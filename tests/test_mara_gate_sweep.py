import copy
from dataclasses import asdict, replace
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

import mara_diagnose as shared
import mara_gate_sweep as gate
from tools.mara_agent import MARAAgent
from tools.mara_diagnostics import score_trace, trace_infer
from tools.mara_gate_sweep import make_arms, score_batch, trace_batch
import test_mara_diagnostics as fixtures

toy = fixtures.toy
real_base_loader = fixtures.real_base_loader


class GateTraceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_CUDA_batch_matches_production_and_scores(self):
        agent, inputs, mask, labels = toy()
        agent = agent.cuda()
        packed = {k: v.repeat(4, *([1]*(v.ndim-1))).cuda() for k, v in inputs.items()}
        spec = make_arms("0")["G0"]
        result = trace_batch(agent, packed, spec)
        with torch.no_grad():
            expected = agent.infer(**packed)
        for key in ("final_prob", "final_logits", "gate_map"):
            self.assertTrue(torch.equal(result[key], expected[key]), key)
        rows = score_batch(result, mask.repeat(4, 1, 1, 1).cuda(), labels.repeat(4).cuda(), agent.cfg)
        self.assertTrue(all(np.isfinite(row["reward"]) for trace in rows for row in trace["records"]))

    def test_batch_matches_production_all_arms_and_preserves_weights(self):
        agent, inputs, mask, labels = toy()
        before = copy.deepcopy(agent.state_dict())
        original = agent.cfg
        packed = {k: v.repeat(3, *([1]*(v.ndim-1))) for k, v in inputs.items()}
        for arm, spec in make_arms("0,-0.01,-0.03,-0.05,-0.10").items():
            with self.subTest(arm=arm):
                result = trace_batch(agent, packed, spec)
                agent.cfg = replace(original, gain_safety_margin=spec["gain_safety_margin"],
                                    freeze_noop_state=spec["freeze_noop_state"],
                                    use_gain_gate=not spec["bypass_gain"])
                with torch.no_grad():
                    expected = agent.infer(**packed)
                agent.cfg = original
                for key in ("final_prob", "final_logits", "gate_map"):
                    self.assertTrue(torch.equal(result[key], expected[key]), (arm, key))
                self.assertEqual(len(result["traces"]), 3)
        self.assertIs(agent.cfg, original)
        for name, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_posthoc_batch_scores_match_existing_scalar_quality_and_returns(self):
        agent, inputs, mask, labels = toy()
        for spec in make_arms("0,-0.05").values():
            result = trace_batch(agent, inputs, spec)
            actual = score_batch(result, mask, labels, agent.cfg)[0]["records"]
            scalar = trace_infer(agent, inputs, intervention=dict(spec, force_refine=False))
            expected = score_trace(scalar, mask, labels, agent.cfg)
            for a, b in zip(actual, expected):
                for key in ("reward", "candidate_net_reward", "before_hr_quality", "after_lr_quality", "return_to_go"):
                    self.assertAlmostEqual(a[key], b[key], places=6)
        with self.assertRaisesRegex(ValueError, "GT"):
            trace_batch(agent, dict(inputs, mask=mask), make_arms("0")["G0"])

    def test_negative_margin_releases_predicted_negative_lower_not_probability(self):
        agent, inputs, _, _ = toy()
        with torch.no_grad():
            agent.gain_lower_head[-1].weight.zero_()
            agent.gain_lower_head[-1].bias.fill_(-.02)
            agent.gain_accept_head[-1].weight.zero_()
            agent.gain_accept_head[-1].bias.fill_(3.)
        specs = make_arms("0,-0.03")
        self.assertEqual(trace_batch(agent, inputs, specs["G0"])["traces"][0]["records"][0]["execute_weight"], 0.)
        self.assertEqual(trace_batch(agent, inputs, specs["G1"])["traces"][0]["records"][0]["execute_weight"], 1.)
        with torch.no_grad():
            agent.gain_accept_head[-1].bias.fill_(-10.)
        self.assertEqual(trace_batch(agent, inputs, specs["G1"])["traces"][0]["records"][0]["execute_weight"], 0.)
        self.assertEqual(agent.cfg.gain_accept_threshold, 0.)

    def test_noop_state_freezes_every_tensor_for_STOP_reject_and_inactive(self):
        agent, inputs, _, _ = toy()
        base = torch.nn.functional.interpolate(inputs["base_prob"], size=(8, 8), mode="bilinear", align_corners=False).repeat(3, 1, 1, 1)
        layers = inputs["layer_maps"].repeat(3, 1, 1, 1)
        extra = inputs["evidence_maps"].repeat(3, 1, 1, 1)
        logits = torch.randn_like(base)
        current = .8*base+.2*logits.softmax(1)
        global_logits = torch.tensor([[-2., 2.]]).repeat(3, 1)
        base_logits = torch.zeros_like(global_logits)
        cumulative = torch.full_like(base[:, :1], .2)
        active = torch.tensor([True, True, False])
        with torch.no_grad():
            agent.gain_lower_head[-1].weight.zero_()
            agent.gain_lower_head[-1].bias.fill_(-1.)
            state = agent._make_state(current, base, layers, 1, extra)
            policy = agent._policy(state, global_logits, active, 1, sample=False,
                                   forced_op=torch.tensor([0, 1, 1]), global_evidence=inputs["global_evidence"].repeat(3, 1))
            result = agent._refine(base, base_logits, logits, current, global_logits,
                                   cumulative, state, layers, policy, active, True, True)
        for key, expected in (("next_prob", current), ("next_logits", logits),
                              ("next_global_logits", global_logits), ("next_cumulative_gate", cumulative)):
            self.assertTrue(torch.equal(result[key], expected), key)
        self.assertTrue(torch.equal(result["candidate_global_logits"][2], global_logits[2]))
        self.assertFalse(torch.equal(result["candidate_global_logits"][0], global_logits[0]))

    def test_legacy_reference_is_explicit_and_drift_is_detectable(self):
        agent, inputs, _, _ = toy()
        original = agent._policy
        def scheduled(*args, **kwargs):
            kwargs["forced_op"] = torch.tensor([int(args[3] == 0)])
            return original(*args, **kwargs)
        arms = make_arms("0")
        with patch.object(agent, "_policy", side_effect=scheduled):
            fixed = trace_batch(agent, inputs, arms["G0"])
            legacy = trace_batch(agent, inputs, arms["L0_legacy"])
        self.assertEqual(fixed["traces"][0]["records"][1]["global_change_max"], 0.)
        self.assertEqual(fixed["traces"][0]["records"][2]["global_change_max"], 0.)
        self.assertGreater(legacy["traces"][0]["records"][1]["global_change_max"], 0.)
        self.assertFalse(torch.equal(fixed["final_logits"], legacy["final_logits"]))

    def test_invalid_margin_grids_fail(self):
        for text in ("0,0", "-0.01,0", "0,nan", "0,inf", "0,0.1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                make_arms(text)


class GatePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.work = root/"run"
        self.ckpt = root/"mara.pth"
        self.agent, self.inputs, self.mask, self.labels = toy()
        payload = dict(epoch=29, mara_config=asdict(self.agent.cfg), mara_agent=self.agent.state_dict(),
                       base_arch="clip_dino", visual_backbone="clip", visual_layers=[0, 1],
                       cls_token_adapter={}, patch_token_adapter={}, prompt_adapter={}, prompt_learner={})
        torch.save(payload, self.ckpt)
        source = root/"source"
        source.mkdir()
        items, samples = [], []
        for i in range(4):
            image = source/f"{i}.png"
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(image)
            mask_path = source/f"mask_{i}.png" if i % 2 else None
            mask = self.mask[0].clone() if i == 3 else torch.zeros(1, 16, 16)
            if mask_path:
                Image.fromarray((mask[0].numpy()*255).astype(np.uint8)).save(mask_path)
            items.append(("part", "bad" if i % 2 else "good", str(image), str(mask_path) if mask_path else None))
            samples.append(dict(image=torch.zeros(3, 16, 16), mask=mask,
                                is_anomaly=i % 2, image_path=str(image)))
        class Dataset:
            def __init__(self, **kwargs):
                self.data_to_iterate = items
            def __getitem__(self, i):
                return samples[i]
        datasets = types.ModuleType("Datasets")
        datasets.DATASET_CLASSES = {"toy": ["part"]}
        datasets.DATASET_REGISTRY = {"toy": (Dataset, types.SimpleNamespace(TEST="test"), str(source))}
        utils = types.ModuleType("tools.utils_up")
        self.forwards = []
        def forward(**kwargs):
            n = len(kwargs["image_info"]["image_path"])
            self.forwards.append(n)
            base = self.inputs["base_prob"].repeat(n, 1, 1, 1)
            zero = torch.zeros(n, 8, 8)
            ev = dict(cross_prob_layers=[base[:, 1]]*2, cross_margin_layers=[zero]*2,
                      awareness_layers=[base[:, 1]]*2, normal_similarity_layers=[zero]*2,
                      anomaly_similarity_layers=[zero]*2, global_margin_layers=[torch.zeros(n)]*2)
            return None, kwargs["image_info"]["mask"], base, self.inputs["base_logits"].repeat(n, 1), ev
        utils.get_anomaly_map = forward
        modules = {"Datasets": datasets, "tools.utils_up": utils, "test_mara": real_base_loader()}
        self.patcher = patch.dict("sys.modules", modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.args = gate.parser().parse_args([
            "--stage", "prepare", "--work_dir", str(self.work), "--mara_ckpt", str(self.ckpt),
            "--datasets", "toy", "--samples_per_category", "4", "--num_shards", "1",
            "--image_size", "16", "--pro_num_th", "21", "--device", "cpu", "--metric_workers", "1",
        ])

    def run_to_audit(self):
        gate.prepare(self.args)
        gate.export(self.args)
        gate.audit(self.args)

    def test_full_pipeline_batches_once_audits_same_images_and_resumes(self):
        before = shared.sha256_file(self.ckpt)
        self.run_to_audit()
        self.assertEqual(self.forwards, [4])
        audit = shared.load_json(self.work/"audit.json")
        self.assertEqual(audit["excluded"], 1)
        self.assertEqual(audit["rows"][1]["issue"], "empty_native_annotation")
        gate.sweep(self.args)
        gate.collect(self.args)
        complete = shared.load_json(self.work/"complete.json")
        self.assertEqual(complete["images"], 3)
        self.assertFalse(list(self.work.rglob("*.npz")))
        with zipfile.ZipFile(self.work/"gate_report.zip") as archive:
            self.assertIn("results/gate_summary.csv", archive.namelist())
            rows = list(csv.DictReader(io.StringIO(archive.read("results/step_trace.csv").decode())))
            self.assertEqual(len(rows), 3*7*3)
            self.assertEqual(len({r["sample_id"] for r in rows}), 3)
            for row in rows:
                if row["arm"] != "L0_legacy" and float(row["execute_weight"]) == 0:
                    self.assertEqual(float(row["global_change_max"]), 0.)
        path = next((self.work/"categories").glob("*.json"))
        modified = path.stat().st_mtime_ns
        gate.export(self.args)
        gate.sweep(self.args)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        self.assertEqual(self.forwards, [4])
        self.assertEqual(shared.sha256_file(self.ckpt), before)

    def test_partial_collection_and_corrupt_cache_fail_closed(self):
        self.run_to_audit()
        with self.assertRaisesRegex(ValueError, "Missing category"):
            gate.collect(self.args)
        cache = next((self.work/"evidence").glob("*.pt"))
        with cache.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "Corrupted"):
            gate.audit(self.args)

    def test_verified_source_cache_reuse_never_loads_Base(self):
        self.run_to_audit()
        second = copy.copy(self.args)
        second.work_dir = str(Path(self.temp.name)/"second")
        second.source_work_dir = str(self.work)
        gate.prepare(second)
        with patch("mara_diagnose.base_export_runtime", side_effect=AssertionError("Base must not load")):
            gate.export(second)
        gate.audit(second)
        self.assertEqual(self.forwards, [4])
        for path in (Path(second.work_dir)/"evidence").glob("*.json"):
            self.assertIn("reused_from", shared.load_json(path))

    def test_source_checkpoint_mismatch_is_rejected(self):
        self.run_to_audit()
        second = copy.copy(self.args)
        second.work_dir = str(Path(self.temp.name)/"second")
        second.source_work_dir = str(self.work)
        with self.ckpt.open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "different MARA"):
            gate.prepare(second)

    def test_old_v1_evidence_and_missing_new_config_field_are_supported(self):
        self.run_to_audit()
        cfg = shared.load_json(self.work/"config.json")
        cfg["protocol"] = "mara_early_stop_ab_v1"
        cfg["mara_config"].pop("freeze_noop_state")
        cfg["original_mara_config"].pop("freeze_noop_state")
        shared.atomic_json(self.work/"config.json", cfg)
        signature = shared.digest(cfg)
        for path in (self.work/"evidence").glob("*.pt"):
            cache = shared.load_torch(path)
            cache["signature"] = signature
            shared.atomic_torch(path, cache)
            shared.atomic_json(path.with_suffix(".json"), dict(signature=signature, sha256=shared.sha256_file(path)))
        second = copy.copy(self.args)
        second.work_dir = str(Path(self.temp.name)/"from_v1")
        second.source_work_dir = str(self.work)
        gate.prepare(second)
        gate.export(second)
        gate.audit(second)
        self.assertEqual(self.forwards, [4])

    def test_sweep_OOM_retries_same_chunk_and_no_duplicate_rows(self):
        self.run_to_audit()
        original = gate.trace_batch
        def limited(agent, inputs, spec):
            if len(inputs["base_prob"]) > 1:
                raise torch.cuda.OutOfMemoryError("synthetic sweep limit")
            return original(agent, inputs, spec)
        with patch.object(gate, "trace_batch", side_effect=limited):
            gate.sweep(self.args)
        gate.collect(self.args)
        with (self.work/"results"/"step_trace.csv").open(encoding="utf8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3*7*3)
        self.assertEqual(len({(r["sample_id"], r["arm"], r["step"]) for r in rows}), len(rows))

    def test_export_OOM_halves_batch_without_skipping_images(self):
        gate.prepare(self.args)
        fake = __import__("sys").modules["tools.utils_up"]
        original = fake.get_anomaly_map
        def limited(**kwargs):
            if len(kwargs["image_info"]["image_path"]) > 2:
                raise torch.cuda.OutOfMemoryError("synthetic memory limit")
            return original(**kwargs)
        with patch.object(fake, "get_anomaly_map", side_effect=limited):
            gate.export(self.args)
        self.assertEqual(self.forwards, [2, 2])
        self.assertEqual(len(list((self.work/"evidence").glob("*.pt"))), 4)

    def test_flag_policy_keeps_empty_GT_consistently(self):
        self.args.empty_gt_policy = "flag"
        self.run_to_audit()
        audited = shared.load_json(self.work/"audit.json")
        self.assertEqual(audited["excluded"], 0)
        self.assertEqual(audited["included"], 4)
        self.assertEqual(audited["rows"][1]["issue"], "empty_native_annotation")

    def test_native_foreground_loss_is_distinguished_from_empty_annotation(self):
        mask = Path(self.temp.name)/"source"/"mask_1.png"
        native = np.zeros((16, 16), np.uint8)
        native[0, 0] = 255
        Image.fromarray(native).save(mask)
        self.run_to_audit()
        audited = shared.load_json(self.work/"audit.json")
        self.assertEqual(audited["rows"][1]["issue"], "foreground_lost_during_preprocess_or_threshold")
        self.assertEqual(audited["rows"][1]["native_nonzero_pixels"], 1)


class GateShellTests(unittest.TestCase):
    def test_two_GPU_stage_order(self):
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_gate_sweep.sh")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual([r.split("|")[0] for r in lines], ["prepare", "export", "export", "audit", "sweep", "sweep", "collect"])
        self.assertIn("sweep|gpu=0|shard=0", lines)
        self.assertIn("sweep|gpu=1|shard=1", lines)

    def test_failure_reports_child_log_and_never_sweeps(self):
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_gate_sweep.sh", FAIL_EXPORT_SHARD="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Log tails follow", result.stdout+result.stderr)
        self.assertFalse(any(r.startswith("sweep") or r.startswith("collect") for r in lines))


if __name__ == "__main__":
    unittest.main()
