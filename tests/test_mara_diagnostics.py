import ast
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
from typing import Dict, Optional, Tuple
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import mara_diagnose as entry
from tools.mara_agent import MARAAgent, MARAConfig, _discounted_returns, quality_score
from tools.dino_single_tower import (
    SEMANTIC_SPATIAL_CHANNELS, DinoSingleTowerDetector,
    config_from_checkpoint as single_tower_config_from_checkpoint,
    semantic_global_dim,
)
from tools.mara_diagnostics import (
    ARMS, PROTOCOL, binary_metrics, checkpoint_config, gradient_probe,
    map_metrics, quality_parts, sampled_pro, score_trace, select_indices,
    trace_infer, validate_cache,
)

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def source_functions(module, filename, names):
    """Execute the actual loader functions without unrelated optional imports."""
    tree = ast.parse((ROOT/filename).read_text(encoding="utf-8"), filename=filename)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"Missing loader functions in {filename}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), module.__dict__)


def real_base_loader():
    # Only heavyweight model factories are stubbed. Runtime restoration, gate
    # config parsing, model-building orchestration and state loading are real.
    evaluator = types.ModuleType("test_mara")
    evaluator.__dict__.update(
        torch=torch, Dict=Dict, Optional=Optional, Tuple=Tuple,
        MARAAgent=MARAAgent, MARAConfig=MARAConfig,
        DinoSingleTowerDetector=DinoSingleTowerDetector,
        single_tower_config_from_checkpoint=single_tower_config_from_checkpoint,
        SEMANTIC_SPATIAL_CHANNELS=SEMANTIC_SPATIAL_CHANNELS,
        semantic_global_dim=semantic_global_dim,
    )
    source_functions(evaluator, "train_mara.py", (
        "normalize_layers_obj", "resolve_hfa_layers", "apply_base_runtime_config",
        "create_visual_backbone_for_mara", "create_single_tower_from_checkpoint",
    ))
    source_functions(evaluator, "train_up.py", ("create_clip_model",))
    source_functions(evaluator, "test_mara.py", (
        "config_from_payload", "apply_payload_runtime_config", "build_models_from_checkpoint",
        "load_mara_checkpoint",
    ))

    class FakeSingle(torch.nn.Module):
        def load_checkpoint_fields(self, payload, strict=True):
            if "dino_single_head" not in payload:
                raise KeyError("dino_single_head")

    def fake_adapter(*args):
        model = torch.nn.Module()
        for name in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter"):
            model.add_module(name, torch.nn.Identity())
        return model

    evaluator.create_model = lambda **kwargs: torch.nn.Linear(1, 1)
    evaluator.create_prompt_learner = lambda *args: torch.nn.Identity()
    evaluator.create_adapter_model = fake_adapter
    evaluator.create_dino_single_tower = lambda **kwargs: FakeSingle()
    evaluator.install_bottleneck_adapters_into_dino = lambda *args, **kwargs: (torch.nn.Identity(), None)
    return evaluator


def native_bash():
    git = shutil.which("git")
    choices = [Path("C:/Program Files/Git/bin/bash.exe")]
    if git:
        choices.append(Path(git).parent.parent/"bin"/"bash.exe")
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        choices.append(Path(found))
    return next((p for p in choices if p.exists()), None)


def shell_path(path):
    value = Path(path).as_posix()
    if os.name == "nt" and len(value) > 2 and value[1] == ":":
        return "/"+value[0].lower()+value[2:]
    return value


def toy():
    torch.manual_seed(7)
    cfg = MARAConfig(
        num_layers=2, evidence_channels=10, global_evidence_dim=2, map_size=8,
        num_regions=4, roi_size=4, hidden_dim=8, max_steps=3, group_size=4,
        reward_conf_weight=0., step_cost=.001, refine_cost=.002,
    )
    agent = MARAAgent(cfg).eval()
    with torch.no_grad():
        agent.op_head.weight.zero_()
        agent.op_head.bias.copy_(torch.tensor([-2., 2.]))
        agent.delta_head.bias.copy_(torch.tensor([-.4, .4]))
        agent.gate_head.bias.fill_(0.)
        agent.score_head[-1].bias.fill_(.2)
        agent.gain_head[-1].bias.fill_(.2)
        agent.gain_lower_head[-1].bias.fill_(.1)
        agent.gain_accept_head[-1].bias.fill_(2.)
    base = torch.softmax(torch.randn(1, 2, 16, 16), 1)
    inputs = dict(base_prob=base, base_logits=torch.tensor([[-.2, .2]]),
                  layer_maps=torch.rand(1, 2, 8, 8),
                  evidence_maps=torch.rand(1, 10, 8, 8), global_evidence=torch.rand(1, 2))
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 5:10, 5:10] = 1
    labels = torch.ones(1, dtype=torch.long)
    return agent, inputs, mask, labels


class TraceTests(unittest.TestCase):
    def test_D0_matches_production_infer(self):
        agent, inputs, _, _ = toy()
        with torch.no_grad():
            expected = agent.infer(**inputs)
        trace = trace_infer(agent, inputs, "D0")
        for key in ("final_prob", "final_logits", "gate_map"):
            self.assertTrue(torch.equal(trace[key], expected[key]), key)
        self.assertEqual(len(trace["records"]), 3)

    def test_D0_matches_STOP_and_inactive_drift(self):
        agent, inputs, mask, labels = toy()
        original = agent._policy

        def scheduled(*args, **kwargs):
            step = args[3]
            kwargs["forced_op"] = torch.tensor([int(step == 0)])
            return original(*args, **kwargs)

        with patch.object(agent, "_policy", side_effect=scheduled):
            with torch.no_grad():
                expected = agent.infer(**inputs)
            trace = trace_infer(agent, inputs)
        self.assertTrue(torch.equal(trace["final_prob"], expected["final_prob"]))
        self.assertTrue(torch.equal(trace["final_logits"], expected["final_logits"]))
        rows = score_trace(trace, mask, labels, agent.cfg)
        self.assertEqual(rows[1]["stop_reason"], "policy_stop")
        self.assertEqual(rows[2]["stop_reason"], "inactive")
        self.assertEqual(rows[1]["pixel_change_max"], 0.)
        self.assertGreater(rows[1]["global_change_max"], 0.)
        self.assertGreater(rows[2]["global_change_max"], 0.)
        self.assertEqual(rows[2]["reward"], 0.)

    def test_true_bypass_retains_spatial_gate(self):
        agent, inputs, _, _ = toy()
        with torch.no_grad():
            agent.gain_lower_head[-1].bias.fill_(-.2)
            agent.gain_accept_head[-1].bias.fill_(-10.)
        original_cfg = agent.cfg
        original_state = copy.deepcopy(agent.state_dict())
        blocked = trace_infer(agent, inputs, "D2")
        bypass = trace_infer(agent, inputs, "D3")
        self.assertFalse(blocked["records"][0]["active_after"])
        self.assertEqual(blocked["records"][0]["execute_weight"], 0.)
        self.assertEqual(sum(r["execute_weight"] for r in bypass["records"]), 3.)
        self.assertGreater(float(bypass["gate_map"].max()), 0.)
        self.assertLess(float(bypass["gate_map"].max()), 1.)
        self.assertIs(agent.cfg, original_cfg)
        for key, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[key]))

    def test_D1_does_not_override_policy_STOP(self):
        agent, inputs, _, _ = toy()
        with torch.no_grad():
            agent.op_head.bias.copy_(torch.tensor([2., -2.]))
        self.assertTrue(torch.equal(trace_infer(agent, inputs, "D1")["final_prob"], inputs["base_prob"]))
        forced = trace_infer(agent, inputs, "D3")
        self.assertEqual([r["op"] for r in forced["records"]], [1, 1, 1])

    def test_GT_is_post_hoc_only(self):
        agent, inputs, mask, labels = toy()
        first = trace_infer(agent, inputs)
        good = score_trace(first, mask, labels, agent.cfg)
        opposite = score_trace(first, 1-mask, 1-labels, agent.cfg)
        self.assertNotEqual(good[0]["after_lr_quality"], opposite[0]["after_lr_quality"])
        second = trace_infer(agent, inputs)
        self.assertTrue(torch.equal(first["final_prob"], second["final_prob"]))
        with self.assertRaisesRegex(ValueError, "GT"):
            trace_infer(agent, dict(inputs, mask=mask))

    def test_quality_matches_training_and_returns(self):
        agent, inputs, mask, labels = toy()
        cfg = replace(agent.cfg, reward_conf_weight=.1)
        actual = quality_parts(inputs["base_prob"], inputs["base_logits"], mask, labels, cfg)
        expected = quality_score(inputs["base_prob"], inputs["base_logits"], mask, labels, cfg)
        self.assertAlmostEqual(actual["quality"], float(expected), places=6)
        rows = score_trace(trace_infer(agent, inputs), mask, labels, agent.cfg)
        rewards = torch.tensor([[row["reward"] for row in rows]])
        expected_returns = _discounted_returns(rewards, agent.cfg.gamma)[0]
        for row, expected_return in zip(rows, expected_returns):
            self.assertAlmostEqual(row["return_to_go"], float(expected_return), places=7)
            self.assertAlmostEqual(
                row["reward"],
                (row["delta_quality_lr"]-row["step_cost"]-row["refine_cost"])*row["active_before"])

    def test_GT_loss_under_downsampling_is_visible(self):
        agent, inputs, _, labels = toy()
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 1, 1] = 1
        rows = score_trace(trace_infer(agent, inputs), mask, labels, agent.cfg)
        self.assertEqual(rows[0]["before_hr_gt_pixels"], 1)
        self.assertEqual(rows[0]["before_lr_gt_pixels"], 0)

    def test_probe_is_read_only_and_has_policy_gradients(self):
        agent, inputs, mask, labels = toy()
        agent.requires_grad_(False)
        before = copy.deepcopy(agent.state_dict())
        rng = torch.get_rng_state().clone()
        probe = gradient_probe(agent, inputs, mask, labels, seed=17)
        self.assertEqual(probe["group_size"], 4)
        self.assertGreater(probe["valid_policy_decisions"], 0)
        self.assertGreater(probe["op_head_grad_l1"], 0.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in agent.parameters()))

    def test_input_validation_and_strict_checkpoint(self):
        agent, inputs, _, _ = toy()
        validate_cache(inputs, agent.cfg)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_cache(dict(inputs, base_logits=torch.full((1, 2), float("nan"))), agent.cfg)
        with self.assertRaisesRegex(ValueError, "MARA checkpoint"):
            checkpoint_config({})
        payload = dict(mara_config=asdict(agent.cfg), mara_agent=agent.state_dict())
        self.assertEqual(checkpoint_config(payload), agent.cfg)
        bad = dict(payload, mara_agent=dict(payload["mara_agent"]))
        bad["mara_agent"].pop("op_head.weight")
        with self.assertRaises(RuntimeError):
            checkpoint_config(bad)


class MetricAndSelectionTests(unittest.TestCase):
    def test_balanced_deterministic_selection(self):
        items = [("c", "a", f"/{i}.png", None if i < 8 else f"/{i}.mask") for i in range(16)]
        indices = select_indices(items, 6, 42)
        self.assertEqual(indices, select_indices(items, 6, 42))
        self.assertEqual(sum(i >= 8 for i in indices), 3)
        self.assertEqual(select_indices(items, 0, 42), list(range(16)))
        sparse = [("c", "a", f"/{i}.png", None if i < 1 else "/mask") for i in range(8)]
        self.assertEqual(len(select_indices(sparse, 6, 42)), 6)

    def test_tied_AUC_and_F1(self):
        gt = np.array([0, 1, 0, 1])
        self.assertEqual(binary_metrics(gt, np.array([0., 1., 0., 1.]))["AUROC"], 1.)
        self.assertEqual(binary_metrics(gt, np.ones(4))["AUROC"], .5)
        self.assertEqual(binary_metrics(gt, np.array([0., 1., 0., 1.]))["F1"], 1.)
        self.assertIsNone(binary_metrics(np.zeros(4), np.ones(4))["AUROC"])

    def test_PRO_matches_reference_loop(self):
        rng = np.random.default_rng(7)
        masks = np.zeros((3, 8, 8), np.uint8)
        masks[0, 1:3, 1:3] = 1
        masks[1, 2:6, 2:6] = 1
        maps = rng.random(masks.shape).astype(np.float32)
        thresholds = np.linspace(float(maps.min()), float(maps.max()), 41, dtype=np.float32)
        fprs, pros = [], []
        for threshold in thresholds:
            pred = maps >= threshold
            overlaps = []
            for gt, score in zip(masks, pred):
                n, components = cv2.connectedComponents(gt)
                for component in range(1, n):
                    region = components == component
                    overlaps.append(score[region].sum()/(region.sum()+1e-8))
            fpr = np.logical_and(pred, 1-masks).sum()/((1-masks).sum()+1e-12)
            if fpr <= .3:
                fprs.append(fpr)
                pros.append(np.mean(overlaps))
        order = np.argsort(fprs)
        x, y = np.asarray(fprs)[order], np.asarray(pros)[order]
        x, indices = np.unique(x, return_index=True)
        y = y[indices]
        if x[0] > 0:
            x, y = np.r_[0., x], np.r_[y[0], y]
        if x[-1] < .3:
            x, y = np.r_[x, .3], np.r_[y, y[-1]]
        reference = np.sum(np.diff(x/.3)*(y[1:]+y[:-1])/2)
        self.assertAlmostEqual(sampled_pro(masks, maps, 41), reference, places=7)
        self.assertIsNone(sampled_pro(np.zeros_like(masks), maps, 41))


class BaseLoaderContractTests(unittest.TestCase):
    def test_old_runtime_omits_overrides_but_preserves_checkpoint_gates(self):
        agent, _, _, _ = toy()
        cfg = replace(agent.cfg, gain_safety_margin=.17, gain_accept_probability=.83, hard_gain_gate=False)
        stored = dict(
            image_size=16, visual_backbone="dino", visual_layers=[0, 1],
            hfa_setting="hfa4", hfa_layers="", hfa_bottleneck=None, dino_bottleneck=256,
            clip_model_name="test", clip_pretrained="test", dino_repo_dir="/fake/dino",
            dino_model_name="test", dino_weights="/fake/weights.pth",
        )
        before = copy.deepcopy(stored)
        payload = dict(mara_config=asdict(cfg), mara_agent=MARAAgent(cfg).state_dict(),
                       visual_backbone="dino", visual_layers=[0, 1],
                       hfa_setting="custom", hfa_layers=[1], hfa_bottleneck=32)
        evaluator = real_base_loader()
        # Reproduce the reported bug using the old sparse Namespace.
        with self.assertRaisesRegex(AttributeError, "gain_safety_margin"):
            evaluator.config_from_payload(payload, types.SimpleNamespace(**stored))
        for architecture in ("clip_dino", "dino_single"):
            with self.subTest(architecture=architecture):
                current = dict(payload, base_arch=architecture)
                if architecture == "dino_single":
                    current.update(dino_single_config=dict(visual_layers=[0, 1], evidence_channels=8),
                                   dino_single_head={})
                runtime = entry.base_export_runtime(stored)
                with patch("torch.hub.load", return_value=torch.nn.Linear(1, 1)):
                    models = evaluator.build_models_from_checkpoint(runtime, torch.device("cpu"), current)
                self.assertEqual(models[5].cfg, cfg)
                self.assertIsNone(runtime.gain_safety_margin)
                self.assertIsNone(runtime.gain_accept_probability)
                self.assertFalse(runtime.disable_hard_gain_gate)
                self.assertEqual(runtime.hfa_layers_runtime, (1,))
                self.assertEqual(runtime.hfa_bottleneck_runtime, 32)
        self.assertEqual(stored, before)


class OfflinePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"run"
        self.root.mkdir()
        self.agent, self.inputs, self.mask, self.labels = toy()
        self.ckpt = Path(self.temp.name)/"mara.pth"
        torch.save(dict(epoch=9, mara_config=asdict(self.agent.cfg), mara_agent=self.agent.state_dict()), self.ckpt)
        self.manifest = [
            dict(id=f"sample_{i}", index=i, shard=0, dataset="toy", category="part",
                 image_path=f"/not_read/{i}", image_sha256="diagnostic",
                 mask_path=None, mask_sha256=None)
            for i in range(4)
        ]
        self.cfg = dict(
            protocol=PROTOCOL, mara_ckpt=str(self.ckpt),
            checkpoint_sha256=entry.sha256_file(self.ckpt),
            manifest_sha256=entry.digest(self.manifest), mara_config=asdict(self.agent.cfg),
            num_shards=1, seed=42, gradient_ids=["sample_0"],
            pro_num_th=21, pro_max_fpr=.3, quality_tolerance=1e-4,
        )
        entry.atomic_json(self.root/"config.json", self.cfg)
        entry.atomic_json(self.root/"manifest.json", self.manifest)
        signature = entry.digest(self.cfg)
        for i, record in enumerate(self.manifest):
            cache = copy.deepcopy(self.inputs)
            cache.update(mask=self.mask.to(torch.uint8) if i % 2 else torch.zeros_like(self.mask, dtype=torch.uint8),
                         labels=torch.tensor([i % 2]), record_id=record["id"], signature=signature)
            path = self.root/"evidence"/f"{record['id']}.pt"
            entry.atomic_torch(path, cache)
            entry.atomic_json(path.with_suffix(".json"), dict(signature=signature, sha256=entry.sha256_file(path)))
        self.args = type("Args", (), dict(work_dir=str(self.root), shard_id=0, device="cpu"))()

    def tearDown(self):
        self.temp.cleanup()

    def test_end_to_end_resume_integrity_and_metrics(self):
        before = entry.sha256_file(self.ckpt)
        entry.diagnose(self.args)
        entry.collect(self.args)
        self.assertEqual(before, entry.sha256_file(self.ckpt))
        self.assertTrue((self.root/"complete.json").exists())
        import csv
        with (self.root/"results"/"step_trace.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4*4*3)
        self.assertEqual({row["arm"] for row in rows}, set(ARMS))
        with (self.root/"results"/"macro_metrics.csv").open(encoding="utf-8") as stream:
            metrics = list(csv.DictReader(stream))
        self.assertEqual(len(metrics), 1+4*3)
        path = self.root/"per_image"/"sample_0.npz"
        modified = path.stat().st_mtime_ns
        entry.diagnose(self.args)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        cache = self.root/"evidence"/"sample_0.pt"
        with cache.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "Corrupted"):
            entry.collect(self.args)

    def test_collect_rejects_partial_and_foreign_outputs(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            entry.collect(self.args)
        entry.diagnose(self.args)
        result = self.root/"per_image"/"sample_0.json"
        obj = entry.load_json(result)
        obj["signature"] = "different_run"
        entry.atomic_json(result, obj)
        with self.assertRaisesRegex(ValueError, "another run"):
            entry.diagnose(self.args)

    def test_checkpoint_change_fails_before_diagnosis(self):
        with self.ckpt.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            entry.diagnose(self.args)

    def test_prepare_export_resume_with_mock_Base(self):
        # Metadata/real cache pipeline, but no dataset/model downloads.
        source = Path(self.temp.name)/"source"
        source.mkdir()
        items = []
        for i in range(4):
            image = source/f"{i}.png"
            image.write_bytes(f"image_{i}".encode())
            mask = source/f"{i}_mask.png" if i % 2 else None
            if mask:
                mask.write_bytes(f"mask_{i}".encode())
            items.append(("part", "Anomaly" if mask else "Normal", str(image), str(mask) if mask else None))
        outer = self

        class FakeDataset:
            def __init__(self, **kwargs):
                self.data_to_iterate = items

            def __getitem__(self, index):
                item = items[index]
                return dict(image=torch.zeros(3, 16, 16),
                            mask=outer.mask[0] if item[3] else torch.zeros(1, 16, 16),
                            is_anomaly=int(item[3] is not None), image_path=item[2])

        datasets = types.ModuleType("Datasets")
        datasets.DATASET_CLASSES = {"toy": {"part"}}
        datasets.DATASET_REGISTRY = {"toy": (FakeDataset, types.SimpleNamespace(TEST="test"), str(source))}
        payload = entry.load_torch(self.ckpt)
        payload.update(base_arch="clip_dino", visual_backbone="clip", visual_layers=[0, 1],
                       cls_token_adapter={}, patch_token_adapter={}, prompt_adapter={}, prompt_learner={})
        torch.save(payload, self.ckpt)
        work = Path(self.temp.name)/"prepared"
        args = entry.parser().parse_args([
            "--stage", "prepare", "--work_dir", str(work), "--mara_ckpt", str(self.ckpt),
            "--datasets", "toy", "--samples_per_category", "2", "--num_shards", "1",
            "--gradient_images", "0", "--image_size", "16",
        ])
        with patch.dict("sys.modules", {"Datasets": datasets}):
            entry.prepare(args)
            config_before = (work/"config.json").read_bytes()
            entry.prepare(args)
            self.assertEqual(config_before, (work/"config.json").read_bytes())
            forwards = []
            utils = types.ModuleType("tools.utils_up")

            def base_forward(**kwargs):
                forwards.append(kwargs["image_info"]["image_path"])
                base = outer.inputs["base_prob"]
                evidence = dict(
                    cross_prob_layers=[base[:, 1]]*2,
                    cross_margin_layers=[torch.zeros(1, 8, 8)]*2,
                    awareness_layers=[base[:, 1]]*2,
                    normal_similarity_layers=[torch.zeros(1, 8, 8)]*2,
                    anomaly_similarity_layers=[torch.zeros(1, 8, 8)]*2,
                    global_margin_layers=[torch.zeros(1)]*2,
                )
                return None, kwargs["image_info"]["mask"], base, outer.inputs["base_logits"], evidence

            utils.get_anomaly_map = base_forward
            evaluator = real_base_loader()
            real_build = evaluator.build_models_from_checkpoint

            def build(runtime, device, checkpoint):
                self.assertEqual(runtime.image_size, 16)
                models = real_build(runtime, device, checkpoint)
                self.assertEqual(models[5].cfg, outer.agent.cfg)
                return models

            evaluator.build_models_from_checkpoint = build
            worker = type("Args", (), dict(work_dir=str(work), shard_id=0, device="cpu"))()
            with patch.dict("sys.modules", {"tools.utils_up": utils, "test_mara": evaluator}):
                entry.export(worker)
                self.assertEqual(len(forwards), 2)
                entry.export(worker)
                self.assertEqual(len(forwards), 2)
            entry.diagnose(worker)
            entry.collect(worker)
            self.assertTrue((work/"complete.json").exists())


class EntrypointTests(unittest.TestCase):
    def test_shell_contract(self):
        script = (ROOT/"run_exp_mara_diagnose.sh").read_text(encoding="utf-8")
        self.assertNotIn("\r", script)
        self.assertIn("flock -n", script)
        self.assertIn("run_workers export", script)
        self.assertIn("run_workers diagnose", script)
        self.assertIn("--stage collect", script)
        self.assertIn("MARA_CKPT must name one existing", script)
        self.assertNotIn("train_mara.py", script)
        self.assertNotIn("git pull", script)
        bash = native_bash()
        if not bash:
            self.skipTest("No native Bash available")
        result = subprocess.run([str(bash), "-n", (ROOT/"run_exp_mara_diagnose.sh").as_posix()],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_mock_shell(self, **overrides):
        bash = native_bash()
        if not bash:
            self.skipTest("No native Bash available")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bin_dir = root/"bin"
        bin_dir.mkdir()
        calls = root/"calls.txt"
        python = bin_dir/"mock_python"
        python.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"${1:-}\" == -c ]]; then exit 0; fi\n"
            "stage=''; shard=''\n"
            "while (( $# )); do\n"
            "  case \"$1\" in\n"
            "    --stage) stage=\"$2\"; shift 2;;\n"
            "    --shard_id) shard=\"$2\"; shift 2;;\n"
            "    *) shift;;\n"
            "  esac\n"
            "done\n"
            "printf '%s|gpu=%s|shard=%s\\n' \"$stage\" \"${CUDA_VISIBLE_DEVICES:-}\" \"$shard\" >> \"$MOCK_CALLS\"\n"
            "if [[ \"$stage\" == export && \"$shard\" == 1 && \"${FAIL_EXPORT_SHARD:-0}\" == 1 ]]; then exit 7; fi\n",
            encoding="utf-8", newline="\n")
        python.chmod(0o755)
        flock = bin_dir/"flock"
        flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
        flock.chmod(0o755)
        checkpoint = root/"mara.pth"
        checkpoint.write_bytes(b"mock: orchestration only")
        # Native Windows sandbox must not see MSYS mkdir probing home ancestors.
        (root/"work").mkdir()
        env = os.environ.copy()
        for key in (
            "MARA_CKPT", "WORK_DIR", "GPU_IDS", "DATASETS", "DATA_ROOTS", "SAMPLES_PER_CATEGORY",
            "STEPS", "GRADIENT_IMAGES", "SEED", "CPU_THREADS", "RUN_EXPORT", "RUN_DIAGNOSE",
            "RUN_COLLECT", "TRAIN_PYTHON", "DINO_REPO_DIR", "DINO_WEIGHTS", "CATEGORIES",
        ):
            env.pop(key, None)
        env.update(
            MARA_CKPT=shell_path(checkpoint), WORK_DIR=shell_path(root/"work"),
            TRAIN_PYTHON=shell_path(python), MOCK_CALLS=shell_path(calls), GPU_IDS="0,1",
            PATH=os.pathsep.join((str(bin_dir), str(bash.parent.parent/"usr"/"bin"),
                                  str(bash.parent), env.get("PATH", ""))),
        )
        env.update(overrides)
        result = subprocess.run([str(bash), (ROOT/"run_exp_mara_diagnose.sh").as_posix()],
                                cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=40)
        lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        return result, lines

    def test_shell_orders_both_workers_before_collect(self):
        result, lines = self.run_mock_shell()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        stages = [line.split("|")[0] for line in lines]
        self.assertEqual(stages[0], "prepare")
        self.assertEqual(stages[1:3], ["export", "export"])
        self.assertEqual(stages[3:5], ["diagnose", "diagnose"])
        self.assertEqual(stages[5], "collect")
        self.assertIn("export|gpu=0|shard=0", lines)
        self.assertIn("export|gpu=1|shard=1", lines)

    def test_shell_worker_failure_never_collects(self):
        result, lines = self.run_mock_shell(FAIL_EXPORT_SHARD="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("worker failed", result.stdout+result.stderr)
        self.assertFalse(any(line.startswith("diagnose") or line.startswith("collect") for line in lines))

    def test_shell_bad_budgets_and_duplicate_GPU_fail_before_python(self):
        for settings in (dict(GPU_IDS="0,0"), dict(STEPS="-1"), dict(RUN_EXPORT="maybe")):
            with self.subTest(settings=settings):
                result, lines = self.run_mock_shell(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
