import copy
import csv
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import sys
import types
from typing import Dict, Optional
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
import torch._dynamo  # Load before patch.dict(sys.modules) fixtures restore imports.
import torch.nn.functional as F

import mara_diagnose as shared
import mara_reward_experiment as reward
from tools.loss import BinaryDiceLoss, FocalLoss
from tools.mara_agent import MARAAgent, quality_score
from tools.mara_reward import ARMS, CachedBatches, EvidenceStore, arm_config, initial_agent, options, scheduler_for, split_source, state_hash
import test_mara_diagnostics as fixtures
import test_mara_gate_sweep as gate_fixtures


def training_module():
    """The actual production training function, without optional Base factories."""
    module = types.ModuleType("train_mara")
    module.__dict__.update(torch=torch, np=np, F=F, Dict=Dict, Optional=Optional,
        MARAAgent=MARAAgent, DinoSingleTowerDetector=torch.nn.Module,
        FocalLoss=FocalLoss, BinaryDiceLoss=BinaryDiceLoss,
        DDP=torch.nn.parallel.DistributedDataParallel)
    fixtures.source_functions(module, "train_mara.py", ("unwrap_mara_agent", "reduce_epoch_metrics", "train_one_epoch"))
    return module


class RewardUnitTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_CUDA_training_and_gpu_cache_eviction(self):
        module = training_module()
        agent, inputs, mask, labels = fixtures.toy()
        cfg = arm_config(agent.cfg, "R1")
        args = options(2, cfg, 1)
        args.is_main_process = False
        args.gain_warmup_epochs = 0
        policy = initial_agent(cfg, 11).cuda()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        cache = dict(inputs, mask=mask, labels=labels)
        store = EvidenceStore(lambda _: cache, "cuda", cpu_gb=0, gpu_gb=0)
        store.gpu_cap = EvidenceStore.size(cache)*2
        for index in range(4):
            store.get(dict(id=str(index)))
        self.assertLessEqual(store.gpu_bytes, store.gpu_cap)
        loader = CachedBatches(store, [dict(id="3")], 1, 0, 0)
        before = state_hash(policy.state_dict())
        meters = module.train_one_epoch(0, policy, optimizer, None, None, None, None, None, None,
                                        loader, torch.device("cuda"), args)
        self.assertTrue(all(math.isfinite(value) for value in meters.values()))
        self.assertNotEqual(before, state_hash(policy.state_dict()))

    def test_matched_initialization_only_reward_cls_differs(self):
        agent, _, _, _ = fixtures.toy()
        configs = {arm: arm_config(agent.cfg, arm) for arm in ARMS}
        hashes = [state_hash(initial_agent(cfg, 7).state_dict()) for cfg in configs.values()]
        self.assertEqual(len(set(hashes)), 1)
        self.assertNotEqual(hashes[0], state_hash(initial_agent(configs["R0"], 8).state_dict()))
        for arm in ("R1", "R2"):
            changed = {key for key in asdict(configs[arm]) if asdict(configs[arm])[key] != asdict(configs["R0"])[key]}
            self.assertEqual(changed, {"reward_cls_weight"})

    def test_reward_coefficient_changes_true_quality_not_display_only(self):
        agent, inputs, mask, labels = fixtures.toy()
        low = dict(inputs, base_logits=torch.tensor([[-2., 2.]]))
        high = dict(inputs, base_logits=torch.tensor([[-3., 3.]]))
        changes = []
        for arm in ARMS:
            cfg = arm_config(agent.cfg, arm)
            before = quality_score(low["base_prob"], low["base_logits"], mask, labels, cfg)
            after = quality_score(high["base_prob"], high["base_logits"], mask, labels, cfg)
            changes.append(float(after-before))
        self.assertGreater(changes[0], changes[1])
        self.assertGreater(changes[1], changes[2])
        self.assertEqual(changes[2], 0.)

    def test_cached_training_identical_to_live_frozen_path(self):
        module = training_module()
        agent, inputs, mask, labels = fixtures.toy()
        cfg = arm_config(agent.cfg, "R0")
        args = options(2, cfg, 1)
        args.is_main_process = False
        args.gain_warmup_epochs = 0
        module.get_anomaly_map = lambda **kw: (None, mask, inputs["base_prob"], inputs["base_logits"], {})
        module.build_mara_evidence = lambda **kw: dict(layer_maps=inputs["layer_maps"],
            extra_maps=inputs["evidence_maps"], global_evidence=inputs["global_evidence"])
        results = []
        for cached in (False, True):
            policy = initial_agent(cfg, 9)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            batch = dict(mask=mask, is_anomaly=labels)
            if cached:
                batch["_mara_cached"] = inputs
            else:
                args.visual_backbone = "clip"
                args.text_source = "prompt_learner"
            torch.manual_seed(100)
            meters = module.train_one_epoch(0, policy, optimizer, None, None, None, None, None, None,
                                            [batch], torch.device("cpu"), args)
            results.append((copy.deepcopy(policy.state_dict()), meters))
        self.assertEqual(results[0][1], results[1][1])
        for key in results[0][0]:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]), key)

    def test_scheduler_exactly_matches_production(self):
        module = types.ModuleType("scheduler")
        module.__dict__.update(torch=torch, math=math, LambdaLR=torch.optim.lr_scheduler.LambdaLR)
        fixtures.source_functions(module, "train_up.py", ("build_warmup_cosine_scheduler",))
        agent, _, _, _ = fixtures.toy()
        args = options(3, agent.cfg)
        left = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.01)
        right = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.01)
        actual = scheduler_for(left, 3, 4, args)
        expected = module.build_warmup_cosine_scheduler(right, 1, 36, .05)
        for _ in range(37):
            self.assertEqual(actual.get_last_lr(), expected.get_last_lr())
            left.step(); right.step()
            actual.step(); expected.step()

    def test_bounded_cache_and_deterministic_order_no_dropped_tail(self):
        _, inputs, mask, labels = fixtures.toy()
        cache = dict(inputs, mask=mask, labels=labels)
        calls = []
        def loader(record):
            calls.append(record["id"])
            return cache
        cap = EvidenceStore.size(cache)*1.5/2**30
        store = EvidenceStore(loader, "cpu", cpu_gb=cap, gpu_gb=0)
        records = [dict(id=str(i)) for i in range(5)]
        store.get(records[0]); store.get(records[0]); store.get(records[1])
        self.assertEqual(calls, ["0", "1"])
        self.assertLessEqual(store.cpu_bytes, store.cpu_cap)
        a, b = CachedBatches(store, records, 2, 10, 0), CachedBatches(store, records, 2, 10, 0)
        self.assertEqual(a.order, b.order)
        self.assertNotEqual(a.order, CachedBatches(store, records, 2, 10, 1).order)
        self.assertEqual(sorted(a.order), list(range(5)))
        self.assertEqual([len(row["is_anomaly"]) for row in a], [2, 2, 1])


class RewardPipelineTests(unittest.TestCase):
    def setUp(self):
        gate_fixtures.GatePipelineTests.setUp(self)
        source = Path(self.temp.name)/"source_train"
        source.mkdir()
        items, samples = [], []
        for i in range(8):
            image = source/f"image_{i}.png"
            Image.fromarray(np.full((16, 16, 3), i, np.uint8)).save(image)
            mask_path = source/f"mask_{i}.png" if i % 2 else None
            mask = self.mask[0].clone() if i % 2 else torch.zeros(1, 16, 16)
            if mask_path:
                Image.fromarray((mask[0].numpy()*255).astype(np.uint8)).save(mask_path)
            items.append(("part", "bad" if i % 2 else "good", str(image), str(mask_path) if mask_path else None))
            samples.append(dict(image=torch.zeros(3, 16, 16), mask=mask, is_anomaly=i % 2, image_path=str(image)))
        class Source:
            def __init__(self, **kwargs):
                self.data_to_iterate = items
            def __getitem__(self, index):
                return samples[index]
        sys.modules["Datasets"].DATASET_CLASSES["visa"] = ["part"]
        sys.modules["Datasets"].DATASET_REGISTRY["visa"] = (Source, types.SimpleNamespace(TEST="test"), str(source))
        self.args = reward.parser().parse_args([
            "--stage", "prepare", "--work_dir", str(self.work), "--mara_ckpt", str(self.ckpt),
            "--datasets", "visa toy", "--samples_per_category", "4", "--num_shards", "1",
            "--image_size", "16", "--pro_num_th", "21", "--device", "cpu", "--metric_workers", "1",
            "--epochs", "2", "--warmup_epochs", "1", "--train_batch_size", "2", "--gpu_cache_gb", "0",
        ])
        self.training = training_module()
        self.training_patcher = patch.dict("sys.modules", {"train_mara": self.training})
        self.training_patcher.start()
        self.addCleanup(self.training_patcher.stop)

    def run_to_audit(self):
        reward.prepare(self.args)
        reward.export(self.args)
        reward.audit(self.args)

    def test_complete_three_arm_pipeline_and_light_report(self):
        self.run_to_audit()
        split = shared.load_json(self.work/"split.json")
        self.assertEqual((len(split["train_ids"]), len(split["calibration_ids"]), len(split["target_ids"])), (6, 2, 3))
        self.assertFalse(set(split["train_ids"]) & set(split["target_ids"]))
        reward.train(self.args)
        reward.collect(self.args)
        all_rows = []
        for arm in ARMS:
            path = self.work/"jobs"/f"{arm}_seed0"
            trained = shared.load_json(path/"training.json")
            self.assertEqual(trained["epochs"], 2)
            self.assertEqual(len(trained["history"]), 2)
            self.assertEqual(trained["history"][-1]["optimizer_steps"], 18)
            self.assertTrue((path/"final.pth").exists())
            all_rows.append(trained)
        self.assertEqual(len({r["initial_state_sha256"] for r in all_rows}), 1)
        with zipfile.ZipFile(self.work/"reward_report.zip") as archive:
            self.assertFalse(any(name.endswith((".pt", ".pth", ".npz")) for name in archive.namelist()))
            rows = list(csv.DictReader(io.StringIO(archive.read("results/step_trace.csv").decode())))
            expected_ids = set(split["calibration_ids"]) | set(split["target_ids"])
            for arm in ARMS:
                actual = {r["sample_id"] for r in rows if r["arm"] == arm}
                self.assertEqual(actual, expected_ids)
            self.assertTrue(all(r["image_label"] in ("0", "1") for r in rows))
            self.assertIn("reward_reference_R0", rows[0])
        # A rerun consumes no Base forwards, and doesn't update a completed policy.
        hashes = [shared.sha256_file(self.work/"jobs"/f"{arm}_seed0"/"final.pth") for arm in ARMS]
        reward.train(self.args)
        self.assertEqual(hashes, [shared.sha256_file(self.work/"jobs"/f"{arm}_seed0"/"final.pth") for arm in ARMS])

    def test_epoch_resume_replays_same_rng_optimizer_scheduler(self):
        self.run_to_audit()
        root, cfg, manifest, signature = reward.load_work(self.work)
        split = shared.load_json(root/"split.json")
        def store():
            return EvidenceStore(lambda record: reward.gate.evidence(root, cfg, record, signature), "cpu", 1, 0)
        job = cfg["jobs"][0]
        original = reward.save_resume
        def interrupt_after_first(*args, **kwargs):
            original(*args, **kwargs)
            if args[2] == 0:
                raise RuntimeError("simulated interruption")
        with patch.object(reward, "save_resume", side_effect=interrupt_after_first):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                reward.train_job(root, cfg, manifest, signature, split, store(), job, torch.device("cpu"))
        _, resumed, trained = reward.train_job(root, cfg, manifest, signature, split, store(), job, torch.device("cpu"))
        # A separate output root uses exactly the same sealed evidence/split.
        reference_root = root/"reference"
        _, reference, _ = reward.train_job(reference_root, cfg, manifest, signature, split, store(), job, torch.device("cpu"))
        self.assertEqual(state_hash(resumed.state_dict()), state_hash(reference.state_dict()))
        self.assertEqual(len(trained["history"]), 2)

    def test_config_changes_and_partial_collection_rejected(self):
        self.run_to_audit()
        self.args.epochs = 3
        with self.assertRaisesRegex(ValueError, "different config"):
            reward.prepare(self.args)
        with self.assertRaises((FileNotFoundError, ValueError)):
            reward.collect(self.args)

    def test_reuses_all_verified_source_caches_without_loading_Base(self):
        self.run_to_audit()
        previous = self.work
        self.args.work_dir = str(previous.parent/"reused")
        self.args.source_work_dir = str(previous)
        before = list(self.forwards)
        reward.prepare(self.args)
        with patch.object(sys.modules["test_mara"], "build_models_from_checkpoint",
                          side_effect=AssertionError("Base should not be loaded")):
            reward.export(self.args)
        self.assertEqual(self.forwards, before)
        root, cfg, manifest, signature = reward.load_work(self.args.work_dir)
        self.assertEqual(len(manifest), 12)
        for record in manifest:
            self.assertTrue(shared.cache_ready(root, record, signature))


class RewardShellTests(unittest.TestCase):
    def test_dual_GPU_barriers_and_failure(self):
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_reward.sh")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual([r.split("|")[0] for r in lines],
                         ["prepare", "export", "export", "audit", "train", "train", "collect"])
        self.assertIn("train|gpu=0|shard=0", lines)
        self.assertIn("train|gpu=1|shard=1", lines)
        result, lines = fixtures.EntrypointTests.run_mock_shell(
            self, script_name="run_exp_mara_reward.sh", FAIL_EXPORT_SHARD="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("worker failed", result.stdout+result.stderr)
        self.assertFalse(any(r.startswith(("train", "collect")) for r in lines))

    def test_native_bash_syntax(self):
        import subprocess
        bash = fixtures.native_bash()
        if not bash:
            self.skipTest("native Bash unavailable")
        subprocess.run([str(bash), "-n", "run_exp_mara_reward.sh"], cwd=fixtures.ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
