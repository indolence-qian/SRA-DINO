import csv
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import upstream_p2 as p2
import upstream_p2_fast as fast
from tools.upstream_p2 import MatchingHead, component_loss, component_supervision, spatial_matching
from tools.upstream_p2_fast import CachedMatchingHead, ValidationPlan, fast_loss, geometry, batch_plan_indices
from tools.upstream_repair import constrained_loss, source_metrics
from tools.vlm_review import file_digest, load_json
import test_upstream_localization as v1_tests


def settings():
    return dict(min_area=2, near_radius=2, small_weight=2., small_fraction=.01, rank_pixels=7, rank_margin=1.,
                hard_fraction=.03, bg_weight=1., residual_weight=.01, component_weight=.1, ranking_weight=.1)


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cached_head_matches_dynamic_head_outputs_and_gradients(self):
        torch.manual_seed(6)
        original, cached = MatchingHead(2, 8, 8), CachedMatchingHead(2, 8, 8)
        with torch.no_grad():
            original.decode[-1].weight.normal_(std=.1)
        cached.load_state_dict(original.state_dict())
        visual, base = torch.randn(2, 2, 8, 4, 4), torch.rand(2, 1, 16, 16)
        emb, valid = torch.randn(2, 2, 2, 8), torch.tensor([[[1., 1.], [1., 0.]]]*2)
        boxes = torch.tensor([[[0., 0., .6, 1.], [.4, 0., 1., 1.]]]*2)
        match = spatial_matching(visual, emb, valid, boxes)
        a, b = original(visual, base, emb, valid, boxes), cached.forward_cached(visual, base, match)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward()
        b.sum().backward()
        for x, y in zip(original.parameters(), cached.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_vectorized_p2a_p2b_losses_and_gradients_match(self):
        torch.manual_seed(4)
        s = settings()
        base, mask = torch.rand(3, 1, 32, 32)*.8+.01, torch.zeros(3, 1, 32, 32)
        mask[0, 0, 4, 4:6] = 1
        mask[0, 0, 7:11, 7:11] = 1
        mask[1] = 1  # no background/ranking; normal third image has zero extras
        gs = [geometry(gt.numpy(), s) for gt in mask[:, 0]]
        labels = torch.stack([component_supervision(gt.numpy(), s["min_area"]) for gt in mask[:, 0]])
        for mode in ("p2a", "p2b"):
            d = torch.randn_like(base)*.5
            x, y = d.clone().requires_grad_(), d.clone().requires_grad_()
            kwargs = {k: s[k] for k in ("bg_weight", "residual_weight", "hard_fraction")}
            if mode == "p2a":
                expected, terms = constrained_loss(base, x, mask, **kwargs)
            else:
                options = {k: s[k] for k in ("component_weight", "ranking_weight", "small_fraction", "small_weight", "near_radius", "rank_pixels", "rank_margin")}
                expected, terms = component_loss(base, x, mask, labels, **options, **kwargs)
            actual, observed = fast_loss(base, y, mask, gs, mode, s)
            torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)
            for key in terms:
                torch.testing.assert_close(terms[key], observed[key], rtol=1e-6, atol=1e-7)
            expected.sum().backward()
            actual.sum().backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-7)

    def test_validation_plan_exact_metrics_and_batch_gathers(self):
        rng = np.random.default_rng(5)
        masks = np.zeros((3, 32, 32), bool)
        masks[0, 2, 3] = 1
        masks[1, 8:12, 3:8] = 1
        base, pred = rng.random(masks.shape).astype(np.float32), rng.random(masks.shape).astype(np.float32)
        plan = ValidationPlan(masks, 42, 512, .01)
        sample, gt = plan.gather(pred)
        base_sample, _ = plan.gather(base)
        expected = source_metrics(masks, pred, base, 42, 512, .01)
        actual = plan.metrics(sample, gt, base_sample)
        self.assertEqual(expected, actual)
        collected, positives = [], []
        for start, end in ((0, 2), (2, 3)):
            s, g = batch_plan_indices(plan, start, end, 32*32)
            values = pred[start:end].ravel()
            collected.extend(values[s.numpy()])
            positives.extend(values[g.numpy()])
        np.testing.assert_array_equal(collected, sample)
        np.testing.assert_array_equal(positives, gt)


class PipelineTests(unittest.TestCase):
    review = v1_tests.PipelineTests.review

    def setUp(self):
        v1_tests.PipelineTests.setUp(self)
        v1_tests.PipelineTests.features(self)
        self.source = self.root
        self.output_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_temp.cleanup)
        self.out = Path(self.output_temp.name)
        self.a = fast.parser().parse_args(["--stage", "prepare", "--source_work_dir", str(self.source), "--work_dir", str(self.out),
                    "--cpu", "--precision", "fp32", "--epochs", "2", "--until_epoch", "1", "--hidden", "8",
                    "--batch_size", "2", "--micro_batch", "2", "--num_shards", "1", "--cpu_threads", "1", "--pack_workers", "2"])
        fast.prepare(self.a)

    def test_source_only_pack_and_six_controls_resume_without_changing_cache(self):
        snapshot = {str(p.relative_to(self.source)): file_digest(p) for p in self.source.rglob("*") if p.is_file()}
        original = fast.p2.old.mask_for
        def no_target(r, size):
            self.assertNotEqual(r["partition"], "eval")
            return original(r, size)
        store = None
        try:
            with patch.object(fast.p2.old, "mask_for", side_effect=no_target):
                fast.diagnostics(self.a)
                fast.pack(self.a)
                root, _, cfg, s, protocol = fast.context(self.a)
                meta = fast.verify_pack(root, protocol)
                self.assertEqual({r["partition"] for r in meta["rows"]}, {"train", "val"})
                store = fast.Store(root, cfg, s, torch.device("cpu"), self.a)
                for loss_mode, mode in fast.JOBS:
                    with patch.object(fast, "selection_score", side_effect=lambda c, b, *args: 0. if c is b else None):
                        fast.train_head(self.a, store, loss_mode, mode)
                        path = self.out/"heads"/loss_mode/mode/"last.pt"
                        checksum = file_digest(path)
                        fast.train_head(self.a, store, loss_mode, mode)
                        self.assertEqual(checksum, file_digest(path))
                        self.a.until_epoch = 2
                        fast.train_head(self.a, store, loss_mode, mode)
                        self.assertEqual(torch.load(path, weights_only=False)["epoch"], 1)
                        self.a.until_epoch = 1
                fast.screen_report(self.a)
                self.assertEqual(len(load_json(self.out/"source_screen.json")), 6)
            models = {}
            for loss_mode, mode in fast.JOBS:
                saved = torch.load(self.out/"heads"/loss_mode/mode/"best.pt", weights_only=False)
                model = fast.CachedMatchingHead(2, 8, 8).eval()
                model.load_state_dict(saved["state_dict"])
                models[f"{loss_mode}/{mode}"] = model, saved
            hashes = fast.head_hashes(self.out)
            original_infer = fast.infer_batch
            inference_ooms = []
            def limited_inference(arrays, shuffled, start, end, *args):
                if end-start > 1:
                    inference_ooms.append(end-start)
                    raise torch.cuda.OutOfMemoryError("Simulated inference memory pressure")
                return original_infer(arrays, shuffled, start, end, *args)
            for key in fast.p2.repair.groups_for(self.source, {"val", "eval"}):
                with patch.object(fast, "infer_batch", side_effect=limited_inference):
                    fast.evaluate_group(self.a, key, torch.device("cpu"), models, hashes)
            self.assertEqual(inference_ooms, [2, 2])
            fast.report(self.a)
            rows = load_json(self.out/"results/summary.json")
            self.assertTrue(all(r["changed_fraction"] == 0 for r in rows))
            self.assertEqual(snapshot, {str(p.relative_to(self.source)): file_digest(p) for p in self.source.rglob("*") if p.is_file()})
        finally:
            if store:
                store.arrays.clear()
                store = None
                gc.collect()

    def test_microbatch_accumulation_preserves_effective_update(self):
        fast.pack(self.a)
        root, _, cfg, s, protocol = fast.context(self.a)
        store = fast.Store(root, cfg, s, torch.device("cpu"), self.a)
        try:
            torch.manual_seed(1)
            first, second = fast.CachedMatchingHead(2, 8, 8), fast.CachedMatchingHead(2, 8, 8)
            second.load_state_dict(first.state_dict())
            indices = store.train_indices[:2]
            x, y = torch.optim.SGD(first.parameters(), lr=.01), torch.optim.SGD(second.parameters(), lr=.01)
            a = fast.effective_step(first, x, store, indices, "real", "p2b", s, protocol, 2)
            b = fast.effective_step(second, y, store, indices, "real", "p2b", s, protocol, 1)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
            for p, q in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
        finally:
            store.arrays.clear()
            gc.collect()

    def test_precision_change_and_old_output_are_rejected(self):
        self.a.precision = "bf16"
        self.a.cpu = False
        with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
            fast.prepare(self.a)
        with tempfile.TemporaryDirectory() as temp:
            self.a.work_dir = temp
            self.a.precision = "fp32"
            p2.prepare(self.a)
            with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
                fast.prepare(self.a)

    def test_training_oom_backoff_keeps_updates_and_optimizer_oom_is_not_retried(self):
        fast.pack(self.a)
        root, _, cfg, s, protocol = fast.context(self.a)
        store = fast.Store(root, cfg, s, torch.device("cpu"), self.a)
        original_batch = store.batch
        seen, updates = [], []
        original_step = torch.optim.AdamW.step
        def limited_batch(indices, mode, supervised=False):
            if supervised:
                seen.append(len(indices))
                if len(indices) > 1:
                    raise torch.cuda.OutOfMemoryError("Simulated training memory pressure")
            return original_batch(indices, mode, supervised)
        def count_step(optimizer, *args, **kwargs):
            updates.append(1)
            return original_step(optimizer, *args, **kwargs)
        try:
            with patch.object(store, "batch", side_effect=limited_batch), patch.object(torch.optim.AdamW, "step", new=count_step):
                fast.train_head(self.a, store, "p2b", "real")
            self.assertEqual(seen, [2, 1, 1, 1, 1])
            self.assertEqual(len(updates), 2)
            last = torch.load(root/"heads/p2b/real/last.pt", weights_only=False)
            self.assertEqual(last["history"][-1]["effective_batch"], 2)
            self.assertEqual(last["history"][-1]["micro_batch"], 1)
            model = fast.CachedMatchingHead(2, 8, 8)
            optimizer = torch.optim.AdamW(model.parameters())
            with patch.object(optimizer, "step", side_effect=torch.cuda.OutOfMemoryError("Optimizer OOM")) as step:
                with self.assertRaisesRegex(RuntimeError, "Optimizer OOM: stop"):
                    fast.effective_step(model, optimizer, store, store.train_indices[:2], "real", "p2b", s, protocol, 1)
                self.assertEqual(step.call_count, 1)
        finally:
            store.arrays.clear()
            gc.collect()

    def test_mutated_tensor_pack_is_rejected(self):
        fast.pack(self.a)
        root, _, _, _, protocol = fast.context(self.a)
        with (root/"packed/base.npy").open("ab") as stream:
            stream.write(b"mutation")
        with self.assertRaisesRegex(ValueError, "modified"):
            fast.verify_pack(root, protocol)

    def test_reuses_completed_prior_diagnostics_readonly(self):
        with tempfile.TemporaryDirectory() as temp:
            prior = p2.parser().parse_args(["--stage", "prepare", "--source_work_dir", str(self.source), "--work_dir", temp,
                                           "--epochs", "2", "--num_shards", "1", "--hidden", "8"])
            p2.prepare(prior)
            p2.quality(prior)
            with patch("torch.cuda.is_available", return_value=False):
                p2.matching(prior)
            before = {str(p.relative_to(temp)): file_digest(p) for p in Path(temp).rglob("*") if p.is_file()}
            self.a.diagnostic_work_dir = temp
            with patch.object(p2, "quality", side_effect=AssertionError("Must reuse completed diagnostics")):
                fast.diagnostics(self.a)
            self.assertEqual(load_json(self.out/"diagnostics.json")["directory"], str(Path(temp).resolve()))
            self.assertEqual(before, {str(p.relative_to(temp)): file_digest(p) for p in Path(temp).rglob("*") if p.is_file()})

    def test_spawn_worker_queue_trains_and_evaluates_complete_matrix(self):
        fast.diagnostics(self.a)
        fast.pack(self.a)
        fast.run_pool(self.a, "train")
        fast.run_pool(self.a, "evaluate")
        fast.report(self.a)
        self.assertEqual(len(load_json(self.out/"source_screen.json")), 6)
        self.assertEqual({r["completed_epochs"] for r in load_json(self.out/"source_screen.json")}, {1})


BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH on Windows")
class ShellTests(unittest.TestCase):
    def test_independent_worker_stage_order_resume_limit_and_fail_stop(self):
        for fail in ("NEVER", "--stage pack"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                shutil.copyfile(p2.old.REPO/"run_exp_upstream_p2_fast.sh", root/"run.sh")
                (root/"source").mkdir()
                (root/"source/sealed.json").write_text("{}")
                (root/"fake_python").write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n', newline="\n")
                (root/"flock").write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
                for path in (root/"fake_python", root/"flock"):
                    path.chmod(0o755)
                env = dict(os.environ, TRAIN_PYTHON="./fake_python", SOURCE_WORK_DIR="./source", WORK_DIR="./output",
                           GPU_IDS="0,1", TRAIN_UNTIL_EPOCH="5", FAIL_STAGE=fail)
                env["PATH"] = ".:"+env.get("PATH", "") if os.name != "nt" else str(root)+os.pathsep+env.get("PATH", "")
                result = subprocess.run([BASH, "run.sh"], cwd=root, env=env, text=True, capture_output=True)
                output = result.stdout+result.stderr
                if fail == "NEVER":
                    self.assertEqual(result.returncode, 0, output)
                    order = [output.index("--stage "+stage) for stage in ("prepare", "diagnostics", "pack", "train", "evaluate", "report")]
                    self.assertEqual(order, sorted(order))
                    self.assertNotIn("torch.distributed.run", output)
                    self.assertIn("--num_shards 2", output)
                    self.assertIn("--epochs 15 --until_epoch 5", output)
                    self.assertIn("GPU=0,1 ARGS=upstream_p2_fast.py --stage train", output)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("--stage train", output)
                    self.assertNotIn("--stage report", output)


if __name__ == "__main__":
    unittest.main()
