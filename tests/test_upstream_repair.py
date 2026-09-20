import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import upstream_repair as repair
from tools.upstream_repair import corrected_probability, constrained_loss, source_metrics, selection_score
from tools.vlm_review import file_digest, load_json
import test_upstream_localization as v1_tests


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_identity_all_probability_ranges_and_gradient(self):
        p = torch.tensor([0., 1e-45, 1e-30, 1e-9, 1e-7, .2, .9, 1.])
        d = torch.zeros_like(p, requires_grad=True)
        q = corrected_probability(p, d)
        self.assertTrue(torch.equal(p, q))
        q.sum().backward()
        self.assertTrue(torch.isfinite(d.grad).all())
        self.assertTrue((d.grad > 0).all())
        self.assertTrue(torch.equal(p, corrected_probability(p, torch.ones_like(p)*100, 0)))

    def test_stable_extreme_corrections_and_endpoint_recovery(self):
        p = torch.tensor([0., 1e-40, .2, .9, 1.])
        for amount in (-10000., -20., 0., 20., 10000.):
            out = corrected_probability(p, torch.full_like(p, amount))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(((out >= 0) & (out <= 1)).all())
        self.assertGreater(float(corrected_probability(p, p*0+20)[0]), .5)
        self.assertLess(float(corrected_probability(p, p*0-20)[-1]), .5)

    def test_no_floor_ranking_loss(self):
        from sklearn.metrics import roc_auc_score
        p = torch.tensor([1e-9, 2e-9, 1e-7, 2e-7])
        labels = [0, 0, 1, 1]
        self.assertEqual(roc_auc_score(labels, p.clamp(1e-5, 1-1e-5)), .5)
        self.assertEqual(roc_auc_score(labels, corrected_probability(p, p*0)), 1.)

    def test_loss_finite_with_zero_base_and_penalizes_bg_increases(self):
        base = torch.zeros(2, 1, 8, 8)
        mask = base.clone()
        mask[0, :, 2:4, 2:4] = 1
        delta = torch.zeros_like(base, requires_grad=True)
        loss, terms = constrained_loss(base, delta, mask)
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertLess(float(delta.grad[0, :, 2:4, 2:4].mean()), 0)
        _, increased = constrained_loss(base, base+20, mask)
        self.assertGreater(float(increased["background"].mean().detach()), float(terms["background"].mean().detach()))

    def test_source_thresholds_conservative_and_guards(self):
        mask = np.zeros((2, 32, 32), bool)
        mask[0, 2, 2] = True
        base = np.full(mask.shape, .1, np.float32)
        new = base.copy()
        new[mask] = .9
        bm = source_metrics(mask, base, base, 42)
        cm = source_metrics(mask, new, base, 42)
        self.assertLessEqual(cm["realized_fpr"], .01)
        self.assertEqual(cm["small_hit_at_fpr"], 1)
        self.assertIsNotNone(selection_score(cm, bm, [cm], [bm]))
        bad = dict(cm, fpr_at_base_threshold=.1)
        self.assertIsNone(selection_score(cm, bm, [bad], [bm]))
        self.assertIsNone(selection_score(dict(cm, pixel_auc_sampled=0), bm, [cm], [bm]))


class PipelineTests(unittest.TestCase):
    # Reuse synthetic v1 CACHE fixture, not the old experiment's test methods.
    review = v1_tests.PipelineTests.review

    def setUp(self):
        v1_tests.PipelineTests.setUp(self)
        v1_tests.PipelineTests.features(self)
        self.source = self.root
        self.output_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_temp.cleanup)
        self.out = Path(self.output_temp.name)
        self.a = repair.parser().parse_args(["--stage", "prepare", "--work_dir", str(self.out),
                                             "--source_work_dir", str(self.source), "--num_shards", "1",
                                             "--epochs", "2", "--batch_size", "2", "--hidden", "8", "--workers", "0"])
        repair.prepare(self.a)

    def test_full_reuse_identity_source_only_training_resume_and_fallback(self):
        snapshot = {str(p.relative_to(self.source)): file_digest(p) for p in self.source.rglob("*") if p.is_file()}
        with patch("torch.cuda.is_available", return_value=False):
            repair.evaluate(self.a, True)
            repair.report(self.a, True)
            rows = load_json(self.out/"p0/summary.json")
            for row in rows:
                if row["mode"] == "zero_residual":
                    b = next(v for v in rows if v["partition"] == row["partition"] and v["category"] == row["category"] and v["mode"] == "base")
                    self.assertEqual(row["PRO_exact"], b["PRO_exact"])
                    self.assertEqual(row["P_AUROC"], b["P_AUROC"])
            original_mask = repair.old.mask_for
            def no_target(r, size):
                self.assertNotEqual(r["partition"], "eval")
                return original_mask(r, size)
            real_score = repair.selection_score
            def only_base(candidate, baseline, *args):
                return real_score(candidate, baseline, *args) if candidate is baseline else None
            with patch.object(repair.old, "mask_for", side_effect=no_target), patch.object(repair, "selection_score", side_effect=only_base):
                for mode in ("visual", "vlm"):
                    self.a.mode = mode
                    repair.train(self.a)
                    checksum = file_digest(self.out/"heads"/mode/"last.pt")
                    repair.train(self.a)
                    self.assertEqual(checksum, file_digest(self.out/"heads"/mode/"last.pt"))
                    selected = torch.load(self.out/"heads"/mode/"best.pt", weights_only=False)
                    self.assertEqual(selected["alpha"], 0)
                    self.assertEqual(selected["selection"], "BASE_FALLBACK")
            repair.evaluate(self.a)
            repair.report(self.a)
            target = load_json(self.out/"results/summary.json")
            for row in target:
                if row["mode"] != "base":
                    self.assertEqual(row["changed_fraction"], 0)
                    self.assertEqual(row["selection"], "BASE_FALLBACK")
        self.assertEqual(snapshot, {str(p.relative_to(self.source)): file_digest(p) for p in self.source.rglob("*") if p.is_file()})

    def test_changed_config_or_nested_output_rejected(self):
        self.a.lr = .02
        with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
            repair.prepare(self.a)
        self.a.work_dir = str(self.source/"child")
        with self.assertRaisesRegex(ValueError, "non-nested"):
            repair.prepare(self.a)

    def test_old_checkpoint_diagnostics_and_accepted_head_selection(self):
        # P0 can inspect actual v1 checkpoints without reinterpreting them as v2.
        head = repair.old.LocalSemanticHead(2, 8, 8)
        with torch.no_grad():
            head.decode[-1].bias.fill_(1.)
        repair.old.save_torch(self.source/"heads/visual/best.pt", dict(state_dict=head.state_dict(), epoch=0,
                              settings=dict(cache_sha=file_digest(self.source/"sealed.json"), hidden=8)))
        with patch("torch.cuda.is_available", return_value=False):
            repair.evaluate(self.a, True)
            repair.report(self.a, True)
            row = next(r for r in load_json(self.out/"p0/summary.json") if r["mode"] == "old_visual")
            self.assertAlmostEqual(row["mean_bg_logit_residual"], 1.)
            self.assertGreater(row["mean_bg_probability_increase"], 0)
            # Exercise persistence of a selected nonzero-alpha trained checkpoint.
            def allow(candidate, baseline, *args):
                return 0. if candidate is baseline else 1.
            with patch.object(repair, "selection_score", side_effect=allow):
                for mode in ("visual", "vlm"):
                    self.a.mode = mode
                    repair.train(self.a)
            selected = torch.load(self.out/"heads/vlm/best.pt", weights_only=False)
            self.assertEqual(selected["selection"], "SOURCE_VALIDATED")
            self.assertEqual(selected["alpha"], .1)
            repair.evaluate(self.a)
            repair.report(self.a)
            row = next(r for r in load_json(self.out/"results/summary.json") if r["mode"] == "vlm")
            self.assertGreater(row["changed_fraction"], 0)


BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH on Windows")
class ShellTests(unittest.TestCase):
    def test_stage_order_two_gpu_ddp_and_fail_stop(self):
        for fail in ("NEVER", "--stage p0"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                shutil.copyfile(repair.old.REPO/"run_exp_upstream_repair.sh", root/"run.sh")
                (root/"source").mkdir()
                (root/"source/sealed.json").write_text("{}")
                (root/"fake_python").write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n', newline="\n")
                (root/"flock").write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
                for p in (root/"fake_python", root/"flock"):
                    p.chmod(0o755)
                env = os.environ.copy()
                env.update(SOURCE_WORK_DIR="./source", WORK_DIR="./out", TRAIN_PYTHON="./fake_python", GPU_IDS="0,1", FAIL_STAGE=fail)
                env["PATH"] = str(root)+os.pathsep+str(Path(BASH).parent)+os.pathsep+env.get("PATH", "")
                result = subprocess.run([BASH, "run.sh"], cwd=root, env=env, text=True, capture_output=True, timeout=30)
                if fail == "NEVER":
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn("GPU=0 ARGS=upstream_repair.py --stage p0", result.stdout)
                    self.assertIn("GPU=1 ARGS=upstream_repair.py --stage p0", result.stdout)
                    self.assertIn("GPU=0,1 ARGS=-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2", result.stdout)
                    self.assertLess(result.stdout.index("--stage p0"), result.stdout.index("--stage train"))
                    self.assertIn("--mode visual", result.stdout)
                    self.assertIn("--mode vlm", result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("--stage train", result.stdout)


if __name__ == "__main__":
    unittest.main()
