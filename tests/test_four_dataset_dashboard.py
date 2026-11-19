import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

import build_four_dataset_dashboard as app
import render_four_dataset_heatmaps as heat
import test_four_dataset_heatmaps as fixtures


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HeatmapTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.work = self.fixture.work
        self.rows = self.fixture.rows
        lock = heat.read_json(self.fixture.predictions / "locked.json")
        for row in self.rows:
            if row["mask_path"] and row["dataset"] != "btad":
                mask = np.zeros((160, 320), np.uint8)
                mask[40:80, 120:200] = 255
                Image.fromarray(mask).save(row["mask_path"])
            visual = np.full((8, 8), -2., np.float32)
            full = visual.copy()
            if row["dataset"] == "mvtec" and row["mask_path"]:
                visual[2:4, 3:5] = full[2:4, 3:5] = 4.
            if row["dataset"] == "btad" and not row["mask_path"]:
                full[1:3, 1:3] = 4.  # a new false positive after fusion
            if row["dataset"] == "mpdd" and row["mask_path"]:
                visual[0:2, 0:2] = full[0:2, 0:2] = 4.  # image TP but wrong location
            if row["dataset"] == "ld" and row["mask_path"]:
                full[2:4, 3:5] = 4.  # text-assisted image and localization improvement
            path = self.fixture.predictions / (row["id"]+".npz")
            np.savez(path, D_visual=visual, D_full=full)
            lock["predictions"][row["id"]] = heat.sha256(path)
            heat.write_json(path.with_suffix(".json"), dict(identity=lock["identity"], sha=heat.sha256(path)))
        heat.write_json(self.fixture.predictions / "locked.json", lock)
        self.args = app.parser().parse_args(["--work-dir", str(self.work), "--max-side", "128"])

    def test_confusion_localization_and_empty_annotation_are_separate(self):
        truth = np.zeros((8, 8), bool)
        truth[2:4, 3:5] = True
        low = np.full((8, 8), -2., np.float32)
        high = low.copy(); high[2:4, 3:5] = 4
        point = app.diagnose({"visual": low, "full": high}, truth, True, {"visual": 1., "full": 1.5})
        self.assertEqual(point["branches"]["visual"]["outcome"], "FN")
        self.assertEqual(point["branches"]["full"]["outcome"], "TP")
        self.assertEqual(point["branches"]["full"]["dice"], 1.)
        self.assertEqual(point["dice_delta"], 1.)
        empty = app.diagnose({"visual": high, "full": high}, np.zeros((8, 8), bool), True, {"visual": 1., "full": 1.5})
        self.assertTrue(empty["empty_anomaly_mask"])
        self.assertIsNone(empty["branches"]["full"]["iou"])
        self.assertEqual(empty["branches"]["full"]["outcome"], "TP")

    def test_end_to_end_html_csv_zip_and_good_bad_cases_without_input_changes(self):
        before = self.fixture.sealed_hashes()
        output, data = app.run(self.args)
        self.assertEqual(len(data["cases"]), 8)
        self.assertEqual({r["full"]["outcome"] for r in data["cases"]}, {"TP", "TN", "FP", "FN"})
        good = next(r for r in data["cases"] if r["dataset"] == "mvtec" and r["label"])
        self.assertEqual(good["full"]["dice"], 1.)
        poor = next(r for r in data["cases"] if r["dataset"] == "mpdd" and r["label"])
        self.assertEqual(poor["full"]["dice"], 0.)
        self.assertEqual(poor["full"]["outcome"], "TP")
        empty = next(r for r in data["cases"] if r["dataset"] == "btad" and r["label"])
        self.assertTrue(empty["empty_anomaly_mask"])
        self.assertIsNone(empty["full"]["dice"])
        ld = next(r for r in data["cases"] if r["dataset"] == "ld" and r["label"])
        self.assertEqual(ld["dice_delta"], 1.)
        self.assertIn("qualityTabs", (output / "index.html").read_text(encoding="utf-8"))
        self.assertNotIn("__DASHBOARD_DATA__", (output / "index.html").read_text(encoding="utf-8"))
        for row in data["cases"]:
            for path in row["files"].values():
                self.assertTrue((output / path).is_file())
        with zipfile.ZipFile(output / "analysis_report.zip") as archive:
            self.assertIn("index.html", archive.namelist())
            self.assertIn("sample_metrics.csv", archive.namelist())
            self.assertFalse(any(name.endswith((".npz", ".pt", ".pth")) for name in archive.namelist()))
        after = {path: heat.sha256(path) for path in before}
        self.assertEqual(before, after)
        self.assertTrue((self.work / "reports" / "HTML_ANALYSIS.html").is_file())
        pointer = heat.read_json(self.work / "reports" / "html_analysis.json")
        self.assertEqual(Path(pointer["index"]), output / "index.html")
        self.assertEqual(pointer["horizon"], 15)

    def test_numpy_truth_matches_existing_torch_evaluation_geometry(self):
        import torch
        from torch.nn import functional as F
        rng = np.random.default_rng(7)
        row = copy.deepcopy(next(r for r in self.rows if r["mask_path"]))
        for h, w, side in ((17, 23, 8), (5, 19, 12), (3, 5, 9), (160, 320, 8)):
            mask = rng.random((h, w)) > .85
            Image.fromarray(mask.astype(np.uint8)).save(row["mask_path"])
            row["native_size"] = [w, h]
            actual, _ = app.evaluation_truth(row, side)
            value = torch.from_numpy(mask.copy()).float()[None, None]
            reduced = min(h, side), min(w, side)
            value = F.adaptive_max_pool2d(value, reduced)
            value = F.interpolate(value, (side, side), mode="nearest")
            self.assertTrue(np.array_equal(actual, value[0, 0].numpy() > 0))

    def test_tied_ranking_and_single_label_auc(self):
        auc, ap = app.binary_ranking([0, 1, 0, 1], [.2, .2, .9, .9])
        self.assertEqual(auc, .5)
        self.assertEqual(ap, .5)
        self.assertIsNone(app.binary_ranking([1, 1], [0, 1])[0])

    def test_published_metrics_are_used_only_for_complete_category_coverage(self):
        rows = [r for r in self.rows if r["dataset"] == "mvtec"]
        path = self.work / "metrics" / "seed42" / "through15" / "mvtec__object.json"
        heat.write_json(path, dict(
            prediction_lock_sha=heat.sha256(self.fixture.predictions / "locked.json"),
            evaluation_mask_hashes={r["id"]: heat.sha256(r["mask_path"]) if r["mask_path"] else None for r in rows},
            metrics=[dict(score=score, P_AP=.75, P_AUROC=.9, PRO_exact=.8, F1_best=.7) for score in ("visual", "full")]))
        self.args.datasets = "mvtec"
        _, data = app.run(self.args)
        self.assertEqual(data["datasets"][0]["full"]["pixel_ap"], .75)
        self.assertEqual(data["categories"][0]["full"]["best_f1"], .7)
        self.args.limit_per_category = 1
        _, preview = app.run(self.args)
        self.assertIsNone(preview["categories"][0]["full"]["pixel_ap"])
        self.assertEqual(preview["coverage"]["selected"], 1)

    def test_json_payload_is_safe_for_untrusted_filename(self):
        text = '</script><script>window.bad=1</script>'
        app.write_html(self.work, {"filename": text})
        html = (self.work / "index.html").read_text(encoding="utf-8")
        payload = html.split('<script type="application/json" id="reportData">', 1)[1].split('</script>', 1)[0]
        self.assertNotIn("<script>window.bad", html)
        self.assertEqual(json.loads(payload)["filename"], text)


@unittest.skipUnless(fixtures.BASH, "Set SRA_TEST_BASH on Windows")
class ShellTests(unittest.TestCase):
    def test_future_entry_runs_experiment_then_html_and_can_rebuild_html_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(fixtures.ROOT / "run_four_datasets_with_html.sh", root / "run.sh")
            for name, stage in (("run_train_test_four_datasets.sh", "EXPERIMENT"), ("run_build_four_dataset_dashboard.sh", "HTML")):
                (root / name).write_text(f'#!/usr/bin/env bash\necho {stage}\n[[ "${{FAIL_STAGE:-}}" != {stage} ]]\n', encoding="utf-8", newline="\n")
            env = os.environ.copy()
            env.update(PIPELINE_MODE="train", WORK_DIR=str(root / "new_run"))
            first = subprocess.run([fixtures.BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stdout+first.stderr)
            self.assertLess(first.stdout.index("EXPERIMENT"), first.stdout.index("HTML"))
            env.pop("PIPELINE_MODE", None)
            env["RUN_EXPERIMENT"] = "0"
            second = subprocess.run([fixtures.BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotIn("EXPERIMENT", second.stdout)
            self.assertIn("HTML", second.stdout)
            env.update(RUN_EXPERIMENT="1", FAIL_STAGE="EXPERIMENT")
            failed = subprocess.run([fixtures.BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            self.assertNotIn("HTML", failed.stdout)

    def test_auto_mode_uses_cache_or_existing_weights_but_never_starts_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(fixtures.ROOT / "run_four_datasets_with_html.sh", root / "run.sh")
            (root / "run_train_test_four_datasets.sh").write_text('#!/usr/bin/env bash\necho EXPERIMENT\n', encoding="utf-8", newline="\n")
            (root / "run_build_four_dataset_dashboard.sh").write_text('#!/usr/bin/env bash\necho "HTML RESUME=$RESUME_PREDICTIONS"\n', encoding="utf-8", newline="\n")
            work = root / "work with spaces"
            env = os.environ.copy()
            for key in ("PIPELINE_MODE", "RUN_EXPERIMENT", "TRAIN_UNTIL_EPOCH", "TRAIN_SEED"):
                env.pop(key, None)
            env["WORK_DIR"] = str(work)
            def run():
                return subprocess.run([fixtures.BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            missing = run()
            self.assertNotEqual(missing.returncode, 0)
            self.assertNotIn("EXPERIMENT", missing.stdout)
            work.mkdir()
            (work / "config.json").write_text('{}', encoding="utf-8")
            resume = run()
            self.assertEqual(resume.returncode, 0, resume.stdout+resume.stderr)
            self.assertIn("HTML RESUME=1", resume.stdout)
            self.assertNotIn("EXPERIMENT", resume.stdout)
            lock = work / "predictions" / "seed42" / "through5" / "locked.json"
            lock.parent.mkdir(parents=True)
            lock.write_text('{}', encoding="utf-8")
            cached = run()
            self.assertEqual(cached.returncode, 0, cached.stdout+cached.stderr)
            self.assertIn("HTML RESUME=0", cached.stdout)
            self.assertNotIn("EXPERIMENT", cached.stdout)
            env["TRAIN_UNTIL_EPOCH"] = "15"
            requested = run()
            self.assertIn("HTML RESUME=1", requested.stdout)
            env["PIPELINE_MODE"] = "bad"
            self.assertNotEqual(run().returncode, 0)

    def test_explicit_train_allocates_new_directory_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(fixtures.ROOT / "run_four_datasets_with_html.sh", root / "run.sh")
            for name, text in (("run_train_test_four_datasets.sh", 'echo "TRAIN WORK=$WORK_DIR"'),
                               ("run_build_four_dataset_dashboard.sh", 'echo "HTML RESUME=$RESUME_PREDICTIONS"')):
                (root / name).write_text('#!/usr/bin/env bash\n'+text+'\n', encoding="utf-8", newline="\n")
            env = os.environ.copy()
            env.pop("WORK_DIR", None)
            env.update(PIPELINE_MODE="train")
            result = subprocess.run([fixtures.BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("TRAIN WORK=./checkpoint/residual_hr_text_four_datasets_20", result.stdout)
            self.assertNotIn("four_datasets_v1", result.stdout)
            self.assertLess(result.stdout.index("TRAIN WORK="), result.stdout.index("HTML RESUME=0"))

    def test_prediction_recovery_passes_run_train_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(fixtures.ROOT / "run_build_four_dataset_dashboard.sh", root / "run.sh")
            (root / "run_train_test_four_datasets.sh").write_text('#!/usr/bin/env bash\necho "RECOVER TRAIN=$RUN_TRAIN PREDICT=$RUN_PREDICT EVAL=$RUN_EVAL"\n', encoding="utf-8", newline="\n")
            (root / "runtime").write_text('#!/usr/bin/env bash\necho HTML\n', encoding="utf-8", newline="\n")
            (root / "runtime").chmod(0o755)
            env = os.environ.copy()
            env.update(TRAIN_PYTHON="./runtime", RESUME_PREDICTIONS="1")
            result = subprocess.run([fixtures.BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("RECOVER TRAIN=0 PREDICT=1 EVAL=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
