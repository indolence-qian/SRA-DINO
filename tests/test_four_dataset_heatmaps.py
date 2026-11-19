"""Cached-score heatmap tests; no network, PyTorch, model weights, or GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

import render_four_dataset_heatmaps as app

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "run"
        self.work.mkdir()
        self.rows = []
        rng = np.random.default_rng(12)
        for dataset in app.TARGETS:
            for anomaly in (False, True):
                stem = f"{dataset}_{int(anomaly)}"
                image = self.root / (stem+".png")
                Image.fromarray(rng.integers(20, 150, (160, 320, 3), dtype=np.uint8)).save(image)
                mask = None
                if anomaly:
                    mask = self.root / (stem+"_mask.png")
                    array = np.zeros((160, 320), np.uint8)
                    if dataset != "btad":
                        array[71, 151] = 1  # 0/1 encoding and a tiny annotation
                    Image.fromarray(array).save(mask)
                self.rows.append(dict(id=hashlib.sha256(stem.encode()).hexdigest()[:24], dataset=dataset,
                    category="gear_teeth" if dataset == "ld" else "object", image_path=str(image),
                    image_sha=app.sha256(image), native_size=[320, 160], mask_path=str(mask) if mask else None, partition="eval"))
        app.write_json(self.work / "manifest.json", self.rows)
        self.cfg = dict(protocol=app.PROTOCOL, manifest_sha=app.sha256(self.work / "manifest.json"), eval_size=8)
        app.write_json(self.work / "config.json", self.cfg)
        app.write_json(self.work / "sealed_eval.json", {"fingerprint": app.fingerprint(self.cfg)})
        self.predictions = self.work / "predictions" / "seed42" / "through15"
        self.predictions.mkdir(parents=True)
        decision = self.work / "decisions" / "seed42" / "through15" / "selection.json"
        app.write_json(decision, dict(fingerprint=app.fingerprint(self.cfg), seed=42, horizon=15,
                                      arms={"D": {"thresholds": {"visual": 1., "full": 1.5}}}))
        identity = dict(fingerprint=app.fingerprint(self.cfg), selection=app.sha256(decision),
                        eval_seal=app.sha256(self.work / "sealed_eval.json"), seed=42, horizon=15)
        hashes = {}
        for row in self.rows:
            low = np.full((8, 8), -2., np.float32)
            low[2:5, 3:6] = 3.
            path = self.predictions / (row["id"]+".npz")
            np.savez(path, D_visual=low, D_full=low+.5)
            hashes[row["id"]] = app.sha256(path)
            app.write_json(path.with_suffix(".json"), dict(identity=identity, sha=hashes[row["id"]]))
        app.write_json(self.predictions / "locked.json", dict(identity=identity, predictions=hashes))
        self.args = app.parser().parse_args(["--work-dir", str(self.work), "--max-side", "128", "--pack"])

    def sealed_hashes(self):
        return {str(p): app.sha256(p) for p in self.work.rglob("*") if p.is_file() and "heatmaps" not in p.parts}

    def test_all_four_datasets_real_render_gallery_zip_and_immutable_inputs(self):
        before = self.sealed_hashes()
        with patch.dict("sys.modules", {"torch": None, "cv2": None, "matplotlib": None}):
            output, records = app.run(self.args)
        self.assertEqual(len(records), 8)
        self.assertEqual({r["dataset"] for r in records}, set(app.TARGETS))
        for row in records:
            with Image.open(output / row["files"]["full_heatmap"]) as image:
                self.assertEqual(image.size, (128, 64))
                self.assertGreater(len(np.unique(np.asarray(image).reshape(-1, 3), axis=0)), 8)
            with Image.open(output / row["files"]["panel"]) as image:
                self.assertGreater(image.width, 700)
            with Image.open(output / row["files"]["prediction"]) as image:
                self.assertEqual(set(np.unique(image)), {0, 255})
        self.assertTrue((output / "index.html").is_file())
        self.assertIn('loading="lazy"', (output / "index.html").read_text(encoding="utf-8"))
        with zipfile.ZipFile(output / "heatmaps.zip") as archive:
            self.assertIn("index.html", archive.namelist())
            self.assertFalse(any(name.endswith((".npz", ".pt", ".pth")) for name in archive.namelist()))
        self.assertEqual(before, self.sealed_hashes())

    def test_resume_reuses_image_files_and_rejects_changed_gt(self):
        output, records = app.run(self.args)
        path = output / records[0]["files"]["panel"]
        before = path.stat().st_mtime_ns
        with patch.object(app, "render_sample", side_effect=AssertionError("Re-rendered completed file")):
            app.run(self.args)
        self.assertEqual(before, path.stat().st_mtime_ns)
        row = next(r for r in self.rows if r["mask_path"])
        Path(row["mask_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "GT changed"):
            app.run(self.args)

    def test_source_scaling_is_shared_and_percentiles_do_not_modify_predictions(self):
        maps = {"visual": np.arange(16, dtype=np.float32).reshape(4, 4),
                "full": np.arange(16, dtype=np.float32).reshape(4, 4)+1}
        before = copy.deepcopy(maps)
        display, _ = app.display_maps(maps, {"visual": 2., "full": 3.})
        self.assertTrue(np.allclose(display["visual"], display["full"]))
        self.assertEqual(float(display["visual"][0, 2]), .5)
        display, info = app.display_maps(maps, {}, "percentile")
        self.assertTrue(np.all(display["full"] >= display["visual"]))
        self.assertIn("cannot be compared", info["note"])
        for key in maps:
            self.assertTrue(np.array_equal(before[key], maps[key]))
        constant, _ = app.display_maps({"visual": np.ones((2, 2)), "full": np.ones((2, 2))}, {}, "percentile")
        self.assertFalse(constant["full"].any())

    def test_tiny_and_zero_gt_masks_are_preserved_with_explicit_image_label(self):
        row = next(r for r in self.rows if r["dataset"] == "ld" and r["mask_path"])
        truth, _ = app.load_truth(row, (32, 16), True)
        self.assertTrue(truth.any())
        row = next(r for r in self.rows if r["dataset"] == "btad" and r["mask_path"])
        truth, _ = app.load_truth(row, (32, 16), True)
        self.assertFalse(truth.any())
        self.args.datasets = "btad"
        _, records = app.run(self.args)
        self.assertEqual(sum(r["is_anomaly"] for r in records), 1)

    def test_modified_prediction_and_missing_lock_fail(self):
        row = self.rows[0]
        path = self.predictions / (row["id"]+".npz")
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Prediction changed"):
            app.run(self.args)
        (self.predictions / "locked.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "No locked predictions"):
            app.run(self.args)

    def test_filter_latest_completed_horizon_and_gt_free_render(self):
        (self.work / "predictions" / "seed42" / "through99").mkdir()
        self.args.datasets, self.args.limit_per_category = "ld", 1
        self.args.no_gt = True
        for row in self.rows:
            if row["mask_path"]:
                Path(row["mask_path"]).unlink()
        output, records = app.run(self.args)
        self.assertEqual(len(records), 1)
        self.assertIn("through15", output.parts)
        self.args.categories = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown category"):
            app.run(self.args)

    def test_invalid_style_and_output_location_fail(self):
        self.args.temperature = 0
        with self.assertRaisesRegex(ValueError, "temperature"):
            app.run(self.args)
        self.args.temperature = 1
        self.args.output_dir = self.predictions / "bad"
        with self.assertRaisesRegex(ValueError, "separate heatmap"):
            app.run(self.args)


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH on Windows")
class ShellTests(unittest.TestCase):
    def test_cpu_only_wrapper_quotes_paths_and_forwards_render_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(ROOT / "run_render_four_dataset_heatmaps.sh", root / "run.sh")
            (root / "runtime").write_text('#!/usr/bin/env bash\nprintf "ARG=%s\\n" "$@"\n', encoding="utf-8", newline="\n")
            (root / "runtime").chmod(0o755)
            env = os.environ.copy()
            env.update(TRAIN_PYTHON="./runtime", WORK_DIR="./run with spaces", NORMALIZATION="percentile", DATASETS="ld", PACK="1", SHOW_GT="0")
            result = subprocess.run([BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("ARG=./run with spaces", result.stdout)
            for value in ("--normalization", "percentile", "--pack", "--no-gt"):
                self.assertIn("ARG="+value, result.stdout)
            self.assertNotIn("--device", result.stdout)


if __name__ == "__main__":
    unittest.main()
