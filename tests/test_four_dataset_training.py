"""Tiny real head training with synthetic encoders; no production fake-data mode."""
import copy
import csv
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

import factorial_ablation_experiment as legacy
import train_test_four_datasets as app
import test_residual_text as previous
from tools.residual_text_runtime import make_record
from tools.residual_text import paired_directions
from tools.text_ablation import TextMatcher
from tools.vlm_review import file_digest, load_json


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = previous.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.base / "four_targets"
        self.args = app.parser().parse_args(["--stage", "prepare"])
        for key, value in vars(self.fixture.args).items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
        self.args.work_dir = str(self.root)
        self.args.arm = "D"
        self.args.eval_datasets = ",".join(app.TARGETS)
        self.args.num_shards = 2
        self.args.calibration_images = 2
        self.catalog = list(copy.deepcopy(self.fixture.catalog))
        rng = np.random.default_rng(193)
        for dataset in ("btad", "mpdd", "ld"):
            for index in range(6):
                # LD category intentionally has only anomaly TEST images.
                if dataset == "ld" and index < 2:
                    continue
                path = self.fixture.base / f"{dataset}_{index}.png"
                Image.fromarray(rng.integers(0, 200, (48, 64, 3), dtype=np.uint8)).save(path)
                mask = None
                if index in (2, 3):
                    mask = path.with_name(path.stem+"_mask.png")
                    array = np.zeros((48, 64), np.uint8)
                    if not (dataset == "btad" and index == 3):
                        array[12:24, 20:32] = 255
                    Image.fromarray(array).save(mask)
                partition = "reference" if index >= 4 else "eval"
                row = make_record(dataset, "target", path, mask, partition)
                self.catalog[1 if index >= 4 else 0].append(row)

    def prepare_source(self):
        cfg = app.prepare(self.args, self.catalog)
        self.args.partition = "source"
        self.export()
        app.fit_stats(self.args)
        states = torch.randn(1, 2, 3, 6, generator=torch.Generator().manual_seed(90))
        directions = paired_directions(states[:, 0], states[:, 1]).numpy()
        with patch.object(legacy, "encode_text_bank", return_value=(directions, states.numpy())):
            app.delegate(self.args, "text_bank")
        return cfg

    def export(self):
        with patch.object(app, "FrozenDinoExtractor", previous.FakeExtractor):
            for shard in range(self.args.num_shards):
                self.args.shard_id = shard
                app.export(self.args)
        app.seal(self.args)

    def test_real_head_training_four_targets_empty_mask_and_explicit_image_labels(self):
        native_arms = copy.deepcopy(legacy.ARMS)
        old_code = {name: file_digest(app.REPO / name) for name in ("factorial_ablation_experiment.py", "residual_text_experiment.py")}
        cfg = self.prepare_source()
        self.assertEqual(cfg["text_readout"], "T2")
        self.assertEqual(set(cfg["code"]), set(app.CODE))
        with app.runtime():
            source = legacy.rows(self.root, "source")
            self.assertEqual({r["dataset"] for r in source}, {"visa"})
            self.assertEqual(set(load_json(self.root / "normal_stats.json")["rms"]), {"residual"})
            self.assertIsInstance(app.new_model(self.root, cfg, "text", 42, "cpu"), TextMatcher)
        self.assertFalse((self.root / "features" / "appearance").exists())
        # Delete every anomalous target mask: neither training nor inference may open it.
        masks = {}
        for row in self.catalog[0]:
            if row["partition"] == "eval" and row["mask_path"]:
                path = Path(row["mask_path"])
                masks[path] = path.read_bytes()
                path.unlink()
        for kind in ("density", "text"):
            self.args.kind = kind
            app.delegate(self.args, "train")
        app.delegate(self.args, "select")
        self.args.partition = "eval"
        self.export()
        for shard in range(self.args.num_shards):
            self.args.shard_id = shard
            app.delegate(self.args, "predict")
        with patch.object(app, "evaluation_truth", side_effect=AssertionError("Target GT before lock")):
            app.delegate(self.args, "lock_predictions")
        for path, data in masks.items():
            path.write_bytes(data)
        for shard in range(self.args.num_shards):
            self.args.shard_id = shard
            app.evaluate(self.args)
        result = app.report(self.args)
        self.assertEqual({r["dataset"] for r in result}, set(app.TARGETS))
        self.assertEqual(len(result), 8)  # full network + same-model visual control
        ld = next(r for r in result if r["dataset"] == "ld" and r["score"] == "full")
        self.assertIsNone(ld["I_AUROC"])
        self.assertEqual(ld["I_AUROC_valid_categories"], 0)
        self.assertEqual(ld["annotation_kind"], "bounding_box")
        btad = next(r for r in result if r["dataset"] == "btad" and r["score"] == "full")
        self.assertEqual(btad["images"], 4)  # zero-mask image was not dropped
        info = load_json(self.root / "metrics" / "seed42" / "through1" / "btad__target.json")
        self.assertEqual(info["metrics"][0]["abnormal_images"], 2)
        self.assertTrue(any(r["image_label"] == 1 and r["native_positive"] == 0 for r in info["mask_audit"]))
        with zipfile.ZipFile(self.root / "four_datasets_report.zip") as archive:
            self.assertIn("reports/FOUR_DATASETS.md", archive.namelist())
            self.assertFalse(any(n.endswith((".npz", ".pt", ".pth")) for n in archive.namelist()))
        with app.runtime():
            decision = load_json(legacy.decision_path(self.root, 42, 1))
            self.assertEqual(set(decision["arms"]), {"D"})
            self.assertGreaterEqual(decision["arms"]["D"]["density_epoch"], 1)
            self.assertEqual(decision["arms"]["D"]["strength"], .5)
        self.assertEqual(legacy.ARMS, native_arms)
        self.assertEqual(old_code, {name: file_digest(app.REPO / name) for name in old_code})
        # Repeat completed extraction and training: cache and optimizer state must resume.
        with patch.object(app, "FrozenDinoExtractor", side_effect=AssertionError("Re-exported completed cache")):
            app.export(self.args)
        for kind in ("density", "text"):
            self.args.kind = kind
            app.delegate(self.args, "train")

    def test_config_change_rejects_without_touching_old_run(self):
        app.prepare(self.args, self.catalog)
        before = file_digest(self.root / "config.json")
        self.args.text_readout = "T1"
        with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
            app.prepare(self.args, self.catalog)
        self.assertEqual(file_digest(self.root / "config.json"), before)

    def test_missing_fourth_target_and_target_training_are_rejected(self):
        catalog = copy.deepcopy(self.catalog)
        catalog[0] = [r for r in catalog[0] if r["dataset"] != "ld"]
        with self.assertRaisesRegex(ValueError, "Four complete"):
            app.prepare(self.args, catalog)
        catalog = copy.deepcopy(self.catalog)
        next(r for r in catalog[0] if r["dataset"] == "ld")["partition"] = "source"
        with self.assertRaisesRegex(ValueError, "Only VisA"):
            app.prepare(self.args, catalog)


@unittest.skipUnless(previous.BASH, "Set SRA_TEST_BASH to native Bash on Windows")
class ShellTests(unittest.TestCase):
    def run_script(self, **overrides):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(app.REPO / "run_train_test_four_datasets.sh", root / "run.sh")
            (root / "runtime").write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-CPU} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n', encoding="utf-8", newline="\n")
            (root / "flock").write_text('#!/usr/bin/env bash\nexit 0\n', encoding="utf-8", newline="\n")
            for name in ("runtime", "flock"):
                (root / name).chmod(0o755)
            env = os.environ.copy()
            env.update(TRAIN_PYTHON="./runtime", WORK_DIR="./work", GPU_IDS="0,1")
            env.update(overrides)
            env["PATH"] = str(root)+os.pathsep+str(Path(previous.BASH).parent)+os.pathsep+env.get("PATH", "")
            return subprocess.run([previous.BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=45)

    def test_single_model_source_first_four_datasets_and_correct_gpu_mapping(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        out = result.stdout
        self.assertIn("--eval_datasets mvtec,btad,mpdd,ld", out)
        self.assertIn("--text_readout T2", out)
        self.assertIn("--until_epoch 15", out)
        self.assertIn("--density_epochs 15 --text_epochs 15", out)
        self.assertEqual(sum("--stage train " in line for line in out.splitlines()), 2)
        self.assertLess(out.rindex("--stage train "), out.index("--stage select "))
        first_target_export = next(out.index(line) for line in out.splitlines() if "--stage export " in line and "--partition eval" in line)
        self.assertLess(out.index("--stage select "), first_target_export)
        self.assertLess(out.index("--stage lock_predictions "), out.index("--stage evaluate "))
        self.assertIn("GPU=1", out)

    def test_worker_failure_blocks_target_tests_and_one_gpu_is_supported(self):
        result = self.run_script(FAIL_STAGE="--stage train ")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("--stage predict ", result.stdout)
        result = self.run_script(GPU_IDS="0")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        result = self.run_script(GPU_IDS="0,0")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("--stage prepare ", result.stdout)


if __name__ == "__main__":
    unittest.main()
