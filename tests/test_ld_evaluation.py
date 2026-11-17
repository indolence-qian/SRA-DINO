"""CPU tests for the LD runner; no downloaded weights or claimed GPU results."""
import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

import evaluate_ld as app
import test_ld_dataset as fixtures
from tools.prepare_ld_dataset import convert

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


def payload(mara=False):
    value = {key: {} for key in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner")}
    if mara:
        value.update(mara_agent={}, mara_config={})
    return value


class CheckpointTests(unittest.TestCase):
    def test_latest_is_numeric_and_not_selected_from_test_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("2.pth", "10.pth", "99_best.pth"):
                (root / name).touch()
            self.assertEqual(app.resolve_checkpoint(root).name, "10.pth")
            (root / "mara_epoch_003.pth").touch()
            with self.assertRaisesRegex(ValueError, "Mixed Base/MARA"):
                app.resolve_checkpoint(root)
            self.assertEqual(app.resolve_checkpoint(root, "mara").name, "mara_epoch_003.pth")

    def test_single_tower_or_incomplete_weights_fail(self):
        self.assertEqual(app.checkpoint_mode(payload()), "base")
        self.assertEqual(app.checkpoint_mode(payload(True)), "mara")
        with self.assertRaisesRegex(ValueError, "single-tower"):
            app.checkpoint_mode(dict(payload(), base_arch="dino_single"))
        with self.assertRaisesRegex(ValueError, "missing"):
            app.checkpoint_mode({"density": {}})

    def test_archive_extracts_only_data_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            archive = parent / "data.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                for name in ("LD_AD_20261116/manifest.json", "ld_adapter/do_not_install.py"):
                    info = tarfile.TarInfo(name)
                    info.size = 2
                    stream.addfile(info, io.BytesIO(b"[]"))
            app.ensure_data(parent / "converted", archive)
            self.assertTrue((parent / "converted" / "manifest.json").exists())
            self.assertFalse((parent / "ld_adapter").exists())
            with tarfile.open(archive, "w:gz") as stream:
                info = tarfile.TarInfo("LD_AD_20261116/../escape.txt")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "Invalid data archive"):
                app.ensure_data(parent / "other", archive)
            self.assertFalse((parent / "escape.txt").exists())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        fixture = fixtures.LDDataTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.root = fixture.root
        # Match the real Cam5 edge case: all TEST images are anomalous.
        cam5 = fixture.source / "轴面5only_20261113"
        csv_path = cam5 / "split.csv"
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keep = []
        for row in rows:
            if row["split"] == "val" and int(row["boxes"]) < 2:
                for directory, suffix in (("images", ".jpg"), ("labels", ".txt")):
                    (cam5 / directory / "val" / (row["stem"]+suffix)).unlink()
            else:
                keep.append(row)
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(keep)
        convert(fixture.source, fixture.output)
        self.data = fixture.output
        ckpt = self.root / "14.pth"
        torch.save(payload(), ckpt)
        dino_repo = self.root / "dino"
        dino_repo.mkdir()
        (dino_repo / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
        self.args = app.parser().parse_args([
            "--weight-path", str(ckpt), "--data-root", str(self.data),
            "--work-dir", str(self.root / "work"), "--device", "cpu", "--workers", "0",
            "--image-size", "32", "--batch-size", "2", "--pro-num-th", "30",
            "--dino-repo-dir", str(dino_repo), "--dino-weights", str(ckpt), "--clip-weights", str(ckpt),
            "--vis-per-category", "1",
        ])

    @staticmethod
    def predictor(batch, index):
        size = batch["image"].shape[-1]
        values = torch.sigmoid(batch["image"].mean(1)).numpy().astype(np.float32)
        return dict(base=values.reshape(len(batch["image"]), size, size))

    def run_pipeline(self):
        cfg, state = app.prepare(self.args)
        app.infer(cfg, self.args, state, self.predictor)
        return cfg, app.report(cfg, self.args)

    def test_full_pipeline_single_label_auroc_zip_and_resume(self):
        cfg, result = self.run_pipeline()
        cam5 = next(r for r in result["categories"] if r["category"] == "shaft_cam5")
        self.assertIsNone(cam5["I_AUROC"])
        self.assertEqual(result["summary"]["base"]["macro"]["I_AUROC"]["valid_categories"], 3)
        self.assertEqual(result["summary"]["base"]["images"], 10)
        self.assertTrue((self.args.work_dir / "REPORT.md").is_file())
        with zipfile.ZipFile(self.args.work_dir / "ld_test_report.zip") as stream:
            self.assertIn("metrics.json", stream.namelist())
            self.assertFalse(any(name.endswith(".npz") for name in stream.namelist()))
            self.assertEqual(len([name for name in stream.namelist() if name.endswith(".png")]), 4)
        # Complete categories are reused and no further inference is allowed.
        def fail(*args):
            raise AssertionError("Recomputed completed predictions")
        app.infer(cfg, self.args, {}, fail)

    def test_changed_settings_cannot_mix_existing_predictions(self):
        self.run_pipeline()
        self.args.threshold = .7
        with self.assertRaisesRegex(ValueError, "WORK_DIR settings"):
            app.prepare(self.args)

    def test_tampered_predictions_are_rejected(self):
        cfg, _ = self.run_pipeline()
        path = self.args.work_dir / "predictions" / "shaft_cam5.npz"
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "modified"):
            app.report(cfg, self.args)

    def test_image_label_is_not_derived_from_resized_mask(self):
        cfg, state = app.prepare(self.args)
        app.infer(cfg, self.args, state, self.predictor)
        path = self.args.work_dir / "predictions" / "shaft_cam5.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        data["masks"][:] = False
        with path.open("wb") as stream:
            np.savez_compressed(stream, **data)
        # The cached explicit image label remains positive even for a vanished mask.
        rows = [r for r in cfg["rows"] if r["category"] == "shaft_cam5"]
        loaded = app.read_prediction(path, cfg, rows)
        self.assertEqual(loaded["labels"].tolist(), [1])


class NetworkDelegationTests(unittest.TestCase):
    def test_mara_delegates_existing_agent_without_gt_oracle_or_weight_updates(self):
        from dataclasses import asdict
        from tools.mara_agent import MARAAgent, MARAConfig
        torch.set_num_threads(1)
        adapter = torch.nn.Module()
        for name in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter"):
            setattr(adapter, name, torch.nn.Linear(1, 1))
        clip, prompt, dino = [torch.nn.Linear(1, 1) for _ in range(3)]
        agent = MARAAgent(MARAConfig(num_layers=2, map_size=8, num_regions=1, roi_size=4, hidden_dim=8, max_steps=1))
        state = {name: getattr(adapter, name).state_dict() for name in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter")}
        state.update(prompt_learner=prompt.state_dict(), mara_agent=agent.state_dict(), mara_config=asdict(agent.cfg))
        before = {key: value.clone() for key, value in agent.state_dict().items()}
        def builder(args, device, payload):
            args.visual_layers, args.hfa_layers_runtime = (2, 4), ()
            args.evidence_feature_channels = args.evidence_semantic_channels = args.evidence_semantic_global_dim = 0
            return clip, prompt, dino, None, adapter, agent, None
        def forward(clip, info, device, model, dino, prompt, index, **kwargs):
            self.assertFalse(bool(info["mask"].any()))
            self.assertFalse(bool(info["is_anomaly"].any()))
            self.assertTrue(kwargs["return_evidence"])
            return None, info["mask"], torch.full((1, 2, 16, 16), .5), torch.zeros(1, 2), {}
        def evidence(**kwargs):
            self.assertFalse(kwargs["include_full_resolution_oracle"])
            return dict(layer_maps=torch.full((1, 2, 8, 8), .5), extra_maps=None, global_evidence=None)
        settings = dict(image_size=512, dino_repo_dir="repo", dino_weights="dino.pth", clip_weights="clip.pt",
                        dino_model_name="dinov3_vitl16", hfa_setting="hfa3")
        replacements = {
            "tools.utils_up": types.SimpleNamespace(get_anomaly_map=forward),
            "test_mara": types.SimpleNamespace(build_models_from_checkpoint=builder, load_mara_checkpoint=lambda *a: None),
            "tools.mara_evidence": types.SimpleNamespace(build_mara_evidence=evidence),
        }
        with patch.dict("sys.modules", replacements):
            model = app.ExistingNetwork(dict(settings=settings, mode="mara", weight_path="mara_epoch_029.pth"), state, torch.device("cpu"))
            result = model(dict(image=torch.ones(1, 3, 16, 16), mask=torch.ones(1, 1, 16, 16),
                                is_anomaly=torch.ones(1, dtype=torch.long), image_path=["query.jpg"]), 0)
        self.assertEqual(set(result), {"base", "mara"})
        self.assertEqual(result["mara"].shape, (1, 16, 16))
        self.assertTrue(all(torch.equal(before[key], value) for key, value in agent.state_dict().items()))
        self.assertFalse(agent.training)

    def test_base_delegates_existing_builders_strict_loads_and_hides_labels(self):
        def adapter(*args):
            model = torch.nn.Module()
            for key in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter"):
                setattr(model, key, torch.nn.Linear(1, 1))
            return model
        modules = []
        def linear(*args):
            result = torch.nn.Linear(1, 1)
            modules.append(result)
            return result
        trained = adapter()
        prompt = torch.nn.Linear(1, 1)
        state = {key: getattr(trained, key).state_dict() for key in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter")}
        state["prompt_learner"] = prompt.state_dict()
        def runtime(args, payload):
            self.assertEqual(payload["dino_weights"], "relocated.pth")
            args.visual_layers = (2, 4)
            args.hfa_layers_runtime = ()
        def forward(clip, info, device, model, dino, prompt, index, **kwargs):
            self.assertFalse(bool(info["mask"].any()))
            self.assertFalse(bool(info["is_anomaly"].any()))
            self.assertEqual(kwargs["visual_layers"], (2, 4))
            maps = torch.full((1, 2, 4, 4), .5)
            return None, info["mask"], maps, torch.zeros(1, 2)
        settings = dict(image_size=512, dino_repo_dir="repo", dino_weights="relocated.pth", clip_weights="clip.pt",
                        dino_model_name="dinov3_vitl16", hfa_setting="hfa3")
        replacements = {
            "tools.utils_up": types.SimpleNamespace(get_anomaly_map=forward),
            "train_mara": types.SimpleNamespace(apply_base_runtime_config=runtime, create_visual_backbone_for_mara=lambda *a: (linear(), None)),
            "train_up": types.SimpleNamespace(create_clip_model=linear, create_prompt_learner=lambda *a: prompt, create_adapter_model=adapter),
        }
        with patch.dict("sys.modules", replacements):
            model = app.ExistingNetwork(dict(settings=settings, mode="base", weight_path="14.pth"), state, torch.device("cpu"))
            output = model(dict(image=torch.ones(1, 3, 4, 4), mask=torch.ones(1, 1, 4, 4),
                                is_anomaly=torch.ones(1, dtype=torch.long), image_path=["query.jpg"]), 0)
        self.assertEqual(output["base"].shape, (1, 4, 4))
        self.assertTrue(all(not m.training for m in modules))
        for parameter in model.adapter.parameters():
            self.assertFalse(parameter.requires_grad)


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH to native Bash on Windows")
class ShellTests(unittest.TestCase):
    def test_quoted_paths_failure_and_argument_forwarding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(ROOT / "run_test_ld.sh", root / "run.sh")
            (root / "runtime").write_text('#!/usr/bin/env bash\nprintf "ARG=%s\\n" "$@"\nexit "${FAIL_CODE:-0}"\n', encoding="utf-8", newline="\n")
            (root / "flock").write_text('#!/usr/bin/env bash\nexit 0\n', encoding="utf-8", newline="\n")
            for name in ("runtime", "flock"):
                (root / name).chmod(0o755)
            env = os.environ.copy()
            env.update(WEIGHT_PATH="./weights with spaces/mara_epoch_029.pth", WORK_DIR="./work with spaces",
                       TRAIN_PYTHON="./runtime", LD_DATA_ROOT="./data with spaces", GPU_ID="1")
            env["PATH"] = str(root)+os.pathsep+str(Path(BASH).parent)+os.pathsep+env.get("PATH", "")
            result = subprocess.run([BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("ARG=./weights with spaces/mara_epoch_029.pth", result.stdout)
            self.assertIn("ARG=--clip-weights", result.stdout)
            env["FAIL_CODE"] = "7"
            result = subprocess.run([BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 7, result.stdout+result.stderr)
            env["GPU_ID"] = "0,1"
            result = subprocess.run([BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GPU_ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
