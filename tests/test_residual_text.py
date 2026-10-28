import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import residual_text_experiment as app
import tools.residual_text_runtime as runtime
from tools.reference_highres import fixed_projection
from tools.residual_text import (
    ResidualDensity, ResidualFlow, ResidualText, SignedConstraintor, coverage_maps,
    density_loss, fuse_views, image_geometry, multi_positive_loss, nearest_residual,
    paired_directions, positive_text_fusion, text_loss,
)
from tools.vlm_review import atomic_json, file_digest, load_json


class ResidualCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_exact_nearest_and_chunk_stable_ties(self):
        query = F.normalize(torch.randn(2, 9, 8), dim=-1)
        bank = F.normalize(torch.randn(2, 17, 8), dim=-1)
        bank[:, 5] = bank[:, 0]
        query[:, 0] = bank[:, 0]
        a = nearest_residual(query, bank, 2, 3)
        b = nearest_residual(query, bank, 99, 99)
        expected = (query @ bank.transpose(1, 2)).argmax(-1)
        self.assertTrue(torch.equal(a[2], expected))
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertTrue(torch.allclose(a[0], b[0]))
        self.assertEqual(int(a[2][0, 0]), 0)
        self.assertEqual(float(a[0][:, 0].abs().max()), 0.)
        selected = torch.stack([bank[i, expected[i]] for i in range(2)])
        self.assertTrue(torch.allclose(a[0], query - selected))
        self.assertTrue(torch.allclose(a[1], a[0].norm(dim=-1)))

    def test_support_from_all_images_is_usable(self):
        first = F.normalize(torch.randn(1, 4, 8), dim=-1)
        second = F.normalize(torch.randn(1, 4, 8), dim=-1)
        residual, _, index = nearest_residual(second, torch.cat([first, second], 1))
        self.assertEqual(float(residual.abs().max()), 0.)
        self.assertTrue((index >= 4).all())

    def test_flow_inverse_and_true_log_jacobian(self):
        model = ResidualFlow(4, 8, 3).double()
        for block in model.blocks:
            torch.nn.init.normal_(block.net[-1].weight, std=.12)
            torch.nn.init.normal_(block.net[-1].bias, std=.08)
        x = torch.randn(5, 4, dtype=torch.double)
        z, determinant = model(x)
        rebuilt, inverse = model.inverse(z)
        self.assertTrue(torch.allclose(x, rebuilt, atol=1e-9))
        self.assertTrue(torch.allclose(determinant, -inverse, atol=1e-9))
        jacobian = torch.autograd.functional.jacobian(lambda a: model(a)[0], x[0])
        self.assertAlmostEqual(float(torch.linalg.slogdet(jacobian)[1]), float(determinant[0]), places=8)

    def test_flow_nll_gaussian_initialization_and_gradients(self):
        model = ResidualFlow(6, 8, 2)
        x = torch.randn(7, 6)
        expected = .5 * (x.square() + np.log(2 * np.pi)).mean(-1)
        self.assertTrue(torch.allclose(model.nll(x), expected, atol=1e-6))
        model.nll(x).mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_signed_constraintor_preserves_zero_and_starts_at_identity(self):
        model = SignedConstraintor(4, 8)
        x = torch.randn(9, 4)
        self.assertTrue(torch.equal(x, model(x)))
        optimizer = torch.optim.SGD(model.parameters(), .1)
        model(x).square().mean().backward()
        optimizer.step()
        self.assertEqual(float(model(torch.zeros_like(x)).abs().max()), 0.)
        self.assertTrue((model(x) < 0).any())

    def test_density_loss_and_scores_use_residual_only(self):
        model = ResidualDensity(2, 4, 8, 2)
        residual = torch.randn(2, 5, 2, 4, 3, 3)
        coverage = torch.zeros(2, 5, 3, 3)
        coverage[0, :, 1, 1] = .01
        loss = density_loss(model, residual, coverage, tokens=3)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertEqual(tuple(model(residual).shape), (2, 5, 2, 3, 3))
        with self.assertRaises(TypeError):
            model(residual, query=torch.randn_like(residual))

    def test_paired_text_directions_and_degenerate_rejection(self):
        normal, changed = torch.randn(3, 2, 7), torch.randn(3, 2, 7)
        expected = F.normalize((F.normalize(changed, dim=-1) - F.normalize(normal, dim=-1)).mean(1), dim=-1)
        self.assertTrue(torch.allclose(paired_directions(normal, changed), expected))
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            paired_directions(normal, normal)

    def test_text_zero_after_training_and_real_direction_dependency(self):
        model = ResidualText(2, 4, torch.randn(3, 7), 8)
        residual = torch.randn(1, 5, 2, 4, 3, 3)
        coverage = torch.zeros(1, 5, 3, 3)
        coverage[:, :, 1, 1] = 1
        typed = torch.zeros(1, 5, 2, 3, 3)
        typed[:, :, 0, 1, 1] = 1
        loss = text_loss(model, residual, coverage, typed)
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.Adam(model.parameters(), .01)
        loss.backward()
        optimizer.step()
        score, _ = model(residual)
        zero, _ = model(torch.zeros_like(residual))
        self.assertEqual(float(zero.abs().max()), 0.)
        model.directions.neg_()
        inverse, _ = model(residual)
        self.assertTrue(torch.allclose(inverse, -score))
        self.assertNotIn("directions", dict(model.named_parameters()))
        self.assertTrue(all(not name.endswith("bias") for name, _ in model.named_parameters()))

    def test_multi_positive_labels_are_not_false_negatives(self):
        logits = torch.tensor([[1., 1., -1.], [2., 0., 0.]], requires_grad=True)
        positive = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
        loss = multi_positive_loss(logits, positive)
        self.assertLess(float(loss), float(F.cross_entropy(logits[:1], torch.tensor([0]))))
        loss.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)

    def test_tiny_native_defect_survives_coverage_pooling(self):
        mask = torch.zeros(80, 100)
        mask[1, 1] = 1
        boxes = image_geometry(100, 80)
        coverage = coverage_maps(mask, boxes, 4)
        self.assertGreater(float(coverage[0].max()), 0.)
        self.assertLess(float(coverage[0].max()), 1.)
        self.assertTrue((coverage[0] == 0).any())

    def test_late_hann_fusion_and_exact_text_fallback(self):
        boxes = torch.tensor([[[0., 0., 1., 1.], [0., 0., .6, .6], [.4, 0., 1., .6],
                                [0., .4, .6, 1.], [.4, .4, 1., 1.]]])
        maps = torch.full((1, 5, 1, 3, 3), 3.)
        maps[:, 0] = 1
        combined = fuse_views(maps, boxes, 19)
        self.assertTrue(torch.allclose(combined, torch.full_like(combined, 1.5), atol=1e-6))
        self.assertTrue(torch.allclose(fuse_views(maps, boxes, 19, high=False), torch.ones_like(combined), atol=1e-6))
        raw = torch.randn(3, 3)
        self.assertIs(positive_text_fusion(raw, torch.full_like(raw, float("nan")), 0), raw)
        self.assertTrue((positive_text_fusion(raw, torch.randn_like(raw), .5) >= raw).all())

    def test_negative_normal_text_baseline_never_boosts_zero_residual(self):
        item = dict(coverage=torch.zeros(5, 2, 2), residual=torch.zeros(5, 1, 4, 2, 2))
        with patch.object(app, "raw_scores", return_value=-torch.ones(1, 1, 1, 2, 2)):
            calibration = app.fit_calibration(None, "text", [item], torch.ones(1, 2), "G",
                                                dict(layers=[0]), torch.device("cpu"))
        self.assertEqual(calibration["center"][0][0], 0.)
        zero = app.apply_calibration(torch.zeros(1, 1, 1, 2, 2), calibration)
        self.assertEqual(float(zero.clamp_min(0).max()), 0.)


class FakeExtractor:
    """Only used in tests. No production synthetic extraction option exists."""
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.projection = fixed_projection(cfg["input_dim"], cfg["feature_dim"], cfg["data_seed"])

    def view(self, image):
        side = self.cfg["image_size"] // self.cfg["patch_size"]
        image = np.asarray(image.resize((side * 2, side * 2))).copy()
        tensor = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255
        rgb = F.adaptive_avg_pool2d(tensor, (side, side))[0].permute(1, 2, 0).reshape(-1, 3)
        features = torch.cat([rgb, rgb.square(), rgb[:, :1] * rgb[:, 1:2], torch.ones(len(rgb), 1)], 1)
        return torch.stack([F.normalize(features + i * .02, dim=-1) for i in range(len(self.cfg["layers"]))])

    def views(self, image):
        boxes = image_geometry(*image.size, self.cfg["tile_fraction"])
        return [self.view(image if i == 0 else image.crop(box)) for i, box in enumerate(boxes)], boxes

    def close(self):
        pass


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "run"
        self.dino_weights = self.base / "dino.pth"
        self.clip_weights = self.base / "clip.pt"
        self.dino_weights.write_bytes(b"test-only")
        self.clip_weights.write_bytes(b"test-only")
        self.args = app.parser().parse_args(["--stage", "prepare", "--work_dir", str(self.root),
            "--device", "cpu", "--dino_weights", str(self.dino_weights), "--clip_weights", str(self.clip_weights),
            "--dino_repo_dir", str(app.REPO / "dinov3"), "--visual_layers", "0,1", "--input_dim", "8",
            "--feature_dim", "4", "--patch_size", "8", "--image_size", "32", "--eval_size", "32",
            "--selection_size", "16", "--reference_count", "2", "--density_epochs", "2", "--text_epochs", "2",
            "--until_epoch", "1", "--batch_size", "2", "--micro_batch", "1", "--workers", "0",
            "--flow_blocks", "2", "--hidden", "8", "--tokens_per_view", "4", "--calibration_images", "4",
            "--precision", "fp32"])
        self.catalog = self.make_catalog()

    def tearDown(self):
        self.temporary.cleanup()

    def make_catalog(self):
        queries, references = [], []
        generator = np.random.default_rng(3)
        for dataset, category in (("visa", "sourceA"), ("visa", "sourceB"), ("mvtec", "target")):
            for index in range(6):
                image = self.base / f"{dataset}_{category}_{index}.png"
                array = generator.integers(0, 180, (48, 64, 3), dtype=np.uint8)
                mask_path = None
                if index in (2, 3):
                    array[8:22, 10:26, 0] = 255
                    mask = np.zeros((48, 64), np.uint8)
                    mask[8:22, 10:26] = 255
                    mask_path = self.base / f"{dataset}_{category}_{index}_mask.png"
                    Image.fromarray(mask).save(mask_path)
                Image.fromarray(array).save(image)
                partition = "reference" if index >= 4 else "source" if dataset == "visa" else "eval"
                row = runtime.make_record(dataset, category, image, mask_path, partition)
                (references if index >= 4 else queries).append(row)
        return queries, references

    def prepare_export(self):
        cfg = app.prepare(self.args, copy.deepcopy(self.catalog))
        with patch.object(app, "FrozenDinoExtractor", FakeExtractor):
            for partition in ("source", "eval"):
                self.args.partition = partition
                app.export(self.args)
                app.seal(self.args)
        app.fit_stats(self.args)
        generator = torch.Generator().manual_seed(90)
        prompts = load_json(self.root / "prompts.json")
        states = torch.randn(len(prompts), 2, 3, 6, generator=generator)
        directions = paired_directions(states[:, 0], states[:, 1]).numpy()
        with patch.object(app, "encode_text_bank", return_value=(directions, states.numpy())):
            app.text_bank(self.args)
        return cfg

    def train_all(self):
        for kind in ("density", "text"):
            for variant in ("G", "HR"):
                self.args.variant = variant
                app.train(self.args, kind)
        app.select(self.args)

    def test_prepare_idempotence_class_holdout_and_all_support(self):
        cfg = app.prepare(self.args, copy.deepcopy(self.catalog))
        app.prepare(self.args, copy.deepcopy(self.catalog))
        rows = app.rows_for(self.root)
        train_classes = {r["category"] for r in rows if r["partition"] == "train"}
        val_classes = {r["category"] for r in rows if r["partition"] == "val"}
        self.assertFalse(train_classes & val_classes)
        self.assertEqual(len(load_json(self.root / "references.json")), 6)
        self.assertTrue(all(r["mask_sha"] is None for r in rows if r["partition"] == "eval"))
        before = file_digest(self.root / "manifest.json")
        self.args.feature_dim = 3
        with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
            app.prepare(self.args, copy.deepcopy(self.catalog))
        self.assertEqual(file_digest(self.root / "manifest.json"), before)
        self.assertEqual(app.checked_config(self.root)["feature_dim"], cfg["feature_dim"])

    def test_reference_duplicate_exclusion_is_enforced(self):
        queries, references = copy.deepcopy(self.catalog)
        references[0]["rgb_sha"] = queries[0]["rgb_sha"]
        with self.assertRaisesRegex(ValueError, "distinct normal TRAIN"):
            app.prepare(self.args, (queries, references))

    def test_generic_only_and_source_region_annotation_guards(self):
        cfg = app.prepare(self.args, copy.deepcopy(self.catalog))
        self.assertEqual(cfg["semantic_scope"], "generic binary direction")
        target = next(r for r in self.catalog[0] if r["partition"] == "eval" and r["mask_path"])
        annotation = self.base / "invalid_semantics.json"
        atomic_json(annotation, dict(version=1, types=[dict(id="crack", normal="intact", abnormal="cracked")],
                                     regions=[dict(image_path=target["image_path"], mask_path=target["mask_path"], type_id="crack")]))
        with self.assertRaisesRegex(ValueError, "source anomalous"):
            runtime.load_annotations(annotation, app.rows_for(self.root))

    def test_explicit_source_semantics_generate_typed_supervision(self):
        annotation = self.base / "semantics.json"
        regions = []
        for row in self.catalog[0]:
            if row["partition"] == "source" and row["mask_path"]:
                regions.append(dict(image_path=row["image_path"], mask_path=row["mask_path"],
                                    type_id="crack" if "_2.png" in row["image_path"] else "contamination"))
        atomic_json(annotation, dict(version=1,
            types=[dict(id="crack", normal="an intact surface", abnormal="a cracked surface"),
                   dict(id="contamination", normal="a clean surface", abnormal="a contaminated surface")], regions=regions))
        self.args.semantic_annotations = str(annotation)
        cfg = self.prepare_export()
        self.assertEqual(cfg["semantic_scope"], "source-region-supervised change directions")
        dataset = app.ResidualDataset(self.root, cfg, app.rows_for(self.root, "train"), labels=True)
        self.assertEqual(dataset[0]["typed"].shape[1], 2)
        self.assertTrue(any(float(x["typed"].sum()) > 0 for x in dataset))

    def test_seal_rejects_modified_cache_and_targets_cannot_be_labeled(self):
        cfg = self.prepare_export()
        with self.assertRaisesRegex(ValueError, "Target supervision"):
            app.ResidualDataset(self.root, cfg, app.rows_for(self.root, "eval"), labels=True)
        row = app.rows_for(self.root, "train")[0]
        path = app.feature_path(self.root, row)
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            app.ResidualDataset(self.root, cfg, [row])

    def test_full_cpu_workflow_no_target_masks_before_lock_and_resume(self):
        target_paths = {r["mask_path"] for r in self.catalog[0] if r["partition"] == "eval" and r["mask_path"]}
        original_open = Image.open
        def guarded_open(path, *args, **kwargs):
            if str(path) in target_paths:
                self.assertTrue((app.prediction_dir(self.root, 42, self.args.until_epoch) / "locked.json").exists(),
                                "Target mask was opened before predictions locked")
            return original_open(path, *args, **kwargs)
        with patch.object(Image, "open", side_effect=guarded_open):
            cfg = self.prepare_export()
            for variant in ("G", "HR"):
                self.args.variant = variant
                app.train(self.args, "density")
            density_sha = file_digest(app.snapshot_path(self.root, 42, "HR", "density", 1))
            for variant in ("G", "HR"):
                self.args.variant = variant
                app.train(self.args, "text")
            self.assertEqual(density_sha, file_digest(app.snapshot_path(self.root, 42, "HR", "density", 1)))
            app.select(self.args)
            app.audit(self.args)
            app.predict(self.args)
            with self.assertRaises(FileNotFoundError):
                app.evaluate(self.args)
            app.lock_predictions(self.args)
            app.evaluate(self.args)
            app.report(self.args)
        self.assertTrue((self.root / "reports/through1/summary.csv").exists())
        self.assertTrue(list((self.root / "panels").rglob("*.png")))
        snapshot = app.snapshot_path(self.root, 42, "HR", "density", 1)
        before = file_digest(snapshot)
        self.args.until_epoch = 2
        self.train_all()
        self.assertEqual(file_digest(snapshot), before)
        self.assertTrue(app.snapshot_path(self.root, 42, "HR", "density", 2).exists())
        app.train(self.args, "text")  # idempotent completed-stage resume
        self.args.until_epoch = 1
        app.train(self.args, "density")  # explicitly reuse the older immutable snapshot
        decision = load_json(app.decision_dir(self.root, 42, 1) / "selection.json")
        self.assertIn(decision["variants"]["HR"]["strength"], cfg["strengths"])
        for selected in decision["variants"].values():
            self.assertEqual(selected["threshold_resolution"], cfg["eval_size"])
            if selected["text_epoch"] == 0:
                self.assertEqual(selected["strength"], 0.)

    def test_resumed_and_uninterrupted_training_match(self):
        self.prepare_export()
        second_root = self.base / "uninterrupted"
        shutil.copytree(self.root, second_root)
        self.args.variant = "G"
        for kind in ("density", "text"):
            self.args.work_dir = str(self.root)
            self.args.until_epoch = 1
            app.train(self.args, kind)
            self.args.until_epoch = 2
            app.train(self.args, kind)
            self.args.work_dir = str(second_root)
            app.train(self.args, kind)
            resumed = runtime.read_torch(app.head_dir(self.root, 42, "G", kind) / "last.pt")
            uninterrupted = runtime.read_torch(app.head_dir(second_root, 42, "G", kind) / "last.pt")
            for key, value in resumed["state_dict"].items():
                self.assertTrue(torch.equal(value, uninterrupted["state_dict"][key]), key)

    def test_direct_frozen_backbone_hooks_without_legacy_adapters(self):
        cfg = app.prepare(self.args, copy.deepcopy(self.catalog))
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch = torch.nn.Conv2d(3, 8, 8, 8)
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])
                self.norm = torch.nn.LayerNorm(8)
                self.n_storage_tokens = 4
            def forward(self, x):
                x = self.patch(x).flatten(2).transpose(1, 2)
                x = torch.cat([x.new_zeros(len(x), 5, 8), x], 1)
                for block in self.blocks:
                    x = block(x)
                return self.norm(x)
        model = TinyBackbone()
        with patch.object(torch.hub, "load", return_value=model) as load:
            extractor = runtime.FrozenDinoExtractor(cfg, torch.device("cpu"))
            with Image.open(self.catalog[0][0]["image_path"]) as image:
                tokens = extractor.view(image)
            self.assertEqual(tuple(tokens.shape), (2, 16, 8))
            self.assertTrue(torch.allclose(tokens.norm(dim=-1), torch.ones(2, 16), atol=1e-6))
            self.assertFalse(tokens.requires_grad)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertFalse(model.training)
            self.assertEqual(load.call_args.kwargs["source"], "local")
            extractor.close()
            self.assertTrue(all(not block._forward_hooks for block in model.blocks))

    def test_frozen_text_encoder_uses_paired_prompts_and_no_image_tower(self):
        cfg = app.prepare(self.args, copy.deepcopy(self.catalog))
        class TinyText(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Linear(99, 99)
                self.text = torch.nn.Linear(5, 6, bias=False)
            def encode_text(self, tokens, cnn, device, adapter, normalize=True):
                self.assertions = (cnn, adapter, normalize)
                return F.normalize(self.text(tokens.float()), dim=-1)
        model = TinyText()
        openai, tokenizer = types.ModuleType("CLIP.openai"), types.ModuleType("CLIP.tokenizer")
        openai.load_openai_model = lambda *a, **k: model
        tokenizer.tokenize = lambda texts: torch.tensor([[len(t), t.count("a"), t.count("o"), t.count("n"), 1] for t in texts])
        with patch.dict(sys.modules, {"CLIP.openai": openai, "CLIP.tokenizer": tokenizer}):
            directions, states = runtime.encode_text_bank(cfg, load_json(self.root / "prompts.json"), torch.device("cpu"))
        self.assertEqual(directions.shape, (1, 6))
        self.assertEqual(states.shape, (1, 2, 3, 6))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertIsInstance(model.visual, torch.nn.Identity)
        self.assertEqual(model.assertions, ("", None, True))


BASH = os.environ.get("ALIGNMENT_TEST_BASH") or os.environ.get("SRA_TEST_BASH")


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH to Git Bash")
class ShellTests(unittest.TestCase):
    def run_script(self, fail="NEVER", gpus="0,1", **flags):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(app.REPO / "run_exp_residual_text.sh", root / "run.sh")
            (root / "fake_python").write_bytes(b'#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]] || exit 1\nexit 0\n')
            (root / "flock").write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
            for name in ("fake_python", "flock"):
                (root / name).chmod(0o755)
            env = os.environ.copy()
            env.update(TRAIN_PYTHON="./fake_python", WORK_DIR="./output", GPU_IDS=gpus, FAIL_STAGE=fail)
            env.update(flags)
            env["PATH"] = str(root) + os.pathsep + str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
            return subprocess.run([BASH, "run.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=45)

    def test_stage_order_two_gpus_and_prediction_lock_before_evaluation(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("text_bank", "fit_stats", "train_density", "train_text", "select", "audit", "predict", "lock_predictions", "evaluate", "report"):
            self.assertIn("--stage " + name, result.stdout)
        self.assertIn("GPU=1", result.stdout)
        self.assertLess(result.stdout.rindex("--stage train_density"), result.stdout.index("--stage train_text"))
        self.assertLess(result.stdout.rindex("--stage predict"), result.stdout.index("--stage lock_predictions"))
        self.assertLess(result.stdout.index("--stage lock_predictions"), result.stdout.index("--stage evaluate"))
        self.assertIn("--reference_count 4", result.stdout)

    def test_worker_error_stops_pipeline_and_duplicate_gpu_rejected(self):
        result = self.run_script(fail="--stage train_text")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("--stage predict", result.stdout)
        self.assertNotIn("--stage select", result.stdout)
        self.assertNotEqual(self.run_script(gpus="0,0").returncode, 0)

    def test_single_gpu_resume_flags(self):
        result = self.run_script(gpus="0", RUN_TRAIN="0", RUN_EXPORT="0", RUN_PREDICT="0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("--stage train_text", result.stdout)
        self.assertNotIn("--stage export", result.stdout)
        self.assertNotIn("--stage predict ", result.stdout)
        self.assertIn("--stage evaluate", result.stdout)


if __name__ == "__main__":
    unittest.main()
