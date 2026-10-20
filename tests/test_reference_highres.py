import json
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

import reference_highres_experiment as pipeline
from tools.reference_highres import (
    ARMS, EvidenceHead, FrozenExtractor, arm_inputs, assemble_tiles, covering_boxes, fixed_projection,
    match_reference, normalized_box, patch_coordinates, stitch_tiles, supervision_loss,
)
from tools.upstream_repair import corrected_probability
from tools.vlm_review import atomic_json, file_digest, load_json


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native_coverage_overlap_and_constant_reconstruction(self):
        for width, height in ((101, 79), (1, 3), (2048, 1000)):
            boxes = covering_boxes(width, height)
            covered = np.zeros((height, width), bool)
            for x1, y1, x2, y2 in boxes:
                covered[y1:y2, x1:x2] = True
            self.assertTrue(covered.all())
            normalized = [normalized_box(b, width, height) for b in boxes]
            rebuilt = stitch_tiles([torch.full((2, 4, 4), 3.) for _ in boxes], normalized, 17)
            self.assertTrue(torch.allclose(rebuilt, torch.full_like(rebuilt, 3.), atol=1e-6))
        with self.assertRaises(ValueError):
            covering_boxes(100, 100, 2, .4)
        with self.assertRaisesRegex(ValueError, "uncovered"):
            stitch_tiles([torch.ones(1, 2, 2)], [(0, 0, .2, .2)], 10)
        # Left/right evidence must remain in its original coordinates after stitching.
        joined = stitch_tiles([torch.zeros(1, 3, 3), torch.ones(1, 3, 3)],
                              [(0, 0, .6, 1), (.4, 0, 1, 1)], 20)
        self.assertEqual(float(joined[..., 0].max()), 0)
        self.assertEqual(float(joined[..., -1].min()), 1)

    def test_correspondence_position_residual_and_chunk_equivalence(self):
        torch.manual_seed(2)
        query = F.normalize(torch.randn(2, 16, 12), dim=-1)
        reference = query.clone()
        xy = patch_coordinates(4)
        small = match_reference(query, reference, xy, xy, topk=1, radius=.1, chunk_size=3)
        self.assertTrue(torch.allclose(small[0], torch.zeros_like(query), atol=1e-7))
        self.assertEqual(float(small[1].max()), 0)
        self.assertEqual(float(small[2].max()), 0)
        a = match_reference(query, reference, xy, xy, topk=3, radius=0, chunk_size=3)
        b = match_reference(query, reference, xy, xy, topk=3, radius=0, chunk_size=100)
        for x, y in zip(a, b):
            self.assertTrue(torch.allclose(x, y, atol=2e-6))
        with self.assertRaisesRegex(ValueError, "Too few"):
            match_reference(query, reference, xy, xy, topk=3, radius=.01)
        # The signed difference and its magnitude preserve anomaly strength.
        changed = -query
        residual, distance, _ = match_reference(changed, reference, xy, xy, topk=1, radius=.1)
        self.assertTrue(torch.allclose(residual, -2 * query, atol=1e-6))
        self.assertTrue(torch.allclose(distance, torch.full_like(distance, 2.), atol=1e-6))

    def arrays(self):
        arrays = {}
        for prefix, grid in (("global", 4), ("high", 8)):
            for key in ("query", "residual"):
                arrays[prefix + "_" + key] = torch.randn(2, 2, 4, grid, grid)
            for key in ("distance", "uncertainty", "prior"):
                arrays[prefix + "_" + key] = torch.rand(2, 2, grid, grid)
            arrays[prefix + "_base"] = torch.full((2, 1, 16, 16), .1 if prefix == "global" else .3)
        return arrays

    def test_factorial_input_isolation_and_identity(self):
        arrays = self.arrays()
        original = {arm: arm_inputs(arrays, arm, 8, .25) for arm in ARMS}
        self.assertTrue(torch.equal(original[ARMS[0]]["base"], arrays["global_base"]))
        self.assertTrue(torch.allclose(original[ARMS[1]]["base"], torch.full_like(arrays["global_base"], .15)))
        for arm in ARMS[:2]:
            self.assertEqual(float(original[arm]["residual"].abs().sum()), 0)
            self.assertEqual(float(original[arm]["reference_valid"].sum()), 0)
        for arm in ARMS[2:]:
            self.assertGreater(float(original[arm]["residual"].abs().sum()), 0)
        for key in list(arrays):
            if key.endswith(("residual", "distance", "uncertainty")):
                arrays[key] += 100
        for arm in ARMS[:2]:
            after = arm_inputs(arrays, arm, 8, .25)
            for key in after:
                self.assertTrue(torch.equal(after[key], original[arm][key]))
        projection = fixed_projection(12, 4)
        self.assertTrue(torch.equal(projection, fixed_projection(12, 4)))
        self.assertTrue(torch.allclose(projection.T @ projection, torch.eye(4) * 3, atol=1e-5))
        # Includes endpoints and very low probabilities: zero delta is bit-for-bit identity.
        base = torch.tensor([0., 1e-12, .3, 1.]).reshape(1, 1, 2, 2)
        self.assertTrue(torch.equal(corrected_probability(base, torch.zeros_like(base)), base))

    def test_head_alignment_gradient_and_recovery(self):
        torch.manual_seed(4)
        inputs = arm_inputs(self.arrays(), ARMS[3], 8, .25)
        head = EvidenceHead(2, 4, torch.randn(2, 8), hidden=8)
        mask = torch.zeros(2, 1, 16, 16)
        mask[0, :, 4:8, 4:8] = 1
        optimizer = torch.optim.Adam(head.parameters(), lr=.003)
        self.assertTrue(torch.equal(corrected_probability(inputs["base"], head(inputs)["delta"]), inputs["base"]))
        initial = None
        for _ in range(6):
            optimizer.zero_grad()
            output = head(inputs)
            loss = supervision_loss(inputs["base"], output, mask).mean()
            initial = float(loss.detach()) if initial is None else initial
            loss.backward()
            self.assertGreater(float(head.to_text.weight.grad.abs().sum()), 0)
            optimizer.step()
        self.assertLess(float(supervision_loss(inputs["base"], head(inputs), mask).mean().detach()), initial)
        self.assertGreater(float(head.reference.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.visual.weight.grad.abs().sum()), 0)

    def test_frozen_extractor_preserves_base_math_and_raw_matching_space(self):
        class PairProjection(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = torch.nn.Linear(input_dim, 768, bias=False)
            def forward(self, value):
                return self.linear(value), value
        class Dino(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
                self.norm = torch.nn.LayerNorm(1024)
                self.n_storage_tokens = 4
            def forward(self, image):
                value = torch.arange(21 * 1024, dtype=torch.float32).reshape(1, 21, 1024)
                value = (value * .013).sin() + image.mean() * (value * .021).cos()
                for block in self.blocks:
                    value = block(value)
                return value
        torch.manual_seed(5)
        dino = Dino()
        adapter = torch.nn.Module()
        adapter.patch_token_adapter = torch.nn.ModuleList([PairProjection(1024) for _ in range(4)])
        adapter.prompt_adapter = torch.nn.ModuleList([PairProjection(768) for _ in range(2)])
        helper = types.ModuleType("utils")
        text = F.normalize(torch.randn(768, 2), dim=0)
        helper.get_text_features_with_prompt_learner = lambda *a, **kw: text
        cfg = dict(image_size=64, feature_dim=4, seed=42)
        with patch.dict(sys.modules, {"utils": helper}), patch("dual_vlm.load_dual", return_value=(None, None, dino, adapter, (0, 1, 2, 3))):
            extractor = FrozenExtractor(cfg, torch.device("cpu"))
            result = extractor.view(Image.new("RGB", (101, 79), (30, 70, 110)))
        full = []
        with torch.inference_mode():
            for index, layer in enumerate(extractor.layers):
                patches = extractor.captured[layer]
                self.assertTrue(torch.equal(result["raw"][index], F.normalize(patches[0], dim=-1)))
                standardized = (patches - patches.mean(1, keepdim=True)) / (patches.std(1, keepdim=True) + 1e-6)
                projected = F.normalize(adapter.patch_token_adapter[index](standardized)[0], dim=-1)
                logits = (100 * projected @ extractor.text.T).transpose(1, 2).reshape(1, 2, 4, 4)
                full.append(F.interpolate(logits, (64, 64), mode="bilinear", align_corners=True).softmax(1)[0, 1])
        self.assertTrue(torch.equal(result["base"], torch.stack(full).mean(0)[None]))
        extractor.close()
        self.assertTrue(all(not block._forward_hooks for block in dino.blocks))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matching_and_head_backward_smoke(self):
        arrays = {key: value.cuda() for key, value in self.arrays().items()}
        inputs = arm_inputs(arrays, ARMS[3], 8, .25)
        head = EvidenceHead(2, 4, torch.randn(2, 8), hidden=8).cuda()
        mask = torch.zeros_like(inputs["base"])
        mask[0, :, 3:6, 3:6] = 1
        loss = supervision_loss(inputs["base"], head(inputs), mask).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(head.to_text.weight.grad).all())
        tokens = F.normalize(torch.randn(2, 16, 12, device="cuda"), dim=-1)
        xy = patch_coordinates(4, device="cuda")
        residual, distance, uncertainty = match_reference(tokens, tokens, xy, xy, topk=1, radius=.1)
        self.assertLess(float(distance.max()), 1e-6)
        self.assertTrue(torch.isfinite(residual).all() and torch.isfinite(uncertainty).all())


class FakeExtractor:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.layers = (0, 1, 2, 3)
        self.text = torch.eye(2, 8).to(device)
        self.project = fixed_projection(1024, cfg["feature_dim"]).to(device)

    def image_views(self, image):
        side = self.cfg["image_size"] // 16
        def view(im):
            # Raw pixels only. No paths, masks, labels or Base ROI selection.
            color = float(np.asarray(im).mean()) / 255
            raw = torch.zeros(4, side * side, 1024, device=self.device)
            raw[..., 0], raw[..., 1] = 1., color
            raw = F.normalize(raw, dim=-1)
            return dict(raw=raw, prior=torch.full((4, side, side), .2 + color * .1, device=self.device),
                        base=torch.full((1, self.cfg["image_size"], self.cfg["image_size"]), .2, device=self.device))
        boxes = covering_boxes(image.width, image.height, self.cfg["tile_grid"], self.cfg["tile_fraction"])
        return view(image), [view(image.crop(b)) for b in boxes], [normalized_box(b, image.width, image.height) for b in boxes]

    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(12)
        self.entries = {}
        for dataset in ("visa", "mvtec"):
            for split, count in (("test", 8 if dataset == "visa" else 4), ("train", 2)):
                entries = []
                for index in range(count):
                    image = self.root / f"{dataset}_{split}_{index}.png"
                    Image.fromarray(rng.integers(0, 255, (79, 101, 3), dtype=np.uint8)).save(image)
                    mask = None
                    if split == "test" and index % 2:
                        mask = self.root / f"mask_{dataset}_{index}.png"
                        target = np.zeros((79, 101), np.uint8)
                        target[5:10, 7:12] = 255
                        Image.fromarray(target).save(mask)
                    state = ("Normal" if dataset == "visa" else "good") if mask is None else "defect"
                    entries.append(("object", state, str(image), str(mask) if mask else None))
                self.entries[(dataset, split)] = entries
        module = types.ModuleType("Datasets")
        entries = self.entries
        class Dataset:
            def __init__(self, source, split, **kwargs):
                self.data_to_iterate = entries[(source, split)]
        splits = types.SimpleNamespace(TRAIN="train", TEST="test")
        module.DATASET_REGISTRY = {n: (Dataset, splits, n) for n in ("visa", "mvtec")}
        module.DATASET_CLASSES = {n: ["object"] for n in ("visa", "mvtec")}
        self.module = module
        base = self.root / "base.pth"
        torch.save({k: {} for k in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner")}, base)
        dino = self.root / "dino.pth"
        dino.write_bytes(b"fake frozen DINO")
        self.args = pipeline.parser().parse_args([
            "--stage", "prepare", "--work_dir", str(self.root / "run"), "--base_ckpt", str(base),
            "--dino_weights", str(dino), "--eval_datasets", "mvtec", "--image_size", "64", "--eval_size", "64",
            "--feature_grid", "8", "--feature_dim", "4", "--reference_pool", "2", "--num_shards", "2",
            "--match_radius", "0", "--epochs", "2", "--batch_size", "2", "--micro_batch", "1",
            "--workers", "0", "--hidden", "8", "--precision", "fp32", "--device", "cpu"])

    def prepare(self):
        with patch.dict(sys.modules, {"Datasets": self.module}):
            pipeline.prepare(self.args)

    def export(self):
        with patch.object(pipeline, "FrozenExtractor", FakeExtractor), patch.object(pipeline, "mask_for", side_effect=AssertionError("export read GT")):
            for shard in range(2):
                self.args.shard_id = shard
                pipeline.export(self.args)
        pipeline.seal(self.args)

    def test_prepare_content_families_and_reference_protocol(self):
        self.prepare()
        self.prepare()
        root = Path(self.args.work_dir)
        rows = pipeline.records(root)
        refs = load_json(root / "references.json")
        self.assertEqual({r["partition"] for r in rows}, {"train", "val", "eval"})
        self.assertTrue(all(r["partition"] == "reference_train" and not r["mask_path"] for r in refs))
        duplicate = dict(refs[0], rgb_sha=rows[0]["rgb_sha"])
        with self.assertRaisesRegex(ValueError, "duplicates a query"):
            pipeline.validate_manifest(rows, [duplicate] + refs[1:])
        # Re-encoding without changing decoded pixels must retain identity.
        source = Path(rows[0]["image_path"])
        copy = self.root / "reencoded.png"
        with Image.open(source) as im:
            im.save(copy, compress_level=0)
        self.assertEqual(pipeline.image_identity(source), pipeline.image_identity(copy))
        self.args.high_weight = .5
        with patch.dict(sys.modules, {"Datasets": self.module}), self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
            pipeline.prepare(self.args)

    def test_export_no_masks_then_cache_corruption_fails(self):
        self.prepare()
        for row in pipeline.records(Path(self.args.work_dir)):
            if row["mask_path"]:
                Path(row["mask_path"]).unlink()
        self.export()
        root = Path(self.args.work_dir)
        self.assertEqual(len(list((root / "features").glob("*.npz"))), 12)
        self.export()  # verified resume needs no model or mask
        row = pipeline.records(root)[0]
        path = root / "features" / f"{row['id']}.npz"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            pipeline.FeatureDataset(root, pipeline.cfg_load(root), [row], ARMS[0], include_mask=False)

    def test_source_only_training_resume_equivalence_and_target_report(self):
        self.prepare()
        self.export()
        root = Path(self.args.work_dir)
        uninterrupted = self.root / "uninterrupted"
        shutil.copytree(root, uninterrupted)
        original = pipeline.mask_for
        def source_only(row, size):
            self.assertNotEqual(row["partition"], "eval", "target mask entered training/selection")
            return original(row, size)
        with patch.object(pipeline, "mask_for", side_effect=source_only):
            for arm in ARMS:
                self.args.mode, self.args.until_epoch = arm, 1
                pipeline.train(self.args)
                self.args.until_epoch = 2
                pipeline.train(self.args)
                before = file_digest(root / "heads" / arm / "last.pt")
                pipeline.train(self.args)
                self.assertEqual(before, file_digest(root / "heads" / arm / "last.pt"))
            self.args.mode, self.args.work_dir = ARMS[0], str(uninterrupted)
            pipeline.train(self.args)
        full = torch.load(uninterrupted / "heads" / ARMS[0] / "last.pt", weights_only=False)
        resumed = torch.load(root / "heads" / ARMS[0] / "last.pt", weights_only=False)
        for key in full["state_dict"]:
            self.assertTrue(torch.equal(full["state_dict"][key], resumed["state_dict"][key]), key)
        self.args.work_dir = str(root)
        for shard in range(2):
            self.args.shard_id = shard
            pipeline.evaluate(self.args)
        pipeline.report(self.args)
        summary = load_json(root / "results" / "summary.json")
        self.assertEqual({r["arm"] for r in summary["means"]}, set(ARMS) | {"base_global", "base_highres"})
        self.assertEqual(len(summary["comparisons"]), 7)
        self.assertTrue(all(r["completed_epochs"] == 2 for r in summary["source_selection"]))
        self.assertTrue((root / "results" / "comparisons.csv").exists())
        # Existing reports must not bypass changed target masks on resume.
        target = next(r for r in pipeline.records(root, "eval") if r["mask_path"])
        Path(target["mask_path"]).write_bytes(b"modified GT")
        self.args.shard_id = 0
        with self.assertRaisesRegex(ValueError, "Target mask changed"):
            pipeline.evaluate(self.args)


BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH to Git Bash on Windows")
class ShellTests(unittest.TestCase):
    def run_script(self, fail="NEVER", gpu_ids="0,1"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(pipeline.REPO / "run_exp_reference_highres.sh", root / "run.sh")
            (root / "base.pth").write_bytes(b"fake")
            (root / "fake_python").write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n', newline="\n")
            (root / "flock").write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
            for name in ("fake_python", "flock"):
                (root / name).chmod(0o755)
            env = os.environ.copy()
            env.update(BASE_CKPT="./base.pth", WORK_DIR="./run", TRAIN_PYTHON="./fake_python", GPU_IDS=gpu_ids, FAIL_STAGE=fail)
            env["PATH"] = str(root) + os.pathsep + str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
            return subprocess.run([BASH, "run.sh"], cwd=root, env=env, text=True, capture_output=True, timeout=30)

    def test_four_modes_two_gpu_workers_and_source_before_target(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--data_roots {}", result.stdout)
        for gpu in (0, 1):
            self.assertIn(f"GPU={gpu} ARGS=reference_highres_experiment.py --stage export", result.stdout)
        for arm in ARMS:
            self.assertIn(f"--mode {arm}", result.stdout)
        self.assertLess(result.stdout.index("--stage preflight"), result.stdout.index("--stage export"))
        self.assertLess(result.stdout.index("--stage seal"), result.stdout.index("--stage train"))
        self.assertLess(result.stdout.rindex("--stage train"), result.stdout.index("--stage evaluate"))

    def test_failure_blocks_later_stages_and_duplicate_gpu_rejected(self):
        result = self.run_script(fail="--stage export")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("--stage train", result.stdout)
        self.assertNotEqual(self.run_script(gpu_ids="0,0").returncode, 0)


if __name__ == "__main__":
    unittest.main()
