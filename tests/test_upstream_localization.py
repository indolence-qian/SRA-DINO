import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import upstream_localization as pipeline
from dual_vlm import atomic_npz
from tools.upstream_localization import (
    LocalSemanticHead, parse_tile, segmentation_loss, source_partition, spatial_semantics, tile_boxes, tile_prompt,
)
from tools.vlm_review import atomic_json, digest, file_digest, load_json


def response(tile_id=0, **kwargs):
    value = dict(tile_id=tile_id, visibility="sufficient", status="possible_defect",
                 observation="A thin dark discontinuity in the metal.", normal_expectation="Continuous intact metal.")
    return json.dumps(value | kwargs)


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_coverage_unknown_mask_and_geometry(self):
        boxes = torch.tensor(tile_boxes())[None]
        emb = torch.ones(1, 9, 2, 8)
        valid = torch.ones(1, 9, 2)
        dense, covered = spatial_semantics(emb, valid, boxes, 31, 33)
        self.assertTrue(torch.all(covered == 1))
        self.assertTrue(torch.all(dense == 1))
        dense, covered = spatial_semantics(emb, valid * 0, boxes, 31, 33)
        self.assertEqual(dense.sum().item(), 0)
        self.assertEqual(covered.sum().item(), 0)
        with self.assertRaises(ValueError):
            tile_boxes(3, .2)
        # A tile's embedding cannot spill to the opposite side of an image.
        valid.zero_()
        valid[:, 0] = 1
        dense, _ = spatial_semantics(emb, valid, boxes, 32, 32)
        self.assertEqual(dense[..., -1, -1].sum().item(), 0)
        self.assertGreater(dense[..., 0, 0].sum().item(), 0)

    def test_parser_allows_honest_abstention_not_normal_pseudolabel(self):
        v = parse_tile(response(visibility="insufficient", status="uncertain", observation="", normal_expectation=""), 0)
        self.assertTrue(v["parse_ok"])
        self.assertFalse(v["semantic_valid"])
        for raw in [response(tile_id=True), response(tile_id=3), response(visibility="maybe"),
                    response(observation=""), response()[:-1] + ',"tile_id":0}', "malformed"]:
            self.assertFalse(parse_tile(raw, 0)["parse_ok"], raw)
        self.assertTrue(parse_tile("```json\n" + response() + "\n```", 0)["parse_ok"])
        self.assertNotIn("heatmap", tile_prompt("capsule", 0).lower())
        self.assertIn("not a normal reference", tile_prompt("capsule", 0))

    def test_identity_gradient_and_recovery_outside_base_candidates(self):
        torch.manual_seed(7)
        head = LocalSemanticHead(2, 8, 8)
        visual = torch.randn(1, 2, 8, 4, 4)
        base = torch.full((1, 1, 16, 16), .001)
        emb = torch.randn(1, 9, 2, 8)
        valid = torch.ones(1, 9, 2)
        boxes = torch.tensor(tile_boxes())[None]
        output = head(visual, base, emb, valid, boxes)
        self.assertTrue(torch.allclose(output.sigmoid(), base, atol=1e-7))
        target = torch.zeros_like(base)
        target[..., 3:6, 3:6] = 1
        optimizer = torch.optim.Adam(head.parameters(), lr=.03)
        initial = float(segmentation_loss(output, target).mean().detach())
        for _ in range(8):
            optimizer.zero_grad()
            output = head(visual, base, emb, valid, boxes)
            loss = segmentation_loss(output, target).mean()
            loss.backward()
            optimizer.step()
        output = head(visual, base, emb, valid, boxes)
        self.assertLess(float(segmentation_loss(output, target).mean().detach()), initial)
        self.assertGreater(float(output.sigmoid()[..., 3:6, 3:6].mean().detach()), .001)
        a = head(visual, base, emb, valid, boxes, use_semantics=False)
        b = head(visual, base, emb * 100, valid, boxes, use_semantics=False)
        self.assertTrue(torch.equal(a, b))
        self.assertGreater(float(head.semantic[0].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            head.decode[-1].weight.zero_()
            head.decode[-1].bias.fill_(20)
        saturated = head(visual, base * 0 + 1e-8, emb, valid, boxes)
        self.assertTrue(torch.all(saturated.sigmoid() > .5), "A residual bound prevents recovery of severe misses")

    def test_source_partition_deterministic_and_stratified(self):
        rows = [dict(id=str(i), category="a", mask_path="mask" if i % 2 else None) for i in range(20)]
        a = source_partition([dict(r) for r in rows], .2, 42)
        b = source_partition([dict(r) for r in reversed(rows)], .2, 42)
        self.assertEqual({r["id"]: r["partition"] for r in a}, {r["id"]: r["partition"] for r in b})
        self.assertEqual(sum(r["partition"] == "val" for r in a), 4)


class FakeTeacher:
    calls = []
    def __init__(self, **kwargs):
        pass

    def generate(self, prompt, images, trace_dir=None):
        tile_id = int(re.search(r'"tile_id":(\d+)', prompt)[1])
        self.calls.append((prompt, [im.size for im in images]))
        raw = response(tile_id)
        atomic_json(trace_dir / "trace.json", dict(raw_response=raw))
        return raw


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = pipeline.parser().parse_args(["--stage", "train", "--work_dir", str(self.root),
                                                  "--epochs", "2", "--batch_size", "2", "--hidden", "8", "--workers", "0"])
        self.rows = []
        rng = np.random.default_rng(4)
        for i, part in enumerate(["train"] * 4 + ["val"] * 2 + ["eval"] * 2):
            image_path = self.root / f"rgb{i}.png"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(image_path)
            mask_path = None
            if i % 2:
                mask_path = self.root / f"gt{i}.png"
                gt = np.zeros((64, 64), np.uint8)
                gt[5:7, 10:12] = 255
                Image.fromarray(gt).save(mask_path)
            self.rows.append(dict(id=str(i), partition=part, dataset="visa" if part != "eval" else "mvtec",
                                  category="object", image_path=str(image_path), image_sha=file_digest(image_path),
                                  mask_path=str(mask_path) if mask_path else None,
                                  mask_sha=file_digest(mask_path) if mask_path else None))
        self.cfg = dict(protocol=pipeline.PROTOCOL, code={p: file_digest(pipeline.REPO / p) for p in pipeline.CODE_FILES},
                        manifest_sha=digest(self.rows), seed=42, num_shards=2, image_size=64, boxes=tile_boxes(),
                        model=dict(path="fake"), gpu_memory=.7, max_model_len=4096, max_tokens=256,
                        teacher_image_size=512, retries=0, max_invalid_ratio=.05)
        atomic_json(self.root / "manifest.json", self.rows)
        atomic_json(self.root / "config.json", self.cfg)
        FakeTeacher.calls = []

    def review(self):
        with patch("tools.vlm_decision.QwenVLLMTeacher", FakeTeacher), patch("dual_vlm.model_signature", return_value=self.cfg["model"]):
            for shard in range(2):
                self.args.shard_id = shard
                pipeline.review(self.args)

    def features(self):
        self.review()
        rng = np.random.default_rng(10)
        for r in self.rows:
            _, sha = pipeline.reviews_for(self.root, r, self.cfg)
            path = self.root / "features" / f"{r['id']}.npz"
            atomic_npz(path, visual=rng.normal(size=(2, 8, 4, 4)).astype(np.float16),
                       base=np.full((1, 64, 64), .1, np.float32),
                       embeddings=rng.normal(size=(9, 2, 8)).astype(np.float16),
                       valid=np.ones((9, 2), np.float32), boxes=np.array(tile_boxes(), np.float32),
                       truncated_texts=np.array(0))
            atomic_json(path.with_suffix(".json"), dict(fingerprint=digest(self.cfg), review_sha=sha))
        pipeline.seal(self.args)

    def test_review_never_opens_masks_or_uses_base_and_resumes(self):
        for r in self.rows:
            if r["mask_path"]:
                Path(r["mask_path"]).unlink()
        with patch.object(pipeline, "mask_for", side_effect=AssertionError("GT leak")):
            self.review()
            self.review()
        self.assertEqual(len(FakeTeacher.calls), 8 * 9)
        for prompt, sizes in FakeTeacher.calls:
            self.assertNotIn(str(self.root), prompt)
            self.assertEqual(len(sizes), 2)
            self.assertEqual(sizes[0], (64, 64))
            self.assertLess(sizes[1][0], 64)
        pipeline.audit_reviews(self.args)

    def test_end_to_end_training_selection_resume_and_paired_evaluation(self):
        self.features()
        target_rows = [r for r in self.rows if r["partition"] == "eval"]
        original_mask = pipeline.mask_for
        def forbid_target(r, size):
            self.assertNotEqual(r["partition"], "eval", "target GT entered head training")
            return original_mask(r, size)
        with patch("torch.cuda.is_available", return_value=False), patch.object(pipeline, "mask_for", side_effect=forbid_target):
            for mode in ("visual", "vlm"):
                self.args.mode = mode
                pipeline.train(self.args)
                saved = file_digest(self.root / "heads" / mode / "last.pt")
                pipeline.train(self.args)
                self.assertEqual(saved, file_digest(self.root / "heads" / mode / "last.pt"))
        calls = len(FakeTeacher.calls)
        self.review()
        self.assertEqual(calls, len(FakeTeacher.calls))
        with patch("torch.cuda.is_available", return_value=False):
            for shard in range(2):
                self.args.shard_id = shard
                pipeline.evaluate(self.args)
        pipeline.report(self.args)
        rows = load_json(self.root / "results/summary.json")
        self.assertEqual({r["mode"] for r in rows}, {"base", "visual", "vlm", "vlm_zero_semantics_DIAGNOSTIC"})
        self.assertTrue(all(r["samples"] == len(target_rows) for r in rows))
        self.assertEqual(len(load_json(self.root / "heads/vlm/history.json")), 2)

    def test_feature_and_review_mutation_fail_closed(self):
        self.features()
        path = self.root / "reviews/0/0.json"
        review = load_json(path)
        review["parsed"]["observation"] = "silently edited"
        atomic_json(path, review)
        with self.assertRaisesRegex(ValueError, "raw response"):
            pipeline.reviews_for(self.root, self.rows[0], self.cfg)
        feature = self.root / "features/0.npz"
        feature.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "modified"):
            pipeline.FeatureDataset(self.root, self.cfg, [self.rows[0]])

    def test_mask_resize_follows_dataset_specific_legacy_thresholds(self):
        path = self.root / "grey.png"
        Image.fromarray(np.full((64, 64), 100, np.uint8)).save(path)
        self.assertEqual(float(pipeline.mask_for(dict(mask_path=str(path), dataset="visa"), 64).sum()), 4096)
        self.assertEqual(float(pipeline.mask_for(dict(mask_path=str(path), dataset="mvtec"), 64).sum()), 0)

    def test_frozen_export_hooks_capture_all_layers_and_no_gt(self):
        self.review()
        basefile, dinofile = self.root / "base.pth", self.root / "dino.pth"
        basefile.write_bytes(b"fake base")
        dinofile.write_bytes(b"fake dino")
        self.cfg.update(base_ckpt=str(basefile), base_sha=file_digest(basefile),
                        dino_weights=str(dinofile), dino_sha=file_digest(dinofile))
        atomic_json(self.root / "config.json", self.cfg)
        # Re-fingerprint synthetic responses after adding the fake model metadata.
        for path in (self.root / "reviews").glob("*/*.json"):
            value = load_json(path)
            value["fingerprint"] = digest(self.cfg)
            atomic_json(path, value)
        adapter = torch.nn.Module()
        adapter.patch_token_adapter = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        clip = types.SimpleNamespace(token_embedding=torch.nn.Embedding(10, 8))
        utils = types.ModuleType("tools.utils_up")
        def get_map(clip, info, device, adapter, dino, prompt, idx, **kwargs):
            self.assertEqual(float(info["mask"].sum()), 0)
            self.assertEqual(info["image"].shape, (1, 3, 64, 64))
            for layer in adapter.patch_token_adapter:
                layer(torch.ones(1, 16, 8))
            base = torch.zeros(1, 2, 64, 64)
            base[:, 1] = .2
            return None, None, base, None, None
        utils.get_anomaly_map = get_map
        semantics = types.ModuleType("tools.semantic_anchor")
        semantics._transform_text_embeddings = lambda clip, embeddings, ids: embeddings[:, 0]
        tokenizer = types.ModuleType("CLIP.tokenizer")
        tokenizer.tokenize = lambda texts, **kwargs: torch.ones(len(texts), 4, dtype=torch.long)
        real_device = torch.device
        for r in self.rows:
            if r["mask_path"]:
                Path(r["mask_path"]).unlink()
        with patch.dict(sys.modules, {"tools.utils_up": utils, "tools.semantic_anchor": semantics, "CLIP.tokenizer": tokenizer}), \
                patch("dual_vlm.load_dual", return_value=(clip, None, None, adapter, [1, 2])), \
                patch.object(pipeline.torch, "device", side_effect=lambda _: real_device("cpu")), \
                patch.object(pipeline.torch.cuda, "set_device"), \
                patch.object(pipeline, "mask_for", side_effect=AssertionError("GT leak")):
            with patch.object(pipeline, "reviews_for", side_effect=AssertionError("Preflight must not require VLM")):
                pipeline.export(self.args, probe=True)
            self.assertFalse((self.root / "features").exists(), "Preflight must not populate the semantic feature cache")
            for shard in range(2):
                self.args.shard_id = shard
                pipeline.export(self.args)
        with np.load(self.root / "features/0.npz") as data:
            self.assertEqual(data["visual"].shape, (2, 8, 4, 4))
            self.assertTrue(np.allclose(data["base"], .2))
            self.assertTrue(np.allclose(np.linalg.norm(data["visual"].astype(np.float32), axis=1), 1, atol=.001))
        self.assertTrue(all(not layer._forward_hooks for layer in adapter.patch_token_adapter))

    def test_prepare_stable_guardrails_and_content_overlap(self):
        dataset_module = types.ModuleType("Datasets")
        original = self.rows
        class Dataset:
            def __init__(self, source, **kwargs):
                # Six source examples (3 of each label) and two target examples.
                selected = original[:6] if source == "SOURCE" else original[6:]
                self.data_to_iterate = [("object", "unused", r["image_path"], r["mask_path"]) for r in selected]
        splits = types.SimpleNamespace(TEST="test")
        dataset_module.DATASET_REGISTRY = {"visa": (Dataset, splits, "SOURCE"), "mvtec": (Dataset, splits, "TARGET")}
        dataset_module.DATASET_CLASSES = {"visa": ["object"], "mvtec": ["object"]}
        base = self.root / "base.pth"
        torch.save({k: {} for k in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner")}, base)
        dino = self.root / "dino.pth"
        dino.write_bytes(b"frozen dino")
        args = pipeline.parser().parse_args(["--stage", "prepare", "--work_dir", str(self.root / "prepared"),
                                             "--base_ckpt", str(base), "--dino_weights", str(dino)])
        with patch.dict(sys.modules, {"Datasets": dataset_module}), patch("dual_vlm.model_signature", return_value={"path": "fake"}):
            pipeline.prepare(args)
            pipeline.prepare(args)
            manifest = load_json(Path(args.work_dir) / "manifest.json")
            self.assertEqual({r["partition"] for r in manifest}, {"train", "val", "eval"})
            args.tile_fraction = .5
            with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
                pipeline.prepare(args)
            args.work_dir = str(self.root / "overlap")
            original[6]["image_path"] = original[0]["image_path"]
            with self.assertRaisesRegex(ValueError, "Identical image"):
                pipeline.prepare(args)


BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH on Windows")
class ShellTests(unittest.TestCase):
    def run_script(self, fail="NEVER", gpu_ids="0,1"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(pipeline.REPO / "run_exp_upstream_localization.sh", root / "run.sh")
            (root / "base").mkdir()
            (root / "fake_python").write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n', newline="\n")
            (root / "flock").write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
            for path in (root / "fake_python", root / "flock"):
                path.chmod(0o755)
            env = os.environ.copy()
            env.update(BASE_CKPT="./base", WORK_DIR="./out", TRAIN_PYTHON="./fake_python", VLM_PYTHON="./fake_python",
                       GPU_IDS=gpu_ids, FAIL_STAGE=fail)
            env["PATH"] = str(root) + os.pathsep + str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
            return subprocess.run([BASH, "run.sh"], cwd=root, env=env, text=True, capture_output=True, timeout=30)

    def test_both_gpu_workers_real_ddp_launch_and_stage_order(self):
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for gpu in (0, 1):
            self.assertIn(f"GPU={gpu} ARGS=upstream_localization.py --stage review", r.stdout)
            self.assertIn(f"GPU={gpu} ARGS=upstream_localization.py --stage export", r.stdout)
        self.assertIn("GPU=0,1 ARGS=-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2", r.stdout)
        self.assertIn("--mode visual", r.stdout)
        self.assertIn("--mode vlm", r.stdout)
        self.assertLess(r.stdout.index("--stage review"), r.stdout.index("--stage export"))
        self.assertLess(r.stdout.index("--stage seal"), r.stdout.index("--stage train"))
        self.assertLess(r.stdout.index("--stage train"), r.stdout.index("--stage evaluate"))

    def test_failure_stops_training_and_duplicate_gpus_rejected(self):
        r = self.run_script(fail="--stage export")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("--stage train", r.stdout)
        r = self.run_script(gpu_ids="0,0")
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
