import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import dual_vlm
from tools.vlm_review import (
    atomic_json, calibrate_map, candidate_rois, digest, load_json, parse_review, review_prompt,
)


class ReviewUnitTests(unittest.TestCase):
    def setUp(self):
        self.base = np.full((32, 32), 0.4, dtype=np.float32)
        self.rois = [{"roi_id": 1, "box": [8, 8, 24, 24], "kind": "peak"}]

    def vote(self, verdict="defect", confidence=0.9):
        return {"parse_ok": True, "regions": [{"roi_id": 1, "verdict": verdict, "confidence": confidence}]}

    def test_candidates_reproducible_distinct_and_in_bounds(self):
        base = self.base.copy()
        base[3, 3], base[28, 28] = 1.0, 0.9
        dis = np.zeros_like(base)
        dis[3, 28] = 1
        rois = candidate_rois(base, dis, seed=42)
        self.assertEqual(rois, candidate_rois(base, dis, seed=42))
        self.assertEqual([r["kind"] for r in rois], ["peak", "peak", "disagreement", "coverage"])
        self.assertEqual(len({tuple(r["box"]) for r in rois}), 4)
        for roi in rois:
            x1, y1, x2, y2 = roi["box"]
            self.assertTrue(0 <= x1 < x2 <= 32 and 0 <= y1 < y2 <= 32)

    def test_uniform_maps_still_produce_coverage(self):
        self.assertEqual(len(candidate_rois(self.base, self.base, seed=1)), 4)

    def test_invalid_evidence_is_rejected(self):
        bad = self.base.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            candidate_rois(bad, self.base)

    def test_prompt_is_no_reference_and_has_no_feature_layer_decision(self):
        prompt = review_prompt("capsule", self.rois)
        self.assertIn("No known-normal reference", prompt)
        self.assertNotIn("preferred_layer", prompt)
        self.assertNotIn("heatmap", prompt)
        self.assertIn("fallible proposal heatmap", review_prompt("capsule", self.rois, True))

    def test_strict_review_accepts_json_and_fences(self):
        value = json.dumps({"regions": self.vote()["regions"]})
        for raw in (value, "```json\n" + value + "\n```"):
            self.assertTrue(parse_review(raw, self.rois)["parse_ok"])

    def test_bad_reviews_fail_closed(self):
        for regions in ([], [{"roi_id": 2, "verdict": "defect", "confidence": 1}],
                        [{"roi_id": 1, "verdict": "defect", "confidence": True}],
                        [{"roi_id": 1, "verdict": "defect", "confidence": float("nan")}],
                        [{"roi_id": 1, "verdict": "refine", "confidence": 0.9}],
                        self.vote()["regions"] * 2):
            with self.subTest(regions=regions):
                self.assertFalse(parse_review(json.dumps({"regions": regions}), self.rois)["parse_ok"])
        self.assertFalse(parse_review("truncated {", self.rois)["parse_ok"])

    def test_noop_is_exact_identity(self):
        for review in ({}, self.vote("uncertain"), self.vote(confidence=0.5)):
            np.testing.assert_array_equal(calibrate_map(self.base, self.rois, review), self.base)
        np.testing.assert_array_equal(calibrate_map(self.base, self.rois, self.vote(), alpha=0), self.base)

    def test_correction_is_local_bounded_and_directional(self):
        plus = calibrate_map(self.base, self.rois, self.vote())
        minus = calibrate_map(self.base, self.rois, self.vote("normal"))
        self.assertGreater(plus[16, 16], self.base[16, 16])
        self.assertLess(minus[16, 16], self.base[16, 16])
        self.assertLessEqual(float(np.abs(plus - self.base).max()), 0.125)
        np.testing.assert_array_equal(plus[:8], self.base[:8])
        np.testing.assert_array_equal(plus[:, :8], self.base[:, :8])
        self.assertLess(plus[8, 8], plus[16, 16])

    def test_overlap_does_not_multiply_logit_budget(self):
        rois = self.rois + [{**self.rois[0], "roi_id": 2}]
        review = self.vote()
        review["regions"].append({**review["regions"][0], "roi_id": 2})
        result = calibrate_map(self.base, rois, review)
        logit = lambda p: np.log(p / (1 - p))
        self.assertLessEqual(float((logit(result) - logit(self.base)).max()), 0.50001)

    def test_control_is_independent_of_teacher(self):
        a = calibrate_map(self.base, self.rois, {}, control=True)
        b = calibrate_map(self.base, self.rois, self.vote(), control=True)
        np.testing.assert_array_equal(a, b)
        self.assertLess(a[16, 16], self.base[16, 16])

    def test_selected_subset_is_deterministic_and_limited(self):
        a = dual_vlm.selected_indices(range(100), 12, 42)
        self.assertEqual(a, dual_vlm.selected_indices(range(100), 12, 42))
        self.assertEqual(len(set(a)), 12)
        self.assertEqual(dual_vlm.selected_indices(range(5), 0, 42), list(range(5)))

    def test_resolve_numeric_checkpoint_and_reject_wrong_arch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("9.pth", "14.pth", "mara_epoch_29.pth"):
                (root / name).touch()
            self.assertEqual(dual_vlm.resolve_checkpoint(root).name, "14.pth")
        for payload in ({"base_arch": "dino_single"}, {"mara_agent": {}}, {}):
            with self.assertRaises(ValueError):
                dual_vlm.check_dual_payload(payload)


class ReviewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = dual_vlm.parser().parse_args(["--stage", "prepare", "--work_dir", str(self.root / "work"),
                                                  "--pro_num_th", "10", "--image_size", "32"])
        self.work = Path(self.args.work_dir)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"mock model")
        self.args.model_id = str(model)
        self.base = self.root / "base.pth"
        import torch
        torch.save({key: {} for key in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner")}, self.base)
        self.args.base_ckpt = str(self.base)

    def prepare(self):
        # Avoid importing dataset/image-model dependencies in this CPU test.
        from types import SimpleNamespace
        with patch.dict("sys.modules", {"Datasets": SimpleNamespace(DATASET_CLASSES={"visa": ["object"]})}):
            dual_vlm.prepare(self.args)
        return load_json(self.work / "config.json")

    def test_prepare_is_resumable_but_rejects_changed_configuration(self):
        cfg = self.prepare()
        self.assertEqual(cfg, self.prepare())
        self.args.alpha = 0.7
        with self.assertRaisesRegex(ValueError, "different settings"):
            self.prepare()
        self.assertEqual(load_json(self.work / "config.json"), cfg)

    def make_manifest(self):
        cfg = self.prepare()
        records = []
        for i in range(2):
            ident = f"sample{i}"
            record = dict(id=ident, fingerprint=digest(cfg), dataset="visa", category="object", index=i,
                          rois=[{"roi_id": 1, "box": [8, 8, 24, 24], "kind": "peak"}],
                          query=f"queries/{ident}.png", evidence=f"evidence/{ident}.npz",
                          evaluation=f"evaluation_only/{ident}.npz")
            dual_vlm.atomic_image(self.work / record["query"], np.full((32, 32, 3), 128, np.uint8))
            mask = np.zeros((32, 32), np.uint8)
            base = np.full((32, 32), 0.2, np.float32)
            if i:
                mask[10:20, 10:20] = 1
                base[10:20, 10:20] = 0.6
            dual_vlm.atomic_npz(self.work / record["evidence"], base=base)
            dual_vlm.atomic_npz(self.work / record["evaluation"], mask=mask)
            records.append(record)
            atomic_json(self.work / f"export_shard_{i}.json", {"fingerprint": digest(cfg), "records": [record]})
        dual_vlm.seal(self.args)
        return cfg, records

    def test_review_never_reads_label_files_and_resumes(self):
        cfg, records = self.make_manifest()
        # Review still works with GT files absent.
        for record in records:
            (self.work / record["evaluation"]).unlink()
        raw = '{"regions":[{"roi_id":1,"verdict":"normal","confidence":0.9}]}'
        with patch("tools.vlm_decision.QwenVLLMTeacher") as teacher:
            teacher.return_value.generate.return_value = raw
            for shard in range(2):
                self.args.shard_id = shard
                dual_vlm.review(self.args)
            self.assertEqual(teacher.return_value.generate.call_count, 2)
            prompt, images = teacher.return_value.generate.call_args.args
            self.assertEqual(len(images), 2)
            self.assertNotIn("evaluation_only", prompt)
            self.assertNotIn("sample", prompt)
            teacher.reset_mock()
            dual_vlm.review(self.args)
            teacher.assert_not_called()

    def test_heatmap_is_optional_and_aligned(self):
        cfg, records = self.make_manifest()
        images = dual_vlm.build_review_images(self.work, records[0], {**cfg, "heatmap": True})
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0].size, images[1].size)
        self.assertEqual(images[2].size, (16, 16))

    def test_invalid_review_retries_then_falls_back(self):
        self.make_manifest()
        with patch("tools.vlm_decision.QwenVLLMTeacher") as teacher:
            teacher.return_value.generate.return_value = "invalid"
            dual_vlm.review(self.args)
            self.assertEqual(teacher.return_value.generate.call_count, 2)
        data = load_json(self.work / "reviews" / "sample0.json")
        self.assertFalse(data["decision"]["parse_ok"])

    def test_incomplete_or_wrong_shards_cannot_be_sealed(self):
        cfg = self.prepare()
        atomic_json(self.work / "export_shard_0.json", {"fingerprint": "wrong", "records": []})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            dual_vlm.seal(self.args)

    def test_evaluation_reuses_real_metrics_and_produces_all_four_modes(self):
        cfg, records = self.make_manifest()
        for i, record in enumerate(records):
            atomic_json(self.work / "reviews" / f"{record['id']}.json", {
                "id": record["id"], "fingerprint": digest(cfg), "seconds": 0.1,
                "decision": {"parse_ok": True, "regions": [{"roi_id": 1, "verdict": "defect" if i else "normal", "confidence": 0.9}]},
            })
        dual_vlm.evaluate(self.args)
        result = load_json(self.work / "results" / "metrics.json")
        row = result["categories"][0]
        self.assertEqual(row["base_PRO"], row["vlm_image_PRO"])
        self.assertEqual(row["vlm_region_I_AUROC"], row["vlm_image_I_AUROC"])
        self.assertEqual(row["valid_ratio"], 1)
        self.assertEqual(row["decisive_roi_accuracy"], 1)
        self.assertTrue((self.work / "results" / "metric_vlm.txt").is_file())
        self.assertIn("control_shrink_PRO", row)

    def test_evaluation_fails_on_missing_reviews(self):
        self.make_manifest()
        with self.assertRaisesRegex(ValueError, "Missing review"):
            dual_vlm.evaluate(self.args)

    def test_evaluation_rejects_excess_invalid_reviews(self):
        cfg, records = self.make_manifest()
        for record in records:
            atomic_json(self.work / "reviews" / f"{record['id']}.json", {
                "id": record["id"], "fingerprint": digest(cfg), "seconds": 0,
                "decision": {"parse_ok": False, "regions": []},
            })
        with self.assertRaisesRegex(ValueError, "Invalid reviews"):
            dual_vlm.evaluate(self.args)

    def test_evaluation_rejects_stale_review(self):
        _, records = self.make_manifest()
        record = records[0]
        atomic_json(self.work / "reviews" / f"{record['id']}.json", {
            "id": record["id"], "fingerprint": "other-run", "seconds": 0,
            "decision": {"parse_ok": True, "regions": []},
        })
        with self.assertRaisesRegex(ValueError, "provenance"):
            dual_vlm.evaluate(self.args)

    def test_export_review_evaluate_smoke_with_mock_backbones(self):
        import torch
        from types import SimpleNamespace
        cfg = self.prepare()
        source = self.root / "source.png"
        Image.new("RGB", (32, 32), (128, 128, 128)).save(source)
        class ToyDataset:
            def __init__(self, **kwargs):
                self.data_to_iterate = [("object", "good", str(source), None)] * 4
            def __len__(self):
                return 4
            def __getitem__(self, i):
                rgb = torch.full((3, 32, 32), 128 / 255)
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                mask = torch.zeros(1, 32, 32)
                if i % 2:
                    mask[:, 10:20, 10:20] = 1
                return {"image": (rgb - mean) / std, "mask": mask,
                        "is_anomaly": i % 2, "image_path": str(source)}
        modules = {"Datasets": SimpleNamespace(DATASET_CLASSES={"visa": ["object"]},
                    DATASET_REGISTRY={"visa": (ToyDataset, SimpleNamespace(TEST="test"), str(self.root))})}
        def fake_forward(clip, batch, *args, **kwargs):
            p = 0.2 + batch["mask"] * 0.5
            return None, batch["mask"], torch.cat([1 - p, p], 1), torch.zeros(len(p), 2), {"cross_prob_layers": [p[:, 0]] * 4}
        # Import real utilities before temporarily replacing the dataset registry.
        import tools.utils_up
        with patch.dict("sys.modules", modules), patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.set_device"), patch("dual_vlm.load_dual", return_value=(None, None, None, None, (5, 11, 17, 23))), \
             patch("tools.utils_up.get_anomaly_map", side_effect=fake_forward) as forward:
            for shard in range(2):
                self.args.shard_id = shard
                dual_vlm.export(self.args)
            self.assertEqual(forward.call_count, 2)
            dual_vlm.export(self.args)
            self.assertEqual(forward.call_count, 2)  # cached samples skip inference
        dual_vlm.seal(self.args)
        records = load_json(self.work / "manifest.json")["records"]
        self.assertEqual(len(records), 4)
        with Image.open(self.work / records[0]["query"]) as image:
            self.assertTrue(np.all(np.asarray(image) == 128))
        def fake_generate(prompt, images):
            return json.dumps({"regions": [{"roi_id": i, "verdict": "uncertain", "confidence": 0.5}
                                           for i in range(1, len(images))]})
        with patch("tools.vlm_decision.QwenVLLMTeacher") as teacher:
            teacher.return_value.generate.side_effect = fake_generate
            for shard in range(2):
                self.args.shard_id = shard
                dual_vlm.review(self.args)
        dual_vlm.evaluate(self.args)
        result = load_json(self.work / "results" / "metrics.json")
        row = result["categories"][0]
        self.assertEqual(row["samples"], 4)
        self.assertEqual(row["changed_pixel_fraction"], 0)
        self.assertEqual(row["vlm_region_PRO"], row["base_PRO"])


if __name__ == "__main__":
    unittest.main()
