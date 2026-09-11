"""Native crop geometry, fail-closed decisions, tiny masks and CPU pipeline tests."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import dual_vlm
from tools.local_vlm_review import (action, calibrate_local, component_counts, local_prompt,
                                   native_crops, parse_local, raw_boxes, small_candidates)
from tools.local_vlm_pipeline import export_local, reference_bank, review_path
from tools.vlm_review import atomic_json, digest, load_json


def decision(candidate_id=1, verdict="defect_supported", visibility="sufficient", reference="unavailable"):
    return dict(candidate_id=candidate_id, verdict=verdict, visibility=visibility, reference_match=reference,
                defect_type="crack", evidence="A thin discontinuity is visible")


class LocalUnitTests(unittest.TestCase):
    def test_single_pixel_and_thin_line_preserved(self):
        base = np.full((128, 128), .1, np.float32)
        base[20, 20] = .8
        base[60:66, 90] = .7
        candidates, supports = small_candidates(base, np.zeros_like(base))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(sorted(c["area"] for c in candidates), [1, 6])
        self.assertTrue(np.all(supports.sum(axis=0) <= 1))
        self.assertEqual(int(supports.sum()), 7)

    def test_plateau_and_uniform_map_not_cropped_to_fake_small_defect(self):
        base = np.full((128, 128), .2, np.float32)
        self.assertEqual(len(small_candidates(base, base)[0]), 0)
        base[10:90, 10:90] = .8
        self.assertEqual(len(small_candidates(base, np.zeros_like(base))[0]), 0)

    def test_area_budget_and_determinism(self):
        base = np.full((128, 128), .1, np.float32)
        for y in (20, 50, 80, 110):
            for x in (20, 50, 80, 110):
                base[y:y+2, x:x+2] = .8
        a, masks = small_candidates(base, base, 20, .001, .002)
        self.assertEqual(a, small_candidates(base, base, 20, .001, .002)[0])
        self.assertLessEqual(masks.sum(), int(base.size*.002))
        self.assertTrue(all(c["area"] <= int(base.size*.001) for c in a))

    def test_invalid_candidate_settings(self):
        base = np.ones((32, 32), np.float32)
        for kwargs in ({"max_candidates": 0}, {"max_area_fraction": .1, "total_area_fraction": .01}, {"min_probability": -1}):
            with self.assertRaises(ValueError):
                small_candidates(base, base, **kwargs)
        base[0, 0] = np.nan
        with self.assertRaises(ValueError):
            small_candidates(base, base)

    def test_raw_non_square_geometry_and_native_detail(self):
        raw = Image.fromarray(np.random.default_rng(4).integers(0, 256, (800, 1600, 3), dtype=np.uint8))
        box = [248, 248, 256, 256]
        tight, context = raw_boxes(box, (512, 512), raw.size)
        images, geom = native_crops(raw, box, (512, 512))
        np.testing.assert_array_equal(images[1], raw.crop(tuple(tight)))
        self.assertEqual(geom["source_size"], [1600, 800])
        self.assertGreater(images[1].width, images[1].height)
        low, _ = native_crops(raw, box, (512, 512), resized_ablation=True)
        self.assertLess(low[1].width, images[1].width)
        for box in ([0, 0, 1, 1], [511, 511, 512, 512]):
            for crop in raw_boxes(box, (512, 512), raw.size):
                self.assertTrue(0 <= crop[0] < crop[2] <= 1600 and 0 <= crop[1] < crop[3] <= 800)

    def test_strict_parsing_and_abstention(self):
        valid = decision()
        self.assertTrue(parse_local(json.dumps(valid), 1)["parse_ok"])
        self.assertTrue(parse_local("```json\n"+json.dumps(valid)+"\n```", 1)["parse_ok"])
        for changed in ({"candidate_id": True}, {"candidate_id": 2}, {"confidence": .9}, {"evidence": ""},
                        {"reference_match": "matched"}, {"verdict": "normal"}):
            self.assertFalse(parse_local(json.dumps({**valid, **changed}), 1)["parse_ok"])
        self.assertEqual(action(parse_local(json.dumps(decision(visibility="insufficient")), 1)), "keep")
        prompt = local_prompt("pcb1", 1)
        self.assertIn("few pixels", prompt)
        self.assertIn("does NOT establish normality", prompt)
        self.assertNotIn("GT", prompt)

    def test_exact_support_identity_and_monotonic_enhancement(self):
        base = np.full((32, 32), .4, np.float32)
        support = np.zeros((1, 32, 32), bool)
        support[0, 10:20, 12] = True  # only the crack, not its observation rectangle
        vote = {"parse_ok": True, **decision()}
        out = calibrate_local(base, support, [vote])
        np.testing.assert_array_equal(out[~support[0]], base[~support[0]])
        self.assertTrue(np.all(out >= base))
        self.assertEqual(np.count_nonzero(out != base), 10)
        base[10, 12], base[11, 12] = 1, 0
        out = calibrate_local(base, support, [vote])
        self.assertEqual(out[10, 12], 1)
        self.assertEqual(out[11, 12], 0)
        self.assertTrue(np.all(out >= base))
        self.assertTrue(np.all(out[support[0]] <= 1))

    def test_normal_uncertain_invalid_preserve_default_base(self):
        base = np.full((32, 32), .4, np.float32)
        supports = np.ones((1, 32, 32), bool)
        for vote in ({"parse_ok": False}, {"parse_ok": True, **decision(verdict="normal_supported")},
                     {"parse_ok": True, **decision(verdict="insufficient_evidence")}):
            np.testing.assert_array_equal(calibrate_local(base, supports, [vote]), base)

    def test_suppression_requires_reference_and_vetoes_high_base(self):
        base = np.full((32, 32), .4, np.float32)
        supports = np.ones((1, 32, 32), bool)
        vote = {"parse_ok": True, **decision(verdict="normal_supported")}
        np.testing.assert_array_equal(calibrate_local(base, supports, [vote], "suppress"), base)
        vote["reference_match"] = "matched"
        self.assertTrue(np.all(calibrate_local(base, supports, [vote], "suppress") < base))
        base[0, 0] = .8
        np.testing.assert_array_equal(calibrate_local(base, supports, [vote], "suppress"), base)

    def test_no_candidates_and_overlap_guards(self):
        base = np.full((32, 32), .4, np.float32)
        np.testing.assert_array_equal(calibrate_local(base, np.zeros((0, 32, 32), bool), []), base)
        with self.assertRaises(ValueError):
            calibrate_local(base, np.ones((2, 32, 32), bool), [{}, {}])

    def test_component_diagnostics_use_gt_only_and_keep_tiny_components(self):
        mask = np.zeros((128, 128), bool)
        mask[10, 10], mask[50:56, 50:56] = True, True
        base = mask.astype(np.float32)
        new = base.copy()
        new[10, 10] = 0
        counts = component_counts(mask, mask, base, new)
        self.assertEqual(counts["small_count"], 1)
        self.assertEqual(counts["small_new_miss"], 1)
        self.assertEqual(counts["medium_count"], 1)


class LocalPipelineTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"mock")
        checkpoint = self.root / "base.pth"
        torch.save({k: {} for k in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner")}, checkpoint)
        self.args = dual_vlm.parser().parse_args(["--stage", "prepare", "--work_dir", str(self.work),
                     "--base_ckpt", str(checkpoint), "--model_id", str(model), "--image_size", "32",
                     "--pro_num_th", "10", "--local_review"])

    def prepare(self):
        with patch.dict("sys.modules", {"Datasets": SimpleNamespace(DATASET_CLASSES={"visa": ["object"]})}):
            dual_vlm.prepare(self.args)
        return load_json(self.work / "config.json")

    def make_manifest(self, flat=False):
        cfg = self.prepare()
        records = []
        raw = self.root / "raw.png"
        Image.new("RGB", (800, 400), (100, 110, 120)).save(raw)
        for i in range(2):
            record = dict(id=f"sample{i}", fingerprint=digest(cfg), dataset="visa", category="object", index=i,
                          evidence=f"evidence/sample{i}.npz", evaluation=f"evaluation_only/sample{i}.npz")
            base = np.full((32, 32), .2, np.float32)
            if not flat:
                base[12:14, 12:14] = .55
            gt = np.zeros_like(base, np.uint8)
            if i:
                gt[12:14, 12:14] = 1
            dual_vlm.atomic_npz(self.work / record["evidence"], base=base)
            dual_vlm.atomic_npz(self.work / record["evaluation"], mask=gt)
            export_local(self.work, record, base, np.zeros_like(base), raw, cfg)
            atomic_json(self.work / f"export_shard_{i}.json", {"fingerprint": digest(cfg), "records": [record]})
            records.append(record)
        dual_vlm.seal(self.args)
        return cfg, records

    def run_review(self):
        with patch("tools.vlm_decision.QwenVLLMTeacher") as teacher:
            teacher.return_value.generate.return_value = json.dumps(decision())
            for shard in (0, 1):
                self.args.shard_id = shard
                dual_vlm.review(self.args)
            return teacher

    def test_prepare_separates_protocols_and_changed_settings(self):
        cfg = self.prepare()
        self.assertEqual(cfg["correction_default"], "enhance_only")
        self.assertEqual(self.prepare(), cfg)
        self.args.local_input = "resized"
        with self.assertRaisesRegex(ValueError, "different settings"):
            self.prepare()

    def test_native_review_uses_no_gt_and_resumes(self):
        cfg, records = self.make_manifest()
        for r in records:
            (self.work / r["evaluation"]).unlink()
        teacher = self.run_review()
        self.assertEqual(teacher.return_value.generate.call_count, 2)
        prompt, images = teacher.return_value.generate.call_args.args
        self.assertEqual(len(images), 2)
        self.assertGreater(images[1].width, 8)
        self.assertNotIn("raw.png", prompt)
        self.assertNotIn("sample", prompt)
        with patch("tools.vlm_decision.QwenVLLMTeacher") as mock:
            dual_vlm.review(self.args)
            mock.assert_not_called()

    def test_evaluation_default_modes_and_decision_diagnostics(self):
        self.make_manifest()
        self.run_review()
        dual_vlm.evaluate(self.args)
        result = load_json(self.work / "results/metrics.json")
        row = result["categories"][0]
        self.assertIn("control_enhance_PRO", row)
        self.assertNotIn("suppress_PRO", row)
        diag = result["diagnostics"][0]
        self.assertEqual(diag["tp"], 1)
        self.assertEqual(diag["fp"], 1)
        self.assertEqual(diag["enhance_changed_pixels"], 8)
        self.assertEqual(diag["enhance_new_fn_pixels_at_0_5"], 0)
        self.assertTrue((self.work / "results/candidate_decisions.csv").exists())

    def test_empty_candidates_need_no_vlm_and_preserve_base(self):
        self.make_manifest(flat=True)
        with patch("tools.vlm_decision.QwenVLLMTeacher") as teacher:
            dual_vlm.review(self.args)
            teacher.assert_not_called()
        dual_vlm.evaluate(self.args)
        result = load_json(self.work / "results/metrics.json")
        self.assertEqual(result["candidate_reviews"], 0)
        self.assertEqual(result["diagnostics"][0]["no_candidate_images"], 2)
        row = result["categories"][0]
        self.assertEqual(row["base_F1"], row["enhance_F1"])

    def test_missing_review_fails(self):
        self.make_manifest()
        with self.assertRaisesRegex(ValueError, "Missing candidate review"):
            dual_vlm.evaluate(self.args)

    def test_invalid_review_retries_and_evaluation_refuses(self):
        self.make_manifest()
        with patch("tools.vlm_decision.QwenVLLMTeacher") as teacher:
            teacher.return_value.generate.return_value = "bad"
            for shard in (0, 1):
                self.args.shard_id = shard
                dual_vlm.review(self.args)
            self.assertEqual(teacher.return_value.generate.call_count, 4)
        with self.assertRaisesRegex(ValueError, "Invalid local reviews"):
            dual_vlm.evaluate(self.args)

    def test_wrong_cached_candidate_refused(self):
        cfg, records = self.make_manifest()
        self.run_review()
        candidate = records[0]["rois"][0]
        path = review_path(self.work, records[0], candidate)
        payload = load_json(path)
        payload["candidate_id"] = 4
        atomic_json(path, payload)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            dual_vlm.evaluate(self.args)

    def test_reference_pool_reads_train_only_and_excludes_anomalies(self):
        normal = self.root / "normal.png"
        Image.new("RGB", (80, 40), (100, 110, 120)).save(normal)
        requested = []
        class Dataset:
            def __init__(self, **kwargs):
                requested.append(kwargs["split"])
                self.data_to_iterate = [("object", "Normal", str(normal), None), ("object", "Anomaly", "must-not-read", None)]
        cfg = self.prepare()
        fake = SimpleNamespace(DATASET_CLASSES={"visa": ["object"]},
               DATASET_REGISTRY={"visa": (Dataset, SimpleNamespace(TRAIN="train"), str(self.root))})
        with patch.dict("sys.modules", {"Datasets": fake}):
            bank = reference_bank(cfg)
        self.assertEqual(requested, ["train"])
        self.assertEqual(len(bank["visa/object"]), 1)
        self.assertEqual(bank["visa/object"][0]["path"], str(normal.resolve()))

    def test_reference_selection_rejects_query_duplicates_and_mismatch(self):
        from tools.local_vlm_pipeline import select_reference
        from tools.vlm_review import file_digest
        cfg = self.prepare()
        query = self.root / "query.png"
        duplicate = self.root / "duplicate.png"
        mismatch = self.root / "mismatch.png"
        Image.new("RGB", (80, 40), (100, 110, 120)).save(query)
        duplicate.write_bytes(query.read_bytes())
        Image.new("RGB", (80, 40), (255, 0, 0)).save(mismatch)
        cfg["reference_bank"] = {"visa/object": [{"path": str(p), "sha256": file_digest(p)} for p in (query, duplicate, mismatch)]}
        with Image.open(query) as img:
            context = img.convert("RGB")
        selected = select_reference(cfg, {"dataset": "visa", "category": "object"}, {"box": [12, 12, 14, 14]}, query, context, (32, 32))
        self.assertIsNone(selected)

    def test_native_export_two_shards_resume_and_evaluate_end_to_end(self):
        import torch
        import tools.utils_up
        cfg = self.prepare()
        raw = self.root / "source.png"
        Image.new("RGB", (800, 400), (100, 110, 120)).save(raw)
        class ToyDataset:
            def __init__(self, **kwargs):
                self.data_to_iterate = [("object", "good", str(raw), None)] * 4
            def __len__(self):
                return 4
            def __getitem__(self, i):
                rgb = torch.full((3, 32, 32), 128/255)
                mean = torch.tensor([.485, .456, .406])[:, None, None]
                std = torch.tensor([.229, .224, .225])[:, None, None]
                mask = torch.zeros(1, 32, 32)
                if i % 2:
                    mask[:, 12:14, 12:14] = 1
                return dict(image=(rgb-mean)/std, mask=mask, is_anomaly=i%2, image_path=str(raw))
        def forward(clip, batch, *args, **kwargs):
            p = .2 + batch["mask"] * .4
            return None, batch["mask"], torch.cat([1-p, p], 1), torch.zeros(len(p), 2), {"cross_prob_layers": [p[:, 0]] * 4}
        fake = SimpleNamespace(DATASET_CLASSES={"visa": ["object"]}, DATASET_REGISTRY={"visa": (ToyDataset, SimpleNamespace(TEST="test"), str(self.root))})
        with patch.dict("sys.modules", {"Datasets": fake}), patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.set_device"), patch("dual_vlm.load_dual", return_value=(None, None, None, None, (5, 11, 17, 23))), \
             patch("tools.utils_up.get_anomaly_map", side_effect=forward) as model:
            for shard in (0, 1):
                self.args.shard_id = shard
                dual_vlm.export(self.args)
            self.assertEqual(model.call_count, 2)
            dual_vlm.export(self.args)
            self.assertEqual(model.call_count, 2)
            # Incomplete crop export is repaired, never treated as completed.
            record = load_json(self.work / "export_shard_1.json")["records"][0]
            (self.work / record["rois"][0]["detail"]).unlink()
            dual_vlm.export(self.args)
            self.assertEqual(model.call_count, 3)
        dual_vlm.seal(self.args)
        records = load_json(self.work / "manifest.json")["records"]
        with Image.open(self.work / records[1]["rois"][0]["detail"]) as crop:
            self.assertEqual(crop.size, (200, 101))  # outward floor/ceil of half-pixel edges
            self.assertTrue(np.all(np.asarray(crop) == [100, 110, 120]))
        self.run_review()
        dual_vlm.evaluate(self.args)
        result = load_json(self.work / "results/metrics.json")
        self.assertEqual(result["candidate_reviews"], 2)
        self.assertEqual(result["diagnostics"][0]["no_candidate_images"], 2)

    def test_optional_suppression_modes_are_reported(self):
        self.args.include_suppression = True
        self.make_manifest()
        self.run_review()
        dual_vlm.evaluate(self.args)
        row = load_json(self.work / "results/metrics.json")["categories"][0]
        self.assertIn("suppress_PRO", row)
        self.assertIn("bidirectional_PRO", row)
        self.assertIn("control_suppress_PRO", row)


if __name__ == "__main__":
    unittest.main()
