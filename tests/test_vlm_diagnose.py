"""CPU-only fixtures for paired diagnosis, source isolation and trace caching."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import dual_vlm
import vlm_diagnose as diag
from tools.local_vlm_review import local_prompt, parse_local
from tools.vlm_review import atomic_json, digest, load_json


class FakeTeacher:
    calls = []

    def __init__(self, **kwargs):
        self.config = kwargs

    def generate(self, prompt, images, trace_dir=None):
        self.calls.append((self.config, prompt, [im.size for im in images]))
        # Do not read any GT: only arm settings influence this synthetic answer.
        neutral = "Visibility means" in prompt
        high = self.config["min_pixels"] > 1024
        ident = 1 if "candidate 1." in prompt else 2
        verdict = "defect_supported" if neutral and high else "insufficient_evidence"
        raw = json.dumps(dict(candidate_id=ident, verdict=verdict,
                              visibility="sufficient" if high else "insufficient",
                              reference_match="unavailable", defect_type="unknown", evidence="synthetic CPU test"))
        trace_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(images):
            im.save(trace_dir / f"post_vision_{i}.png")
        atomic_json(trace_dir / "trace.json", dict(input_sizes=[list(im.size) for im in images],
                    post_vision_sizes=[list(im.size) for im in images], processor_probe_tokens=[16,16],
                    finish_reason="stop", output_tokens=24, raw_response=raw))
        return raw


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src, self.out, self.model = self.root / "source", self.root / "diagnosis", self.root / "model"
        self.model.mkdir()
        (self.model / "model.safetensors").write_bytes(b"fixture, not a real model")
        atomic_json(self.model / "config.json", {"model_type": "fixture"})
        original = dict(local_review=True, teacher_image_size=512, gpu_memory=.7, max_model_len=4096,
                        max_tokens=512, retries=1, alpha=.25, local_prompt="local",
                        model={"path": str(self.model)})
        atomic_json(self.src / "config.json", original)
        records = []
        for index in range(3):
            ident = f"sample{index}"
            r = dict(id=ident, index=index, dataset="visa", category=f"category{index}",
                     fingerprint=digest(original), query=f"queries/{ident}.png",
                     evidence=f"evidence/{ident}.npz", supports=f"supports/{ident}.npz",
                     evaluation=f"evaluation_only/{ident}.npz", rois=[])
            masks = np.zeros((2,32,32), bool)
            masks[0,4:8,4:8] = True
            masks[1,20:25,20:25] = True
            dual_vlm.atomic_npz(self.src / r["supports"], masks=masks)
            dual_vlm.atomic_npz(self.src / r["evidence"], base=np.full((32,32), .45, np.float32))
            dual_vlm.atomic_npz(self.src / r["evaluation"], mask=masks[0].astype(np.uint8))
            dual_vlm.atomic_image(self.src / r["query"], np.full((32,32,3), 50+index*30, np.uint8))
            for i in range(2):
                c = dict(roi_id=i+1, area=int(masks[i].sum()), box=[4,4,8,8] if i==0 else [20,20,25,25],
                         context=f"local_crops/{ident}_{i+1}_context.png", detail=f"local_crops/{ident}_{i+1}_detail.png",
                         geometry=dict(source_size=[64,64], detail_box=[8,8,16,16], context_box=[0,0,32,32]))
                for name, side in (("context",32), ("detail",8)):
                    dual_vlm.atomic_image(self.src / c[name], np.full((side,side,3), 50+index*30, np.uint8))
                r["rois"].append(c)
            records.append(r)
        atomic_json(self.src / "manifest.json", dict(fingerprint=digest(original), records=records))
        self.args = SimpleNamespace(source_dir=str(self.src), work_dir=str(self.out), model_id=str(self.model),
                                    candidates=0, seed=42, num_shards=2, min_pixels=65536, sanity_samples=2,
                                    shard_id=0, variant="A_original")
        FakeTeacher.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def prepare_and_review(self):
        real_asset = diag.safe_asset
        def forbid_gt(root, relative):
            if "evaluation_only" in relative:
                raise AssertionError("GT opened before evaluate")
            return real_asset(root, relative)
        with patch.object(diag, "safe_asset", side_effect=forbid_gt):
            diag.prepare(self.args)
            with patch("tools.vlm_decision.QwenVLLMTeacher", FakeTeacher):
                for variant in diag.VARIANTS:
                    self.args.variant = variant
                    for shard in range(2):
                        self.args.shard_id = shard
                        diag.review(self.args)

    def test_paired_pipeline_no_gt_in_prepare_or_review(self):
        self.prepare_and_review()
        self.assertEqual(len(FakeTeacher.calls), 24)
        diag.evaluate(self.args)
        report = load_json(self.out / "results/summary.json")
        rows = {r["variant"]: r for r in report["summaries"]}
        self.assertEqual(rows["A_original"]["keep"], 6)
        self.assertEqual(rows["D_prompt_pixels"]["enhance"], 6)
        self.assertEqual(rows["D_prompt_pixels"]["tp"], 3)
        self.assertEqual(rows["D_prompt_pixels"]["fp"], 3)
        with (self.out / "results/pixel_effects.csv").open() as f:
            pixels = list(csv.DictReader(f))
        self.assertTrue(any(r["variant"] == "GT_overlap_diagnostic_ONLY" for r in pixels))
        self.assertTrue((self.out / "input_audit/sample0_1.png").exists())
        self.assertFalse((self.src / "results").exists())
        with patch("tools.vlm_decision.QwenVLLMTeacher", side_effect=AssertionError("Cache should skip engine")):
            diag.review(self.args)
        self.assertEqual(len(FakeTeacher.calls), 24)

    def test_selection_deterministic_count_and_visible_sentinel(self):
        records = load_json(self.src / "manifest.json")["records"]
        a = diag.select_candidates(records, 3, 42, {"sample2_2"})
        self.assertEqual(a, diag.select_candidates(records, 3, 42, {"sample2_2"}))
        self.assertEqual(len(a), 3)
        self.assertIn("sample2_2", {x["key"] for x in a})
        self.assertEqual(len({x["key"] for x in diag.select_candidates(records, 0, 42)}), 6)

    def test_neutral_prompt_does_not_fill_abstention_answer(self):
        prompt = diag.neutral_prompt("pcb1", 1)
        self.assertIn("Visibility means", prompt)
        self.assertNotIn('"verdict":"insufficient_evidence"', prompt)
        self.assertNotIn("GT", prompt)
        r, c = load_json(self.src / "manifest.json")["records"][0], {"roi_id": 1}
        self.assertEqual(diag.original_prompt(r,c,{"local_prompt":"generic"}), local_prompt(r["category"],1,generic=True))

    def test_prepare_does_not_overwrite_changed_settings(self):
        diag.prepare(self.args)
        before = (self.out / "diagnosis_config.json").read_bytes()
        self.args.min_pixels = 32768
        with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
            diag.prepare(self.args)
        self.assertEqual(before, (self.out / "diagnosis_config.json").read_bytes())

    def test_changed_source_or_code_rejected(self):
        diag.prepare(self.args)
        with patch.object(diag, "code_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "code changed"):
                diag.load_config(self.args)
        p = self.src / "local_crops/sample0_1_detail.png"
        Image.new("RGB", (8,8), "white").save(p)
        with self.assertRaisesRegex(ValueError, "Source changed"):
            diag.load_config(self.args)

    def test_source_and_output_cannot_overlap(self):
        self.args.work_dir = str(self.src / "nested")
        with self.assertRaisesRegex(ValueError, "separate"):
            diag.prepare(self.args)

    def test_missing_arm_stops_evaluation_before_gt(self):
        diag.prepare(self.args)
        with patch.object(np, "load", side_effect=AssertionError("No GT before paired results")):
            with self.assertRaises(FileNotFoundError):
                diag.evaluate(self.args)

    def test_swapped_review_and_tampered_trace_rejected(self):
        self.prepare_and_review()
        cfg = diag.load_config(self.args)
        a, b = self.out / "A_original/sample0_1.json", self.out / "A_original/sample0_2.json"
        original = a.read_bytes()
        a.write_bytes(b.read_bytes())
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            diag.validate_review(a, cfg)
        a.write_bytes(original)
        p = self.out / "A_original/traces/sample0_1/0/post_vision_0.png"
        Image.new("RGB", (16,16)).save(p)
        with self.assertRaisesRegex(ValueError, "Changed trace"):
            diag.validate_review(a, cfg)

    def test_sanity_outputs_are_separate(self):
        diag.prepare(self.args)
        with patch("tools.vlm_decision.QwenVLLMTeacher", FakeTeacher):
            for shard in range(2):
                self.args.shard_id = shard
                diag.sanity(self.args)
        self.assertEqual(len(list((self.out / "sanity").glob("*.json"))), 6)
        self.assertFalse((self.out / "results").exists())

    def test_controls_preserve_count_and_exact_outside_identity(self):
        masks = np.zeros((3,16,16), bool)
        masks[0,0,0:2] = True
        masks[1,2,0:2] = True
        masks[2,4,0:4] = True
        votes = [diag.control_vote(True), diag.control_vote(False), diag.control_vote(False)]
        ctrl = diag.matched_control(masks, votes, 42)
        self.assertEqual(sum(diag.action(v)=="enhance" for v in ctrl), 1)
        base = np.full((16,16), .45, np.float32)
        gt = masks[0]
        row = diag.pixel_effect("test", "id", base, gt, masks, votes, .25)
        self.assertEqual(row["changed_pixels"], 2)
        self.assertEqual(row["changed_gt_pixels"], 2)
        self.assertEqual(row["changed_background_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
