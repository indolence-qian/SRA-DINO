import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from tools.vlm_decision import (
    ACTION_NAMES,
    QwenVLLMTeacher,
    build_vlm_prompt,
    candidate_boxes_from_map,
    decision_target_map,
    parse_vlm_decision,
)


class VLMDecisionTests(unittest.TestCase):
    def make_teacher(self, **kwargs):
        processor = MagicMock()
        processor.image_processor.patch_size = 16
        processor.apply_chat_template.return_value = "rendered prompt"
        vision = MagicMock(return_value=(["image inputs"], None, {}))
        llm = MagicMock()
        llm.return_value.generate.return_value = [
            SimpleNamespace(outputs=[SimpleNamespace(text="teacher JSON")])
        ]
        modules = {
            "qwen_vl_utils": SimpleNamespace(process_vision_info=vision),
            "vllm": SimpleNamespace(LLM=llm, SamplingParams=MagicMock()),
            "transformers": SimpleNamespace(
                AutoProcessor=SimpleNamespace(
                    from_pretrained=MagicMock(return_value=processor)
                )
            ),
        }
        with patch.dict("sys.modules", modules), patch.dict("os.environ"):
            teacher = QwenVLLMTeacher(**kwargs)
        return teacher, llm, vision

    def test_teacher_uses_bounded_image_only_engine(self):
        _, llm, _ = self.make_teacher(
            max_images=6, teacher_image_size=448, max_model_len=3072
        )
        config = llm.call_args.kwargs
        self.assertEqual(config["limit_mm_per_prompt"], {"image": 6, "video": 0})
        self.assertEqual(config["mm_processor_kwargs"], {
            "min_pixels": 1024, "max_pixels": 448 ** 2,
        })
        self.assertEqual(config["max_num_seqs"], 1)
        self.assertEqual(config["max_num_batched_tokens"], 3072)
        self.assertTrue(config["enforce_eager"])
        self.assertEqual(config["tensor_parallel_size"], 1)
        self.assertEqual(config["gpu_memory_utilization"], 0.70)

    def test_teacher_request_and_preprocessing_share_pixel_cap(self):
        teacher, llm, vision = self.make_teacher(teacher_image_size=448)
        # Even a conflicting per-request utility default cannot drop our cap.
        vision.return_value = (["image inputs"], None, {"max_pixels": 999999})
        result = teacher.generate("inspect", [Image.new("RGB", (64, 64))])
        self.assertEqual(result, "teacher JSON")
        image_item = vision.call_args.args[0][0]["content"][0]
        self.assertEqual(image_item["max_pixels"], 448 ** 2)
        self.assertEqual(image_item["min_pixels"], 1024)
        request = llm.return_value.generate.call_args.args[0][0]
        self.assertEqual(request["mm_processor_kwargs"], {
            "min_pixels": 1024, "max_pixels": 448 ** 2,
        })
        self.assertEqual(request["multi_modal_data"], {"image": ["image inputs"]})

    def test_teacher_accepts_empty_utility_kwargs(self):
        teacher, llm, vision = self.make_teacher()
        vision.return_value = (["image inputs"], None, None)
        teacher.generate("inspect", [Image.new("RGB", (32, 32))])
        request = llm.return_value.generate.call_args.args[0][0]
        self.assertEqual(request["mm_processor_kwargs"]["max_pixels"], 512 ** 2)

    def test_teacher_rejects_bad_limits_before_loading_engine(self):
        for kwargs in (
            {"teacher_image_size": 0}, {"max_images": 0},
            {"max_tokens": 4096}, {"max_tokens": 0},
            {"min_pixels": 0}, {"min_pixels": 512**2+1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QwenVLLMTeacher(**kwargs)

    def test_diagnostic_budget_and_trace_are_opt_in(self):
        from pathlib import Path
        import tempfile
        import json
        teacher, llm, vision = self.make_teacher(min_pixels=65536)
        image = Image.new("RGB", (256,256))
        vision.return_value = ([image], None, {})
        teacher._processor.image_processor.merge_size = 2
        teacher._processor.image_processor.return_value = {"image_grid_thw": torch.tensor([[1,16,16]])}
        with tempfile.TemporaryDirectory() as folder:
            teacher.generate("inspect", [Image.new("RGB", (8,8))], trace_dir=folder)
            trace = json.loads((Path(folder)/"trace.json").read_text())
            self.assertEqual(trace["input_sizes"], [[8,8]])
            self.assertEqual(trace["post_vision_sizes"], [[256,256]])
            self.assertEqual(trace["processor_probe_tokens"], [64])
            self.assertEqual(trace["pixel_budget"]["min_pixels"], 65536)
            self.assertTrue((Path(folder)/"post_vision_0.png").exists())
        request = llm.return_value.generate.call_args.args[0][0]
        self.assertEqual(request["mm_processor_kwargs"]["min_pixels"], 65536)

    def test_teacher_rejects_empty_or_excess_images(self):
        teacher, llm, vision = self.make_teacher(max_images=1)
        for images in ([], [Image.new("RGB", (32, 32))] * 2):
            with self.assertRaises(ValueError):
                teacher.generate("inspect", images)
        vision.assert_not_called()
        llm.return_value.generate.assert_not_called()

    def test_fenced_json_is_parsed_and_constrained(self):
        response = """```json
        {"image_anomaly_probability": 1.4, "action": "REFINE",
         "regions": [{"box": [-5, 20, 1100, 900], "defect_probability": 0.8}],
         "preferred_layer": 99, "confidence": 0.7}
        ```"""
        decision = parse_vlm_decision(response, num_layers=4)
        self.assertTrue(decision.parse_ok)
        self.assertEqual(decision.action, "refine")
        self.assertEqual(decision.preferred_layer, 3)
        self.assertEqual(decision.image_anomaly_probability, 1.0)
        self.assertEqual(decision.regions[0]["box"], [0.0, 20.0, 1000.0, 900.0])

    def test_invalid_response_fails_closed(self):
        decision = parse_vlm_decision("this is not JSON", num_layers=4)
        self.assertFalse(decision.parse_ok)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.confidence, 0.0)

    def test_peak_boxes_and_dense_region_target(self):
        anomaly_map = torch.zeros(8, 8)
        anomaly_map[6, 2] = 1.0
        boxes = candidate_boxes_from_map(anomaly_map, num_rois=2, roi_fraction=0.25)
        self.assertEqual(len(boxes), 2)
        prompt = build_vlm_prompt("capsule", boxes, num_layers=4)
        self.assertIn("capsule", prompt)
        self.assertIn("preferred_layer", prompt)
        response = (
            '{"image_anomaly_probability":0.9,"action":"refine",'
            f'"regions":[{{"box":{boxes[0]},"defect_probability":0.75}}],'
            '"preferred_layer":1,"confidence":0.8}'
        )
        decision = parse_vlm_decision(response, 4)
        target = decision_target_map(decision, map_size=16)
        self.assertEqual(tuple(target.shape), (1, 16, 16))
        self.assertAlmostEqual(float(target.max()), 0.75, places=5)
        self.assertIn(decision.action, ACTION_NAMES)


if __name__ == "__main__":
    unittest.main()
