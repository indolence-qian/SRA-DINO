import unittest

import torch

from tools.vlm_decision import (
    ACTION_NAMES,
    build_vlm_prompt,
    candidate_boxes_from_map,
    decision_target_map,
    parse_vlm_decision,
)


class VLMDecisionTests(unittest.TestCase):
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
