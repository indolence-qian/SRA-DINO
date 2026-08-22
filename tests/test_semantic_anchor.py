import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.semantic_anchor import (
    SemanticAnchorAligner,
    load_external_semantic_anchors,
    load_semantic_anchor_descriptions,
)


class SemanticAnchorAlignerTests(unittest.TestCase):
    def test_matching_final_clip_features_have_near_zero_loss(self):
        normal_bank = torch.tensor([[1.0, 0.0, 0.0]])
        anomaly_bank = torch.tensor([[0.0, 1.0, 0.0]])
        aligner = SemanticAnchorAligner(
            normal_bank=normal_bank,
            anomaly_bank=anomaly_bank,
            margin=0.2,
            separation_weight=0.5,
        )
        prompt_features = torch.stack([normal_bank[0], anomaly_bank[0]])
        output = aligner(prompt_features)
        self.assertLess(float(output["loss"].detach()), 1e-5)
        self.assertGreater(float(output["direction_similarity"].detach()), 0.999)
        self.assertGreater(float(output["normal_similarity"].detach()), 0.999)
        self.assertGreater(float(output["anomaly_similarity"].detach()), 0.999)

    def test_loss_backpropagates_to_final_prompt_features_without_projector(self):
        aligner = SemanticAnchorAligner(
            normal_bank=torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]),
            anomaly_bank=torch.tensor([[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]]),
        )
        prompt_features = torch.nn.Parameter(
            torch.tensor([[0.1, 0.9, 0.2], [0.9, 0.1, -0.2]])
        )
        output = aligner(prompt_features)
        output["loss"].backward()
        self.assertGreater(float(prompt_features.grad.abs().sum()), 0.0)
        self.assertEqual(sum(param.numel() for param in aligner.parameters()), 0)

    def test_margin_is_capped_by_frozen_teacher_geometry(self):
        normal = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
        anomaly = F.normalize(torch.tensor([[0.99, 0.1]]), dim=-1)
        aligner = SemanticAnchorAligner(normal, anomaly, margin=0.2)
        teacher_gap = 1.0 - float(aligner.teacher_pair_similarity)
        self.assertLessEqual(float(aligner.adaptive_margin), 0.9 * teacher_gap + 1e-7)
        self.assertLess(float(aligner.adaptive_margin), 0.2)

    def test_reference_prompt_gap_prevents_teacher_center_collapse(self):
        normal = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
        anomaly = F.normalize(torch.tensor([[0.99, 0.1]]), dim=-1)
        aligner = SemanticAnchorAligner(
            normal,
            anomaly,
            margin=0.2,
            reference_prompt_gap=0.18,
        )
        self.assertAlmostEqual(float(aligner.adaptive_margin), 0.162, places=6)

    def test_description_sidecar_supplies_multi_anchor_banks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "anchors.pt"
            torch.save(
                {
                    "anchors": torch.eye(2),
                    "descriptions": {"normal": "intact object", "anomaly": "broken object"},
                    "embedding_model": "external-test-model",
                    "clip_encoded": False,
                },
                path,
            )
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "description_banks": {
                            "normal": ["intact object", "uniform surface"],
                            "anomaly": ["broken object", "cracked surface", "foreign matter"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            banks, metadata = load_semantic_anchor_descriptions(str(path))
        self.assertEqual(len(banks["normal"]), 2)
        self.assertEqual(len(banks["anomaly"]), 3)
        self.assertEqual(metadata["anchor_space"], "frozen_clip_text")
        self.assertFalse(metadata["external_embedding_used_for_training"])

    def test_legacy_external_embeddings_remain_auditable_but_are_not_training_space(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "anchors.pt"
            torch.save(
                {
                    "anchors": torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
                    "labels": ["normal", "anomaly"],
                    "embedding_model": "gemini-embedding-2",
                    "clip_encoded": False,
                },
                path,
            )
            anchors, metadata = load_external_semantic_anchors(str(path))
        self.assertEqual(tuple(anchors.shape), (2, 2))
        self.assertTrue(torch.allclose(anchors.norm(dim=1), torch.ones(2)))
        self.assertFalse(metadata["legacy_embedding_used_for_training"])


if __name__ == "__main__":
    unittest.main()
