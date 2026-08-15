import tempfile
import unittest
from pathlib import Path

import torch

from tools.semantic_anchor import SemanticAnchorAligner, load_external_semantic_anchors


class SemanticAnchorAlignerTests(unittest.TestCase):
    def test_matching_context_prototypes_have_near_zero_loss(self):
        anchors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        aligner = SemanticAnchorAligner(
            prompt_dim=3,
            anchors=anchors,
            margin=0.2,
            separation_weight=0.5,
        )
        normal_ctx = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
        anomaly_ctx = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]])
        output = aligner(normal_ctx, anomaly_ctx)
        self.assertLess(float(output["loss"].detach()), 1e-6)
        self.assertGreater(float(output["normal_similarity"].detach()), 0.999)
        self.assertGreater(float(output["anomaly_similarity"].detach()), 0.999)

    def test_anchor_loss_backpropagates_to_prompt_tokens_and_projector(self):
        anchors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        aligner = SemanticAnchorAligner(prompt_dim=3, anchors=anchors)
        normal_ctx = torch.nn.Parameter(torch.tensor([[[[0.0, 1.0, 0.2]]]]))
        anomaly_ctx = torch.nn.Parameter(torch.tensor([[[[1.0, 0.0, -0.2]]]]))
        output = aligner(normal_ctx, anomaly_ctx)
        output["loss"].backward()
        self.assertGreater(float(normal_ctx.grad.abs().sum()), 0.0)
        self.assertGreater(float(anomaly_ctx.grad.abs().sum()), 0.0)
        self.assertGreater(float(aligner.projector.weight.grad.abs().sum()), 0.0)

    def test_external_anchor_payload_is_normalized_and_auditable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "anchors.pt"
            torch.save(
                {
                    "anchors": torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
                    "labels": ["normal", "anomaly"],
                    "generator_model": "gemini-3.5-flash",
                    "embedding_model": "gemini-embedding-2",
                    "clip_encoded": False,
                },
                path,
            )
            anchors, metadata = load_external_semantic_anchors(str(path))
        self.assertEqual(tuple(anchors.shape), (2, 2))
        self.assertTrue(torch.allclose(anchors.norm(dim=1), torch.ones(2)))
        self.assertEqual(metadata["generator_model"], "gemini-3.5-flash")
        self.assertFalse(metadata["clip_encoded"])

    def test_clip_anchor_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clip_anchors.pt"
            torch.save(
                {
                    "anchors": torch.eye(2),
                    "labels": ["normal", "anomaly"],
                    "embedding_model": "clip",
                    "clip_encoded": True,
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "CLIP-encoded anchors are not accepted"):
                load_external_semantic_anchors(str(path))


if __name__ == "__main__":
    unittest.main()
