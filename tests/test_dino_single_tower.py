import unittest

import torch
import torch.nn.functional as F

from tools.dino_single_tower import DinoSingleTowerConfig, DinoVisualPrototypeHead
from tools.mara_agent import MARAAgent, MARAConfig
from tools.mara_evidence import build_mara_evidence


class DinoSingleTowerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = DinoSingleTowerConfig(
            visual_layers=(1, 2),
            input_dim=16,
            embed_dim=8,
            normal_prototypes=2,
            anomaly_prototypes=3,
            attention_heads=2,
            evidence_channels=3,
            temperature=0.1,
            topk_ratio=0.25,
        )
        self.head = DinoVisualPrototypeHead(self.config)
        self.cls_tokens = [torch.randn(2, 1, 16) for _ in self.config.visual_layers]
        self.patch_tokens = [torch.randn(2, 16, 16) for _ in self.config.visual_layers]

    def test_visual_head_outputs_valid_probabilities_and_evidence(self):
        output = self.head(self.cls_tokens, self.patch_tokens, output_size=(32, 32))
        self.assertEqual(tuple(output["prob"].shape), (2, 2, 32, 32))
        self.assertEqual(tuple(output["global_logits"].shape), (2, 2))
        self.assertEqual(len(output["layer_logits"]), 2)
        self.assertTrue(
            torch.allclose(
                output["prob"].sum(dim=1),
                torch.ones(2, 32, 32),
                atol=1e-5,
            )
        )
        evidence = output["evidence"]
        self.assertEqual(len(evidence["feature_layers"]), 2)
        self.assertEqual(tuple(evidence["feature_layers"][0].shape), (2, 3, 4, 4))

    def test_dense_and_global_losses_reach_visual_prototypes(self):
        output = self.head(self.cls_tokens, self.patch_tokens, output_size=(16, 16))
        mask = torch.randint(0, 2, (2, 16, 16), dtype=torch.float32)
        labels = torch.tensor([0, 1])
        loss = (
            F.nll_loss(output["prob"].clamp_min(1e-6).log(), mask.long())
            + F.cross_entropy(output["global_logits"], labels)
            + 0.01 * output["prototype_regularization"]
        )
        loss.backward()
        self.assertIsNotNone(self.head.normal_bank.grad)
        self.assertIsNotNone(self.head.anomaly_bank.grad)
        self.assertGreater(float(self.head.normal_bank.grad.norm()), 0.0)
        self.assertGreater(float(self.head.anomaly_bank.grad.norm()), 0.0)
        self.assertIsNotNone(self.head.layer_adapters[0].project.weight.grad)

    def test_mara_receives_compressed_dino_feature_channels(self):
        output = self.head(self.cls_tokens, self.patch_tokens, output_size=(32, 32))
        packed = build_mara_evidence(
            evidence=output["evidence"],
            fallback_prob=output["prob"],
            num_layers=2,
            map_size=8,
            feature_channels_per_layer=3,
        )
        # Four scalar evidence maps + three DINO feature channels per layer,
        # plus two cross-layer disagreement maps.
        self.assertEqual(tuple(packed["extra_maps"].shape), (2, 16, 8, 8))
        self.assertEqual(tuple(packed["layer_maps"].shape), (2, 2, 8, 8))
        self.assertEqual(tuple(packed["global_evidence"].shape), (2, 2))

        agent = MARAAgent(
            MARAConfig(
                num_layers=2,
                evidence_channels=16,
                global_evidence_dim=2,
                map_size=8,
                roi_size=4,
                num_regions=4,
                hidden_dim=16,
                max_steps=1,
                group_size=1,
            )
        ).eval()
        refined = agent.infer(
            base_prob=output["prob"],
            base_logits=output["global_logits"],
            layer_maps=packed["layer_maps"],
            evidence_maps=packed["extra_maps"],
            global_evidence=packed["global_evidence"],
        )
        self.assertEqual(tuple(refined["final_prob"].shape), (2, 2, 32, 32))
        self.assertTrue(torch.isfinite(refined["final_prob"]).all())

    def test_config_round_trip_preserves_tuple_fields(self):
        restored = DinoSingleTowerConfig.from_dict(self.config.to_dict())
        self.assertEqual(restored, self.config)
        self.assertIsInstance(restored.visual_layers, tuple)
        self.assertIsInstance(restored.hfa_layers, tuple)


if __name__ == "__main__":
    unittest.main()
