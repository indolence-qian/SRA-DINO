import unittest

import torch

from tools.mara_agent import MARAAgent, MARAConfig, _baseline_anchored_advantages


class BaselineAnchoredAdvantageTests(unittest.TestCase):
    def test_returns_below_base_never_receive_positive_advantage(self):
        returns = torch.tensor(
            [
                [0.0, 0.0],
                [-0.20, -0.10],
                [-0.05, 0.04],
                [0.10, 0.20],
            ]
        )
        active = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        base_mask = torch.tensor([True, False, False, False])

        advantages = _baseline_anchored_advantages(
            returns=returns,
            active_mask=active,
            base_trajectory_mask=base_mask,
            group_size=4,
            margin=0.01,
            negative_scale=1.0,
            clip=5.0,
        )

        self.assertLess(float(advantages[1, 0]), 0.0)
        self.assertLess(float(advantages[2, 0]), 0.0)
        self.assertGreater(float(advantages[3, 0]), 0.0)
        self.assertGreater(float(advantages[2, 1]), 0.0)


class HierarchicalPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = MARAConfig(
            num_layers=2,
            map_size=8,
            num_regions=4,
            roi_size=4,
            hidden_dim=8,
            max_steps=2,
            group_size=4,
        )

    @staticmethod
    def _grad_norm(module):
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.abs().sum())
        return total

    def test_stop_action_only_updates_op_policy_head(self):
        agent = MARAAgent(self.cfg)
        state = torch.rand(3, 4 + self.cfg.num_layers, 8, 8)
        global_logits = torch.zeros(3, 2)
        active = torch.ones(3, dtype=torch.bool)
        actions = {
            "region": torch.zeros(3, dtype=torch.long),
            "layer": torch.zeros(3, dtype=torch.long),
            "op": torch.zeros(3, dtype=torch.long),
        }

        output = agent._policy(
            state=state,
            global_logits=global_logits,
            active=active,
            step_idx=0,
            sample=False,
            actions=actions,
        )
        (-output["logprob"].sum()).backward()

        self.assertEqual(self._grad_norm(agent.region_head), 0.0)
        self.assertEqual(self._grad_norm(agent.layer_head), 0.0)
        self.assertGreater(self._grad_norm(agent.op_head), 0.0)

    def test_base_trajectory_has_zero_return_during_refiner_warmup(self):
        agent = MARAAgent(self.cfg)
        base_prob = torch.softmax(torch.randn(1, 2, 8, 8), dim=1)
        base_logits = torch.randn(1, 2)
        layer_maps = torch.rand(1, self.cfg.num_layers, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        labels = torch.zeros(1, dtype=torch.long)

        output = agent.rollout(
            base_prob=base_prob,
            base_logits=base_logits,
            layer_maps=layer_maps,
            mask=mask,
            labels=labels,
            group_size=self.cfg.group_size,
            sample=True,
            apply_gain_gate=False,
            force_refine=True,
        )

        trajectory = output["trajectory"]
        self.assertTrue(bool(trajectory["base_trajectory_mask"][0]))
        self.assertTrue(torch.allclose(output["returns"][0], torch.tensor(0.0)))
        self.assertTrue(torch.equal(trajectory["op_actions"][0], torch.zeros(self.cfg.max_steps, dtype=torch.long)))
        self.assertTrue(torch.equal(trajectory["op_actions"][1:], torch.ones(3, self.cfg.max_steps, dtype=torch.long)))

    def test_fixed_trajectory_replay_backpropagates_all_training_heads(self):
        agent = MARAAgent(self.cfg)
        base_prob = torch.softmax(torch.randn(1, 2, 8, 8), dim=1)
        base_logits = torch.randn(1, 2)
        layer_maps = torch.rand(1, self.cfg.num_layers, 8, 8)
        mask = torch.randint(0, 2, (1, 1, 8, 8)).float()
        labels = torch.ones(1, dtype=torch.long)

        with torch.no_grad():
            behavior = agent.rollout(
                base_prob=base_prob,
                base_logits=base_logits,
                layer_maps=layer_maps,
                mask=mask,
                labels=labels,
                group_size=self.cfg.group_size,
                sample=True,
                apply_gain_gate=False,
                force_refine=True,
            )
        replay = agent.rollout(
            base_prob=base_prob,
            base_logits=base_logits,
            layer_maps=layer_maps,
            mask=mask,
            labels=labels,
            group_size=self.cfg.group_size,
            sample=False,
            trajectory=behavior["trajectory"],
            apply_gain_gate=False,
        )
        loss = (
            replay["final_prob"].mean()
            + replay["policy_loss"]
            + replay["gain_loss"]
            + replay["gain_consistency_loss"]
            + replay["op_aux_loss"]
        )
        loss.backward()

        self.assertGreater(self._grad_norm(agent.region_head), 0.0)
        self.assertGreater(self._grad_norm(agent.layer_head), 0.0)
        self.assertGreater(self._grad_norm(agent.op_head), 0.0)
        self.assertGreater(self._grad_norm(agent.gain_lower_head), 0.0)


if __name__ == "__main__":
    unittest.main()
