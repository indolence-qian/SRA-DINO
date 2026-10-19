import copy
import csv
from dataclasses import asdict, replace
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

import torch

import mara_diagnose as shared
import mara_ppo_experiment as experiment
from tools.mara_agent import MARAAgent, MARAConfig, build_candidate_masks, quality_score
from tools.mara_reward import EvidenceStore, state_hash
from tools.mara_ppo import (
    PROTOCOL, ARMS, PPOConfig, MARAPPOAgent, agent_from_payload,
    environment_config, optimizers, return_targets, score_trajectory, update,
)
import test_mara_diagnostics as fixtures
import test_mara_reward_experiment as reward_fixtures


def toy(device="cpu", use_value=True):
    old, inputs, mask, labels = fixtures.toy()
    agent = MARAPPOAgent(environment_config(old.cfg), PPOConfig(trajectories=4, update_epochs=2, minibatch_size=5,
                                                               use_value=use_value)).to(device)
    agent.initialize_refiner(old.state_dict())
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return agent, inputs, mask.to(device), labels.to(device)


class ReturnTests(unittest.TestCase):
    def test_long_term_credit_negative_first_step_and_terminal(self):
        r = torch.tensor([[-.01, .02, .05], [.02, 0., 0.]])
        done = torch.tensor([[False, False, True], [False, True, True]])
        valid = torch.tensor([[True, True, True], [True, True, False]])
        values = torch.tensor([[.02, .03, .01], [.1, -.1, 100.]])
        returns, advantages = return_targets(r, done, valid, values)
        torch.testing.assert_close(returns, torch.tensor([[.06, .07, .05], [.02, 0., 0.]]))
        torch.testing.assert_close(advantages[0], torch.full((3,), .04))
        torch.testing.assert_close(advantages, (returns-values)*valid)
        _, mc = return_targets(r, done, valid, values, use_value=False)
        self.assertTrue(torch.equal(mc, returns))
        # No bootstrap even if an (irrelevant) trailing V is huge.
        self.assertEqual(float(returns[1, 0]), float(r[1, 0]))

    def test_discounted_mc_and_gae(self):
        r = torch.tensor([[1., 2., 3.]])
        d = torch.tensor([[False, False, True]])
        valid = torch.ones_like(d)
        v = torch.tensor([[.1, .2, .3]])
        returns, a = return_targets(r, d, valid, v, gamma=.9)
        torch.testing.assert_close(a, returns-v)
        _, td = return_targets(r, d, valid, v, gamma=.9, lam=0)
        torch.testing.assert_close(td, torch.tensor([[1+.9*.2-.1, 2+.9*.3-.2, 3-.3]]))

    def test_invalid_config_fails(self):
        for kwargs in (dict(gamma=0), dict(gae_lambda=2), dict(trajectories=0), dict(actor_lr=float("nan"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PPOConfig(**kwargs)


class PPOTests(unittest.TestCase):
    def test_observation_contains_transition_memory_and_no_GT(self):
        agent, inputs, mask, labels = toy()
        result = agent.trajectory(inputs, keep_observations=True)
        obs = result["steps"][0]["observation"]
        self.assertEqual(obs["spatial"].shape, (1, 7+2+10, 8, 8))
        self.assertEqual(obs["context"].shape, (1, 5+2))
        self.assertEqual(float(obs["spatial"][:, -1].sum()), 0.)
        self.assertEqual(float(obs["context"][:, -1]), 1.)
        self.assertTrue(torch.equal(obs["context"][:, 2:4], inputs["base_logits"]))
        self.assertFalse(obs["spatial"].requires_grad)
        with self.assertRaisesRegex(ValueError, "GT"):
            agent.trajectory(dict(inputs, labels=labels))
        with self.assertRaisesRegex(ValueError, "GT"):
            agent.trajectory(dict(inputs, mask=mask))

    def test_exact_joint_entropy_and_STOP_log_probability(self):
        agent, inputs, _, _ = toy()
        obs = agent.trajectory(inputs, keep_observations=True)["steps"][0]["observation"]
        actions = dict(op=torch.zeros(1, dtype=torch.long), region=torch.tensor([2]), layer=torch.tensor([1]))
        policy = agent.actor(obs, actions=actions)
        p = {k: v.softmax(-1) for k, v in policy["logits"].items()}
        joint = torch.cat([p["op"][:, :1], (p["op"][:, 1, None, None]*p["region"][:, :, None]*p["layer"][:, None, :]).flatten(1)], 1)
        torch.testing.assert_close(joint.sum(1), torch.ones(1))
        torch.testing.assert_close(policy["entropy"], -(joint*joint.log()).sum(1))
        torch.testing.assert_close(policy["logprob"], p["op"][:, 0].log())
        policy["logprob"].sum().backward()
        self.assertEqual(float(agent.actor.layer.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(agent.actor.region[-1].weight.grad.abs().sum()), 0.)

    def test_vectorized_ROI_matches_inclusive_reference(self):
        torch.manual_seed(1)
        anomaly, uncertainty = torch.rand(3, 1, 8, 8), torch.rand(3, 1, 8, 8)
        masks, boxes = build_candidate_masks(anomaly, uncertainty, 16, 4)
        reference = torch.zeros_like(masks)
        integer = (boxes*7).round().long()
        for b in range(3):
            for k in range(16):
                y1, x1, y2, x2 = integer[b, k]
                reference[b, k, y1:y2+1, x1:x2+1] = 1
        self.assertTrue(torch.equal(masks, reference))

    def test_STOP_after_REFINE_freezes_every_state_and_HR_reward_telescopes(self):
        agent, inputs, mask, labels = toy()
        forward = agent.actor.forward
        calls = []
        def scheduled(observation, **kwargs):
            op = 1 if not calls else 0
            calls.append(op)
            actions = {"op": torch.tensor([op]), "region": torch.tensor([0]), "layer": torch.tensor([0])}
            return forward(observation, actions=actions)
        with patch.object(agent.actor, "forward", side_effect=scheduled):
            result = agent.trajectory(inputs, keep_observations=True)
        score_trajectory(result, mask, labels, agent.cfg, agent.ppo)
        steps = result["steps"]
        self.assertEqual([int(s["valid"]) for s in steps], [1, 1, 0])
        self.assertGreater(float(steps[0]["pixel_change"]), 0.)
        for step in steps[1:]:
            self.assertEqual(float(step["pixel_change"]), 0.)
            self.assertEqual(float(step["global_change"]), 0.)
            self.assertEqual(float(step["reward"]), 0.)
        for key in ("spatial", "context"):
            # Time changes, but memory Z/C and current/base global logits do not.
            left, right = steps[1]["observation"][key], steps[2]["observation"][key]
            if key == "spatial":
                self.assertTrue(torch.equal(left[:, -3:], right[:, -3:]))
            else:
                self.assertTrue(torch.equal(left[:, :-1], right[:, :-1]))
        expected = steps[-1]["quality_after"]-steps[0]["quality_before"]-.003
        torch.testing.assert_close(steps[0]["return"], expected)
        scalar = quality_score(steps[0]["after_hr"], steps[0]["after_logits"], mask, labels, agent.cfg)
        torch.testing.assert_close(scalar, steps[0]["quality_after"])

    def check_training(self, device, use_value=True):
        agent, inputs, mask, labels = toy(device, use_value)
        before = state_hash(agent.transition.state_dict())
        actor_before, critic_before = state_hash(agent.actor.state_dict()), state_hash(agent.critic.state_dict())
        torch.manual_seed(12)
        batch, summary = agent.collect(inputs, mask, labels)
        self.assertGreater(summary["decisions"], 0)
        self.assertEqual(summary["trajectories"], 4)
        self.assertTrue(all(not t.requires_grad for t in batch["observation"].values()))
        replay = agent.actor(batch["observation"], actions=batch["actions"])
        torch.testing.assert_close(replay["logprob"], batch["old_logprob"], atol=1e-6, rtol=1e-6)
        observations = {k: v.clone() for k, v in batch["observation"].items()}
        advantage = batch["advantage"].clone()
        # Parameter updates MUST NOT regenerate a trajectory or call the Refiner.
        with patch.object(agent, "trajectory", side_effect=AssertionError("no transition replay")), \
             patch("tools.mara_ppo.build_candidate_masks", side_effect=AssertionError("no behavior ROI regeneration")):
            metrics = update(agent, batch, *optimizers(agent))
        self.assertGreater(metrics["optimizer_steps"], 0)
        self.assertEqual(before, state_hash(agent.transition.state_dict()))
        self.assertNotEqual(actor_before, state_hash(agent.actor.state_dict()))
        self.assertEqual(critic_before == state_hash(agent.critic.state_dict()), not use_value)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in agent.transition.parameters()))
        self.assertTrue(torch.equal(advantage, batch["advantage"]))
        self.assertTrue(all(torch.equal(v, batch["observation"][k]) for k, v in observations.items()))
        return agent, inputs

    def test_real_CPU_training_freezes_environment_and_uses_stored_states(self):
        self.check_training("cpu")

    def test_zero_baseline_control_has_no_critic_updates(self):
        self.check_training("cpu", False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_CUDA_training_and_checkpoint_roundtrip(self):
        agent, inputs = self.check_training("cuda")
        payload = dict(mara_algorithm=PROTOCOL, mara_config=asdict(agent.cfg), ppo_config=asdict(agent.ppo),
                       mara_ppo_agent=agent.state_dict())
        loaded = agent_from_payload(payload).cuda()
        for k in ("final_prob", "final_logits", "gate_map"):
            self.assertTrue(torch.equal(agent.infer(**inputs)[k], loaded.infer(**inputs)[k]))
        bad = copy.deepcopy(payload)
        bad["mara_ppo_agent"].pop("critic.value.2.bias")
        with self.assertRaises(RuntimeError):
            agent_from_payload(bad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_production_spatial_dimensions_CUDA_update(self):
        # Real 29x128x128 actor state and 512x512 reward maps. Small batch only:
        # the local GPU is not a 24GB 4090; this is not a full-budget VRAM claim.
        cfg = environment_config(MARAConfig(num_layers=4, evidence_channels=18, global_evidence_dim=4))
        env = MARAAgent(cfg)
        with torch.no_grad():
            env.delta_head.bias.copy_(torch.tensor([-.1, .1]))
        agent = MARAPPOAgent(cfg, PPOConfig(trajectories=1, update_epochs=1, minibatch_size=1)).cuda()
        agent.initialize_refiner(env.state_dict())
        inputs = dict(base_prob=torch.rand(1, 2, 512, 512, device="cuda").softmax(1),
                      base_logits=torch.zeros(1, 2, device="cuda"),
                      layer_maps=torch.rand(1, 4, 128, 128, device="cuda"),
                      evidence_maps=torch.rand(1, 18, 128, 128, device="cuda"),
                      global_evidence=torch.rand(1, 4, device="cuda"))
        mask = torch.zeros(1, 1, 512, 512, device="cuda")
        mask[:, :, 32:64, 32:64] = 1
        batch, _ = agent.collect(inputs, mask, torch.ones(1, dtype=torch.long, device="cuda"))
        self.assertEqual(batch["observation"]["spatial"].shape[1:], (29, 128, 128))
        frozen = state_hash(agent.transition.state_dict())
        update(agent, batch, *optimizers(agent))
        self.assertEqual(frozen, state_hash(agent.transition.state_dict()))

    def test_ordinary_evaluator_dispatches_PPO_strictly(self):
        agent, inputs, _, _ = toy()
        evaluator = fixtures.real_base_loader()
        payload = dict(mara_algorithm=PROTOCOL, mara_config=asdict(agent.cfg), ppo_config=asdict(agent.ppo),
                       mara_ppo_agent=agent.state_dict(), base_arch="clip_dino", visual_backbone="clip", visual_layers=[0, 1],
                       cls_token_adapter={}, patch_token_adapter={}, prompt_adapter={}, prompt_learner={})
        runtime = shared.base_export_runtime(dict(image_size=16, visual_backbone="clip", visual_layers=[0, 1],
                                                 hfa_setting="hfa4", hfa_layers="", hfa_bottleneck=None, dino_bottleneck=8,
                                                 clip_model_name="test", clip_pretrained="test"))
        models = evaluator.build_models_from_checkpoint(runtime, torch.device("cpu"), payload)
        self.assertIsInstance(models[5], MARAPPOAgent)
        evaluator.load_mara_checkpoint("toy.pth", payload, models[4], models[1], models[3], models[5], torch.device("cpu"))
        self.assertEqual(state_hash(agent.state_dict()), state_hash(models[5].state_dict()))
        runtime.gain_safety_margin = -.1
        with self.assertRaisesRegex(ValueError, "no gain"):
            evaluator.config_from_payload(payload, runtime)


class PPOPipelineTests(unittest.TestCase):
    def setUp(self):
        reward_fixtures.RewardPipelineTests.setUp(self)
        self.args = experiment.parser().parse_args([
            "--stage", "prepare", "--work_dir", str(self.work), "--mara_ckpt", str(self.ckpt),
            "--datasets", "visa toy", "--samples_per_category", "4", "--num_shards", "1",
            "--image_size", "16", "--pro_num_th", "21", "--device", "cpu", "--metric_workers", "1",
            "--epochs", "2", "--train_batch_size", "2", "--trajectories", "2", "--ppo_epochs", "2",
            "--minibatch_size", "4", "--value_probe_trajectories", "4", "--gpu_cache_gb", "0",
        ])

    def run_to_audit(self):
        experiment.prepare(self.args)
        experiment.export(self.args)
        experiment.audit(self.args)

    def test_full_paired_pipeline_and_small_report_and_no_retrain_on_resume(self):
        self.run_to_audit()
        experiment.train(self.args)
        experiment.collect(self.args)
        split = shared.load_json(self.work/"split.json")
        trained = [shared.load_json(self.work/"jobs"/f"{a}_seed0"/"training.json") for a in ARMS]
        self.assertEqual(len({r["initial_state_sha256"] for r in trained}), 1)
        self.assertEqual(len({r["refiner_state_sha256"] for r in trained}), 1)
        self.assertNotEqual(trained[0]["final_state_sha256"], trained[1]["final_state_sha256"])
        with zipfile.ZipFile(self.work/"ppo_report.zip") as archive:
            self.assertFalse(any(n.endswith((".pt", ".pth", ".npz")) for n in archive.namelist()))
            self.assertIn("results/value_probe.csv", archive.namelist())
            rows = list(csv.DictReader(io.StringIO(archive.read("results/step_trace.csv").decode())))
            for arm in ARMS:
                self.assertEqual({r["sample_id"] for r in rows if r["arm"] == arm}, set(split["calibration_ids"]+split["target_ids"]))
            self.assertEqual(len(rows), 2*5*3)
        before = [shared.sha256_file(self.work/"jobs"/f"{a}_seed0"/"final.pth") for a in ARMS]
        experiment.train(self.args)
        self.assertEqual(before, [shared.sha256_file(self.work/"jobs"/f"{a}_seed0"/"final.pth") for a in ARMS])
        payload = shared.load_torch(self.work/"jobs"/"V_seed0"/"final.pth")
        self.assertNotIn("mara_agent", payload)
        self.assertEqual(payload["mara_algorithm"], PROTOCOL)
        agent_from_payload(payload)

    def test_exact_epoch_resume_restores_actor_critic_optimizers_and_RNG(self):
        self.run_to_audit()
        root, cfg, manifest, signature = experiment.load_work(self.work)
        split = shared.load_json(root/"split.json")
        def store():
            return EvidenceStore(lambda r: experiment.gate.evidence(root, cfg, r, signature), "cpu", 1, 0)
        job = cfg["jobs"][0]
        save = experiment.save_resume
        def interrupt(*args, **kwargs):
            save(*args, **kwargs)
            if args[2] == 0:
                raise RuntimeError("simulated interruption")
        with patch.object(experiment, "save_resume", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "simulated"):
            experiment.train_job(root, cfg, manifest, signature, split, store(), job, torch.device("cpu"))
        _, resumed, _ = experiment.train_job(root, cfg, manifest, signature, split, store(), job, torch.device("cpu"))
        _, reference, _ = experiment.train_job(root/"reference", cfg, manifest, signature, split, store(), job, torch.device("cpu"))
        self.assertEqual(state_hash(resumed.state_dict()), state_hash(reference.state_dict()))

    def test_source_reuse_and_partial_collection_and_config_changes(self):
        self.run_to_audit()
        with self.assertRaises((ValueError, FileNotFoundError)):
            experiment.collect(self.args)
        self.args.epochs = 3
        with self.assertRaisesRegex(ValueError, "different config"):
            experiment.prepare(self.args)
        self.args.epochs = 2
        self.args.source_work_dir, self.args.work_dir = str(self.work), str(self.work.parent/"reused")
        experiment.prepare(self.args)
        with patch.object(sys.modules["test_mara"], "build_models_from_checkpoint", side_effect=AssertionError("no Base rebuild")):
            experiment.export(self.args)


class PPOShellTests(unittest.TestCase):
    def test_dual_GPU_barriers_and_failure(self):
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_ppo.sh")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        stages = [r.split("|")[0] for r in lines]
        self.assertEqual(stages, ["prepare", "export", "export", "audit", "train", "train", "collect"])
        self.assertIn("train|gpu=0|shard=0", lines)
        self.assertIn("train|gpu=1|shard=1", lines)
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_ppo.sh", FAIL_EXPORT_SHARD="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(r.startswith(("train", "collect")) for r in lines))


if __name__ == "__main__":
    unittest.main()
