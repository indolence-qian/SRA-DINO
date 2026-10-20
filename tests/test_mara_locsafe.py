import copy
import csv
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

import mara_diagnose as shared
import mara_locsafe_experiment as experiment
from tools.mara_locsafe import (ARMS, PROTOCOL, LocSafeConfig, LocSafeAgent, RegionRank,
    nested_split, oracle_actions, search_candidates, tail_mean)
from tools.mara_ppo import agent_from_payload, optimizers, score_trajectory, update
from tools.mara_reward import state_hash
from test_mara_ppo import toy
import test_mara_diagnostics as fixtures
import test_mara_reward_experiment as reward_fixtures


def local_toy(arm="P0", device="cpu"):
    old, inputs, mask, labels = toy(device)
    agent = LocSafeAgent(old.cfg, old.ppo, LocSafeConfig(rank_points=16), arm).to(device)
    agent.load_state_dict(old.state_dict(), strict=False)
    return agent, inputs, mask, labels


def scheduled(agent, inputs, ops=(1, 0, 0)):
    op = torch.tensor([ops], dtype=torch.long, device=inputs["base_prob"].device)
    return agent.trajectory(inputs, sample=False, action_sequence=dict(op=op, region=torch.zeros_like(op), layer=torch.zeros_like(op)))


class LocSafeTests(unittest.TestCase):
    def test_P0_is_exact_original_reward_and_GT_free_execution(self):
        agent, inputs, mask, labels = local_toy()
        actual = scheduled(agent, inputs)
        expected = copy.deepcopy(actual)
        before = torch.get_rng_state().clone()
        agent.score(actual, mask, labels)
        score_trajectory(expected, mask, labels, agent.cfg, agent.ppo)
        for a, b in zip(actual["steps"], expected["steps"]):
            for key in ("reward", "return", "advantage", "quality_before", "quality_after"):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "GT"):
            agent.trajectory(dict(inputs, mask=mask))
        normal_gt = torch.zeros_like(mask)
        result = scheduled(agent, inputs)
        self.assertTrue(torch.equal(result["final_prob"], actual["final_prob"]))
        agent.score(result, normal_gt, 1-labels)
        self.assertTrue(torch.equal(result["final_prob"], actual["final_prob"]))

    def test_P1_removes_only_classification_from_reward(self):
        agent, inputs, mask, labels = local_toy("P1")
        result = scheduled(agent, inputs)
        # Synthetic classification-only improvement, with identical probability maps.
        step = result["steps"][0]
        step["after_hr"] = step["before_hr"].clone()
        step["after_logits"] = step["before_logits"]+torch.tensor([[-2., 2.]])
        agent.score(result, mask, labels)
        torch.testing.assert_close(step["reward"], torch.tensor([-.003]))
        self.assertNotEqual(float(step["reward"]), float(step["reference_reward"]))

    def test_ranking_equal_weights_components_and_uses_hard_background(self):
        settings = LocSafeConfig(rank_points=100)
        rank = RegionRank(settings)
        mask = torch.zeros(1, 1, 8, 8)
        mask[0, 0, 0, 0] = 1
        mask[0, 0, 4:7, 4:7] = 1
        prob = torch.full((1, 2, 8, 8), .2)
        prob[0, 1, 0, 0] = .1
        prob[0, 1, 4:7, 4:7] = .9
        before = torch.get_rng_state().clone()
        torch.testing.assert_close(rank(prob, mask, torch.ones(1)), torch.tensor([.1]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(float(rank(prob, mask, torch.zeros(1))), 0.)
        self.assertEqual(len(rank.cache), 1)
        prob[0, 1, 2, 2] = 1.
        self.assertGreater(float(rank(prob, mask, torch.ones(1))), .1)

    def test_terminal_risk_once_including_early_STOP_and_long_term_return(self):
        p3, inputs, mask, labels = local_toy("P3")
        p2 = copy.deepcopy(p3)
        p2.arm = "P2"
        for ops in ((1, 0, 0), (1, 1, 1), (0, 0, 0)):
            with self.subTest(ops=ops):
                a, b = scheduled(p3, inputs, ops), scheduled(p2, inputs, ops)
                mask, labels = torch.zeros_like(mask), torch.zeros_like(labels)
                p3.score(a, mask, labels)
                p2.score(b, mask, labels)
                costs = sum(s["risk_cost"] for s in a["steps"])
                expected = (tail_mean(a["final_prob"])-tail_mean(inputs["base_prob"])).clamp_min(0)
                torch.testing.assert_close(costs, expected)
                torch.testing.assert_close(a["steps"][0]["return"], b["steps"][0]["return"]-p3.dual.float()*costs)
                for s in a["steps"]:
                    if not bool(s["valid"] & s["done"]):
                        self.assertEqual(float(s["risk_cost"]), 0.)
                if ops == (1, 0, 0):
                    self.assertTrue(torch.equal(a["steps"][0]["reward"], b["steps"][0]["reward"]))

    def test_dual_next_epoch_update_is_conditional_risk_correct(self):
        agent, _, _, _ = local_toy("P3")
        agent.settings = replace(agent.settings, normal_fraction=.4)
        agent._normal_risks = [.01, .03]
        agent._normal_reference = [-2., 0.]
        row = agent.finish_epoch()
        self.assertAlmostEqual(row["dual_next"], 1+10*.4*(.02-.001))
        self.assertEqual(float(agent.dual), 1.)  # Held-out score matches trained lambda.
        self.assertEqual(row["normal_reference_catastrophes"], 1)
        self.assertEqual(row["normal_risk_max"], .03)
        agent.arm = "P2"
        self.assertEqual(agent.finish_epoch()["dual_next"], 1.)

    def test_actor_critic_learn_refiner_and_inputs_immutable_and_payload_roundtrip(self):
        agent, inputs, mask, labels = local_toy("P2")
        frozen = state_hash(agent.transition.state_dict())
        evidence = state_hash(inputs)
        initial = state_hash(agent.actor.state_dict())
        batch, summary = agent.collect(inputs, mask, labels)
        self.assertIn("reference_return_mean", summary)
        update(agent, batch, *optimizers(agent))
        self.assertNotEqual(initial, state_hash(agent.actor.state_dict()))
        self.assertEqual(frozen, state_hash(agent.transition.state_dict()))
        self.assertEqual(evidence, state_hash(inputs))
        payload = dict(mara_algorithm=PROTOCOL, mara_config=asdict(agent.cfg), ppo_config=asdict(agent.ppo),
                       locsafe_config=asdict(agent.settings), locsafe_arm=agent.arm, mara_ppo_agent=agent.state_dict())
        self.assertEqual(state_hash(agent_from_payload(payload).state_dict()), state_hash(agent.state_dict()))

    def test_candidate_pool_includes_STOP_and_never_restarts_done_paths(self):
        choices = oracle_actions([[]], 3, 16, 4)
        self.assertEqual(len(choices), 65)
        self.assertIn(((0, 0, 0),)*3, choices)
        completed = oracle_actions([[(1, 0, 0), (0, 0, 0)]], 3, 16, 4)
        self.assertEqual(completed, [((1, 0, 0), (0, 0, 0), (0, 0, 0))])
        agent, inputs, mask, labels = local_toy("P2")
        rows, prefixes, best, probability = search_candidates(agent, inputs, mask, labels, [[]], 1, 2, 3)
        self.assertEqual(len(rows), 1+agent.cfg.num_regions*agent.cfg.num_layers)
        self.assertGreaterEqual(best["proxy_gain"], -1e-6)
        rows, _, best, _ = search_candidates(agent, inputs, mask, labels, prefixes, 2, 2, 3)
        self.assertEqual(probability.shape, (16, 16))
        self.assertGreaterEqual(best["proxy_gain"], -1e-6)

    def test_nested_split_is_disjoint_stable_and_excludes_historical_and_target(self):
        manifest = [dict(id=str(i), category="a", dataset="visa" if i < 12 else "target") for i in range(14)]
        audit = dict(rows=[dict(sample_id=str(i), image_label=i % 2, included=True) for i in range(14)])
        historical = dict(source="visa", train_ids=[str(i) for i in range(10)], calibration_ids=["10", "11"], target_ids=["12", "13"])
        result = nested_split(manifest, audit, historical, 7)
        self.assertEqual(result, nested_split(list(reversed(manifest)), audit, historical, 7))
        self.assertEqual(len(result["train_ids"]), 8)
        self.assertEqual(len(result["calibration_ids"]), 2)
        self.assertFalse(set(result["calibration_ids"]) & {"10", "11", "12", "13"})

    def test_source_selection_guardrails_and_no_fake_Base_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = dict(epochs=30, selection_bg_tolerance=.0002, selection_auc_tolerance=.002)
            for epoch, pro, fp, auc in ((0, 1., 0., 1.), (1, .99, .1, .99), (5, .91, .001, .98), (10, .92, .001, .98), (15, .99, .001, .5)):
                shared.atomic_json(root/"monitor/validation"/f"epoch_{epoch:03d}.json", dict(epoch=epoch,
                    macro=dict(PRO=pro, base_PRO=.9, background_FPR_at_05=fp, base_background_FPR_at_05=.001,
                               P_AUROC=auc, base_P_AUROC=.98)))
            result = experiment.select_epoch(root, cfg)
            self.assertEqual(result["epoch"], 10)
            cfg["selection_bg_tolerance"] = 0.
            for path in (root/"monitor/validation").glob("*.json"):
                value = shared.load_json(path)
                value["macro"]["PRO"] = .8
                shared.atomic_json(path, value)
            self.assertEqual(experiment.select_epoch(root, cfg)["status"], "FAILED_guardrails_FINAL_diagnostic_only")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_29_channel_128_grid_512_HR_CUDA_reward_and_update(self):
        from tools.mara_agent import MARAAgent, MARAConfig
        from tools.mara_ppo import PPOConfig, environment_config
        cfg = environment_config(MARAConfig(num_layers=4, evidence_channels=18, global_evidence_dim=4))
        refiner = MARAAgent(cfg)
        agent = LocSafeAgent(cfg, PPOConfig(trajectories=1, update_epochs=1, minibatch_size=1), LocSafeConfig(), "P3").cuda()
        agent.initialize_refiner(refiner.state_dict())
        inputs = dict(base_prob=torch.rand(1, 2, 512, 512, device="cuda").softmax(1),
                      base_logits=torch.zeros(1, 2, device="cuda"), layer_maps=torch.rand(1, 4, 128, 128, device="cuda"),
                      evidence_maps=torch.rand(1, 18, 128, 128, device="cuda"), global_evidence=torch.rand(1, 4, device="cuda"))
        mask = torch.zeros(1, 1, 512, 512, device="cuda")
        mask[:, :, 10:12, 10:12] = 1
        mask[:, :, 100:120, 100:120] = 1
        batch, _ = agent.collect(inputs, mask, torch.ones(1, dtype=torch.long, device="cuda"))
        frozen = state_hash(agent.transition.state_dict())
        update(agent, batch, *optimizers(agent))
        self.assertEqual(state_hash(agent.transition.state_dict()), frozen)


class LocSafePipelineTests(unittest.TestCase):
    def setUp(self):
        reward_fixtures.RewardPipelineTests.setUp(self)
        self.args = experiment.parser().parse_args([
            "--stage", "prepare", "--work_dir", str(self.work), "--mara_ckpt", str(self.ckpt),
            "--datasets", "visa toy", "--samples_per_category", "4", "--num_shards", "1",
            "--image_size", "16", "--pro_num_th", "21", "--device", "cpu", "--metric_workers", "1",
            "--epochs", "1", "--train_batch_size", "2", "--trajectories", "2", "--ppo_epochs", "1",
            "--minibatch_size", "4", "--value_probe_trajectories", "4", "--gpu_cache_gb", "0",
            "--beam_per_category", "2", "--beam_width", "2", "--oracle_per_category", "2", "--preview_samples", "0",
        ])

    def run_to_audit(self):
        experiment.prepare(self.args)
        experiment.export(self.args)
        experiment.audit(self.args)

    def test_full_pipeline_population_separation_and_light_report_and_resume(self):
        self.run_to_audit()
        with self.assertRaises(FileNotFoundError):
            experiment.train(self.args)  # Cannot bypass O.
        with self.assertRaises(FileNotFoundError):
            experiment.evaluate(self.args)  # No targets before all training.
        experiment.oracle(self.args)
        experiment.oracle_report(self.args)
        with patch.object(experiment, "refresh_job"):
            experiment.train(self.args)
        experiment.evaluate(self.args)
        experiment.collect(self.args)
        split = shared.load_json(self.work/"split.json")
        self.assertEqual((len(split["train_ids"]), len(split["calibration_ids"]), len(split["historical_calibration_ids"])), (4, 2, 2))
        trained = [shared.load_json(self.work/"jobs"/f"{a}_seed0/training.json") for a in ARMS]
        self.assertEqual(len({v["initial_state_sha256"] for v in trained}), 1)
        self.assertEqual(len({v["refiner_state_sha256"] for v in trained}), 1)
        hashes = [shared.sha256_file(self.work/"jobs"/f"{a}_seed0/selected.pth") for a in ARMS]
        with patch.object(experiment.legacy, "train_epoch", side_effect=AssertionError("no retraining")):
            experiment.train(self.args)
        self.assertEqual(hashes, [shared.sha256_file(self.work/"jobs"/f"{a}_seed0/selected.pth") for a in ARMS])
        with zipfile.ZipFile(self.work/"locsafe_report.zip") as archive:
            names = archive.namelist()
            self.assertFalse(any(n.endswith((".pt", ".pth", ".npz")) for n in names))
            self.assertIn("oracle/index.html", names)
            self.assertIn("visual_report/index.html", names)
            self.assertIn("results/risk_summary.csv", names)
            self.assertIn("results/value_calibration_summary.csv", names)
            self.assertIn("visual_report/value_calibration_by_label.png", names)
            self.assertTrue(any(n.startswith("oracle/previews/") and n.endswith(".png") for n in names))
            rows = list(csv.DictReader(io.StringIO(archive.read("results/metrics.csv").decode())))
            self.assertEqual({r["role"] for r in rows}, {"source_dev", "source_historical", "target_full", "target_additional"})
            self.assertEqual({r["arm"] for r in rows}, {"Base", *ARMS})
            trace = list(csv.DictReader(io.StringIO(archive.read("results/step_trace.csv").decode())))
            self.assertTrue(all("reference_return" in r and "risk_cost" in r for r in trace))

    def test_exact_P3_resume_restores_dual_optimizer_and_RNG(self):
        self.args.epochs = 2
        self.run_to_audit()
        root, cfg, manifest, signature = experiment.load_work(self.work)
        split, audited = experiment.checked_split(root, signature)
        store = experiment.legacy.evidence_store(root, cfg, signature, audited, "cpu", 0)
        job = cfg["jobs"][-1]
        original = experiment.save_resume
        def stop(*values):
            original(*values)
            if values[2] == 0:
                raise RuntimeError("interrupted after committed weights")
        with patch.object(experiment, "save_resume", side_effect=stop), self.assertRaisesRegex(RuntimeError, "interrupted"):
            experiment.train_job(root, cfg, manifest, signature, split, audited, store, job, torch.device("cpu"))
        experiment.train_job(root, cfg, manifest, signature, split, audited, store, job, torch.device("cpu"))
        reference = root/"reference"
        experiment.train_job(reference, cfg, manifest, signature, split, audited, store, job, torch.device("cpu"))
        a = shared.load_torch(root/"jobs"/job["id"]/"final.pth")
        b = shared.load_torch(reference/"jobs"/job["id"]/"final.pth")
        self.assertEqual(state_hash(a["mara_ppo_agent"]), state_hash(b["mara_ppo_agent"]))


class LocSafeShellTests(unittest.TestCase):
    def test_two_GPU_barriers_and_failure(self):
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_locsafe.sh")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual([r.split("|")[0] for r in lines], ["prepare", "export", "export", "audit", "oracle", "oracle", "oracle_report", "train", "train", "evaluate", "evaluate", "collect"])
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_locsafe.sh", FAIL_EXPORT_SHARD="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(r.startswith(("oracle", "train", "evaluate", "collect")) for r in lines))
        result, lines = fixtures.EntrypointTests.run_mock_shell(self, script_name="run_exp_mara_locsafe.sh", ORACLE_ONLY="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(r.startswith(("train", "evaluate", "collect")) for r in lines))


if __name__ == "__main__":
    unittest.main()
