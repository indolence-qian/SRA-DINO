import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import matplotlib.pyplot
import numpy as np
from PIL import Image
import torch

import mara_diagnose as shared
import mara_ppo_experiment as experiment
from tools.mara_ppo import PROTOCOL, optimizers, update
from tools.mara_ppo_monitor import (
    EpochTelemetry, diagnostic_context, monitoring_selection, refresh_job,
    refresh_overview, package_process, read_csv,
)
from tools.mara_reward import state_hash
from test_mara_ppo import toy


class MonitoringTests(unittest.TestCase):
    def test_context_preserves_all_RNG_modes_weights_even_on_error(self):
        agent, _, _, _ = toy()
        agent.train()
        agent.actor.layer.eval()  # Mixed submodule modes must be restored exactly.
        modes = [m.training for m in agent.modules()]
        weights = state_hash(agent.state_dict())
        python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "diagnostic"):
            with diagnostic_context(agent):
                random.random()
                np.random.rand()
                torch.rand(3)
                self.assertFalse(agent.actor.training)
                raise RuntimeError("diagnostic")
        self.assertEqual(python, random.getstate())
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in agent.modules()])
        self.assertEqual(weights, state_hash(agent.state_dict()))

    def test_update_telemetry_does_not_change_parameters_or_sampling_RNG(self):
        agent, inputs, mask, labels = toy()
        batch, _ = agent.collect(inputs, mask, labels)
        reference = copy.deepcopy(agent)
        logged = []
        torch.manual_seed(543)
        expected = update(reference, batch, *optimizers(reference))
        rng = torch.get_rng_state().clone()
        torch.manual_seed(543)
        actual = update(agent, batch, *optimizers(agent), diagnostic_callback=logged.append)
        self.assertEqual(expected, actual)
        self.assertEqual(len(logged), expected["optimizer_steps"])
        self.assertEqual(state_hash(agent.state_dict()), state_hash(reference.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_fixed_source_selection_balanced_spread_and_no_target(self):
        manifest, rows = [], []
        for category in ("a", "b", "c", "d"):
            for label in (0, 1):
                for n in range(6):
                    identity = f"{category}_{label}_{n}"
                    manifest.append(dict(id=identity, category=category, dataset="visa"))
                    rows.append(dict(sample_id=identity, image_label=label, included=True))
        split = dict(source="visa", calibration_ids=[r["id"] for r in manifest])
        selected, previews = monitoring_selection(manifest, split, dict(rows=rows), 8, 8)
        self.assertEqual((len(selected), len(previews)), (32, 8))
        lookup = {r["id"]: r for r in selected}
        self.assertEqual({lookup[i]["category"] for i in previews}, {"a", "b", "c", "d"})
        self.assertEqual((selected, previews), monitoring_selection(list(reversed(manifest)), split, dict(rows=rows), 8, 8))
        manifest[0]["dataset"] = "mvtec"
        with self.assertRaisesRegex(ValueError, "Target forbidden"):
            monitoring_selection(manifest, split, dict(rows=rows))

    def test_only_committed_actual_batches_are_plotted_and_packaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = dict(id="V_seed0", arm="V", seed=0)
            directory = root/"jobs"/job["id"]
            telemetry = EpochTelemetry(directory, job, 1)
            telemetry.record("batch", dict(batch=1, loss=.3, return_mean=.1))
            telemetry.close()
            telemetry.commit()
            failed = EpochTelemetry(directory, job, 2)
            failed.record("batch", dict(batch=1, loss=999.))
            failed.close()  # No commit -> never count an interrupted epoch.
            history = [dict(job=job["id"], epoch=1, loss=.3, return_mean=.1)]
            shared.write_csv(directory/"training.csv", history)
            refresh_job(directory)
            rows = read_csv(directory/"monitor"/"batch_metrics.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["loss"]), .3)
            png = directory/"monitor"/"curves"/"epoch_training.png"
            with Image.open(png) as image:
                image.verify()
            cfg = dict(jobs=[job])
            refresh_overview(root, cfg)
            (directory/"monitor"/"unexpected_weights.pth").write_bytes(b"not a visual artifact")
            (directory/"monitor"/"partial.tmp.png").write_bytes(b"in-progress image")
            with zipfile.ZipFile(package_process(root, cfg)) as archive:
                self.assertIn("visual_report/index.html", archive.namelist())
                self.assertFalse(any("live_epoch" in n or "/epochs/" in n for n in archive.namelist()))
                self.assertIn("jobs/V_seed0/monitor/batch_metrics.csv", archive.namelist())
                self.assertFalse(any(".tmp" in n or n.endswith(".pth") for n in archive.namelist()))

    def test_old_report_can_plot_without_checkpoints_GT_or_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = dict(protocol=PROTOCOL, manifest_sha256=shared.digest([]), num_shards=2,
                       mara_ckpt="/does/not/exist.pth", jobs=[dict(id="V_seed0", arm="V", seed=0)],
                       mara_config=dict(max_steps=3))
            shared.atomic_json(root/"config.json", cfg)
            shared.atomic_json(root/"manifest.json", [])
            shared.write_csv(root/"results"/"training.csv", [dict(job="V_seed0", epoch=1, loss=.2, value_loss=.01)])
            args = experiment.parser().parse_args(["--stage", "visualize", "--work_dir", str(root), "--curves_only", "--device", "cpu"])
            with patch.object(experiment, "evidence_store", side_effect=AssertionError("no cache loading")):
                experiment.visualize(args)
            self.assertTrue((root/"ppo_process_report.zip").exists())
            self.assertEqual(read_csv(root/"jobs"/"V_seed0"/"monitor"/"batch_metrics.csv"), [])
            self.assertIn("Not recorded", (root/"jobs"/"V_seed0"/"monitor"/"index.html").read_text())


if __name__ == "__main__":
    unittest.main()
