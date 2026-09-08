"""Exercise shell routing with stub pipelines; never start a GPU workload."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch


BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH to a native Bash executable on Windows")
class ExperimentEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(ROOT / "run_exp.sh", self.root / "run_exp.sh")
        # Echo the dispatch and forwarded settings instead of running training.
        for name, marker in (
            ("train_mara_visa.sh", "DUAL"),
            ("run_exp_dino_single.sh", "SINGLE"),
        ):
            (self.root / name).write_text(
                '#!/usr/bin/env bash\n'
                f'echo "ROUTE={marker} DATASET=${{DATASET:-}} HFA=${{HFA_SETTING:-}} '
                'GPUS=${CUDA_VISIBLE_DEVICES:-} LATEST=${TEST_EVAL_LATEST_ONLY:-} '
                'VLM=${RUN_VLM_CACHE:-} ARGS=$*"\n', encoding="utf-8"
            )

    def run_entry(self, **settings):
        env = os.environ.copy()
        for key in (
            "BASE_ARCH", "SOURCE_DATASET", "DATASET", "HFA_SETTING", "GPU_IDS",
            "NPROC_PER_NODE", "DEVICE", "RUN_BASE", "RUN_MARA", "RUN_TEST",
            "RUN_VLM_CACHE", "RUN_VLM_DISTILL", "BASE_CKPT", "TEST_EVAL_LATEST_ONLY",
        ):
            env.pop(key, None)
        # Ensure the checkpoint guard uses the same Python/torch as these tests.
        env["PATH"] = os.pathsep.join((
            str(Path(sys.executable).parent), str(Path(BASH).parent), env.get("PATH", ""),
        ))
        env.update(settings)
        return subprocess.run(
            [BASH, "run_exp.sh", "sentinel"], cwd=self.root, env=env,
            capture_output=True, text=True, timeout=60,
        )

    def test_default_restores_dual_tower(self):
        result = self.run_entry()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ROUTE=DUAL DATASET=visa HFA=hfa3 GPUS=0,1 LATEST=1", result.stdout)
        self.assertIn("ARGS=sentinel", result.stdout)

    def test_explicit_single_preserves_vlm_route(self):
        result = self.run_entry(BASE_ARCH="dino_single", RUN_VLM_CACHE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ROUTE=SINGLE", result.stdout)
        self.assertIn("VLM=1 ARGS=sentinel", result.stdout)

    def test_dual_rejects_unsupported_vlm_stages(self):
        for stage in ("RUN_VLM_CACHE", "RUN_VLM_DISTILL"):
            with self.subTest(stage=stage):
                result = self.run_entry(**{stage: "1"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("supports only dino_single", result.stderr)
                self.assertNotIn("ROUTE=", result.stdout)

    def test_custom_settings_reach_dual_pipeline(self):
        result = self.run_entry(
            SOURCE_DATASET="mvtec", HFA_SETTING="hfa2", GPU_IDS="2, 3",
            TEST_EVAL_LATEST_ONLY="0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DATASET=mvtec HFA=hfa2 GPUS=2,3 LATEST=0", result.stdout)

    def test_bad_arch_gpu_count_and_test_only_fail_before_dispatch(self):
        for settings in (
            {"BASE_ARCH": "typo"}, {"GPU_IDS": "0"},
            {"NPROC_PER_NODE": "abc"}, {"RUN_MARA": "0", "RUN_TEST": "1"},
        ):
            with self.subTest(settings=settings):
                result = self.run_entry(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("ROUTE=", result.stdout)

    def test_resume_accepts_legacy_and_explicit_dual_metadata(self):
        for payload in ({}, {"base_arch": "clip_dino"}):
            with self.subTest(payload=payload):
                path = self.root / "base.pth"
                torch.save(payload, path)
                result = self.run_entry(RUN_BASE="0", BASE_CKPT=path.as_posix())
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Verified dual-tower checkpoint", result.stdout)
                self.assertIn("ROUTE=DUAL", result.stdout)

    def test_resume_rejects_single_checkpoint_even_if_renamed(self):
        for payload in ({"base_arch": "dino_single"}, {"dino_single_config": {}}):
            with self.subTest(payload=payload):
                path = self.root / "looks_like_dual.pth"
                torch.save(payload, path)
                result = self.run_entry(RUN_BASE="0", BASE_CKPT=path.as_posix())
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Expected a CLIP+DINO checkpoint", result.stderr)
                self.assertNotIn("ROUTE=", result.stdout)

    def test_resume_rejects_missing_checkpoint(self):
        result = self.run_entry(RUN_BASE="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("existing dual-tower BASE_CKPT", result.stderr)


if __name__ == "__main__":
    unittest.main()
