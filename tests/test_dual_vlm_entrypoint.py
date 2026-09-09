"""Shell integration tests use stub runtimes: no CUDA, no model download."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH to native Bash on Windows")
class DualVLMEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(ROOT / "run_exp_dual_vlm.sh", self.root / "run.sh")
        (self.root / "base.pth").touch()
        runtime = self.root / "runtime"
        runtime.write_text(
            '#!/usr/bin/env bash\nset -eu\n'
            '[[ "${1:-}" != -c ]] || { echo PREFLIGHT; exit 0; }\n'
            'stage=""\nfor arg in "$@"; do\n'
            '  if [[ "${previous:-}" == --stage ]]; then stage="$arg"; fi\n'
            '  previous="$arg"\ndone\n'
            'echo "STAGE=$stage GPU=${CUDA_VISIBLE_DEVICES:-} ARGS=$*"\n'
            '[[ "$stage" != "${FAIL_STAGE:-NEVER}" ]]\n',
            encoding="utf-8", newline="\n",
        )
        runtime.chmod(0o755)

    def run_script(self, **overrides):
        env = os.environ.copy()
        for key in ("NPROC_PER_NODE", "RUN_EXPORT", "RUN_REVIEW", "RUN_EVAL", "VLM_HEATMAP", "FAIL_STAGE"):
            env.pop(key, None)
        env.update(BASE_CKPT="./base.pth", WORK_DIR="./work", TRAIN_PYTHON="./runtime", VLM_PYTHON="./runtime",
                   VLM_MODEL_ID="./model", DATASETS="visa", MAX_PER_CATEGORY="40", GPU_IDS="0,1")
        env["PATH"] = str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        return subprocess.run([BASH, "run.sh"], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=30)

    def test_stage_order_and_gpu_mapping(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        out = result.stdout
        self.assertLess(out.index("STAGE=prepare"), out.index("STAGE=export"))
        self.assertLess(out.rindex("STAGE=export"), out.index("STAGE=seal"))
        self.assertLess(out.index("STAGE=seal"), out.index("STAGE=review"))
        self.assertLess(out.rindex("STAGE=review"), out.index("STAGE=evaluate"))
        for stage in ("export", "review"):
            for gpu in (0, 1):
                self.assertIn(f"STAGE={stage} GPU={gpu}", out)
        self.assertIn("STAGE=evaluate GPU= ARGS", out)

    def test_worker_failure_prevents_later_stages(self):
        result = self.run_script(FAIL_STAGE="export")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("STAGE=review", result.stdout)
        self.assertNotIn("STAGE=evaluate", result.stdout)

    def test_resume_can_skip_completed_export(self):
        result = self.run_script(RUN_EXPORT="0")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("STAGE=export", result.stdout)
        self.assertIn("STAGE=review", result.stdout)

    def test_bad_gpu_configuration_fails_before_work(self):
        for overrides in ({"GPU_IDS": "0,0"}, {"GPU_IDS": "foo"}, {"NPROC_PER_NODE": "3"}):
            result = self.run_script(**overrides)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("STAGE=", result.stdout)

    def test_missing_base_fails_early(self):
        result = self.run_script(BASE_CKPT="./missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BASE_CKPT", result.stderr)


if __name__ == "__main__":
    unittest.main()
