"""Run the diagnostic shell with stub runtimes; never loads GPUs or weights."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH to native Bash on Windows")
class DiagnosticEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(ROOT / "run_exp_vlm_diagnose.sh", self.root / "run.sh")
        (self.root / "source").mkdir()
        for name in ("manifest.json", "config.json"):
            (self.root / "source" / name).write_text("{}")
        runtime = self.root / "runtime"
        runtime.write_text(
            '#!/usr/bin/env bash\nset -eu\n'
            '[[ "${1:-}" != -c ]] || { echo PREFLIGHT; exit 0; }\n'
            'stage=""; variant=""\nfor arg in "$@"; do\n'
            '  if [[ "${previous:-}" == --stage ]]; then stage="$arg"; fi\n'
            '  if [[ "${previous:-}" == --variant ]]; then variant="$arg"; fi\n'
            '  previous="$arg"\ndone\n'
            'echo "STAGE=$stage VARIANT=$variant GPU=${CUDA_VISIBLE_DEVICES:-} ARGS=$*"\n'
            'if [[ "$stage" == review && "${FAIL_GPU:-none}" == "${CUDA_VISIBLE_DEVICES:-}" ]]; then exit 7; fi\n',
            encoding="utf-8", newline="\n")
        runtime.chmod(0o755)

    def run_script(self, **overrides):
        env = os.environ.copy()
        for k in ("NPROC_PER_NODE", "RUN_REVIEW", "RUN_SANITY", "RUN_EVAL", "VLM_MODEL_ID", "FAIL_GPU"):
            env.pop(k, None)
        env.update(SOURCE_WORK_DIR="./source", WORK_DIR="./diagnosis", TRAIN_PYTHON="./runtime",
                   VLM_PYTHON="./runtime", GPU_IDS="0,1", DIAG_CANDIDATES="96")
        env["PATH"] = str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        return subprocess.run([BASH, "run.sh"], cwd=self.root, env=env, capture_output=True, text=True, timeout=30)

    def test_fast_cached_workers_and_stage_order(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        out = result.stdout
        self.assertEqual(out.count("STAGE=review"), 8)
        for arm in ("A_original", "B_prompt", "C_pixels", "D_prompt_pixels"):
            for gpu in (0,1):
                self.assertIn(f"STAGE=review VARIANT={arm} GPU={gpu}", out)
        self.assertLess(out.index("STAGE=prepare"), out.index("STAGE=review"))
        self.assertLess(out.rindex("STAGE=review"), out.index("STAGE=sanity"))
        self.assertLess(out.rindex("STAGE=sanity"), out.index("STAGE=evaluate"))
        self.assertNotIn("STAGE=export", out)

    def test_failure_stops_later_arms(self):
        result = self.run_script(FAIL_GPU="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("VARIANT=B_prompt", result.stdout)
        self.assertNotIn("STAGE=evaluate", result.stdout)

    def test_cpu_only_resume(self):
        result = self.run_script(RUN_REVIEW="0", RUN_SANITY="0")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertNotIn("STAGE=review", result.stdout)
        self.assertIn("STAGE=evaluate", result.stdout)

    def test_bad_gpu_and_missing_source_fail_early(self):
        for args in ({"GPU_IDS":"0,0"}, {"NPROC_PER_NODE":"3"}, {"SOURCE_WORK_DIR":"./absent"}):
            result = self.run_script(**args)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("STAGE=", result.stdout)


if __name__ == "__main__":
    unittest.main()
