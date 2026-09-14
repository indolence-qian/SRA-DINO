import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASH=os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name!="nt" else None)
ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH,"Set SRA_TEST_BASH on Windows")
class RepairEntrypointTests(unittest.TestCase):
    def run_script(self,fail=""):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            shutil.copyfile(ROOT/"run_exp_vlm_repair.sh",root/"run.sh")
            (root/"run_exp_dual_vlm_local.sh").write_text(
                '#!/usr/bin/env bash\n'
                'echo "ARM=$WORK_DIR PARSER=$LOCAL_PARSER CONTEXT=$LOCAL_CONTEXT_FACTOR MIN=$LOCAL_CONTEXT_MINIMUM REF=$NORMAL_REFERENCE SUPPRESS=$INCLUDE_SUPPRESSION"\n'
                '[[ "$WORK_DIR" != *"${FAIL_ARM:-NEVER}" ]]\n',encoding="utf-8",newline="\n")
            env=os.environ.copy()
            env.update(MODE="ablation",WORK_DIR="./out",FAIL_ARM=fail or "NEVER")
            env["PATH"]=str(Path(BASH).parent)+os.pathsep+env.get("PATH","")
            return subprocess.run([BASH,"run.sh"],cwd=root,env=env,text=True,capture_output=True,timeout=20)

    def test_separate_arms_and_suppression_disabled(self):
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("R0_repaired PARSER=repair_v2 CONTEXT=4 MIN=32 REF=0 SUPPRESS=0",result.stdout)
        self.assertIn("R1_context PARSER=repair_v2 CONTEXT=8 MIN=64 REF=0 SUPPRESS=0",result.stdout)
        self.assertIn("R2_context_reference PARSER=repair_v2 CONTEXT=8 MIN=64 REF=1 SUPPRESS=0",result.stdout)

    def test_failure_stops_next_arm(self):
        result=self.run_script("R1_context")
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn("R2_context_reference",result.stdout)
