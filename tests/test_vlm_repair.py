import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tools.local_vlm_protocol import parse_repaired, repaired_prompt
from tools.local_vlm_review import action, parse_local, native_crops
from tools.local_vlm_pipeline import structure_similarity
from tools.vlm_review import load_json, atomic_json, digest, file_digest
import vlm_diagnose as diag
import reparse_vlm_diagnosis as replay
import test_vlm_diagnose as fixtures
import test_local_vlm as local_fixtures
import dual_vlm


def vote(**changes):
    return dict(candidate_id=1, verdict="insufficient_evidence", visibility="sufficient",
                reference_match="unavailable", defect_type="unknown", evidence="Visible but ambiguous") | changes


class ParserRepairTests(unittest.TestCase):
    def test_empty_unknown_type_is_warning_not_new_action(self):
        raw=json.dumps(vote(defect_type=""))
        d,a=parse_repaired(raw,1)
        self.assertTrue(d["parse_ok"])
        self.assertEqual(d["defect_type"],"unknown")
        self.assertEqual(action(d),"keep")
        self.assertIn("empty_defect_type_normalized",a["parse_warnings"])
        self.assertFalse(parse_local(raw,1)["parse_ok"])

    def test_long_explanation_preserved_with_defect_decision(self):
        raw=json.dumps(vote(verdict="defect_supported",defect_type="solder blob",evidence="x"*563))
        d,a=parse_repaired(raw,1)
        self.assertEqual(action(d),"enhance")
        self.assertEqual(len(d["evidence"]),563)
        self.assertIn("evidence_over_400_preserved",a["parse_warnings"])

    def test_strict_core_and_evidence_guards(self):
        for changes in ({"candidate_id":True},{"candidate_id":2},{"verdict":"bad"},
                        {"visibility":"maybe"},{"reference_match":"matched"},
                        {"evidence":""},{"defect_type":None},
                        {"verdict":"defect_supported","defect_type":""},{"extra":1}):
            with self.subTest(changes=changes):
                d,a=parse_repaired(json.dumps(vote(**changes)),1)
                self.assertFalse(d["parse_ok"])
                self.assertTrue(a["parse_errors"])
        d,a=parse_repaired('{"candidate_id":1,"candidate_id":2}',1)
        self.assertIn("duplicate_key",a["parse_errors"])
        self.assertFalse(parse_repaired('[]',1)[0]["parse_ok"])
        self.assertFalse(parse_repaired('x'*65537,1)[0]["parse_ok"])

    def test_prompt_does_not_force_visibility_or_defect(self):
        p=repaired_prompt("pcb1",1)
        self.assertIn("out of focus",p)
        self.assertIn("Do not force",p)
        self.assertNotIn('"verdict":"insufficient_evidence"',p)

    def test_context_change_preserves_detail(self):
        im=Image.fromarray(np.random.default_rng(42).integers(0,256,(800,1200,3),dtype=np.uint8))
        a,g=native_crops(im,[200,200,208,208],(512,512))
        b,h=native_crops(im,[200,200,208,208],(512,512),context_factor=8,context_minimum=64)
        np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(g["detail_box"],h["detail_box"])
        self.assertGreater(b[0].width,a[0].width)

    def test_structure_prefilter_rejects_flat_reference(self):
        image=Image.fromarray(np.random.default_rng(1).integers(0,256,(32,32),dtype=np.uint8))
        self.assertAlmostEqual(structure_similarity(image,image),1,places=5)
        self.assertEqual(structure_similarity(image,Image.new("RGB",(32,32))),0)


class OfflineReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DiagnosisTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_replay_no_engine_last_answer_only_source_unchanged(self):
        f=self.fixture
        f.prepare_and_review()
        oldcfg=load_json(f.out/"diagnosis_config.json")
        # Simulate old code fingerprints: only the explicit migration may load them.
        oldcfg["code_hashes"]={"old_revision":"legacy"}
        atomic_json(f.out/"diagnosis_config.json",oldcfg)
        for p in f.out.glob('*/*.json'):
            data=load_json(p)
            data["fingerprint"]=digest(oldcfg)
            if p.parent.name=="B_prompt":
                raw=json.dumps(vote(candidate_id=data["decision"]["candidate_id"],defect_type=""))
                data["raw_responses"]=[json.dumps(vote(verdict="defect_supported",defect_type="crack")),raw]
                data["decision"]=parse_local(raw,data["decision"]["candidate_id"])
                trace_path=p.parent/data["final_trace"]
                trace=load_json(trace_path)
                trace["raw_response"]=raw
                atomic_json(trace_path,trace)
                data["trace_hashes"][data["final_trace"]]=file_digest(trace_path)
            atomic_json(p,data)
        snapshots={str(p):p.read_bytes() for p in f.out.rglob('*') if p.is_file()}
        target=f.root/"replayed"
        with patch('tools.vlm_decision.QwenVLLMTeacher',side_effect=AssertionError("Must not call VLM")):
            replay.run(f.out,target)
        self.assertTrue((target/"results/summary.json").exists())
        p=next((target/"B_prompt").glob('*.json'))
        d=load_json(p)
        self.assertTrue(d["decision"]["parse_ok"])
        self.assertEqual(action(d["decision"]),"keep")
        self.assertTrue(d["parse_warnings"])
        for name,content in snapshots.items():
            self.assertEqual(Path(name).read_bytes(),content)

    def test_migration_rejects_overlapping_directories(self):
        f=self.fixture
        with self.assertRaisesRegex(ValueError,"separate"):
            replay.run(f.out,f.out/"child")


class RepairedLocalPipelineTests(unittest.TestCase):
    def test_policy_is_fingerprinted_and_long_reason_reaches_evaluation(self):
        f=local_fixtures.LocalPipelineTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.args.local_parser="repair_v2"
        f.args.local_prompt="repaired"
        f.args.local_context_factor=8
        f.args.local_teacher_min_pixels=65536
        cfg,records=f.make_manifest()
        self.assertEqual(cfg["local_parser"],"repair_v2")
        self.assertTrue(cfg["local_code_hashes"])
        with patch("tools.vlm_decision.QwenVLLMTeacher") as teacher:
            teacher.return_value.generate.return_value=json.dumps(vote(verdict="defect_supported",defect_type="crack",evidence="x"*500))
            for shard in (0,1):
                f.args.shard_id=shard
                dual_vlm.review(f.args)
            self.assertEqual(teacher.call_args.kwargs["min_pixels"],65536)
            self.assertIn("out of focus",teacher.return_value.generate.call_args.args[0])
        dual_vlm.evaluate(f.args)
        cached=load_json(f.work/"local_reviews/sample0_1.json")
        self.assertTrue(cached["decision"]["parse_ok"])
        self.assertIn("evidence_over_400_preserved",cached["parse_warnings"])
        self.assertIn("evidence_over_400_preserved",(f.work/"results/candidate_decisions.csv").read_text())
        with patch("dual_vlm.file_digest",return_value="different"):
            with self.assertRaisesRegex(ValueError,"code changed"):
                dual_vlm.config(f.args)


if __name__=="__main__":
    unittest.main()
