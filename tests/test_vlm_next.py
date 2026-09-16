import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import dual_vlm
import vlm_next_diagnose as diag
from tools.vlm_next_diagnostics import (ARMS, audit_prompt, parse_audit, correction, exact_pro, paired_selection)
from tools.vlm_review import atomic_json, digest, file_digest, load_json


def response(reference=False):
    return dict(candidate_id=1,verdict='defect_supported',visibility='sufficient',
                reference_match='unmatched' if reference else 'unavailable',defect_type='crack',
                evidence='A discontinuity is visible.',query_observation='A thin dark line.',
                reference_observation='Smooth surface.' if reference else 'not_provided',
                local_difference='Different local structure.' if reference else 'No reference comparison.')


class Teacher:
    calls=[]
    def __init__(self,**kwargs):
        self.kwargs=kwargs
    def generate(self,prompt,images,trace_dir=None):
        self.calls.append((prompt,[np.asarray(im).copy() for im in images]))
        raw=json.dumps(response(len(images)==3))
        trace_dir.mkdir(parents=True,exist_ok=True)
        for i,im in enumerate(images):
            im.save(trace_dir/f'post_vision_{i}.png')
        atomic_json(trace_dir/'trace.json',dict(raw_response=raw,input_sizes=[list(im.size) for im in images]))
        return raw


class HelperTests(unittest.TestCase):
    def test_protocol_valid_and_duplicate_rejected(self):
        d,a=parse_audit(json.dumps(response()),1,False)
        self.assertTrue(d['parse_ok'])
        self.assertFalse(a['semantic_warnings'])
        raw=json.dumps(response())[:-1]+',"candidate_id":2}'
        self.assertFalse(parse_audit(raw,1,False)[0]['parse_ok'])
        for update in ({'query_observation':''},{'candidate_id':True},{'extra':1},{'visibility':'maybe'}):
            self.assertFalse(parse_audit(json.dumps(response()|update),1,False)[0]['parse_ok'])

    def test_semantic_warnings_preserve_decision(self):
        obj=response(True)|dict(reference_match='unavailable',reference_observation='No reference provided.',
                                query_observation='No discernible structure; out of focus.')
        d,a=parse_audit(json.dumps(obj),1,True)
        self.assertTrue(d['parse_ok'])
        self.assertIn('denies_provided_reference',a['semantic_warnings'])
        self.assertIn('visibility_text_conflict',a['semantic_warnings'])

    def test_reference_arm_prompts_identical_and_no_gt_fields(self):
        p=audit_prompt('pcb',1,True)
        self.assertIn('Image 3: NORMAL TRAIN REFERENCE DETAIL',p)
        self.assertIn('NOT guaranteed',p)
        self.assertNotIn('shuffled',p)
        self.assertNotIn('gt_',p)

    def test_correction_boundaries_and_oracle_distinctions(self):
        base=np.full((4,4),.45,np.float32)
        s=np.zeros((2,4,4),bool)
        s[0,0,:2]=True
        s[1,3,:2]=True
        gt=np.zeros((4,4),bool)
        gt[0,0]=True
        votes=[dict(parse_ok=True,verdict='insufficient_evidence',visibility='sufficient')]*2
        candidate=correction(base,s,votes,gt,'gt_candidate_DIAGNOSTIC_ONLY',.25)
        pixel=correction(base,s,votes,gt,'gt_pixel_DIAGNOSTIC_ONLY',.25)
        all_map=correction(base,s,votes,gt,'all_candidates',.25)
        self.assertEqual(int((candidate!=base).sum()),2)
        self.assertEqual(int((pixel!=base).sum()),1)
        self.assertEqual(int((all_map!=base).sum()),4)
        np.testing.assert_array_equal(base,correction(base,s,votes,gt,'vlm_cached',.25))
        for m in (candidate,pixel,all_map):
            self.assertTrue(np.all(m>=base))
            np.testing.assert_array_equal(m[~s.any(axis=0)],base[~s.any(axis=0)])
        empty=np.zeros((0,4,4),bool)
        np.testing.assert_array_equal(correction(base,empty,[],gt,'gt_pixel_DIAGNOSTIC_ONLY',.25),base)

    def test_exact_pro_perfect_reversed_ties(self):
        gt=np.array([[[1,0],[0,0]]],bool)
        self.assertAlmostEqual(exact_pro(gt,gt.astype(float)),1.)
        self.assertAlmostEqual(exact_pro(gt,1-gt.astype(float)),0.)
        self.assertAlmostEqual(exact_pro(gt,np.ones_like(gt,float)),.15)
        self.assertIsNone(exact_pro(np.zeros_like(gt),np.ones_like(gt,float)))

    def test_exact_pro_vertical_segment_not_diagonal(self):
        # One negative before a single positive: correct partial area is 2/3,
        # NOT 5/6 from collapsing duplicate FPR coordinates into a diagonal.
        gt=np.array([[[1,0,0,0]]],bool)
        scores=np.array([[[.8,.9,.2,.1]]])
        self.assertAlmostEqual(exact_pro(gt,scores,1.),2/3)

    def test_exact_pro_equal_region_weighting(self):
        gt=np.array([[[1,0,1,1]]],bool)
        scores=np.array([[[.9,.8,.1,.1]]])
        self.assertAlmostEqual(exact_pro(gt,scores,.3),.5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.src,self.out,self.model=[self.root/x for x in ('source','out','model')]
        self.model.mkdir()
        atomic_json(self.model/'config.json',{})
        (self.model/'model.safetensors').write_bytes(b'fake-not-a-model')
        self.original=dict(local_review=True,teacher_image_size=512,local_parser='repair_v2',
                           model=dict(path=str(self.model)),pro_max_fpr=.3,
                           local_code_hashes={'historic_revision':'allowed-read-only'})
        atomic_json(self.src/'config.json',self.original)
        records=[]
        for i in range(3):
            r=dict(id=f's{i}',dataset='visa',category='object',index=i,fingerprint=digest(self.original),
                   evidence=f'evidence/s{i}.npz',supports=f'supports/s{i}.npz',evaluation=f'evaluation_only/s{i}.npz')
            base=np.full((8,8),.2,np.float32)
            base[2:4,2:4]=.45
            supports=np.zeros((1,8,8),bool)
            supports[0,2:4,2:4]=True
            gt=np.zeros((8,8),np.uint8)
            if i:
                gt[2,2]=1
            dual_vlm.atomic_npz(self.src/r['evidence'],base=base)
            dual_vlm.atomic_npz(self.src/r['supports'],masks=supports)
            dual_vlm.atomic_npz(self.src/r['evaluation'],mask=gt)
            c=dict(roi_id=1,area=4,reference_source_sha256='shared-normal-source',geometry={})
            for k,value in (('context',20+i),('detail',40+i),('reference',80+i)):
                c[k]=f'local_crops/s{i}_{k}.png'
                dual_vlm.atomic_image(self.src/c[k],np.full((16,16,3),value,np.uint8))
            r['rois']=[c]
            v={k:v for k,v in response(True).items() if k not in ('query_observation','reference_observation','local_difference')}
            atomic_json(self.src/f'local_reviews/s{i}_1.json',dict(id=r['id'],candidate_id=1,
                        fingerprint=digest(self.original),raw_responses=[json.dumps(v)],decision={'parse_ok':True,**v}))
            records.append(r)
        atomic_json(self.src/'manifest.json',dict(fingerprint=digest(self.original),records=records))
        self.args=diag.parser().parse_args(['--stage','prepare','--source_dir',str(self.src),
                                           '--work_dir',str(self.out),'--candidates','0'])
        Teacher.calls=[]

    def run_review(self):
        with patch('tools.vlm_decision.QwenVLLMTeacher',Teacher):
            for i in range(2):
                self.args.shard_id=i
                diag.review(self.args)

    def test_end_to_end_no_gt_before_eval_source_immutable(self):
        snapshot={p:p.read_bytes() for p in self.src.rglob('*') if p.is_file()}
        real=diag.safe_asset
        def forbid_gt(root,name):
            self.assertNotIn('evaluation_only',name)
            return real(root,name)
        with patch.object(diag,'safe_asset',side_effect=forbid_gt):
            diag.prepare(self.args)
            diag.prepare(self.args)
            self.run_review()
        self.assertEqual(len(Teacher.calls),9)
        cfg=load_json(self.out/'next_config.json')
        self.assertEqual(len(cfg['selected']),3)
        for item in cfg['selected']:
            self.assertEqual(item['reference_source_sha256'],item['donor_source_sha256'])
            self.assertNotEqual(item['reference'],item['shuffled_reference'])
        diag.evaluate_reference(self.args)
        with patch('tools.vlm_decision.QwenVLLMTeacher',side_effect=AssertionError('No VLM in action eval')):
            diag.evaluate_action(self.args)
            diag.review(self.args)
        report=load_json(self.out/'action_results/summary.json')
        self.assertEqual(len(report['means']),5)
        with (self.out/'action_results/pixel_effects.csv').open() as f:
            rows={r['mode']:r for r in csv.DictReader(f)}
        self.assertEqual(rows['gt_pixel_DIAGNOSTIC_ONLY']['changed_bg_pixels'],'0')
        self.assertGreater(int(rows['gt_candidate_DIAGNOSTIC_ONLY']['changed_bg_pixels']),0)
        for p,data in snapshot.items():
            self.assertEqual(p.read_bytes(),data)

    def test_stage_prepare_without_gt_files(self):
        for p in (self.src/'evaluation_only').glob('*.npz'):
            p.unlink()
        diag.prepare(self.args)
        self.run_review()

    def test_invalid_response_limit_checked_before_gt(self):
        diag.prepare(self.args)
        class BadTeacher(Teacher):
            def generate(self,prompt,images,trace_dir=None):
                super().generate(prompt,images,trace_dir)
                trace=load_json(trace_dir/'trace.json')
                trace['raw_response']='invalid'
                atomic_json(trace_dir/'trace.json',trace)
                return 'invalid'
        with patch('tools.vlm_decision.QwenVLLMTeacher',BadTeacher):
            for i in range(2):
                self.args.shard_id=i
                diag.review(self.args)
        for p in (self.src/'evaluation_only').glob('*.npz'):
            p.unlink()
        with self.assertRaisesRegex(ValueError,'too many invalid'):
            diag.evaluate_reference(self.args)

    def test_source_mutation_rejected(self):
        diag.prepare(self.args)
        (self.src/'local_crops/s0_detail.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Source changed'):
            diag.load_config(self.args)

    def test_settings_code_and_overlapping_output_rejected(self):
        diag.prepare(self.args)
        self.args.alphas='.5'
        with self.assertRaisesRegex(ValueError,'NEW WORK_DIR'):
            diag.prepare(self.args)
        with patch.object(diag,'code_hashes',return_value={}):
            with self.assertRaisesRegex(ValueError,'code changed'):
                diag.load_config(self.args)
        self.args.work_dir=str(self.src/'child')
        with self.assertRaisesRegex(ValueError,'non-overlapping'):
            diag.prepare(self.args)

    def test_missing_mismatched_trace_and_invalid_reviews(self):
        diag.prepare(self.args)
        with self.assertRaises(FileNotFoundError):
            diag.evaluate_reference(self.args)
        self.run_review()
        cfg=load_json(self.out/'next_config.json')
        item=cfg['selected'][0]
        path=self.out/ARMS[0]/f"{item['key']}.json"
        payload=load_json(path)
        payload['decision']['verdict']='normal_supported'
        atomic_json(path,payload)
        with self.assertRaisesRegex(ValueError,'differs'):
            diag.validate_review(path,cfg,item,ARMS[0])

    def test_selection_repeatable_gt_independent(self):
        diag.prepare(self.args)
        cfg=load_json(self.out/'next_config.json')
        pool=[{k:v for k,v in x.items() if k not in ('shuffled_reference','donor_key','donor_source_sha256')} for x in cfg['selected']]
        a,n=paired_selection(pool,2,42)
        b,m=paired_selection(pool[::-1],2,42)
        self.assertEqual(a,b)
        self.assertEqual(n,m)


BASH=os.environ.get('SRA_TEST_BASH') or (shutil.which('bash') if os.name!='nt' else None)


@unittest.skipUnless(BASH,'Set SRA_TEST_BASH on Windows')
class ShellTests(unittest.TestCase):
    def run_script(self,fail=''):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            shutil.copyfile(Path(__file__).resolve().parents[1]/'run_exp_vlm_next.sh',root/'run.sh')
            (root/'source').mkdir()
            (root/'source/manifest.json').write_text('{}')
            (root/'fake_python').write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n'
                                           '[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n',newline='\n')
            # Git Bash does not enforce POSIX executable bits; Linux does.
            (root/'fake_python').chmod(0o755)
            (root/'flock').write_text('#!/usr/bin/env bash\nexit 0\n',newline='\n')
            (root/'flock').chmod(0o755)
            env=os.environ.copy()
            env.update(SOURCE_WORK_DIR='./source',WORK_DIR='./out',TRAIN_PYTHON='./fake_python',
                       VLM_PYTHON='./fake_python',GPU_IDS='0,1',FAIL_STAGE=fail or 'NEVER')
            env['PATH']=str(root)+os.pathsep+str(Path(BASH).parent)+os.pathsep+env.get('PATH','')
            return subprocess.run([BASH,'run.sh'],cwd=root,env=env,text=True,capture_output=True,timeout=20)
    def test_two_workers_and_pipeline_order(self):
        r=self.run_script()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('GPU=0 ARGS=vlm_next_diagnose.py --stage review',r.stdout)
        self.assertIn('GPU=1 ARGS=vlm_next_diagnose.py --stage review',r.stdout)
        self.assertLess(r.stdout.index('--stage action_eval'),r.stdout.index('--stage review'))
        self.assertIn('--stage reference_eval',r.stdout)
    def test_action_failure_stops_vlm(self):
        r=self.run_script('--stage action_eval')
        self.assertNotEqual(r.returncode,0)
        self.assertNotIn('--stage review',r.stdout)


if __name__=='__main__':
    unittest.main()
