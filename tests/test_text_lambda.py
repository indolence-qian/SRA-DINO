import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

import lambda_text_experiment as app
from tools.text_lambda import (DEFAULT_GRID,basis_maps,best_lambda,diagnostics,name_for,needs_extension,parse_grid,readouts,score)
import test_text_alignment_ablation as previous
from tools.vlm_review import file_digest,load_json
from PIL import Image


class CoreTests(unittest.TestCase):
    def test_grid_zero_exactness_and_selection_ties_extension(self):
        specs=readouts(DEFAULT_GRID)
        self.assertEqual(len(specs),22)
        self.assertEqual(sum(v['kind']!='increment' for v in specs.values()),19)
        visual=np.ones((2,2),np.float32)
        self.assertIs(score({'V':visual,'A_T1':np.full((2,2),np.nan)},dict(kind='fusion',arm='T1',strength=0)),visual)
        points={v:dict(P_AP=.4,PRO_exact=.4) for v in DEFAULT_GRID}
        self.assertEqual(best_lambda(points,DEFAULT_GRID),0.)
        self.assertFalse(needs_extension(points,DEFAULT_GRID))
        points[4]=dict(P_AP=.6,PRO_exact=.6)
        self.assertTrue(needs_extension(points,DEFAULT_GRID))
        points[2]=dict(P_AP=.6,PRO_exact=.6)
        self.assertFalse(needs_extension(points,DEFAULT_GRID))
        with self.assertRaises(ValueError): parse_grid('0,0.5,1,nan')

    def test_positive_clamp_is_before_global_local_fusion(self):
        torch.set_num_threads(1)
        boxes=torch.tensor([[[0,0,1,1],[0,0,.6,.6],[.4,0,1,.6],[0,.4,.6,1],[.4,.4,1,1]]],dtype=torch.float32)
        v=torch.zeros(1,5,1,3,3); t=torch.ones_like(v); t[:,0]=-1
        cfg=dict(settings=dict(high_weight=.25),visual_calibration={})
        with patch.object(app.app.base,'calibrated_map',side_effect=lambda model,kind,*args:v if kind=='density' else t):
            values=basis_maps(None,{'T1':None},{'text':{'T1':{'calibration':{}}}},{'boxes':boxes},None,cfg,16)
        self.assertTrue(np.allclose(values['A_T1'],.25))
        wrong=app.app.fuse_views(t,boxes,16,True,.25).clamp_min(0)[0,0].numpy()
        self.assertFalse(np.allclose(values['A_T1'],wrong))

    def test_zero_annotation_keeps_image_label_and_threshold_uses_strict_greater(self):
        values=np.zeros((8,8),np.float32); gt=np.zeros((8,8),bool)
        point=diagnostics(values,gt,True,0.)
        self.assertEqual(point['outcome'],'FN'); self.assertIsNone(point['dice'])
        values[1,1]=1
        point=diagnostics(values,gt,True,0.)
        self.assertEqual(point['outcome'],'TP'); self.assertEqual(point['fp_pixels'],1)
        self.assertIsNone(point['iou'])

    def test_markdown_publication_preserves_historical_results(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'report.md'; path.write_text('Original historical results\n',encoding='utf-8')
            app.publish(path,'NEW A'); app.publish(path,'NEW B')
            text=path.read_text(encoding='utf-8')
            self.assertTrue(text.startswith('Original historical results\n'))
            self.assertNotIn('NEW A',text); self.assertIn('NEW B',text)
            self.assertEqual(text.count(app.SECTION_START),1)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture=previous.WorkflowTests(); self.fixture.setUp()
        self.fixture.setup_text(); self.fixture.train_all()
        self.root=self.fixture.fixture.fixture.base/'lambda'
        self.args=app.parser().parse_args(['--stage','prepare','--work_dir',str(self.root),
            '--text_work_dir',str(self.fixture.root),'--until_epoch','1','--device','cpu','--html_map_side','32'])
        self.old_hashes=self.hash_inputs()

    def tearDown(self): self.fixture.tearDown()

    def hash_inputs(self):
        paths=list(self.fixture.root.rglob('*'))+list(self.fixture.fixture.root.rglob('*'))
        return {str(p):file_digest(p) for p in paths if p.is_file()}

    def export(self,partition):
        self.args.partition=partition
        for shard in range(2): self.args.shard_id=shard; app.export(self.args)
        app.seal(self.args)

    def test_end_to_end_no_retraining_no_early_gt_full_html_and_resume(self):
        source,_=app.app.parent(app.app.config(self.fixture.root))
        target_paths={r['mask_path'] for r in app.app.f.rows(source,'eval') if r['mask_path']}
        empty_row=next(r for r in app.app.f.rows(source,'eval') if r['mask_path'])
        Image.fromarray(np.zeros((empty_row['native_size'][1],empty_row['native_size'][0]),np.uint8)).save(empty_row['mask_path'])
        original_truth=app.app.evaluation_truth
        def guarded(row,side):
            if row['mask_path'] in target_paths:
                self.assertTrue(app.prediction_lock_path(self.root,42).exists(),'Target GT opened before prediction lock')
            return original_truth(row,side)
        with patch.object(app.app,'train',side_effect=AssertionError('No training')), patch.object(app.app,'evaluation_truth',side_effect=guarded):
            cfg=app.prepare(self.args)
            # prepare returns None; configuration is independently loaded from disk.
            self.export('source'); app.select(self.args); self.export('eval')
            with self.assertRaises(FileNotFoundError): app.evaluate(self.args)
            app.lock_predictions(self.args)
            for shard in range(2): self.args.shard_id=shard; app.evaluate(self.args)
            app.report(self.args); app.package(self.args)
        self.assertEqual(self.old_hashes,self.hash_inputs())
        selected=load_json(app.selection_path(self.root)); expected=len(selected['specs'])
        with (self.root/'reports/summary.csv').open(encoding='utf-8-sig') as stream: summary=list(csv.DictReader(stream))
        self.assertEqual(len(summary),expected)
        for path in (self.root/'metrics/seed42').glob('*.json'):
            for row in load_json(path)['metrics']:
                self.assertEqual(row['images'],row['pixel_images_included'])
                self.assertEqual(row['abnormal_images'],2)
                self.assertEqual(row['empty_anomaly_masks'],1)
        self.assertEqual(set(selected['chosen']['42']),{'T1','T2','T3'})
        html=(self.root/'html/index.html').read_text(encoding='utf-8')
        self.assertNotIn('__LAMBDA_DATA__',html)
        self.assertIn('逐图好坏',html); self.assertIn('image',html.lower())
        self.assertTrue(list((self.root/'html/assets').glob('*.js')))
        with zipfile.ZipFile(self.root/'lambda_text_html_resources.zip') as z:
            self.assertIn('index.html',z.namelist()); self.assertTrue(any(n.endswith('.jpg') for n in z.namelist()))
        with zipfile.ZipFile(next(self.root.glob('lambda_text_analysis_*.zip'))) as z:
            self.assertIn('source_selection.json',z.namelist()); self.assertFalse(any(n.endswith('.npz') for n in z.namelist()))
        # New target results do not feed back into the already frozen source lambda.
        choice_sha=file_digest(app.selection_path(self.root)); app.select(self.args)
        self.assertEqual(choice_sha,file_digest(app.selection_path(self.root)))
        self.export('eval')
        # Leave a persistent, task-owned report for optional real browser QA.
        if os.environ.get('LAMBDA_QA_DIR'):
            shutil.copytree(self.root/'html',os.environ['LAMBDA_QA_DIR'],dirs_exist_ok=True)

    def test_half_lambda_matches_original_fusion_and_source_extension_is_global(self):
        app.prepare(self.args); self.export('source')
        cfg=app.config(self.root); text_root,text,source,pcfg=app.context(cfg)
        selected=app.app.decision(text_root,text,42,1)
        visual=app.app.visual_model(text,'cpu')
        texts={a:app.app.load_snapshot(text_root,text,a,42,1,'cpu')[0] for a in ('T1','T2','T3')}
        row=app.records(cfg,'source')[0]
        item=app.app.f.Features(source,pcfg,[row],'D',False)[0]
        batch=app.app.base.batch_to(item,'cpu'); scale=app.app.f.scale_for(source,pcfg,'D','cpu')
        with torch.inference_mode():
            basis=basis_maps(visual,texts,selected,batch,scale,text,text['settings']['eval_size'])
            old=app.app.score_maps(visual,texts,selected['text'],batch,scale,text)
        self.assertTrue(np.array_equal(basis['V'],old['V']))
        for arm in ('T1','T2','T3'):
            self.assertTrue(np.allclose(score(basis,readouts(DEFAULT_GRID)[name_for(arm,.5)]),old[arm],atol=2e-5,rtol=2e-5))
        def forced_source(root,cfg,seed,specs):
            return {name:dict(point=dict(P_AP=.2+(spec['strength'] or 0)*.01 if spec['arm']=='T1' else .2,PRO_exact=.2),threshold=.1)
                    for name,spec in specs.items()}
        with patch.object(app,'source_pass',side_effect=forced_source): app.select(self.args)
        decision=load_json(app.selection_path(self.root))
        self.assertIn(8.,decision['grid'])
        self.assertEqual(decision['chosen']['42']['T1'],8.)
        for arm in ('T1','T2','T3'): self.assertIn(name_for(arm,8),decision['specs'])

    def test_tampered_basis_and_changed_configuration_are_rejected(self):
        app.prepare(self.args); self.export('source'); app.select(self.args); self.export('eval'); app.lock_predictions(self.args)
        cfg=app.config(self.root); row=app.records(cfg,'eval')[0]
        path=app.basis_path(self.root,42,'eval',row); path.write_bytes(path.read_bytes()+b'tamper')
        with patch.object(app.app,'evaluation_truth',side_effect=AssertionError('GT read too early')):
            with self.assertRaisesRegex(ValueError,'Basis changed after seal'): app.evaluate(self.args)
        self.args.html_map_side=64
        with self.assertRaisesRegex(ValueError,'NEW WORK_DIR'): app.prepare(self.args)


@unittest.skipUnless(previous.original.BASH,'Set SRA_TEST_BASH to Git Bash')
class ShellTests(unittest.TestCase):
    def run_script(self,fail='NEVER',**flags):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); shutil.copyfile(app.REPO/'run_exp_text_lambda.sh',root/'run.sh')
            (root/'old').mkdir(); (root/'old/.run.lock').touch()
            (root/'fake_python').write_bytes(b'#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]] || exit 1\nexit 0\n')
            (root/'flock').write_bytes(b'#!/usr/bin/env bash\nexit 0\n')
            for name in ('fake_python','flock'): (root/name).chmod(0o755)
            env=os.environ.copy(); env.update(TRAIN_PYTHON='./fake_python',TEXT_WORK_DIR='./old',WORK_DIR='./new',FAIL_STAGE=fail)
            env.update(flags); env['PATH']=str(root)+os.pathsep+str(Path(previous.original.BASH).parent)+os.pathsep+env.get('PATH','')
            return subprocess.run([previous.original.BASH,'run.sh'],cwd=root,env=env,capture_output=True,text=True,timeout=45)

    def test_source_choice_and_all_seed_locks_precede_evaluation_and_html(self):
        r=self.run_script(TRAIN_SEEDS='42,43'); self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertNotIn('--stage train ',r.stdout)
        self.assertLess(r.stdout.index('--stage select'),r.stdout.index('--partition eval'))
        self.assertLess(r.stdout.rindex('--stage lock_predictions'),r.stdout.index('--stage evaluate'))
        self.assertIn('--stage report',r.stdout); self.assertIn('--stage package',r.stdout)
        r=self.run_script(RUN_EXPORT='0',RUN_EVAL='0',UPDATE_REPO_REPORT='0')
        self.assertEqual(r.returncode,0,r.stdout+r.stderr); self.assertNotIn('--stage export',r.stdout)
        self.assertNotIn('--stage evaluate',r.stdout); self.assertNotIn('--publish_report',r.stdout)

    def test_worker_failure_blocks_report_and_invalid_inputs_fail(self):
        r=self.run_script(fail='--stage export'); self.assertNotEqual(r.returncode,0)
        self.assertNotIn('--stage evaluate',r.stdout); self.assertNotIn('--stage report',r.stdout)
        self.assertNotEqual(self.run_script(GPU_IDS='0,0').returncode,0)
        self.assertNotEqual(self.run_script(TRAIN_SEEDS='42,42').returncode,0)


if __name__=='__main__': unittest.main()
