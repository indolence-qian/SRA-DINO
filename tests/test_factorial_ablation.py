import copy
import csv
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

import factorial_ablation_experiment as app
import test_residual_text as previous
from tools.residual_text import paired_directions
from tools.vlm_review import atomic_json, file_digest, load_json
from tools.residual_text_runtime import save_torch, read_torch


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_appearance_is_reference_independent_and_residual_matches_before_projection(self):
        cfg=dict(image_size=4,patch_size=2,layers=[0],feature_dim=4,match_chunk=2,bank_chunk=3)
        class Extractor:
            projection=torch.eye(4)
        views=[torch.eye(4)[None] for _ in range(5)]
        first=[[v.clone() for v in views]]
        second=[[v[:,:1].repeat(1,4,1) for v in views]]
        appearance=app.appearance_features(Extractor(),views,cfg)
        r1=app.export_residuals(Extractor(),views,first,cfg)
        r2=app.export_residuals(Extractor(),views,second,cfg)
        self.assertFalse(r1['residual'].any())
        self.assertGreater(float(abs(r2['residual']).sum()),0)
        self.assertTrue(np.array_equal(appearance,app.appearance_features(Extractor(),views,cfg)))
        expected=(views[0]-second[0][0]).permute(0,2,1).reshape(1,4,2,2).numpy()
        self.assertTrue(np.array_equal(r2['residual'][0],expected))

    def test_markdown_replaces_only_result_region_and_rejects_bad_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'report.md'
            path.write_text('Keep before\n<!-- AUTO_RESULTS_START -->\nold\n<!-- AUTO_RESULTS_END -->\nKeep after',encoding='utf-8')
            app.publish_markdown(path,'| actual | values |')
            value=path.read_text(encoding='utf-8')
            self.assertTrue(value.startswith('Keep before\n'))
            self.assertTrue(value.endswith('\nKeep after'))
            self.assertIn('| actual | values |',value)
            self.assertNotIn('\nold\n',value)
            path.write_text('missing',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'markers'):
                app.publish_markdown(path,'new')

    def test_class_macro_then_seed_mean_not_image_weighted(self):
        values=[]
        for seed in (1,2):
            for value,images in ((.2,100),(.8,1)):
                values.append(dict(dataset='d',score='full',arm='A',seed=seed,images=images,
                                   **{m:value+(.1 if seed==2 else 0) for m in app.METRICS}))
        per_seed=app.summarize(values,('dataset','score','arm','seed'))
        result=app.summarize(per_seed,('dataset','score','arm'))[0]
        self.assertAlmostEqual(result['I_AUROC'],.55)
        self.assertAlmostEqual(result['I_AUROC_std'],.05)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture=previous.WorkflowTests(); self.fixture.setUp()
        self.root=self.fixture.base/'factorial'
        self.args=app.parser().parse_args(['--stage','prepare'])
        for name,value in vars(self.fixture.args).items():
            if hasattr(self.args,name): setattr(self.args,name,value)
        self.args.work_dir=str(self.root); self.args.num_shards=2
        self.args.calibration_images=2

    def tearDown(self):
        self.fixture.tearDown()

    def prepare(self):
        return app.prepare(self.args,copy.deepcopy(self.fixture.catalog))

    def export(self,partition):
        self.args.partition=partition
        with patch.object(app,'FrozenDinoExtractor',previous.FakeExtractor):
            for shard in range(app.config(self.root)['num_shards']):
                self.args.shard_id=shard; app.export(self.args)
        app.seal(self.args)

    def setup_training(self):
        cfg=self.prepare(); self.export('source'); app.fit_stats(self.args)
        states=torch.randn(1,2,3,6,generator=torch.Generator().manual_seed(90))
        directions=paired_directions(states[:,0],states[:,1]).numpy()
        with patch.object(app,'encode_text_bank',return_value=(directions,states.numpy())):
            app.text_bank(self.args)
        return cfg

    def train_all(self):
        for arm in app.ARMS:
            for kind in ('density','text'):
                self.args.arm=arm; self.args.kind=kind; app.train(self.args)
        app.select(self.args)

    def predict_all(self):
        for shard in range(app.config(self.root)['num_shards']):
            self.args.shard_id=shard; app.predict(self.args)

    def test_complete_four_arms_no_target_labels_before_lock_reports_and_package(self):
        cfg=self.setup_training()
        train=app.rows(self.root,'train')
        cal=set(cfg['calibration_ids'])
        self.assertTrue(cal)
        self.assertTrue(all(not r['mask_path'] for r in train if r['id'] in cal))
        self.assertTrue(any(not r['mask_path'] for r in train if r['id'] not in cal))
        target=next(r for r in app.rows(self.root,'eval') if r['mask_path'])
        mask=Path(target['mask_path']); contents=mask.read_bytes(); mask.unlink()
        self.train_all()
        self.export('eval')
        with patch.object(app,'evaluation_mask',side_effect=AssertionError('Target GT opened too early')):
            self.predict_all()
            with self.assertRaises(FileNotFoundError): app.evaluate(self.args)
            app.lock_predictions(self.args)
        mask.write_bytes(contents)
        for shard in range(cfg['num_shards']):
            self.args.shard_id=shard; app.evaluate(self.args)
        app.report(self.args); app.package(self.args)
        decision=load_json(app.decision_path(self.root,42,1))
        self.assertEqual(set(decision['arms']),set(app.ARMS))
        self.assertTrue(all(s['density_kind']=='density' and s['density_epoch']>=1 and s['text_epoch']>=1
                            and s['strength']==.5 for s in decision['arms'].values()))
        out=self.root/'reports/through1'
        with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
            summary=list(csv.DictReader(stream))
        self.assertEqual({r['arm']+'_'+r['score'] for r in summary},set(app.MAPS))
        self.assertIn('Pixel F1-best',(out/'消融试验.md').read_text(encoding='utf-8'))
        self.assertTrue(list((self.root/'panels').rglob('*.png')))
        with zipfile.ZipFile(next(self.root.glob('factorial_analysis_*.zip'))) as archive:
            self.assertIn('reports/through1/消融试验.md',archive.namelist())
            self.assertIn('heads/seed42/A/density/history.json',archive.namelist())
            self.assertFalse(any(n.endswith(('.npz','.pt')) for n in archive.namelist()))
        with patch.object(app,'FrozenDinoExtractor',side_effect=AssertionError('Re-encoding completed cache')):
            app.export(self.args)
        before=file_digest(app.prediction_dir(self.root,42,1)/(target['id']+'.npz'))
        self.predict_all()
        self.assertEqual(before,file_digest(app.prediction_dir(self.root,42,1)/(target['id']+'.npz')))
        mask.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'masks changed'): app.report(self.args)

    def test_raw_inputs_do_not_read_residual_bank_and_global_ignores_locals(self):
        cfg=self.prepare(); self.export('source')
        records=app.rows(self.root,'train')
        a=app.Features(self.root,cfg,records,'A')
        b=app.Features(self.root,cfg,records,'B')
        c=app.Features(self.root,cfg,records,'C')
        d=app.Features(self.root,cfg,records,'D')
        self.assertEqual(a[0]['residual'].shape[0],1)
        self.assertEqual(c[0]['residual'].shape[0],5)
        self.assertTrue(torch.equal(a[0]['residual'],c[0]['residual'][:1]))
        self.assertTrue(torch.equal(b[0]['residual'],d[0]['residual'][:1]))
        self.assertFalse(torch.equal(a[0]['residual'],b[0]['residual']))
        original=app.load_feature
        def changed(*args,**kwargs):
            value=original(*args,**kwargs); value['features'][1:]+=1; return value
        before=a[0]['residual'].clone(); local=c[0]['residual'].clone()
        with patch.object(app,'load_feature',side_effect=changed):
            self.assertTrue(torch.equal(before,a[0]['residual']))
            self.assertFalse(torch.equal(local,c[0]['residual']))
        for row in records:
            app.feature_path(self.root,row,'residual').unlink()
        # Still usable when ALL residual tensors are absent.
        with patch.object(app,'export_residuals',side_effect=AssertionError('NN bypass')):
            raw=app.Features(self.root,cfg,records,'A')
            self.assertTrue(torch.equal(before,raw[0]['residual']))

    def test_cache_tamper_and_config_change_fail_without_overwriting(self):
        cfg=self.prepare(); self.export('source')
        sha=file_digest(self.root/'config.json')
        self.args.text_strength=.25
        with self.assertRaisesRegex(ValueError,'NEW WORK_DIR'): self.prepare()
        self.assertEqual(sha,file_digest(self.root/'config.json'))
        self.args.text_strength=.5
        row=app.rows(self.root,'train')[0]
        path=app.feature_path(self.root,row,'appearance'); path.write_bytes(path.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'changed after sealing'):
            app.Features(self.root,cfg,[row],'A')

    def test_trained_only_selection_rejects_epoch_zero_or_distance(self):
        cfg=self.setup_training(); self.args.arm='A'; self.args.kind='density'
        app.train(self.args)
        path=app.head_dir(self.root,42,'A','density')/'through1.pt'
        meta=read_torch(path); meta['epoch']=0; save_torch(path,meta)
        with self.assertRaisesRegex(ValueError,'fallback forbidden'):
            app.load_snapshot(self.root,cfg,'A','density',42,1,'cpu')
        meta['epoch']=1; meta['kind']='distance'; save_torch(path,meta)
        with self.assertRaisesRegex(ValueError,'fallback forbidden'):
            app.load_snapshot(self.root,cfg,'A','density',42,1,'cpu')

    def test_resumed_and_uninterrupted_training_match_and_old_snapshot_survives(self):
        self.setup_training()
        full=self.fixture.base/'full'
        shutil.copytree(self.root,full)
        self.args.arm='A'; self.args.kind='density'
        self.args.work_dir=str(full); self.args.until_epoch=2; app.train(self.args)
        self.args.work_dir=str(self.root); self.args.until_epoch=1; app.train(self.args)
        snapshot=app.head_dir(self.root,42,'A','density')/'through1.pt'; sha=file_digest(snapshot)
        self.args.until_epoch=2; app.train(self.args)
        first=read_torch(app.head_dir(full,42,'A','density')/'last.pt')
        resumed=read_torch(app.head_dir(self.root,42,'A','density')/'last.pt')
        self.assertEqual(sha,file_digest(snapshot))
        for key,value in first['state_dict'].items(): self.assertTrue(torch.equal(value,resumed['state_dict'][key]),key)
        app.train(self.args)


@unittest.skipUnless(previous.BASH,'Set SRA_TEST_BASH to Git Bash')
class ShellTests(unittest.TestCase):
    def run_script(self,fail='NEVER',gpus='0,1',**flags):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); shutil.copyfile(app.REPO/'run_exp_factorial_ablation.sh',root/'run.sh')
            (root/'fake_python').write_bytes(b'#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]] || exit 1\nexit 0\n')
            (root/'flock').write_bytes(b'#!/usr/bin/env bash\nexit 0\n')
            for name in ('fake_python','flock'): (root/name).chmod(0o755)
            env=os.environ.copy(); env.update(TRAIN_PYTHON='./fake_python',WORK_DIR='./output',GPU_IDS=gpus,FAIL_STAGE=fail)
            env.update(flags)
            env['PATH']=str(root)+os.pathsep+str(Path(previous.BASH).parent)+os.pathsep+env.get('PATH','')
            return subprocess.run([previous.BASH,'run.sh'],cwd=root,env=env,capture_output=True,text=True,timeout=45)

    def test_all_four_models_same_text_setting_and_prediction_lock_before_evaluation(self):
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        for arm in app.ARMS: self.assertIn('--arm '+arm,result.stdout)
        for kind in ('density','text'): self.assertIn('--kind '+kind,result.stdout)
        self.assertIn('GPU=1',result.stdout)
        self.assertIn('--text_strength 0.5',result.stdout)
        self.assertIn('--publish_report',result.stdout)
        self.assertLess(result.stdout.rindex('--stage train '),result.stdout.index('--stage predict'))
        self.assertLess(result.stdout.index('--stage lock_predictions'),result.stdout.index('--stage evaluate'))

    def test_worker_failure_blocks_evaluation_and_duplicate_gpu_is_rejected(self):
        result=self.run_script(fail='--stage train ')
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('--stage predict',result.stdout)
        self.assertNotIn('--stage evaluate',result.stdout)
        self.assertNotEqual(self.run_script(gpus='0,0').returncode,0)

    def test_single_gpu_metrics_only_resume(self):
        result=self.run_script(gpus='0',RUN_EXPORT='0',RUN_TRAIN='0',RUN_PREDICT='0',UPDATE_REPO_REPORT='0')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotIn('--stage train ',result.stdout)
        self.assertNotIn('--stage export ',result.stdout)
        self.assertNotIn('--publish_report',result.stdout)
        self.assertIn('--stage evaluate',result.stdout)


if __name__=='__main__': unittest.main()
