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
from PIL import Image
import torch
from torch.nn import functional as F

import factorial_ablation_experiment as factorial
import text_alignment_ablation as app
from tools.residual_text import ResidualText, paired_directions, text_loss
from tools.text_ablation import (MODES, TextMatcher, bank_audit, corrected_metrics,
                                 evaluation_truth, validate_prompts)
from tools.vlm_review import file_digest, load_json
from tests import test_factorial_ablation as previous
from tests import test_residual_text as original


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.states = torch.randn(4,2,3,7,generator=torch.Generator().manual_seed(9))

    def test_original_delta_parity_equal_parameters_zero_and_gradients(self):
        directions = paired_directions(self.states[:,0],self.states[:,1])[:1]
        torch.manual_seed(7); old = ResidualText(2,4,directions,8)
        models = []
        for mode in MODES.values():
            torch.manual_seed(7); models.append(TextMatcher(2,4,self.states,mode,8))
        for model in models:
            for a,b in zip(old.parameters(),model.parameters()):
                self.assertTrue(torch.equal(a,b))
            self.assertNotIn('directions',dict(model.named_parameters()))
        residual = torch.randn(1,5,2,4,3,3)
        self.assertTrue(torch.allclose(old(residual)[0],models[0](residual)[0],atol=1e-6))
        labels = torch.zeros(1,5,3,3); labels[:,:,1,1]=1
        for model in models:
            self.assertEqual(float(model(torch.zeros_like(residual))[0].abs().max()),0.)
            loss = text_loss(model,residual,labels); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_two_state_softmax_equivalence_and_mean_pair_scores(self):
        z = torch.randn(11,7)
        a = TextMatcher(1,4,self.states,'single_delta',8)
        b = TextMatcher(1,4,self.states,'normal_abnormal',8)
        c = TextMatcher(1,4,self.states,'mean_paired_delta',8)
        length = (b.abnormal-b.normal).norm()
        self.assertTrue(torch.allclose(b.readout(z),length*a.readout(z),atol=1e-6))
        y = torch.rand(11)
        logits = b.state_logits(z)*4
        logits[:,1]-=2
        ce = -(torch.stack([1-y,y],-1)*logits.log_softmax(-1)).sum(-1).mean()
        bce = F.binary_cross_entropy_with_logits(4*b.readout(z)[:,0]-2,y)
        self.assertTrue(torch.allclose(ce,bce,atol=1e-6))
        expected = torch.stack([z@d for d in c.directions]).mean(0)
        self.assertTrue(torch.allclose(c.readout(z)[:,0],expected,atol=1e-6))
        self.assertTrue(torch.allclose(c.readout(z)[:,0],z@c.directions.mean(0),atol=1e-6))
        self.assertLess(bank_audit(self.states)['single_equivalence_max_error'],1e-6)

    def test_prompt_definition_and_degenerate_directions(self):
        p = load_json(app.REPO/'configs/text_ablation_prompts_v1.json')
        self.assertEqual(len(validate_prompts(p)),24)
        p.append(p[-1])
        with self.assertRaises(ValueError): validate_prompts(p)
        bad = self.states.clone(); bad[:,1]=bad[:,0]
        with self.assertRaisesRegex(ValueError,'Degenerate'): bank_audit(bad)

    def test_masks_preserve_01_palette_color_tiny_defects_and_image_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'mask.png'; row = dict(id='x',native_size=[64,48],mask_path=str(path))
            values = np.zeros((48,64),np.uint8); values[1,1]=1
            for kind in ('L','P','RGB','RGBA'):
                if kind in ('RGB','RGBA'):
                    array = np.zeros((48,64,len(kind)),np.uint8); array[1,1,0]=1
                    if kind=='RGBA': array[:,:,3]=255
                    im = Image.fromarray(array)
                else:
                    im = Image.fromarray(values)
                    if kind=='P': im=im.convert('P')
                im.save(path)
                mask,audit = evaluation_truth(row,8)
                self.assertTrue(mask.any(),kind)
                self.assertEqual(audit['native_positive'],1,kind)
                self.assertEqual(audit['image_label'],1)
            # An empty mask does not change the explicit anomaly image label.
            truth = np.zeros((3,8,8),bool); truth[2,1,1]=True
            maps = np.zeros((3,8,8)); maps[1]=.5; maps[2]=1
            point = corrected_metrics(truth,maps,[0,1,1],.2)
            self.assertEqual(point['abnormal_images'],2)
            self.assertEqual(point['I_AUROC'],1.)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = previous.WorkflowTests(); self.fixture.setUp()
        self.parent_cfg = self.fixture.setup_training()
        self.fixture.args.arm='D'; self.fixture.args.kind='density'
        factorial.train(self.fixture.args); self.fixture.export('eval')
        self.root = self.fixture.fixture.base/'text'
        self.args = app.parser().parse_args(['--stage','prepare','--work_dir',str(self.root),
            '--base_work_dir',str(self.fixture.root),'--base_until_epoch','1',
            '--epochs','2','--until_epoch','1','--device','cpu'])
        self.parent_hashes = self.hash_parent()

    def tearDown(self): self.fixture.tearDown()

    def hash_parent(self):
        return {str(p.relative_to(self.fixture.root)):file_digest(p) for p in self.fixture.root.rglob('*') if p.is_file()}

    def setup_text(self):
        cfg = app.prepare(self.args); app.prepare(self.args)
        states = torch.randn(len(cfg['prompts']),2,3,6,generator=torch.Generator().manual_seed(90))
        with patch.object(app,'encode_text_bank',return_value=(paired_directions(states[:,0],states[:,1]).numpy(),states.numpy())):
            app.text_bank(self.args)
        return cfg

    def train_all(self):
        for arm in MODES:
            self.args.arm=arm; app.train(self.args)
        app.select(self.args)

    def predict_all(self):
        for shard in range(2):
            self.args.shard_id=shard; app.predict(self.args)

    def test_full_workflow_source_only_shared_flow_report_and_tamper_guards(self):
        targets = {r['mask_path'] for r in factorial.rows(self.fixture.root,'eval') if r['mask_path']}
        opened = Image.open
        def guarded(path,*args,**kwargs):
            if str(path) in targets:
                self.assertTrue((app.prediction_dir(self.root,42,1)/'locked.json').exists(),'Premature target label access')
            return opened(path,*args,**kwargs)
        with patch.object(Image,'open',side_effect=guarded):
            cfg = self.setup_text(); self.train_all()
            self.predict_all()
            with self.assertRaises(FileNotFoundError): app.evaluate(self.args)
            app.lock_predictions(self.args)
            for shard in range(2):
                self.args.shard_id=shard; app.evaluate(self.args)
            app.report(self.args); app.package(self.args)
        self.assertEqual(self.parent_hashes,self.hash_parent())
        with (self.root/'reports/through1/summary.csv').open(encoding='utf-8-sig') as stream:
            summary=list(csv.DictReader(stream))
        self.assertEqual({r['arm']+('_text' if r['score']=='text' else '') for r in summary},set(app.MAPS))
        report=(self.root/'reports/through1/对比实验.md').read_text(encoding='utf-8')
        self.assertIn('T1-T2',report); self.assertIn('T3-T1',report)
        self.assertTrue(list((self.root/'mask_audit').glob('*.json')))
        with zipfile.ZipFile(next(self.root.glob('text_comparison_analysis_*.zip'))) as archive:
            self.assertIn('reports/through1/对比实验.md',archive.namelist())
            self.assertFalse(any(n.endswith(('.pt','.npz')) for n in archive.namelist()))
        self.predict_all()  # idempotent resume
        path=next(app.prediction_dir(self.root,42,1).glob('*.npz')); contents=path.read_bytes()
        path.write_bytes(contents+b'tamper')
        with self.assertRaisesRegex(ValueError,'Predictions changed after evaluation'): app.report(self.args)
        path.write_bytes(contents)
        # Original test labels changed -> stale report is rejected.
        Path(next(iter(targets))).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed masks'): app.report(self.args)

    def test_empty_original_anomaly_mask_stops_evaluation_with_audit(self):
        self.setup_text(); self.train_all(); self.predict_all(); app.lock_predictions(self.args)
        row=next(r for r in factorial.rows(self.fixture.root,'eval') if r['mask_path'])
        Image.fromarray(np.zeros((row['native_size'][1],row['native_size'][0]),np.uint8)).save(row['mask_path'])
        self.args.shard_id=0
        with self.assertRaisesRegex(ValueError,'Empty original anomaly mask'): app.evaluate(self.args)
        audit=load_json(self.root/'mask_audit'/(row['dataset']+'__'+row['category']+'.json'))
        self.assertTrue(any(v['image_label'] and not v['native_positive'] for v in audit['images']))
        self.assertFalse(list((self.root/'metrics').rglob('*.json')))

    def test_resumed_training_matches_uninterrupted_and_preserves_old_snapshot(self):
        self.setup_text(); full=self.fixture.fixture.base/'text_full'
        shutil.copytree(self.root,full)
        self.args.arm='T3'; self.args.work_dir=str(full); self.args.until_epoch=2; app.train(self.args)
        self.args.work_dir=str(self.root); self.args.until_epoch=1; app.train(self.args)
        old=app.head_dir(self.root,42,'T3')/'through1.pt'; sha=file_digest(old)
        self.args.until_epoch=2; app.train(self.args)
        a=app.read_torch(app.head_dir(full,42,'T3')/'last.pt')
        b=app.read_torch(app.head_dir(self.root,42,'T3')/'last.pt')
        for k,v in a['state_dict'].items(): self.assertTrue(torch.equal(v,b['state_dict'][k]),k)
        self.assertEqual(sha,file_digest(old)); app.train(self.args)

    def test_configuration_and_frozen_flow_changes_fail(self):
        self.setup_text(); self.args.text_strength=.25
        with self.assertRaisesRegex(ValueError,'NEW WORK_DIR'): app.prepare(self.args)
        self.args.text_strength=.5
        cfg=app.config(self.root)
        parent=Path(cfg['base_work_dir']); name=next(n for n in cfg['parent_artifacts'] if n.endswith('.pt'))
        (parent/name).write_bytes((parent/name).read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'parent artifact changed'): app.config(self.root)


@unittest.skipUnless(original.BASH,'Set SRA_TEST_BASH to Git Bash')
class ShellTests(unittest.TestCase):
    def run_script(self,fail='NEVER',gpus='0,1',**flags):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); shutil.copyfile(app.REPO/'run_exp_text_alignment_ablation.sh',root/'run.sh')
            (root/'parent').mkdir(); (root/'parent/.run.lock').touch()
            (root/'fake_python').write_bytes(b'#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]] || exit 1\nexit 0\n')
            (root/'flock').write_bytes(b'#!/usr/bin/env bash\nexit 0\n')
            for name in ('fake_python','flock'): (root/name).chmod(0o755)
            env=os.environ.copy(); env.update(TRAIN_PYTHON='./fake_python',WORK_DIR='./output',BASE_WORK_DIR='./parent',GPU_IDS=gpus,FAIL_STAGE=fail)
            env.update(flags); env['PATH']=str(root)+os.pathsep+str(Path(original.BASH).parent)+os.pathsep+env.get('PATH','')
            return subprocess.run([original.BASH,'run.sh'],cwd=root,env=env,capture_output=True,text=True,timeout=45)

    def test_three_heads_shared_visual_order_and_metrics_only_resume(self):
        r=self.run_script(); self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        for arm in MODES: self.assertIn('--arm '+arm,r.stdout)
        self.assertNotIn('--stage export',r.stdout); self.assertNotIn('--kind density',r.stdout)
        self.assertLess(r.stdout.rindex('--stage train '),r.stdout.index('--stage predict'))
        self.assertLess(r.stdout.index('--stage lock_predictions'),r.stdout.index('--stage evaluate'))
        r=self.run_script(gpus='0',RUN_TRAIN='0',RUN_PREDICT='0',UPDATE_REPO_REPORT='0')
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertNotIn('--stage train ',r.stdout); self.assertNotIn('--stage predict ',r.stdout)
        self.assertIn('--stage evaluate ',r.stdout); self.assertNotIn('--publish_report',r.stdout)

    def test_worker_failure_and_invalid_settings_stop_later_stages(self):
        r=self.run_script(fail='--stage train ')
        self.assertNotEqual(r.returncode,0); self.assertNotIn('--stage evaluate',r.stdout)
        self.assertNotEqual(self.run_script(gpus='0,0').returncode,0)
        self.assertNotEqual(self.run_script(EPOCHS='3',TRAIN_UNTIL_EPOCH='5').returncode,0)


if __name__=='__main__': unittest.main()
