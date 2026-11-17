import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import warnings
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

import repair_text_evaluation as repair
import test_text_alignment_ablation as previous
from tools.vlm_review import file_digest,load_json


class MetricTests(unittest.TestCase):
    def test_empty_anomaly_retained_in_both_image_and_pixel_evaluation(self):
        truth=np.zeros((3,8,8),bool); truth[2,1,1]=True
        maps=np.zeros((3,8,8),np.float32); maps[1]=.8; maps[2,1,1]=.6
        labels=np.array([0,1,1],bool)
        full,=repair.metric_scopes(truth,maps,labels,.4)
        self.assertEqual(full['I_AUROC'],1.)
        self.assertEqual(full['abnormal_images'],2)
        self.assertEqual(full['images'],3)
        self.assertEqual(full['pixel_images_included'],3)
        self.assertEqual(full['empty_anomaly_masks'],1)
        self.assertLess(full['P_AP'],1.)  # High scores on zero GT count as pixel FPs.
        expected=repair.app.corrected_metrics(truth,maps,labels,.4)
        for m in repair.app.METRICS: self.assertEqual(full[m],expected[m])

    def test_complete_masks_preserve_original_metrics_and_all_zero_gt_is_safe(self):
        truth=np.zeros((2,8,8),bool); truth[1,1,1]=True
        maps=truth.astype(np.float32); labels=np.array([0,1],bool)
        full,=repair.metric_scopes(truth,maps,labels,.5)
        old=repair.app.corrected_metrics(truth,maps,labels,.5)
        for m in repair.app.METRICS:
            self.assertEqual(full[m],old[m])
        truth[:]=False; maps[1]=1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            full,=repair.metric_scopes(truth,maps,labels,.5)
        self.assertIsNone(full['P_AP'])  # No foreground in the entire class.
        self.assertEqual(full['I_AUROC'],1.)
        self.assertEqual(full['background_FPR_source_threshold'],.5)
        self.assertEqual(full['pixel_images_included'],2)

    def test_btad_directory_labels_are_independent_of_mask_pixels(self):
        row=dict(dataset='btad',image_path='/data/03/test/ko/0030.bmp',mask_path='/data/03/ground_truth/ko/0030.bmp')
        self.assertTrue(repair.dataset_label(row))
        row.update(image_path='/data/03/test/ok/0030.bmp',mask_path=None)
        self.assertFalse(repair.dataset_label(row))
        row['mask_path']='/data/03/ground_truth/ko/0030.bmp'
        with self.assertRaisesRegex(ValueError,'contradiction'): repair.dataset_label(row)

    def test_unavailable_class_not_silently_removed_from_macro_mean(self):
        rows=[dict(dataset='d',category='one',I_AUROC=.6,P_AP=.9),
              dict(dataset='d',category='two',I_AUROC=.8,P_AP=None)]
        point=repair.aggregate(rows,('dataset',),('I_AUROC','P_AP'))[0]
        self.assertAlmostEqual(point['I_AUROC'],.7)
        self.assertIsNone(point['P_AP']); self.assertIsNone(point['P_AP_std'])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture=previous.WorkflowTests(); self.fixture.setUp()
        self.fixture.setup_text(); self.fixture.train_all(); self.fixture.predict_all()
        repair.app.lock_predictions(self.fixture.args)
        self.root=self.fixture.fixture.fixture.base/'reeval'
        self.args=repair.parser().parse_args(['--stage','prepare','--work_dir',str(self.fixture.root),
            '--reeval_dir',str(self.root),'--until_epoch','1'])

    def tearDown(self): self.fixture.tearDown()

    def run_evaluation(self):
        repair.prepare(self.args); repair.audit(self.args)
        for shard in range(2):
            self.args.shard_id=shard; repair.evaluate(self.args)
        repair.report(self.args); repair.package(self.args)

    def test_recover_real_failure_without_mutating_old_artifacts_or_retraining(self):
        source,_=repair.app.parent(repair.app.config(self.fixture.root))
        row=next(r for r in repair.app.f.rows(source,'eval') if r['mask_path'])
        Image.fromarray(np.zeros((row['native_size'][1],row['native_size'][0]),np.uint8)).save(row['mask_path'])
        self.fixture.args.shard_id=0
        with self.assertRaisesRegex(ValueError,'Empty original anomaly mask'):
            repair.app.evaluate(self.fixture.args)
        before={str(p):file_digest(p) for p in self.fixture.root.rglob('*') if p.is_file()}
        with patch.object(repair.app,'train',side_effect=AssertionError('Retraining forbidden')), patch.object(repair.app,'predict',side_effect=AssertionError('Reprediction forbidden')):
            self.run_evaluation()
        self.assertEqual(before,{str(p):file_digest(p) for p in self.fixture.root.rglob('*') if p.is_file()})
        with (self.root/'reports/summary.csv').open(encoding='utf-8-sig') as stream:
            summary=list(csv.DictReader(stream))
        self.assertEqual(len(summary),7)
        self.assertEqual({r['scope'] for r in summary},{'full_set'})
        self.assertTrue(all(r['P_AP']!='' for r in summary))
        with (self.root/'empty_masks.csv').open(encoding='utf-8-sig') as stream:
            empty=list(csv.DictReader(stream))
        self.assertEqual(len(empty),1); self.assertEqual(empty[0]['mask_path'],row['mask_path'])
        self.assertTrue(list((self.root/'empty_mask_previews').glob('*.png')))
        self.assertTrue(list((self.root/'panels').rglob('*.png')))
        self.assertTrue((self.root/'reports/selected_epochs.csv').is_file())
        text=(self.root/'reports/对比实验.md').read_text(encoding='utf-8')
        self.assertIn('保留全部样本',text)
        self.assertIn('T1-T2',text); self.assertIn('T3-T1',text)
        with zipfile.ZipFile(next(self.root.glob('text_reevaluation_analysis_*.zip'))) as archive:
            self.assertIn('reports/对比实验.md',archive.namelist())
            self.assertIn('empty_masks.csv',archive.namelist())
            self.assertIn('provenance/official_zero_mask_reference.json',archive.namelist())
            self.assertFalse(any(n.endswith(('.pt','.npz')) for n in archive.namelist()))
        self.run_evaluation()  # Idempotent resume.
        Image.fromarray(np.ones((row['native_size'][1],row['native_size'][0]),np.uint8)).save(row['mask_path'])
        with self.assertRaisesRegex(ValueError,'Target mask changed'): repair.report(self.args)

    def test_corrupt_prediction_rejected_before_any_target_mask_is_read(self):
        repair.prepare(self.args)
        path=next(repair.app.prediction_dir(self.fixture.root,42,1).glob('*.npz'))
        path.write_bytes(path.read_bytes()+b'tamper')
        with patch.object(repair.app,'evaluation_truth',side_effect=AssertionError('GT read too early')):
            with self.assertRaisesRegex(ValueError,'Prediction changed after lock'): repair.audit(self.args)

    def test_complete_dataset_recovery_and_missing_prediction_lock(self):
        self.run_evaluation()
        with (self.root/'reports/summary.csv').open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual({r['scope'] for r in rows},{'full_set'})
        self.fixture.args.shard_id=0; repair.app.evaluate(self.fixture.args)
        old=load_json(self.fixture.root/'metrics/seed42/through1/mvtec__target.json')['metrics']
        new=load_json(self.root/'metrics/seed42/mvtec__target.json')['records']
        lookup={(r['arm'],r['score']):r for r in old}
        for r in new:
            for m in repair.app.METRICS: self.assertEqual(r[m],lookup[(r['arm'],r['score'])][m])
        (repair.app.prediction_dir(self.fixture.root,42,1)/'locked.json').unlink()
        with patch.object(repair.app,'evaluation_truth',side_effect=AssertionError('GT read too early')):
            with self.assertRaises(FileNotFoundError): repair.prepare(self.args)


@unittest.skipUnless(previous.original.BASH,'Set SRA_TEST_BASH to Git Bash')
class ShellTests(unittest.TestCase):
    def test_new_full_runner_keeps_original_training_and_uses_v3_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); shutil.copyfile(repair.REPO/'run_exp_text_comparison_v3.sh',root/'run.sh')
            (root/'run_exp_text_alignment_ablation.sh').write_bytes(b'#!/usr/bin/env bash\necho "TRAIN EVAL=$RUN_EVAL PACKAGE=$RUN_PACKAGE WORK=$WORK_DIR UNTIL=$TRAIN_UNTIL_EPOCH"\n[[ "${FAIL_TRAIN:-0}" == 0 ]]\n')
            (root/'run_repair_text_evaluation.sh').write_bytes(b'#!/usr/bin/env bash\necho "REEVAL WORK=$WORK_DIR UNTIL=$TRAIN_UNTIL_EPOCH"\n')
            env=os.environ.copy(); env.update(WORK_DIR='./original',TRAIN_UNTIL_EPOCH='15')
            env['PATH']=str(Path(previous.original.BASH).parent)+os.pathsep+env.get('PATH','')
            r=subprocess.run([previous.original.BASH,'run.sh'],cwd=root,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertIn('TRAIN EVAL=0 PACKAGE=0 WORK=./original UNTIL=15',r.stdout)
            self.assertIn('REEVAL WORK=./original UNTIL=15',r.stdout)
            env['FAIL_TRAIN']='1'
            r=subprocess.run([previous.original.BASH,'run.sh'],cwd=root,env=env,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(r.returncode,0); self.assertNotIn('REEVAL WORK=',r.stdout)

    def run_script(self,fail='NEVER',**flags):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); shutil.copyfile(repair.REPO/'run_repair_text_evaluation.sh',root/'run.sh')
            (root/'original').mkdir(); (root/'original/.run.lock').touch()
            (root/'fake_python').write_bytes(b'#!/usr/bin/env bash\necho "ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]] || exit 1\nexit 0\n')
            (root/'flock').write_bytes(b'#!/usr/bin/env bash\nexit 0\n')
            for name in ('fake_python','flock'): (root/name).chmod(0o755)
            env=os.environ.copy(); env.update(TRAIN_PYTHON='./fake_python',WORK_DIR='./original',REEVAL_DIR='./new',FAIL_STAGE=fail)
            env.update(flags); env['PATH']=str(root)+os.pathsep+str(Path(previous.original.BASH).parent)+os.pathsep+env.get('PATH','')
            return subprocess.run([previous.original.BASH,'run.sh'],cwd=root,env=env,capture_output=True,text=True,timeout=45)

    def test_cpu_recovery_orders_audit_before_metrics_and_never_trains_or_predicts(self):
        r=self.run_script(); self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertLess(r.stdout.index('--stage prepare'),r.stdout.index('--stage audit'))
        self.assertLess(r.stdout.index('--stage audit'),r.stdout.index('--stage evaluate'))
        self.assertNotIn('--stage train ',r.stdout); self.assertNotIn('--stage predict ',r.stdout)
        self.assertIn('--publish_report',r.stdout)
        r=self.run_script(RUN_EVAL='0'); self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertNotIn('--stage evaluate',r.stdout); self.assertNotIn('--stage report',r.stdout)
        self.assertIn('--stage audit',r.stdout); self.assertIn('--stage package',r.stdout)

    def test_failed_worker_stops_report(self):
        r=self.run_script(fail='--stage evaluate'); self.assertNotEqual(r.returncode,0)
        self.assertNotIn('--stage report',r.stdout)
        self.assertNotEqual(self.run_script(EVAL_WORKERS='0').returncode,0)


if __name__=='__main__': unittest.main()
