import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import upstream_semantic_audit as audit
import upstream_repair as repair
import test_upstream_repair as repair_tests
from tools.upstream_semantic_audit import component_rows, intervene, pixel_regions, shuffled_semantics
from tools.vlm_review import digest, file_digest, load_json


class InterventionTests(unittest.TestCase):
    def test_known_text_sensitive_and_text_blind_heads(self):
        torch.set_num_threads(1)
        torch.manual_seed(6)
        model = repair.ConstrainedHead(2, 8, 8).eval()
        batch = dict(visual=torch.randn(1, 2, 8, 4, 4), base=torch.full((1, 1, 16, 16), .1),
                     embeddings=torch.randn(1, 9, 2, 8), valid=torch.ones(1, 9, 2),
                     boxes=torch.tensor(repair.old.tile_boxes())[None])
        with torch.no_grad():
            model.decode[-1].weight.normal_(0, .5)
            zero, mode = intervene(batch, 'zero_text')
            real = repair.old.head_forward(model, batch, 'vlm')
            changed = repair.old.head_forward(model, zero, mode)
            self.assertGreater(float((real-changed).abs().max()), 1e-6)
            # A head that discards the text channels can execute the semantic
            # module while being perfectly insensitive to its content.
            model.fuse[0].weight[:, 8:8+32].zero_()
            real = repair.old.head_forward(model, batch, 'vlm')
            changed = repair.old.head_forward(model, zero, mode)
            self.assertTrue(torch.equal(real, changed))

    def test_other_image_donors_keep_roles_and_missing_coverage(self):
        e = np.arange(3*2*2*4).reshape(3, 2, 2, 4).astype(np.float32)
        valid = np.ones((3, 2, 2), np.float32)
        valid[1:, 0, 0] = 0  # same-tile donor unavailable; same-role fallback
        before = valid.copy()
        changed, stats, donors = shuffled_semantics(e, valid, ['a', 'b', 'c'], 42)
        self.assertTrue(np.array_equal(before, valid))
        self.assertTrue(np.array_equal(changed[valid == 0], e[valid == 0]))
        self.assertEqual(sum(x['unavailable_slots'] for x in stats), 0)
        for row in donors:
            self.assertNotEqual(row['image_id'], row['donor_id'])
            i, j = 'abc'.index(row['image_id']), 'abc'.index(row['donor_id'])
            self.assertTrue(np.array_equal(changed[i, row['tile'], row['role']], e[j, row['donor_tile'], row['role']]))
        again = shuffled_semantics(e, valid, ['a', 'b', 'c'], 42)
        self.assertTrue(np.array_equal(changed, again[0]))
        one, summary, donors = shuffled_semantics(e[:1], valid[:1], ['a'], 42)
        self.assertTrue(np.array_equal(one, e[:1]))
        self.assertEqual(summary[0]['unavailable_slots'], 4)
        self.assertEqual(donors, [])

    def test_zero_content_preserves_coverage_and_branch_off_removes_it(self):
        batch = {'embeddings': torch.ones(1, 2, 2, 8), 'valid': torch.ones(1, 2, 2)}
        out, mode = intervene(batch, 'zero_text')
        self.assertEqual(mode, 'vlm')
        self.assertEqual(out['embeddings'].sum().item(), 0)
        self.assertTrue(torch.equal(out['valid'], batch['valid']))
        self.assertEqual(intervene(batch, 'branch_off')[1], 'visual')
        self.assertGreater(batch['embeddings'].sum().item(), 0)

    def test_components_and_near_background_are_not_confused(self):
        gt = np.zeros((64, 64), bool)
        gt[10:12, 10:12] = True
        gt[30:35, 30:35] = True
        base = np.full(gt.shape, 1e-7, np.float32)
        prob = base.copy()
        prob[10:12, 10:12] = .9
        rows = component_rows(gt, base, prob, .5, .5)
        self.assertTrue(rows[0]['small'])
        self.assertTrue(rows[0]['recovered_at_05'])
        self.assertFalse(rows[1]['small'])
        areas = {r['region']: r for r in pixel_regions(gt, base, prob, prob*0, 2)}
        self.assertEqual(areas['small_gt']['pixels'], 4)
        self.assertEqual(areas['large_gt']['pixels'], 25)
        self.assertEqual(areas['low_base_gt']['pixels'], 29)
        self.assertEqual(sum(areas[k]['pixels'] for k in ('small_gt', 'large_gt', 'near_background', 'far_background')), gt.size)


class PipelineTests(unittest.TestCase):
    review = repair_tests.PipelineTests.review

    def setUp(self):
        repair_tests.PipelineTests.setUp(self)
        self.repair_root = self.out
        self.audit_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.audit_temp.cleanup)
        self.audit_root = Path(self.audit_temp.name)
        self.args = audit.parser().parse_args(['--stage', 'prepare', '--repair_dir', str(self.repair_root),
                                               '--work_dir', str(self.audit_root), '--num_shards', '1',
                                               '--batch_size', '1', '--workers', '0', '--visuals_per_category', '0'])
        settings = load_json(self.repair_root/'repair_config.json')
        for mode in ('visual', 'vlm'):
            torch.manual_seed(19)
            model = repair.ConstrainedHead(2, 8, 8)
            with torch.no_grad():
                model.decode[-1].weight.normal_(0, .3)
            payload = dict(state_dict=model.state_dict(), fingerprint=digest([settings, mode]),
                           epoch=0, alpha=.1, selection='SOURCE_VALIDATED', shape=[2, 8, 4, 4])
            repair.old.save_torch(self.repair_root/f'heads/{mode}/best.pt', payload)
            repair.old.save_torch(self.repair_root/f'heads/{mode}/last.pt', dict(payload, epoch=1))

    def test_paired_audit_detects_semantics_without_modifying_or_training(self):
        snapshot = {str(p): file_digest(p) for root in (self.source, self.repair_root) for p in root.rglob('*') if p.is_file()}
        with patch('torch.cuda.is_available', return_value=False), patch.object(repair, 'train', side_effect=AssertionError('No training')):
            audit.prepare(self.args)
            audit.evaluate(self.args)
            audit.report(self.args)
            before = {str(p): file_digest(p) for p in (self.audit_root/'groups').glob('*.json')}
            audit.evaluate(self.args)
            self.assertEqual(before, {str(p): file_digest(p) for p in (self.audit_root/'groups').glob('*.json')})
        result = load_json(self.audit_root/'summary.json')
        self.assertEqual(len(result['metrics']), 20)
        self.assertTrue((self.audit_root/'comparisons.csv').exists())
        self.assertTrue(all(r['visual_text_invariant'] and r['alpha_zero_identity'] for r in result['controls']))
        for r in result['participation']:
            if r['intervention'] == 'shuffled_text':
                self.assertGreater(r['images_with_changed_shuffle'], 0)
                self.assertGreater(r['images_raw_affected'], 0)
                self.assertGreater(r['images_pixels_affected'], 0)
        self.assertEqual(snapshot, {str(p): file_digest(p) for root in (self.source, self.repair_root) for p in root.rglob('*') if p.is_file()})
        # Completed results cannot be silently reused with changed settings.
        self.args.batch_size = 2
        with self.assertRaisesRegex(ValueError, 'NEW WORK_DIR'):
            audit.prepare(self.args)

    def test_alpha_zero_reports_raw_influence_but_no_pixel_change(self):
        path = self.repair_root/'heads/vlm/best.pt'
        saved = torch.load(path, weights_only=False)
        saved['alpha'] = 0.
        saved['selection'] = 'BASE_FALLBACK'
        repair.old.save_torch(path, saved)
        self.args.checkpoints = 'best'
        self.args.partitions = 'val'
        with patch('torch.cuda.is_available', return_value=False):
            audit.prepare(self.args)
            audit.evaluate(self.args)
            audit.report(self.args)
        rows = load_json(self.audit_root/'summary.json')['participation']
        shuffled = next(r for r in rows if r['intervention'] == 'shuffled_text')
        self.assertGreater(shuffled['images_raw_affected'], 0)
        self.assertEqual(shuffled['images_pixels_affected'], 0)
        self.assertTrue(all(r['images_changed_from_base'] == 0 for r in rows))

    def test_nested_paths_and_checkpoint_mutation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-nested'):
            audit.non_nested([self.repair_root, self.repair_root/'audit', self.source])
        audit.prepare(self.args)
        with (self.repair_root/'heads/vlm/last.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
            audit.context(self.args)


BASH = os.environ.get('SRA_TEST_BASH') or (shutil.which('bash') if os.name != 'nt' else None)


@unittest.skipUnless(BASH, 'Set SRA_TEST_BASH on Windows')
class ShellTests(unittest.TestCase):
    def test_two_gpu_sharding_and_worker_failure_stops_report(self):
        for failure in ('NEVER', '--stage evaluate'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copyfile(repair.old.REPO/'run_exp_upstream_semantic_audit.sh', root/'run.sh')
                (root/'repair').mkdir()
                (root/'repair/repair_config.json').write_text('{}')
                (root/'fake_python').write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n', newline='\n')
                (root/'flock').write_text('#!/usr/bin/env bash\nexit 0\n', newline='\n')
                for p in (root/'fake_python', root/'flock'):
                    p.chmod(0o755)
                env = dict(os.environ, REPAIR_WORK_DIR='./repair', WORK_DIR='./out', TRAIN_PYTHON='./fake_python', GPU_IDS='0,1', FAIL_STAGE=failure)
                env['PATH'] = str(root)+os.pathsep+str(Path(BASH).parent)+os.pathsep+env.get('PATH', '')
                result = subprocess.run([BASH, 'run.sh'], cwd=root, env=env, text=True, capture_output=True, timeout=30)
                if failure == 'NEVER':
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('GPU=0 ARGS=upstream_semantic_audit.py --stage evaluate', result.stdout)
                    self.assertIn('GPU=1 ARGS=upstream_semantic_audit.py --stage evaluate', result.stdout)
                    self.assertIn('--stage report', result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('--stage report', result.stdout)


if __name__ == '__main__':
    unittest.main()
