#!/usr/bin/env python3
"""Read-only P0/P1 checkpoint audit: does local semantic content affect pixels?"""
import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import upstream_repair as repair
from tools.upstream_repair import ConstrainedHead, corrected_probability, source_metrics
from tools.upstream_semantic_audit import (
    ARMS, component_rows, difference, intervene, pixel_regions, save_panel, shuffled_semantics,
)
from tools.vlm_review import atomic_json, digest, file_digest, load_json

FILES = ('upstream_semantic_audit.py', 'tools/upstream_semantic_audit.py', 'run_exp_upstream_semantic_audit.sh')


def non_nested(paths):
    paths = [Path(p).resolve() for p in paths]
    for i, x in enumerate(paths):
        for y in paths[i+1:]:
            if x == y or x.is_relative_to(y) or y.is_relative_to(x):
                raise ValueError('Audit, repair and source must be separate non-nested directories')


def inputs(repair_dir):
    return repair.context(SimpleNamespace(work_dir=str(repair_dir)))


def prepare(a):
    repair_root, source, cfg, settings = inputs(a.repair_dir)
    root = Path(a.work_dir).resolve()
    non_nested([root, repair_root, source])
    if (a.num_shards < 1 or a.batch_size < 1 or a.visuals_per_category < 0 or a.near_radius < 0
            or a.pixel_epsilon <= 0 or a.residual_epsilon <= 0):
        raise ValueError('Invalid audit settings')
    checkpoints = list(dict.fromkeys(a.checkpoints.split(',')))
    partitions = list(dict.fromkeys(a.partitions.split(',')))
    if not checkpoints or set(checkpoints)-{'best', 'last'} or not partitions or set(partitions)-{'val', 'eval'}:
        raise ValueError('checkpoints: best,last; partitions: val,eval')
    paths = [repair_root/'heads/vlm'/f'{k}.pt' for k in set(checkpoints)|{'best'}]
    paths.append(repair_root/'heads/visual/best.pt')
    value = dict(protocol='upstream_semantic_audit_v1', repair_dir=str(repair_root),
                 repair_config_sha=file_digest(repair_root/'repair_config.json'),
                 checkpoints=checkpoints, partitions=partitions,
                 heads={str(p.relative_to(repair_root)): file_digest(p) for p in sorted(paths)},
                 code={p: file_digest(repair.old.REPO/p) for p in FILES},
                 **{k: getattr(a, k) for k in ('num_shards', 'batch_size', 'visuals_per_category',
                                              'near_radius', 'pixel_epsilon', 'residual_epsilon')})
    path = root/'audit_config.json'
    if path.exists() and load_json(path) != value:
        raise ValueError('Audit settings/checkpoints changed; use NEW WORK_DIR')
    if not path.exists() and root.exists() and any(p.name != '.run.lock' for p in root.iterdir()):
        raise ValueError('Output directory is not empty; use NEW WORK_DIR')
    rows = [r for r in repair.old.records_for(source) if r['partition'] in partitions]
    if not rows or set(partitions)-{r['partition'] for r in rows}:
        raise ValueError('Requested partition has no records')
    repair.old.FeatureDataset(source, cfg, rows, verify=True)
    atomic_json(path, value)
    print('Verified sealed inputs. No training, Qwen calls, alpha search or target selection.', flush=True)


def context(a):
    root = Path(a.work_dir).resolve()
    audit = load_json(root/'audit_config.json')
    repair_root, source, cfg, settings = inputs(audit['repair_dir'])
    non_nested([root, repair_root, source])
    if file_digest(repair_root/'repair_config.json') != audit['repair_config_sha']:
        raise ValueError('Repair config changed')
    for p, sha in audit['heads'].items():
        if file_digest(repair_root/p) != sha:
            raise ValueError('Checkpoint changed; use NEW WORK_DIR')
    for p, sha in audit['code'].items():
        if file_digest(repair.old.REPO/p) != sha:
            raise ValueError('Audit code changed; use NEW WORK_DIR')
    return root, audit, repair_root, source, cfg, settings


def load_head(path, mode, settings, shape, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['fingerprint'] != digest([settings, mode]):
        raise ValueError('Checkpoint fingerprint mismatch')
    if 'shape' in saved and list(saved['shape']) != list(shape):
        raise ValueError('Checkpoint shape mismatch')
    model = ConstrainedHead(shape[0], shape[1], settings['hidden']).to(device).eval()
    model.load_state_dict(saved['state_dict'])
    return model, saved


def group_path(root, key, checkpoint):
    # Hashed group names do not interpret dataset/category strings as paths.
    return root/'groups'/f'{digest([list(key), checkpoint])[:24]}.json'


def evaluate(a):
    from tools.vlm_next_diagnostics import map_metrics
    root, audit, repair_root, source, cfg, settings = context(a)
    if not 0 <= a.shard_id < audit['num_shards']:
        raise ValueError('Invalid shard ID')
    device = repair.device_for()
    shape = load_json(source/'sealed.json')['shape']
    visual, _ = load_head(repair_root/'heads/visual/best.pt', 'visual', settings, shape, device)
    best = torch.load(repair_root/'heads/vlm/best.pt', map_location='cpu', weights_only=False)
    if best['fingerprint'] != digest([settings, 'vlm']):
        raise ValueError('Selected VLM checkpoint fingerprint mismatch')
    alpha = float(best['alpha'])
    groups = list(repair.groups_for(source, set(audit['partitions'])).items())
    for key, records in groups[a.shard_id::audit['num_shards']]:
        seed = int(digest([cfg['seed'], list(key)])[:8], 16)
        bank, validity = [], []
        for r in records:
            with np.load(source/'features'/f"{r['id']}.npz") as f:
                bank.append(f['embeddings'].astype(np.float32))
                validity.append(f['valid'].astype(np.float32))
        shuffled, input_stats, donors = shuffled_semantics(bank, validity, [r['id'] for r in records], seed)
        # Deterministic, label-independent visualization sample.
        panel_ids = {r['id'] for r in sorted(records, key=lambda r: digest([seed, r['id']]))[:audit['visuals_per_category']]}
        for checkpoint in audit['checkpoints']:
            path = group_path(root, key, checkpoint)
            if path.exists():
                previous = load_json(path)
                if previous['fingerprint'] != digest(audit):
                    raise ValueError('Stale audit group')
                print(f'Resume: {key} {checkpoint} already complete', flush=True)
                continue
            model, saved = load_head(repair_root/'heads/vlm'/f'{checkpoint}.pt', 'vlm', settings, shape, device)
            maps, masks, images, regions = defaultdict(list), [], [], []
            calls = defaultdict(int)
            def hook(name):
                def count(*unused):
                    calls[name] += 1
                return count
            hooks = [getattr(model, k).register_forward_hook(hook(k)) for k in ('visual', 'semantic', 'fuse', 'decode')]
            batches = repair.loader(source, cfg, records, dict(settings, batch_size=audit['batch_size']), a.workers)
            offset = 0
            try:
                with torch.inference_mode():
                    for batch in batches:
                        gt = batch.pop('mask')[:, 0].numpy().astype(bool)
                        batch = {k: v.to(device) for k, v in batch.items()}
                        count = len(gt)
                        other = torch.from_numpy(shuffled[offset:offset+count]).to(device)
                        altered, _ = intervene(batch, 'shuffled_text', other)
                        visual_a = repair.old.head_forward(visual, batch, 'visual')
                        visual_b = repair.old.head_forward(visual, altered, 'visual')
                        if not torch.equal(visual_a, visual_b):
                            raise AssertionError('Pure visual negative control responds to text')
                        if not torch.equal(corrected_probability(batch['base'], visual_a, 0), batch['base']):
                            raise AssertionError('alpha=0 must reproduce Base exactly')
                        predictions, residuals = {}, {}
                        for arm in ARMS:
                            modified, mode = intervene(batch, arm, other)
                            delta = repair.old.head_forward(model, modified, mode)
                            prob = corrected_probability(batch['base'], delta, alpha)
                            if not torch.isfinite(delta).all() or not torch.isfinite(prob).all():
                                raise FloatingPointError('Nonfinite head output')
                            residuals[arm] = delta[:, 0].cpu().numpy()
                            predictions[arm] = prob[:, 0].cpu().numpy()
                        base = batch['base'][:, 0].cpu().numpy()
                        maps['base'].extend(base)
                        masks.extend(gt)
                        for arm in ARMS:
                            maps[arm].extend(predictions[arm])
                        for i in range(count):
                            r = records[offset+i]
                            meta = dict(partition=key[0], dataset=key[1], category=key[2], checkpoint=checkpoint,
                                        epoch=saved['epoch']+1, alpha=alpha, image_id=r['id'])
                            for arm in ARMS:
                                d, p = residuals[arm][i], predictions[arm][i]
                                active = np.asarray(validity[offset+i]) > 0
                                emb = bank[offset+i][active]
                                images.append(dict(**meta, arm=arm, **input_stats[offset+i],
                                                   valid_fraction=float(active.mean()),
                                                   real_embedding_abs_mean=float(np.abs(emb).mean()) if emb.size else 0.,
                                                   raw_residual_abs_mean=float(np.abs(d).mean()),
                                                   raw_residual_abs_max=float(np.abs(d).max()),
                                                   **difference(d, residuals['real'][i], 'raw_vs_real', audit['residual_epsilon']),
                                                   **difference(p, predictions['real'][i], 'prob_vs_real', audit['pixel_epsilon']),
                                                   **difference(p, base[i], 'prob_vs_base', audit['pixel_epsilon'])))
                                regions.extend(dict(**meta, arm=arm, **v) for v in pixel_regions(
                                    gt[i], base[i], p, d, audit['near_radius']))
                            if r['id'] in panel_ids:
                                name = digest([list(key), r['id']])[:20]+'.png'
                                save_panel(root/'panels'/checkpoint/name, r['image_path'], gt[i],
                                           dict(base=base[i], **{k: v[i] for k, v in predictions.items()}))
                        offset += count
                        print(f'Audit {"/".join(key)} {checkpoint}: {offset}/{len(records)}', flush=True)
            finally:
                for handle in hooks:
                    handle.remove()
            gt = np.asarray(masks)
            base = np.asarray(maps.pop('base'))
            metrics, components = [], []
            for arm in ('base',)+ARMS:
                prediction = base if arm == 'base' else np.asarray(maps.pop(arm))
                operating = source_metrics(gt, prediction, base, seed, settings['val_pixels'], settings['fpr'])
                metrics.append(dict(partition=key[0], dataset=key[1], category=key[2], checkpoint=checkpoint,
                                    epoch=saved['epoch']+1, alpha=alpha, arm=arm, samples=len(records),
                                    **map_metrics(gt, prediction),
                                    region_recall_at_fpr_DIAGNOSTIC=operating['region_recall_at_fpr'],
                                    small_hit_at_fpr_DIAGNOSTIC=operating['small_hit_at_fpr'],
                                    threshold_DIAGNOSTIC=operating['threshold'],
                                    realized_fpr_DIAGNOSTIC=operating['realized_fpr'],
                                    background_FPR_at_05=float(np.mean(prediction[~gt] >= .5))))
                for r, mask, b, p in zip(records, gt, base, prediction):
                    components.extend(dict(partition=key[0], dataset=key[1], category=key[2],
                                           checkpoint=checkpoint, arm=arm, image_id=r['id'], **v)
                                      for v in component_rows(mask, b, p, operating['threshold'], operating['base_threshold']))
            atomic_json(path, dict(fingerprint=digest(audit), key=list(key), checkpoint=checkpoint,
                                   alpha=alpha, best_selection=best['selection'], diagnostic_only=True,
                                   forward_calls=dict(calls), visual_text_invariant=True, alpha_zero_identity=True,
                                   donors=donors, images=images, metrics=metrics, components=components, regions=regions))
            print(f'Finished {key} {checkpoint}; alpha={alpha}; calls={dict(calls)}', flush=True)


def report(a):
    root, audit, _, source, _, _ = context(a)
    all_rows = {k: [] for k in ('images', 'metrics', 'components', 'regions')}
    controls, donors = [], []
    for key in repair.groups_for(source, set(audit['partitions'])):
        for checkpoint in audit['checkpoints']:
            value = load_json(group_path(root, key, checkpoint))
            if value['fingerprint'] != digest(audit) or value['key'] != list(key) or value['checkpoint'] != checkpoint:
                raise ValueError('Stale/mismatched audit results')
            for k in all_rows:
                all_rows[k].extend(value[k])
            controls.append({k: value[k] for k in ('key', 'checkpoint', 'forward_calls', 'visual_text_invariant', 'alpha_zero_identity')})
            donors.extend(dict(partition=key[0], dataset=key[1], category=key[2], checkpoint=checkpoint, **v) for v in value['donors'])
    means, participation = [], []
    for part, dataset, checkpoint in sorted({(r['partition'], r['dataset'], r['checkpoint']) for r in all_rows['metrics']}):
        for arm in ('base',)+ARMS:
            rows = [r for r in all_rows['metrics'] if (r['partition'], r['dataset'], r['checkpoint'], r['arm']) == (part, dataset, checkpoint, arm)]
            mean = dict(rows[0], category='MEAN', samples=sum(r['samples'] for r in rows))
            for name in rows[0]:
                if name in ('partition', 'dataset', 'category', 'checkpoint', 'epoch', 'alpha', 'arm', 'samples'):
                    continue
                values = [r[name] for r in rows if r[name] is not None]
                mean[name] = float(np.mean(values)) if values else None
            means.append(mean)
        for arm in ARMS:
            rows = [r for r in all_rows['images'] if (r['partition'], r['dataset'], r['checkpoint'], r['arm']) == (part, dataset, checkpoint, arm)]
            participation.append(dict(partition=part, dataset=dataset, checkpoint=checkpoint, intervention=arm,
                images=len(rows), images_with_valid_text=sum(r['valid_slots'] > 0 for r in rows),
                images_with_changed_shuffle=sum(r['changed_slots'] > 0 for r in rows),
                unavailable_donor_slots=sum(r['unavailable_slots'] for r in rows),
                images_raw_affected=sum(r['raw_vs_real_abs_max'] > audit['residual_epsilon'] for r in rows),
                images_pixels_affected=sum(r['prob_vs_real_abs_max'] > audit['pixel_epsilon'] for r in rows),
                images_changed_from_base=sum(r['prob_vs_base_abs_max'] > audit['pixel_epsilon'] for r in rows),
                mean_raw_difference=float(np.mean([r['raw_vs_real_abs_mean'] for r in rows])),
                mean_probability_difference=float(np.mean([r['prob_vs_real_abs_mean'] for r in rows]))))
    all_rows['metrics'].extend(means)
    comparisons = []
    grouped = defaultdict(dict)
    for row in all_rows['metrics']:
        grouped[(row['partition'], row['dataset'], row['category'], row['checkpoint'])][row['arm']] = row
    for key, arms in grouped.items():
        for control in ('base', 'zero_text', 'branch_off', 'shuffled_text'):
            row = dict(zip(('partition', 'dataset', 'category', 'checkpoint'), key))
            row['control'] = control
            for metric in ('PRO_exact', 'P_AUROC', 'I_AUROC', 'region_recall_at_fpr_DIAGNOSTIC', 'small_hit_at_fpr_DIAGNOSTIC'):
                x, y = arms['real'][metric], arms[control][metric]
                row['real_minus_control_'+metric] = None if x is None or y is None else x-y
            comparisons.append(row)
    for name, rows in all_rows.items():
        if rows:
            repair.write_csv(root/(name+'.csv'), rows)
    if donors:
        repair.write_csv(root/'donors.csv', donors)
    repair.write_csv(root/'participation.csv', participation)
    repair.write_csv(root/'comparisons.csv', comparisons)
    atomic_json(root/'summary.json', dict(diagnostic_only=True, metrics=means, participation=participation,
                                         controls=controls, configuration=audit))
    lines = ['# Semantic participation audit', '',
             'Inference interventions only. No training, checkpoint selection or alpha search.',
             'Both checkpoints use the source-selected BEST alpha. LAST is diagnostic only.',
             'zero_text preserves coverage; branch_off removes both text and coverage.',
             'shuffled_text uses other images in the same category/partition and preserves query coverage.',
             'A donor can have identical text: inspect changed_slots before interpreting a null result.',
             'All GT/FPR metrics and panels are offline diagnostics, not deployment thresholds.', '',
             '| Partition/dataset | Checkpoint | Intervention | Images | Raw vs real | Pixels vs real | Pixels vs Base |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in participation:
        lines.append(f"| {row['partition']}/{row['dataset']} | {row['checkpoint']} | {row['intervention']} | {row['images']} | {row['images_raw_affected']} | {row['images_pixels_affected']} | {row['images_changed_from_base']} |")
    lines.extend(['', '| Partition/dataset | Checkpoint | Control | Real minus control PRO (pp) |',
                  '|---|---|---|---:|'])
    for row in comparisons:
        if row['category'] == 'MEAN':
            value = row['real_minus_control_PRO_exact']
            shown = 'n.a.' if value is None else f'{100*value:+.6f}'
            lines.append(f"| {row['partition']}/{row['dataset']} | {row['checkpoint']} | {row['control']} | {shown} |")
    lines.extend(['', 'Raw/pixel affected compares each intervention with REAL at the same checkpoint/alpha.',
                  'REAL rows therefore have zero intervention difference; use images_changed_from_base for head activity.',
                  'Nonzero sensitivity establishes participation, not accuracy improvement or semantic correctness.',
                  'If raw changes but pixels do not, inspect alpha and numerical attenuation by Base.',
                  'If only branch_off changes output, coverage/bias may explain the response rather than text content.',
                  'If REAL does not beat SHUFFLED on source diagnostics, useful image-specific grounding is unproven.',
                  'Pixel region rows provide sums and counts; divide sums by pixels, excluding empty regions.',
                  'No changes to model design should be selected using target evaluation labels.'])
    (root/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'evaluate', 'report'), required=True)
    p.add_argument('--work_dir', required=True)
    p.add_argument('--repair_dir', default='./checkpoint/upstream_P0_P1_v1')
    p.add_argument('--checkpoints', default='best,last')
    p.add_argument('--partitions', default='val,eval')
    for k, v in dict(num_shards=2, shard_id=0, workers=2, batch_size=4, visuals_per_category=3, near_radius=8).items():
        p.add_argument('--'+k, type=int, default=v)
    p.add_argument('--pixel_epsilon', type=float, default=1e-6)
    p.add_argument('--residual_epsilon', type=float, default=1e-6)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    dict(prepare=prepare, evaluate=evaluate, report=report)[args.stage](args)
