#!/usr/bin/env python3
"""Post-R2 paired reference audit + full-source offline correction diagnosis.

No training. Source directories are immutable. prepare/review NEVER open GT.
Old source code hashes are provenance, not a requirement to run old code.
"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from tools.local_vlm_review import action, component_counts, parse_local
from tools.vlm_review import atomic_json, digest, file_digest, load_json
from tools.vlm_next_diagnostics import (ARMS, audit_prompt, parse_audit, paired_selection,
                                        correction, map_metrics)

PROTOCOL = 'post_r2_reference_action_diagnosis_v1'
CODE_FILES = ('vlm_next_diagnose.py', 'tools/vlm_next_diagnostics.py', 'tools/vlm_decision.py',
              'tools/local_vlm_protocol.py', 'tools/local_vlm_review.py', 'tools/vlm_review.py', 'dual_vlm.py')
MODES = ('base', 'vlm_cached', 'all_candidates', 'gt_candidate_DIAGNOSTIC_ONLY', 'gt_pixel_DIAGNOSTIC_ONLY')


def code_hashes():
    root = Path(__file__).resolve().parent
    return {p: file_digest(root / p) for p in CODE_FILES}


def safe_asset(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f'Missing/out-of-root asset: {relative}')
    return path


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows for {path.name}')
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def candidate_key(record, candidate):
    return f"{record['id']}_{candidate['roi_id']}"


def source_vote(src, cfg, record, candidate):
    name = f"local_reviews/{candidate_key(record, candidate)}.json"
    payload = load_json(safe_asset(src, name))
    if (payload.get('fingerprint') != digest(cfg) or payload.get('id') != record['id']
            or payload.get('candidate_id') != candidate['roi_id']):
        raise ValueError(f'Mismatched source decision: {name}')
    raw = payload['raw_responses'][-1]
    parsed = parse_local(raw, candidate['roi_id'], 'reference' in candidate,
                         policy=cfg.get('local_parser', 'legacy'))
    if parsed != payload['decision'] or not parsed['parse_ok']:
        raise ValueError(f'Invalid/inconsistent source decision: {name}; finish R2 review first')
    return parsed, name


def prepare(args):
    from dual_vlm import model_signature
    src, out = Path(args.source_dir).resolve(), Path(args.work_dir).resolve()
    if src == out or src.is_relative_to(out) or out.is_relative_to(src):
        raise ValueError('Source and diagnostic output must be separate, non-overlapping directories')
    if args.candidates < 0 or args.num_shards < 1 or args.retries < 0 or not 0 < args.gpu_memory < 1:
        raise ValueError('Invalid count/shards/runtime settings')
    if not 0 <= args.max_invalid_ratio <= 1 or not 0 < args.max_tokens < args.max_model_len:
        raise ValueError('Invalid review budget')
    alphas = sorted(set(float(x) for x in args.alphas.split(',')))
    if not alphas or not all(np.isfinite(x) and 0 < x <= 2 for x in alphas):
        raise ValueError('Alphas must be finite, in (0,2]')
    original = load_json(src / 'config.json')
    manifest = load_json(src / 'manifest.json')
    if not original.get('local_review') or manifest['fingerprint'] != digest(original):
        raise ValueError('SOURCE_WORK_DIR must be a sealed local-VLM export, normally R2_context_reference')
    if not 1024 <= args.min_pixels <= original['teacher_image_size']**2:
        raise ValueError('Invalid minimum visual pixel budget')
    records, pool, assets = manifest['records'], [], {}
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Empty or duplicate source records')
    for name in ('config.json', 'manifest.json'):
        assets[name] = file_digest(src / name)
    for r in records:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']):
            raise ValueError('Unsafe source sample ID')
        if r['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Mixed source fingerprints')
        for name in ('evidence', 'supports'):
            assets[r[name]] = file_digest(safe_asset(src, r[name]))
        if [c['roi_id'] for c in r['rois']] != list(range(1, len(r['rois'])+1)):
            raise ValueError('Candidate IDs must correspond to support-mask order')
        for c in r['rois']:
            _, name = source_vote(src, original, r, c)
            assets[name] = file_digest(safe_asset(src, name))
            for field in ('context', 'detail', 'reference'):
                if field in c:
                    assets[c[field]] = file_digest(safe_asset(src, c[field]))
            if 'reference' in c and c.get('reference_source_sha256'):
                pool.append(dict(key=candidate_key(r,c), sample_id=r['id'], candidate_id=c['roi_id'],
                                 dataset=r['dataset'], category=r['category'], area=c['area'],
                                 context=c['context'], detail=c['detail'], reference=c['reference'],
                                 reference_sha256=assets[c['reference']],
                                 reference_source_sha256=c['reference_source_sha256'],
                                 geometry=c.get('geometry', {})))
    selected, eligible = paired_selection(pool, args.candidates, args.seed)
    if not selected:
        raise ValueError('No pairable normal references; use complete R2 output with reference metadata')
    cfg = dict(protocol=PROTOCOL, source_dir=str(src), original=original, records=records,
               source_assets=assets, code_hashes=code_hashes(), selected=selected,
               source_reference_candidates=len(pool), pairable_candidates=eligible,
               requested_candidates=args.candidates, seed=args.seed, num_shards=args.num_shards,
               min_pixels=args.min_pixels, gpu_memory=args.gpu_memory,
               max_tokens=args.max_tokens, max_model_len=args.max_model_len, retries=args.retries,
               max_invalid_ratio=args.max_invalid_ratio, alphas=alphas,
               model=model_signature(args.model_id or original['model']['path']),
               notes='P0 selected paired candidates only; retrieved is NOT guaranteed aligned. '
                     'P1 evaluates ALL source images/candidates using cached R2 decisions. '
                     'GT is evaluation-only. Never choose deployment alpha using these test results.')
    path = out / 'next_config.json'
    if path.exists() and load_json(path) != cfg:
        raise ValueError('Settings/source/code changed. Use NEW WORK_DIR; existing output preserved')
    if path.exists():
        print('Prepare unchanged; reusing immutable configuration', flush=True)
        return
    atomic_json(path, cfg)
    write_csv(out / 'selection.csv', [{k:v for k,v in x.items() if k != 'geometry'} for x in selected])
    print(f'Prepared P0: {len(selected)} paired candidates / {eligible} pairable / {len(pool)} reference candidates. '
          f'P1: ALL {len(records)} source images. No GT opened.', flush=True)


def load_config(args):
    cfg = load_json(Path(args.work_dir) / 'next_config.json')
    if cfg['protocol'] != PROTOCOL or cfg['code_hashes'] != code_hashes():
        raise ValueError('Diagnostic code changed; use NEW WORK_DIR')
    for name, expected in cfg['source_assets'].items():
        if file_digest(safe_asset(cfg['source_dir'], name)) != expected:
            raise ValueError(f'Source changed: {name}')
    return cfg


def image_paths(item, arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    paths = [item['context'], item['detail']]
    if arm != ARMS[0]:
        paths.append(item['reference'] if arm == ARMS[1] else item['shuffled_reference'])
    return paths


def validate_review(path, cfg, item, arm):
    payload = load_json(path)
    if payload.get('fingerprint') != digest(cfg) or payload.get('key') != item['key'] or payload.get('arm') != arm:
        raise ValueError(f'Mismatched review cache: {path}')
    expected_prompt = audit_prompt(item['category'], item['candidate_id'], arm != ARMS[0])
    if payload.get('prompt_sha256') != digest(expected_prompt):
        raise ValueError('Cached prompt mismatch')
    parsed, audit = parse_audit(payload['raw_responses'][-1], item['candidate_id'], arm != ARMS[0])
    if parsed != payload['decision'] or audit != payload['audit']:
        raise ValueError('Cached decision differs from final raw response')
    if not payload.get('trace_hashes'):
        raise ValueError('Missing input trace')
    for name, expected in payload['trace_hashes'].items():
        if file_digest(safe_asset(path.parent, name)) != expected:
            raise ValueError('Changed trace; do not mix caches')
    return payload


def review(args):
    from dual_vlm import model_signature
    from tools.vlm_decision import QwenVLLMTeacher
    cfg, out = load_config(args), Path(args.work_dir)
    if not 0 <= args.shard_id < cfg['num_shards']:
        raise ValueError('Invalid shard')
    if model_signature(cfg['model']['path']) != cfg['model']:
        raise ValueError('VLM model changed')
    pending = []
    for item in cfg['selected'][args.shard_id::cfg['num_shards']]:
        for arm in ARMS:
            path = out / arm / f"{item['key']}.json"
            if path.exists() and validate_review(path,cfg,item,arm)['decision']['parse_ok']:
                continue
            pending.append((item,arm,path))
    if not pending:
        print(f'Shard {args.shard_id}: all reviews cached', flush=True)
        return
    teacher = QwenVLLMTeacher(model_id=cfg['model']['path'], gpu_memory_utilization=cfg['gpu_memory'],
                             max_model_len=cfg['max_model_len'], max_tokens=cfg['max_tokens'], max_images=3,
                             teacher_image_size=cfg['original']['teacher_image_size'], min_pixels=cfg['min_pixels'])
    for i,(item,arm,path) in enumerate(pending):
        images, image_audit = [], []
        for name in image_paths(item,arm):
            with Image.open(safe_asset(cfg['source_dir'],name)) as im:
                images.append(im.convert('RGB'))
            image_audit.append(dict(asset=name,sha256=cfg['source_assets'][name],size=list(images[-1].size)))
        prompt = audit_prompt(item['category'],item['candidate_id'],arm != ARMS[0])
        started, responses, hashes = time.monotonic(), [], {}
        for attempt in range(cfg['retries']+1):
            trace_dir = path.parent / 'traces' / item['key'] / str(attempt)
            raw = teacher.generate(prompt + (' Previous JSON was invalid; return all nine fields.' if attempt else ''),
                                   images,trace_dir=trace_dir)
            if not (trace_dir/'trace.json').is_file():
                raise ValueError('VLM did not produce mandatory request trace')
            trace=load_json(trace_dir/'trace.json')
            if trace.get('raw_response')!=raw or trace.get('input_sizes')!=[list(im.size) for im in images]:
                raise ValueError('Request trace does not match supplied images/response')
            responses.append(raw)
            for p in trace_dir.iterdir():
                if p.is_file():
                    hashes[str(p.relative_to(path.parent))]=file_digest(p)
            decision,audit = parse_audit(raw,item['candidate_id'],arm != ARMS[0])
            if decision['parse_ok']:
                break
        atomic_json(path,dict(fingerprint=digest(cfg),key=item['key'],arm=arm,decision=decision,audit=audit,
                             raw_responses=responses,prompt_sha256=digest(prompt),input_images=image_audit,
                             trace_hashes=hashes,seconds=time.monotonic()-started))
        print(f"[{args.shard_id}] {i+1}/{len(pending)} {arm} {decision['verdict']} valid={decision['parse_ok']}",flush=True)


def evaluate_reference(args):
    cfg,out = load_config(args),Path(args.work_dir)
    # Validate ALL responses before opening ANY GT.
    cached={}
    for item in cfg['selected']:
        for arm in ARMS:
            cached[(item['key'],arm)] = validate_review(out/arm/f"{item['key']}.json",cfg,item,arm)
    for arm in ARMS:
        invalid=sum(not cached[(x['key'],arm)]['decision']['parse_ok'] for x in cfg['selected'])
        if invalid/len(cfg['selected']) > cfg['max_invalid_ratio']:
            raise ValueError(f'{arm}: too many invalid responses ({invalid}); resume review')
    records={r['id']:r for r in cfg['records']}
    labels,eval_hashes={},{}
    for ident in {x['sample_id'] for x in cfg['selected']}:
        r=records[ident]
        path=safe_asset(cfg['source_dir'],r['evaluation'])
        eval_hashes[r['evaluation']]=file_digest(path)
        with np.load(path,allow_pickle=False) as z:
            gt=z['mask']>0
        with np.load(safe_asset(cfg['source_dir'],r['supports']),allow_pickle=False) as z:
            supports=z['masks'].astype(bool)
        for c,s in zip(r['rois'],supports):
            labels[candidate_key(r,c)] = bool(gt[s].any())
    rows, summaries, paired = [], [], []
    for arm in ARMS:
        counts=Counter()
        for item in cfg['selected']:
            payload=cached[(item['key'],arm)]
            d,a=payload['decision'],payload['audit']
            gt=labels[item['key']]
            choice=action(d)
            counts.update(total=1,gt_positive=int(gt),valid=int(d['parse_ok']),enhance=int(choice=='enhance'),
                          true_enhance=int(gt and choice=='enhance'),false_enhance=int(not gt and choice=='enhance'),
                          keep=int(choice=='keep'),normal_on_gt=int(gt and choice=='suppress'),
                          reference_matched=int(d['reference_match']=='matched'),
                          reference_unavailable=int(d['reference_match']=='unavailable'),
                          semantic_warnings=int(bool(a['semantic_warnings'])))
            rows.append(dict(arm=arm,key=item['key'],category=item['category'],area=item['area'],
                             gt_contains_defect=gt,has_reference=arm!=ARMS[0],action=choice,**d,
                             **a['observations'],parse_errors=';'.join(a['parse_errors']),
                             semantic_warnings=';'.join(a['semantic_warnings'])))
        summaries.append(dict(arm=arm,**counts,precision=counts['true_enhance']/counts['enhance'] if counts['enhance'] else None,
                              recall=counts['true_enhance']/counts['gt_positive'] if counts['gt_positive'] else None))
    for item in cfg['selected']:
        paired.append(dict(key=item['key'],category=item['category'],gt_contains_defect=labels[item['key']],
                           **{arm:action(cached[(item['key'],arm)]['decision']) for arm in ARMS}))
    target=out/'reference_results'
    write_csv(target/'candidate_decisions.csv',rows)
    write_csv(target/'summary.csv',summaries)
    write_csv(target/'paired_transitions.csv',paired)
    atomic_json(target/'summary.json',dict(summaries=summaries,evaluation_hashes=eval_hashes,
                selected=len(cfg['selected']),source_reference_candidates=cfg['source_reference_candidates'],
                pairable_candidates=cfg['pairable_candidates'],note=cfg['notes']))
    print(json_summary(summaries),flush=True)


def json_summary(value):
    import json
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)


def evaluate_action(args):
    """Full-source offline selector/action isolation. Does not instantiate a VLM."""
    cfg,out=load_config(args),Path(args.work_dir)
    groups=defaultdict(list)
    for r in cfg['records']:
        groups[(r['dataset'],r['category'])].append(r)
    rows,pixels,eval_hashes=[],[],{}
    for (dataset,category),records in sorted(groups.items()):
        items=[]
        for r in records:
            with np.load(safe_asset(cfg['source_dir'],r['evidence']),allow_pickle=False) as z:
                base=z['base'].copy()
            with np.load(safe_asset(cfg['source_dir'],r['supports']),allow_pickle=False) as z:
                supports=z['masks'].astype(bool)
            votes=[source_vote(cfg['source_dir'],cfg['original'],r,c)[0] for c in r['rois']]
            path=safe_asset(cfg['source_dir'],r['evaluation'])
            eval_hashes[r['evaluation']]=file_digest(path)
            with np.load(path,allow_pickle=False) as z:
                gt=z['mask']>0
            if supports.shape != (len(votes),*base.shape) or gt.shape!=base.shape or np.any(supports.sum(axis=0)>1):
                raise ValueError('Mismatched or overlapping supports/GT')
            items.append((base,supports,votes,gt))
        masks=np.stack([x[3] for x in items])
        for alpha in cfg['alphas']:
            for mode in MODES:
                if mode=='base' and alpha!=cfg['alphas'][0]:
                    continue
                maps,stats=[],Counter()
                for base,supports,votes,gt in items:
                    updated=correction(base,supports,votes,gt,mode,alpha)
                    union=supports.any(axis=0)
                    changed=updated!=base
                    if np.any(changed & ~union):
                        raise AssertionError('Correction escaped candidate support')
                    stats.update(pixels=base.size,gt_pixels=int(gt.sum()),candidate_pixels=int(union.sum()),
                                 covered_gt_pixels=int((union & gt).sum()),changed_pixels=int(changed.sum()),
                                 changed_gt_pixels=int((changed & gt).sum()),changed_bg_pixels=int((changed & ~gt).sum()),
                                 newly_detected_gt_pixels=int(((base<.5)&(updated>=.5)&gt).sum()),
                                 new_fp_pixels=int(((base<.5)&(updated>=.5)&~gt).sum()),
                                 new_fn_pixels=int(((base>=.5)&(updated<.5)&gt).sum()))
                    stats.update(component_counts(gt,union,base,updated))
                    maps.append(updated)
                label_alpha=0 if mode=='base' else alpha
                row=dict(dataset=dataset,category=category,samples=len(records),mode=mode,alpha=label_alpha,
                         **map_metrics(masks,np.stack(maps),cfg['original']['pro_max_fpr']))
                rows.append(row)
                pixels.append(dict(dataset=dataset,category=category,mode=mode,alpha=label_alpha,**stats))
                print(f"[action] {category} {mode} alpha={label_alpha}: PRO_exact={row['PRO_exact']}",flush=True)
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['dataset'],r['mode'],r['alpha'])].append(r)
    means=[]
    for (dataset,mode,alpha),values in grouped.items():
        result=dict(dataset=dataset,mode=mode,alpha=alpha,categories=len(values))
        for metric in ('PRO_exact','P_AUROC','I_AUROC','F1_best'):
            available=[v[metric] for v in values if v[metric] is not None]
            result[metric]=float(np.mean(available)) if available else None
            result[metric+'_valid_categories']=len(available)
        means.append(result)
    target=out/'action_results'
    write_csv(target/'metrics.csv',rows)
    write_csv(target/'summary.csv',means)
    write_csv(target/'pixel_effects.csv',pixels)
    pixel_groups=defaultdict(Counter)
    for r in pixels:
        pixel_groups[(r['dataset'],r['mode'],r['alpha'])].update(
            {k:v for k,v in r.items() if k not in ('dataset','category','mode','alpha')})
    pixel_summary=[]
    for (dataset,mode,alpha),counts in pixel_groups.items():
        row=dict(dataset=dataset,mode=mode,alpha=alpha,**counts)
        for output,numerator,denominator in (
                ('candidate_gt_coverage','covered_gt_pixels','gt_pixels'),
                ('changed_pixel_gt_fraction','changed_gt_pixels','changed_pixels'),
                ('small_candidate_recall','small_candidate_hit','small_count'),
                ('small_base_recall','small_base_hit','small_count'),
                ('small_updated_recall','small_updated_hit','small_count')):
            row[output]=counts[numerator]/counts[denominator] if counts[denominator] else None
        row['small_new_hits']=counts['small_updated_hit']-counts['small_base_hit']
        pixel_summary.append(row)
    write_csv(target/'pixel_summary.csv',pixel_summary)
    atomic_json(target/'summary.json',dict(means=means,evaluation_hashes=eval_hashes,note=cfg['notes'],
                metric_note='PRO_exact uses all score thresholds; do NOT compare absolute values directly to historical sampled PRO. '
                            'GT candidate means any overlap, not perfect segmentation. GT pixel is support-restricted, not a deployable result. '
                            'F1_best uses test thresholds for reporting only.'))
    print(json_summary(means),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','review','reference_eval','action_eval'),required=True)
    p.add_argument('--source_dir',default='')
    p.add_argument('--work_dir',required=True)
    p.add_argument('--model_id',default='')
    p.add_argument('--candidates',type=int,default=192)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num_shards',type=int,default=2)
    p.add_argument('--shard_id',type=int,default=0)
    p.add_argument('--min_pixels',type=int,default=65536)
    p.add_argument('--gpu_memory',type=float,default=.70)
    p.add_argument('--max_tokens',type=int,default=768)
    p.add_argument('--max_model_len',type=int,default=4096)
    p.add_argument('--retries',type=int,default=1)
    p.add_argument('--max_invalid_ratio',type=float,default=.05)
    p.add_argument('--alphas',default='0.25')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    dict(prepare=prepare,review=review,reference_eval=evaluate_reference,action_eval=evaluate_action)[args.stage](args)
