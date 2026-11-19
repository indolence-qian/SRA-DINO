#!/usr/bin/env python3
"""Frozen T1/T2/T3 lambda sweep: source-only choices, exact positive-view fusion.

Exports V and A once from sealed residual features. Does not train, extract
DINO, change text calibration, or derive increments from rounded old maps.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

import text_alignment_ablation as app
import repair_text_evaluation as repair
from tools.text_lambda import (ARMS,DEFAULT_GRID,basis_maps,best_lambda,diagnostics,html_asset,
                              name_for,needs_extension,parse_grid,quality,readouts,score)
from tools.residual_text_runtime import atomic_npz
from tools.vlm_review import atomic_json,digest,file_digest,load_json

REPO=Path(__file__).resolve().parent
PROTOCOL="frozen_text_lambda_source_selection_v1"
CODE=("lambda_text_experiment.py","tools/text_lambda.py","run_exp_text_lambda.sh","asset/text_lambda_dashboard.html","repair_text_evaluation.py")
SECTION_START="<!-- LAMBDA_RESULTS_START -->"
SECTION_END="<!-- LAMBDA_RESULTS_END -->"


def context(cfg):
    text_root=Path(cfg['text_work_dir']); text=app.config(text_root)
    if file_digest(text_root/'config.json')!=cfg['text_config_sha']: raise ValueError('Text configuration changed')
    source,pcfg=app.parent(text)
    for seed in cfg['seeds']:
        if file_digest(app.decision_path(text_root,seed,cfg['horizon']))!=cfg['decisions'][str(seed)]:
            raise ValueError('Frozen source selection changed')
        app.decision(text_root,text,seed,cfg['horizon'])
    return text_root,text,source,pcfg


def config(root):
    cfg=load_json(Path(root)/'config.json')
    if cfg['protocol']!=PROTOCOL: raise ValueError('Wrong experiment directory')
    for name,sha in cfg['code'].items():
        if file_digest(REPO/name)!=sha: raise ValueError(f'Code changed: {name}; use NEW WORK_DIR')
    context(cfg)
    return cfg


def records(cfg,partition):
    _,_,source,pcfg=context(cfg)
    if partition=='eval': return app.f.rows(source,'eval')
    cal=set(pcfg['calibration_ids'])
    return [r for r in app.f.rows(source,'source') if r['partition']=='val' or r['id'] in cal]


def prepare(args):
    root,text_root=Path(args.work_dir).resolve(),Path(args.text_work_dir).resolve()
    text=app.config(text_root); source,pcfg=app.parent(text)
    if any(root==p or root in p.parents or p in root.parents for p in (text_root,source)):
        raise ValueError('Use a separate, non-nested WORK_DIR')
    grid=parse_grid(args.lambda_grid); seeds=sorted(int(s) for s in args.train_seeds.split(','))
    if (not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0 or args.num_shards<1
            or not 1<=args.until_epoch<=text['settings']['text_epochs'] or not 32<=args.html_map_side<=256):
        raise ValueError('Invalid seeds/horizon/shards/HTML size')
    decisions={}
    for seed in seeds:
        app.decision(text_root,text,seed,args.until_epoch)
        decisions[str(seed)]=file_digest(app.decision_path(text_root,seed,args.until_epoch))
    cal=set(pcfg['calibration_ids']); manifest=app.f.rows(source,'source')
    if not cal or {r['id'] for r in manifest if r['id'] in cal}!=cal or any(r['partition']!='train' or r['mask_path'] for r in manifest if r['id'] in cal):
        raise ValueError('Source CAL must contain only held-out normal TRAIN queries')
    cfg=dict(protocol=PROTOCOL,text_work_dir=str(text_root),text_config_sha=file_digest(text_root/'config.json'),
        horizon=args.until_epoch,seeds=seeds,decisions=decisions,base_grid=grid,num_shards=args.num_shards,
        selection_size=text['settings']['selection_size'],eval_size=text['settings']['eval_size'],
        html_map_side=args.html_map_side,code={n:file_digest(REPO/n) for n in CODE},
        extension='Add 8 for EVERY arm/seed iff any source curve selects 4 and Q(4)>Q(2); one extension only',
        selection='Source VAL class-macro mean(P_AP,AUPRO); ties <=1e-12 choose smaller lambda; 0 allowed',
        source_mask_policy='Original source head selection mask/size; target evaluation uses as-provided v3 decoder',
        calibration='Frozen V/T centers and IQR; each final threshold from SAME source normal CAL pixel quantile',
        target_mask_policy='All target images and masks retained, including official zero anomaly masks')
    app.f.immutable(root/'config.json',cfg)
    for name,path in (('text_config.json',text_root/'config.json'),('manifest.json',source/'manifest.json'),
                      ('references.json',source/'references.json'),('normal_stats.json',source/'normal_stats.json'),
                      ('text_bank.json',text_root/'text_bank.json')):
        app.f.immutable(root/'provenance'/name,load_json(path))
    for seed in seeds:
        app.f.immutable(root/'provenance'/f'seed{seed}'/'selection.json',load_json(app.decision_path(text_root,seed,args.until_epoch)))
    print('Base grid: 19 fused configurations + 3 actual increment-only diagnostics; all heads/calibration frozen.',flush=True)
    print('Source selection may add lambda=8 to ALL arms. No target labels before all target predictions lock.',flush=True)
    queries=app.f.rows(source,'eval')+[r for r in app.f.rows(source,'source') if r['partition']=='val' or r['id'] in cal]
    pixels=sum((cfg['selection_size'] if r['partition']=='val' else cfg['eval_size'])**2 for r in queries)
    print(f'New fp32 V/A cache estimate: {len(seeds)*pixels*16/2**30:.2f} GiB plus HTML resources.',flush=True)


def basis_path(root,seed,partition,row):
    return Path(root)/'basis'/f'seed{seed}'/partition/(row['id']+'.npz')


def basis_identity(cfg,seed,partition):
    return dict(fingerprint=digest(cfg),seed=seed,partition=partition,source_selection=cfg['decisions'][str(seed)])


def load_basis(root,cfg,seed,partition,row,verify=True):
    path=basis_path(root,seed,partition,row)
    if verify:
        meta=load_json(path.with_suffix('.json'))
        if meta['identity']!=basis_identity(cfg,seed,partition) or meta['sha']!=file_digest(path):
            raise ValueError('Basis identity/hash mismatch')
    with np.load(path,allow_pickle=False) as data: values={n:data[n].copy() for n in data.files}
    side=cfg['selection_size'] if row['partition']=='val' else cfg['eval_size']
    if (set(values)!={'V',*("A_"+a for a in ARMS)} or any(v.shape!=(side,side) or v.dtype!=np.float32
            or not np.isfinite(v).all() for v in values.values()) or any((values['A_'+a]<0).any() for a in ARMS)):
        raise ValueError('Invalid fp32 V/A basis; do not substitute relu(fused raw text)')
    return values


def seal_path(root,seed,partition): return Path(root)/'basis'/f'seed{seed}'/(partition+'_sealed.json')


@torch.inference_mode()
def export(args):
    root=Path(args.work_dir); cfg=config(root)
    if not 0<=args.shard_id<cfg['num_shards']: raise ValueError('Invalid shard')
    if args.train_seed not in cfg['seeds']: raise ValueError('Unknown training seed')
    text_root,text,source,pcfg=context(cfg)
    if args.partition=='eval': source_decision(root,cfg)
    model=None
    for i,(key,queries) in enumerate(app.f.groups(records(cfg,args.partition)).items()):
        if i%cfg['num_shards']!=args.shard_id: continue
        pending=[]
        for row in queries:
            path=basis_path(root,args.train_seed,args.partition,row)
            if path.exists() and path.with_suffix('.json').exists():
                load_basis(root,cfg,args.train_seed,args.partition,row)
            else:
                if seal_path(root,args.train_seed,args.partition).exists(): raise ValueError('Cannot regenerate sealed basis')
                pending.append(row)
        if not pending: continue
        if model is None:
            device=app.base.device_for(args,pcfg)
            selected=app.decision(text_root,text,args.train_seed,cfg['horizon'])
            model=app.visual_model(text,device)
            texts={a:app.load_snapshot(text_root,text,a,args.train_seed,cfg['horizon'],device)[0] for a in ARMS}
            scale=app.f.scale_for(source,pcfg,'D',device)
        dataset=app.f.Features(source,pcfg,pending,'D',False)
        for row,item in zip(pending,dataset):
            side=text['settings']['selection_size'] if row['partition']=='val' else text['settings']['eval_size']
            values=basis_maps(model,texts,selected,app.base.batch_to(item,device),scale,text,side)
            path=basis_path(root,args.train_seed,args.partition,row); atomic_npz(path,**values)
            app.f.immutable(path.with_suffix('.json'),dict(identity=basis_identity(cfg,args.train_seed,args.partition),sha=file_digest(path)))
        print(f'export {args.partition} seed{args.train_seed} {key}',flush=True)


def seal(args):
    root=Path(args.work_dir); cfg=config(root); hashes={}
    for row in records(cfg,args.partition):
        load_basis(root,cfg,args.train_seed,args.partition,row)
        path=basis_path(root,args.train_seed,args.partition,row)
        hashes[row['id']]=dict(sha=file_digest(path),metadata=file_digest(path.with_suffix('.json')))
    app.f.immutable(seal_path(root,args.train_seed,args.partition),dict(identity=basis_identity(cfg,args.train_seed,args.partition),basis=hashes))


def checked_basis(root,cfg,seed,partition,row,lock,load=True):
    path=basis_path(root,seed,partition,row)
    if lock['basis'][row['id']]!=dict(sha=file_digest(path),metadata=file_digest(path.with_suffix('.json'))):
        raise ValueError('Basis changed after seal')
    return load_basis(root,cfg,seed,partition,row,verify=False) if load else None


def source_pass(root,cfg,seed,specs):
    _,text,_,pcfg=context(cfg)
    lock=load_json(seal_path(root,seed,'source'))
    if lock['identity']!=basis_identity(cfg,seed,'source'): raise ValueError('Source seal mismatch')
    vals=[]; cals=[]
    for row in records(cfg,'source'):
        app.unchanged(row,masks=True)
        values=checked_basis(root,cfg,seed,'source',row,lock)
        (vals if row['partition']=='val' else cals).append((row,values))
    truth=[app.base.evaluation_mask(r,text['settings']['selection_size']) for r,_ in vals]
    cats=[r['category'] for r,_ in vals]
    result={}
    for name,spec in specs.items():
        maps=[score(b,spec) for _,b in vals]
        point=app.base.metric_point(truth,maps,cats,pcfg['val_fpr'])
        # VAL-derived threshold is deliberately discarded. Operating thresholds
        # always use held-out normal CAL, independently for EACH lambda.
        point.pop('threshold',None)
        sample=[score(b,spec).ravel()[::max(1,b['V'].size//8192)] for _,b in cals]
        threshold=float(np.quantile(np.concatenate(sample),1-pcfg['val_fpr'],method='higher'))
        result[name]=dict(point=point,threshold=threshold)
        print(f'source seed{seed} {name}: Q={quality(point):.6f}',flush=True)
    return result


def selection_path(root): return Path(root)/'source_selection.json'


def source_decision(root,cfg):
    value=load_json(selection_path(root))
    if value['fingerprint']!=digest(cfg): raise ValueError('Wrong source decision')
    for seed in cfg['seeds']:
        if value['source_seals'][str(seed)]!=file_digest(seal_path(root,seed,'source')):
            raise ValueError('Source basis seal changed')
    return value


def select(args):
    root=Path(args.work_dir); cfg=config(root)
    for row in records(cfg,'source'): app.unchanged(row,masks=True)
    if selection_path(root).exists(): source_decision(root,cfg); return
    points={}; triggers=[]; base_specs=readouts(cfg['base_grid'])
    for seed in cfg['seeds']:
        cache=root/'source_grid'/f'seed{seed}_base.json'
        ident=dict(fingerprint=digest(cfg),source_seal=file_digest(seal_path(root,seed,'source')))
        if cache.exists():
            saved=load_json(cache)
            if saved['identity']!=ident: raise ValueError('Source curve cache mismatch')
            points[str(seed)]=saved['points']
        else:
            points[str(seed)]=source_pass(root,cfg,seed,base_specs)
            app.f.immutable(cache,dict(identity=ident,points=points[str(seed)]))
        for arm in ARMS:
            values={v:points[str(seed)][name_for(arm,v)]['point'] for v in cfg['base_grid']}
            if needs_extension(values,cfg['base_grid']): triggers.append(dict(seed=seed,arm=arm))
    grid=cfg['base_grid']+([8.] if triggers else [])
    specs=readouts(grid)
    if triggers:
        extra={k:v for k,v in specs.items() if k not in base_specs}
        for seed in cfg['seeds']: points[str(seed)].update(source_pass(root,cfg,seed,extra))
    chosen={}
    for seed in cfg['seeds']:
        chosen[str(seed)]={a:best_lambda({v:points[str(seed)][name_for(a,v)]['point'] for v in grid},grid) for a in ARMS}
    app.f.immutable(selection_path(root),dict(fingerprint=digest(cfg),grid=grid,specs=specs,extension_triggers=triggers,
        points=points,chosen=chosen,source_seals={str(s):file_digest(seal_path(root,s,'source')) for s in cfg['seeds']}))
    print(f'FROZEN source-only lambdas: {chosen}; grid={grid}',flush=True)


def prediction_lock_path(root,seed): return Path(root)/'predictions'/f'seed{seed}'/'locked.json'


def lock_predictions(args):
    root=Path(args.work_dir); cfg=config(root); selected=source_decision(root,cfg)
    seal_data=load_json(seal_path(root,args.train_seed,'eval'))
    if seal_data['identity']!=basis_identity(cfg,args.train_seed,'eval'): raise ValueError('Target seal mismatch')
    rows=records(cfg,'eval')
    if set(seal_data['basis'])!={r['id'] for r in rows}: raise ValueError('Incomplete target basis')
    for row in rows: checked_basis(root,cfg,args.train_seed,'eval',row,seal_data)
    app.f.immutable(prediction_lock_path(root,args.train_seed),dict(fingerprint=digest(cfg),
        source_selection=file_digest(selection_path(root)),basis_seal=file_digest(seal_path(root,args.train_seed,'eval')),
        seed=args.train_seed,readouts=selected['specs']))


def locked(root,cfg):
    selected=source_decision(root,cfg)
    for seed in cfg['seeds']:
        lock=load_json(prediction_lock_path(root,seed))
        if lock!=dict(fingerprint=digest(cfg),source_selection=file_digest(selection_path(root)),
                     basis_seal=file_digest(seal_path(root,seed,'eval')),seed=seed,readouts=selected['specs']):
            raise ValueError('Target predictions/source choices not completely locked')
    return selected


def verify_target_basis(root,cfg):
    rows=records(cfg,'eval')
    for seed in cfg['seeds']:
        value=load_json(seal_path(root,seed,'eval'))
        if value['identity']!=basis_identity(cfg,seed,'eval') or set(value['basis'])!={r['id'] for r in rows}:
            raise ValueError('Invalid/incomplete target seal')
        for row in rows: checked_basis(root,cfg,seed,'eval',row,value,load=False)


def diagnostic_means(basis,truth):
    result={}
    for arm in ARMS:
        a=basis['A_'+arm]
        result[arm]=dict(foreground_sum=float(a[truth].sum(dtype=np.float64)),background_sum=float(a[~truth].sum(dtype=np.float64)),
            foreground_pixels=int(truth.sum()),background_pixels=int((~truth).sum()),
            foreground_positive=int((a[truth]>1e-6).sum()),background_positive=int((a[~truth]>1e-6).sum()))
    return result


def evaluate(args):
    root=Path(args.work_dir); cfg=config(root); selected=locked(root,cfg)
    verify_target_basis(root,cfg)
    _,text,_,_=context(cfg)
    if not 0<=args.shard_id<cfg['num_shards']: raise ValueError('Invalid shard')
    for index,(key,rows) in enumerate(app.f.groups(records(cfg,'eval')).items()):
        if index%cfg['num_shards']!=args.shard_id: continue
        # First target label access, AFTER every seed's deterministic V+lambda*A
        # predictions and ALL source choices are locked.
        for row in rows: app.unchanged(row)
        truth=np.stack([app.evaluation_truth(r,text['settings']['eval_size'])[0] for r in rows])
        labels=np.array([repair.dataset_label(r) for r in rows],bool)
        mask_hashes={r['id']:file_digest(r['mask_path']) if r['mask_path'] else None for r in rows}
        app.f.immutable(root/'mask_audit'/('__'.join(key)+'.json'),dict(mask_hashes=mask_hashes,
            images=len(rows),abnormal_images=int(labels.sum()),pixel_images_included=len(rows),
            empty_anomaly_masks=int((labels&~truth.reshape(len(rows),-1).any(1)).sum())))
        for seed in cfg['seeds']:
            seal_data=load_json(seal_path(root,seed,'eval'))
            basis=[checked_basis(root,cfg,seed,'eval',r,seal_data) for r in rows]
            point_records=[]; cases=[]; inc=[]
            thresholds={n:v['threshold'] for n,v in selected['points'][str(seed)].items()}
            for name,spec in selected['specs'].items():
                part=root/'metric_parts'/f'seed{seed}'/'__'.join(key)/(name+'.json')
                identity=dict(fingerprint=digest(cfg),prediction_lock=file_digest(prediction_lock_path(root,seed)),mask_hashes=mask_hashes)
                if part.exists():
                    saved=load_json(part)
                    if saved['identity']!=identity: raise ValueError('Partial metric provenance changed')
                    point=saved['point']
                else:
                    maps=np.stack([score(b,spec) for b in basis])
                    point=app.corrected_metrics(truth,maps,labels,thresholds[name])
                    point.update(dataset=key[0],category=key[1],seed=seed,readout=name,**spec,
                                 pixel_images_included=len(rows),visual_sha=text['visual_sha'],
                                 empty_anomaly_masks=int((labels&~truth.reshape(len(rows),-1).any(1)).sum()))
                    app.f.immutable(part,dict(identity=identity,point=point))
                point_records.append(point)
                print(f'evaluate seed{seed} {key} {name} ({len(point_records)}/{len(selected["specs"])})',flush=True)
            for row,b,gt,label in zip(rows,basis,truth,labels):
                sample={n:diagnostics(score(b,spec),gt,bool(label),thresholds[n]) for n,spec in selected['specs'].items()}
                asset_key=f'seed{seed}_'+row['id']
                payload,files=html_asset(root,row,b,gt,selected['specs'],thresholds,cfg['html_map_side'],asset_key)
                js='window.LAMBDA_CASES=window.LAMBDA_CASES||{};window.LAMBDA_CASES['+json.dumps(asset_key)+']='+json.dumps(payload,separators=(',',':'))+';'
                (root/'html'/files['payload']).write_text(js,encoding='utf-8')
                cases.append(dict(id=row['id'],key=asset_key,seed=seed,dataset=key[0],category=key[1],
                    label=bool(label),empty_mask=bool(label and not gt.any()),annotation_kind='as_provided_mask',files=files,maps=sample))
                inc.append(diagnostic_means(b,gt))
            means=[]
            for arm in ARMS:
                values={k:sum(v[arm][k] for v in inc) for k in inc[0][arm]}
                means.append(dict(dataset=key[0],category=key[1],seed=seed,arm=arm,**values,
                    foreground_mean=values['foreground_sum']/values['foreground_pixels'] if values['foreground_pixels'] else None,
                    background_mean=values['background_sum']/values['background_pixels'] if values['background_pixels'] else None,
                    foreground_coverage=values['foreground_positive']/values['foreground_pixels'] if values['foreground_pixels'] else None,
                    background_coverage=values['background_positive']/values['background_pixels'] if values['background_pixels'] else None))
            app.f.immutable(root/'metrics'/f'seed{seed}'/('__'.join(key)+'.json'),dict(fingerprint=digest(cfg),
                prediction_lock=file_digest(prediction_lock_path(root,seed)),mask_hashes=mask_hashes,metrics=point_records,increments=means,cases=cases))
        print(f'evaluated {key}: {len(rows)} images, ALL masks retained; {len(selected["specs"])} readouts',flush=True)


def table(rows):
    lines=['| 数据集 | 读出/方案 | λ | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO | F1-best | 背景FPR↓ | 像素召回 |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join([r['dataset'],r.get('readout',r.get('arm','')),str(r.get('strength','源选'))]+[
            '—' if r.get(m) is None else f'{100*r[m]:.2f}'+(f" ± {100*r[m+'_std']:.2f}" if r.get(m+'_std') else '') for m in app.METRICS])+' |')
    return lines


def publish(path,text):
    path=Path(path); previous=path.read_text(encoding='utf-8')
    if SECTION_START not in previous:
        previous+='\n\n## λ对比实验实测结果\n\n'+SECTION_START+'\n\n待运行。\n\n'+SECTION_END+'\n'
    if previous.count(SECTION_START)!=1 or previous.count(SECTION_END)!=1: raise ValueError('Bad lambda result markers')
    before,rest=previous.split(SECTION_START); _,after=rest.split(SECTION_END)
    temporary=path.with_suffix('.md.tmp'); temporary.write_text(before+SECTION_START+'\n\n'+text+'\n'+SECTION_END+after,encoding='utf-8'); temporary.replace(path)


def report(args):
    root=Path(args.work_dir); cfg=config(root); selected=locked(root,cfg)
    metrics=[]; increments=[]; cases=[]
    for key,rows in app.f.groups(records(cfg,'eval')).items():
        hashes={r['id']:file_digest(r['mask_path']) if r['mask_path'] else None for r in rows}
        for seed in cfg['seeds']:
            value=load_json(root/'metrics'/f'seed{seed}'/('__'.join(key)+'.json'))
            if value['fingerprint']!=digest(cfg) or value['prediction_lock']!=file_digest(prediction_lock_path(root,seed)) or value['mask_hashes']!=hashes:
                raise ValueError('Stale metrics/masks')
            if len(value['metrics'])!=len(selected['specs']) or {r['readout'] for r in value['metrics']}!=set(selected['specs']):
                raise ValueError('Incomplete lambda metrics')
            for r in value['metrics']:
                if r['images']!=len(rows) or r['pixel_images_included']!=len(rows) or r['seed']!=seed or (r['dataset'],r['category'])!=key:
                    raise ValueError('Wrong category/count/seed')
            seal_data=load_json(seal_path(root,seed,'eval'))
            for row in rows: checked_basis(root,cfg,seed,'eval',row,seal_data,load=False)
            metrics.extend(value['metrics']); increments.extend(value['increments']); cases.extend(value['cases'])
    per_seed=repair.aggregate(metrics,('dataset','seed','readout'),app.METRICS)
    summary=repair.aggregate(per_seed,('dataset','readout'),app.METRICS)
    category=repair.aggregate(metrics,('dataset','category','readout'),app.METRICS)
    for r in summary+category: r.update(selected['specs'][r['readout']])
    chosen_records=[]
    for seed in cfg['seeds']:
        subset=[r for r in metrics if r['seed']==seed]
        for arm in ('V',*ARMS):
            lam=0. if arm=='V' else selected['chosen'][str(seed)][arm]
            name=name_for(arm,lam)
            for r in subset:
                if r['readout']==name: chosen_records.append(dict(r,arm=arm,strength=lam))
    chosen_seed=repair.aggregate(chosen_records,('dataset','seed','arm'),app.METRICS)
    chosen=repair.aggregate(chosen_seed,('dataset','arm'),app.METRICS)
    chosen_category=repair.aggregate(chosen_records,('dataset','category','arm'),app.METRICS)
    contrasts=[]
    for seed in cfg['seeds']:
        for dataset in sorted({r['dataset'] for r in chosen_seed}):
            lookup={r['arm']:r for r in chosen_seed if r['seed']==seed and r['dataset']==dataset}
            for label,a,b in (('T1-T2','T1','T2'),('T3-T1','T3','T1'),*((a+'-V',a,'V') for a in ARMS)):
                contrasts.append(dict(seed=seed,dataset=dataset,comparison=label,**{m+'_pp':
                    100*(lookup[a][m]-lookup[b][m]) if lookup[a][m] is not None and lookup[b][m] is not None else None for m in app.METRICS}))
    gains=repair.aggregate(contrasts,('dataset','comparison'),tuple(m+'_pp' for m in app.METRICS))
    inc_seed=app.summarize(increments,('dataset','seed','arm'),('foreground_mean','background_mean','foreground_coverage','background_coverage'))
    inc_summary=app.summarize(inc_seed,('dataset','arm'),('foreground_mean','background_mean','foreground_coverage','background_coverage'))
    out=root/'reports'; out.mkdir(parents=True,exist_ok=True)
    for name,rows in (('metrics',metrics),('summary',summary),('per_seed',per_seed),('per_category',category),
                      ('selected_summary',chosen),('selected_per_seed',chosen_seed),('selected_per_category',chosen_category),('contrasts',gains),
                      ('increment_regions',increments),('increment_summary',inc_summary)):
        app.base.write_csv(out/(name+'.csv'),rows)
    lines=['# 冻结文本头的λ对比结果','',f'文本快照through{cfg["horizon"]}；种子{cfg["seeds"]}；源选择网格{selected["grid"]}。',
        f'基本19组融合+3组增量诊断；本次共{len(selected["specs"])}种读出。λ=8仅由源验证触发，触发记录：{selected["extension_triggers"]}。',
        '无重训、无DINO重提取；每组源选λ对全部目标数据集统一使用。所有目标图与发布mask保留，包括全零异常mask。',
        'V/T的校准不变；最终二值阈值分别来自同一源正常CAL集。F1-best仅为目标最优阈值诊断。百分数，类别宏平均后跨种子平均。','',
        '## 源验证选择的全局λ','','| 文本种子 | T1 | T2 | T3 |','|---|---:|---:|---:|']
    for seed in cfg['seeds']: lines.append('| '+' | '.join([str(seed)]+[str(selected['chosen'][str(seed)][a]) for a in ARMS])+' |')
    lines+=['','## 各组源选λ后的完整方法比较','',*table(chosen),'','## 所有固定λ的同口径对比','',*table(summary),
            '','## 源选方案的配对增益（百分点）','','| 数据集 | 对比 | I-AUROC | P-AP | AUPRO |','|---|---|---:|---:|---:|']
    for r in gains: lines.append('| '+' | '.join([r['dataset'],r['comparison']]+['—' if r[m+'_pp'] is None else f'{r[m+"_pp"]:+.2f}' for m in ('I_AUROC','P_AP','PRO_exact')])+' |')
    lines+=['','## 实际进入融合的文本增量落点','','前景均值/覆盖仅在存在标注前景的类别内定义；背景按发布mask统计。先类内像素汇总，再类别宏平均。全零异常mask保留为图像异常标签。','',
            '| 数据集 | 文本 | 缺陷区均值 | 背景均值 | 缺陷区覆盖 | 背景覆盖 |','|---|---|---:|---:|---:|---:|']
    for r in inc_summary: lines.append('| '+' | '.join([r['dataset'],r['arm']]+['—' if r[m] is None else f'{r[m]:.5f}' for m in ('foreground_mean','background_mean')]+['—' if r[m] is None else f'{100*r[m]:.2f}%' for m in ('foreground_coverage','background_coverage')])+' |')
    text='\n'.join(lines)+'\n'; (out/'lambda_comparison.md').write_text(text,encoding='utf-8')
    if args.publish_report: publish(REPO/'对比实验.md',text)
    html=root/'html'; html.mkdir(exist_ok=True)
    payload=dict(protocol=PROTOCOL,seeds=cfg['seeds'],grid=selected['grid'],chosen=selected['chosen'],specs=selected['specs'],
        summary=summary,category=category,selected=chosen,selected_category=chosen_category,increments=inc_summary,cases=cases,source=selected['points'])
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    template=(REPO/'asset/text_lambda_dashboard.html').read_text(encoding='utf-8')
    if template.count('__LAMBDA_DATA__')!=1: raise ValueError('Invalid HTML template')
    (html/'index.html').write_text(template.replace('__LAMBDA_DATA__',encoded),encoding='utf-8')
    atomic_json(html/'analysis.json',payload)
    print(f'Results: {out/"lambda_comparison.md"}; offline HTML: {html/"index.html"}',flush=True)


def package(args):
    root=Path(args.work_dir); cfg=config(root)
    path=root/('lambda_text_analysis_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    candidates=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md','.log') and 'basis' not in p.relative_to(root).parts and 'html' not in p.relative_to(root).parts]
    candidates.extend(seal_path(root,s,p) for s in cfg['seeds'] for p in ('source','eval'))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(candidates): archive.write(p,p.relative_to(root).as_posix())
    html_zip=root/'lambda_text_html_resources.zip'
    if not (root/'html/index.html').is_file(): raise FileNotFoundError('Generate complete HTML report before packaging')
    with zipfile.ZipFile(html_zip,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((root/'html').rglob('*')):
            if p.is_file(): archive.write(p,p.relative_to(root/'html').as_posix())
    print(f'Analysis ZIP: {path}\nComplete offline HTML/resources ZIP: {html_zip}',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('prepare','export','seal','select','lock_predictions','evaluate','report','package'))
    p.add_argument('--work_dir',default='./checkpoint/text_lambda_v1_through15')
    p.add_argument('--text_work_dir',default='./checkpoint/text_comparison_v1')
    p.add_argument('--until_epoch',type=int,default=15); p.add_argument('--train_seeds',default='42')
    p.add_argument('--train_seed',type=int,default=42); p.add_argument('--lambda_grid',default='0,0.1,0.25,0.5,1,2,4')
    p.add_argument('--num_shards',type=int,default=2); p.add_argument('--shard_id',type=int,default=0)
    p.add_argument('--html_map_side',type=int,default=96); p.add_argument('--partition',choices=('source','eval'),default='source')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--publish_report',action='store_true')
    return p


if __name__=='__main__':
    arguments=parser().parse_args(); globals()[arguments.stage](arguments)
