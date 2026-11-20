#!/usr/bin/env python3
"""Pinned T3/lambda=2 fusion and portable reports from the completed lambda run.

No training, feature extraction, checkpoint selection or target threshold fitting.
The original source-selected experiment remains immutable.
"""
import argparse
import base64
from collections import defaultdict
import csv
import hashlib
import json
import math
import mimetypes
from pathlib import Path, PurePosixPath
import shutil
import zipfile

REPO = Path(__file__).resolve().parent
PRESET = REPO / 'configs/fixed_t3_lambda2_v1.json'
METRICS = ('I_AUROC','I_AP','P_AUROC','P_AP','PRO_exact','F1_best',
           'background_FPR_source_threshold','pixel_recall_source_threshold')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def load_preset():
    cfg=read(PRESET)
    if (cfg['protocol']!='fixed_t3_lambda2_v1' or cfg['arm']!='T3' or cfg['text_strength']!=2
            or cfg['readout']!='T3_l2' or cfg['text_mode']!='mean_paired_delta'
            or cfg['prompt_pairs']!=24 or cfg['renormalize_mean_delta']):
        raise ValueError('Pinned T3/lambda=2 preset changed; use a new version')
    return cfg


def fuse(basis):
    """Use fp32 V/A exported by lambda_text_experiment; A is already view-clamped."""
    cfg=load_preset()
    return basis['V']+cfg['text_strength']*basis['A_'+cfg['arm']]


def load_frozen_models(text_work_dir, device='cuda:0'):
    """Reusable inference interface for existing global/local residual batches."""
    import text_alignment_ablation as legacy
    preset=load_preset();root=Path(text_work_dir);cfg=legacy.config(root)
    selected=legacy.decision(root,cfg,preset['seed'],preset['horizon'])
    if selected['visual_sha']!=preset['visual_sha'] or selected['text']['T3']['sha']!=preset['text_sha']:
        raise ValueError('Expected the pinned visual/T3 weights')
    if len(cfg['prompts'])!=24 or len(cfg['templates'])!=3:
        raise ValueError('Expected 24 paired prompts and three templates')
    text,_,sha=legacy.load_snapshot(root,cfg,'T3',preset['seed'],preset['horizon'],device)
    if sha!=preset['text_sha']: raise ValueError('T3 snapshot changed')
    source,parent=legacy.parent(cfg)
    return dict(visual=legacy.visual_model(cfg,device),text=text,selected=selected,cfg=cfg,
                scale=legacy.f.scale_for(source,parent,'D',device),device=device)


def predict_residual_batch(models, batch, size=None):
    """The batch contains the original matched/rescaled residuals and view boxes."""
    import torch
    from tools.text_lambda import basis_maps
    with torch.inference_mode():
        values=basis_maps(models['visual'],{'T3':models['text']},models['selected'],batch,
                          models['scale'],models['cfg'],size or models['cfg']['settings']['eval_size'])
    return dict(visual=values['V'],increment=values['A_T3'],full=fuse(values))


def sample_cases(cases, manifest, fraction):
    if not math.isfinite(fraction) or not 0<fraction<=1: raise ValueError('sample-fraction must be in (0,1]')
    byid={r['id']:r for r in manifest}; groups=defaultdict(list)
    for case in cases:
        row=byid[case['id']]
        if case['label']!=bool(row['mask_path']): raise ValueError('Image label changed')
        groups[case['dataset'],case['category'],case['label']].append(case)
    sampled=[]; strata=[]
    for key,items in sorted(groups.items()):
        items=sorted(items,key=lambda c:(PurePosixPath(byid[c['id']]['image_path'].replace('\\','/')).name,c['id']))
        count=max(1,math.ceil(len(items)*fraction))
        chosen=[items[(2*i+1)*len(items)//(2*count)] for i in range(count)]
        sampled+=chosen
        strata.append(dict(dataset=key[0],category=key[1],label=key[2],total=len(items),selected=count))
    return sampled,dict(fraction=fraction,total=len(cases),selected=len(sampled),strata=strata,
                       method='filename_systematic_midpoints_by_dataset_category_image_label',ids=[c['id'] for c in sampled])


def resource(base, relative):
    path=PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError('Nonportable HTML resource')
    result=(base/relative).resolve()
    if base.resolve() not in result.parents or not result.is_file(): raise FileNotFoundError(result)
    return result


def export_report(lambda_dir, output_dir, sample_fraction=.1):
    root=Path(lambda_dir).resolve(); output=Path(output_dir).resolve(); preset=load_preset()
    if root==output or root in output.parents or output in root.parents: raise ValueError('Use a separate output directory')
    if output.exists(): raise FileExistsError('Use a NEW output directory; existing reports are preserved')
    cfg=read(root/'config.json'); decision=read(root/'source_selection.json')
    if digest(cfg)!=preset['lambda_fingerprint'] or decision['fingerprint']!=digest(cfg): raise ValueError('Wrong frozen lambda experiment')
    for file,expected in cfg['code'].items():
        if file_sha(REPO/file)!=expected: raise ValueError('Original sealed code changed: '+file)
    head=read(root/'provenance/seed42/selection.json')
    if file_sha(root/'provenance/seed42/selection.json')!=cfg['decisions']['42']:
        raise ValueError('Frozen checkpoint selection changed')
    if head['visual_sha']!=preset['visual_sha'] or head['text']['T3']['sha']!=preset['text_sha']:
        raise ValueError('Frozen model weights differ from the pinned configuration')
    text=read(root/'provenance/text_config.json')
    if file_sha(root/'provenance/text_config.json')!=cfg['text_config_sha']:
        raise ValueError('Original text configuration changed')
    if len(text['prompts'])!=24 or len(text['templates'])!=3 or text['settings']['high_weight']!=.25 or text['settings']['reference_count']!=4:
        raise ValueError('Prompt bank or visual settings differ')
    lock_path=root/'predictions/seed42/locked.json'; lock=read(lock_path)
    if (lock['fingerprint']!=digest(cfg) or lock['source_selection']!=file_sha(root/'source_selection.json')
            or lock['basis_seal']!=file_sha(root/'basis/seed42/eval_sealed.json')):
        raise ValueError('Prediction lock differs')
    name=preset['readout']; names=('V',name)
    if decision['specs'][name]!=dict(arm='T3',strength=2.,kind='fusion'): raise ValueError('Wrong fixed readout')
    specs={n:decision['specs'][n] for n in names}
    manifest=[r for r in read(root/'provenance/manifest.json') if r['partition']=='eval']
    groups=defaultdict(list)
    for row in manifest:groups[row['dataset'],row['category']].append(row)
    metrics=[]; cases=[]
    for key,queries in sorted(groups.items()):
        data=read(root/'metrics/seed42'/('__'.join(key)+'.json'))
        if data['fingerprint']!=digest(cfg) or data['prediction_lock']!=file_sha(lock_path): raise ValueError('Stale metrics')
        if {c['id'] for c in data['cases']}!={r['id'] for r in queries}: raise ValueError('Incomplete case cohort')
        for n in names:
            point=next(r for r in data['metrics'] if r['readout']==n)
            if point['images']!=point['pixel_images_included'] or point['images']!=len(queries): raise ValueError('Excluded images')
            if point['threshold']!=decision['points']['42'][n]['threshold']: raise ValueError('Source threshold changed')
            metrics.append(point)
        for c in data['cases']:
            cases.append(dict(c,maps={n:c['maps'][n] for n in names}))
    summary=[]
    for dataset in sorted({r['dataset'] for r in metrics}):
        for n in names:
            members=[r for r in metrics if r['dataset']==dataset and r['readout']==n]
            summary.append(dict(dataset=dataset,readout=n,**specs[n],**{m:sum(r[m] for r in members)/len(members) if all(r[m] is not None for r in members) else None for m in METRICS}))
    sampled,sampling=sample_cases(cases,manifest,sample_fraction)
    # Validate the complete resource closure before creating the new report.
    resources={rel:resource(root/'html',rel) for c in sampled for rel in c['files'].values()}
    for c in sampled:
        body=resources[c['files']['payload']].read_text(encoding='utf-8')
        if f'window.LAMBDA_CASES[{json.dumps(c["key"])}]=' not in body: raise ValueError('Wrong case payload')
    output.mkdir(parents=True)
    for rel,source in resources.items():
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for c in sampled:
        path=resources[c['files']['original']]
        mime=mimetypes.guess_type(path.name)[0] or 'image/jpeg'
        # file:// images taint Canvas in ordinary browsers. A data URL keeps the
        # modal's pixel reads portable without browser flags or a local server.
        c['preview_data_url']='data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
    for filename,rows in (('summary.csv',summary),('per_category.csv',metrics)):
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (output/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    payload=dict(protocol=preset['protocol'],seeds=[42],specs=specs,summary=summary,category=metrics,cases=sampled,
                 cohort_size=len(cases),sampling=sampling,increments=[],preset=preset)
    template=(REPO/'asset/fixed_t3_dashboard.html').read_text(encoding='utf-8')
    if template.count('__LAMBDA_DATA__')!=1: raise ValueError('Invalid report template')
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    (output/'index.html').write_text(template.replace('__LAMBDA_DATA__',encoded),encoding='utf-8')
    shutil.copyfile(output/'index.html',output/'gallery.html')
    write(output/'analysis.json',payload);write(output/'sampling.json',sampling);write(output/'fixed_config.json',preset)
    write(output/'provenance.json',dict(lambda_config_sha=file_sha(root/'config.json'),prediction_lock_sha=file_sha(lock_path),
        fixed_readout=name,thresholds={n:decision['points']['42'][n]['threshold'] for n in names},metrics='full cohort; copied frozen readouts',
        selection_status=preset['evaluation_status'],source_dir=str(root)))
    lines=['# 固定配置：T3＋λ=2','','已有目标集参与配置选择；此处为开发结果，未重训。全量指标与抽样图库分开。','',
           '| 数据集 | 方案 | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO | F1-best | 背景FPR | 像素召回 |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        lines.append('| '+' | '.join([r['dataset'],r['readout']]+['N/A' if r[m] is None else f'{100*r[m]:.2f}' for m in METRICS])+' |')
    (output/'固定配置结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files={p.relative_to(output).as_posix():file_sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'asset_manifest.json',files)
    with zipfile.ZipFile(output/'fixed_t3_lambda2_resources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted([*files,'asset_manifest.json']):archive.write(output/rel,rel)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lambda-dir',default='./checkpoint/text_lambda_v1_through15')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--sample-fraction',type=float,default=.1)
    args=p.parse_args();export_report(args.lambda_dir,args.output_dir,args.sample_fraction)
    print('Fixed T3/lambda=2 report and complete offline resources: '+args.output_dir)


if __name__=='__main__':main()
