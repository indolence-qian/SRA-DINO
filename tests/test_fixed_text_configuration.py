import base64
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import fixed_text_configuration as app


class FixedConfigurationTests(unittest.TestCase):
    def test_pinned_fusion_and_prompt_geometry(self):
        p=app.load_preset()
        self.assertEqual((p['arm'],p['text_strength'],p['readout']),('T3',2.,'T3_l2'))
        self.assertEqual((p['prompt_pairs'],p['templates_per_pair']),(24,3))
        self.assertFalse(p['renormalize_mean_delta'])
        v=np.array([[1.,2.],[3.,4.]],np.float32);a=np.array([[0.,.5],[1.,2.]],np.float32)
        np.testing.assert_array_equal(app.fuse({'V':v,'A_T3':a}),v+2*a)
        bad=dict(p,text_strength=4.)
        with patch.object(app,'read',return_value=bad),self.assertRaises(ValueError):app.load_preset()

    def test_sampling_uses_filename_and_label_not_scores(self):
        cases=[];manifest=[]
        for ds in ('a','b'):
            for label in (False,True):
                for i in range(11):
                    key=f'{ds}_{int(label)}_{i}'
                    manifest.append(dict(id=key,image_path=f'/data/{10-i:03d}.png',mask_path='mask' if label else None))
                    cases.append(dict(id=key,dataset=ds,category='c',label=label,maps=dict(score=i)))
        sampled,info=app.sample_cases(cases,manifest,.1)
        self.assertEqual(len(sampled),8)
        self.assertEqual([r['selected'] for r in info['strata']],[2]*4)
        for c in cases:c['maps']['score']=-c['maps']['score']
        again,_=app.sample_cases(list(reversed(cases)),manifest,.1)
        self.assertEqual([c['id'] for c in sampled],[c['id'] for c in again])
        for f in (0,-1,float('nan'),1.1):
            with self.assertRaises(ValueError):app.sample_cases(cases,manifest,f)

    def test_inference_loads_only_pinned_t3_and_uses_same_fusion(self):
        import torch
        import text_alignment_ablation as legacy
        import tools.text_lambda as maps
        preset=app.load_preset()
        selected=dict(visual_sha=preset['visual_sha'],text={'T3':{'sha':preset['text_sha']}})
        cfg=dict(prompts=[{}]*24,templates=['a','b','c'],settings=dict(eval_size=32))
        with patch.object(legacy,'config',return_value=cfg),patch.object(legacy,'decision',return_value=selected),\
             patch.object(legacy,'load_snapshot',return_value=('text',{},preset['text_sha'])) as load,\
             patch.object(legacy,'parent',return_value=('source',{})),patch.object(legacy,'visual_model',return_value='visual'),\
             patch.object(legacy.f,'scale_for',return_value='scale'):
            models=app.load_frozen_models('/unused','cpu')
            self.assertEqual(load.call_args.args[2],'T3')
        values={'V':np.ones((2,2),np.float32),'A_T3':np.full((2,2),.25,np.float32)}
        with patch.object(maps,'basis_maps',return_value=values) as basis:
            result=app.predict_residual_batch(models,{})
            self.assertEqual(set(basis.call_args.args[1]),{'T3'})
        np.testing.assert_array_equal(result['full'],np.full((2,2),1.5,np.float32))

    def fixture(self,root):
        preset=app.load_preset();manifest=[]
        cfg=dict(protocol='frozen_text_lambda_source_selection_v1',code={},seeds=[42],horizon=15)
        text=dict(prompts=[dict(id=i) for i in range(24)],templates=['a','b','c'],settings=dict(high_weight=.25,reference_count=4))
        head=dict(visual_sha=preset['visual_sha'],text=dict(T3=dict(sha=preset['text_sha'])))
        app.write(root/'provenance/text_config.json',text);app.write(root/'provenance/seed42/selection.json',head)
        cfg.update(text_config_sha=app.file_sha(root/'provenance/text_config.json'),decisions={'42':app.file_sha(root/'provenance/seed42/selection.json')})
        app.write(root/'config.json',cfg);fingerprint=app.digest(cfg);preset['lambda_fingerprint']=fingerprint
        specs={'V':dict(arm='V',strength=0.,kind='visual'),'T3_l2':dict(arm='T3',strength=2.,kind='fusion')}
        decision=dict(fingerprint=fingerprint,specs=specs,points={'42':{n:dict(threshold=1.) for n in specs}})
        app.write(root/'source_selection.json',decision);app.write(root/'basis/seed42/eval_sealed.json',{})
        lock=dict(fingerprint=fingerprint,source_selection=app.file_sha(root/'source_selection.json'),basis_seal=app.file_sha(root/'basis/seed42/eval_sealed.json'))
        app.write(root/'predictions/seed42/locked.json',lock)
        png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jW0kAAAAASUVORK5CYII=')
        point=dict(score=2.,threshold=1.,outcome='TP',tp_pixels=1,fp_pixels=0,fn_pixels=0,dice=1.,iou=1.,recall=1.,background_fpr=0.)
        q=dict(lo=0.,hi=1.,data=base64.b64encode(bytes(8)).decode())
        for ds in ('mvtec','btad','mpdd'):
            cases=[]
            for i in range(4):
                key=f'{ds}{i}';label=i>=2
                row=dict(id=key,dataset=ds,category='test',partition='eval',image_path=f'/source/{key}.png',mask_path='mask' if label else None)
                manifest.append(row)
                files=dict(original=f'assets/{key}.png',gt=f'assets/{key}_gt.png',payload=f'assets/{key}.js')
                cases.append(dict(id=key,key=key,seed=42,dataset=ds,category='test',label=label,empty_mask=False,files=files,maps={n:point for n in specs}))
                for rel in (files['original'],files['gt']):
                    p=root/'html'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(png)
                payload=dict(side=2,maps={'V':q,'A_T3':q},binary={n:'AA==' for n in specs},gt='AA==')
                (root/'html'/files['payload']).write_text('window.LAMBDA_CASES=window.LAMBDA_CASES||{};window.LAMBDA_CASES['+json.dumps(key)+']='+json.dumps(payload)+';',encoding='utf-8')
            metrics=[]
            for n in specs:
                metrics.append(dict(dataset=ds,category='test',readout=n,images=4,pixel_images_included=4,threshold=1.,**specs[n],**{m:.7 if n=='V' else .8 for m in app.METRICS}))
            app.write(root/'metrics/seed42'/f'{ds}__test.json',dict(fingerprint=fingerprint,prediction_lock=app.file_sha(root/'predictions/seed42/locked.json'),cases=cases,metrics=metrics))
        app.write(root/'provenance/manifest.json',manifest)
        return preset

    def test_report_keeps_full_metrics_samples_images_and_packages_all_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'old';output=Path(temp)/'fixed';preset=self.fixture(root)
            before={str(p):app.file_sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(app,'load_preset',return_value=preset):
                result=app.export_report(root,output)
                with self.assertRaises(FileExistsError):app.export_report(root,output)
            self.assertEqual(len(result),6)
            self.assertTrue(all(r['P_AP']==(.7 if r['readout']=='V' else .8) for r in result))
            payload=app.read(output/'analysis.json')
            self.assertEqual(payload['cohort_size'],12);self.assertEqual(len(payload['cases']),6)
            self.assertEqual(set(payload['specs']),{'V','T3_l2'})
            with zipfile.ZipFile(output/'fixed_t3_lambda2_resources.zip') as archive:
                for c in payload['cases']:
                    for rel in c['files'].values():self.assertIn(rel,archive.namelist())
                for rel in ('index.html','gallery.html','summary.csv','asset_manifest.json'):self.assertIn(rel,archive.namelist())
            self.assertEqual(before,{str(p):app.file_sha(p) for p in root.rglob('*') if p.is_file()})
            if os.environ.get('FIXED_T3_QA_DIR'):shutil.copytree(output,os.environ['FIXED_T3_QA_DIR'],dirs_exist_ok=True)

    def test_rejects_wrong_weights_missing_assets_and_nonportable_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'old';preset=self.fixture(root)
            with patch.object(app,'load_preset',return_value=dict(preset,text_sha='wrong')):
                with self.assertRaisesRegex(ValueError,'weights'):app.export_report(root,Path(temp)/'wrong')
            for rel in ('../x','/x','C:/x','assets/../x'):
                with self.assertRaises(ValueError):app.resource(root,rel)
            (root/'html/assets/mvtec1.png').unlink()
            with patch.object(app,'load_preset',return_value=preset),self.assertRaises(FileNotFoundError):app.export_report(root,Path(temp)/'missing')
            self.assertFalse((Path(temp)/'missing').exists())


if __name__=='__main__':unittest.main()
