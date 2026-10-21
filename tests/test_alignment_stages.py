import csv
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import alignment_stages_experiment as app
from tools.alignment_stages import (
    AlignmentHead, ResidualHead, TeacherExtractor, best_source, bundle_prediction, eligible, head_bundle,
    head_loss, normal_cells, teacher_loss,
)
from tools.reference_highres import EvidenceHead, arm_inputs
from tools.upstream_repair import constrained_loss
from tools.vlm_review import file_digest, load_json
from tests.test_reference_highres import FakeExtractor
import tests.test_reference_highres as legacy_tests


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = dict(feature_grid=8, image_size=16, high_weight=.25, semantic_weight=.1,
                        background_weight=1., residual_weight=.01, keep_weight=.1)
        self.old = {}
        for prefix, side in (("global",4),("high",8)):
            for key in ("query","residual"):
                self.old[prefix+"_"+key] = torch.randn(2,2,4,side,side)
            for key in ("distance","uncertainty","prior"):
                self.old[prefix+"_"+key] = torch.rand(2,2,side,side)
            self.old[prefix+"_base"] = torch.rand(2,1,16,16)*.8+.1
        self.batch = dict(old=self.old,similarity=torch.tensor([.8,.9]))
        self.anchors = torch.randn(2,8)

    def test_h1_exact_legacy_parity_including_nonzero_delta_and_loss(self):
        torch.manual_seed(42)
        old = EvidenceHead(2,4,self.anchors,8)
        torch.manual_seed(42)
        head = AlignmentHead("H1",2,4,self.anchors,8)
        for key,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,head.state_dict()[key]),key)
        with torch.no_grad():
            old.delta.weight.normal_()
            old.delta.bias.fill_(.1)
        head.load_state_dict(old.state_dict())
        inputs = arm_inputs(self.old,"B_highres",8,.25)
        a,b = old(inputs),head(inputs)
        for key in ("delta","semantic_logits"):
            self.assertTrue(torch.equal(a[key],b[key]),key)
        mask = torch.zeros(2,1,16,16); mask[0,:,2:4,2:4]=1
        main,_ = constrained_loss(inputs["base"],a["delta"],mask)
        expected = main+.1*F.binary_cross_entropy_with_logits(a["semantic_logits"],mask,reduction="none").flatten(1).mean(1)
        bundle=head_bundle(head,"H1",self.batch,self.cfg)
        self.assertTrue(torch.equal(expected,head_loss(bundle,self.batch,mask,self.cfg)))

    def test_text_removal_and_margin_only_path(self):
        h2=AlignmentHead("H2",2,4,self.anchors,8)
        self.assertFalse(hasattr(h2,"to_text"))
        self.assertFalse(hasattr(h2,"anchors"))
        h3=AlignmentHead("H3",2,4,self.anchors,8)
        self.assertEqual(h3.delta.in_channels,1)
        inputs=arm_inputs(self.old,"B_highres",8,.25)
        with torch.no_grad():
            h3.to_text.weight.zero_(); h3.to_text.bias.normal_(); h3.delta.weight.fill_(.3)
        first=h3(inputs)["delta"]
        inputs["query"] += 100
        self.assertTrue(torch.equal(first,h3(inputs)["delta"]))
        self.assertNotIn("semantic_logits",h2(inputs))

    def add_views(self):
        self.batch["views"]=[]
        for i in range(5):
            self.batch["views"].append(dict(query=torch.randn(2,2,4,4,4),prior=torch.rand(2,2,4,4),
                base=torch.full((2,1,16,16),.2),teacher=F.normalize(torch.randn(2,8,4,4),dim=1).requires_grad_()))
        boxes=[[0,0,1,1],[0,0,.6,.6],[.4,0,1,.6],[0,.4,.6,1],[.4,.4,1,1]]
        self.batch["boxes"]=torch.tensor([boxes,boxes])
        # Match the synthetic frozen late readout at zero delta.
        self.batch["old"]["global_base"].fill_(.2)
        self.batch["old"]["high_base"].fill_(.2)

    def test_late_identity_original_coordinates_and_detached_teacher(self):
        self.add_views()
        head=AlignmentHead("T11",2,4,self.anchors,8)
        bundle=head_bundle(head,"T11",self.batch,self.cfg)
        self.assertTrue(torch.equal(bundle_prediction(bundle,0,self.cfg),bundle["base"]))
        self.assertTrue(torch.allclose(bundle_prediction(bundle,1,self.cfg),bundle["base"],atol=1e-7))
        xy=bundle["views"][2]["coords"]
        self.assertGreater(float(xy[:,0].min()),.4)
        self.assertLess(float(xy[:,1].max()),.6)
        mask=torch.zeros(2,1,16,16); mask[0,:,2,3]=1
        loss=teacher_loss(bundle,self.batch,mask,self.cfg).mean()
        loss.backward()
        self.assertGreater(float(head.to_text.weight.grad.abs().sum()),0)
        self.assertTrue(all(v["teacher"].grad is None for v in self.batch["views"]))
        self.assertFalse(normal_cells(mask,4)[0,0,0,0])
        self.assertEqual(int((~normal_cells(mask,4)).sum()),1)
        full=torch.ones_like(mask)
        self.assertEqual(float(teacher_loss(bundle,self.batch,full,self.cfg).sum()),0)

    def test_reference_gate_initialization_inputs_and_gradients(self):
        torch.manual_seed(9); fixed=ResidualHead("R2",2,4,8)
        torch.manual_seed(9); learned=ResidualHead("R3",2,4,8)
        for key,value in fixed.state_dict().items():
            self.assertTrue(torch.equal(value,learned.state_dict()[key]))
        inputs=arm_inputs(self.old,"D_highres_reference",8,.25)
        inputs["reference_similarity"]=self.batch["similarity"]
        self.assertTrue(torch.equal(learned(inputs)["gate"],torch.full((2,1,16,16),.5)))
        with torch.no_grad(): learned.delta.weight.normal_()
        output=learned(inputs)
        output["delta"].square().mean().backward()
        self.assertGreater(float(learned.gate[-1].weight.grad.abs().sum()),0)

    def test_pareto_selection_and_ties(self):
        baseline=dict(PRO_exact=.8,P_AP=.5,alpha=0.,epoch=0)
        bad=dict(PRO_exact=.9,P_AP=.49,alpha=1.,epoch=1)
        self.assertFalse(eligible(bad,baseline))
        self.assertEqual(best_source([bad],baseline),baseline)
        a=dict(PRO_exact=.81,P_AP=.5,alpha=.5,epoch=1)
        b=dict(a,alpha=.1,epoch=2)
        self.assertEqual(best_source([a,b],baseline),b)

    def test_teacher_uses_standardized_original_adapter_and_is_detached(self):
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.linear=torch.nn.Linear(1024,8)
            def forward(self,value):
                return self.linear(value),value
        extractor=object.__new__(TeacherExtractor)
        extractor.layers=(0,1)
        extractor.captured={i:torch.randn(1,4,1024,requires_grad=True) for i in extractor.layers}
        extractor.adapter=torch.nn.Module()
        extractor.adapter.patch_token_adapter=torch.nn.ModuleList([Projection(),Projection()])
        with patch("tools.reference_highres.FrozenExtractor.view",return_value={}):
            result=extractor.view(None)["teacher"]
        expected=[]
        for i in extractor.layers:
            p=extractor.captured[i]
            s=(p-p.mean(1,keepdim=True))/(p.std(1,keepdim=True)+1e-6)
            expected.append(F.normalize(extractor.adapter.patch_token_adapter[i](s)[0][0],dim=-1))
        expected=F.normalize(torch.stack(expected).mean(0),dim=-1).T.reshape(8,2,2)
        self.assertTrue(torch.allclose(result,expected))
        self.assertFalse(result.requires_grad)

    def test_late_exact_endpoints_retain_classification_gradient_without_probability_floor(self):
        self.add_views()
        for probability,target,sign in ((0.,1.,-1),(1.,0.,1)):
            head=AlignmentHead("T01",2,4,self.anchors,8)
            for view in self.batch["views"]: view["base"].fill_(probability)
            for key in ("global_base","high_base"): self.batch["old"][key].fill_(probability)
            bundle=head_bundle(head,"T01",self.batch,self.cfg)
            predicted=bundle_prediction(bundle,1,self.cfg)
            self.assertTrue(torch.equal(predicted,bundle["base"]))
            mask=torch.full_like(predicted,target)
            head_loss(bundle,self.batch,mask,self.cfg).mean().backward()
            gradient=float(head.delta.bias.grad)
            self.assertTrue(math.isfinite(gradient))
            self.assertGreater(sign*gradient,.5)


class FakeTeacher(FakeExtractor):
    def image_views(self,image):
        g,tiles,boxes=super().image_views(image)
        if getattr(self,"include_teacher",True):
            for view in [g]+tiles:
                raw=view["raw"].mean(0)[:,:8]
                side=int(raw.shape[0]**.5)
                view["teacher"]=F.normalize(raw,dim=-1).T.reshape(8,side,side)
        return g,tiles,boxes


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=legacy_tests.PipelineTests("test_prepare_content_families_and_reference_protocol")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare(); self.fixture.export()
        self.old=Path(self.fixture.args.work_dir)
        self.root=self.fixture.root/"alignment"
        self.args=app.parser().parse_args(["--stage","prepare","--work_dir",str(self.root),"--v1_dir",str(self.old),
            "--epochs","2","--until_epoch","1","--workers","0","--device","cpu","--num_shards","2"])
        app.prepare(self.args)

    def extra(self,partition):
        self.args.partition=partition
        with patch.object(app,"TeacherExtractor",FakeTeacher),patch.object(app.legacy,"mask_for",side_effect=AssertionError("export used GT")):
            for shard in range(2):
                self.args.shard_id=shard; app.export_extra(self.args)
        app.seal_extra(self.args)

    def test_mask_free_extra_export_and_seal_failure(self):
        self.extra("source"); self.extra("eval")
        cfg=app.config(self.root)
        row=app.selected_rows(cfg,"eval")[0]
        sample=app.FeatureData(self.root,cfg,[row],extra=True,masks=False)[0]
        self.assertNotIn("mask",sample)
        self.assertTrue(all("teacher" not in v for v in sample["views"]))
        with self.assertRaisesRegex(ValueError,"Target masks"):
            app.FeatureData(self.root,cfg,[row],masks=True)
        path=self.root/"extra"/f"{row['id']}.npz"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"Cache modified"):
            app.FeatureData(self.root,cfg,[row],extra=True)[0]

    def test_three_stages_resume_freeze_report_and_target_isolation(self):
        self.extra("source")
        full=self.fixture.root/"full"
        shutil.copytree(self.root,full)
        original=app.legacy.mask_for
        def source_only(row,size):
            self.assertNotEqual(row["partition"],"eval")
            return original(row,size)
        with patch.object(app.legacy,"mask_for",side_effect=source_only):
            for mode in ("H1","H2","H3","T10","T01","T11"):
                self.args.mode=mode; self.args.until_epoch=1; app.train(self.args)
                self.args.until_epoch=2; app.train(self.args)
                before=file_digest(app.run_dir(self.root,42,mode)/"last.pt")
                app.train(self.args)
                self.assertEqual(before,file_digest(app.run_dir(self.root,42,mode)/"last.pt"))
            self.args.mode="H3"; self.args.work_dir=str(full); app.train(self.args)
            self.args.work_dir=str(self.root)
            a=torch.load(app.run_dir(full,42,"H3")/"last.pt",weights_only=False)
            b=torch.load(app.run_dir(self.root,42,"H3")/"last.pt",weights_only=False)
            for key in a["state_dict"]:
                self.assertTrue(torch.equal(a["state_dict"][key],b["state_dict"][key]),key)
            self.args.continue_on_fallback=True
            self.assertTrue(app.stage_gate(self.args)); app.select_front(self.args)
            app.fit_distance(self.args); app.calibrate_r1(self.args)
            self.assertTrue(app.reference_gate(self.args))
            parent=load_json(self.root/"decisions"/"seed42_front.json")
            if parent["checkpoint"]:
                parent_sha=file_digest(parent["checkpoint"])
            for mode in ("R2","R3"):
                self.args.mode=mode; app.train(self.args)
            if parent["checkpoint"]:
                self.assertEqual(parent_sha,file_digest(parent["checkpoint"]))
            app.source_audit(self.args)
        self.extra("eval")
        for shard in range(2):
            self.args.shard_id=shard; app.evaluate(self.args)
        app.report(self.args)
        with patch.object(app,"TeacherExtractor",FakeTeacher),patch.object(app.legacy,"mask_for",side_effect=AssertionError("online benchmark read target labels")):
            self.args.benchmark_images=1
            app.benchmark(self.args)
        self.assertTrue((self.root/"reports"/"through2"/"end_to_end_seed42.csv").exists())
        self.assertTrue((self.root/"diagnostics"/"seed42"/"H3_through2.json").exists())
        with (self.root/"reports"/"through2"/"summary.csv").open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual({r["mode"] for r in rows},{"H0","H1","H2","H3","T00","T10","T01","T11","R0","R1","R2","R3"})
        self.assertEqual({r["selector"] for r in rows},{"primary","bce"})
        self.assertTrue((self.root/"reports"/"through2"/"comparisons.csv").exists())
        selected=app.run_dir(self.root,42,"H2")/"best_primary.pt"
        original_bytes=selected.read_bytes()
        value=torch.load(selected,weights_only=False); value["point"]["threshold"]+=.01
        torch.save(value,selected)
        with self.assertRaisesRegex(ValueError,"Stale evaluation"):
            app.report(self.args)
        selected.write_bytes(original_bytes)
        target=next(r for r in app.selected_rows(app.config(self.root),"eval") if r["mask_path"])
        Path(target["mask_path"]).write_bytes(b"changed GT")
        self.args.shard_id=0
        with self.assertRaisesRegex(ValueError,"mask changed"):
            app.evaluate(self.args)

    def test_source_fallback_skips_and_parent_hash_separates_reference_runs(self):
        cfg=app.config(self.root)
        point=dict(PRO_exact=.5,P_AP=.1,loss=1.,threshold=.2,alpha=0.,epoch=0)
        for mode in ("H1","H2","H3"):
            out=app.run_dir(self.root,42,mode)
            app.legacy.save_torch(out/"last.pt",dict(identity=dict(fingerprint=app.digest(cfg)),epoch=1))
            app.legacy.save_torch(out/"best_primary.pt",dict(identity=dict(fingerprint=app.digest(cfg)),point=point,baseline=point))
        self.assertFalse(app.stage_gate(self.args))
        app.select_front(self.args)
        parent=load_json(self.root/"decisions"/"seed42_front.json")
        self.assertEqual(parent["mode"],"H0")
        self.assertFalse(app.reference_gate(self.args))
        self.assertNotEqual(app.run_dir(self.root,42,"R2",dict(parent,sha="a")),app.run_dir(self.root,42,"R2",dict(parent,sha="b")))


BASH=os.environ.get("ALIGNMENT_TEST_BASH") or os.environ.get("SRA_TEST_BASH")


@unittest.skipUnless(BASH,"Set SRA_TEST_BASH to Git Bash")
class ShellTests(unittest.TestCase):
    def run_shell(self,fail="NEVER",gate=0,gpus="0,1"):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            shutil.copyfile(app.REPO/"run_exp_alignment_stages.sh",root/"run.sh")
            (root/"v1").mkdir()
            for name in ("config.json","sealed.json"):
                (root/"v1"/name).write_text("{}")
            (root/"fake_python").write_bytes(b'#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]] || exit 1\nif [[ "$*" == *"--stage gate"* || "$*" == *"--stage reference_gate"* ]]; then exit "${GATE_CODE:-0}"; fi\nexit 0\n')
            (root/"flock").write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
            for name in ("fake_python","flock"):
                (root/name).chmod(0o755)
            env=os.environ.copy(); env.update(TRAIN_PYTHON="./fake_python",V1_DIR="./v1",WORK_DIR="./output",GPU_IDS=gpus,FAIL_STAGE=fail,GATE_CODE=str(gate))
            env["PATH"]=str(root)+os.pathsep+str(Path(BASH).parent)+os.pathsep+env.get("PATH","")
            return subprocess.run([BASH,"run.sh"],cwd=root,env=env,text=True,capture_output=True,timeout=45)

    def test_order_all_modes_and_dual_gpu(self):
        result=self.run_shell(); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        for mode in ("H1","H2","H3","T10","T01","T11","R2","R3"):
            self.assertIn("--mode "+mode,result.stdout)
        self.assertLess(result.stdout.index("--mode H3"),result.stdout.index("--mode T10"))
        self.assertLess(result.stdout.index("--stage select"),result.stdout.index("--mode R2"))
        self.assertLess(result.stdout.rindex("--stage train"),result.stdout.index("--stage evaluate"))
        self.assertIn("GPU=1",result.stdout)

    def test_skips_are_not_failures_but_worker_errors_stop(self):
        result=self.run_shell(gate=3); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotIn("--mode T10",result.stdout); self.assertNotIn("--mode R2",result.stdout)
        self.assertIn("--stage calibrate_r1",result.stdout)
        result=self.run_shell(fail="--mode T01")
        self.assertNotEqual(result.returncode,0); self.assertNotIn("--stage evaluate",result.stdout)
        self.assertNotEqual(self.run_shell(gpus="0,0").returncode,0)


if __name__=="__main__":
    unittest.main()
