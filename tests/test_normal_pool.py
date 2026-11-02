import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import normal_pool_experiment as app
import test_residual_text as previous
from tools.normal_residual_pool import normal_pool, second_moment, variation_scale, raw_maps, image_scores
from tools.vlm_review import atomic_json, load_json, file_digest


class NormalPoolCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_self_image_is_excluded_including_all_local_crops(self):
        cfg = dict(image_size=4, patch_size=2, layers=[0], match_chunk=2, bank_chunk=3)
        class Encoder:
            projection = torch.eye(4)
        supports = [[torch.eye(4)[i:i+1].repeat(4, 1)[None] for _ in range(5)] for i in range(3)]
        records = [dict(id=str(i),rgb_sha=str(i),dataset="d",category="c",partition="reference",mask_path=None)
                   for i in range(3)]
        pool, mapping = normal_pool(Encoder(),supports,records,cfg)
        self.assertEqual(pool["residual"].shape,(3,5,1,4,2,2))
        self.assertTrue(np.allclose(pool["distance"],np.sqrt(2)))
        for i, row in enumerate(mapping):
            self.assertNotIn(str(i),row["bank_ids"])
            self.assertEqual(row["bank_ids"],[str(j) for j in range(3) if j!=i])
        # If the held-out image's local patches leaked, local residuals would be zero.
        self.assertTrue((np.linalg.norm(pool["residual"],axis=3)>0).all())
        bad = copy.deepcopy(records)
        bad[1]["rgb_sha"]=bad[0]["rgb_sha"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            normal_pool(Encoder(),supports,bad,cfg)
        with self.assertRaisesRegex(ValueError,"at least two"):
            normal_pool(Encoder(),supports[:1],records[:1],cfg)
        bad=copy.deepcopy(records)
        bad[0]["mask_path"]="abnormal.png"
        with self.assertRaisesRegex(ValueError,"known-normal"):
            normal_pool(Encoder(),supports,bad,cfg)

    def test_second_moment_equal_image_view_weights_no_centering(self):
        vectors=np.ones((2,5,1,2,2,2),np.float16)
        vectors[1,0]=3
        vectors[:,1:]=2
        expected=np.array([[[5.,5.],[4.,4.]]])
        self.assertTrue(np.array_equal(second_moment(vectors),expected))

    def test_four_normal_references_use_exactly_three_other_images(self):
        cfg=dict(image_size=4,patch_size=2,layers=[0],match_chunk=2,bank_chunk=3)
        class Encoder:
            projection=torch.eye(4)
        supports=[[torch.eye(4)[i:i+1].repeat(4,1)[None] for _ in range(5)] for i in range(4)]
        records=[dict(id=str(i),rgb_sha=str(i),dataset="d",category="c",partition="reference",mask_path=None)
                 for i in range(4)]
        pool,mapping=normal_pool(Encoder(),supports,records,cfg)
        self.assertEqual(pool["residual"].shape,(4,5,1,4,2,2))
        self.assertTrue(np.allclose(pool["distance"],np.sqrt(2)))
        for i,item in enumerate(mapping):
            self.assertEqual(item["bank_ids"],[str(j) for j in range(4) if j!=i])

    def test_variation_suppresses_common_directions_not_all_large_residuals(self):
        source=np.array([[[100.,1.],[100.,1.]]])
        residual=np.zeros((5,1,2,1,2),np.float32)
        residual[:,:,0,0,0]=10
        residual[:,:,1,0,1]=2
        result=raw_maps(residual,np.zeros((5,1,1,2)),source,source)
        self.assertGreater(result["P"][0,0,0,0],result["P"][0,0,0,1])
        self.assertLess(result["N"][0,0,0,0],result["N"][0,0,0,1])

    def test_shrinkage_floor_zero_preservation_and_rho_one_control(self):
        source=np.ones((2,2,4))
        support=np.zeros_like(source)
        self.assertTrue((variation_scale(source,support,0)>=.1).all())
        residual=np.zeros((5,2,4,2,2))
        result=raw_maps(residual,np.zeros((5,2,2,2)),source,support)
        self.assertTrue(all(not x.any() for x in result.values()))
        residual=np.ones_like(residual)
        result=raw_maps(residual,np.zeros((5,2,2,2)),source,support,rho=1)
        self.assertTrue(np.array_equal(result["N"],result["S"]))
        with self.assertRaisesRegex(ValueError,"Degenerate"):
            variation_scale(np.zeros_like(source),support)
        with self.assertRaises(ValueError):
            variation_scale(source,support,rho=float("nan"))

    def test_top_fraction_does_not_modify_pixel_map(self):
        maps=np.arange(100,dtype=float).reshape(1,10,10)
        before=maps.copy()
        maximum,top=image_scores(maps,.05)
        self.assertEqual(maximum[0],99)
        self.assertEqual(top[0],97)
        self.assertTrue(np.array_equal(maps,before))


class NormalPoolWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture=previous.WorkflowTests()
        self.fixture.setUp()
        self.root=self.fixture.base/"normal_pool"
        self.args=app.parser().parse_args(["--stage","prepare","--work_dir",str(self.root),
            "--device","cpu","--dino_repo_dir",str(app.REPO/"dinov3"),
            "--dino_weights",str(self.fixture.dino_weights),"--visual_layers","0,1",
            "--input_dim","8","--feature_dim","4","--patch_size","8",
            "--image_size","32","--eval_size","32","--reference_count","2",
            "--calibration_images","2","--num_shards","1","--precision","fp32"])

    def tearDown(self):
        self.fixture.tearDown()

    def prepare(self):
        return app.prepare(self.args,copy.deepcopy(self.fixture.catalog))

    def export(self,partition):
        self.args.partition=partition
        cfg=app.config(self.root)
        with patch.object(app,"FrozenDinoExtractor",previous.FakeExtractor):
            for shard in range(cfg["num_shards"]):
                self.args.shard_id=shard
                app.export(self.args)
        app.seal(self.args)

    def predictions(self,partition):
        self.args.partition=partition
        for shard in range(app.config(self.root)["num_shards"]):
            self.args.shard_id=shard
            app.predict(self.args)

    def evaluation(self,partition):
        self.args.partition=partition
        app.lock_predictions(self.args)
        for shard in range(app.config(self.root)["num_shards"]):
            self.args.shard_id=shard
            app.evaluate(self.args)
        app.report(self.args)

    def test_full_cpu_pipeline_source_prior_is_train_only_and_resume(self):
        self.args.num_shards=2
        cfg=self.prepare()
        before=file_digest(self.root/"config.json")
        self.prepare()
        self.assertEqual(before,file_digest(self.root/"config.json"))
        self.export("source")
        app.fit(self.args)
        app.fit(self.args)
        model=app.model_for(self.root,cfg)
        model_sha=file_digest(self.root/"normal_model.json")
        (self.root/"normal_model.sha.json").unlink()
        app.fit(self.args)  # Recover a fit interrupted between model and hash.
        self.assertEqual(model_sha,file_digest(self.root/"normal_model.json"))
        app.model_for(self.root,cfg)
        self.assertEqual({x[1] for x in model["prior_classes"]},set(cfg["train_categories"]))
        self.assertFalse(set(cfg["val_categories"]) & {x[1] for x in model["prior_classes"]})
        expected=np.stack([app.pool_moment(self.root,cfg,tuple(k)) for k in model["prior_classes"]]).mean(0)
        self.assertTrue(np.allclose(model["source_prior"],expected))
        self.predictions("source")
        self.evaluation("source")
        self.export("eval")
        self.predictions("eval")
        self.evaluation("eval")
        with patch.object(app,"FrozenDinoExtractor",side_effect=AssertionError("No re-encoding on resume")):
            for part in ("source","eval"):
                self.args.partition=part
                app.export(self.args)
        self.assertTrue((self.root/"reports/eval/summary.csv").exists())
        self.assertTrue(list((self.root/"panels/eval").rglob("*.png")))
        self.assertEqual(set(model["thresholds"]),set(app.ARMS))
        app.package(self.args)
        archive=next(self.root.glob("normal_pool_analysis_*.zip"))
        with zipfile.ZipFile(archive) as package:
            self.assertIn("reports/eval/summary.csv",package.namelist())
            self.assertFalse(any(n.endswith(".npz") for n in package.namelist()))

    def test_target_gt_is_not_accessed_before_predictions_are_locked(self):
        self.prepare()
        target=next(r for r in self.fixture.catalog[0] if r["partition"]=="eval" and r["mask_path"])
        path=Path(target["mask_path"])
        original=path.read_bytes()
        path.unlink()
        self.export("source")
        app.fit(self.args)
        self.export("eval")
        with patch.object(app,"evaluation_mask",side_effect=AssertionError("Target mask accessed early")):
            self.predictions("eval")
            with self.assertRaises(FileNotFoundError):
                app.evaluate(self.args)
            app.lock_predictions(self.args)
        path.write_bytes(original)
        self.evaluation("eval")

    def test_pool_and_normal_model_tampering_are_rejected(self):
        cfg=self.prepare()
        self.export("source")
        app.fit(self.args)
        path=self.root/"normal_model.json"
        model=load_json(path)
        model["source_prior"][0][0][0]+=1
        atomic_json(path,model)
        with self.assertRaisesRegex(ValueError,"Normal statistics changed"):
            app.model_for(self.root,cfg)
        category_key=next(iter(app.groups(app.rows(self.root,cfg,"source"))))
        path=app.pool_path(self.root,category_key)
        path.write_bytes(path.read_bytes()+b"tampered")
        with self.assertRaisesRegex(ValueError,"changed after sealing"):
            app.checked_seal(self.root,cfg,"source")

    def test_read_only_v1_reuse_and_old_results_unchanged(self):
        self.fixture.prepare_export()
        old_root=self.fixture.root
        old_files={str(p):file_digest(p) for p in old_root.rglob("*") if p.is_file()}
        self.args.reuse_dir=str(old_root)
        cfg=app.prepare(self.args)
        self.export("source")
        app.fit(self.args)
        self.export("eval")
        for partition in ("source","eval"):
            for row in app.rows(self.root,cfg,partition):
                meta=load_json(app.feature_meta(self.root,row))
                self.assertEqual(meta["origin"],"verified_read_only_v1")
                self.assertTrue(old_root in Path(meta["path"]).parents)
                self.assertFalse((self.root/"features"/(row["id"]+".npz")).exists())
        self.assertEqual(old_files,{str(p):file_digest(p) for p in old_root.rglob("*") if p.is_file()})
        self.predictions("eval")
        self.evaluation("eval")

    def test_reuse_setting_mismatch_and_nested_paths_fail_before_writes(self):
        self.fixture.prepare_export()
        self.args.reuse_dir=str(self.fixture.root)
        self.args.feature_dim=3
        with self.assertRaisesRegex(ValueError,"Reuse setting differs"):
            app.prepare(self.args)
        self.assertFalse(self.root.exists())
        self.args.work_dir=str(self.fixture.root/"nested")
        with self.assertRaisesRegex(ValueError,"separate"):
            app.prepare(self.args)

    def test_configuration_change_preserves_previous_run(self):
        self.prepare()
        sha=file_digest(self.root/"config.json")
        self.args.pool_prior_weight=.8
        with self.assertRaisesRegex(ValueError,"NEW WORK_DIR"):
            self.prepare()
        self.assertEqual(file_digest(self.root/"config.json"),sha)

    def test_reference_content_cannot_cross_classes(self):
        catalog=copy.deepcopy(self.fixture.catalog)
        support=catalog[1]
        first=support[0]
        other=next(r for r in support if app.key(r)!=app.key(first))
        other["rgb_sha"]=first["rgb_sha"]
        with self.assertRaisesRegex(ValueError,"crosses category/dataset"):
            app.prepare(self.args,catalog)
        self.assertFalse(self.root.exists())


@unittest.skipUnless(previous.BASH, "Set SRA_TEST_BASH to Git Bash")
class NormalPoolShellTests(unittest.TestCase):
    def run_script(self, gpus="0,1", fail="NEVER", **flags):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            shutil.copyfile(app.REPO/"run_exp_normal_pool.sh",root/"run.sh")
            (root/"fake_python").write_bytes(b'#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]] || exit 1\nexit 0\n')
            (root/"flock").write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
            for name in ("fake_python","flock"):
                (root/name).chmod(0o755)
            env=os.environ.copy()
            env.update(TRAIN_PYTHON="./fake_python",WORK_DIR="./output",GPU_IDS=gpus,
                       REUSE_DIR="none",FAIL_STAGE=fail)
            env.update(flags)
            env["PATH"]=str(root)+os.pathsep+str(Path(previous.BASH).parent)+os.pathsep+env.get("PATH","")
            return subprocess.run([previous.BASH,"run.sh"],cwd=root,env=env,capture_output=True,text=True,timeout=45)

    def test_order_two_gpu_and_lock_before_gt_evaluation(self):
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        output=result.stdout
        self.assertIn("GPU=1",output)
        self.assertIn("--pool_prior_weight 0.5",output)
        self.assertNotIn("--stage train",output)
        self.assertNotIn("--stage text_bank",output)
        self.assertLess(output.index("--stage fit"),output.index("--stage predict"))
        self.assertLess(output.index("--stage lock_predictions"),output.index("--stage evaluate"))
        self.assertLess(output.rindex("--stage evaluate"),output.index("--stage package"))

    def test_worker_failure_and_duplicate_gpu_do_not_continue(self):
        result=self.run_script(fail="--stage export")
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn("--stage fit",result.stdout)
        self.assertNotIn("--stage predict",result.stdout)
        self.assertNotEqual(self.run_script(gpus="0,0").returncode,0)

    def test_single_gpu_resume_without_export_or_prediction(self):
        result=self.run_script(gpus="0",RUN_EXPORT="0",RUN_PREDICT="0")
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotIn("--stage export",result.stdout)
        self.assertNotIn("--stage predict ",result.stdout)
        self.assertIn("--stage evaluate",result.stdout)


if __name__=="__main__":
    unittest.main()
