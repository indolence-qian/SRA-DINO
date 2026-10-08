import os
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import upstream_p2 as p2
from tools.upstream_p2 import MatchingHead, component_loss, component_supervision, spatial_matching
from tools.upstream_repair import corrected_probability, constrained_loss
from tools.vlm_review import atomic_json, file_digest, load_json
import test_upstream_localization as v1_tests


class MatchingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cosine_spatial_variation_and_tile_support(self):
        visual = torch.tensor([[[[[1., 0., 1., 0.]], [[0., 1., 0., 1.]]]]])
        emb = torch.tensor([[[[1., 0.], [0., 1.]]]])
        valid = torch.ones(1, 1, 2)
        boxes = torch.tensor([[[0., 0., .5, 1.]]])
        match = spatial_matching(visual*7, emb*3, valid, boxes)
        self.assertEqual(match.shape, (1, 5, 1, 4))
        torch.testing.assert_close(match[0, 0, 0], torch.tensor([1., 0., 0., 0.]))
        torch.testing.assert_close(match[0, 1, 0], torch.tensor([0., 1., 0., 0.]))
        torch.testing.assert_close(match[0, 2, 0], torch.tensor([1., -1., 0., 0.]))
        self.assertTrue(torch.equal(spatial_matching(visual, emb, valid*0, boxes), match*0))

    def test_unpaired_roles_never_fabricate_a_difference(self):
        visual = torch.ones(1, 1, 2, 2, 2)
        emb = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]])
        valid = torch.tensor([[[1., 0.], [0., 1.]]])
        boxes = torch.tensor([[[0., 0., 1., 1.], [0., 0., 1., 1.]]])
        match = spatial_matching(visual, emb, valid, boxes)
        self.assertTrue((match[:, -2:] == 1).all())
        self.assertTrue((match[:, 2] == 0).all())

    def test_head_exact_identity_visual_invariance_and_semantic_gradient(self):
        torch.manual_seed(5)
        head = MatchingHead(2, 8, 8)
        visual, emb = torch.randn(2, 2, 8, 4, 4), torch.randn(2, 1, 2, 8)
        base = torch.full((2, 1, 16, 16), 1e-9)
        valid, boxes = torch.ones(2, 1, 2), torch.tensor([[[0., 0., 1., 1.]]]*2)
        delta = head(visual, base, emb, valid, boxes)
        self.assertTrue(torch.equal(corrected_probability(base, delta), base))
        with torch.no_grad():
            head.decode[-1].weight.normal_(std=.1)
        a = head(visual, base, emb, valid, boxes, use_semantics=False)
        b = head(visual, base, emb*999, valid, boxes, use_semantics=False)
        self.assertTrue(torch.equal(a, b))
        delta = head(visual, base, emb, valid, boxes)
        delta.square().mean().backward()
        self.assertGreater(float(head.semantic[0].weight.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(corrected_probability(base, delta, 0), base))


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_component_balance_small_regions_get_stronger_pixel_gradient(self):
        base = torch.full((1, 1, 32, 32), 1e-9)
        mask = torch.zeros_like(base)
        mask[0, 0, 3, 3:5] = 1
        mask[0, 0, 20:24, 20:24] = 1
        labels = component_supervision(mask[0, 0].numpy())[None]
        delta = torch.zeros_like(base, requires_grad=True)
        _, terms = component_loss(base, delta, mask, labels, small_fraction=.01)
        terms["component"].sum().backward()
        small = float(delta.grad[0, 0, 3, 3:5].abs().mean())
        large = float(delta.grad[0, 0, 20:24, 20:24].abs().mean())
        self.assertGreater(small, large*10)
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_ranking_suppresses_near_background_and_lifts_gt(self):
        base = torch.full((1, 1, 16, 16), .1)
        mask = torch.zeros_like(base)
        mask[0, 0, 7:9, 7:9] = 1
        labels = component_supervision(mask[0, 0].numpy())[None]
        delta = torch.zeros_like(base, requires_grad=True)
        _, terms = component_loss(base, delta, mask, labels, near_radius=2)
        terms["ranking"].sum().backward()
        self.assertLess(float(delta.grad[mask > 0].mean()), 0)
        self.assertGreater(float(delta.grad[0, 0, 6, 7:9].sum()), 0)
        self.assertEqual(float(delta.grad[0, 0, 0, 0]), 0)

    def test_normal_and_ignored_tiny_regions_have_no_extra_loss(self):
        base = torch.zeros(2, 1, 16, 16)
        mask = torch.zeros_like(base)
        mask[1, 0, 1, 1] = 1
        labels = torch.stack([component_supervision(gt.numpy(), min_area=2) for gt in mask[:, 0]])
        delta = torch.zeros_like(base, requires_grad=True)
        loss, terms = component_loss(base, delta, mask, labels)
        self.assertTrue((terms["component"] == 0).all())
        self.assertTrue((terms["ranking"] == 0).all())
        torch.testing.assert_close(loss, constrained_loss(base, delta, mask)[0])
        loss.sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())


class PipelineTests(unittest.TestCase):
    review = v1_tests.PipelineTests.review

    def setUp(self):
        v1_tests.PipelineTests.setUp(self)
        v1_tests.PipelineTests.features(self)
        self.source = self.root
        self.output_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_temp.cleanup)
        self.out = Path(self.output_temp.name)
        self.a = p2.parser().parse_args(["--stage", "prepare", "--work_dir", str(self.out), "--source_work_dir", str(self.source),
                                       "--num_shards", "1", "--epochs", "2", "--batch_size", "2", "--hidden", "8", "--workers", "0"])
        p2.prepare(self.a)

    def test_full_source_only_pipeline_controls_resume_and_readonly_cache(self):
        snapshot = {str(p.relative_to(self.source)): file_digest(p) for p in self.source.rglob("*") if p.is_file()}
        original = p2.old.mask_for
        def no_target(r, size):
            self.assertNotEqual(r["partition"], "eval")
            return original(r, size)
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(p2.old, "mask_for", side_effect=no_target):
                p2.quality(self.a)
                p2.matching(self.a)
                for loss in p2.LOSSES:
                    self.a.loss = loss
                    for mode in p2.MODES:
                        self.a.mode = mode
                        # A strict fallback exercises identity despite training.
                        score = p2.selection_score
                        def only_base(c, b, *args):
                            return score(c, b, *args) if c is b else None
                        with patch.object(p2, "selection_score", side_effect=only_base):
                            p2.train(self.a)
                            path = self.out/"heads"/loss/mode/"last.pt"
                            checksum = file_digest(path)
                            p2.train(self.a)
                            self.assertEqual(checksum, file_digest(path))
                        saved = torch.load(path, weights_only=False)
                        self.assertEqual(saved["alpha"], 0)
                        terms = saved["history"][0]["loss_terms"]
                        self.assertEqual(terms["component"] == 0, loss == "p2a")
            p2.evaluate(self.a)
            p2.report(self.a)
            with patch.object(p2, "P2Dataset", side_effect=AssertionError("Completed evaluation must resume")):
                p2.evaluate(self.a)
        rows = load_json(self.out/"results/summary.json")
        self.assertTrue(all(r["changed_fraction"] == 0 for r in rows if r["intervention"] == "selected_alpha"))
        self.assertTrue(any(r["changed_fraction"] > 0 for r in rows if r["intervention"] == "alpha1_DIAGNOSTIC"))
        with (self.out/"results/comparisons.csv").open(encoding="utf-8-sig") as stream:
            comparisons = list(csv.DictReader(stream))
        self.assertEqual({r["contrast"] for r in comparisons}, {"head_vs_base", "real_vs_visual", "real_vs_shuffled", "p2b_vs_p2a"})
        self.assertTrue(all(float(r["delta_PRO_exact_pp"]) == 0 for r in comparisons))
        self.assertEqual(snapshot, {str(p.relative_to(self.source)): file_digest(p) for p in self.source.rglob("*") if p.is_file()})

    def test_shuffled_bank_partition_category_roles_and_self_exclusion(self):
        _, source, cfg, s = p2.context(self.a)
        dataset = p2.P2Dataset(source, cfg, self.rows, s, "shuffled")
        by_id = {r["id"]: r for r in self.rows}
        for donor in dataset.donors:
            query, reference = by_id[donor["image_id"]], by_id[donor["donor_id"]]
            self.assertNotEqual(query["id"], reference["id"])
            self.assertEqual((query["partition"], query["category"]), (reference["partition"], reference["category"]))
        self.assertTrue(all(v["changed_slots"] > 0 for v in dataset.audit))

    def test_quality_resume_preserves_manual_annotations_and_missing_trace_is_labeled(self):
        p2.quality(self.a)
        path = self.out/"quality/manual_review.csv"
        path.write_text("human completed annotations", encoding="utf-8")
        p2.quality(self.a)
        self.assertEqual(path.read_text(encoding="utf-8"), "human completed annotations")
        self.assertIn("缺少实际 post-vision trace", (self.out/"quality/index.html").read_text(encoding="utf-8"))
        self.assertEqual(load_json(self.out/"quality/complete.json")["human_accuracy"], "NOT_MEASURED")

    def test_actual_teacher_inputs_are_copied_and_hash_checked(self):
        for path in (self.source/"traces").rglob("trace.json"):
            trace = load_json(path)
            files = []
            for index in range(2):
                image = path.parent/f"post_vision_{index}.png"
                Image.new("RGB", (28, 28), (index*100, 10, 20)).save(image)
                files.append(dict(file=image.name, sha256=file_digest(image)))
            trace["processed_images"] = files
            atomic_json(path, trace)
        p2.quality(self.a)
        self.assertNotIn("缺少实际 post-vision trace", (self.out/"quality/index.html").read_text(encoding="utf-8"))
        sample = next((self.out/"quality/samples").iterdir())
        self.assertTrue(load_json(sample/"review.json")["exact_post_vision_inputs"])
        with Image.open(sample/"source_gt_crop.png") as im:
            self.assertEqual(im.size, (28, 28))
        trace_dir = self.source/load_json(sample/"review.json")["review"]["attempts"][-1]["trace"]
        (trace_dir/"post_vision_1.png").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "trace image changed"):
            p2.quality(self.a)

    def test_source_selected_nonzero_alpha_and_interrupted_best_is_preserved(self):
        with patch("torch.cuda.is_available", return_value=False):
            p2.quality(self.a)
            p2.matching(self.a)
            self.a.loss, self.a.mode = "p2a", "real"
            with patch.object(p2, "selection_score", side_effect=lambda c, b, *args: 0. if c is b else 1.):
                p2.train(self.a)
            path = self.out/"heads/p2a/real/best.pt"
            selected = torch.load(path, weights_only=False)
            self.assertEqual(selected["selection"], "SOURCE_VALIDATED")
            self.assertEqual(selected["alpha"], .1)
            checksum = file_digest(path)
            # Simulate a crash after best was written but before last was saved.
            (self.out/"heads/p2a/real/last.pt").unlink()
            with patch.object(p2, "selection_score", side_effect=lambda c, b, *args: 0. if c is b else .5):
                p2.train(self.a)
            self.assertEqual(file_digest(path), checksum)

    def test_stage_order_and_changed_settings_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            p2.matching(self.a)
        with self.assertRaises(FileNotFoundError):
            p2.train(self.a)
        self.a.lr = .02
        with self.assertRaisesRegex(ValueError, "NEW WORK_DIR"):
            p2.prepare(self.a)
        self.a.work_dir = str(self.source/"child")
        with self.assertRaisesRegex(ValueError, "non-nested"):
            p2.prepare(self.a)

    def test_mutated_cache_is_rejected(self):
        path = self.source/"features/0.npz"
        with path.open("ab") as stream:
            stream.write(b"mutation")
        with self.assertRaisesRegex(ValueError, "modified after sealing"):
            p2.quality(self.a)


BASH = os.environ.get("SRA_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "Set SRA_TEST_BASH on Windows")
class ShellTests(unittest.TestCase):
    def test_stage_order_six_ddp_controls_and_fail_stop(self):
        for fail in ("NEVER", "--stage matching"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                shutil.copyfile(p2.old.REPO/"run_exp_upstream_p2.sh", root/"run.sh")
                (root/"source").mkdir()
                (root/"source/sealed.json").write_text("{}")
                (root/"fake_python").write_text('#!/usr/bin/env bash\necho "GPU=${CUDA_VISIBLE_DEVICES:-none} ARGS=$*"\n[[ "$*" != *"${FAIL_STAGE:-NEVER}"* ]]\n', newline="\n")
                (root/"flock").write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
                (root/"fake_python").chmod(0o755)
                (root/"flock").chmod(0o755)
                env = dict(os.environ, TRAIN_PYTHON="./fake_python", SOURCE_WORK_DIR="./source", WORK_DIR="./output", GPU_IDS="0,1", FAIL_STAGE=fail)
                env["PATH"] = ".:"+env.get("PATH", "") if os.name != "nt" else str(root)+os.pathsep+env.get("PATH", "")
                result = subprocess.run([BASH, "run.sh"], cwd=root, env=env, text=True, capture_output=True)
                output = result.stdout+result.stderr
                if fail == "NEVER":
                    self.assertEqual(result.returncode, 0, output)
                    stages = [output.index("--stage "+s) for s in ("prepare", "quality", "matching", "train", "evaluate", "report")]
                    self.assertEqual(stages, sorted(stages))
                    self.assertEqual(output.count("--stage train"), 6)
                    self.assertEqual(output.count("--nproc_per_node=2"), 6)
                    self.assertIn("GPU=0 ARGS=upstream_p2.py --stage evaluate", output)
                    self.assertIn("GPU=1 ARGS=upstream_p2.py --stage evaluate", output)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("--stage train", output)
                    self.assertNotIn("--stage report", output)


if __name__ == "__main__":
    unittest.main()
