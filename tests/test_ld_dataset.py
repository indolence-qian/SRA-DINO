import csv
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from tools.prepare_ld_dataset import TASKS, convert, read_boxes, validate
from tools.install_ld_dataset import install


class LDDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / "original", self.root / "converted"
        rng = np.random.default_rng(9)
        for task in TASKS:
            directory = self.source / task
            rows = []
            for split, timestamp in (("train", "20260101000000"), ("val", "20260102000000")):
                for index, label in enumerate(("", "5 .25 .25 .1 .1\n", "2 .5 .5 .5 .5\n5 .1 .1 .1 .1\n")):
                    stem = f"Cam2_{timestamp}_{index}_OK"
                    image = directory / "images" / split / (stem + ".jpg")
                    text = directory / "labels" / split / (stem + ".txt")
                    image.parent.mkdir(parents=True, exist_ok=True)
                    text.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rng.integers(0, 255, (40, 80, 3), dtype=np.uint8)).save(image)
                    text.write_text(label, encoding="utf-8")
                    rows.append(dict(split=split, group=f"Cam2_{timestamp}", stem=stem, camera="Cam2",
                                     date=timestamp[:8], source="fixture", boxes=len(label.splitlines())))
            with (directory / "split.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def tearDown(self):
        self.temp.cleanup()

    def test_conversion_semantics_and_real_mixed_batch(self):
        report = convert(self.source, self.output)
        rows = validate(self.output)
        self.assertEqual(report["total_images"], 24)
        self.assertEqual(report["total_masks"], 8)
        defect = next(r for r in rows if r["is_anomaly"])
        with Image.open(self.output / defect["mask_path"]) as image:
            mask = np.array(image)
        self.assertEqual(int((mask > 0).sum()), 40 * 20)
        self.assertEqual(mask[4, 8], 0)  # nuisance rectangle is excluded
        self.assertTrue(defect["filename_label_disagreement"])
        from Datasets.ld import Dataset, DatasetSplit
        for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
            dataset = Dataset(self.output, "ALL", resize=32, imagesize=32, split=split)
            self.assertEqual(len(dataset), 12)
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=3)))
            self.assertEqual(batch["image"].shape, (3, 3, 32, 32))
            self.assertEqual(batch["mask"].dtype, torch.float32)
            self.assertEqual(batch["is_anomaly"].tolist(), [0, 0, 1])
            self.assertGreater(float(batch["mask"][2].sum()), 0)
        with self.assertRaisesRegex(ValueError, "no independent VAL"):
            Dataset(self.output, split=DatasetSplit.VAL)

    def test_damaged_copy_is_rejected(self):
        convert(self.source, self.output)
        path = next(self.output.glob("*/test/defect/*.jpg"))
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate(self.output)

    def test_missing_label_is_not_assumed_normal(self):
        next(self.source.glob("*/labels/train/*.txt")).unlink()
        with self.assertRaisesRegex(ValueError, "Image/label/split.csv mismatch"):
            convert(self.source, self.output)

    def test_duplicate_image_content_is_rejected(self):
        image = next(self.source.glob("*/images/train/*.jpg"))
        target = next(self.source.glob("*/images/val/*.jpg"))
        shutil.copyfile(image, target)
        with self.assertRaisesRegex(ValueError, "Duplicate file image"):
            convert(self.source, self.output)

    def test_invalid_yolo_box_is_rejected(self):
        path = self.root / "bad.txt"
        for line in ("2 nan .5 .2 .2", "2 .1 .5 .4 .2", "2.5 .5 .5 .1 .1", "2 .5 .5 .1"):
            path.write_text(line, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_boxes(path, (80, 40))

    def test_installer_preserves_old_registry_and_is_idempotent(self):
        repo, bundle = self.root / "repo", self.root / "bundle"
        (repo / "Datasets").mkdir(parents=True)
        bundle.mkdir()
        init = repo / "Datasets" / "__init__.py"
        original = 'DATASET_REGISTRY = {"mvtec": (1, 2, "/old/path")}\nDATASET_CLASSES = {"mvtec": {"bottle"}}\n'
        init.write_text(original, encoding="utf-8")
        for name in ("ld.py", "prepare_ld_dataset.py", "check_ld_dataset.py"):
            (bundle / name).write_text("# fixture\n", encoding="utf-8")
        install(repo, bundle)
        first = init.read_bytes()
        install(repo, bundle)
        self.assertEqual(first, init.read_bytes())
        self.assertIn('"/old/path"', init.read_text(encoding="utf-8"))
        self.assertEqual((init.parent / "__init__.py.before_ld.bak").read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
