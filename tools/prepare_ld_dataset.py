"""Convert the supplied LD YOLO asset to an audited anomaly-detection dataset.

Only the original train/val split and authoritative YOLO labels are used.
The generated rectangular masks are BOX annotations, not segmentation truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw


TASKS = {
    "齿面_20261113": "gear_teeth",
    "轴面26_20261113": "shaft_cam26",
    "轴面5only_20261113": "shaft_cam5",
    "轴面9only_20261113": "shaft_cam9",
}
CLASS_NAMES = ("heipi", "wmc", "kepeng", "shengxiu", "youwu", "bscyichang", "queliao")
NUISANCE_CLASS = 5


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_boxes(path, size):
    width, height = size
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(f"Expected YOLO class/cx/cy/w/h: {path}:{number}")
        class_value, cx, cy, bw, bh = map(float, tokens)
        values = (class_value, cx, cy, bw, bh)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite annotation: {path}:{number}")
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < len(CLASS_NAMES) or min(bw, bh) <= 0:
            raise ValueError(f"Invalid class or box: {path}:{number}")
        xyxy = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        if min(xyxy) < -1e-6 or max(xyxy) > 1 + 1e-6:
            raise ValueError(f"Box outside image: {path}:{number}")
        x1, y1, x2, y2 = xyxy
        bounds = [max(0, math.floor(x1 * width)), max(0, math.floor(y1 * height)),
                  min(width, math.ceil(x2 * width)), min(height, math.ceil(y2 * height))]
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError(f"Empty pixel box: {path}:{number}")
        result.append(dict(class_id=class_id, class_name=CLASS_NAMES[class_id],
                           is_defect=class_id != NUISANCE_CLASS, yolo=[cx, cy, bw, bh],
                           pixel_xyxy_exclusive=bounds))
    return result


def box_mask(size, boxes):
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    for box in boxes:
        if box["is_defect"]:
            x1, y1, x2, y2 = box["pixel_xyxy_exclusive"]
            mask[y1:y2, x1:x2] = 255
    return mask


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be separate from the original asset")
    if destination.exists():
        raise FileExistsError(f"Use a new output directory: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    manifests, summaries, identities, rgb_identities = [], [], defaultdict(list), defaultdict(list)
    destination.mkdir(parents=True)
    for task_name, category in TASKS.items():
        task = source / task_name
        csv_path = task / "split.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = [(r["split"], r["stem"]) for r in rows]
        if len(keys) != len(set(keys)) or {r["split"] for r in rows} != {"train", "val"}:
            raise ValueError(f"Invalid/duplicate CSV records: {csv_path}")
        groups = {s: {r["group"] for r in rows if r["split"] == s} for s in ("train", "val")}
        if groups["train"] & groups["val"]:
            raise ValueError(f"Acquisition group leaks across splits: {task_name}")
        images = {(p.parent.name, p.stem): p for p in task.glob("images/*/*") if p.suffix.lower() == ".jpg"}
        labels = {(p.parent.name, p.stem): p for p in task.glob("labels/*/*.txt")}
        if set(keys) != set(images) or set(images) != set(labels):
            raise ValueError(f"Image/label/split.csv mismatch: {task_name}")
        annotation_dir = destination / "annotations" / category
        annotation_dir.mkdir(parents=True)
        shutil.copyfile(csv_path, annotation_dir / "original_split.csv")
        counters = {s: Counter() for s in ("train", "val")}
        sizes = {s: Counter() for s in ("train", "val")}
        box_counts = {s: Counter() for s in ("train", "val")}
        for split in ("train", "test"):
            for state in ("good", "defect"):
                (destination / category / split / state).mkdir(parents=True)
        for row in sorted(rows, key=lambda r: (r["split"], r["stem"])):
            original_split, stem = row["split"], row["stem"]
            key = original_split, stem
            image_path, label_path = images[key], labels[key]
            with Image.open(image_path) as image:
                image.load()  # Decode every image, rather than just checking headers.
                size = image.size
                rgb = image.convert("RGB")
                rgb_sha = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
            boxes = read_boxes(label_path, size)
            if len(boxes) != int(row["boxes"]):
                raise ValueError(f"CSV/YOLO box count differs: {image_path}")
            camera = stem.split("_", 1)[0]
            expected_group = re.match(r"(Cam\d+_\d{14})_", stem)
            if row["camera"] != camera or expected_group is None or expected_group[1] != row["group"]:
                raise ValueError(f"CSV/filename acquisition group differs: {image_path}")
            anomaly = any(b["is_defect"] for b in boxes)
            output_split, state = ("train" if original_split == "train" else "test"), ("defect" if anomaly else "good")
            relative_image = Path(category) / output_split / state / image_path.name
            output_image = destination / relative_image
            shutil.copyfile(image_path, output_image)
            image_sha = sha256(image_path)
            if sha256(output_image) != image_sha:
                raise ValueError(f"Image copy checksum differs: {image_path}")
            relative_mask = None
            mask_sha = None
            if anomaly:
                mask_dir = Path(category) / "ground_truth"
                if output_split == "train":
                    mask_dir /= "train"
                relative_mask = mask_dir / "defect" / (stem + "_mask.png")
                output_mask = destination / relative_mask
                output_mask.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(box_mask(size, boxes)).save(output_mask)
                mask_sha = sha256(output_mask)
            relative_label = Path("annotations") / category / original_split / label_path.name
            (destination / relative_label).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(label_path, destination / relative_label)
            suffix_label = stem.rsplit("_", 1)[-1]
            mismatch = suffix_label in ("OK", "NG") and (suffix_label == "NG") != anomaly
            nuisance_only = bool(boxes) and not anomaly
            record = dict(category=category, source_task=task_name, original_split=original_split,
                          split=output_split, group=row["group"], camera=row["camera"], date=row["date"],
                          source=row["source"], stem=stem, image_path=relative_image.as_posix(),
                          mask_path=relative_mask.as_posix() if relative_mask else None,
                          label_path=relative_label.as_posix(), native_size=list(size), is_anomaly=int(anomaly),
                          nuisance_only=nuisance_only, boxes=boxes, filename_label_disagreement=mismatch,
                          image_sha256=image_sha, rgb_sha256=rgb_sha, label_sha256=sha256(label_path),
                          mask_sha256=mask_sha, annotation_kind="bounding_box")
            manifests.append(record)
            identities[image_sha].append(record)
            rgb_identities[rgb_sha].append(record)
            counters[original_split].update(images=1, abnormal=int(anomaly), normal=int(not anomaly),
                                            nuisance_only=int(nuisance_only), filename_label_disagreements=int(mismatch))
            sizes[original_split][f"{size[0]}x{size[1]}"] += 1
            box_counts[original_split].update(b["class_name"] for b in boxes)
        for split in ("train", "val"):
            summaries.append(dict(category=category, original_split=split, split="train" if split == "train" else "test",
                                  **dict(counters[split]), groups=len(groups[split]),
                                  sizes=dict(sizes[split]), boxes=dict(box_counts[split])))
        print(f"Converted {category}: {len(rows)} images", flush=True)
    for kind, mapping in (("file", identities), ("decoded RGB", rgb_identities)):
        duplicates = [v for v in mapping.values() if len(v) > 1]
        if duplicates:
            raise ValueError(f"Duplicate {kind} image content; review before packaging: {duplicates[0][0]['image_path']}")
    original_readme = source / "数据集说明.md"
    if original_readme.is_file():
        shutil.copyfile(original_readme, destination / "annotations" / "original_dataset_notes.md")
    report = dict(version=1, dataset="ld", name=destination.name, annotation_kind="bounding_box",
                  image_label_source="YOLO labels, excluding class 5; filename OK/NG is never used",
                  split_policy="Original train -> train; original val -> test; no resampling",
                  nuisance_class=NUISANCE_CLASS, task_mapping=TASKS, classes=dict(enumerate(CLASS_NAMES)),
                  total_images=len(manifests), total_masks=sum(r["is_anomaly"] for r in manifests),
                  group_overlap=0, duplicate_image_files=0, duplicate_decoded_rgb=0, categories=summaries,
                  warnings=["Rectangular masks are box-level proxies, not pixel-accurate segmentation ground truth.",
                            "shaft_cam5 TEST has no normal images; image AUROC is undefined, not zero.",
                            "The independent good-product false-positive test set mentioned in the source README is absent.",
                            "Empty labels/nuisance-only images are annotation-normal, not independently certified good parts.",
                            "The original README Cam9 train count is stale: actual count is 39, not 36."])
    write_json(destination / "manifest.json", manifests)
    write_json(destination / "audit_report.json", report)
    return report


def validate(root):
    """Full independent disk validation; requires only Pillow and NumPy."""
    root = Path(root).resolve()
    rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    seen, split_groups = set(), defaultdict(set)
    for row in rows:
        image_path = root / row["image_path"]
        label_path = root / row["label_path"]
        if row["image_path"] in seen:
            raise ValueError(f"Duplicate manifest image: {image_path}")
        seen.add(row["image_path"])
        if sha256(image_path) != row["image_sha256"] or sha256(label_path) != row["label_sha256"]:
            raise ValueError(f"Image/label checksum mismatch: {image_path}")
        with Image.open(image_path) as image:
            image.load()
            size = image.size
            rgb = image.convert("RGB")
            rgb_sha = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        if list(size) != row["native_size"] or rgb_sha != row["rgb_sha256"]:
            raise ValueError(f"Image geometry/content mismatch: {image_path}")
        boxes = read_boxes(label_path, size)
        if boxes != row["boxes"] or int(any(b["is_defect"] for b in boxes)) != row["is_anomaly"]:
            raise ValueError(f"Annotation/image label mismatch: {image_path}")
        expected_split = "train" if row["original_split"] == "train" else "test"
        if row["split"] != expected_split or image_path.parent.name != ("defect" if row["is_anomaly"] else "good"):
            raise ValueError(f"Split/directory mismatch: {image_path}")
        split_groups[row["category"], row["split"]].add(row["group"])
        if row["is_anomaly"]:
            path = root / row["mask_path"]
            if sha256(path) != row["mask_sha256"]:
                raise ValueError(f"Mask checksum mismatch: {path}")
            with Image.open(path) as image:
                if image.mode != "L" or image.size != size:
                    raise ValueError(f"Mask mode/geometry mismatch: {path}")
                actual = np.asarray(image)
            expected = box_mask(size, boxes)
            if not actual.any() or not np.array_equal(actual, expected):
                raise ValueError(f"Mask does not match source boxes: {path}")
        elif row["mask_path"] is not None:
            raise ValueError(f"Normal image has an anomaly mask: {image_path}")
    for category in {r["category"] for r in rows}:
        if split_groups[category, "train"] & split_groups[category, "test"]:
            raise ValueError(f"Acquisition group leakage: {category}")
    actual_images = {p.relative_to(root).as_posix() for category in TASKS.values()
                     for p in (root / category).glob("*/*/*.jpg")}
    if seen != actual_images:
        raise ValueError("Manifest does not cover exactly the dataset images")
    actual_masks = {p.relative_to(root).as_posix() for category in TASKS.values()
                    for p in (root / category / "ground_truth").rglob("*.png")}
    if actual_masks != {r["mask_path"] for r in rows if r["mask_path"]}:
        raise ValueError("Manifest does not cover exactly the masks")
    print(f"PASS: {len(rows)} images, {len(actual_masks)} masks, labels, hashes, geometry and split isolation", flush=True)
    return rows


def preview(root):
    root = Path(root)
    rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = []
    for category in TASKS.values():
        for anomaly in (False, True):
            eligible = [r for r in rows if r["category"] == category and bool(r["is_anomaly"]) == anomaly]
            # Prefer a test sample; Cam5 has no normal TEST, so show a TRAIN normal explicitly.
            eligible.sort(key=lambda r: (r["split"] != "test", r["image_path"]))
            samples.append(eligible[0])
    canvas = Image.new("RGB", (960, 4 * 300), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(samples):
        with Image.open(root / row["image_path"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((470, 265))
        sx, sy = thumb.width / row["native_size"][0], thumb.height / row["native_size"][1]
        overlay = ImageDraw.Draw(thumb)
        for box in row["boxes"]:
            x1, y1, x2, y2 = box["pixel_xyxy_exclusive"]
            overlay.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="red" if box["is_defect"] else "cyan", width=2)
        x, y = (i % 2) * 480 + 5, (i // 2) * 300
        draw.text((x, y + 4), f"{row['category']} / {row['split']} / {'defect' if row['is_anomaly'] else 'good'}", fill="black")
        canvas.paste(thumb, (x, y + 28))
    canvas.save(root / "annotation_preview.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        if args.source is None:
            parser.error("--source is required for conversion")
        convert(args.source, args.output)
    validate(args.output)
    preview(args.output)


if __name__ == "__main__":
    main()
