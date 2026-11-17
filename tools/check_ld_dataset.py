"""Verify LD disk integrity and exercise the real PyTorch Dataset/DataLoader."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(root, image_size=512, workers=0, full_batches=False, disk_audit=True):
    import torch
    torch.set_num_threads(2)
    from tools.prepare_ld_dataset import validate
    if disk_audit:
        validate(root)
    os.environ["LD_DATA_ROOT"] = str(Path(root).resolve())
    from Datasets import DATASET_CLASSES, DATASET_REGISTRY
    cls, splits, configured_root = DATASET_REGISTRY["ld"]
    if Path(configured_root).resolve() != Path(root).resolve():
        raise ValueError("LD_DATA_ROOT was not applied; check registration before testing")
    totals = Counter()
    for split in (splits.TRAIN, splits.TEST):
        for category in sorted(DATASET_CLASSES["ld"]):
            dataset = cls(source=configured_root, classname=category, split=split,
                          resize=image_size, imagesize=image_size)
            labels = Counter(state for _, state, _, _ in dataset.data_to_iterate)
            if not full_batches:
                # A mixed normal/abnormal batch where possible, using the actual loader.
                selected = [next(i for i, e in enumerate(dataset.data_to_iterate) if e[1] == state)
                            for state in sorted(labels)]
                samples = torch.utils.data.Subset(dataset, selected)
            else:
                samples = dataset
            loader = torch.utils.data.DataLoader(samples, batch_size=4, num_workers=workers, shuffle=False)
            checked = 0
            for batch in loader:
                count = len(batch["image"])
                if tuple(batch["image"].shape) != (count, 3, image_size, image_size):
                    raise ValueError("Unexpected image batch shape")
                if tuple(batch["mask"].shape) != (count, 1, image_size, image_size):
                    raise ValueError("Unexpected mask batch shape")
                if batch["mask"].dtype != torch.float32 or not bool(torch.isfinite(batch["image"]).all()):
                    raise ValueError("Invalid image/mask tensor")
                if not bool(((batch["mask"] == 0) | (batch["mask"] == 1)).all()):
                    raise ValueError("Mask is not binary")
                for label, mask in zip(batch["is_anomaly"], batch["mask"]):
                    if int(label) == 0 and bool(mask.any()):
                        raise ValueError("Normal image has foreground mask")
                checked += count
            totals[split.value] += len(dataset)
            print(f"PASS {split.value}/{category}: {len(dataset)} images, labels={dict(labels)}, batched={checked}", flush=True)
            if split is splits.TEST and len(labels) < 2:
                print("  NOTE: image-level AUROC is undefined for this single-label category; legacy tests may print 0.", flush=True)
    print(f"PASS real PyTorch loading: {dict(totals)}; image size={image_size}, workers={workers}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("LD_DATA_ROOT", "/mnt/qfg/Tate_qfg/data/LD_AD_20261116")))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--full-batches", action="store_true")
    parser.add_argument("--skip-disk-audit", action="store_true")
    args = parser.parse_args()
    check(args.data_root, args.image_size, args.workers, args.full_batches, not args.skip_disk_audit)
