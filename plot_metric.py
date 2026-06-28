#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


DATASET_RE = re.compile(r"^-+Dataset:\s*(?P<ds>.+?)\s*-+$")
HEADER_RE  = re.compile(r"^Classname\s+(?P<label>.+?)_epoch_(?P<epoch>\d+)\s*$")
MEAN_RE    = re.compile(
    r"^(?P<key>mean|avg|average|overall)\s+"
    r"(?P<auc>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    r"/"
    r"(?P<f1>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*$",
    re.IGNORECASE
)


def parse_metric_txt(path: Path, summary_key: str = "mean"):
    """
    Parse text file that contains repeating blocks:
      ----------Dataset: xxx----------
      Classname    AUC_xxx/F1_xxx_epoch_k
      ...
      mean          0.123/0.456

    Returns:
      data: dict[dataset] -> dict with fields:
            - label: e.g. "AUC_Pixel/F1_Pixel"
            - points: list[(epoch, auc, f1)]
      bad_lines: list[(lineno, content)]
    """
    summary_key = summary_key.lower()
    data = defaultdict(lambda: {"label": None, "points": []})

    current_ds = None
    current_epoch = None
    current_label = None
    bad_lines = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            m_ds = DATASET_RE.match(line)
            if m_ds:
                current_ds = m_ds.group("ds").strip()
                current_epoch = None
                current_label = None
                continue

            m_h = HEADER_RE.match(line)
            if m_h:
                current_label = m_h.group("label").strip()
                current_epoch = int(m_h.group("epoch"))
                if current_ds is not None and data[current_ds]["label"] is None:
                    data[current_ds]["label"] = current_label
                continue

            m_mean = MEAN_RE.match(line)
            if m_mean and current_ds is not None and current_epoch is not None:
                key = m_mean.group("key").lower()
                if key != summary_key:
                    continue
                auc = float(m_mean.group("auc"))
                f1  = float(m_mean.group("f1"))
                data[current_ds]["points"].append((current_epoch, auc, f1))
                continue

            # ignore per-class lines; record only truly unexpected lines
            # (optional: comment out next two lines if you don't want warnings)
            if (line.startswith("----------Dataset:") or line.startswith("Classname")):
                bad_lines.append((lineno, line))

    # sort & deduplicate (keep last if duplicate epoch)
    for ds in list(data.keys()):
        pts = data[ds]["points"]
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda x: x[0])
        dedup = {}
        for ep, auc, f1 in pts_sorted:
            dedup[ep] = (auc, f1)
        data[ds]["points"] = [(ep, dedup[ep][0], dedup[ep][1]) for ep in sorted(dedup.keys())]

    return dict(data), bad_lines


def moving_average(y: np.ndarray, window: int):
    if window is None or window <= 1:
        return y
    window = min(window, len(y))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def plot_one_dataset(ds_name: str, label: str, points, outpath: Path, smooth: int = 0):
    epochs = np.array([p[0] for p in points], dtype=int)
    aucs   = np.array([p[1] for p in points], dtype=float)
    f1s    = np.array([p[2] for p in points], dtype=float)

    if smooth and smooth > 1:
        aucs_s = moving_average(aucs, smooth)
        f1s_s  = moving_average(f1s, smooth)
    else:
        aucs_s, f1s_s = aucs, f1s

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(epochs, aucs_s, label="mean AUC")
    axes[0].set_ylabel("AUC")
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    axes[0].legend()

    axes[1].plot(epochs, f1s_s, label="mean F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    axes[1].legend()

    title = f"{ds_name} | {label} (summary=mean)"
    if smooth and smooth > 1:
        title += f" | smooth={smooth}"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to metric txt file")
    ap.add_argument("--outdir", default=None, help="Output directory (default: same as txt)")
    ap.add_argument("--dataset", default=None, help="Only plot this dataset name (e.g., mvtec)")
    ap.add_argument("--smooth", type=int, default=0, help="Moving average window; 0/1 means no smoothing")
    ap.add_argument("--summary_key", default="mean", help="Summary row keyword: mean/avg/average/overall")
    ap.add_argument("--show", action="store_true", help="Show plot windows (useful on local with GUI)")
    args = ap.parse_args()

    txt_path = Path(args.path).expanduser().resolve()
    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else txt_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    data, bad_lines = parse_metric_txt(txt_path, summary_key=args.summary_key)

    if args.dataset is not None:
        # case-insensitive match
        target = args.dataset.lower()
        data = {k: v for k, v in data.items() if k.lower() == target}

    if not data:
        raise ValueError("No dataset/mean points parsed. Please check file format or summary_key.")

    saved = []
    for ds_name, obj in data.items():
        points = obj["points"]
        if not points:
            continue
        label = obj["label"] or "AUC/F1"
        outpath = outdir / f"{ds_name}_mean_auc_f1.png"
        plot_one_dataset(ds_name, label, points, outpath, smooth=args.smooth)
        saved.append(outpath)

    # Optional CSV export for convenience
    csv_path = outdir / "parsed_mean_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("dataset,epoch,mean_auc,mean_f1\n")
        for ds_name, obj in data.items():
            for ep, auc, f1 in obj["points"]:
                f.write(f"{ds_name},{ep},{auc:.8f},{f1:.8f}\n")

    print("[OK] Saved figures:")
    for p in saved:
        print(f"  - {p}")
    print(f"[OK] Saved parsed CSV: {csv_path}")

    if bad_lines:
        warn_path = outdir / "parse_warnings.txt"
        with warn_path.open("w", encoding="utf-8") as f:
            for ln, content in bad_lines:
                f.write(f"Line {ln}: {content}\n")
        print(f"[WARN] Some unexpected header-like lines were seen. Saved: {warn_path}")

    if args.show:
        # Re-open last dataset plot for display (optional)
        # If you want interactive display of all plots, modify accordingly.
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        last = saved[-1]
        img = mpimg.imread(last)
        plt.figure(figsize=(10, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Preview: {last.name}")
        plt.show()


if __name__ == "__main__":
    main()
