#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"""^epoch_(?P<epoch>\d+):\s*
        awareness_loss=(?P<awareness>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*
        \tseg_loss=(?P<seg>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*
        \tglobal_anomaly_loss=(?P<global>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*
        \ttotal_loss=(?P<total>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*$
    """,
    re.VERBOSE,
)


def parse_loss_file(path: Path):
    epochs, awareness, seg, global_loss, total = [], [], [], [], []
    bad_lines = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                bad_lines.append((i, line))
                continue

            epochs.append(int(m.group("epoch")))
            awareness.append(float(m.group("awareness")))
            seg.append(float(m.group("seg")))
            global_loss.append(float(m.group("global")))
            total.append(float(m.group("total")))

    if len(epochs) == 0:
        raise ValueError(f"No valid lines parsed from {path}")

    # sort by epoch in case file order is not strictly increasing
    order = np.argsort(epochs)
    epochs = np.array(epochs)[order]
    awareness = np.array(awareness)[order]
    seg = np.array(seg)[order]
    global_loss = np.array(global_loss)[order]
    total = np.array(total)[order]

    return epochs, awareness, seg, global_loss, total, bad_lines


def moving_average(y: np.ndarray, window: int):
    if window <= 1:
        return y
    window = min(window, len(y))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser(description="Visualize losses from a training log txt file.")
    parser.add_argument("--path", type=str, required=True, help="Path to loss txt file")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: same as txt)")
    parser.add_argument("--smooth", type=int, default=0, help="Moving average window size (e.g., 5). 0/1 = no smoothing")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    txt_path = Path(args.path).expanduser().resolve()
    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else txt_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    epochs, awareness, seg, global_loss, total, bad_lines = parse_loss_file(txt_path)

    # optional smoothing
    if args.smooth and args.smooth > 1:
        awareness_s = moving_average(awareness, args.smooth)
        seg_s = moving_average(seg, args.smooth)
        global_s = moving_average(global_loss, args.smooth)
        total_s = moving_average(total, args.smooth)
    else:
        awareness_s, seg_s, global_s, total_s = awareness, seg, global_loss, total

    # ---- Figure 1: all curves in one plot ----
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, awareness_s, label="awareness_loss")
    plt.plot(epochs, seg_s, label="seg_loss")
    plt.plot(epochs, global_s, label="global_anomaly_loss")
    plt.plot(epochs, total_s, label="total_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves ({txt_path.name})" + (f" | smooth={args.smooth}" if args.smooth and args.smooth > 1 else ""))
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    oneplot_path = outdir / "loss_curves_all_in_one.png"
    plt.tight_layout()
    plt.savefig(oneplot_path, dpi=200)

    # ---- Figure 2: 2x2 subplots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    axes[0].plot(epochs, awareness_s)
    axes[0].set_title("awareness_loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5)

    axes[1].plot(epochs, seg_s)
    axes[1].set_title("seg_loss")
    axes[1].grid(True, linestyle="--", linewidth=0.5)

    axes[2].plot(epochs, global_s)
    axes[2].set_title("global_anomaly_loss")
    axes[2].grid(True, linestyle="--", linewidth=0.5)

    axes[3].plot(epochs, total_s)
    axes[3].set_title("total_loss")
    axes[3].grid(True, linestyle="--", linewidth=0.5)

    for ax in axes[2:]:
        ax.set_xlabel("Epoch")
    for ax in axes[0::2]:
        ax.set_ylabel("Loss")

    fig.suptitle(f"Loss Curves (2x2) - {txt_path.name}" + (f" | smooth={args.smooth}" if args.smooth and args.smooth > 1 else ""))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    subplots_path = outdir / "loss_curves_2x2.png"
    fig.savefig(subplots_path, dpi=200)

    # ---- Report parse issues (if any) ----
    if bad_lines:
        bad_path = outdir / "loss_parse_errors.txt"
        with bad_path.open("w", encoding="utf-8") as f:
            for ln, content in bad_lines:
                f.write(f"Line {ln}: {content}\n")
        print(f"[WARN] {len(bad_lines)} lines could not be parsed. Saved: {bad_path}")

    print(f"[OK] Saved plots:\n  - {oneplot_path}\n  - {subplots_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
