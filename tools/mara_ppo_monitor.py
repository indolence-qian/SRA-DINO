"""Offline/live scientific diagnostics. Never select a checkpoint or use targets.

No TensorBoard/web service required: Agg PNGs and a self-contained local HTML
index, durable per-epoch telemetry, fixed source-only monitoring and image grids.
"""
from collections import defaultdict
from contextlib import contextmanager
import csv
import html
import io
import json
import math
import os
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image
import torch

import mara_diagnose as shared
from tools.mara_diagnostics import map_metrics, stable_id
from tools.mara_ppo import INPUTS, score_trajectory
from tools.mara_reward import state_hash

SCHEMA = "ppo_observability_v1"


@contextmanager
def diagnostic_context(agent):
    """Observation must not consume any training RNG or change module modes."""
    python, numpy = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in agent.modules()]
    devices = sorted({p.device.index for p in agent.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            agent.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        for module, mode in modes:
            module.training = mode


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def atomic_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem+f".{os.getpid()}.tmp.png")
    try:
        fig.savefig(temporary, dpi=115, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        pyplot().close(fig)


def atomic_html(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_history(root, job):
    rows = read_csv(Path(root)/"jobs"/job["id"]/"training.csv")
    return rows or [r for r in read_csv(Path(root)/"results"/"training.csv") if r["job"] == job["id"]]


def number(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else float("nan")
    except (KeyError, ValueError, TypeError):
        return float("nan")


def panels(path, rows, groups, x="epoch", title="", group_key=None):
    plt = pyplot()
    fig, axes = plt.subplots(math.ceil(len(groups)/2), 2, figsize=(13, 3.2*math.ceil(len(groups)/2)), squeeze=False)
    series = defaultdict(list)
    for row in rows:
        series[str(row.get(group_key, "")) if group_key else ""].append(row)
    for ax, (label, keys) in zip(axes.flat, groups):
        plotted = False
        for name, values in sorted(series.items()):
            for key in keys:
                points = [(number(r, x), number(r, key)) for r in values]
                points = [(a, b) for a, b in points if math.isfinite(a) and math.isfinite(b)]
                if points:
                    xx, yy = zip(*points)
                    ax.plot(xx, yy, linewidth=1.1, label=(name+" " if name else "")+key,
                            marker="." if x == "epoch" else None, markersize=3)
                    plotted = True
        ax.set(title=label, xlabel=x)
        ax.grid(alpha=.2)
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "Not recorded / not applicable", ha="center", transform=ax.transAxes)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    atomic_figure(fig, path)


class EpochTelemetry:
    """Live JSONL + atomic committed epoch CSVs; restart truncates only its own
    uncommitted epoch, so an interrupted update is never counted twice.
    """
    def __init__(self, directory, job, epoch):
        self.root = Path(directory)/"monitor"
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch, self.job = epoch, job
        self.path = self.root/f"live_epoch_{epoch:03d}.jsonl"
        self.stream = self.path.open("w", encoding="utf-8")
        self.rows = {"batch": [], "update": [], "transition": []}

    def record(self, kind, row):
        row = dict(job=self.job["id"], arm=self.job["arm"], seed=self.job["seed"], epoch=self.epoch, **row)
        self.rows[kind].append(row)
        self.stream.write(json.dumps(dict(kind=kind, **row), allow_nan=False)+"\n")
        self.stream.flush()

    def close(self):
        self.stream.close()

    def commit(self):
        for kind, rows in self.rows.items():
            shared.write_csv(self.root/"epochs"/f"{self.epoch:03d}_{kind}.csv", rows)
        shared.atomic_json(self.root/"epochs"/f"{self.epoch:03d}_complete.json", dict(epoch=self.epoch, schema=SCHEMA))


def monitoring_selection(manifest, split, audited, per_category=8, preview_count=8):
    holdout = set(split["calibration_ids"])
    labels = {r["sample_id"]: r["image_label"] for r in audited["rows"] if r["included"]}
    strata = defaultdict(lambda: defaultdict(list))
    for record in manifest:
        if record["id"] in holdout:
            if record["dataset"] != split["source"]:
                raise ValueError("Target forbidden in intermediate monitoring")
            strata[record["category"]][labels[record["id"]]].append(record)
    selected = []
    for category, buckets in sorted(strata.items()):
        buckets = {k: sorted(v, key=lambda r: stable_id(SCHEMA, r["id"])) for k, v in buckets.items()}
        count = 0
        while any(buckets.values()) and (per_category == 0 or count < per_category):
            for label in (0, 1):
                if buckets.get(label) and (per_category == 0 or count < per_category):
                    selected.append(buckets[label].pop(0))
                    count += 1
    # Category-round-robin and normal/anomaly interleaving; never rank examples
    # by reward, quality, PRO or whether they "look good".
    previews = []
    for label in (0, 1):
        candidates = defaultdict(list)
        for record in selected:
            if labels[record["id"]] == label:
                candidates[record["category"]].append(record)
        quota = math.ceil(preview_count/2) if label == 0 else preview_count//2
        chosen = 0
        while any(candidates.values()) and chosen < quota:
            for records in candidates.values():
                if records and chosen < quota:
                    previews.append(records.pop(0)["id"])
                    chosen += 1
    # Fill shortages without duplicating a sample.
    previews += [r["id"] for r in selected if r["id"] not in previews][:max(0, preview_count-len(previews))]
    return selected, previews[:preview_count]


def preview_grid(path, record, batch, result, index, epoch, job):
    plt = pyplot()
    from matplotlib.patches import Rectangle
    with Image.open(record["image_path"]) as source:
        image = np.asarray(source.convert("RGB").resize((256, 256)))
    mask = batch["mask"][index, 0].cpu().numpy()
    base = batch["base_prob"][index, 1].cpu().numpy()
    steps = result["steps"]
    fig, axes = plt.subplots(3, 3+len(steps), figsize=(2.7*(3+len(steps)), 8.0), squeeze=False)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original (display resized)")
    axes[0, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("GT (reward only)")
    axes[0, 2].imshow(base, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("Base P; scale [0,1]")
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("ROI boxes on image")
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(Image.fromarray((mask*255).astype(np.uint8)).resize((256, 256)), cmap="Greens", alpha=.35, vmin=0, vmax=255)
    axes[1, 1].set_title("GT overlay")
    axes[1, 2].imshow(base, cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("Base (same scale)")
    axes[2, 2].imshow(np.zeros_like(base), cmap="viridis", vmin=0, vmax=1)
    axes[2, 2].set_title("Base cumulative C=0")
    axes[2, 0].text(0, .65, "Row 1: probability P [0,1]\nRow 2: actual step delta P\nRow 3: cumulative spatial gate C [0,1]\n\nROI colors: cyan / yellow / red", fontsize=9)
    metadata = []
    for t, step in enumerate(steps):
        p = step["after_hr"][index, 1].cpu().numpy()
        delta = p-step["before_hr"][index, 1].cpu().numpy()
        active, op = bool(step["valid"][index]), int(step["actions"]["op"][index])
        label = "inactive" if not active else "REFINE" if op else "STOP"
        reward, value, ret = [float(step[k][index]) for k in ("reward", "old_value", "return")]
        axes[0, 3+t].imshow(p, cmap="magma", vmin=0, vmax=1)
        region, layer = int(step["actions"]["region"][index]), int(step["actions"]["layer"][index])
        axes[0, 3+t].set_title(f"Step {t+1}: {label}; R{region} L{layer}\nr={reward:+.4f}; V={value:+.4f}\nG_greedy={ret:+.4f}", fontsize=9)
        limit = max(.01, float(np.abs(delta).max()))
        axes[1, 3+t].imshow(delta, cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[1, 3+t].set_title(f"Actual delta P; +/-{limit:.3f}")
        gate = step["cumulative_gate"][index, 0].cpu().numpy()
        axes[2, 3+t].imshow(gate, cmap="viridis", vmin=0, vmax=1)
        axes[2, 3+t].set_title(f"Cumulative C; max={gate.max():.3f}")
        box = step["box"][index].cpu().tolist()
        if active and op:
            y1, x1, y2, x2 = box
            axes[1, 0].add_patch(Rectangle((x1*255, y1*255), (x2-x1)*255, (y2-y1)*255,
                                            fill=False, linewidth=1.5, edgecolor=("cyan", "yellow", "red")[t % 3]))
        metadata.append(dict(step=t+1, status=label, reward=reward, v_prediction=value, greedy_return=ret,
                             box=box, layer=int(step["actions"]["layer"][index]), region=int(step["actions"]["region"][index])))
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{job['id']} | epoch {epoch} | SOURCE HOLDOUT {record['category']} | {record['id']}\n"
                 "GT is post-hoc only. V estimates stochastic returns, not this one greedy trajectory.", fontsize=10)
    fig.tight_layout()
    atomic_figure(fig, path)
    return metadata


def monitor_epoch(directory, agent, cfg, manifest, split, audited, store, job, epoch, every=5, per_category=8, previews=8):
    root = Path(directory)/"monitor"
    selected, preview_ids = monitoring_selection(manifest, split, audited, per_category, previews)
    identity = dict(schema=SCHEMA, signature=shared.digest(cfg), split_sha256=shared.digest(split), job=job,
                    selected_ids=[r["id"] for r in selected], preview_ids=preview_ids,
                    note="Fixed source-only MARA holdout; no target monitoring/selection")
    config_path = root/"selection.json"
    if config_path.exists() and shared.load_json(config_path) != identity:
        raise ValueError("Monitoring subset changed; do not mix validation populations")
    shared.atomic_json(config_path, identity)
    path = root/"validation"/f"epoch_{epoch:03d}.json"
    weights = state_hash(agent.state_dict())
    if path.exists():
        recorded = shared.load_json(path)
        if recorded["identity"] != shared.digest(identity) or recorded["state_sha256"] != weights:
            raise ValueError("Stale source monitoring result")
        return
    groups = defaultdict(list)
    for record in selected:
        groups[record["category"]].append(record)
    metrics, trajectories, images = [], [], []
    with diagnostic_context(agent):
        for category, records in sorted(groups.items()):
            maps, bases, masks, labels = [], [], [], []
            for offset in range(0, len(records), cfg["eval_batch_size"]):
                chunk = records[offset:offset+cfg["eval_batch_size"]]
                batch = store.batch(chunk)
                result = agent.trajectory({k: batch[k] for k in INPUTS if k in batch}, sample=False)
                agent.score(result, batch["mask"], batch["labels"])
                maps.extend(result["final_prob"][:, 1].cpu().numpy())
                bases.extend(batch["base_prob"][:, 1].cpu().numpy())
                masks.extend(batch["mask"][:, 0].cpu().numpy().astype(bool))
                labels.extend(batch["labels"].cpu().tolist())
                for i, record in enumerate(chunk):
                    trajectories.append(dict(sample_id=record["id"], label=int(batch["labels"][i]),
                        return_value=float(result["steps"][0]["return"][i]),
                        first_stop=int(result["steps"][0]["actions"]["op"][i]) == 0,
                        depth=sum(float(s["execute"][i]) for s in result["steps"])))
                    if record["id"] in preview_ids:
                        target = root/"previews"/f"epoch_{epoch:03d}"/f"{record['id']}.png"
                        steps = preview_grid(target, record, batch, result, i, epoch, job)
                        images.append(dict(sample_id=record["id"], category=category, label=int(batch["labels"][i]),
                                           path=target.relative_to(root).as_posix(), steps=steps))
            for name, probability in (("Base", bases), (job["arm"], maps)):
                metrics.append(dict(category=category, arm=name, **map_metrics(np.stack(masks), np.stack(probability),
                    np.asarray(labels), cfg["pro_num_th"], cfg["pro_max_fpr"])))
    macro = {key: shared.mean([r[key] for r in metrics if r["arm"] == job["arm"]])
             for key in ("PRO", "P_AUROC", "F1", "background_FPR_at_05")}
    macro.update({"base_"+key: shared.mean([r[key] for r in metrics if r["arm"] == "Base"])
                  for key in ("PRO", "P_AUROC", "background_FPR_at_05")})
    macro.update(first_stop_fraction=shared.mean([r["first_stop"] for r in trajectories]),
                 refine_mean=shared.mean([r["depth"] for r in trajectories]),
                 return_mean=shared.mean([r["return_value"] for r in trajectories]))
    shared.atomic_json(path, dict(identity=shared.digest(identity), state_sha256=weights, schema=SCHEMA, epoch=epoch, every=every,
                                 selected_images=len(selected), metrics=metrics, macro=macro, previews=images))


def refresh_job(directory, history=None):
    """Rebuild figures from recorded evidence only; old missing batches stay missing."""
    directory = Path(directory)
    root = directory/"monitor"
    root.mkdir(parents=True, exist_ok=True)
    if history is None:
        history = read_csv(directory/"training.csv")
    committed = {int(number(r, "epoch")) for r in history}
    batches, updates, transitions = [], [], []
    for epoch in sorted(committed):
        marker = root/"epochs"/f"{epoch:03d}_complete.json"
        if marker.exists():
            for kind, out in (("batch", batches), ("update", updates), ("transition", transitions)):
                out.extend(read_csv(root/"epochs"/f"{epoch:03d}_{kind}.csv"))
    for values in (batches, updates):
        for i, row in enumerate(values):
            row["record_index"] = i+1
    for name, rows in (("batch_metrics", batches), ("update_metrics", updates), ("transition_samples", transitions)):
        shared.write_csv(root/f"{name}.csv", rows)
    panels(root/"curves"/"epoch_training.png", history, [
        ("Loss components (not segmentation losses)", ["loss", "policy_loss", "value_loss"]),
        ("Return and old baseline", ["return_mean", "old_value_mean"]),
        ("Behavior", ["first_stop_fraction", "refine_mean", "multi_step_fraction"]),
        ("PPO stability", ["approx_kl", "clip_fraction"]),
        ("Value fit and policy exploration", ["explained_variance", "entropy"]),
        ("Pre-clipping gradient norms", ["actor_grad_norm", "critic_grad_norm"])], title=directory.name+" | epoch training")
    panels(root/"curves"/"batch_training.png", batches, [
        ("Batch losses (raw; no smoothing)", ["loss", "policy_loss", "value_loss"]),
        ("Batch returns", ["return_mean", "old_value_mean"]),
        ("Advantage spread", ["advantage_mean", "advantage_std"]),
        ("STOP and multistep", ["first_stop_fraction", "multi_step_fraction"]),
        ("Policy entropy", ["entropy"]), ("Timing and memory", ["batch_seconds", "gpu_allocated_gb"])],
        x="record_index", title=directory.name+" | per rollout batch")
    if updates:
        panels(root/"curves"/"optimizer_updates.png", updates, [
            ("Optimizer-step loss components", ["policy_loss", "value_loss"]),
            ("Total objective", ["loss"]), ("Policy ratio stability", ["approx_kl", "clip_fraction"]),
            ("Pre-clipping gradient norms", ["actor_grad_norm", "critic_grad_norm"])],
            x="record_index", title=directory.name+" | every PPO optimizer update, raw values")
    behavior = [dict(r) for r in history]
    layer_keys = sorted({k for r in behavior for k in r if k.startswith("layer_") and k.endswith("_executions")})
    depth_keys = sorted({k for r in behavior for k in r if k.startswith("depth_")})
    for row in behavior:
        total = sum(number(row, k) for k in layer_keys)
        for key in layer_keys:
            row[key.replace("_executions", "_share")] = number(row, key)/max(1, total)
        for name in ("normal", "anomaly"):
            if number(row, name+"_count") == 0:
                row[name+"_return_mean"] = row[name+"_refine_mean"] = float("nan")
    panels(root/"curves"/"policy_diagnostics.png", behavior, [
        ("Executed layer distribution", [k.replace("_executions", "_share") for k in layer_keys]),
        ("Trajectory depth distribution", depth_keys),
        ("Normal / anomaly returns", ["normal_return_mean", "anomaly_return_mean"]),
        ("Normal / anomaly refine depth", ["normal_refine_mean", "anomaly_refine_mean"]),
        ("Learning rates", ["actor_lr", "critic_lr"]),
        ("Advantage spread / entropy weight", ["advantage_std", "entropy_coef"])],
        title=directory.name+" | policy behavior, source training only")
    if any("dual_used" in r for r in history):
        panels(root/"curves"/"locsafe_risk.png", history, [
            ("Normal tail-probability damage", ["normal_risk_mean", "normal_risk_p95", "normal_risk_worst5_mean", "normal_risk_max"]),
            ("Fixed P0 reference return", ["reference_return_mean", "normal_reference_worst5_mean"]),
            ("Lagrange multiplier (P3 only)", ["dual_used", "dual_next"]),
            ("Ranking improvement", ["rank_improvement_mean"])], title="LocSafe | full-epoch tail risk, not mean loss only")
    validations = [shared.load_json(p) for p in sorted((root/"validation").glob("epoch_*.json"))]
    if validations:
        panels(root/"curves"/"source_validation.png", [dict(epoch=r["epoch"], **r["macro"]) for r in validations], [
            ("Fixed SOURCE holdout PRO", ["PRO", "base_PRO"]), ("Pixel ranking", ["P_AUROC", "base_P_AUROC"]),
            ("Background FPR at .5", ["background_FPR_at_05", "base_background_FPR_at_05"]), ("Greedy net return", ["return_mean"]),
            ("Greedy STOP", ["first_stop_fraction"]), ("Greedy refine depth", ["refine_mean"])],
            title="SOURCE-only monitoring | fixed population; consult protocol for selection")
    if transitions:
        plt = pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        x = np.array([number(r, "old_value") for r in transitions])
        y = np.array([number(r, "return") for r in transitions])
        # Render a deterministic bounded set, not outcome-selected examples.
        stride = max(1, math.ceil(len(x)/4000))
        axes[0].scatter(x[::stride], y[::stride], s=3, alpha=.25)
        finite = np.r_[x[np.isfinite(x)], y[np.isfinite(y)]]
        if len(finite):
            lo, hi = float(finite.min()), float(finite.max())
            axes[0].plot([lo, hi], [lo, hi], color="red", linewidth=.8)
        axes[0].set(xlabel="V_old(s)", ylabel="sampled G", title="On-policy transitions; not independent MC means")
        for ax, key in zip(axes[1:], ("return", "advantage")):
            values = np.array([number(r, key) for r in transitions])
            ax.hist(values[np.isfinite(values)], bins=45)
            ax.axvline(0, color="red", linewidth=.8)
            ax.set_title(key+" distribution (logged subset)")
        fig.tight_layout()
        atomic_figure(fig, root/"curves"/"value_return_distributions.png")
    figures = "".join(f'<h3>{html.escape(p.stem)}</h3><a href="{p.relative_to(root).as_posix()}"><img src="{p.relative_to(root).as_posix()}"></a>'
                      for p in sorted((root/"curves").glob("*.png")))
    galleries = []
    for result in reversed(validations):
        cards = "".join(f'<figure><a href="{r["path"]}"><img loading="lazy" src="{r["path"]}"></a>'
                        f'<figcaption>{html.escape(r["category"])} / label {r["label"]} / {html.escape(r["sample_id"])}</figcaption></figure>'
                        for r in result["previews"])
        galleries.append(f'<details><summary>Epoch {result["epoch"]}: {result["selected_images"]} source monitor images</summary>{cards}</details>')
    atomic_html(root/"index.html", page(directory.name+" | PPO process diagnostics",
        '<p>Refresh this file after a committed epoch. Raw PNG/CSV files are downloadable below. '
        'No target data is used here. Check the experiment protocol for checkpoint selection. Old missing telemetry is NOT reconstructed.</p>'
        '<p>Policy loss need not monotonically decrease. MC value loss is zero by design. '
        'V is remaining stochastic return, not current map quality or one greedy return.</p>'
        f'<p>Recorded: {len(history)} epochs, {len(batches)} rollout batches, {len(updates)} optimizer updates. '
        'Not recorded means unavailable, NOT a measured zero.</p>'
        '<p>'+" | ".join(f'<a href="{n}.csv">{n}.csv</a>' for n in ("batch_metrics", "update_metrics", "transition_samples"))+"</p>"
        +figures+"<h2>Fixed source examples through time</h2>"+"".join(galleries)))


def page(title, body):
    return '<!doctype html><html><head><meta charset="utf-8"><title>'+html.escape(title)+'''</title>
<style>body{font:15px system-ui;max-width:1450px;margin:28px auto;padding:0 22px;color:#182438;background:#f5f7fb}
img{max-width:100%;height:auto;background:white;border:1px solid #dae0ea}h1,h2{color:#15365d}
figure{margin:14px 0}details{padding:14px;background:white;margin:12px 0}a{color:#1464a5}p{line-height:1.6}</style>
</head><body><h1>'''+html.escape(title)+"</h1>"+body+"</body></html>"


def refresh_overview(root, cfg):
    root = Path(root)
    directory = root/"visual_report"
    history, links = [], []
    for job in cfg["jobs"]:
        path = root/"jobs"/job["id"]
        rows = read_history(root, job)
        history.extend(rows)
        if (path/"monitor"/"index.html").exists():
            links.append(f'<li><a href="../jobs/{job["id"]}/monitor/index.html">{job["id"]}: batch curves, source validation and image trajectories</a></li>')
    panels(directory/"training_comparison.png", history, [
        ("Total objective", ["loss"]), ("Policy loss", ["policy_loss"]), ("Value loss (MC=0)", ["value_loss"]),
        ("Training return", ["return_mean"]), ("STOP", ["first_stop_fraction"]), ("Refine depth", ["refine_mean"])],
        group_key="job", title=cfg.get("protocol", "PPO")+" | raw rewards may differ across arms")
    final = read_csv(root/"results"/"macro_metrics.csv")
    if cfg.get("protocol") == "mara_ppo_locsafe_v1":
        final = [r for r in final if r.get("role") in ("source_dev", "target_full")]
    if final:
        plt = pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        datasets = sorted({r["dataset"] for r in final})
        for ax, metric in zip(axes.flat, ("PRO", "P_AUROC", "F1", "background_FPR_at_05")):
            arms = ("Base", *cfg.get("arms", ("V", "MC")))
            width = .8/len(arms)
            for offset, arm in enumerate(arms):
                means, stds = [], []
                for dataset in datasets:
                    values = {r["seed"]: number(r, metric) for r in final if r["dataset"] == dataset and r["arm"] == arm
                              and (arm == "Base" or int(r["step"]) == cfg["mara_config"]["max_steps"])}
                    values = [v for v in values.values() if math.isfinite(v)]
                    means.append(float(np.mean(values)) if values else float("nan"))
                    stds.append(float(np.std(values)) if len(values) > 1 else 0.)
                ax.bar(np.arange(len(datasets))+(offset-(len(arms)-1)/2)*width, means, width=width*.95, yerr=stds, label=arm, capsize=3)
            ax.set_xticks(np.arange(len(datasets)), datasets)
            ax.set_title(metric+" (raw 0-1 units)")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Category macro; seed mean +/- std (not CI); consult protocol for checkpoint selection")
        fig.tight_layout()
        atomic_figure(fig, directory/"final_metrics.png")
    probe = read_csv(root/"results"/"value_probe.csv")
    if probe:
        plt = pyplot()
        fig, ax = plt.subplots(figsize=(7, 5))
        for arm in sorted({r["arm"] for r in probe}):
            rows = [r for r in probe if r["arm"] == arm]
            ax.errorbar([number(r, "v_prediction") for r in rows], [number(r, "mc_return_mean") for r in rows],
                        yerr=[number(r, "mc_standard_error") for r in rows], fmt="o", markersize=3, alpha=.6, label=arm)
        values = [number(r, key) for r in probe for key in ("v_prediction", "mc_return_mean")]
        values = [v for v in values if math.isfinite(v)]
        if values:
            ax.plot([min(values), max(values)], [min(values), max(values)], color="red", linewidth=.8, label="V=MC mean")
        ax.set(xlabel="V(s0)", ylabel="independent stochastic rollout mean G", title="SOURCE value calibration; bars = MC standard error")
        ax.grid(alpha=.2)
        ax.legend()
        atomic_figure(fig, directory/"value_calibration.png")
    figures = "".join(f'<h2>{html.escape(p.stem)}</h2><img src="{p.name}">' for p in sorted(directory.glob("*.png")))
    atomic_html(directory/"index.html", page("MARA PPO: training process", '<p>Actual recorded history; refresh after each completed epoch. '
        'Each job has a separate live report while GPUs train independently. No targets are monitored mid-training.</p>'
        +figures+'<ul>'+"".join(links)+"</ul>"))


def visual_files(root, cfg):
    root = Path(root)
    allowed = {".png", ".csv", ".json", ".html"}
    for path in sorted((root/"visual_report").rglob("*")):
        if path.is_file() and path.suffix in allowed and ".tmp" not in path.name:
            yield path
    for job in cfg["jobs"]:
        monitor = root/"jobs"/job["id"]/"monitor"
        for path in sorted(monitor.rglob("*")):
            if (path.is_file() and path.suffix in allowed and ".tmp" not in path.name
                    and "epochs" not in path.relative_to(monitor).parts and not path.name.startswith("live_epoch_")):
                yield path


def package_process(root, cfg):
    """Small, separate process snapshot, available before final target testing."""
    root = Path(root)
    target = root/"ppo_process_report.zip"
    temporary = target.with_name(target.name+f".{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in visual_files(root, cfg):
                archive.write(path, path.relative_to(root).as_posix())
            for name in ("config.json", "manifest.json", "audit.json", "split.json"):
                if (root/name).exists():
                    archive.write(root/name, name)
            for path in sorted((root/"results").glob("*.csv")):
                archive.write(path, "results/"+path.name)
            for job in cfg["jobs"]:
                rows = read_history(root, job)
                if rows:
                    output = io.StringIO(newline="")
                    fields = sorted(set().union(*(r.keys() for r in rows)))
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                    archive.writestr(f"jobs/{job['id']}/training.csv", output.getvalue())
            archive.writestr("README.txt", "PROCESS SNAPSHOT, not a new final benchmark. Open visual_report/index.html.\n"
                "Only recorded data / source monitoring. Jobs may be at different committed epochs. No model weights or large caches.\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
