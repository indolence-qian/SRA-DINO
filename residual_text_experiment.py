#!/usr/bin/env python3
"""Residual-first DINO + normal support + high resolution + text-delta experiment.

Independent of all legacy Base/H/T/R checkpoints and caches. Target masks are
opened only by evaluate, after all target predictions have been sealed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from tools.residual_text import (ResidualDensity, ResidualText, density_loss, text_loss,
                                 fuse_views, positive_text_fusion)
from tools.residual_text_runtime import (
    GENERIC_PROMPT, TEMPLATES, FrozenDinoExtractor, atomic_npz, encode_text_bank,
    evaluation_mask, export_residuals, load_annotations, make_record, normalized_boxes,
    read_torch, save_torch, source_labels, unchanged,
)
from tools.vlm_review import atomic_json, digest, file_digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "residual_first_highres_text_v1"
ARMS = ("D_G", "D_HR", "R_G", "R_HR", "T_G", "T_HR", "F_G", "F_HR")
FEATURE_KEYS = {"residual", "distance", "match_index", "boxes", "pixel_boxes"}
CODE_FILES = ("residual_text_experiment.py", "tools/residual_text.py", "tools/residual_text_runtime.py",
              "run_exp_residual_text.sh", "tools/reference_highres.py", "tools/vlm_review.py",
              "tools/vlm_next_diagnostics.py", "CLIP/openai.py", "CLIP/model.py",
              "CLIP/transformer.py", "CLIP/tokenizer.py")


def device_for(args, cfg=None):
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; CPU is only for small smoke tests")
        torch.cuda.set_device(device)
        if cfg and cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported: use PRECISION=fp32 and a new WORK_DIR")
    return device


def rows_for(root, partition=None):
    rows = load_json(Path(root) / "manifest.json")
    if partition == "source":
        return [r for r in rows if r["partition"] in ("train", "val")]
    return [r for r in rows if partition is None or r["partition"] == partition]


def load_catalog(args):
    from Datasets import DATASET_CLASSES, DATASET_REGISTRY
    targets = args.eval_datasets.replace(",", " ").split()
    names = [args.source] + targets
    if (not targets or len(set(names)) != len(names)
            or not set(names) <= {"visa", "mvtec", "btad", "mpdd"}):
        raise ValueError("Use one source and distinct target industrial datasets")
    import json
    roots = json.loads(args.data_roots)
    if not isinstance(roots, dict) or not set(roots) <= set(DATASET_REGISTRY):
        raise ValueError("DATA_ROOTS_JSON must map dataset names to directories")
    if args.source_root:
        roots[args.source] = args.source_root
    if args.target_root:
        if len(targets) != 1:
            raise ValueError("TARGET_ROOT needs a single target; otherwise use DATA_ROOTS_JSON")
        roots[targets[0]] = args.target_root
    queries, candidates = [], []
    for name in names:
        dataset_cls, splits, default = DATASET_REGISTRY[name]
        for category in sorted(DATASET_CLASSES[name]):
            for split, target in ((splits.TEST, queries), (splits.TRAIN, candidates)):
                dataset = dataset_cls(source=roots.get(name, default), classname=category, split=split,
                                      resize=args.image_size, imagesize=args.image_size)
                entries = sorted(dataset.data_to_iterate,
                                 key=lambda e: digest([args.data_seed, name, category, str(e[2])]))
                if target is queries and args.limit_per_category:
                    entries = entries[:args.limit_per_category]
                for _, state, image, mask in entries:
                    if target is candidates and (mask or str(state).lower() not in ("normal", "good", "ok")):
                        continue
                    partition = "reference" if target is candidates else "source" if name == args.source else "eval"
                    target.append(make_record(name, category, image, mask, partition))
    return queries, candidates


def prepare(args, catalog=None):
    root = Path(args.work_dir)
    layers = [int(x) for x in args.visual_layers.split(",")]
    strengths = sorted(set(float(x) for x in args.text_strengths.split(",")) | {0.})
    if (len(layers) != len(set(layers)) or not layers or min(layers) < 0
            or args.image_size < 32 or args.patch_size < 1 or args.image_size % args.patch_size
            or not 2 <= args.feature_dim <= args.input_dim or args.input_dim < 2
            or args.reference_count not in (1, 2, 4) or not .5 <= args.tile_fraction < 1
            or not 0 < args.val_fraction < .5 or not 0 < args.high_weight < 1
            or min(args.density_epochs, args.text_epochs, args.batch_size, args.micro_batch,
                   args.tokens_per_view, args.flow_blocks, args.hidden, args.num_shards,
                   args.match_chunk, args.bank_chunk, args.calibration_images, args.selection_size,
                   args.eval_size) < 1 or args.workers < 0 or args.limit_per_category < 0
            or args.micro_batch > args.batch_size or min(args.density_lr, args.text_lr) <= 0
            or not all(math.isfinite(x) for x in (args.density_lr, args.text_lr, args.tile_fraction,
                                                   args.val_fraction, args.high_weight, args.val_fpr))
            or not 0 < args.val_fpr < 1 or not all(math.isfinite(x) and 0 <= x <= 1 for x in strengths)):
        raise ValueError("Invalid residual experiment geometry/training/protocol settings")
    for path in (args.dino_weights, args.clip_weights):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Explicit local pretrained weights required: {path}")
    if not Path(args.dino_repo_dir, "hubconf.py").is_file():
        raise FileNotFoundError("DINO_REPO_DIR must be the local DINO torch.hub repository")
    rows, candidates = catalog if catalog is not None else load_catalog(args)
    source = [r for r in rows if r["partition"] == "source"]
    categories = sorted({r["category"] for r in source}, key=lambda c: digest([args.data_seed, c]))
    if len(categories) < 2:
        raise ValueError("Class-held-out source validation requires at least two source categories")
    count = max(1, min(len(categories) - 1, round(len(categories) * args.val_fraction)))
    val_categories = categories[:count]
    for row in source:
        row["partition"] = "val" if row["category"] in val_categories else "train"
    for partition in ("train", "val"):
        flags = {bool(r["mask_path"]) for r in rows if r["partition"] == partition}
        if flags != {False, True}:
            raise ValueError(f"Source {partition} needs normal and abnormal queries; increase LIMIT_PER_CATEGORY")
    if not any(r["partition"] == "eval" for r in rows):
        raise ValueError("No target queries")
    seen, ids = {}, set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate query id")
        ids.add(row["id"])
        if seen.setdefault(row["rgb_sha"], row["partition"]) != row["partition"]:
            raise ValueError("Decoded query content crosses source/val/target partitions")
    references, reference_families = [], {}
    for key in sorted({(r["dataset"], r["category"]) for r in rows}):
        eligible = sorted([r for r in candidates if (r["dataset"], r["category"]) == key],
                          key=lambda r: digest([args.data_seed, "reference", r["rgb_sha"]]))
        chosen, ref_sha = [], set()
        for row in eligible:
            if row["mask_path"] or row["rgb_sha"] in seen or row["rgb_sha"] in ref_sha:
                continue
            if reference_families.setdefault(row["rgb_sha"], key) != key:
                raise ValueError("Normal reference content crosses category/dataset support banks")
            chosen.append(row)
            ref_sha.add(row["rgb_sha"])
            if len(chosen) == args.reference_count:
                break
        if len(chosen) != args.reference_count:
            raise ValueError(f"Need {args.reference_count} distinct normal TRAIN references without query duplicates: {key}")
        references.extend(chosen)
    prompts, regions = load_annotations(args.semantic_annotations, rows)
    prompts = [GENERIC_PROMPT] + prompts
    root.mkdir(parents=True, exist_ok=True)
    values = dict(manifest=rows, references=references, prompts=prompts, semantic_regions=regions)
    # Serialize sidecars before config only on first prepare; a changed rerun is
    # rejected without touching any existing artifact.
    cfg = {k: v for k, v in vars(args).items() if k not in {
        "stage", "work_dir", "partition", "shard_id", "device", "variant", "train_seed",
        "train_seeds", "until_epoch", "probe", "source_root", "target_root", "data_roots"}}
    cfg.update(protocol=PROTOCOL, layers=layers, strengths=strengths, val_categories=val_categories,
               dino_weights=str(Path(args.dino_weights).resolve()), clip_weights=str(Path(args.clip_weights).resolve()),
               dino_repo_dir=str(Path(args.dino_repo_dir).resolve()),
               dino_sha=file_digest(args.dino_weights), clip_sha=file_digest(args.clip_weights),
               code={p: file_digest(REPO / p) for p in CODE_FILES},
               backbone_code=digest({str(p.relative_to(args.dino_repo_dir)): file_digest(p)
                                     for p in sorted(Path(args.dino_repo_dir).rglob("*.py"))}),
               torch_version=str(torch.__version__), templates=list(TEMPLATES),
               setting="K normal TRAIN references; no target anomalies for training; no target fine-tuning",
               semantic_scope="generic binary direction" if len(prompts) == 1 else "source-region-supervised change directions")
    # Hash canonical values, not pretty-print bytes, so prospective config can
    # be compared safely before writing anything.
    cfg.update({name + "_sha": digest(value) for name, value in values.items()})
    if (root / "config.json").exists():
        previous = load_json(root / "config.json")
        if previous != cfg:
            raise ValueError("Settings/data/code changed; use NEW WORK_DIR (legacy caches are not compatible)")
        checked_config(root)
        print("prepare: unchanged configuration", flush=True)
        return cfg
    if any((root / f"{name}.json").exists() for name in values):
        raise ValueError("Incomplete pre-existing prepare sidecars; use a new directory")
    for name, value in values.items():
        atomic_json(root / f"{name}.json", value)
    atomic_json(root / "config.json", cfg)
    side = args.image_size // args.patch_size
    estimate = len(rows) * 5 * len(layers) * side * side * (2 * args.feature_dim + 8)
    print(f"Prepared {len(rows)} queries, K={args.reference_count}, held-out source classes={val_categories}", flush=True)
    print(f"Residual cache estimate={estimate/2**30:.1f} GiB; eight output maps require additional space.", flush=True)
    print(f"Text supervision: {cfg['semantic_scope']}; no Base checkpoint, HFA, Prompt Adapter or RL.", flush=True)
    return cfg


def checked_config(root):
    # Canonical JSON hashes allow immutable sidecars and safe prepare idempotence.
    root = Path(root)
    cfg = load_json(root / "config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong experiment protocol")
    for path, sha in cfg["code"].items():
        if file_digest(REPO / path) != sha:
            raise ValueError(f"Code changed: {path}; use NEW WORK_DIR")
    for name in ("manifest", "references", "semantic_regions", "prompts"):
        if digest(load_json(root / f"{name}.json")) != cfg[name + "_sha"]:
            raise ValueError(f"{name} changed")
    return cfg


def feature_path(root, row):
    return Path(root) / "features" / row["partition"] / f"{row['id']}.npz"


def seal_path(root, partition):
    return Path(root) / f"sealed_{partition}.json"


def export(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard id")
    assigned = rows_for(root, args.partition)[args.shard_id::cfg["num_shards"]]
    if args.probe:
        assigned = assigned[:1]
    pending = []
    for row in assigned:
        path = feature_path(root, row)
        if path.exists() and path.with_suffix(".json").exists():
            meta = load_json(path.with_suffix(".json"))
            if meta["fingerprint"] != digest(cfg) or meta["sha"] != file_digest(path):
                raise ValueError("Changed feature export; use NEW WORK_DIR")
        else:
            if seal_path(root, args.partition).exists():
                raise ValueError("Cannot regenerate a sealed cache")
            pending.append(row)
    if not pending:
        print(f"export {args.partition}/{args.shard_id}: already complete", flush=True)
        return
    device = device_for(args, cfg)
    current_backbone = digest({str(p.relative_to(cfg["dino_repo_dir"])): file_digest(p)
                               for p in sorted(Path(cfg["dino_repo_dir"]).rglob("*.py"))})
    if current_backbone != cfg["backbone_code"]:
        raise ValueError("DINO backbone code changed after prepare; use NEW WORK_DIR")
    extractor = FrozenDinoExtractor(cfg, device)
    references = load_json(root / "references.json")
    current_key, normal_views, reference_meta = None, None, None
    try:
        for row in pending:
            unchanged(row)
            key = row["dataset"], row["category"]
            if key != current_key:
                normal_views, reference_meta = [], []
                for reference in references:
                    if (reference["dataset"], reference["category"]) != key:
                        continue
                    unchanged(reference)
                    if reference["rgb_sha"] == row["rgb_sha"]:
                        raise ValueError("Self-reference leakage")
                    with Image.open(reference["image_path"]) as image:
                        views, boxes = extractor.views(image.convert("RGB"))
                    normal_views.append([v.cpu() for v in views])
                    reference_meta.append(dict(id=reference["id"], pixel_boxes=boxes,
                                               native_size=reference["native_size"]))
                if len(normal_views) != cfg["reference_count"]:
                    raise ValueError("Incomplete normal support bank")
                current_key = key
            tick = time.monotonic()
            with Image.open(row["image_path"]) as image:
                views, boxes = extractor.views(image.convert("RGB"))
            arrays = export_residuals(extractor, views, normal_views, cfg)
            arrays.update(boxes=normalized_boxes(boxes, row["native_size"]),
                          pixel_boxes=np.asarray(boxes, np.int32))
            path = feature_path(root, row)
            atomic_npz(path, **arrays)
            atomic_json(path.with_suffix(".json"), dict(schema="residual_only_5views_v1", id=row["id"],
                        fingerprint=digest(cfg), sha=file_digest(path), references=reference_meta,
                        query_seconds=time.monotonic() - tick,
                        timing_note="five query encodings + matching; excludes support encoding; not per-arm latency"))
            print(f"export {args.partition}/{args.shard_id} {row['id']} {time.monotonic()-tick:.2f}s", flush=True)
    finally:
        extractor.close()


def seal(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    features, metadata = {}, {}
    side = cfg["image_size"] // cfg["patch_size"]
    for row in rows_for(root, args.partition):
        path = feature_path(root, row)
        meta = load_json(path.with_suffix(".json"))
        if (meta["id"] != row["id"] or meta["fingerprint"] != digest(cfg)
                or meta["schema"] != "residual_only_5views_v1"
                or len(meta["references"]) != cfg["reference_count"] or meta["sha"] != file_digest(path)):
            raise ValueError("Export identity/reference count mismatch")
        with np.load(path, allow_pickle=False) as data:
            expected = (5, len(cfg["layers"]), cfg["feature_dim"], side, side)
            if set(data.files) != FEATURE_KEYS or data["residual"].shape != expected:
                raise ValueError("Not a residual-only per-view cache (legacy caches are forbidden)")
            if data["distance"].shape != (5, len(cfg["layers"]), side, side):
                raise ValueError("Invalid distance maps")
            if data["boxes"].shape != (5, 4) or data["pixel_boxes"].shape != (5, 4):
                raise ValueError("Invalid view coordinates")
            if any(not np.isfinite(data[k]).all() for k in data.files):
                raise ValueError("Nonfinite cached arrays")
        features[row["id"]], metadata[row["id"]] = file_digest(path), file_digest(path.with_suffix(".json"))
    value = dict(fingerprint=digest(cfg), features=features, metadata=metadata)
    path = seal_path(root, args.partition)
    if path.exists() and load_json(path) != value:
        raise ValueError("Sealed cache changed")
    atomic_json(path, value)
    print(f"sealed {args.partition}: {len(features)} residual-only entries", flush=True)


class ResidualDataset(torch.utils.data.Dataset):
    def __init__(self, root, cfg, rows, labels=False):
        self.root, self.cfg, self.rows, self.labels = Path(root), cfg, rows, labels
        self.annotations = load_json(self.root / "semantic_regions.json") if labels else []
        self.type_ids = [p["id"] for p in load_json(self.root / "prompts.json")[1:]]
        seals = {}
        for row in rows:
            partition = "eval" if row["partition"] == "eval" else "source"
            if partition not in seals:
                seals[partition] = load_json(seal_path(self.root, partition))
            value = seals[partition]
            path = feature_path(self.root, row)
            if (value["fingerprint"] != digest(cfg) or value["features"][row["id"]] != file_digest(path)
                    or value["metadata"][row["id"]] != file_digest(path.with_suffix(".json"))):
                raise ValueError("Cache integrity failure")
            if labels and row["partition"] not in ("train", "val"):
                raise ValueError("Target supervision is forbidden")
            unchanged(row, masks=labels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(feature_path(self.root, row), allow_pickle=False) as data:
            result = {k: torch.from_numpy(data[k].astype(np.float32)) for k in ("residual", "distance", "boxes")}
            if self.labels:
                coverage, typed = source_labels(row, data["pixel_boxes"].tolist(), result["residual"].shape[-1],
                                                self.annotations, self.type_ids)
                result.update(coverage=coverage, typed=typed)
        result["index"] = torch.tensor(index)
        return result


def calibration_rows(root, cfg):
    groups = defaultdict(list)
    for row in rows_for(root, "train"):
        groups[row["category"]].append(row)
    for values in groups.values():
        values.sort(key=lambda r: digest([cfg["data_seed"], "calibrate", r["id"]]))
    result = []
    for index in range(max(map(len, groups.values()))):
        for category in sorted(groups):
            if index < len(groups[category]):
                result.append(groups[category][index])
                if len(result) >= cfg["calibration_images"]:
                    return result
    return result


def fit_stats(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    path = root / "normal_stats.json"
    identity = dict(fingerprint=digest(cfg), cache_sha=file_digest(seal_path(root, "source")))
    if path.exists():
        if load_json(path)["identity"] != identity:
            raise ValueError("Normal residual statistics changed")
        return
    dataset = ResidualDataset(root, cfg, calibration_rows(root, cfg), labels=True)
    sums, counts = np.zeros((len(cfg["layers"]), 2)), np.zeros((len(cfg["layers"]), 2))
    for item in dataset:
        for view in range(5):
            normal = item["coverage"][view] == 0
            for layer in range(len(cfg["layers"])):
                x = item["residual"][view, layer, :, normal].numpy().astype(np.float64)
                sums[layer, int(view > 0)] += np.square(x).sum()
                counts[layer, int(view > 0)] += x.size
    if (counts == 0).any():
        raise ValueError("No wholly normal source cells available for calibration")
    scale = np.sqrt(sums / counts)
    if (scale < 1e-8).any() or not np.isfinite(scale).all():
        raise ValueError("Degenerate normal residual statistics; inspect support/query matching")
    atomic_json(path, dict(identity=identity, rms=scale.tolist(),
                          note="source TRAIN normal-only RMS; no centering, no per-image minmax",
                          rows=[r["id"] for r in dataset.rows]))


def stats_for(root, cfg, device):
    value = load_json(Path(root) / "normal_stats.json")
    identity = dict(fingerprint=digest(cfg), cache_sha=file_digest(seal_path(root, "source")))
    if value["identity"] != identity:
        raise ValueError("Stale normal statistics")
    scale = torch.tensor(value["rms"], dtype=torch.float32, device=device)
    if scale.shape != (len(cfg["layers"]), 2) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid source normal residual scale")
    return scale


def scaled_residual(batch, scale, variant):
    residual = batch["residual"][:, :1] if variant == "G" else batch["residual"]
    types = [0] + [1] * (residual.shape[1] - 1)
    denominator = scale[:, types].T[None, :, :, None, None, None]
    return residual / denominator.clamp_min(1e-8)


def text_bank(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    path = root / "text_bank.npz"
    identity = dict(fingerprint=digest(cfg), prompts_sha=cfg["prompts_sha"], clip_sha=cfg["clip_sha"])
    if path.exists() and path.with_suffix(".json").exists():
        meta = load_json(path.with_suffix(".json"))
        if meta["identity"] != identity or meta["sha"] != file_digest(path):
            raise ValueError("Frozen text bank changed")
        return
    directions, states = encode_text_bank(cfg, load_json(root / "prompts.json"), device_for(args, cfg))
    atomic_npz(path, directions=directions, paired_states=states)
    atomic_json(path.with_suffix(".json"), dict(identity=identity, sha=file_digest(path),
                text_dim=int(directions.shape[-1]), trainable_prompts=False, encoder_frozen=True))


def directions_for(root, cfg, device):
    path = Path(root) / "text_bank.npz"
    meta = load_json(path.with_suffix(".json"))
    if meta["identity"]["fingerprint"] != digest(cfg) or meta["sha"] != file_digest(path):
        raise ValueError("Frozen text bank identity mismatch")
    with np.load(path, allow_pickle=False) as data:
        return torch.from_numpy(data["directions"].copy()).float().to(device)


def batch_to(item, device):
    return {k: v[None].to(device) for k, v in item.items() if k != "index"}


def raw_scores(model, kind, batch, scale, variant):
    if kind == "distance":
        return batch["distance"][:, :1] if variant == "G" else batch["distance"]
    residual = scaled_residual(batch, scale, variant)
    if kind == "density":
        return model(residual)
    if kind == "text":
        return model(residual)[0][:, :, :1]
    raise ValueError("Unknown scorer")


def apply_calibration(scores, calibration):
    # score channels are DINO levels for density/distance, one for generic text.
    index = [0] + [1] * (scores.shape[1] - 1)
    center = scores.new_tensor(calibration["center"])[:, index].T[None, :, :, None, None]
    scale = scores.new_tensor(calibration["scale"])[:, index].T[None, :, :, None, None]
    return ((scores - center) / scale).mean(2, keepdim=True)


@torch.inference_mode()
def fit_calibration(model, kind, dataset, scale, variant, cfg, device):
    if model is not None:
        model.eval()
    channels = 1 if kind == "text" else len(cfg["layers"])
    values = [[[] for _ in range(2)] for _ in range(channels)]
    for item in dataset:
        batch = batch_to(item, device)
        scores = raw_scores(model, kind, batch, scale, variant)[0]
        for view in range(scores.shape[0]):
            normal = batch["coverage"][0, view] == 0
            for channel in range(channels):
                sample = scores[view, channel][normal].detach().float().cpu().numpy()
                values[channel][int(view > 0)].append(sample[::max(1, len(sample) // 256)])
    centers, scales = np.zeros((channels, 2)), np.ones((channels, 2))
    for channel in range(channels):
        for view_type in range(2 if variant == "HR" else 1):
            data = np.concatenate(values[channel][view_type])
            if not len(data) or not np.isfinite(data).all():
                raise ValueError("No finite normal source calibration evidence")
            center = float(np.quantile(data, .95 if kind == "text" else .5))
            # A negative normal text baseline must not turn raw zero evidence
            # into a positive fusion increment. Preserve the zero-residual audit
            # through score calibration, not only inside the bias-free MLP.
            centers[channel, view_type] = max(0., center) if kind == "text" else center
            scales[channel, view_type] = max(float(np.quantile(data, .75) - np.quantile(data, .25)), 1e-4)
    return dict(center=centers.tolist(), scale=scales.tolist(), kind=kind,
                note="source TRAIN normal cells; text center=max(0,p95), others=median")


def calibrated_map(model, kind, batch, scale, variant, calibration):
    return apply_calibration(raw_scores(model, kind, batch, scale, variant), calibration)


def metric_point(masks, maps, categories, fpr):
    from tools.vlm_next_diagnostics import map_metrics
    from sklearn.metrics import average_precision_score
    by_category = defaultdict(list)
    for index, category in enumerate(categories):
        by_category[category].append(index)
    metrics = []
    background = []
    for indices in by_category.values():
        truth = np.stack([masks[i] for i in indices])
        predicted = np.stack([maps[i] for i in indices])
        point = map_metrics(truth, predicted)
        point["P_AP"] = float(average_precision_score(truth.ravel(), predicted.ravel())) if truth.any() else None
        metrics.append(point)
        for mask, value in zip(truth, predicted):
            index = np.arange(0, value.size, max(1, value.size // 8192))
            normal = ~mask.ravel()[index]
            background.append(value.ravel()[index][normal])
    result = {key: float(np.mean([p[key] for p in metrics if p[key] is not None]))
              if any(p[key] is not None for p in metrics) else None
              for key in ("PRO_exact", "P_AP", "P_AUROC", "I_AUROC")}
    normal = np.concatenate(background)
    if not len(normal):
        raise ValueError("Source validation has no normal pixels")
    result["threshold"] = float(np.quantile(normal, 1 - fpr, method="higher"))
    return result


@torch.inference_mode()
def validate_scorer(model, kind, dataset, scale, variant, calibration, cfg, device):
    if model is not None:
        model.eval()
    masks, maps, categories = [], [], []
    for item in dataset:
        batch = batch_to(item, device)
        scores = calibrated_map(model, kind, batch, scale, variant, calibration)
        full = fuse_views(scores, batch["boxes"], cfg["selection_size"], variant == "HR", cfg["high_weight"])
        row = dataset.rows[int(item["index"])]
        masks.append(evaluation_mask(row, cfg["selection_size"]))
        maps.append(full[0, 0].cpu().numpy())
        categories.append(row["category"])
    return metric_point(masks, maps, categories, cfg["val_fpr"])


def quality(point):
    return tuple(point[key] if point[key] is not None else -1. for key in ("PRO_exact", "P_AP"))


def eligible(point, baseline):
    a, b = quality(point), quality(baseline)
    return all(x >= y - 1e-12 for x, y in zip(a, b)) and any(x > y + 1e-6 for x, y in zip(a, b))


def head_dir(root, seed, variant, kind):
    return Path(root) / "heads" / f"seed{seed}" / f"{kind}_{variant}"


def requested_horizon(args, cfg):
    until = args.until_epoch or min(cfg["density_epochs"], cfg["text_epochs"])
    if not 1 <= until <= min(cfg["density_epochs"], cfg["text_epochs"]):
        raise ValueError("UNTIL_EPOCH must be within both originally planned training horizons")
    return until


def snapshot_path(root, seed, variant, kind, until):
    return head_dir(root, seed, variant, kind) / f"through{until}" / "best.pt"


def new_model(root, cfg, kind, seed, device):
    if kind == "density":
        return ResidualDensity(len(cfg["layers"]), cfg["feature_dim"], cfg["hidden"],
                               cfg["flow_blocks"], seed).to(device)
    return ResidualText(len(cfg["layers"]), cfg["feature_dim"], directions_for(root, cfg, device),
                         cfg["hidden"]).to(device)


def training_identity(root, cfg, kind, seed, variant):
    value = dict(fingerprint=digest(cfg), cache_sha=file_digest(seal_path(root, "source")),
                 stats_sha=file_digest(Path(root) / "normal_stats.json"), kind=kind, seed=seed, variant=variant)
    if kind == "text":
        value["text_bank_sha"] = file_digest(Path(root) / "text_bank.npz")
    return value


def train(args, kind):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    device = device_for(args, cfg)
    until = requested_horizon(args, cfg)
    seed, variant = args.train_seed, args.variant
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = new_model(root, cfg, kind, seed, device)
    train_data = ResidualDataset(root, cfg, rows_for(root, "train"), labels=True)
    val_data = ResidualDataset(root, cfg, rows_for(root, "val"), labels=True)
    cal_data = ResidualDataset(root, cfg, calibration_rows(root, cfg), labels=True)
    scale = stats_for(root, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg[kind + "_lr"], weight_decay=.01)
    steps = math.ceil(len(train_data) / cfg["batch_size"]) * cfg[kind + "_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: .05 + .95 * (1 + math.cos(math.pi * min(step, steps) / steps)) / 2)
    out = head_dir(root, seed, variant, kind)
    identity = training_identity(root, cfg, kind, seed, variant)
    start, history = 0, []
    if (out / "last.pt").exists():
        saved = read_torch(out / "last.pt", device)
        if saved["identity"] != identity or saved["best_sha"] != file_digest(out / "best.pt"):
            raise ValueError("Training resume identity/checkpoint mismatch")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, history = saved["epoch"], saved["history"]
        if start >= until:
            target = snapshot_path(root, seed, variant, kind, until)
            if not target.exists() or read_torch(target)["identity"] != identity:
                raise ValueError("Requested older training snapshot is missing; never replace it with a later checkpoint")
            print(f"{kind}_{variant}/seed{seed}: through{until} already complete", flush=True)
            return
        best = read_torch(out / "best.pt")
        baseline = saved["baseline"]
    else:
        if kind == "density":
            calibration = fit_calibration(None, "distance", cal_data, scale, variant, cfg, device)
            point = validate_scorer(None, "distance", val_data, scale, variant, calibration, cfg, device)
            selected_kind = "distance"
        else:
            calibration = fit_calibration(model, "text", cal_data, scale, variant, cfg, device)
            point = validate_scorer(model, "text", val_data, scale, variant, calibration, cfg, device)
            selected_kind = "text"
        baseline = point
        best = dict(identity=identity, state_dict=model.state_dict(), epoch=0, kind=selected_kind,
                    calibration=calibration, point=point, baseline=baseline)
        save_torch(out / "best.pt", best)
        best = read_torch(out / "best.pt")
    for epoch in range(start, until):
        tick = time.monotonic()
        # Epoch-local RNG makes interrupted/resumed runs reproduce the same
        # sample order and token selections without depending on validation RNG.
        torch.manual_seed(seed + 100003 * (epoch + 1))
        model.train()
        generator = torch.Generator().manual_seed(cfg["data_seed"] + epoch)
        loader = torch.utils.data.DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True,
                                            generator=generator, num_workers=cfg["workers"])
        loss_sum, samples = 0., 0
        for batch in loader:
            count = len(batch["residual"])
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, count, cfg["micro_batch"]):
                part = {k: v[offset:offset + cfg["micro_batch"]].to(device)
                        for k, v in batch.items() if k != "index"}
                residual = scaled_residual(part, scale, variant)
                views = residual.shape[1]
                if kind == "density":
                    loss = density_loss(model, residual, part["coverage"][:, :views],
                                         cfg["tokens_per_view"], cfg["high_weight"])
                else:
                    loss = text_loss(model, residual, part["coverage"][:, :views],
                                     part["typed"][:, :views], cfg["high_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {kind} loss")
                weight = len(residual) / count
                (loss * weight).backward()
                loss_sum += float(loss.detach()) * len(residual)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            samples += count
        calibration = fit_calibration(model, kind, cal_data, scale, variant, cfg, device)
        point = validate_scorer(model, kind, val_data, scale, variant, calibration, cfg, device)
        allowed = kind == "text" or eligible(point, baseline)
        if allowed and quality(point) > quality(best["point"]):
            save_torch(out / "best.pt", dict(identity=identity, state_dict=model.state_dict(), epoch=epoch + 1,
                                             kind=kind, calibration=calibration, point=point, baseline=baseline))
            best = read_torch(out / "best.pt")
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / samples, source_val=point,
                            selected_epoch=best["epoch"], selected_kind=best["kind"],
                            seconds=time.monotonic() - tick))
        save_torch(out / "last.pt", dict(identity=identity, state_dict=model.state_dict(),
                   optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch + 1,
                   best_sha=file_digest(out / "best.pt"), baseline=baseline, history=history))
        atomic_json(out / "history.json", history)
        print(f"{kind}_{variant}/seed{seed} epoch={epoch+1}/{cfg[kind+'_epochs']} "
              f"loss={loss_sum/samples:.5f} sourcePRO={point['PRO_exact']} AP={point['P_AP']} "
              f"selected={best['kind']}@{best['epoch']}", flush=True)
    target = snapshot_path(root, seed, variant, kind, until)
    if target.exists():
        if file_digest(target) != file_digest(out / "best.pt"):
            raise ValueError("Immutable training snapshot already differs")
    else:
        # Byte-for-byte copy of the selected checkpoint, written atomically.
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes((out / "best.pt").read_bytes())
        temporary.replace(target)


def load_snapshot(root, cfg, kind, variant, seed, until, device):
    path = snapshot_path(root, seed, variant, kind, until)
    saved = read_torch(path, device)
    if saved["identity"] != training_identity(root, cfg, kind, seed, variant):
        raise ValueError("Selected model identity mismatch")
    model = new_model(root, cfg, kind, seed, device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, saved, file_digest(path)


def decision_dir(root, seed, until):
    return Path(root) / "decisions" / f"seed{seed}" / f"through{until}"


@torch.inference_mode()
def select(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    device, until = device_for(args, cfg), requested_horizon(args, cfg)
    scale = stats_for(root, cfg, device)
    dataset = ResidualDataset(root, cfg, rows_for(root, "val"), labels=True)
    calibration_data = ResidualDataset(root, cfg, calibration_rows(root, cfg), labels=True)
    result, model_hashes = {}, {}
    for variant in ("G", "HR"):
        density, density_meta, density_sha = load_snapshot(root, cfg, "density", variant, args.train_seed, until, device)
        text, text_meta, text_sha = load_snapshot(root, cfg, "text", variant, args.train_seed, until, device)
        distance_cal = fit_calibration(None, "distance", calibration_data, scale, variant, cfg, device)
        masks, categories = [], []
        maps = {"D": [], "R": [], "T": []}
        fused = {strength: [] for strength in cfg["strengths"]}
        operating_background = {key: [] for key in ("D", "R", "T")}
        fusion_background = {strength: [] for strength in cfg["strengths"]}
        for item in dataset:
            batch = batch_to(item, device)
            distance = calibrated_map(None, "distance", batch, scale, variant, distance_cal)
            residual = calibrated_map(density, density_meta["kind"], batch, scale, variant, density_meta["calibration"])
            semantic = calibrated_map(text, "text", batch, scale, variant, text_meta["calibration"])
            full = lambda x: fuse_views(x, batch["boxes"], cfg["selection_size"], variant == "HR", cfg["high_weight"])[0, 0].cpu().numpy()
            row = dataset.rows[int(item["index"])]
            operating_mask = evaluation_mask(row, cfg["eval_size"])
            sample_index = np.arange(0, operating_mask.size, max(1, operating_mask.size // 8192))
            normal_index = sample_index[~operating_mask.ravel()[sample_index]]
            def background(x):
                value = fuse_views(x, batch["boxes"], cfg["eval_size"], variant == "HR", cfg["high_weight"])
                return value[0, 0].cpu().numpy().ravel()[normal_index]
            for key, value in (("D", distance), ("R", residual), ("T", semantic)):
                maps[key].append(full(value))
                operating_background[key].append(background(value))
            for strength in fused:
                value = positive_text_fusion(residual, semantic, strength)
                fused[strength].append(full(value))
                fusion_background[strength].append(background(value))
            masks.append(evaluation_mask(row, cfg["selection_size"]))
            categories.append(row["category"])
        points = {key: metric_point(masks, value, categories, cfg["val_fpr"]) for key, value in maps.items()}
        candidates = {strength: metric_point(masks, value, categories, cfg["val_fpr"]) for strength, value in fused.items()}
        allowed = [s for s in candidates if s == 0 or
                   (text_meta["epoch"] > 0 and eligible(candidates[s], points["R"]))]
        chosen = max(allowed, key=lambda s: (*quality(candidates[s]), -s))
        # Model/strength selection uses the fixed selection grid; operating
        # thresholds use the FINAL evaluation geometry, still source-only.
        for key, values in operating_background.items():
            points[key]["threshold"] = float(np.quantile(np.concatenate(values), 1 - cfg["val_fpr"], method="higher"))
        for strength, values in fusion_background.items():
            candidates[strength]["threshold"] = float(np.quantile(np.concatenate(values),
                                                                   1 - cfg["val_fpr"], method="higher"))
        result[variant] = dict(density_kind=density_meta["kind"], density_epoch=density_meta["epoch"],
                               text_epoch=text_meta["epoch"], distance_calibration=distance_cal,
                               density_calibration=density_meta["calibration"], text_calibration=text_meta["calibration"],
                               strength=chosen, source_points=dict(points, F=candidates[chosen]),
                               threshold_resolution=cfg["eval_size"],
                               strength_scan={str(s): p for s, p in candidates.items()})
        model_hashes[variant] = dict(density=density_sha, text=text_sha)
    value = dict(fingerprint=digest(cfg), horizon=until, seed=args.train_seed,
                 models=model_hashes, variants=result, semantic_scope=cfg["semantic_scope"],
                 source_selection_resolution=cfg["selection_size"])
    path = decision_dir(root, args.train_seed, until) / "selection.json"
    if path.exists() and load_json(path) != value:
        raise ValueError("Locked source decision differs; use a new experiment directory")
    atomic_json(path, value)
    print(f"Source-locked fusion strengths: G={result['G']['strength']}, HR={result['HR']['strength']}", flush=True)


def locked_models(root, cfg, seed, until, device):
    path = decision_dir(root, seed, until) / "selection.json"
    decision = load_json(path)
    if decision["fingerprint"] != digest(cfg) or decision["horizon"] != until or decision["seed"] != seed:
        raise ValueError("Source selection identity mismatch")
    models = {}
    for variant in ("G", "HR"):
        models[variant] = {}
        for kind in ("density", "text"):
            model, _, sha = load_snapshot(root, cfg, kind, variant, seed, until, device)
            if sha != decision["models"][variant][kind]:
                raise ValueError("Model changed after source selection")
            models[variant][kind] = model
    return decision, models


def prediction_dir(root, seed, until):
    return Path(root) / "predictions" / f"seed{seed}" / f"through{until}"


@torch.inference_mode()
def predict_one(batch, cfg, scale, decision, models):
    output = {}
    for variant in ("G", "HR"):
        selected, model = decision["variants"][variant], models[variant]
        distance = calibrated_map(None, "distance", batch, scale, variant, selected["distance_calibration"])
        residual = calibrated_map(model["density"], selected["density_kind"], batch, scale,
                                   variant, selected["density_calibration"])
        semantic = calibrated_map(model["text"], "text", batch, scale, variant, selected["text_calibration"])
        fused = positive_text_fusion(residual, semantic, selected["strength"])
        for name, maps in (("D", distance), ("R", residual), ("T", semantic), ("F", fused)):
            output[f"{name}_{variant}"] = fuse_views(maps, batch["boxes"], cfg["eval_size"],
                                                      variant == "HR", cfg["high_weight"])[0, 0].cpu().numpy()
        typed, _ = model["text"](scaled_residual(batch, scale, variant))
        if typed.shape[2] > 1:
            output[f"types_{variant}"] = fuse_views(typed[:, :, 1:], batch["boxes"], cfg["eval_size"],
                                                      variant == "HR", cfg["high_weight"])[0].cpu().numpy()
    return output


def prediction_identity(root, cfg, seed, until):
    return dict(fingerprint=digest(cfg), seed=seed, horizon=until,
                selection_sha=file_digest(decision_dir(root, seed, until) / "selection.json"),
                cache_sha=file_digest(seal_path(root, "eval")))


@torch.inference_mode()
def predict(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    until, device = requested_horizon(args, cfg), device_for(args, cfg)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard id")
    identity = prediction_identity(root, cfg, args.train_seed, until)
    out = prediction_dir(root, args.train_seed, until)
    assigned = rows_for(root, "eval")[args.shard_id::cfg["num_shards"]]
    dataset = ResidualDataset(root, cfg, assigned, labels=False)
    decision, models = locked_models(root, cfg, args.train_seed, until, device)
    scale = stats_for(root, cfg, device)
    for item in dataset:
        row = dataset.rows[int(item["index"])]
        path = out / f"{row['id']}.npz"
        if path.exists() and path.with_suffix(".json").exists():
            meta = load_json(path.with_suffix(".json"))
            if meta["identity"] != identity or meta["sha"] != file_digest(path):
                raise ValueError("Target prediction changed")
            continue
        if (out / "locked.json").exists():
            raise ValueError("Cannot replace locked target predictions")
        arrays = predict_one(batch_to(item, device), cfg, scale, decision, models)
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite target scores")
        atomic_npz(path, **arrays)
        atomic_json(path.with_suffix(".json"), dict(identity=identity, sha=file_digest(path), id=row["id"]))
        print(f"predict/seed{args.train_seed}/{args.shard_id} {row['id']}", flush=True)


def lock_predictions(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    until = requested_horizon(args, cfg)
    out = prediction_dir(root, args.train_seed, until)
    identity = prediction_identity(root, cfg, args.train_seed, until)
    hashes = {}
    for row in rows_for(root, "eval"):
        path = out / f"{row['id']}.npz"
        meta = load_json(path.with_suffix(".json"))
        if meta["identity"] != identity or meta["sha"] != file_digest(path) or meta["id"] != row["id"]:
            raise ValueError("Cannot lock incomplete/mismatched target predictions")
        with np.load(path, allow_pickle=False) as data:
            if not set(ARMS) <= set(data.files):
                raise ValueError("Missing target comparison arm")
            if any(data[key].shape != (cfg["eval_size"], cfg["eval_size"]) for key in ARMS):
                raise ValueError("Target prediction geometry mismatch")
        hashes[row["id"]] = file_digest(path)
    value = dict(identity=identity, predictions=hashes,
                 note="All target predictions finalized before target masks are opened")
    path = out / "locked.json"
    if path.exists() and load_json(path) != value:
        raise ValueError("Target prediction lock changed")
    atomic_json(path, value)


def metric_row(masks, maps, threshold):
    from tools.vlm_next_diagnostics import map_metrics
    from sklearn.metrics import average_precision_score
    from scipy.ndimage import label
    point = map_metrics(masks, maps)
    point["P_AP"] = float(average_precision_score(masks.ravel(), maps.ravel())) if masks.any() else None
    prediction = maps > threshold
    normal = ~masks
    point["background_FPR_source_threshold"] = float(prediction[normal].mean()) if normal.any() else None
    point["pixel_recall_source_threshold"] = float(prediction[masks].mean()) if masks.any() else None
    total, hit = 0, 0
    for mask, predicted in zip(masks, prediction):
        components, count = label(mask, structure=np.ones((3, 3)))
        for index in range(1, count + 1):
            region = components == index
            if region.sum() <= mask.size * .001:
                total += 1
                hit += int(predicted[region].mean() >= .1)
    point.update(small_components=total, small_hits_source_threshold=hit,
                 threshold=threshold, images=len(masks), abnormal_images=int(masks.reshape(len(masks), -1).any(1).sum()))
    return point


def write_panel(root, cfg, row, predictions, seed, until):
    """Fixed hash-selected examples; fixed sigmoid score display, no per-image minmax."""
    from PIL import ImageDraw
    size = min(cfg["eval_size"], 384)
    with Image.open(row["image_path"]) as image:
        original = image.convert("RGB").resize((size, size))
    truth = evaluation_mask(row, size)
    tiles = [("query", original), ("GT (evaluation only)", Image.fromarray((truth * 255).astype(np.uint8)).convert("RGB"))]
    for name in ("D_G", "D_HR", "R_HR", "T_HR", "F_HR"):
        score = torch.from_numpy(predictions[name])[None, None]
        value = F.interpolate(score, (size, size), mode="bilinear", align_corners=False)[0, 0].sigmoid().numpy()
        color = np.stack([value, .15 * (1 - value), 1 - value], -1)
        overlay = (.55 * np.asarray(original) + .45 * color * 255).clip(0, 255).astype(np.uint8)
        tiles.append((name, Image.fromarray(overlay)))
    columns = 4
    canvas = Image.new("RGB", (columns * size, 2 * (size + 26)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(tiles):
        x, y = index % columns * size, index // columns * (size + 26)
        draw.text((x + 5, y + 5), title, fill="black")
        canvas.paste(image, (x, y + 26))
    path = Path(root) / "panels" / f"seed{seed}" / f"through{until}" / row["dataset"] / row["category"] / f"{row['id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    until = requested_horizon(args, cfg)
    out = prediction_dir(root, args.train_seed, until)
    locked = load_json(out / "locked.json")
    if locked["identity"] != prediction_identity(root, cfg, args.train_seed, until):
        raise ValueError("No valid target prediction lock")
    decision = load_json(decision_dir(root, args.train_seed, until) / "selection.json")
    groups = defaultdict(list)
    for row in rows_for(root, "eval"):
        groups[(row["dataset"], row["category"])].append(row)
    for index, (key, rows) in enumerate(sorted(groups.items())):
        if index % cfg["num_shards"] != args.shard_id:
            continue
        paths = [out / f"{row['id']}.npz" for row in rows]
        if any(locked["predictions"][r["id"]] != file_digest(p) for r, p in zip(rows, paths)):
            raise ValueError("Target maps changed after lock")
        # This is the FIRST target-mask access in the pipeline.
        masks = np.stack([evaluation_mask(row, cfg["eval_size"]) for row in rows])
        metrics = []
        for arm in ARMS:
            maps = []
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    maps.append(data[arm].copy())
            variant = arm.split("_")[1]
            threshold = decision["variants"][variant]["source_points"][arm[0]]["threshold"]
            point = metric_row(masks, np.stack(maps), threshold)
            point.update(dataset=key[0], category=key[1], arm=arm, seed=args.train_seed, horizon=until)
            metrics.append(point)
        # Record target-label hashes ONLY in evaluation artifacts, never config/cache.
        label_hashes = {r["id"]: file_digest(r["mask_path"]) if r["mask_path"] else None for r in rows}
        path = Path(root) / "metrics" / f"seed{args.train_seed}" / f"through{until}" / f"{key[0]}_{key[1]}.json"
        atomic_json(path, dict(prediction_lock_sha=file_digest(out / "locked.json"), metrics=metrics,
                               evaluation_mask_hashes=label_hashes))
        for row in sorted(rows, key=lambda r: digest([cfg["data_seed"], "panel", r["id"]]))[:2]:
            with np.load(out / f"{row['id']}.npz", allow_pickle=False) as data:
                write_panel(root, cfg, row, {k: data[k].copy() for k in ARMS}, args.train_seed, until)
        print(f"evaluated/seed{args.train_seed} {key}", flush=True)


@torch.inference_mode()
def audit(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    until, device = requested_horizon(args, cfg), device_for(args, cfg)
    decision, models = locked_models(root, cfg, args.train_seed, until, device)
    scale = stats_for(root, cfg, device)
    data = ResidualDataset(root, cfg, rows_for(root, "val")[:1], labels=False)
    batch = batch_to(data[0], device)
    checks = {}
    for variant in ("G", "HR"):
        residual = scaled_residual(batch, scale, variant)
        text = models[variant]["text"]
        score, _ = text(residual)
        zero, _ = text(torch.zeros_like(residual))
        # A sign-flipped language direction must flip its actual score. This
        # tests wiring, NOT semantic understanding or unseen-word generalization.
        original = text.directions.clone()
        text.directions.neg_()
        reversed_score, _ = text(residual)
        text.directions.copy_(original)
        zero_error, flip_error = float(zero.abs().max()), float((reversed_score + score).abs().max())
        zero_increment = apply_calibration(zero[:, :, :1], decision["variants"][variant]["text_calibration"]).clamp_min(0)
        increment_error = float(zero_increment.abs().max())
        if zero_error != 0 or increment_error != 0 or flip_error > 1e-5:
            raise ValueError("Text interface zero-residual/direction wiring audit failed")
        zero_density = models[variant]["density"](torch.zeros_like(residual))
        spatial_range = float((zero_density.flatten(-2).max(-1).values -
                               zero_density.flatten(-2).min(-1).values).abs().max())
        if spatial_range > 1e-5:
            raise ValueError("Zero residual produced a spatially varying density map")
        checks[variant] = dict(zero_residual_text_max=zero_error, direction_flip_error=flip_error,
                               zero_residual_text_increment_max=increment_error,
                               zero_residual_density_spatial_range=spatial_range,
                               density_kind=decision["variants"][variant]["density_kind"],
                               selected_strength=decision["variants"][variant]["strength"])
    atomic_json(decision_dir(root, args.train_seed, until) / "interface_audit.json",
                dict(checks=checks, semantic_scope=cfg["semantic_scope"],
                     note="Wiring checks are not evidence of fine-grained language understanding"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(args):
    root, cfg = Path(args.work_dir), checked_config(args.work_dir)
    until = requested_horizon(args, cfg)
    seeds = [int(x) for x in args.train_seeds.split(",")]
    categories = sorted({(r["dataset"], r["category"]) for r in rows_for(root, "eval")})
    rows, selections = [], []
    for seed in seeds:
        lock = prediction_dir(root, seed, until) / "locked.json"
        for dataset, category in categories:
            path = root / "metrics" / f"seed{seed}" / f"through{until}" / f"{dataset}_{category}.json"
            value = load_json(path)
            if value["prediction_lock_sha"] != file_digest(lock):
                raise ValueError("Report contains stale target metrics")
            rows.extend(value["metrics"])
        selected = load_json(decision_dir(root, seed, until) / "selection.json")
        for variant, point in selected["variants"].items():
            selections.append(dict(seed=seed, variant=variant, density_kind=point["density_kind"],
                                    density_epoch=point["density_epoch"], text_epoch=point["text_epoch"],
                                    strength=point["strength"], **{f"source_{k}": v for k, v in point["source_points"]["F"].items()}))
    numeric = ("PRO_exact", "P_AP", "P_AUROC", "I_AUROC", "background_FPR_source_threshold",
               "pixel_recall_source_threshold")
    per_seed, summary = [], []
    for dataset in sorted({r["dataset"] for r in rows}):
        for arm in ARMS:
            values = []
            for seed in seeds:
                subset = [r for r in rows if (r["dataset"], r["arm"], r["seed"]) == (dataset, arm, seed)]
                mean = {k: float(np.mean([r[k] for r in subset if r[k] is not None]))
                        if any(r[k] is not None for r in subset) else None for k in numeric}
                point = dict(dataset=dataset, arm=arm, seed=seed, categories=len(subset), **mean)
                per_seed.append(point)
                values.append(point)
            mean = dict(dataset=dataset, arm=arm, seeds=len(seeds))
            for key in numeric:
                valid = [r[key] for r in values if r[key] is not None]
                mean[key] = float(np.mean(valid)) if valid else None
                mean[key + "_seed_std"] = float(np.std(valid)) if valid else None
            summary.append(mean)
    comparisons = []
    for dataset in sorted({r["dataset"] for r in summary}):
        lookup = {r["arm"]: r for r in summary if r["dataset"] == dataset}
        for label, a, b in (("residual_learning_global", "R_G", "D_G"),
                            ("highres_distance", "D_HR", "D_G"),
                            ("highres_residual", "R_HR", "R_G"),
                            ("text_increment_global", "F_G", "R_G"),
                            ("text_increment_highres", "F_HR", "R_HR")):
            comparisons.append(dict(dataset=dataset, comparison=label,
                **{key + "_pp": 100 * (lookup[a][key] - lookup[b][key])
                   if lookup[a][key] is not None and lookup[b][key] is not None else None
                   for key in ("PRO_exact", "P_AP", "P_AUROC", "I_AUROC")}))
    out = root / "reports" / f"through{until}"
    for name, values in (("metrics", rows), ("per_seed", per_seed), ("summary", summary),
                         ("comparisons", comparisons), ("source_selection", selections)):
        write_csv(out / f"{name}.csv", values)
    lines = ["# Residual-first high-resolution text experiment", "",
             f"Protocol: {PROTOCOL}; K={cfg['reference_count']}; text: {cfg['semantic_scope']}.",
             "All target predictions were sealed before target masks were read.",
             "D=distance, R=residual distribution, T=text only, F=source-selected fusion; G=global, HR=global+local.",
             "density_kind=distance means source-selected fallback, not successful Flow learning.",
             "text_epoch=0 means an untrained projector; strength=0 means text did not qualify for fusion.",
             "Binary generic text alignment is NOT proof of fine-grained or unseen-defect language understanding.",
             f"Selection resolution={cfg['selection_size']}; final evaluation resolution={cfg['eval_size']}.",
             "F1_best (metrics.csv) is an evaluation oracle, not a deployed threshold.", "",
             "| Dataset | Arm | PRO | Pixel AP | Pixel AUROC | Image AUROC |", "|---|---|---:|---:|---:|---:|"]
    number = lambda x: "NA" if x is None else f"{100*x:.2f}"
    lines.extend(f"| {r['dataset']} | {r['arm']} | " + " | ".join(number(r[k]) for k in
                 ("PRO_exact", "P_AP", "P_AUROC", "I_AUROC")) + " |" for r in summary)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("prepare", "text_bank", "export", "seal", "fit_stats",
                   "train_density", "train_text", "select", "audit", "predict", "lock_predictions", "evaluate", "report"))
    p.add_argument("--work_dir", default="./checkpoint/residual_text_v1")
    p.add_argument("--partition", choices=("source", "eval"), default="source")
    p.add_argument("--variant", choices=("G", "HR"), default="G")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--probe", action="store_true")
    for name, default in dict(source="visa", eval_datasets="mvtec,btad,mpdd", source_root="", target_root="",
                              data_roots="{}", dino_repo_dir="./dinov3", dino_model_name="dinov3_vitl16",
                              dino_weights="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                              clip_weights="./CLIP/ckpt/ViT-L-14-336px.pt", visual_layers="5,11,17,23",
                              semantic_annotations="", text_strengths="0,0.05,0.1,0.25,0.5", train_seeds="42").items():
        p.add_argument("--" + name, default=default)
    for name, default in dict(data_seed=42, train_seed=42, image_size=512, eval_size=512, selection_size=256,
                              patch_size=16, input_dim=1024, feature_dim=256, reference_count=4,
                              match_chunk=128, bank_chunk=4096, density_epochs=15, text_epochs=15,
                              until_epoch=5, batch_size=8, micro_batch=1, workers=2, flow_blocks=8,
                              hidden=256, tokens_per_view=256, calibration_images=64, limit_per_category=0,
                              num_shards=1, shard_id=0).items():
        p.add_argument("--" + name, type=int, default=default)
    for name, default in dict(tile_fraction=.6, high_weight=.25, val_fraction=.25, val_fpr=.01,
                              density_lr=1e-4, text_lr=1e-4).items():
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return p


def main():
    args = parser().parse_args()
    if args.stage in ("train_density", "train_text"):
        train(args, args.stage.removeprefix("train_"))
    else:
        globals()[args.stage](args)


if __name__ == "__main__":
    main()
