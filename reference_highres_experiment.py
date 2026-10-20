#!/usr/bin/env python3
"""Source-supervised 2x2 experiment: high resolution x normal-reference evidence.

Frozen CLIP+DINO; four matched lightweight heads; source-only epoch/alpha selection.
No VLM, MARA, target-mask training, or implicit residual/text-space equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from tools.reference_highres import (
    ARMS, EvidenceHead, FrozenExtractor, arm_inputs, arm_options, assemble_tiles,
    covering_boxes, pack_view, patch_coordinates, supervision_loss,
)
from tools.upstream_repair import corrected_probability, reference_logits
from tools.vlm_review import atomic_json, digest, file_digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "reference_highres_factorial_v1"
CODE_FILES = ("reference_highres_experiment.py", "tools/reference_highres.py", "dual_vlm.py",
              "tools/upstream_repair.py", "tools/upstream_localization.py", "tools/vlm_review.py",
              "tools/utils_up.py", "train_mara.py", "train_up.py", "tools/promptLearner.py",
              "tools/bottleneckAdapter.py", "CLIP/adapter.py", "utils.py",
              "tools/vlm_next_diagnostics.py")


def atomic_npz(path, **arrays):
    """Uncompressed arrays: avoid repeated DEFLATE work during four head runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def image_identity(path):
    """Decoded RGB identity catches identical images with different containers."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        return sha, list(rgb.size)


def record_for(dataset, category, entry, partition):
    _, state, image, mask = entry
    path = str(Path(image).resolve())
    rgb_sha, size = image_identity(path)
    return dict(id=digest([dataset, category, path])[:24], dataset=dataset, category=category,
                partition=partition, image_path=path, image_sha=file_digest(path), rgb_sha=rgb_sha,
                native_size=size, mask_path=str(Path(mask).resolve()) if mask else None,
                mask_sha=file_digest(mask) if mask else None, state=str(state))


def source_split(rows, fraction, seed):
    """Split decoded-content families before creating normal-reference records."""
    strata = defaultdict(dict)
    for row in rows:
        strata[(row["category"], bool(row["mask_path"]))].setdefault(row["rgb_sha"], []).append(row)
    for families in strata.values():
        ordered = sorted(families, key=lambda sha: digest([seed, sha]))
        if len(ordered) < 2:
            raise ValueError("Need >=2 unique source images per category/normal-anomaly stratum; increase LIMIT_PER_CATEGORY")
        count = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
        for index, sha in enumerate(ordered):
            for row in families[sha]:
                row["partition"] = "val" if index < count else "train"


def validate_manifest(rows, references):
    seen, ids = {}, set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate query record")
        ids.add(row["id"])
        previous = seen.setdefault(row["rgb_sha"], row["partition"])
        if previous != row["partition"]:
            raise ValueError("Identical decoded image content crosses train/val/target partitions")
    ref_seen = {}
    for reference in references:
        if reference["partition"] != "reference_train" or reference["mask_path"]:
            raise ValueError("References must be known-normal TRAIN images")
        if reference["rgb_sha"] in seen:
            raise ValueError("Normal reference duplicates a query (including re-encoded images)")
        key = (reference["dataset"], reference["category"])
        prior = ref_seen.setdefault(reference["rgb_sha"], key)
        if prior != key:
            raise ValueError("Normal reference content crosses dataset/category banks")
    keys = {(r["dataset"], r["category"]) for r in references}
    if any((r["dataset"], r["category"]) not in keys for r in rows):
        raise ValueError("A query category has no valid normal TRAIN reference")


def cfg_load(root):
    root = Path(root)
    cfg = load_json(root / "config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong experiment protocol")
    for name, sha in cfg["code"].items():
        if file_digest(REPO / name) != sha:
            raise ValueError(f"Code changed: {name}; use NEW WORK_DIR")
    for name in ("manifest", "references"):
        if digest(load_json(root / f"{name}.json")) != cfg[name + "_sha"]:
            raise ValueError(f"{name} changed; use NEW WORK_DIR")
    return cfg


def records(root, partition=None):
    rows = load_json(Path(root) / "manifest.json")
    return [r for r in rows if partition is None or r["partition"] == partition]


def unchanged(row, masks=False):
    if file_digest(row["image_path"]) != row["image_sha"]:
        raise ValueError(f"Query/reference image changed: {row['id']}")
    if masks and row["mask_path"] and file_digest(row["mask_path"]) != row["mask_sha"]:
        raise ValueError(f"Supervision mask changed: {row['id']}")


def prepare(args):
    from Datasets import DATASET_CLASSES, DATASET_REGISTRY
    from dual_vlm import check_dual_payload, resolve_checkpoint
    root = Path(args.work_dir)
    targets = args.eval_datasets.replace(",", " ").split()
    supported = {"visa", "mvtec", "btad", "mpdd"}
    if (args.source not in supported or not targets or not set(targets) <= supported
            or args.source in targets or len(set(targets)) != len(targets)):
        raise ValueError("Use one source and distinct held-out target datasets")
    if (not .05 <= args.val_fraction <= .5 or args.limit_per_category < 0 or args.num_shards < 1
            or args.image_size < 64 or args.image_size % 16 or args.feature_grid < args.image_size // 16
            or not 1 <= args.feature_dim <= 1024 or args.reference_count < 1
            or args.reference_pool < args.reference_count or not 0 < args.high_weight <= 1
            or args.match_topk < 1 or args.match_temperature <= 0 or args.match_radius < 0
            or args.spatial_weight < 0 or args.match_chunk < 1 or args.epochs < 1
            or args.batch_size < 1 or not 1 <= args.micro_batch <= args.batch_size
            or args.hidden < 8 or args.hidden % 8 or args.lr <= 0 or args.workers < 0
            or min(args.semantic_weight, args.background_weight, args.residual_weight) < 0
            or args.eval_size < 32 or not 0 < args.val_fpr < 1 or args.precision not in ("bf16", "fp32")):
        raise ValueError("Invalid geometry, matching, training or evaluation settings")
    covering_boxes(100, 100, args.tile_grid, args.tile_fraction)
    alphas = sorted(set(float(v) for v in args.alphas.split(",")) | {0.})
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in alphas):
        raise ValueError("Alphas must be finite in [0,1]; Base fallback 0 is always included")
    base = resolve_checkpoint(args.base_ckpt)
    payload = torch.load(base, map_location="cpu", weights_only=False)
    check_dual_payload(payload)
    if payload.get("visual_backbone", "dino") != "dino":
        raise ValueError("Expected a DINO visual backbone")
    # Honor the actual Base architecture; explicit weight/repository CLI paths remain portable.
    layers = payload.get("visual_layers", [int(v) for v in args.visual_layers.split(",")])
    layer_string = ",".join(str(int(v)) for v in layers)
    if len(layers) != 4 or len(set(layers)) != 4:
        raise ValueError("This matched legacy experiment requires four distinct Base feature layers")
    # load_dual historically honors paths embedded in payloads. Reject nonportable conflicts
    # instead of silently using different frozen weights than the audited CLI paths.
    for key in ("dino_weights", "dino_repo_dir", "dino_model_name"):
        if payload.get(key) and str(payload[key]) != str(getattr(args, key)):
            raise ValueError(f"Base embeds {key}={payload[key]!r}; pass that matching value explicitly")
    if not Path(args.dino_weights).is_file():
        raise FileNotFoundError(args.dino_weights)
    roots = json.loads(args.data_roots)
    if not isinstance(roots, dict) or not set(roots) <= supported or not all(isinstance(v, str) and v for v in roots.values()):
        raise ValueError("--data_roots must be a JSON map of dataset names to local directories")
    if args.source_root:
        roots[args.source] = args.source_root
    if args.target_root:
        if len(targets) != 1:
            raise ValueError("TARGET_ROOT requires a single target; use DATA_ROOTS_JSON for multiple targets")
        roots[targets[0]] = args.target_root
    rows, datasets = [], [args.source] + targets
    for name in datasets:
        dataset_cls, splits, default_root = DATASET_REGISTRY[name]
        source = roots.get(name, default_root)
        for category in sorted(DATASET_CLASSES[name]):
            dataset = dataset_cls(source=source, classname=category, split=splits.TEST,
                                  resize=args.image_size, imagesize=args.image_size)
            entries = sorted(dataset.data_to_iterate, key=lambda e: digest([args.seed, name, category, str(e[2])]))
            if args.limit_per_category:
                entries = entries[:args.limit_per_category]
            rows.extend(record_for(name, category, e, "source" if name == args.source else "eval") for e in entries)
    source_split([r for r in rows if r["partition"] == "source"], args.val_fraction, args.seed)
    query_shas = {r["rgb_sha"] for r in rows}
    references = []
    for name in datasets:
        dataset_cls, splits, default_root = DATASET_REGISTRY[name]
        for category in sorted(DATASET_CLASSES[name]):
            dataset = dataset_cls(source=roots.get(name, default_root), classname=category, split=splits.TRAIN,
                                  resize=args.image_size, imagesize=args.image_size)
            normal = "Normal" if name == "visa" else "ok" if name == "btad" else "good"
            entries = sorted((e for e in dataset.data_to_iterate if e[1] == normal and not e[3]),
                             key=lambda e: digest([args.seed, name, category, "normal_train", str(e[2])]))
            chosen, seen = [], set()
            for entry in entries:
                row = record_for(name, category, entry, "reference_train")
                if row["rgb_sha"] in query_shas or row["rgb_sha"] in seen:
                    continue
                chosen.append(row)
                seen.add(row["rgb_sha"])
                if len(chosen) == args.reference_pool:
                    break
            if len(chosen) < args.reference_count:
                raise ValueError(f"Insufficient nonduplicated normal TRAIN images: {name}/{category}")
            references.extend(chosen)
    validate_manifest(rows, references)
    counts = Counter(r["partition"] for r in rows)
    if any(counts[p] == 0 for p in ("train", "val", "eval")):
        raise ValueError("Empty experiment partition")
    excluded = {"stage", "work_dir", "shard_id", "mode", "until_epoch", "device", "source_root", "target_root", "data_roots"}
    cfg = {k: v for k, v in vars(args).items() if k not in excluded}
    cfg.update(protocol=PROTOCOL, base_ckpt=str(base), base_sha=file_digest(base),
               dino_sha=file_digest(args.dino_weights), visual_layers=layer_string,
               clip_sha=file_digest(args.clip_pretrained) if Path(args.clip_pretrained).is_file() else None,
               eval_datasets=targets, data_roots=roots, alphas=alphas,
               manifest_sha=digest(rows), references_sha=digest(references),
               code={p: file_digest(REPO / p) for p in CODE_FILES},
               reference_protocol="known-normal same-category TRAIN only; target-normal support, not strict zero-shot",
               selection="source-val BCE+positive-image Dice; epoch and alpha only; includes per-arm frozen-readout fallback",
               mask_geometry="nearest resize; VisA >0, other datasets >127")
    for name, value in (("config", cfg), ("manifest", rows), ("references", references)):
        path = root / f"{name}.json"
        if path.exists() and load_json(path) != value:
            raise ValueError("Settings/data/code changed; use NEW WORK_DIR")
    for name, value in (("manifest", rows), ("references", references), ("config", cfg)):
        atomic_json(root / f"{name}.json", value)
    low = args.image_size // 16
    approximate = (2 * len(layers) * (2 * args.feature_dim + 3) * (low * low + args.feature_grid ** 2)
                   + 8 * args.image_size ** 2)
    print(f"Prepared {dict(counts)}, refs={len(references)}, estimated cache={approximate * len(rows) / 2**30:.2f} GiB", flush=True)
    print("A/B: single-image; C/D: same-category known-normal TRAIN support. Only source masks train/select heads.", flush=True)
    print("Source TEST supervision is inherited; its val may already have been seen by the frozen Base.", flush=True)


def device_for(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def verify_weights(cfg):
    for key, sha in (("base_ckpt", "base_sha"), ("dino_weights", "dino_sha"), ("clip_pretrained", "clip_sha")):
        if cfg[sha] and file_digest(cfg[key]) != cfg[sha]:
            raise ValueError(f"Frozen weights changed: {key}")


def cpu_view(view):
    return {key: value.detach().cpu() for key, value in view.items()}


def export(args, probe=False):
    root, started = Path(args.work_dir), time.monotonic()
    cfg = cfg_load(root)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard id")
    assigned = records(root)[:1] if probe else records(root)[args.shard_id::cfg["num_shards"]]
    pending = []
    for row in assigned:
        path = root / "features" / f"{row['id']}.npz"
        meta = path.with_suffix(".json")
        if not probe and path.exists() and meta.exists():
            audit = load_json(meta)
            if audit["fingerprint"] != digest(cfg) or audit["sha"] != file_digest(path):
                raise ValueError("Changed feature cache; use NEW WORK_DIR")
            continue
        if not probe and (root / "sealed.json").exists():
            raise ValueError("Cannot regenerate a sealed cache; use NEW WORK_DIR")
        pending.append(row)
    if not pending:
        print(f"export shard={args.shard_id}: already complete", flush=True)
        return
    verify_weights(cfg)
    device = device_for(args)
    extractor = FrozenExtractor(cfg, device)
    reference_rows = load_json(root / "references.json")
    current_key, bank = None, []
    try:
        anchors = extractor.text.float().cpu().numpy()
        anchor_path = root / "anchors.npz"
        if anchor_path.exists():
            with np.load(anchor_path, allow_pickle=False) as data:
                if data["anchors"].shape != anchors.shape or not np.allclose(data["anchors"], anchors, atol=1e-6):
                    raise ValueError("Text anchors changed")
        else:
            atomic_npz(anchor_path, anchors=anchors)
        for index, row in enumerate(pending):
            unchanged(row)
            key = row["dataset"], row["category"]
            if key != current_key:
                bank = []
                for ref in reference_rows:
                    if (ref["dataset"], ref["category"]) != key:
                        continue
                    unchanged(ref)
                    with Image.open(ref["image_path"]) as image:
                        g, tiles, boxes = extractor.image_views(image.convert("RGB"))
                    bank.append(dict(row=ref, global_view=cpu_view(g), tiles=[cpu_view(t) for t in tiles], boxes=boxes,
                                     descriptor=F.normalize(g["raw"].mean((0, 1)), dim=0).cpu()))
                current_key = key
            if any(ref["row"]["rgb_sha"] == row["rgb_sha"] for ref in bank):
                raise ValueError("Query/self-reference leakage")
            tick = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with Image.open(row["image_path"]) as image:
                native = image.convert("RGB")
                global_view, tiles, boxes = extractor.image_views(native)
            descriptor = F.normalize(global_view["raw"].mean((0, 1)), dim=0).cpu()
            chosen = sorted(bank, key=lambda ref: (-float(descriptor @ ref["descriptor"]), ref["row"]["id"]))[:cfg["reference_count"]]
            if len(chosen) != cfg["reference_count"]:
                raise ValueError("Missing normal reference bank")
            side = cfg["image_size"] // 16
            xy = patch_coordinates(side, device=device)
            global_ref = torch.cat([ref["global_view"]["raw"].to(device) for ref in chosen], dim=1)
            global_xy = xy.repeat(len(chosen), 1)
            low = pack_view(global_view, global_ref, xy, global_xy, extractor.project, cfg)
            high_refs, high_xy = [], []
            for ref in chosen:
                for tile, box in zip(ref["tiles"], ref["boxes"]):
                    high_refs.append(tile["raw"].to(device))
                    high_xy.append(patch_coordinates(side, box, device))
            high_ref, reference_xy = torch.cat(high_refs, 1), torch.cat(high_xy)
            packed_tiles = [pack_view(tile, high_ref, patch_coordinates(side, box, device), reference_xy,
                                      extractor.project, cfg) for tile, box in zip(tiles, boxes)]
            high = assemble_tiles(packed_tiles, boxes, cfg["feature_grid"], cfg["image_size"])
            arrays = {prefix + "_" + key: value.cpu().numpy().astype(np.float32 if key == "base" else np.float16)
                      for prefix, view in (("global", low), ("high", high)) for key, value in view.items()}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError("Nonfinite frozen features")
            if probe:
                # Independently verify that the unmodified Base is reproduced.
                from tools.utils_up import get_anomaly_map
                with Image.open(row["image_path"]) as im:
                    rgb = np.asarray(im.convert("RGB").resize((cfg["image_size"],) * 2, Image.Resampling.BILINEAR)).copy()
                x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
                x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor([.229, .224, .225])[:, None, None]
                # Prompt is reconstructed from the SAME frozen checkpoint, never a new initialization.
                from train_up import create_prompt_learner
                prompt = create_prompt_learner(extractor.clip, device)
                payload = torch.load(cfg["base_ckpt"], map_location="cpu", weights_only=False)
                prompt.load_state_dict(payload["prompt_learner"], strict=True)
                prompt.eval()
                info = dict(image=x[None], image_path=[row["image_path"]], mask=torch.zeros(1, 1, *x.shape[-2:]))
                with torch.inference_mode():
                    _, _, expected, _ = get_anomaly_map(extractor.clip, info, device, extractor.adapter, extractor.dino,
                                                       prompt, 1, visual_layers=extractor.layers)
                error = float((expected[0, 1:2] - low["base"]).abs().max())
                if error > 1e-5:
                    raise ValueError(f"Base parity preflight failed: max error={error}")
                print(f"Preflight: Base max-error={error:.3g}; low={low['query'].shape}; high={high['query'].shape}; anchors={anchors.shape}", flush=True)
                return
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.monotonic() - tick
            path = root / "features" / f"{row['id']}.npz"
            atomic_npz(path, **arrays)
            atomic_json(path.with_suffix(".json"), dict(fingerprint=digest(cfg), sha=file_digest(path),
                        references=[ref["row"]["id"] for ref in chosen],
                        reference_similarity=[float(descriptor @ ref["descriptor"]) for ref in chosen],
                        boxes=boxes, native_size=row["native_size"], query_seconds=seconds,
                        query_views=1 + len(tiles), peak_allocated_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                                                                 if device.type == "cuda" else None)))
            print(f"export shard={args.shard_id} {index+1}/{len(pending)} {key} seconds={seconds:.2f}", flush=True)
    finally:
        extractor.close()
        print(f"export shard={args.shard_id} total_seconds={time.monotonic()-started:.1f} (includes bank encoding)", flush=True)


def seal(args):
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    references = {r["id"]: r for r in load_json(root / "references.json")}
    files, metadata_files, layers = {}, {}, len(cfg["visual_layers"].split(","))
    for row in records(root):
        path = root / "features" / f"{row['id']}.npz"
        audit = load_json(path.with_suffix(".json"))
        sha = file_digest(path)
        if audit["fingerprint"] != digest(cfg) or audit["sha"] != sha:
            raise ValueError("Stale/modified export")
        if len(set(audit["references"])) != cfg["reference_count"]:
            raise ValueError("Wrong reference count")
        for ident in audit["references"]:
            ref = references[ident]
            if ((ref["dataset"], ref["category"]) != (row["dataset"], row["category"])
                    or ref["rgb_sha"] == row["rgb_sha"]):
                raise ValueError("Invalid reference provenance")
        with np.load(path, allow_pickle=False) as data:
            for prefix, grid in (("global", cfg["image_size"] // 16), ("high", cfg["feature_grid"])):
                for key in ("query", "residual", "distance", "uncertainty", "prior", "base"):
                    shape = ((layers, cfg["feature_dim"], grid, grid) if key in ("query", "residual") else
                             (1, cfg["image_size"], cfg["image_size"]) if key == "base" else (layers, grid, grid))
                    value = data[prefix + "_" + key]
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError("Invalid feature cache shape/values")
                    if key in ("prior", "base", "uncertainty") and ((value < 0).any() or (value > 1.001).any()):
                        raise ValueError("Invalid cached probability/uncertainty")
        files[row["id"]] = sha
        metadata_files[row["id"]] = file_digest(path.with_suffix(".json"))
    with np.load(root / "anchors.npz", allow_pickle=False) as data:
        if data["anchors"].shape[0] != 2 or not np.isfinite(data["anchors"]).all():
            raise ValueError("Invalid frozen text anchors")
    value = dict(fingerprint=digest(cfg), features=files, metadata=metadata_files,
                 anchors_sha=file_digest(root / "anchors.npz"))
    if (root / "sealed.json").exists() and load_json(root / "sealed.json") != value:
        raise ValueError("Sealed cache changed; use NEW WORK_DIR")
    atomic_json(root / "sealed.json", value)
    print(f"Sealed {len(files)} label-free feature records", flush=True)


def mask_for(row, size):
    if not row["mask_path"]:
        return torch.zeros(1, size, size)
    with Image.open(row["mask_path"]) as image:
        array = np.asarray(image.convert("L").resize((size, size), Image.Resampling.NEAREST))
    return torch.from_numpy((array > (0 if row["dataset"] == "visa" else 127)).copy()).float()[None]


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, root, cfg, rows, arm, include_mask=True):
        self.root, self.cfg, self.rows, self.arm, self.include_mask = Path(root), cfg, rows, arm, include_mask
        sealed = load_json(self.root / "sealed.json")
        if sealed["fingerprint"] != digest(cfg) or sealed["anchors_sha"] != file_digest(self.root / "anchors.npz"):
            raise ValueError("Stale feature seal")
        for row in rows:
            if file_digest(self.root / "features" / f"{row['id']}.npz") != sealed["features"][row["id"]]:
                raise ValueError("Feature cache changed after sealing")
            if file_digest(self.root / "features" / f"{row['id']}.json") != sealed["metadata"][row["id"]]:
                raise ValueError("Reference selection metadata changed after sealing")
            if include_mask and row["mask_path"] and file_digest(row["mask_path"]) != row["mask_sha"]:
                raise ValueError("Mask changed after prepare")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        high, _ = arm_options(self.arm)
        prefixes = ("global", "high") if high else ("global",)
        with np.load(self.root / "features" / f"{row['id']}.npz", allow_pickle=False) as data:
            arrays = {key: torch.from_numpy(data[key].astype(np.float32))[None]
                      for key in data.files if any(key.startswith(prefix + "_") for prefix in prefixes)}
        inputs = {k: v[0] for k, v in arm_inputs(arrays, self.arm, self.cfg["feature_grid"], self.cfg["high_weight"]).items()}
        if self.include_mask:
            inputs["mask"] = mask_for(row, self.cfg["image_size"])
        return inputs


def anchors_for(root):
    with np.load(Path(root) / "anchors.npz", allow_pickle=False) as data:
        return torch.from_numpy(data["anchors"].copy())


def model_for(root, cfg, device):
    return EvidenceHead(len(cfg["visual_layers"].split(",")), cfg["feature_dim"], anchors_for(root), cfg["hidden"]).to(device)


def autocast(device, cfg):
    from contextlib import nullcontext
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and cfg["precision"] == "bf16" else nullcontext()


def segmentation_score(base, delta, mask, alpha):
    probability = corrected_probability(base, delta, alpha)
    logits = reference_logits(base) + alpha * delta.double()
    bce = F.binary_cross_entropy_with_logits(logits, mask.double(), reduction="none").flatten(1).mean(1)
    p, y = probability.flatten(1), mask.flatten(1)
    dice = (1 - (2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)) * (y.sum(1) > 0)
    return bce + dice


@torch.inference_mode()
def validate(model, loader, cfg, device):
    model.eval()
    totals, count, backgrounds = {a: 0. for a in cfg["alphas"]}, 0, {a: [] for a in cfg["alphas"]}
    for batch in loader:
        for start in range(0, len(batch["base"]), cfg["micro_batch"]):
            part = {k: v[start:start + cfg["micro_batch"]].to(device) for k, v in batch.items()}
            with autocast(device, cfg):
                delta = model(part)["delta"]
            for alpha in totals:
                totals[alpha] += float(segmentation_score(part["base"], delta, part["mask"], alpha).sum())
                p = corrected_probability(part["base"], delta, alpha)
                # Deterministic source-only pixels: identical grid for every arm/epoch/alpha.
                p = F.interpolate(p, (cfg["eval_size"],) * 2, mode="bilinear", align_corners=False)
                target = F.interpolate(part["mask"], (cfg["eval_size"],) * 2, mode="nearest")
                bg = (target == 0).flatten(1)
                flat = p.flatten(1)
                indices = torch.arange(0, flat.shape[1], max(1, flat.shape[1] // 8192), device=device)
                backgrounds[alpha].append(flat[:, indices][bg[:, indices]].float().cpu().numpy())
            count += len(part["base"])
    if not count:
        raise ValueError("Empty source validation set")
    values = {a: totals[a] / count for a in totals}
    thresholds = {a: float(np.quantile(np.concatenate(backgrounds[a]), 1 - cfg["val_fpr"], method="higher"))
                  for a in totals}
    return values, thresholds


def train(args):
    from torch.utils.data import DataLoader
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    arm_options(args.mode)
    until = args.until_epoch or cfg["epochs"]
    if not 1 <= until <= cfg["epochs"]:
        raise ValueError("UNTIL_EPOCH must be within the originally planned EPOCHS")
    device = device_for(args)
    if device.type == "cuda" and cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unsupported; use PRECISION=fp32 in a new WORK_DIR")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = model_for(root, cfg, device)
    train_data = FeatureDataset(root, cfg, records(root, "train"), args.mode)
    val_data = FeatureDataset(root, cfg, records(root, "val"), args.mode)
    val_loader = DataLoader(val_data, batch_size=cfg["micro_batch"], num_workers=cfg["workers"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=.01)
    total_steps = math.ceil(len(train_data) / cfg["batch_size"]) * cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .05 + .95 * (1 + math.cos(math.pi * step / max(1, total_steps))) / 2)
    out = root / "heads" / args.mode
    identity = dict(fingerprint=digest(cfg), cache_sha=file_digest(root / "sealed.json"), arm=args.mode)
    start, history = 0, []
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Head/cache configuration mismatch")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best, history = saved["epoch"] + 1, saved["best_loss"], saved["history"]
        if not (out / "best.pt").exists():
            raise ValueError("Missing best checkpoint for resumed training")
    else:
        values, thresholds = validate(model, val_loader, cfg, device)
        best = values[0.]
        save_torch(out / "best.pt", dict(state_dict=model.state_dict(), identity=identity,
                   epoch=-1, alpha=0., source_val_loss=best, threshold=thresholds[0.]))
    for epoch in range(start, until):
        tick = time.monotonic()
        model.train()
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
                            num_workers=cfg["workers"])
        loss_sum, samples = 0., 0
        for batch in loader:
            count = len(batch["base"])
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, count, cfg["micro_batch"]):
                part = {k: v[offset:offset + cfg["micro_batch"]].to(device) for k, v in batch.items()}
                with autocast(device, cfg):
                    output = model(part)
                    losses = supervision_loss(part["base"], output, part["mask"], cfg["semantic_weight"],
                                               cfg["background_weight"], cfg["residual_weight"])
                    loss = losses.sum() / count
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite head loss")
                loss.backward()
                loss_sum += float(losses.detach().sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            samples += count
        values, thresholds = validate(model, val_loader, cfg, device)
        alpha = min(cfg["alphas"], key=lambda a: (values[a], a))
        if values[alpha] < best:
            best = values[alpha]
            save_torch(out / "best.pt", dict(state_dict=model.state_dict(), identity=identity,
                       epoch=epoch, alpha=alpha, source_val_loss=best, threshold=thresholds[alpha]))
        current = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / samples,
                            source_val={str(a): v for a, v in values.items()},
                            best_source_val_loss=best, selected_alpha=current["alpha"],
                            best_epoch=current["epoch"] + 1, seconds=time.monotonic() - tick,
                            samples_per_second=samples / max(time.monotonic() - tick, 1e-6)))
        save_torch(out / "last.pt", dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), epoch=epoch, identity=identity, best_loss=best, history=history))
        atomic_json(out / "history.json", history)
        print(f"{args.mode} epoch={epoch+1}/{cfg['epochs']} loss={loss_sum/samples:.5f} best={best:.5f} alpha={current['alpha']}", flush=True)
    print(f"{args.mode}: completed {max(start, until)} planned epochs (resume uses same data order/optimizer)", flush=True)


def small_metrics(masks, maps, baseline, threshold, base_threshold):
    from scipy.ndimage import label
    counts = Counter(total=0, hit=0, recovered=0, lost=0)
    for mask, prob, base in zip(masks, maps, baseline):
        cc, count = label(mask, structure=np.ones((3, 3)))
        for component in range(1, count + 1):
            support = cc == component
            if support.sum() > mask.size * .001:
                continue
            hit = float((prob[support] > threshold).mean()) >= .1
            old = float((base[support] > base_threshold).mean()) >= .1
            counts.update(total=1, hit=int(hit), recovered=int(hit and not old), lost=int(old and not hit))
    return dict(small_components=counts["total"], small_hits_source_threshold=counts["hit"],
                recovered_small_source_threshold=counts["recovered"], lost_small_source_threshold=counts["lost"])


def source_base_thresholds(root, cfg):
    """Only source-val masks are consulted; targets never calibrate thresholds."""
    values = {name: [] for name in ("base_global", "base_highres")}
    for row in records(root, "val"):
        mask = mask_for(row, cfg["eval_size"])[0].numpy().astype(bool)
        with np.load(root / "features" / f"{row['id']}.npz", allow_pickle=False) as data:
            low = data["global_base"][0]
            high = (1 - cfg["high_weight"]) * low + cfg["high_weight"] * data["high_base"][0]
        for name, maps in (("base_global", low), ("base_highres", high)):
            maps = F.interpolate(torch.from_numpy(maps)[None, None], (cfg["eval_size"],) * 2,
                                 mode="bilinear", align_corners=False)[0, 0].numpy()
            stride = max(1, maps.size // 8192)
            values[name].append(maps.ravel()[::stride][~mask.ravel()[::stride]])
    return {name: float(np.quantile(np.concatenate(parts), 1 - cfg["val_fpr"], method="higher"))
            for name, parts in values.items()}


def evaluate(args):
    from tools.vlm_next_diagnostics import map_metrics
    from sklearn.metrics import average_precision_score
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard id")
    device = device_for(args)
    models, metadata, hashes = {}, {}, {}
    for arm in ARMS:
        path = root / "heads" / arm / "best.pt"
        value = torch.load(path, map_location="cpu", weights_only=False)
        expected = dict(fingerprint=digest(cfg), cache_sha=file_digest(root / "sealed.json"), arm=arm)
        if value["identity"] != expected:
            raise ValueError("Evaluation head/cache mismatch")
        model = model_for(root, cfg, device)
        model.load_state_dict(value["state_dict"])
        models[arm], metadata[arm], hashes[arm] = model.eval(), value, file_digest(path)
    thresholds = source_base_thresholds(root, cfg)
    thresholds.update({arm: value["threshold"] for arm, value in metadata.items()})
    groups = defaultdict(list)
    for row in records(root, "eval"):
        groups[(row["dataset"], row["category"])].append(row)
    for dataset, category in sorted(groups)[args.shard_id::cfg["num_shards"]]:
        group = groups[(dataset, category)]
        path = root / "results" / f"{dataset}_{category}.json"
        # Verify data even when a previous evaluation can otherwise be reused.
        FeatureDataset(root, cfg, group, ARMS[0], include_mask=False)
        for row in group:
            if row["mask_path"] and file_digest(row["mask_path"]) != row["mask_sha"]:
                raise ValueError("Target mask changed after prepare")
        if path.exists():
            old = load_json(path)
            if old["fingerprint"] == digest(cfg) and old["head_hashes"] == hashes:
                continue
        # Validate the immutable cache without opening target masks in model inputs.
        predictions, masks, head_seconds = defaultdict(list), [], Counter()
        for row in group:
            with np.load(root / "features" / f"{row['id']}.npz", allow_pickle=False) as data:
                arrays = {key: torch.from_numpy(data[key].astype(np.float32))[None].to(device) for key in data.files}
            low, high = arrays["global_base"], (1 - cfg["high_weight"]) * arrays["global_base"] + cfg["high_weight"] * arrays["high_base"]
            outputs = dict(base_global=low, base_highres=high)
            with torch.inference_mode():
                for arm, model in models.items():
                    inputs = arm_inputs(arrays, arm, cfg["feature_grid"], cfg["high_weight"])
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tick = time.monotonic()
                    with autocast(device, cfg):
                        delta = model(inputs)["delta"]
                    outputs[arm] = corrected_probability(inputs["base"], delta, metadata[arm]["alpha"])
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    head_seconds[arm] += time.monotonic() - tick
                for mode, prob in outputs.items():
                    prob = F.interpolate(prob, (cfg["eval_size"],) * 2, mode="bilinear", align_corners=False)
                    predictions[mode].append(prob[0, 0].float().cpu().numpy())
            # Target labels enter only AFTER every model prediction for this query.
            if row["mask_path"] and file_digest(row["mask_path"]) != row["mask_sha"]:
                raise ValueError("Target mask changed after prepare")
            masks.append(mask_for(row, cfg["eval_size"])[0].numpy().astype(bool))
        masks = np.asarray(masks)
        global_maps = np.asarray(predictions["base_global"])
        rows = []
        for mode, values in predictions.items():
            maps = np.asarray(values)
            metrics = map_metrics(masks, maps)
            metrics["P_AP"] = float(average_precision_score(masks.ravel(), maps.ravel())) if masks.any() else None
            background = ~masks
            threshold = thresholds[mode]
            rows.append(dict(dataset=dataset, category=category, arm=mode, samples=len(group), **metrics,
                             **small_metrics(masks, maps, global_maps, threshold, thresholds["base_global"]),
                             source_threshold=threshold,
                             background_FPR_source_threshold=float((maps[background] > threshold).mean()) if background.any() else None,
                             pixel_recall_source_threshold=float((maps[masks] > threshold).mean()) if masks.any() else None,
                             changed_fraction=float((np.abs(maps-global_maps) > 1e-6).mean()),
                             head_seconds_per_image=head_seconds[mode] / len(group) if mode in models else None,
                             alpha=metadata[mode]["alpha"] if mode in metadata else 0.,
                             selected_epoch=metadata[mode]["epoch"] + 1 if mode in metadata else 0))
        atomic_json(path, dict(fingerprint=digest(cfg), head_hashes=hashes, rows=rows))
        print(f"evaluated {dataset}/{category} samples={len(group)}", flush=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(args):
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    hashes = {arm: file_digest(root / "heads" / arm / "best.pt") for arm in ARMS}
    groups = sorted({(r["dataset"], r["category"]) for r in records(root, "eval")})
    rows = []
    for dataset, category in groups:
        value = load_json(root / "results" / f"{dataset}_{category}.json")
        if value["fingerprint"] != digest(cfg) or value["head_hashes"] != hashes:
            raise ValueError("Evaluation stale/missing; run evaluate again")
        rows.extend(value["rows"])
    means = []
    metric_names = ("PRO_exact", "P_AUROC", "I_AUROC", "P_AP", "F1_best", "background_FPR_source_threshold",
                    "pixel_recall_source_threshold", "changed_fraction", "head_seconds_per_image")
    count_names = ("small_components", "small_hits_source_threshold", "recovered_small_source_threshold", "lost_small_source_threshold")
    for dataset in cfg["eval_datasets"]:
        for arm in ("base_global", "base_highres") + ARMS:
            selected = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            value = dict(dataset=dataset, category="MEAN", arm=arm, samples=sum(r["samples"] for r in selected))
            for key in metric_names:
                valid = [r[key] for r in selected if r[key] is not None]
                value[key] = float(np.mean(valid)) if valid else None
            value.update({key: sum(r[key] for r in selected) for key in count_names})
            means.append(value)
    comparisons = []
    for dataset in cfg["eval_datasets"]:
        indexed = {r["arm"]: r for r in means if r["dataset"] == dataset}
        contrasts = (("matched_head", ARMS[0], "base_global"), ("frozen_highres", "base_highres", "base_global"),
                     ("highres_no_reference", ARMS[1], ARMS[0]), ("reference_lowres", ARMS[2], ARMS[0]),
                     ("reference_highres", ARMS[3], ARMS[1]), ("highres_with_reference", ARMS[3], ARMS[2]))
        for name, candidate, baseline in contrasts:
            comparisons.append(dict(dataset=dataset, contrast=name, candidate=candidate, baseline=baseline,
                **{f"delta_{key}_pp": (100 * (indexed[candidate][key] - indexed[baseline][key])
                    if indexed[candidate][key] is not None and indexed[baseline][key] is not None else None)
                   for key in ("PRO_exact", "P_AUROC", "P_AP", "I_AUROC")}))
        comparisons.append(dict(dataset=dataset, contrast="highres_reference_interaction", candidate="D-B-C+A", baseline="factorial",
            **{f"delta_{key}_pp": (100 * (indexed[ARMS[3]][key] - indexed[ARMS[1]][key] - indexed[ARMS[2]][key] + indexed[ARMS[0]][key])
                if all(indexed[arm][key] is not None for arm in ARMS) else None)
               for key in ("PRO_exact", "P_AUROC", "P_AP", "I_AUROC")}))
    write_csv(root / "results" / "metrics.csv", rows)
    write_csv(root / "results" / "summary.csv", means)
    write_csv(root / "results" / "comparisons.csv", comparisons)
    screen = []
    for arm in ARMS:
        value = torch.load(root / "heads" / arm / "best.pt", map_location="cpu", weights_only=False)
        last = torch.load(root / "heads" / arm / "last.pt", map_location="cpu", weights_only=False)
        screen.append(dict(arm=arm, completed_epochs=last["epoch"] + 1, selected_epoch=value["epoch"] + 1,
                           alpha=value["alpha"], source_val_loss=value["source_val_loss"], threshold=value["threshold"]))
    write_csv(root / "source_screen.csv", screen)
    atomic_json(root / "results" / "summary.json", dict(config=cfg, means=means, comparisons=comparisons, source_selection=screen))
    lines = ["# Global/local normal-reference experiment", "", cfg["reference_protocol"], "",
             "A/B use a single query; C/D receive target-category normal TRAIN support. All heads use source masks only.",
             "Source Base may have seen head-val images. Test labels select neither epochs, alpha nor thresholds.",
             "F1_best is an evaluation threshold oracle. Source thresholds are shared across target categories.",
             "Head latency excludes frozen extraction; per-query extraction audit includes ALL branches and references,",
             "and is not a separate deployment latency measurement for each arm.", "",
             "| Dataset | Arm | PRO | Pixel AUROC | Pixel AP | Image AUROC |", "|---|---|---:|---:|---:|---:|"]
    def number(value):
        return "NA" if value is None else f"{100*value:.2f}"
    lines.extend(f"| {r['dataset']} | {r['arm']} | " + " | ".join(number(r[k]) for k in ("PRO_exact", "P_AUROC", "P_AP", "I_AUROC")) + " |" for r in means)
    lines.extend(["", "Factorial gains and D-B-C+A interaction: comparisons.csv. All deltas are percentage points.",
                  "Checkpoint selected_epoch=0/alpha=0 means a frozen-readout fallback, not a learned improvement."])
    (root / "results" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("prepare", "preflight", "export", "seal", "train", "evaluate", "report"))
    p.add_argument("--work_dir", required=True)
    for key, default in dict(base_ckpt="", source="visa", eval_datasets="mvtec,btad,mpdd", source_root="", target_root="", data_roots="{}",
                             dino_repo_dir="./dinov3", dino_model_name="dinov3_vitl16", dino_weights="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                             hfa_setting="hfa3", clip_model_name="ViT-L-14-336", clip_pretrained="openai", visual_layers="5,11,17,23",
                             device="cuda:0", precision="bf16", alphas="0,0.25,0.5,1").items():
        p.add_argument("--" + key, default=default)
    for key, default in dict(seed=42, num_shards=2, shard_id=0, limit_per_category=0, image_size=512, eval_size=512,
                             feature_grid=64, feature_dim=64, tile_grid=2, reference_pool=4, reference_count=1,
                             match_topk=3, match_chunk=128, dino_bottleneck=256, epochs=15, until_epoch=0,
                             batch_size=8, micro_batch=2, workers=2, hidden=64).items():
        p.add_argument("--" + key, type=int, default=default)
    for key, default in dict(val_fraction=.2, tile_fraction=.6, high_weight=.25, match_temperature=.07,
                             spatial_weight=.15, match_radius=.35, lr=1e-4, semantic_weight=.1,
                             background_weight=1., residual_weight=.01, val_fpr=.01).items():
        p.add_argument("--" + key, type=float, default=default)
    p.add_argument("--mode", choices=ARMS, default=ARMS[0])
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.stage == "preflight":
        export(arguments, probe=True)
    else:
        globals()[arguments.stage](arguments)
