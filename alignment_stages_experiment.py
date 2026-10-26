#!/usr/bin/env python3
"""Immutable-cache, source-selected H/T/R experiments. No RL or target training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import reference_highres_experiment as legacy
from tools.alignment_cache import (
    EXTRA_LAYOUT, VISUAL_FIELDS, repair_interrupted_export, verify_execution, visual_parity, visual_view,
)
from tools.alignment_stages import (
    CORE_HEADS, HEAD_MODES, KEEP, LATE, AlignmentHead, ResidualHead, TeacherExtractor,
    batch_stitch, best_source, binary_selection_loss, bundle_prediction, bundle_training_logits, early_inputs, eligible,
    frozen_high, head_bundle, head_loss, probability_loss, selection_key, teacher_loss,
)
from tools.reference_highres import assemble_tiles, pack_view, patch_coordinates
from tools.upstream_repair import corrected_probability, constrained_loss, reference_logits
from tools.vlm_review import atomic_json, digest, file_digest, load_json

PROTOCOL = "alignment_three_stage_v2"
REPO = Path(__file__).resolve().parent
NEW_CODE = ("alignment_stages_experiment.py", "tools/alignment_stages.py", "run_exp_alignment_stages.sh", "tools/alignment_cache.py")


def execution_code():
    return {p: file_digest(REPO / p) for p in NEW_CODE}


def prepare(args):
    root, old = Path(args.work_dir).resolve(), Path(args.v1_dir).resolve()
    if root == old or old in root.parents or root in old.parents:
        raise ValueError("Use an independent WORK_DIR, never a v1 cache directory")
    original = legacy.cfg_load(old)
    sealed = load_json(old / "sealed.json")
    if sealed["fingerprint"] != digest(original) or file_digest(old / "anchors.npz") != sealed["anchors_sha"]:
        raise ValueError("Invalid v1 seal/anchors")
    if args.epochs < 1 or args.keep_weight < 0 or args.workers < 0 or args.num_shards < 1:
        raise ValueError("Invalid epochs/teacher weight/workers")
    if original["tile_grid"] != 2 or original["tile_fraction"] != .6 or original["high_weight"] != .25:
        raise ValueError("This protocol fixes the v1 2x2/.6/.25 geometry")
    alphas = sorted(set(float(x) for x in args.alphas.split(",")) | {0.})
    if not all(np.isfinite(a) and 0 <= a <= 1 for a in alphas):
        raise ValueError("Invalid alpha grid")
    panel = []
    rows = legacy.records(old)
    for key in sorted({(r["dataset"], r["category"]) for r in rows if r["partition"] == "eval"}):
        group = [r for r in rows if (r["dataset"], r["category"]) == key]
        panel.extend(r["id"] for r in sorted(group, key=lambda r: digest([42, "panel", r["id"]]))[:2])
    cfg = dict(original, protocol=PROTOCOL, v1_root=str(old), v1_config_sha=file_digest(old / "config.json"),
               v1_seal_sha=file_digest(old / "sealed.json"), original_protocol=original["protocol"],
               epochs=args.epochs, alphas=alphas, keep_weight=args.keep_weight, workers=args.workers, num_shards=args.num_shards,
               data_seed=original["seed"], projection_seed=original["seed"], reference_seed=original["seed"],
               new_code=execution_code(), extra_layout=EXTRA_LAYOUT, panel_ids=panel,
               selection="source-category macro PRO/AP Pareto against alpha=0; PRO,AP,alpha,epoch order",
               reference_alphas=[0., .05, .1, .25, .5], target_labels="evaluation only after prediction")
    path = root / "config.json"
    if path.exists():
        previous = load_json(path)
        if previous != cfg:
            repaired = verify_execution(root, previous, execution_code())
            compatible = dict(previous, new_code=cfg["new_code"], extra_layout=EXTRA_LAYOUT)
            if not repaired or compatible != cfg:
                raise ValueError("Configuration changed: use NEW WORK_DIR; old results were not overwritten")
            cfg = previous  # Keep H checkpoint fingerprints and config bytes unchanged.
    else:
        atomic_json(path, cfg)
    side=cfg["image_size"]//16
    layers=len(cfg["visual_layers"].split(","))
    teacher_dim=legacy.anchors_for(old).shape[1]
    per_query=5*(2*layers*(cfg["feature_dim"]+1)*side*side+4*cfg["image_size"]**2)
    source_count=sum(r["partition"] in ("train","val") for r in rows)
    estimate=(len(rows)*per_query+source_count*5*2*teacher_dim*side*side)/2**30
    print(f"Additional full H/T/R cache estimate={estimate:.1f} GiB; free={shutil.disk_usage(root).free/2**30:.1f} GiB (later stages may be skipped)",flush=True)
    print(f"Prepared {PROTOCOL}; queries={len(rows)}; source={original['source']}; data/projection/reference seeds fixed={original['seed']}", flush=True)
    print("H jobs read v1 features; T jobs require an additional source-only teacher/per-view export. Old cache is read-only.", flush=True)


def config(root):
    cfg = load_json(Path(root) / "config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong three-stage protocol")
    verify_execution(root, cfg, execution_code())
    old = Path(cfg["v1_root"])
    if file_digest(old / "config.json") != cfg["v1_config_sha"] or file_digest(old / "sealed.json") != cfg["v1_seal_sha"]:
        raise ValueError("v1 experiment identity changed")
    legacy.cfg_load(old)
    seal = load_json(old / "sealed.json")
    if file_digest(old / "anchors.npz") != seal["anchors_sha"]:
        raise ValueError("Frozen anchors changed")
    return cfg


def selected_rows(cfg, partition):
    rows = legacy.records(cfg["v1_root"])
    return [r for r in rows if r["partition"] in (("train", "val") if partition == "source" else (partition,))]


def checked_npz(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Cache modified: {path}")
    return np.load(io.BytesIO(data), allow_pickle=False)


def old_record(cfg, row):
    root = Path(cfg["v1_root"])
    seal = load_json(root / "sealed.json")
    audit_path = root / "features" / f"{row['id']}.json"
    if file_digest(audit_path) != seal["metadata"][row["id"]]:
        raise ValueError("Reference metadata changed")
    return seal, load_json(audit_path)


def repair_export(args):
    root = Path(args.work_dir)
    if (root / "config.json").exists():
        old = load_json(root / "config.json")
        original = legacy.cfg_load(old["v1_root"])
        seal = load_json(Path(old["v1_root"]) / "sealed.json")
        if (seal["fingerprint"] != digest(original)
                or file_digest(Path(old["v1_root"]) / "config.json") != old["v1_config_sha"]
                or file_digest(Path(old["v1_root"]) / "sealed.json") != old["v1_seal_sha"]
                or file_digest(Path(old["v1_root"]) / "anchors.npz") != seal["anchors_sha"]):
            raise ValueError("Invalid/changed v1 identity, seal or anchors")
        for row in legacy.records(old["v1_root"]):
            if row["partition"] in ("train", "val"):
                legacy.unchanged(row, masks=True)
    repair_interrupted_export(root, execution_code())


def export_extra(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    part = args.partition
    if part not in ("source", "eval") or not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid partition/shard")
    rows = selected_rows(cfg, part)[args.shard_id::cfg["num_shards"]]
    pending = []
    for row in rows:
        path = root / "extra" / f"{row['id']}.npz"
        if path.exists() and path.with_suffix(".json").exists():
            audit = load_json(path.with_suffix(".json"))
            if (audit["fingerprint"] != digest(cfg) or audit["sha"] != file_digest(path)
                    or audit.get("layout") != EXTRA_LAYOUT):
                raise ValueError("Extra cache changed; use NEW_WORK_DIR")
        else:
            if (root / f"extra_{part}_seal.json").exists():
                raise ValueError("Cannot regenerate sealed extra features")
            pending.append(row)
    if not pending:
        print(f"extra {part}/{args.shard_id}: complete", flush=True)
        return
    original = legacy.cfg_load(cfg["v1_root"])
    legacy.verify_weights(original)
    device = legacy.device_for(args)
    extractor = TeacherExtractor(original, device)
    started = time.monotonic()
    try:
        for index, row in enumerate(pending):
            legacy.unchanged(row)  # Never open a mask in feature export.
            seal, audit = old_record(cfg, row)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            tick = time.monotonic()
            extractor.include_teacher = row["partition"] in ("train", "val")
            with Image.open(row["image_path"]) as image:
                global_view, tiles, boxes = extractor.image_views(image.convert("RGB"))
            # T consumes no reference evidence. R continues reading the sealed v1
            # residual/distance/entropy, not a fresh and potentially different top-k.
            low = visual_view(global_view, extractor.project)
            packed = [visual_view(view, extractor.project) for view in tiles]
            high = assemble_tiles(packed, boxes, cfg["feature_grid"], cfg["image_size"])
            with checked_npz(Path(cfg["v1_root"]) / "features" / f"{row['id']}.npz", seal["features"][row["id"]]) as old:
                parity = visual_parity(low, high, old)
            atomic_json(root / "parity" / part / f"{row['id']}.json",
                        dict(id=row["id"], fingerprint=digest(cfg), execution_code=execution_code(), **parity))
            if not parity["passed"]:
                failed = {k: v for k, v in parity["fields"].items() if not v["passed"]}
                raise ValueError(f"Visual/Base parity failed {row['id']}: {failed}; details: {root / 'parity' / part}")
            arrays = {}
            for v, (data, view) in enumerate(zip([low] + packed, [global_view] + tiles)):
                for field, value in data.items():
                    arrays[f"v{v}_{field}"] = value.cpu().numpy().astype(np.float32 if field == "base" else np.float16)
                if "teacher" in view:
                    arrays[f"v{v}_teacher"] = view["teacher"].cpu().numpy().astype(np.float16)
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Nonfinite extra features")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.monotonic() - tick
            path = root / "extra" / f"{row['id']}.npz"
            legacy.atomic_npz(path, **arrays)
            atomic_json(path.with_suffix(".json"), dict(fingerprint=digest(cfg), sha=file_digest(path),
                        layout=EXTRA_LAYOUT, parity=parity,
                        old_sha=seal["features"][row["id"]], partition=row["partition"],
                        references=audit["references"], reference_similarity=audit["reference_similarity"],
                        boxes=[[0., 0., 1., 1.]] + boxes, query_seconds=seconds,
                        peak_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                        reference_policy="sealed v1 only; no matching in extra export", teacher=extractor.include_teacher))
            print(f"extra {part}/{args.shard_id} {index+1}/{len(pending)} seconds={seconds:.2f}", flush=True)
            if args.probe:
                break
    finally:
        extractor.close()
        print(f"extra wall={time.monotonic()-started:.1f}s (visual/teacher only; no reference encoding)", flush=True)


def seal_extra(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    files, metadata = {}, {}
    for row in selected_rows(cfg, args.partition):
        path = root / "extra" / f"{row['id']}.npz"
        audit = load_json(path.with_suffix(".json"))
        old_seal, old_audit = old_record(cfg, row)
        if (audit["fingerprint"] != digest(cfg) or audit["old_sha"] != old_seal["features"][row["id"]]
                or audit["references"] != old_audit["references"] or audit["partition"] != row["partition"]
                or audit.get("layout") != EXTRA_LAYOUT or not audit.get("parity", {}).get("passed")):
            raise ValueError("Extra cache identity/provenance changed")
        with checked_npz(path, audit["sha"]) as data:
            side, layers = cfg["image_size"] // 16, len(cfg["visual_layers"].split(","))
            for i in range(5):
                for key in VISUAL_FIELDS:
                    shape = ((layers, cfg["feature_dim"], side, side) if key == "query" else
                             (1, cfg["image_size"], cfg["image_size"]) if key == "base" else (layers, side, side))
                    value = data[f"v{i}_{key}"]
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError("Invalid extra tensor")
                if row["partition"] in ("train", "val"):
                    teacher = data[f"v{i}_teacher"]
                    if teacher.shape != (legacy.anchors_for(cfg["v1_root"]).shape[1], side, side) or not np.isfinite(teacher).all():
                        raise ValueError("Invalid teacher tensor")
                elif f"v{i}_teacher" in data:
                    raise ValueError("Target teacher must not become a training input")
        files[row["id"]], metadata[row["id"]] = audit["sha"], file_digest(path.with_suffix(".json"))
    value = dict(fingerprint=digest(cfg), layout=EXTRA_LAYOUT, files=files, metadata=metadata)
    path = root / f"extra_{args.partition}_seal.json"
    if path.exists() and load_json(path) != value:
        raise ValueError("Extra seal changed")
    atomic_json(path, value)


class FeatureData(torch.utils.data.Dataset):
    def __init__(self, root, cfg, rows, extra=False, masks=False):
        self.root, self.cfg, self.rows, self.extra, self.masks = Path(root), cfg, rows, extra, masks
        if masks and any(r["partition"] == "eval" for r in rows):
            raise ValueError("Target masks cannot enter the training/validation loader")
        self.old_seal = load_json(Path(cfg["v1_root"]) / "sealed.json")
        self.seals = {}
        if extra:
            for part in {"eval" if r["partition"] == "eval" else "source" for r in rows}:
                seal = load_json(self.root / f"extra_{part}_seal.json")
                if seal["fingerprint"] != digest(cfg) or seal.get("layout") != EXTRA_LAYOUT:
                    raise ValueError("Extra seal fingerprint changed")
                self.seals[part] = seal

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        old_root = Path(self.cfg["v1_root"])
        with checked_npz(old_root / "features" / f"{row['id']}.npz", self.old_seal["features"][row["id"]]) as data:
            arrays = {key: torch.from_numpy(data[key].astype(np.float32)) for key in data.files}
        audit_path = old_root / "features" / f"{row['id']}.json"
        if file_digest(audit_path) != self.old_seal["metadata"][row["id"]]:
            raise ValueError("Old reference audit changed")
        audit = load_json(audit_path)
        result = dict(old=arrays, similarity=float(np.mean(audit["reference_similarity"])), row=row)
        if self.extra:
            part = "eval" if row["partition"] == "eval" else "source"
            seal = self.seals[part]
            path = self.root / "extra" / f"{row['id']}.npz"
            if file_digest(path.with_suffix(".json")) != seal["metadata"][row["id"]]:
                raise ValueError("Extra reference audit changed")
            metadata = load_json(path.with_suffix(".json"))
            with checked_npz(path, seal["files"][row["id"]]) as data:
                result["views"] = [{key.split("_", 1)[1]: torch.from_numpy(data[key].astype(np.float32))
                                    for key in data.files if key.startswith(f"v{i}_")} for i in range(5)]
            result["boxes"] = torch.tensor(metadata["boxes"], dtype=torch.float32)
        if self.masks:
            result["mask"] = source_mask(row, self.cfg["image_size"])
        return result


def source_mask(row, size):
    if row["partition"] not in ("train", "val"):
        raise ValueError("Source supervision only")
    if row["mask_path"] and file_digest(row["mask_path"]) != row["mask_sha"]:
        raise ValueError("Source mask changed")
    return legacy.mask_for(row, size)


def collate_samples(values):
    return values


def batch_to(values, device):
    result = dict(old={key: torch.stack([v["old"][key] for v in values]).to(device) for key in values[0]["old"]},
                  similarity=torch.tensor([v["similarity"] for v in values], device=device))
    if "views" in values[0]:
        result["views"] = [{key: torch.stack([v["views"][i][key] for v in values]).to(device)
                            for key in values[0]["views"][i]} for i in range(5)]
        result["boxes"] = torch.stack([v["boxes"] for v in values]).to(device)
    if "mask" in values[0]:
        result["mask"] = torch.stack([v["mask"] for v in values]).to(device)
    return result


def run_dir(root, seed, mode, parent=None):
    name = "reference_" + digest(parent)[:16] if mode in ("R1", "R2", "R3") else "heads"
    return Path(root) / "runs" / f"seed{seed}" / name / mode


def make_model(cfg, mode, device):
    layers = len(cfg["visual_layers"].split(","))
    if mode in HEAD_MODES:
        return AlignmentHead(mode, layers, cfg["feature_dim"], legacy.anchors_for(cfg["v1_root"]), cfg["hidden"]).to(device)
    return ResidualHead(mode, layers, cfg["feature_dim"], cfg["hidden"]).to(device)


def load_checkpoint(cfg, path, device):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["identity"]["fingerprint"] != digest(cfg):
        raise ValueError("Head fingerprint changed")
    extra_sha=value["identity"].get("extra_sha")
    if extra_sha is not None and file_digest(Path(path).resolve().parents[4]/"extra_source_seal.json")!=extra_sha:
        raise ValueError("Training teacher/per-view seal changed")
    model = make_model(cfg, value["identity"]["mode"], device)
    model.load_state_dict(value["state_dict"], strict=True)
    return model.eval(), value


class FrozenFront:
    def __init__(self, cfg, parent, device):
        self.cfg, self.parent, self.model, self.metadata = cfg, parent, None, None
        if parent and parent["mode"] != "H0":
            path = Path(parent["checkpoint"])
            if file_digest(path) != parent["sha"]:
                raise ValueError("Locked frontend changed; select again into a new reference run")
            self.model, self.metadata = load_checkpoint(cfg, path, device)
            if self.metadata["identity"]["mode"]!=parent["mode"] or self.metadata["point"]!=parent["point"]:
                raise ValueError("Locked frontend metadata changed")
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    @property
    def extra(self):
        return self.parent is not None and self.parent["mode"] in LATE

    @torch.no_grad()
    def __call__(self, batch):
        if self.model is None:
            return frozen_high(batch, self.cfg)
        bundle = head_bundle(self.model, self.parent["mode"], batch, self.cfg)
        return bundle_prediction(bundle, self.metadata["point"]["alpha"], self.cfg)


def prediction_bundle(model, mode, batch, cfg, front=None):
    if mode in HEAD_MODES:
        return head_bundle(model, mode, batch, cfg)
    base = front(batch)
    if mode == "R1":
        inputs = early_inputs(batch, cfg, reference=True)
        distance = inputs["distance"].mean(1, keepdim=True)
        distance = F.interpolate(distance, base.shape[-2:], mode="bilinear", align_corners=False)
        delta = (distance - model["median"]) / model["iqr"]
        output = dict(delta=delta)
    else:
        output = model(early_inputs(batch, cfg, reference=True))
    return dict(outputs=[output], views=[], base=base, boxes=None, mode=mode)


def source_metrics(masks, maps):
    from sklearn.metrics import average_precision_score
    from tools.vlm_next_diagnostics import exact_pro
    return dict(PRO_exact=exact_pro(masks, maps), P_AP=float(average_precision_score(masks.ravel(), maps.ravel())) if masks.any() else None)


@torch.no_grad()
def validate(root, cfg, model, mode, device, front=None):
    rows = selected_rows(cfg, "val")
    if any(r["partition"] != "val" for r in rows):
        raise ValueError("Source validation only")
    extra = mode in ("T10", "T01", "T11") or front is not None and front.extra
    alphas = cfg["reference_alphas"] if mode == "R1" else cfg["alphas"]
    totals = {a: dict(categories=[], loss=0., backgrounds=[]) for a in alphas}
    diagnostics = defaultdict(list)
    count = 0
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        dataset = FeatureData(root, cfg, group, extra=extra, masks=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["micro_batch"], num_workers=cfg["workers"], collate_fn=collate_samples)
        predictions, masks = {a: [] for a in alphas}, []
        for samples in loader:
            batch = batch_to(samples, device)
            with legacy.autocast(device, cfg):
                bundle = prediction_bundle(model, mode, batch, cfg, front)
            mask = batch["mask"]
            for alpha in alphas:
                probability = bundle_prediction(bundle, alpha, cfg)
                totals[alpha]["loss"] += float(binary_selection_loss(probability, mask, bundle_training_logits(bundle,alpha,cfg)).sum())
                predictions[alpha].extend(probability[:, 0].float().cpu().numpy())
                flat, bg = probability.flatten(1), (mask == 0).flatten(1)
                indices = torch.arange(0, flat.shape[1], max(1, flat.shape[1] // 8192), device=device)
                totals[alpha]["backgrounds"].append(flat[:, indices][bg[:, indices]].float().cpu().numpy())
            delta = reference_logits(bundle_prediction(bundle, 1., cfg)) - reference_logits(bundle["base"])
            for label, support in (("normal", mask == 0), ("defect", mask > 0)):
                values = delta[support].float().cpu().numpy()
                diagnostics["delta_" + label].extend(values[::max(1, len(values) // 4096)].tolist())
            for output in bundle["outputs"]:
                for name in ("hidden_term", "semantic_term", "gate"):
                    if name in output:
                        diagnostics[name].append(float(output[name].float().abs().mean()))
            masks.extend(mask[:, 0].cpu().numpy().astype(bool))
            count += len(samples)
        masks = np.asarray(masks)
        metric_cache = {}
        for alpha in alphas:
            maps = np.asarray(predictions[alpha])
            key = hashlib.sha256(maps.tobytes()).hexdigest()
            if key not in metric_cache:
                metric_cache[key] = source_metrics(masks, maps)
            totals[alpha]["categories"].append(dict(category=category, **metric_cache[key]))
    if not count:
        raise ValueError("Empty source validation")
    result = []
    for alpha in alphas:
        data = totals[alpha]
        point = dict(alpha=alpha, loss=data["loss"] / count, categories=data["categories"])
        for name in ("PRO_exact", "P_AP"):
            valid = [c[name] for c in data["categories"] if c[name] is not None]
            if len(valid) != len(data["categories"]):
                raise ValueError("Source val needs positive defects/background in every category")
            point[name] = float(np.mean(valid))
            if not np.isfinite(point[name]):
                raise ValueError("Nonfinite source selection metric")
        background = np.concatenate(data["backgrounds"])
        if not len(background):
            raise ValueError("No source calibration background")
        point["threshold"] = float(np.quantile(background, 1 - cfg["val_fpr"], method="higher"))
        result.append(point)
    diag = {key: dict(mean=float(np.mean(values)), quantiles=np.quantile(values, [0, .05, .5, .95, 1]).tolist(),
                      sampled_values=len(values), positive_fraction=float((np.asarray(values) > 0).mean()))
            for key, values in diagnostics.items() if values}
    return result, diag


def point_checkpoint(model, identity, point, baseline):
    return dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, identity=identity,
                point=point, baseline=baseline, fallback=point["alpha"] == 0.)


def train(args):
    root, cfg, mode = Path(args.work_dir), config(args.work_dir), args.mode
    if mode not in HEAD_MODES + ("R2", "R3"):
        raise ValueError("Train H/T heads or R2/R3")
    until = args.until_epoch or cfg["epochs"]
    if not 1 <= until <= cfg["epochs"]:
        raise ValueError("Keep the originally planned horizon; until must be within it")
    seed, device = args.train_seed, legacy.device_for(args)
    parent = load_json(root / "decisions" / f"seed{seed}_front.json") if mode.startswith("R") else None
    front = FrozenFront(cfg, parent, device) if parent else None
    extra = mode in ("T10", "T01", "T11") or front is not None and front.extra
    extra_sha = file_digest(root / "extra_source_seal.json") if extra else None
    identity = dict(fingerprint=digest(cfg), mode=mode, train_seed=seed, parent=parent, extra_sha=extra_sha)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_model(cfg, mode, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=.01)
    dataset = FeatureData(root, cfg, selected_rows(cfg, "train"), extra=extra, masks=True)
    steps = math.ceil(len(dataset) / cfg["batch_size"]) * cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .05 + .95 * (1 + math.cos(math.pi * step / max(1, steps))) / 2)
    out = run_dir(root, seed, mode, parent)
    start, history = 0, []
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Training identity/front/cache changed")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, history, baseline = saved["epoch"], saved["history"], saved["baseline"]
        for selector in ("primary", "bce"):
            if not (out / f"best_{selector}.pt").exists():
                raise ValueError("Missing selected checkpoint")
    else:
        model.eval()
        values, diagnostics = validate(root, cfg, model, mode, device, front)
        baseline = dict(next(p for p in values if p["alpha"] == 0), epoch=0)
        value = point_checkpoint(model, identity, baseline, baseline)
        for selector in ("primary", "bce"):
            legacy.save_torch(out / f"best_{selector}.pt", value)
        legacy.save_torch(out / "epoch_0000.pt", value)
        atomic_json(out / "validation_0000.json", dict(values=values, diagnostics=diagnostics))
    flop_count = [0]
    def count_conv(module, _inputs, output):
        if model.training:
            flop_count[0] += 2 * output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    handles = [module.register_forward_hook(count_conv) for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    try:
        for epoch in range(start, until):
            tick, flop_count[0] = time.monotonic(), 0
            model.train()
            generator = torch.Generator().manual_seed(cfg["data_seed"] + epoch)
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
                                                num_workers=cfg["workers"], collate_fn=collate_samples)
            loss_sum, count, gradient_sums = 0., 0, defaultdict(float)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for samples in loader:
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(samples), cfg["micro_batch"]):
                    batch = batch_to(samples[offset:offset + cfg["micro_batch"]], device)
                    with legacy.autocast(device, cfg):
                        bundle = prediction_bundle(model, mode, batch, cfg, front)
                        if mode in HEAD_MODES:
                            losses = head_loss(bundle, batch, batch["mask"], cfg)
                        else:
                            losses,_ = constrained_loss(bundle["base"],bundle["outputs"][0]["delta"],batch["mask"],
                                                        cfg["background_weight"],cfg["residual_weight"])
                        loss = losses.sum() / len(samples)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    loss_sum += float(losses.detach().sum())
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and (name.startswith("to_text") or name.startswith("delta") or name.startswith("gate")):
                        gradient_sums[name] += float(parameter.grad.float().norm())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                count += len(samples)
            train_seconds = time.monotonic() - tick
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
            model.eval()
            val_tick = time.monotonic()
            values, diagnostics = validate(root, cfg, model, mode, device, front)
            candidates = [dict(v, epoch=epoch + 1) for v in values]
            primary = best_source(candidates, baseline)
            bce = min(candidates + [baseline], key=lambda p: (p["loss"], p["alpha"], p["epoch"]))
            for selector, point in (("primary", primary), ("bce", bce)):
                path = out / f"best_{selector}.pt"
                previous = torch.load(path, map_location="cpu", weights_only=False)["point"]
                improves = ((eligible(point, baseline) and (not eligible(previous, baseline) or selection_key(point) < selection_key(previous)))
                            if selector == "primary" else (point["loss"], point["alpha"], point["epoch"]) < (previous["loss"], previous["alpha"], previous["epoch"]))
                if improves:
                    legacy.save_torch(path, point_checkpoint(model, identity, point, baseline))
            completed = point_checkpoint(model, identity, dict(min(candidates, key=lambda p: (p["loss"], p["alpha"])), epoch=epoch+1), baseline)
            legacy.save_torch(out / f"epoch_{epoch+1:04d}.pt", completed)
            atomic_json(out / f"validation_{epoch+1:04d}.json", dict(values=candidates, diagnostics=diagnostics))
            selected = {s: torch.load(out / f"best_{s}.pt", map_location="cpu", weights_only=False)["point"] for s in ("primary", "bce")}
            history.append(dict(epoch=epoch+1, loss=loss_sum / count, train_seconds=train_seconds,
                                validation_seconds=time.monotonic()-val_tick, queries=count,
                                forward_conv_flops_per_query=flop_count[0]/count,
                                flop_scope="head Conv2d forward only; excludes backbone/backward/normalization",
                                peak_training_gib=peak, parameters=sum(p.numel() for p in model.parameters()),
                                gradient_norm_sums=dict(gradient_sums), selected=selected))
            atomic_json(out / "history.json", history)
            legacy.save_torch(out / "last.pt", dict(state_dict=model.state_dict(), identity=identity,
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch+1,
                              baseline=baseline, history=history))
            print(f"{mode} seed={seed} {epoch+1}/{cfg['epochs']} train={loss_sum/count:.5f} selected={selected['primary']['epoch']}/alpha={selected['primary']['alpha']}", flush=True)
    finally:
        for handle in handles:
            handle.remove()


def stage_gate(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    states = []
    for mode in ("H1", "H2", "H3"):
        path = run_dir(root, args.train_seed, mode)
        last = torch.load(path / "last.pt", map_location="cpu", weights_only=False)
        if last["identity"]["fingerprint"] != digest(cfg) or last["epoch"] != args.until_epoch:
            raise ValueError("Stage one is incomplete/stale")
        value = torch.load(path / "best_primary.pt", map_location="cpu", weights_only=False)
        states.append(dict(mode=mode, eligible=eligible(value["point"], value["baseline"]), completed=last["epoch"]))
    allowed = any(s["eligible"] for s in states) or args.continue_on_fallback
    atomic_json(root / "decisions" / f"seed{args.train_seed}_gate.json", dict(allowed=allowed, source_only=True,
                explicitly_forced=bool(args.continue_on_fallback), states=states, until=args.until_epoch))
    print(f"Source-only stage-two gate: {allowed}", flush=True)
    return allowed


def select_front(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    modes = ["H3"]
    gate = load_json(root / "decisions" / f"seed{args.train_seed}_gate.json")
    if args.front_level >= 2 and gate["allowed"]:
        modes.extend(("T10", "T01", "T11"))
    choices = []
    for mode in modes:
        out = run_dir(root, args.train_seed, mode)
        last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        value = torch.load(out / "best_primary.pt", map_location="cpu", weights_only=False)
        if last["epoch"] != args.until_epoch or value["identity"]["fingerprint"] != digest(cfg):
            raise ValueError("Frontend selection requires all matched jobs to finish")
        if eligible(value["point"], value["baseline"]):
            choices.append((mode, out / "best_primary.pt", value))
    if choices:
        mode, path, value = min(choices, key=lambda item: (selection_key(item[2]["point"]), item[0]))
        parent = dict(mode=mode, checkpoint=str(path.resolve()), sha=file_digest(path), point=value["point"],
                      selector="primary", seed=args.train_seed, fingerprint=digest(cfg))
    else:
        parent = dict(mode="H0", checkpoint=None, sha=None, point=None, selector="primary", seed=args.train_seed,
                      fingerprint=digest(cfg))
    atomic_json(root / "decisions" / f"seed{args.train_seed}_front.json", parent)
    print(f"Locked source-selected frontend: {parent['mode']}; new parent hashes create NEW reference runs, never stale resumes", flush=True)


def fit_distance(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    rows = selected_rows(cfg, "train")
    for row in rows:
        if row["mask_path"] and file_digest(row["mask_path"]) != row["mask_sha"]:
            raise ValueError("Source calibration mask changed")
    existing = root / "distance_stats.json"
    if existing.exists():
        if load_json(existing)["fingerprint"] != digest(cfg):
            raise ValueError("Stale distance calibration")
        return
    data = FeatureData(root, cfg, rows, masks=True)
    samples = []
    for values in torch.utils.data.DataLoader(data, batch_size=cfg["micro_batch"], num_workers=cfg["workers"], collate_fn=collate_samples):
        batch = batch_to(values, torch.device("cpu"))
        distance = early_inputs(batch, cfg, reference=True)["distance"].mean(1, keepdim=True)
        distance = F.interpolate(distance, batch["mask"].shape[-2:], mode="bilinear", align_corners=False)
        for d, mask in zip(distance, batch["mask"]):
            normal = d[mask == 0].numpy()
            samples.append(normal[::max(1, len(normal)//8192)])
    values = np.concatenate(samples)
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    if not np.isfinite(q75 - q25) or q75 - q25 <= 1e-8:
        raise ValueError("Source normal distance IQR degenerate; R1 cannot be calibrated")
    stats = dict(fingerprint=digest(cfg), median=float(median), iqr=float(q75-q25), sample_count=len(values),
                 origin="source TRAIN normal pixels; deterministic subsampling; no target statistics")
    path = root / "distance_stats.json"
    if path.exists() and load_json(path) != stats:
        raise ValueError("Distance calibration changed")
    atomic_json(path, stats)


def calibrate_r1(args):
    root, cfg, device = Path(args.work_dir), config(args.work_dir), legacy.device_for(args)
    parent = load_json(root / "decisions" / f"seed{args.train_seed}_front.json")
    front = FrozenFront(cfg, parent, device)
    stats = load_json(root / "distance_stats.json")
    if stats["fingerprint"] != digest(cfg):
        raise ValueError("Wrong distance calibration")
    values, diagnostics = validate(root, cfg, stats, "R1", device, front)
    baseline = dict(next(p for p in values if p["alpha"] == 0), epoch=0)
    candidates = [dict(p, epoch=0) for p in values]
    primary = best_source(candidates, baseline)
    bce = min(candidates, key=lambda p: (p["loss"], p["alpha"]))
    out = run_dir(root, args.train_seed, "R1", parent)
    atomic_json(out / "selection.json", dict(parent=parent, stats=stats, primary=primary, bce=bce,
                baseline=baseline, diagnostics=diagnostics, candidates=candidates, fingerprint=digest(cfg)))
    print(f"R1 source-selected lambda={primary['alpha']}", flush=True)


def reference_gate(args):
    root = Path(args.work_dir)
    config(root)
    parent = load_json(root / "decisions" / f"seed{args.train_seed}_front.json")
    allowed = parent["mode"] != "H0" or args.continue_on_fallback
    atomic_json(root / "decisions" / f"seed{args.train_seed}_reference_gate.json", dict(allowed=allowed,
                parent=parent, source_only=True, explicitly_forced=bool(args.continue_on_fallback)))
    print(f"Source-only reference-training gate: {allowed}", flush=True)
    return allowed


def evaluation_specs(args, cfg, device):
    root, seed, specs = Path(args.work_dir), args.train_seed, []
    gate = load_json(root / "decisions" / f"seed{seed}_gate.json")
    parent = load_json(root / "decisions" / f"seed{seed}_front.json")
    front = FrozenFront(cfg, parent, device)
    modes = ["H1", "H2", "H3"]
    if (run_dir(root,seed,"H1_aux0")/"last.pt").exists():
        modes.append("H1_aux0")
    if args.front_level >= 2 and gate["allowed"]:
        modes.extend(("T10", "T01", "T11"))
    for mode in modes:
        out = run_dir(root, seed, mode)
        last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if last["epoch"] != args.until_epoch:
            raise ValueError("Matched evaluation requires all completed horizons equal UNTIL_EPOCH")
        for selector in ("primary", "bce"):
            path = out / f"best_{selector}.pt"
            model, value = load_checkpoint(cfg, path, device)
            if value["identity"]["mode"]!=mode:
                raise ValueError("Head mode identity changed")
            specs.append(dict(mode=mode, actual_mode=mode, selector=selector, model=model, point=value["point"],
                              sha=file_digest(path), path=str(path), parent=None, front=None, fallback=value["fallback"]))
            if mode == "H3" and args.front_level >= 2:
                specs.append(dict(specs[-1], mode="T00", alias="H3"))
    baseline = torch.load(run_dir(root, seed, "H3") / "best_primary.pt", map_location="cpu", weights_only=False)["baseline"]
    specs.insert(0, dict(mode="H0", actual_mode="H0", selector="primary", point=baseline, model=None,
                         sha=None, path=None, parent=None, front=None, fallback=True))
    if args.front_level >= 3:
        r1path = run_dir(root, seed, "R1", parent) / "selection.json"
        r1 = load_json(r1path)
        if r1["fingerprint"] != digest(cfg) or r1["parent"] != parent:
            raise ValueError("R1 frontend/calibration changed")
        specs.append(dict(mode="R0", actual_mode="R0", selector="primary", point=r1["baseline"], model=None,
                          sha=digest(parent), path=None, parent=parent, front=front, fallback=True))
        for selector in ("primary", "bce"):
            specs.append(dict(mode="R1", actual_mode="R1", selector=selector, point=r1[selector], model=r1["stats"],
                              sha=file_digest(r1path), path=str(r1path), parent=parent, front=front,
                              fallback=r1[selector]["alpha"] == 0))
        rg = load_json(root / "decisions" / f"seed{seed}_reference_gate.json")
        if rg["parent"] != parent:
            raise ValueError("Reference gate is stale")
        if rg["allowed"]:
            for mode in ("R2", "R3"):
                out = run_dir(root, seed, mode, parent)
                last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
                if last["epoch"] != args.until_epoch:
                    raise ValueError("Reference runs must finish their matched horizon")
                for selector in ("primary", "bce"):
                    path = out / f"best_{selector}.pt"
                    model, value = load_checkpoint(cfg, path, device)
                    if value["identity"]["mode"]!=mode:
                        raise ValueError("Reference mode identity changed")
                    if value["identity"]["parent"] != parent:
                        raise ValueError("Reference head is bound to another frontend")
                    specs.append(dict(mode=mode, actual_mode=mode, selector=selector, point=value["point"],
                                      model=model, sha=file_digest(path), path=str(path), parent=parent,
                                      front=front, fallback=value["fallback"]))
    return specs


def spec_prediction(spec, batch, cfg):
    mode = spec["actual_mode"]
    if mode == "H0":
        return frozen_high(batch, cfg)
    if mode == "R0":
        return spec["front"](batch)
    bundle = prediction_bundle(spec["model"], mode, batch, cfg, spec["front"])
    return bundle_prediction(bundle, spec["point"]["alpha"], cfg)


def save_panel(root, cfg, row, arrays, maps, mask, seed, until):
    """Scientific audit image. Labels are accessed only after predictions are locked."""
    from PIL import ImageDraw
    size = 256
    def heat(value):
        value = np.clip(np.asarray(value, np.float32), 0, 1)
        rgb = np.stack([value, 1 - np.abs(2 * value - 1), 1-value], -1)
        return Image.fromarray((rgb*255).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)
    with Image.open(row["image_path"]) as im:
        query = im.convert("RGB").resize((size, size))
    _, audit = old_record(cfg, row)
    refs = {r["id"]: r for r in load_json(Path(cfg["v1_root"]) / "references.json")}
    with Image.open(refs[audit["references"][0]]["image_path"]) as im:
        reference = im.convert("RGB").resize((size, size))
    panels = [("QUERY (display only)", query), ("NORMAL TRAIN reference", reference), ("GT (evaluation only)", heat(mask))]
    for key in ("H0/primary", "H1/primary", "H2/primary", "H3/primary", "T11/primary", "R0/primary", "R1/primary", "R2/primary", "R3/primary"):
        if key in maps:
            panels.append((key + " probability [0,1]", heat(maps[key])))
    uncertainty = .75 * arrays["global_uncertainty"].mean(0) + .25 * F.interpolate(
        torch.from_numpy(arrays["high_uncertainty"].mean(0))[None, None], arrays["global_uncertainty"].shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    panels.append(("Match entropy [0,1]", heat(uncertainty)))
    if (root/"distance_stats.json").exists():
        stats=load_json(root/"distance_stats.json")
        d=F.interpolate(torch.from_numpy(arrays["global_distance"].mean(0))[None,None],
                        arrays["high_distance"].shape[-2:],mode="bilinear",align_corners=False)[0,0].numpy()
        d=.75*d+.25*arrays["high_distance"].mean(0)
        panels.append(("Distance (source scale)",heat(.5+(d-stats["median"])/(4*stats["iqr"]))))
    columns = 3
    canvas = Image.new("RGB", (columns*size, math.ceil(len(panels)/columns)*(size+25)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x, y = (index % columns)*size, (index//columns)*(size+25)
        draw.text((x+4, y+4), label, fill="black")
        canvas.paste(panel, (x, y+25))
    path = root / "panels" / f"seed{seed}" / f"through{until}" / f"{row['dataset']}_{row['category']}_{row['id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluation_signature(root,cfg,specs):
    extra=any(s["actual_mode"] in LATE or s["front"] is not None and s["front"].extra for s in specs)
    return digest(dict(config=digest(cfg),specs=[{k:v for k,v in s.items() if k not in ("model","front")} for s in specs],
                       extra_sha=file_digest(Path(root)/"extra_eval_seal.json") if extra else None))


def evaluate(args):
    from sklearn.metrics import average_precision_score
    from tools.vlm_next_diagnostics import map_metrics
    root, cfg, device = Path(args.work_dir), config(args.work_dir), legacy.device_for(args)
    specs = evaluation_specs(args, cfg, device)
    extra = any(s["actual_mode"] in LATE or s["front"] is not None and s["front"].extra for s in specs)
    signature = evaluation_signature(root,cfg,specs)
    groups = defaultdict(list)
    for row in selected_rows(cfg, "eval"):
        groups[(row["dataset"], row["category"])].append(row)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid evaluation shard")
    for (dataset, category), rows in sorted(groups.items())[args.shard_id::cfg["num_shards"]]:
        path = root / "results" / f"seed{args.train_seed}" / f"through{args.until_epoch}" / f"{dataset}_{category}.json"
        source = FeatureData(root, cfg, rows, extra=extra, masks=False)
        # Audit the immutable cache even if metrics from an earlier invocation exist.
        if path.exists() and load_json(path).get("signature") == signature:
            for i, row in enumerate(rows):
                source[i]
                legacy.unchanged(row, masks=True)
            print(f"eval cached {dataset}/{category}", flush=True)
            continue
        predictions, masks, timings = defaultdict(list), [], defaultdict(float)
        for index, row in enumerate(rows):
            legacy.unchanged(row)  # No target mask is opened here.
            sample = source[index]
            batch = batch_to([sample], device)
            locked = {}
            with torch.inference_mode(), legacy.autocast(device, cfg):
                for spec in specs:
                    key = spec["mode"] + "/" + spec["selector"]
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tick = time.monotonic()
                    probability = spec_prediction(spec, batch, cfg)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings[key] += time.monotonic()-tick
                    probability = F.interpolate(probability, (cfg["eval_size"],)*2, mode="bilinear", align_corners=False)
                    locked[key] = probability[0, 0].float().cpu().numpy()
            # The target mask enters only AFTER all selector/mode predictions for this image.
            if row["mask_path"] and file_digest(row["mask_path"]) != row["mask_sha"]:
                raise ValueError("Target mask changed")
            mask = legacy.mask_for(row, cfg["eval_size"])[0].numpy().astype(bool)
            masks.append(mask)
            for key, values in locked.items():
                predictions[key].append(values)
            if row["id"] in cfg["panel_ids"]:
                save_panel(root, cfg, row, {k:v.numpy() for k,v in sample["old"].items()}, locked, mask, args.train_seed, args.until_epoch)
        masks, baseline = np.asarray(masks), np.asarray(predictions["H0/primary"])
        records = []
        for spec in specs:
            key = spec["mode"] + "/" + spec["selector"]
            maps = np.asarray(predictions[key])
            threshold = spec["point"]["threshold"]
            reference = spec["mode"].startswith("R")
            oldmaps = np.asarray(predictions["R0/primary"]) if reference else baseline
            baseline_spec = next(s for s in specs if s["mode"] == ("R0" if reference else "H0") and s["selector"] == "primary")
            scores = map_metrics(masks, maps)
            scores["P_AP"] = float(average_precision_score(masks.ravel(), maps.ravel()))
            bg = ~masks
            oracle = float(np.quantile(maps[bg], .99, method="higher"))
            fixed = legacy.small_metrics(masks, maps, maps, oracle, oracle)
            records.append(dict(dataset=dataset, category=category, mode=spec["mode"], selector=spec["selector"],
                seed=args.train_seed, completed_epochs=args.until_epoch, samples=len(rows), **scores,
                **legacy.small_metrics(masks, maps, oldmaps, threshold, baseline_spec["point"]["threshold"]),
                small_hits_target_fpr1_EVAL_ONLY=fixed["small_hits_source_threshold"],
                target_fpr1_threshold_EVAL_ONLY=oracle, source_threshold=threshold,
                background_FPR_source_threshold=float((maps[bg] > threshold).mean()),
                pixel_recall_source_threshold=float((maps[masks] > threshold).mean()),
                changed_fraction=float((np.abs(maps-oldmaps)>1e-6).mean()),
                head_seconds_per_query=timings[key]/len(rows), latency_scope="cached evidence + head only; NOT end-to-end",
                selected_epoch=spec["point"]["epoch"], alpha=spec["point"]["alpha"], fallback=spec["fallback"],
                normal_reference_count=cfg["reference_count"] if reference and spec["mode"] != "R0" else 0,
                parent_mode=spec["parent"]["mode"] if spec["parent"] else None,
                head_sha=spec["sha"], run_id=digest([digest(cfg), args.train_seed, spec["mode"], spec["sha"]])[:20]))
        atomic_json(path, dict(signature=signature, fingerprint=digest(cfg), rows=records))
        print(f"evaluated seed={args.train_seed} {dataset}/{category} modes={len(specs)}", flush=True)


def write_csv(path, rows):
    if rows:
        legacy.write_csv(Path(path), rows)


@torch.inference_mode()
def source_audit(args):
    from sklearn.metrics import average_precision_score, roc_auc_score
    root,cfg,device=Path(args.work_dir),config(args.work_dir),legacy.device_for(args)
    specs=evaluation_specs(args,cfg,device)
    has_teacher=(root/"extra_source_seal.json").exists()
    rows=selected_rows(cfg,"val")
    for spec in specs:
        mode=spec["actual_mode"]
        if spec["selector"]!="primary" or mode not in HEAD_MODES or spec["mode"]=="T00":
            continue
        data=FeatureData(root,cfg,rows,extra=has_teacher or mode in LATE,masks=True)
        categories=defaultdict(lambda:dict(margin=[],mask=[]))
        drift=[]
        for samples in torch.utils.data.DataLoader(data,batch_size=cfg["micro_batch"],num_workers=cfg["workers"],collate_fn=collate_samples):
            batch=batch_to(samples,device)
            with legacy.autocast(device,cfg):
                bundle=head_bundle(spec["model"],mode,batch,cfg)
            if "semantic_logits" not in bundle["outputs"][0]:
                continue
            logits=[out["semantic_logits"] for out in bundle["outputs"]]
            margin=logits[0]
            if mode in LATE:
                high=batch_stitch(logits[1:],batch["boxes"][:,1:].cpu().tolist(),cfg["image_size"])
                margin=(1-cfg["high_weight"])*margin+cfg["high_weight"]*high
            for i,sample in enumerate(samples):
                item=categories[sample["row"]["category"]]
                item["margin"].append(margin[i,0].float().cpu().numpy())
                item["mask"].append(batch["mask"][i,0].cpu().numpy().astype(bool))
            if has_teacher:
                teacher_bundle=dict(bundle,mode="T11" if mode in LATE else "T10")
                drift.extend(teacher_loss(teacher_bundle,batch,batch["mask"],cfg).float().cpu().tolist())
        metrics=[]
        for category,values in categories.items():
            masks,margins=np.asarray(values["mask"]),np.asarray(values["margin"])
            metrics.append(dict(category=category,margin_P_AP=float(average_precision_score(masks.ravel(),margins.ravel())),
                                margin_P_AUROC=float(roc_auc_score(masks.ravel(),margins.ravel()))))
        atomic_json(root/"diagnostics"/f"seed{args.train_seed}"/f"{mode}_through{args.until_epoch}.json",
            dict(fingerprint=digest(cfg),checkpoint_sha=spec["sha"],point=spec["point"],fallback=spec["fallback"],
                 margin_metrics=metrics,teacher_cosine_error=float(np.mean(drift)) if drift else None,
                 teacher_scope="source normal cells only; null if teacher unavailable or text branch removed",
                 delta_parameters={name:p.detach().float().cpu().tolist() for name,p in spec["model"].named_parameters() if name.startswith("delta")}))


def online_features(extractor,cfg,row,bank,with_reference):
    """Deployment-like features: no NPZ I/O and no teacher or GT access."""
    with Image.open(row["image_path"]) as im:
        global_view,tiles,boxes=extractor.image_views(im.convert("RGB"))
    side=cfg["image_size"]//16
    device=extractor.device
    xy=patch_coordinates(side,device=device)
    descriptor=F.normalize(global_view["raw"].mean((0,1)),dim=0).cpu()
    selected=sorted(bank,key=lambda r:(-float(descriptor@r["descriptor"]),r["id"]))[:cfg["reference_count"]] if with_reference else []
    def basic(view):
        q=(view["raw"]@extractor.project).permute(0,2,1).reshape(len(extractor.layers),cfg["feature_dim"],side,side)
        return dict(query=q,residual=torch.zeros_like(q),distance=torch.zeros_like(view["prior"]),
                    uncertainty=torch.zeros_like(view["prior"]),prior=view["prior"],base=view["base"])
    if with_reference:
        global_ref=torch.cat([r["global"]["raw"].to(device) for r in selected],1)
        low=pack_view(global_view,global_ref,xy,xy.repeat(len(selected),1),extractor.project,cfg)
        refs,positions=[],[]
        for r in selected:
            for view,box in zip(r["tiles"],r["boxes"]):
                refs.append(view["raw"].to(device)); positions.append(patch_coordinates(side,box,device))
        ref,positions=torch.cat(refs,1),torch.cat(positions)
        packed=[pack_view(v,ref,patch_coordinates(side,box,device),positions,extractor.project,cfg) for v,box in zip(tiles,boxes)]
    else:
        low,packed=basic(global_view),[basic(v) for v in tiles]
    high=assemble_tiles(packed,boxes,cfg["feature_grid"],cfg["image_size"])
    def quantize(view):
        return {k:(v.float() if k=="base" else v.half().float()) for k,v in view.items()}
    old={prefix+"_"+k:v for prefix,view in (("global",low),("high",high)) for k,v in quantize(view).items()}
    return dict(old=old,views=[quantize(v) for v in [low]+packed],boxes=torch.tensor([[0.,0.,1.,1.]]+boxes),
                similarity=float(np.mean([float(descriptor@r["descriptor"]) for r in selected])) if selected else 0.)


@torch.inference_mode()
def benchmark(args):
    root,cfg,device=Path(args.work_dir),config(args.work_dir),legacy.device_for(args)
    specs=[s for s in evaluation_specs(args,cfg,device) if s["selector"]=="primary" and s["mode"]!="T00"]
    if args.benchmark_mode!="all":
        specs=[s for s in specs if s["mode"]==args.benchmark_mode]
    if not specs or args.benchmark_images<1:
        raise ValueError("No requested benchmark mode/images")
    original=legacy.cfg_load(cfg["v1_root"]); legacy.verify_weights(original)
    extractor=TeacherExtractor(original,device); extractor.include_teacher=False
    rows=selected_rows(cfg,"eval")
    # One deterministic image per category before taking a second image anywhere.
    groups=defaultdict(list)
    for row in rows: groups[(row["dataset"],row["category"])].append(row)
    rows=[min(group,key=lambda r:digest([42,"benchmark",r["id"]])) for _,group in sorted(groups.items())][:args.benchmark_images]
    references=load_json(Path(cfg["v1_root"])/"references.json")
    banks={}; setup_seconds=0.; result=[]
    try:
        if any(s["mode"] in ("R1","R2","R3") for s in specs):
            for key in {(r["dataset"],r["category"]) for r in rows}:
                tick=time.monotonic(); banks[key]=[]
                for ref in references:
                    if (ref["dataset"],ref["category"])!=key: continue
                    legacy.unchanged(ref)
                    with Image.open(ref["image_path"]) as im: g,tiles,boxes=extractor.image_views(im.convert("RGB"))
                    banks[key].append(dict(id=ref["id"],tiles=[legacy.cpu_view(v) for v in tiles],boxes=boxes,
                        descriptor=F.normalize(g["raw"].mean((0,1)),dim=0).cpu(),**{"global":legacy.cpu_view(g)}))
                if device.type=="cuda": torch.cuda.synchronize(device)
                setup_seconds+=time.monotonic()-tick
        for spec in specs:
            reference=spec["mode"] in ("R1","R2","R3")
            times=[]; peaks=[]
            for iteration,row in enumerate([rows[0]]+rows):
                legacy.unchanged(row)
                if device.type=="cuda":
                    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                tick=time.monotonic()
                sample=online_features(extractor,cfg,row,banks.get((row["dataset"],row["category"]),[]),reference)
                batch=batch_to([sample],device)
                with legacy.autocast(device,cfg): predicted=spec_prediction(spec,batch,cfg)
                if not bool(torch.isfinite(predicted).all()): raise ValueError("Nonfinite online prediction")
                if device.type=="cuda": torch.cuda.synchronize(device)
                if iteration:
                    times.append(time.monotonic()-tick)
                    if device.type=="cuda": peaks.append(torch.cuda.max_memory_allocated(device)/2**30)
            result.append(dict(seed=args.train_seed,mode=spec["mode"],images=len(times),seconds_mean=float(np.mean(times)),
                seconds_median=float(np.median(times)),peak_gib=max(peaks) if peaks else None,
                reference_bank_setup_seconds=setup_seconds if reference else 0.,head_sha=spec["sha"],
                timing_scope="image decode + DINO global/4 tiles + matching if used + head; warmup excluded; reference bank pre-encoded"))
    finally:
        extractor.close()
    write_csv(root/"reports"/f"through{args.until_epoch}"/f"end_to_end_seed{args.train_seed}.csv",result)
    print("End-to-end audit written; reference-bank setup reported separately",flush=True)


def report(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    seeds = [int(s) for s in args.train_seeds.split(",")]
    rows, means, screens = [], [], []
    for seed in seeds:
        seed_args=argparse.Namespace(**vars(args)); seed_args.train_seed=seed
        specs=evaluation_specs(seed_args,cfg,torch.device("cpu"))
        expected=evaluation_signature(root,cfg,specs)
        for spec in specs:
            screens.append(dict(seed=seed,mode=spec["mode"],selector=spec["selector"],
                **{k:spec["point"][k] for k in ("epoch","alpha","PRO_exact","P_AP","loss","threshold")},
                fallback=spec["fallback"],head_sha=spec["sha"],parent_mode=spec["parent"]["mode"] if spec["parent"] else None))
        for dataset, category in sorted({(r["dataset"], r["category"]) for r in selected_rows(cfg, "eval")}):
            path = root / "results" / f"seed{seed}" / f"through{args.until_epoch}" / f"{dataset}_{category}.json"
            value = load_json(path)
            if value["fingerprint"] != digest(cfg) or value["signature"] != expected:
                raise ValueError("Stale evaluation report")
            rows.extend(value["rows"])
    metrics = ("PRO_exact", "P_AUROC", "P_AP", "I_AUROC", "F1_best", "background_FPR_source_threshold", "pixel_recall_source_threshold", "head_seconds_per_query")
    for key in sorted({(r["seed"], r["dataset"], r["mode"], r["selector"]) for r in rows}):
        seed, dataset, mode, selector = key
        group = [r for r in rows if (r["seed"],r["dataset"],r["mode"],r["selector"]) == key]
        value = dict(seed=seed, dataset=dataset, mode=mode, selector=selector, categories=len(group),
                     samples=sum(r["samples"] for r in group), alpha=group[0]["alpha"], selected_epoch=group[0]["selected_epoch"],
                     fallback=group[0]["fallback"], parent_mode=group[0]["parent_mode"])
        for name in metrics:
            valid = [r[name] for r in group if r[name] is not None]
            value[name] = float(np.mean(valid)) if valid else None
        for name in ("small_components", "small_hits_source_threshold", "recovered_small_source_threshold", "lost_small_source_threshold", "small_hits_target_fpr1_EVAL_ONLY"):
            value[name] = sum(r[name] for r in group)
        means.append(value)
    comparisons = []
    contrasts = (("H1","H0"),("H2","H1"),("H3","H1"),("T10","T00"),("T01","T00"),("T11","T10"),("T11","T01"),("R1","R0"),("R2","R0"),("R3","R2"))
    for seed in seeds:
        for dataset in cfg["eval_datasets"]:
            index = {r["mode"]:r for r in means if r["seed"]==seed and r["dataset"]==dataset and r["selector"]=="primary"}
            for a,b in contrasts:
                if a in index and b in index:
                    comparisons.append(dict(seed=seed,dataset=dataset,candidate=a,baseline=b,
                        **{f"delta_{k}_pp":100*(index[a][k]-index[b][k]) if index[a][k] is not None and index[b][k] is not None else None for k in ("PRO_exact","P_AP","P_AUROC","I_AUROC")}))
            if all(k in index for k in ("T00","T10","T01","T11")):
                comparisons.append(dict(seed=seed,dataset=dataset,candidate="T11-T10-T01+T00",baseline="interaction",
                    **{f"delta_{k}_pp":100*(index["T11"][k]-index["T10"][k]-index["T01"][k]+index["T00"][k]) if all(index[m][k] is not None for m in ("T00","T10","T01","T11")) else None for k in ("PRO_exact","P_AP","P_AUROC","I_AUROC")}))
    aggregate = []
    for key in sorted({(r["dataset"],r["mode"],r["selector"]) for r in means}):
        group = [r for r in means if (r["dataset"],r["mode"],r["selector"])==key]
        value = dict(dataset=key[0],mode=key[1],selector=key[2],seeds=len(group),fallback_runs=sum(r["fallback"] for r in group))
        for metric in metrics[:5]:
            numbers = [r[metric] for r in group if r[metric] is not None]
            value[metric+"_mean"] = float(np.mean(numbers)) if numbers else None
            value[metric+"_std"] = float(np.std(numbers, ddof=1)) if len(numbers)>1 else None
        aggregate.append(value)
    out = root / "reports" / f"through{args.until_epoch}"
    write_csv(out/"metrics.csv",rows)
    write_csv(out/"summary.csv",means)
    write_csv(out/"comparisons.csv",comparisons)
    write_csv(out/"seed_summary.csv",aggregate)
    write_csv(out/"source_screen.csv",screens)
    lines = ["# Three-stage alignment results", "", "Primary: source PRO/AP Pareto; bce: diagnostic source-loss selector, never a target-selected alternative.",
             "T00 is an alias of H3, not an independently trained model. Missing T/R modes can mean a source-only gate skipped them.",
             "Reference modes use same-category normal TRAIN support. FPR1 EVAL_ONLY is not a deployment threshold.",
             "Head timings exclude frozen extraction and are NOT end-to-end FPS. Single-seed standard deviations are unavailable, not zero.", "",
             "| Seed | Dataset | Mode | Selector | PRO | Pixel AP | Fallback |", "|---|---|---|---|---:|---:|---|"]
    lines.extend(f"| {r['seed']} | {r['dataset']} | {r['mode']} | {r['selector']} | {100*r['PRO_exact']:.2f} | {100*r['P_AP']:.2f} | {r['fallback']} |" for r in means)
    out.mkdir(parents=True,exist_ok=True)
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Report: {out/'README.md'}",flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=("prepare","repair_export","export","seal","train","gate","select","reference_gate","fit_distance","calibrate_r1","evaluate","report","audit","benchmark"))
    p.add_argument("--work_dir",required=True)
    p.add_argument("--v1_dir",default="")
    p.add_argument("--partition",choices=("source","eval"),default="source")
    p.add_argument("--mode",default="H1",choices=HEAD_MODES+("R2","R3"))
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--train_seed",type=int,default=42)
    p.add_argument("--train_seeds",default="42")
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--until_epoch",type=int,default=5)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--keep_weight",type=float,default=.1)
    p.add_argument("--alphas",default="0,0.05,0.1,0.25,0.5,1")
    p.add_argument("--front_level",type=int,choices=(1,2,3),default=3)
    p.add_argument("--shard_id",type=int,default=0)
    p.add_argument("--num_shards",type=int,default=2)
    p.add_argument("--benchmark_images",type=int,default=4)
    p.add_argument("--benchmark_mode",default="all")
    p.add_argument("--continue_on_fallback",action="store_true")
    p.add_argument("--probe",action="store_true")
    return p


def main():
    args = parser().parse_args()
    stages = dict(prepare=prepare,repair_export=repair_export,export=export_extra,seal=seal_extra,train=train,gate=stage_gate,select=select_front,
                  reference_gate=reference_gate,fit_distance=fit_distance,calibrate_r1=calibrate_r1,evaluate=evaluate,report=report,
                  audit=source_audit,benchmark=benchmark)
    result = stages[args.stage](args)
    if args.stage in ("gate","reference_gate") and not result:
        raise SystemExit(3)  # Deliberate source-only skip; every other failure is fatal.


if __name__ == "__main__":
    main()
