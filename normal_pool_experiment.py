#!/usr/bin/env python3
"""First normal-residual-pool experiment: frozen DINO, no Flow/text retraining.

All eight arms are reported, with no learned-head or distance fallback. Existing
residual_first_hr_text_v1 caches may be verified and reused READ ONLY.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import datetime
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

import residual_text_experiment as legacy
from tools.normal_residual_pool import normal_pool, second_moment, raw_maps, calibrate_views, image_scores
from tools.residual_text import fuse_views
from tools.residual_text_runtime import (FrozenDinoExtractor, atomic_npz, evaluation_mask,
                                        export_residuals, normalized_boxes, unchanged)
from tools.vlm_review import atomic_json, digest, file_digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "normal_residual_pool_v1"
ARMS = tuple(f"{family}_{view}" for family in ("D", "P", "S", "N") for view in ("G", "HR"))
CODE_FILES = tuple(dict.fromkeys(("normal_pool_experiment.py", "tools/normal_residual_pool.py",
                                 "run_exp_normal_pool.sh") + legacy.CODE_FILES))


def key(row):
    return row["dataset"], row["category"]


def group_name(value):
    return "__".join(value)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Immutable artifact changed: {path}; use NEW WORK_DIR")
    else:
        atomic_json(path, value)


def config(root):
    root = Path(root)
    cfg = load_json(root / "config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong protocol; use a NEW WORK_DIR separate from previous runs")
    for name, sha in cfg["code"].items():
        if file_digest(REPO / name) != sha:
            raise ValueError(f"Code changed: {name}; use NEW WORK_DIR")
    for name in ("manifest", "references"):
        if file_digest(root / f"{name}.json") != cfg[name + "_file_sha"]:
            raise ValueError(f"Changed {name}")
    return cfg


def rows(root, cfg, partition, prediction=False):
    manifest = load_json(Path(root) / "manifest.json")
    if partition == "eval":
        return [r for r in manifest if r["partition"] == "eval"]
    if prediction:
        return [r for r in manifest if r["partition"] == "val"]
    calibration = set(cfg["calibration_ids"])
    return [r for r in manifest if r["partition"] == "val" or r["id"] in calibration]


def groups(values):
    result = defaultdict(list)
    for row in values:
        result[key(row)].append(row)
    return dict(sorted(result.items()))


def prepare(args, catalog=None):
    root = Path(args.work_dir).resolve()
    if (args.reference_count not in (2, 4) or args.image_size < 32 or args.patch_size < 1
            or args.image_size % args.patch_size or not 2 <= args.feature_dim <= args.input_dim
            or args.eval_size < 8 or args.num_shards < 1 or args.calibration_images < 1
            or min(args.match_chunk, args.bank_chunk) < 1 or not 0 < args.val_fraction < .5
            or not .5 < args.tile_fraction < 1 or not 0 < args.high_weight < 1
            or not 0 <= args.pool_prior_weight <= 1 or not 0 < args.floor_ratio <= 1
            or not 0 < args.val_fpr < 1 or not 0 < args.top_fraction <= 1):
        raise ValueError("Invalid normal-pool settings (mutual matching requires K=2 or 4)")
    layers = [int(x) for x in args.visual_layers.split(",")]
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        raise ValueError("Invalid DINO layer list")
    if not Path(args.dino_weights).is_file() or not Path(args.dino_repo_dir, "hubconf.py").is_file():
        raise FileNotFoundError("Explicit local DINO weights and DINO_REPO_DIR are required")
    reuse = Path(args.reuse_dir).resolve() if args.reuse_dir else None
    old_cfg = None
    if reuse:
        if root == reuse or root in reuse.parents or reuse in root.parents:
            raise ValueError("WORK_DIR and reuse directory must be separate, non-nested directories")
        old_cfg = legacy.checked_config(reuse)
        manifest = copy.deepcopy(load_json(reuse / "manifest.json"))
        references = copy.deepcopy(load_json(reuse / "references.json"))
        categories = sorted({r["category"] for r in manifest if r["partition"] == "train"})
        val_categories = old_cfg["val_categories"]
    else:
        queries, candidates = catalog if catalog is not None else legacy.load_catalog(args)
        manifest = copy.deepcopy(queries)
        source_categories = sorted({r["category"] for r in manifest if r["partition"] == "source"},
                                   key=lambda c: digest([args.data_seed, c]))
        if len(source_categories) < 2:
            raise ValueError("Need at least two source classes")
        count = max(1, min(len(source_categories) - 1, round(len(source_categories) * args.val_fraction)))
        val_categories = source_categories[:count]
        categories = sorted(set(source_categories) - set(val_categories))
        for row in manifest:
            if row["partition"] == "source":
                row["partition"] = "val" if row["category"] in val_categories else "train"
        all_query_sha = {r["rgb_sha"] for r in manifest}
        references = []
        for category_key in sorted({key(r) for r in manifest}):
            eligible = sorted([r for r in candidates if key(r) == category_key],
                              key=lambda r: digest([args.data_seed, "reference", r["rgb_sha"]]))
            chosen, seen = [], set()
            for row in eligible:
                if row["mask_path"] or row["rgb_sha"] in all_query_sha or row["rgb_sha"] in seen:
                    continue
                chosen.append(copy.deepcopy(row))
                seen.add(row["rgb_sha"])
                if len(chosen) == args.reference_count:
                    break
            if len(chosen) != args.reference_count:
                raise ValueError(f"Need K distinct normal TRAIN references without query duplicates: {category_key}")
            references.extend(chosen)
    # Partition and RGB checks are also applied to reused manifests.
    seen = {}
    for row in manifest:
        if seen.setdefault(row["rgb_sha"], row["partition"]) != row["partition"]:
            raise ValueError("Decoded image content crosses data partitions")
    if len({r["id"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate query id")
    reference_families = {}
    for category_key, support in groups(references).items():
        if (len(support) != args.reference_count or any(r["mask_path"] or r["partition"] != "reference" for r in support)
                or len({r["rgb_sha"] for r in support}) != len(support)
                or any(r["rgb_sha"] in seen for r in support)):
            raise ValueError(f"Invalid normal supports: {category_key}")
        for row in support:
            if reference_families.setdefault(row["rgb_sha"], category_key) != category_key:
                raise ValueError("Normal reference content crosses category/dataset groups")
    if set(groups(references)) != set(groups(manifest)):
        raise ValueError("Reference groups do not cover all query classes")
    for partition in ("train", "val"):
        if {bool(r["mask_path"]) for r in manifest if r["partition"] == partition} != {False, True}:
            raise ValueError(f"Source {partition} needs normal and abnormal images")
    normal_groups = groups([r for r in manifest if r["partition"] == "train" and not r["mask_path"]])
    if len(normal_groups) != len(categories) or args.calibration_images < len(categories):
        raise ValueError("Calibration must cover every source TRAIN class with normal queries")
    for values in normal_groups.values():
        values.sort(key=lambda r: digest([args.data_seed, "normal_pool_cal", r["id"]]))
    calibration = []
    for index in range(max(map(len, normal_groups.values()))):
        for values in normal_groups.values():
            if index < len(values) and len(calibration) < args.calibration_images:
                calibration.append(values[index]["id"])
    cfg = {k: v for k, v in vars(args).items() if k not in
           {"stage", "work_dir", "partition", "shard_id", "device", "reuse_dir"}}
    cfg.update(protocol=PROTOCOL, layers=layers, val_categories=val_categories,
               train_categories=categories, calibration_ids=calibration,
               reuse_dir=str(reuse) if reuse else "", dino_sha=file_digest(args.dino_weights),
               dino_repo_dir=str(Path(args.dino_repo_dir).resolve()),
               dino_weights=str(Path(args.dino_weights).resolve()),
               backbone_code=digest({str(p.relative_to(args.dino_repo_dir)): file_digest(p)
                                     for p in sorted(Path(args.dino_repo_dir).rglob("*.py"))}),
               code={name: file_digest(REPO / name) for name in CODE_FILES},
               note="Statistics-only pool experiment; no Flow/text training or target parameter updates",
               matching_note="Support pool is K-to-(K-1); queries use full K; source CAL uses the same full-K queries; no target FPR guarantee")
    if old_cfg:
        checks = ("source", "eval_datasets", "data_seed", "image_size", "patch_size", "input_dim",
                  "feature_dim", "reference_count", "layers", "tile_fraction", "precision", "dino_sha", "backbone_code", "dino_model_name",
                  "val_fraction", "limit_per_category")
        for field in checks:
            if cfg[field] != old_cfg[field]:
                raise ValueError(f"Reuse setting differs: {field}; match V1 or run fresh")
        cfg["reuse_config_sha"] = file_digest(reuse / "config.json")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() and load_json(root / "config.json").get("protocol") != PROTOCOL:
        raise ValueError("Legacy work directory forbidden; use NEW WORK_DIR")
    write_once(root / "manifest.json", manifest)
    write_once(root / "references.json", references)
    cfg.update(manifest_file_sha=file_digest(root / "manifest.json"),
               references_file_sha=file_digest(root / "references.json"))
    write_once(root / "config.json", cfg)
    side = args.image_size // args.patch_size
    pool_gib = len(references) * 5 * len(layers) * side * side * (2 * args.feature_dim + 8) / 2**30
    prediction_gib = (len(calibration) + sum(r["partition"] in ("val", "eval") for r in manifest)) * len(ARMS) * args.eval_size**2 * 4 / 2**30
    print(f"Prepared K={args.reference_count}; normal CAL={len(calibration)}, VAL classes={val_categories}", flush=True)
    print(f"New normal pools ~{pool_gib:.1f} GiB; prediction maps ~{prediction_gib:.1f} GiB; query residuals additional if not reused", flush=True)
    print(f"Query cache reuse (read-only): {cfg['reuse_dir'] or 'disabled'}", flush=True)
    return cfg


def pool_path(root, category_key):
    return Path(root) / "normal_pools" / (group_name(category_key) + ".npz")


def feature_meta(root, row):
    return Path(root) / "features" / (row["id"] + ".json")


def load_feature(root, cfg, row):
    meta = load_json(feature_meta(root, row))
    if meta["fingerprint"] != digest(cfg) or meta["id"] != row["id"] or file_digest(meta["path"]) != meta["sha"]:
        raise ValueError("Query residual cache identity/digest mismatch")
    with np.load(meta["path"], allow_pickle=False) as value:
        if set(value.files) != legacy.FEATURE_KEYS:
            raise ValueError("Only residual-only five-view caches may be used")
        data = {name: value[name].copy() for name in value.files}
    side, levels = cfg["image_size"] // cfg["patch_size"], len(cfg["layers"])
    if (data["residual"].shape != (5, levels, cfg["feature_dim"], side, side)
            or data["distance"].shape != (5, levels, side, side)
            or data["match_index"].shape != data["distance"].shape
            or any(data[name].shape != (5, 4) for name in ("boxes", "pixel_boxes"))
            or any(not np.isfinite(v).all() for v in data.values())):
        raise ValueError("Invalid per-view cache shapes or nonfinite values")
    return data


def reuse_feature(root, cfg, row):
    reuse = Path(cfg["reuse_dir"])
    if file_digest(reuse / "config.json") != cfg["reuse_config_sha"]:
        raise ValueError("Reuse configuration changed")
    path = legacy.feature_path(reuse, row)
    seal = load_json(legacy.seal_path(reuse, "eval" if row["partition"] == "eval" else "source"))
    old_meta = load_json(path.with_suffix(".json"))
    old_cfg = load_json(reuse / "config.json")
    expected_refs = [r["id"] for r in load_json(Path(root) / "references.json") if key(r) == key(row)]
    if (seal["fingerprint"] != digest(old_cfg) or old_meta["fingerprint"] != digest(old_cfg)
            or old_meta["schema"] != "residual_only_5views_v1" or old_meta["id"] != row["id"]
            or [r["id"] for r in old_meta["references"]] != expected_refs
            or seal["features"][row["id"]] != file_digest(path)
            or seal["metadata"][row["id"]] != file_digest(path.with_suffix(".json"))
            or old_meta["sha"] != seal["features"][row["id"]]):
        raise ValueError("V1 residual cache/reference provenance mismatch")
    write_once(feature_meta(root, row), dict(id=row["id"], fingerprint=digest(cfg),
               path=str(path.resolve()), sha=old_meta["sha"], origin="verified_read_only_v1"))


def export(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard index")
    support_groups = groups(load_json(root / "references.json"))
    extractor = None
    try:
        for index, (category_key, queries) in enumerate(groups(rows(root, cfg, args.partition)).items()):
            if index % cfg["num_shards"] != args.shard_id:
                continue
            support = support_groups[category_key]
            for row in support + queries:
                unchanged(row)
            pool = pool_path(root, category_key)
            pending = [r for r in queries if not feature_meta(root, r).exists()]
            if (pool.exists() and pool.with_suffix(".json").exists()
                    and (not pending or cfg["reuse_dir"])):
                if cfg["reuse_dir"]:
                    for row in pending:
                        reuse_feature(root, cfg, row)
                print(f"export {category_key}: cached", flush=True)
                continue
            if (root / f"sealed_{args.partition}.json").exists():
                raise ValueError("Cannot regenerate sealed export")
            if extractor is None:
                current_code = digest({str(p.relative_to(cfg["dino_repo_dir"])): file_digest(p)
                                       for p in sorted(Path(cfg["dino_repo_dir"]).rglob("*.py"))})
                if current_code != cfg["backbone_code"]:
                    raise ValueError("DINO implementation changed")
                extractor = FrozenDinoExtractor(cfg, legacy.device_for(args, cfg))
            normal_views = []
            for row in support:
                with Image.open(row["image_path"]) as image:
                    tokens, _ = extractor.views(image.convert("RGB"))
                normal_views.append([v.cpu() for v in tokens])
            if not pool.exists() or not pool.with_suffix(".json").exists():
                device_views = [[v.to(extractor.device) for v in views] for views in normal_views]
                arrays, correspondence = normal_pool(extractor, device_views, support, cfg)
                atomic_npz(pool, **arrays)
                write_once(pool.with_suffix(".json"), dict(schema=PROTOCOL, fingerprint=digest(cfg),
                    sha=file_digest(pool), category=list(category_key), references=[r["id"] for r in support],
                    correspondence=correspondence, pool_reference_count=len(support)-1,
                    query_reference_count=len(support), second_moment=arrays["second_moment"].tolist(),
                    vector_shape=list(arrays["residual"].shape)))
                del arrays, device_views
            for row in pending:
                if cfg["reuse_dir"]:
                    reuse_feature(root, cfg, row)
                else:
                    with Image.open(row["image_path"]) as image:
                        views, boxes = extractor.views(image.convert("RGB"))
                    arrays = export_residuals(extractor, views, normal_views, cfg)
                    arrays.update(boxes=normalized_boxes(boxes, row["native_size"]), pixel_boxes=np.asarray(boxes, np.int32))
                    path = root / "features" / (row["id"] + ".npz")
                    atomic_npz(path, **arrays)
                    write_once(feature_meta(root, row), dict(id=row["id"], fingerprint=digest(cfg),
                               path=str(path.resolve()), sha=file_digest(path), origin="fresh_dino"))
                print(f"export {args.partition} {row['id']}", flush=True)
    finally:
        if extractor is not None:
            extractor.close()


def pool_moment(root, cfg, category_key):
    path = pool_path(root, category_key)
    meta = load_json(path.with_suffix(".json"))
    support = [r["id"] for r in load_json(Path(root) / "references.json") if key(r) == category_key]
    if (meta["fingerprint"] != digest(cfg) or meta["sha"] != file_digest(path)
            or meta["references"] != support or len(meta["correspondence"]) != len(support)):
        raise ValueError("Normal pool identity mismatch")
    for record, query in zip(meta["correspondence"], support):
        if record != dict(query_id=query, bank_ids=[r for r in support if r != query]):
            raise ValueError("Self-image leakage or wrong normal-pool correspondence")
    moment = np.asarray(meta["second_moment"], np.float64)
    if moment.shape != (len(cfg["layers"]), 2, cfg["feature_dim"]) or not np.isfinite(moment).all() or (moment < 0).any():
        raise ValueError("Invalid normal second moments")
    return moment


def seal(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    features, pools = {}, {}
    for category_key, queries in groups(rows(root, cfg, args.partition)).items():
        moment = pool_moment(root, cfg, category_key)
        path = pool_path(root, category_key)
        side, levels, count = cfg["image_size"] // cfg["patch_size"], len(cfg["layers"]), cfg["reference_count"]
        with np.load(path, allow_pickle=False) as pool:
            expected = (count, 5, levels, cfg["feature_dim"], side, side)
            if (set(pool.files) != {"residual", "distance", "match_index", "second_moment"}
                    or pool["residual"].shape != expected
                    or pool["distance"].shape != (count, 5, levels, side, side)
                    or pool["match_index"].shape != pool["distance"].shape
                    or any(not np.isfinite(pool[n]).all() for n in pool.files)
                    or (pool["match_index"] < 0).any()
                    or (pool["match_index"][:, :1] >= (count-1)*side*side).any()
                    or (pool["match_index"][:, 1:] >= (count-1)*4*side*side).any()
                    or not np.allclose(second_moment(pool["residual"]), moment, rtol=1e-10, atol=1e-15)
                    or not np.allclose(pool["second_moment"], moment, rtol=1e-10, atol=1e-15)):
                raise ValueError("Invalid normal residual pool arrays/moments/correspondence indices")
        pools[group_name(category_key)] = dict(sha=file_digest(path), metadata=file_digest(path.with_suffix(".json")))
        for row in queries:
            load_feature(root, cfg, row)
            features[row["id"]] = file_digest(feature_meta(root, row))
    write_once(root / f"sealed_{args.partition}.json", dict(fingerprint=digest(cfg), features=features, pools=pools))


def checked_seal(root, cfg, partition):
    value = load_json(Path(root) / f"sealed_{partition}.json")
    if value["fingerprint"] != digest(cfg):
        raise ValueError("Wrong export seal")
    for row in rows(root, cfg, partition):
        if value["features"][row["id"]] != file_digest(feature_meta(root, row)):
            raise ValueError("Feature pointer changed after sealing")
    for category_key in groups(rows(root, cfg, partition)):
        path = pool_path(root, category_key)
        entry = value["pools"][group_name(category_key)]
        if entry["sha"] != file_digest(path) or entry["metadata"] != file_digest(path.with_suffix(".json")):
            raise ValueError("Normal pool changed after sealing")
    return value


def fit_identity(root, cfg):
    return dict(fingerprint=digest(cfg), source_seal=file_digest(Path(root) / "sealed_source.json"))


def fit(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    checked_seal(root, cfg, "source")
    path = root / "normal_model.json"
    identity = fit_identity(root, cfg)
    if path.exists():
        if load_json(path)["identity"] != identity:
            raise ValueError("Normal model identity mismatch")
        if path.with_suffix(".sha.json").exists():
            model_for(root, cfg)
            return
        # An interrupted first fit can leave the atomic model without its hash.
        # Recompute it and require exact equality before completing the pair.
    # Training classes only: validation/target pools MUST NOT enter this prior.
    prior_keys = [(cfg["source"], category) for category in cfg["train_categories"]]
    moments = {k: pool_moment(root, cfg, k) for k in groups(rows(root, cfg, "source"))}
    prior = np.stack([moments[k] for k in prior_keys]).mean(0)
    calibration_ids = set(cfg["calibration_ids"])
    queries = [r for r in rows(root, cfg, "source") if r["id"] in calibration_ids]
    if any(r["mask_path"] or r["partition"] != "train" for r in queries):
        raise ValueError("CAL must contain only known-normal source TRAIN queries")
    values = {f: [[[] for _ in range(2)] for _ in cfg["layers"]] for f in ("D", "P", "S", "N")}
    for row in queries:
        data = load_feature(root, cfg, row)
        score = raw_maps(data["residual"], data["distance"], prior, moments[key(row)],
                         cfg["pool_prior_weight"], cfg["floor_ratio"])
        for family, maps in score.items():
            for layer in range(len(cfg["layers"])):
                for view in range(5):
                    flat = maps[view, layer].ravel()
                    values[family][layer][int(view > 0)].append(flat[::max(1, len(flat)//256)])
    calibrations = {}
    for family, levels in values.items():
        center, scale = np.zeros((len(levels), 2)), np.ones((len(levels), 2))
        for layer, views in enumerate(levels):
            for view_type, arrays in enumerate(views):
                array = np.concatenate(arrays)
                center[layer, view_type] = np.median(array)
                scale[layer, view_type] = max(float(np.quantile(array, .75)-np.quantile(array, .25)), 1e-4)
        calibrations[family] = dict(center=center.tolist(), scale=scale.tolist())
    model = dict(identity=identity, source_prior=prior.tolist(), prior_classes=[list(k) for k in prior_keys],
                 calibration_ids=cfg["calibration_ids"], calibration=calibrations,
                 pool_prior_weight=cfg["pool_prior_weight"], floor_ratio=cfg["floor_ratio"])
    normal_values = {arm: [] for arm in ARMS}
    for row in queries:
        data = load_feature(root, cfg, row)
        for arm, value in predict_maps(data, moments[key(row)], model, cfg).items():
            normal_values[arm].append(value.ravel()[::max(1, value.size//8192)])
    model["thresholds"] = {arm: float(np.quantile(np.concatenate(arrays), 1-cfg["val_fpr"], method="higher"))
                            for arm, arrays in normal_values.items()}
    write_once(path, model)
    write_once(path.with_suffix(".sha.json"), dict(identity=identity, sha=file_digest(path)))
    print("Fitted normal statistics/CAL thresholds only; no Flow, CLIP or gradient training", flush=True)


def predict_maps(data, support, model, cfg):
    raw = raw_maps(data["residual"], data["distance"], model["source_prior"], support,
                   cfg["pool_prior_weight"], cfg["floor_ratio"])
    boxes = torch.from_numpy(data["boxes"].astype(np.float32))[None]
    result = {}
    for family, score in raw.items():
        views = torch.from_numpy(calibrate_views(score, model["calibration"][family]).astype(np.float32))[None]
        for variant in ("G", "HR"):
            result[f"{family}_{variant}"] = fuse_views(views, boxes, cfg["eval_size"], variant == "HR",
                                                        cfg["high_weight"])[0, 0].numpy()
    return result


def model_for(root, cfg):
    path = Path(root) / "normal_model.json"
    model = load_json(path)
    seal = load_json(path.with_suffix(".sha.json"))
    if (model["identity"] != fit_identity(root, cfg) or seal["identity"] != model["identity"]
            or seal["sha"] != file_digest(path)):
        raise ValueError("Normal statistics changed")
    return model


def prediction_identity(root, cfg, partition):
    return dict(fingerprint=digest(cfg), partition=partition,
                normal_model=file_digest(Path(root)/"normal_model.json"),
                export_seal=file_digest(Path(root)/f"sealed_{partition}.json"))


def predict(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    checked_seal(root, cfg, args.partition)
    model = model_for(root, cfg)
    identity = prediction_identity(root, cfg, args.partition)
    out = root / "predictions" / args.partition
    for index, (category_key, queries) in enumerate(groups(rows(root, cfg, args.partition, True)).items()):
        if index % cfg["num_shards"] != args.shard_id:
            continue
        moment = pool_moment(root, cfg, category_key)
        for row in queries:
            path = out / (row["id"] + ".npz")
            if path.exists() and path.with_suffix(".json").exists():
                meta = load_json(path.with_suffix(".json"))
                if meta["identity"] != identity or meta["sha"] != file_digest(path):
                    raise ValueError("Prediction identity changed")
                continue
            if (out / "locked.json").exists():
                raise ValueError("Cannot replace a locked prediction")
            data = load_feature(root, cfg, row)
            atomic_npz(path, **predict_maps(data, moment, model, cfg))
            write_once(path.with_suffix(".json"), dict(identity=identity, sha=file_digest(path)))
        print(f"predicted {args.partition} {category_key}", flush=True)


def lock_predictions(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    checked_seal(root, cfg, args.partition)
    model_for(root, cfg)
    out, hashes = root / "predictions" / args.partition, {}
    identity = prediction_identity(root, cfg, args.partition)
    for row in rows(root, cfg, args.partition, True):
        path = out / (row["id"] + ".npz")
        meta = load_json(path.with_suffix(".json"))
        if meta["identity"] != identity or meta["sha"] != file_digest(path):
            raise ValueError("Prediction changed before lock")
        with np.load(path, allow_pickle=False) as maps:
            if set(maps.files) != set(ARMS) or any(maps[a].shape != (cfg["eval_size"],)*2
                    or not np.isfinite(maps[a]).all() for a in ARMS):
                raise ValueError("Invalid prediction arrays")
        hashes[row["id"]] = meta["sha"]
    write_once(out / "locked.json", dict(identity=identity, predictions=hashes))


def panel(root, cfg, row, maps, partition):
    size = min(256, cfg["eval_size"])
    with Image.open(row["image_path"]) as image:
        original = image.convert("RGB").resize((size, size))
    tiles = [("query", original), ("GT (evaluation only)", Image.fromarray(
        (evaluation_mask(row, size)*255).astype(np.uint8)).convert("RGB"))]
    for arm in ("D_HR", "P_HR", "S_HR", "N_HR"):
        value = torch.from_numpy(maps[arm])[None, None]
        p = torch.nn.functional.interpolate(value, (size, size), mode="bilinear", align_corners=False)[0, 0].sigmoid().numpy()
        color = np.stack([p, .15*(1-p), 1-p], -1)*255
        tiles.append((arm, Image.fromarray((.55*np.asarray(original)+.45*color).clip(0,255).astype(np.uint8))))
    canvas = Image.new("RGB", (3*size, 2*(size+24)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, tile) in enumerate(tiles):
        x, y = i%3*size, i//3*(size+24)
        draw.text((x+4, y+4), name, fill="black")
        canvas.paste(tile, (x,y+24))
    path = Path(root)/"panels"/partition/group_name(key(row))/(row["id"]+".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(args):
    from sklearn.metrics import average_precision_score, roc_auc_score
    root, cfg = Path(args.work_dir), config(args.work_dir)
    checked_seal(root, cfg, args.partition)
    model = model_for(root, cfg)
    out = root / "predictions" / args.partition
    lock = load_json(out / "locked.json")  # target masks are forbidden before this exists
    if lock["identity"] != prediction_identity(root, cfg, args.partition):
        raise ValueError("Invalid prediction lock")
    for index, (category_key, queries) in enumerate(groups(rows(root, cfg, args.partition, True)).items()):
        if index % cfg["num_shards"] != args.shard_id:
            continue
        paths = [out/(r["id"]+".npz") for r in queries]
        if any(lock["predictions"][r["id"]] != file_digest(p) for r,p in zip(queries,paths)):
            raise ValueError("Predictions changed after lock")
        for row in queries:
            unchanged(row, masks=args.partition == "source")
        # First target-label content access. No masks in pools, fit or predict.
        truth = np.stack([evaluation_mask(r, cfg["eval_size"]) for r in queries])
        labels = truth.reshape(len(truth),-1).any(1)
        metrics = []
        for arm in ARMS:
            values = []
            for path in paths:
                with np.load(path, allow_pickle=False) as maps:
                    values.append(maps[arm].copy())
            maps = np.stack(values)
            point = legacy.metric_row(truth, maps, model["thresholds"][arm])
            maximum, top = image_scores(maps, cfg["top_fraction"])
            both = labels.any() and not labels.all()
            point.update(I_AP=float(average_precision_score(labels, maximum)) if labels.any() else None,
                         I_AUROC_topmean=float(roc_auc_score(labels,top)) if both else None,
                         I_AP_topmean=float(average_precision_score(labels,top)) if labels.any() else None,
                         dataset=category_key[0], category=category_key[1], arm=arm,
                         partition=args.partition, data_seed=cfg["data_seed"])
            metrics.append(point)
        write_once(root/"metrics"/args.partition/(group_name(category_key)+".json"), dict(
            prediction_lock_sha=file_digest(out/"locked.json"), metrics=metrics,
            evaluation_mask_hashes={r["id"]:file_digest(r["mask_path"]) if r["mask_path"] else None for r in queries}))
        for row in sorted(queries,key=lambda r:digest([cfg["data_seed"],"pool_panel",r["id"]]))[:2]:
            with np.load(out/(row["id"]+".npz"),allow_pickle=False) as maps:
                panel(root,cfg,row,{a:maps[a].copy() for a in ARMS},args.partition)
        print(f"evaluated {args.partition} {category_key}",flush=True)


def report(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    all_rows = []
    for category_key in groups(rows(root,cfg,args.partition,True)):
        value = load_json(root/"metrics"/args.partition/(group_name(category_key)+".json"))
        if value["prediction_lock_sha"] != file_digest(root/"predictions"/args.partition/"locked.json"):
            raise ValueError("Stale metrics")
        all_rows.extend(value["metrics"])
    numeric = ("PRO_exact","P_AP","P_AUROC","I_AUROC","I_AP","I_AUROC_topmean","I_AP_topmean",
               "background_FPR_source_threshold","pixel_recall_source_threshold")
    summary = []
    for dataset in sorted({r["dataset"] for r in all_rows}):
        for arm in ARMS:
            selected = [r for r in all_rows if r["dataset"]==dataset and r["arm"]==arm]
            mean = {n:float(np.mean([r[n] for r in selected if r[n] is not None]))
                    if any(r[n] is not None for r in selected) else None for n in numeric}
            summary.append(dict(dataset=dataset,arm=arm,categories=len(selected),**mean))
    comparisons = []
    for dataset in sorted({r["dataset"] for r in summary}):
        lookup = {r["arm"]:r for r in summary if r["dataset"]==dataset}
        for name,a,b in (("highres","D_HR","D_G"),("projection","P_HR","D_HR"),
                         ("source_channel_scale","S_HR","P_HR"),("support_pool_increment","N_HR","S_HR"),
                         ("total_pool_increment","N_HR","P_HR")):
            comparisons.append(dict(dataset=dataset,comparison=name,**{n+"_pp":100*(lookup[a][n]-lookup[b][n])
                if lookup[a][n] is not None and lookup[b][n] is not None else None for n in numeric}))
    out=root/"reports"/args.partition
    for name,values in (("metrics",all_rows),("summary",summary),("comparisons",comparisons)):
        legacy.write_csv(out/(name+".csv"),values)
    lines=["# Normal residual pool experiment", "", "D=full-dimensional distance; P=projected distance; S=source channel scale; N=source+normal-support-pool channel scale.",
           "All arms are direct outputs: no hidden Flow fallback, no text, no gradient training.",
           "All rows use the same references and query features. G=global; HR=global+four native crops.",
           "Source normal CAL only for score scaling/thresholds; source VAL is class-held-out. No target-label selection.",
           "Support mutual matching uses K-1 OTHER images; query uses K. This is a variation prior, NOT a 1% target FPR guarantee.",
           "I_AUROC uses max pooling; I_AUROC_topmean is a separate fixed top-fraction diagnostic, never target-selected.",
           "One support seed is not evidence of robustness. F1_best is an evaluation oracle.","",
           "| Dataset | Arm | PRO | Pixel AP | Pixel AUROC | Image AUROC (max) | Image AUROC (top mean) |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| "+row["dataset"]+" | "+row["arm"]+" | "+" | ".join(
            f"{100*row[n]:.2f}" if row[n] is not None else "n.a." for n in numeric[:4]+("I_AUROC_topmean",))+" |")
    out.mkdir(parents=True,exist_ok=True)
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


def package(args):
    root=Path(args.work_dir)
    config(root)
    path=root/("normal_pool_analysis_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+".zip")
    candidates=[root/name for name in ("config.json","manifest.json","references.json","normal_model.json","normal_model.sha.json",
                                       "sealed_source.json","sealed_eval.json")]
    for folder in ("normal_pools","metrics","reports","panels"):
        candidates.extend(p for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".json",".csv",".md",".png"))
    candidates.extend((root/"predictions").glob("*/locked.json"))
    candidates.extend(root.glob("run_*.log"))
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(set(candidates)):
            if item.is_file():
                archive.write(item,item.relative_to(root).as_posix())
    print(f"Analysis package (no huge raw pools/query caches/prediction tensors): {path}",flush=True)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage",required=True,choices=("prepare","export","seal","fit","predict","lock_predictions","evaluate","report","package"))
    result.add_argument("--work_dir",required=True)
    result.add_argument("--reuse_dir",default="")
    result.add_argument("--partition",choices=("source","eval"),default="source")
    result.add_argument("--device",default="cuda:0")
    result.add_argument("--source",default="visa")
    result.add_argument("--eval_datasets",default="mvtec,btad,mpdd")
    result.add_argument("--source_root",default="")
    result.add_argument("--target_root",default="")
    result.add_argument("--data_roots",default="{}")
    result.add_argument("--dino_repo_dir",default="./dinov3")
    result.add_argument("--dino_model_name",default="dinov3_vitl16")
    result.add_argument("--dino_weights",default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    result.add_argument("--visual_layers",default="5,11,17,23")
    result.add_argument("--precision",choices=("fp32","bf16"),default="bf16")
    for name,default in dict(data_seed=42,image_size=512,eval_size=512,patch_size=16,input_dim=1024,
            feature_dim=256,reference_count=4,match_chunk=128,bank_chunk=4096,num_shards=2,
            shard_id=0,calibration_images=64,limit_per_category=0).items():
        result.add_argument("--"+name,type=int,default=default)
    for name,default in dict(tile_fraction=.6,high_weight=.25,val_fraction=.25,val_fpr=.01,
                            pool_prior_weight=.5,floor_ratio=.01,top_fraction=.01).items():
        result.add_argument("--"+name,type=float,default=default)
    return result


if __name__=="__main__":
    arguments=parser().parse_args()
    globals()[arguments.stage](arguments)
