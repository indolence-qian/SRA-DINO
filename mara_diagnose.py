#!/usr/bin/env python3
"""A: step/return/gradient audit. B: paired frozen D0-D3 interventions.

prepare -> export (independent GPU workers) -> diagnose -> collect.
No training, no checkpoint selection, no VLM calls. Uses corrected STOP semantics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import asdict, replace
import csv
import hashlib
import gc
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from tools.mara_agent import MARAAgent
from tools.mara_diagnostics import (
    ARMS, PROTOCOL, checkpoint_config, gradient_probe, map_metrics, move_inputs,
    score_trace, select_indices, stable_id, trace_infer, validate_cache,
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    initial = ("sample_id", "dataset", "category", "arm", "step")
    fields = set().union(*(row.keys() for row in rows)) if rows else set()
    fields = [k for k in initial if k in fields]+sorted(fields-set(initial))
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_torch(path):
    # Local, user-specified checkpoints/caches, never remote untrusted downloads.
    return torch.load(path, map_location="cpu", weights_only=False)


def check_signature(path, expected):
    if Path(path).exists() and load_json(path).get("signature") != expected:
        raise ValueError(f"Output belongs to another run: {path}; use a new WORK_DIR")


def load_work(work_dir, verify_checkpoint=True, protocol=PROTOCOL):
    root = Path(work_dir).resolve()
    cfg = load_json(root/"config.json")
    manifest = load_json(root/"manifest.json")
    if cfg.get("protocol") != protocol or digest(manifest) != cfg["manifest_sha256"]:
        raise ValueError("Work directory protocol/manifest mismatch")
    if verify_checkpoint and sha256_file(cfg["mara_ckpt"]) != cfg["checkpoint_sha256"]:
        raise ValueError("MARA checkpoint changed since prepare")
    return root, cfg, manifest, digest(cfg)


def prepare(args, protocol=PROTOCOL, arms=None, extra_config=None):
    ckpt = Path(args.mara_ckpt).resolve()
    if not ckpt.is_file():
        raise ValueError("MARA_CKPT must name ONE explicit checkpoint file, not a directory")
    root = Path(args.work_dir).resolve()
    repository = Path(__file__).resolve().parent
    if root == repository or root == root.parent or repository.is_relative_to(root) or ckpt.is_relative_to(root):
        raise ValueError("WORK_DIR must not be a broad directory or contain the checkpoint")
    if root.exists() and not (root/"config.json").exists():
        unrelated = [p for p in root.iterdir() if p.name != ".run.lock" and not (
            p.name.startswith("run_") and p.suffix == ".log")]
        if unrelated:
            raise ValueError("Nonempty WORK_DIR has no matching config; use a new directory")
    payload = load_torch(ckpt)
    original_cfg = checkpoint_config(payload)
    required_base = ("dino_single_config", "dino_single_head") if payload.get("base_arch") == "dino_single" else (
        "cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner")
    if any(key not in payload for key in required_base):
        raise ValueError("MARA checkpoint is missing embedded Base state; use the original complete MARA checkpoint")
    from Datasets import DATASET_CLASSES, DATASET_REGISTRY
    runtime_cfg = replace(original_cfg, max_steps=args.steps) if args.steps else original_cfg
    roots = json.loads(args.data_roots)
    if not isinstance(roots, dict) or any(not isinstance(v, str) for v in roots.values()):
        raise ValueError("DATA_ROOTS must be a JSON object of dataset names to paths")
    unsupported = set(roots)-set(DATASET_REGISTRY)
    if unsupported:
        raise ValueError(f"Unknown DATA_ROOTS datasets: {sorted(unsupported)}")
    datasets = args.datasets.split()
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("DATASETS must be a nonempty unique list")
    manifest = []
    for dataset_name in datasets:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        dataset_cls, split_cls, default_root = DATASET_REGISTRY[dataset_name]
        source = str(Path(roots.get(dataset_name, default_root)).resolve())
        if not Path(source).is_dir():
            raise FileNotFoundError(f"Missing dataset {dataset_name}: {source}; set DATA_ROOTS")
        categories = sorted(DATASET_CLASSES[dataset_name])
        if args.categories:
            requested = set(args.categories.split(","))
            categories = [c for c in categories if c in requested]
            if not categories:
                raise ValueError(f"No requested categories in {dataset_name}")
        for category in categories:
            dataset = dataset_cls(source=source, split=split_cls.TEST, classname=category,
                                  resize=args.image_size, imagesize=args.image_size)
            items = dataset.data_to_iterate
            limits = (extra_config or {}).get("dataset_sample_limits", {})
            indices = select_indices(items, limits.get(dataset_name, args.samples_per_category), args.seed)
            if not indices:
                raise ValueError(f"No samples: {dataset_name}/{category}")
            for index in indices:
                item = items[index]
                image_path = str(Path(item[2]).resolve())
                mask_path = str(Path(item[3]).resolve()) if item[3] is not None else None
                record = dict(
                    id=stable_id(dataset_name, category, image_path), dataset=dataset_name,
                    category=category, source=source, dataset_index=index,
                    image_path=image_path, image_sha256=sha256_file(image_path),
                    mask_path=mask_path,
                    mask_sha256=sha256_file(mask_path) if mask_path else None,
                    sampling_stratum="mask_present" if mask_path else "mask_absent",
                )
                record["index"] = len(manifest)
                record["shard"] = record["index"] % args.num_shards
                manifest.append(record)
    if len({r["id"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate selected image identity")
    if extra_config and extra_config.get("category_shards"):
        owners = {key: i % args.num_shards for i, key in enumerate(sorted(
            {(r["dataset"], r["category"]) for r in manifest}))}
        for record in manifest:
            record["shard"] = owners[(record["dataset"], record["category"])]
    gradient_ids = [r["id"] for r in sorted(
        manifest, key=lambda r: stable_id(args.seed, "gradient", r["id"]))[:args.gradient_images]]
    runtime = dict(
        image_size=args.image_size,
        visual_backbone=payload.get("visual_backbone", "dino"),
        visual_layers=list(payload.get("visual_layers", (5, 11, 17, 23))),
        text_source=args.text_source, hfa_setting="hfa4", hfa_layers="",
        hfa_bottleneck=None, dino_bottleneck=256,
        clip_model_name=args.clip_model_name, clip_pretrained=args.clip_pretrained,
        dino_repo_dir=args.dino_repo_dir or payload.get("dino_repo_dir", "./dinov3"),
        dino_model_name=payload.get("dino_model_name", "dinov3_vitl16"),
        dino_weights=args.dino_weights or payload.get("dino_weights", "./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    )
    if len(runtime["visual_layers"]) != runtime_cfg.num_layers:
        raise ValueError("Base visual_layers do not match MARA num_layers")
    backbone_artifact = None
    if runtime["visual_backbone"] == "dino":
        if not Path(runtime["dino_repo_dir"]).is_dir() or not Path(runtime["dino_weights"]).is_file():
            raise FileNotFoundError("DINO repo/weights missing; set DINO_REPO_DIR and DINO_WEIGHTS")
        runtime["dino_repo_dir"] = str(Path(runtime["dino_repo_dir"]).resolve())
        runtime["dino_weights"] = str(Path(runtime["dino_weights"]).resolve())
        backbone_artifact = dict(path=runtime["dino_weights"], sha256=sha256_file(runtime["dino_weights"]))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    cfg = dict(
        protocol=protocol, git_revision=revision, mara_ckpt=str(ckpt),
        checkpoint_sha256=sha256_file(ckpt), checkpoint_epoch_zero_based=payload.get("epoch"),
        base_ckpt_metadata=payload.get("base_ckpt"), base_arch=payload.get("base_arch", "clip_dino"),
        original_mara_config=asdict(original_cfg), mara_config=asdict(runtime_cfg), runtime=runtime,
        backbone_artifact=backbone_artifact, image_size=args.image_size, datasets=datasets,
        samples_per_category=args.samples_per_category, categories=args.categories,
        num_shards=args.num_shards, seed=args.seed, gradient_ids=gradient_ids,
        manifest_sha256=digest(manifest), pro_num_th=args.pro_num_th, pro_max_fpr=args.pro_max_fpr,
        quality_tolerance=args.quality_tolerance, arms=ARMS if arms is None else arms,
        sampling_note="mask-presence stratified; this diagnostic subset is not an independent performance claim",
        selection_note="No target checkpoint/threshold selection; GT is post-hoc scoring only",
        gradient_note="Frozen checkpoint counterfactual post-warmup probe; historical W_GRPO/warmup are not stored",
    )
    if extra_config:
        cfg.update(extra_config)
    if (root/"config.json").exists() and load_json(root/"config.json") != cfg:
        raise ValueError("WORK_DIR has a different config; do not overwrite. Use a new WORK_DIR")
    atomic_json(root/"config.json", cfg)
    atomic_json(root/"manifest.json", manifest)
    print(f"Prepared {len(manifest)} paired samples; checkpoint={ckpt}; "
          f"horizon saved={original_cfg.max_steps} effective={runtime_cfg.max_steps}; config={digest(cfg)}", flush=True)
    if runtime_cfg.max_steps < 2:
        print("WARNING: horizon < 2; no multi-step comparison is possible. "
              "Use STEPS=3 with a new WORK_DIR for an explicitly overridden-horizon diagnosis.", flush=True)


def verify_source(record):
    for path_key, hash_key in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
        if record[path_key] and sha256_file(record[path_key]) != record[hash_key]:
            raise ValueError(f"Source changed: {record[path_key]}")


def cache_ready(root, record, signature):
    path, metadata = root/"evidence"/f"{record['id']}.pt", root/"evidence"/f"{record['id']}.json"
    if not path.exists() or not metadata.exists():
        return False
    check_signature(metadata, signature)
    if sha256_file(path) != load_json(metadata)["sha256"]:
        raise ValueError(f"Corrupted evidence cache: {path}")
    return True


def base_export_runtime(runtime_config):
    # The shared evaluator also accepts CLI gate overrides. Export must leave
    # them neutral so the checkpoint's production settings are not changed.
    # Add them here, not to config.json, so already prepared runs remain usable.
    return argparse.Namespace(**dict(
        runtime_config, gain_safety_margin=None, gain_accept_probability=None,
        disable_hard_gain_gate=False,
    ))


def export(args, protocol=PROTOCOL):
    root, cfg, manifest, signature = load_work(args.work_dir, protocol=protocol)
    assigned = [r for r in manifest if r["shard"] == args.shard_id]
    pending = []
    for record in assigned:
        verify_source(record)
        if not cache_ready(root, record, signature):
            pending.append(record)
    if not pending:
        print(f"Export shard {args.shard_id}: all {len(assigned)} caches already valid", flush=True)
        return
    from Datasets import DATASET_REGISTRY
    from tools.dino_single_tower import collate_anomaly_batch, forward_dino_single_batch
    from tools.mara_evidence import build_mara_evidence
    from tools.utils_up import get_anomaly_map
    from test_mara import build_models_from_checkpoint, load_mara_checkpoint
    artifact = cfg.get("backbone_artifact")
    if artifact and sha256_file(artifact["path"]) != artifact["sha256"]:
        raise ValueError("DINO weights changed since prepare")
    payload = load_torch(cfg["mara_ckpt"])
    # Allow explicitly recorded file relocation, not an architecture change.
    payload = dict(payload, **{k: cfg["runtime"][k] for k in ("dino_repo_dir", "dino_weights")})
    runtime = base_export_runtime(cfg["runtime"])
    device = torch.device(args.device)
    models = build_models_from_checkpoint(runtime, device, payload)
    clip, prompts, dino, dino_adapters, model, agent, single = models
    load_mara_checkpoint(cfg["mara_ckpt"], payload, model, prompts, dino_adapters, agent, device, single)
    if model is not None:
        for name in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter"):
            getattr(model, name).load_state_dict(payload[name], strict=True)
        prompts.load_state_dict(payload["prompt_learner"], strict=True)
    for module in models:
        if isinstance(module, torch.nn.Module):
            module.eval().requires_grad_(False)
    agent.cfg = type(agent.cfg)(**cfg["mara_config"])
    dataset = None
    current_key = None
    def forward_batch(records, samples):
        with torch.no_grad():
            image_info = collate_anomaly_batch(samples)
            if single is not None:
                mask, base, logits, evidence = forward_dino_single_batch(single, image_info, device)
            else:
                _, mask, base, logits, evidence = get_anomaly_map(
                    clip_model=clip, image_info=image_info, device=device, model=model,
                    Dino_model=dino, prompt_learner=prompts, idx=records[0]["index"],
                    visual_backbone=runtime.visual_backbone, visual_layers=runtime.visual_layers,
                    text_source=runtime.text_source, return_evidence=True)
            packed = build_mara_evidence(
                evidence, base, agent.cfg.num_layers, agent.cfg.map_size,
                feature_channels_per_layer=runtime.evidence_feature_channels,
                semantic_spatial_channels=runtime.evidence_semantic_channels,
                semantic_global_dim=runtime.evidence_semantic_global_dim)
            batch_cache = dict(
                base_prob=base.detach().float().cpu(), base_logits=logits.detach().float().cpu(),
                layer_maps=packed["layer_maps"].float().cpu(),
                evidence_maps=packed["extra_maps"].float().cpu() if agent.cfg.evidence_channels else None,
                global_evidence=packed["global_evidence"].float().cpu() if agent.cfg.global_evidence_dim else None,
                mask=(mask > .5).to(torch.uint8).cpu(),
                labels=image_info["is_anomaly"].long().cpu(),
            )
            if base.shape[-2:] != (cfg["image_size"], cfg["image_size"]):
                raise ValueError("Base output does not match prepared image_size")
            if len(base) != len(records):
                raise ValueError("Base returned the wrong batch dimension")
            return [{**{k: v[i:i+1].clone() if v is not None else None for k, v in batch_cache.items()},
                     "record_id": record["id"], "signature": signature} for i, record in enumerate(records)]

    batch_size = int(cfg.get("export_batch_size", 1))
    position = 0
    with ThreadPoolExecutor(max_workers=int(cfg.get("decode_workers", 1))) as decoder:
        while position < len(pending):
            record = pending[position]
            key = (record["dataset"], record["category"])
            if key != current_key:
                dataset_cls, split_cls, _ = DATASET_REGISTRY[record["dataset"]]
                dataset = dataset_cls(source=record["source"], split=split_cls.TEST, classname=record["category"],
                                      resize=cfg["image_size"], imagesize=cfg["image_size"])
                current_key = key
            records = []
            for item in pending[position:position+batch_size]:
                if (item["dataset"], item["category"]) != key:
                    break
                records.append(item)
            samples = list(decoder.map(lambda r: dataset[r["dataset_index"]], records))
            if any(str(Path(s["image_path"]).resolve()) != r["image_path"] for r, s in zip(records, samples)):
                raise ValueError("Dataset ordering changed since prepare")
            caches = None
            try:
                caches = forward_batch(records, samples)
            except torch.cuda.OutOfMemoryError:
                if len(records) == 1:
                    raise
                batch_size = max(1, len(records)//2)
                print(f"Export OOM backoff: batch={batch_size}; precision unchanged", flush=True)
            if caches is None:
                gc.collect()
                torch.cuda.empty_cache()
                continue
            for record, cache in zip(records, caches):
                save_export_cache(root, cfg, record, cache, agent.cfg, signature)
            position += len(records)
            peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.
            print(f"Export {args.shard_id} {position}/{len(pending)} {record['dataset']}/{record['category']} "
                  f"batch={len(records)} peak={peak:.2f}GiB", flush=True)


def save_export_cache(root, cfg, record, cache, agent_cfg, signature):
    validate_cache(cache, agent_cfg)
    if cache["mask"].shape != cache["base_prob"][:, :1].shape or cache["labels"].shape != (1,) or int(cache["labels"][0]) not in (0, 1):
        raise ValueError("Invalid GT geometry")
    path = root/"evidence"/f"{record['id']}.pt"
    atomic_torch(path, cache)
    atomic_json(path.with_suffix(".json"), dict(signature=signature, sha256=sha256_file(path)))


def result_ready(root, record, signature):
    path, maps = root/"per_image"/f"{record['id']}.json", root/"per_image"/f"{record['id']}.npz"
    if not path.exists() or not maps.exists():
        return False
    check_signature(path, signature)
    result = load_json(path)
    if sha256_file(maps) != result["maps_sha256"]:
        raise ValueError(f"Corrupted prediction maps: {maps}")
    metadata = root/"evidence"/f"{record['id']}.json"
    if load_json(metadata)["sha256"] != result["evidence_sha256"]:
        raise ValueError("Prediction maps belong to a different evidence export")
    return True


def diagnose(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    assigned = [r for r in manifest if r["shard"] == args.shard_id]
    payload = load_torch(cfg["mara_ckpt"])
    original_cfg = checkpoint_config(payload)
    agent = MARAAgent(type(original_cfg)(**cfg["mara_config"])).to(args.device)
    agent.load_state_dict(payload["mara_agent"], strict=True)
    agent.eval().requires_grad_(False)
    for count, record in enumerate(assigned):
        if not cache_ready(root, record, signature):
            raise ValueError(f"Missing evidence cache for {record['id']}; run export first")
        if result_ready(root, record, signature):
            print(f"Diagnose {args.shard_id}: reuse {record['id']}", flush=True)
            continue
        cache = load_torch(root/"evidence"/f"{record['id']}.pt")
        if cache["signature"] != signature or cache["record_id"] != record["id"]:
            raise ValueError("Cache identity mismatch")
        validate_cache(cache, agent.cfg)
        inputs = move_inputs(cache, args.device)
        mask, labels = cache["mask"].to(args.device), cache["labels"].to(args.device)
        rows, maps = [], []
        for arm in ARMS:
            trace = trace_infer(agent, inputs, arm)
            scored = score_trace(trace, mask, labels, agent.cfg)
            for row in scored:
                row.update(sample_id=record["id"], dataset=record["dataset"],
                           category=record["category"], arm=arm)
            rows.extend(scored)
            maps.append(np.stack([p[0, 1].cpu().numpy() for p in trace["maps"]]))
            del trace
        probe = gradient_probe(
            agent, inputs, mask.float(), labels, seed=cfg["seed"]+int(stable_id(record["id"], "gradient")[:8], 16)
        ) if record["id"] in cfg["gradient_ids"] else None
        path = root/"per_image"/f"{record['id']}.npz"
        atomic_npz(path, maps=np.stack(maps), base=cache["base_prob"][0, 1].numpy(),
                   mask=cache["mask"][0, 0].numpy(), label=cache["labels"].numpy())
        atomic_json(path.with_suffix(".json"), dict(
            signature=signature, sample_id=record["id"], dataset=record["dataset"],
            category=record["category"], image_label=int(labels.item()), rows=rows, gradient_probe=probe,
            evidence_sha256=load_json(root/"evidence"/f"{record['id']}.json")["sha256"],
            maps_sha256=sha256_file(path)))
        print(f"Diagnose {args.shard_id} {count+1}/{len(assigned)} {record['dataset']}/{record['category']}", flush=True)


def mean(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def collect(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    by_category = defaultdict(list)
    traces, gradients, gradient_trajectories = [], [], []
    for record in manifest:
        if not cache_ready(root, record, signature) or not result_ready(root, record, signature):
            raise ValueError(f"Missing/invalid result {record['id']}; collect never accepts partial runs")
        result = load_json(root/"per_image"/f"{record['id']}.json")
        by_category[(record["dataset"], record["category"])].append(result)
        traces.extend(result["rows"])
        probe = result.get("gradient_probe")
        if probe:
            metadata = dict(sample_id=record["id"], dataset=record["dataset"], category=record["category"])
            gradients.append(dict(metadata, **{k: v for k, v in probe.items()
                                               if k not in ("return_to_go", "advantages", "active_mask", "policy_mask", "base_trajectory_mask", "sampled_ops")}))
            for trajectory in range(probe["group_size"]):
                for step, op in enumerate(probe["sampled_ops"][trajectory]):
                    gradient_trajectories.append(dict(
                        metadata, trajectory=trajectory, step=step, op=op,
                        base_anchor=probe["base_trajectory_mask"][trajectory],
                        active=probe["active_mask"][trajectory][step],
                        policy_mask=probe["policy_mask"][trajectory][step],
                        return_to_go=probe["return_to_go"][trajectory][step],
                        advantage=probe["advantages"][trajectory][step]))
    metrics, stops = [], []
    metric_names = ("PRO", "P_AUROC", "I_AUROC", "F1", "background_FPR_at_05")
    for (dataset, category), results in sorted(by_category.items()):
        arrays = []
        for result in results:
            with np.load(root/"per_image"/f"{result['sample_id']}.npz", allow_pickle=False) as data:
                arrays.append({k: data[k] for k in data.files})
        masks = np.stack([d["mask"] for d in arrays]).astype(bool)
        labels = np.array([int(d["label"][0]) for d in arrays])
        base = np.stack([d["base"] for d in arrays])
        baseline = map_metrics(masks, base, labels, cfg["pro_num_th"], cfg["pro_max_fpr"])
        metrics.append(dict(dataset=dataset, category=category, arm="Base", step=0,
                            images=len(results), diagnostic_only=True, **baseline))
        memo = {hashlib.sha256(base.tobytes()).hexdigest(): baseline}
        for arm_index, arm in enumerate(ARMS):
            rows = [r for result in results for r in result["rows"] if r["arm"] == arm]
            active = [r for r in rows if r["active_before"]]
            attempted = [r for r in active if r["op"] == 1]
            terminal = [r for r in active if r["stop_reason"] not in ("continue", "horizon")]
            stops.append(dict(
                dataset=dataset, category=category, arm=arm, images=len(results),
                active_decisions=len(active), attempted_refinements=len(attempted),
                executed_refinements=sum(r["execute_weight"] > 0 for r in attempted),
                execution_weight_sum=sum(r["execute_weight"] for r in attempted),
                policy_stop_images=sum(r["stop_reason"] == "policy_stop" for r in terminal),
                gain_reject_images=sum(r["stop_reason"].startswith("gain_reject") for r in terminal),
                return_mean=mean([r["return_to_go"] for r in rows if r["step"] == 0]),
                costless_return_mean=mean([r["return_to_go_without_cost"] for r in rows if r["step"] == 0]),
                positive_candidate_fraction=mean([r["candidate_delta_quality_lr"] > 0 for r in active]),
                net_positive_candidate_fraction=mean([r["candidate_net_reward"] > 0 for r in active]),
                cost_flipped_candidate_fraction=mean([
                    r["candidate_delta_quality_lr"] > 0 and r["candidate_net_reward"] <= 0 for r in active]),
                beneficial_candidate_rejected_fraction=mean([
                    r["execute_weight"] == 0 for r in attempted if r["candidate_net_reward"] > cfg["quality_tolerance"]]),
                stop_output_drift_fraction=mean([
                    r["global_change_max"] > cfg["quality_tolerance"] or r["pixel_change_max"] > cfg["quality_tolerance"]
                    for r in rows if not r["active_before"] or r["op"] == 0]),
                vanished_gt_images=sum(
                    r["before_hr_gt_pixels"] > 0 and r["before_lr_gt_pixels"] == 0
                    for r in rows if r["step"] == 0),
            ))
            for step in range(cfg["mara_config"]["max_steps"]):
                prediction = np.stack([d["maps"][arm_index, step] for d in arrays])
                key = hashlib.sha256(prediction.tobytes()).hexdigest()
                if key not in memo:
                    memo[key] = map_metrics(masks, prediction, labels, cfg["pro_num_th"], cfg["pro_max_fpr"])
                score = memo[key]
                step_rows = [r for r in rows if r["step"] == step]
                metrics.append(dict(
                    dataset=dataset, category=category, arm=arm, step=step+1,
                    images=len(results), diagnostic_only=True,
                    quality_gain_hr=mean([r["after_hr_quality"]-result["rows"][0]["before_hr_quality"]
                                         for result in results for r in result["rows"] if r["arm"] == arm and r["step"] == step]),
                    mean_step_reward=mean([r["reward"] for r in step_rows]),
                    **score,
                    **{f"delta_{k}": score[k]-baseline[k] if score[k] is not None and baseline[k] is not None else None for k in metric_names}))
        print(f"Metrics {dataset}/{category}: {len(results)} paired images", flush=True)
    macro = []
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["dataset"], row["arm"], row["step"])].append(row)
    for (dataset, arm, step), values in sorted(groups.items()):
        numerical = [*metric_names, *[f"delta_{k}" for k in metric_names], "quality_gain_hr", "mean_step_reward"]
        macro.append(dict(dataset=dataset, arm=arm, step=step, categories=len(values),
                          images=sum(r["images"] for r in values), diagnostic_only=True,
                          **{k: mean([r.get(k) for r in values]) for k in numerical},
                          **{f"valid_categories_{k}": sum(r.get(k) is not None for r in values) for k in metric_names}))
    comparisons = []
    reference = {(r["dataset"], r["category"], r["step"]): r for r in metrics if r["arm"] == "D0"}
    for row in metrics:
        if row["arm"] in ("D1", "D2", "D3"):
            baseline = reference[(row["dataset"], row["category"], row["step"])]
            comparisons.append(dict(
                dataset=row["dataset"], category=row["category"], arm=row["arm"], step=row["step"],
                **{f"{k}_minus_D0": row[k]-baseline[k] if row[k] is not None and baseline[k] is not None else None
                   for k in metric_names}))
    out = root/"results"
    write_csv(out/"step_trace.csv", traces)
    write_csv(out/"stop_summary.csv", stops)
    write_csv(out/"metrics.csv", metrics)
    write_csv(out/"macro_metrics.csv", macro)
    write_csv(out/"comparisons.csv", comparisons)
    write_csv(out/"gradient_probe.csv", gradients)
    write_csv(out/"gradient_trajectories.csv", gradient_trajectories)
    report = [
        "# MARA early-stop A/B diagnostics", "",
        "DIAGNOSTIC ONLY. No retraining, no checkpoint/threshold selection. Corrected no-op state semantics.",
        f"Checkpoint: {cfg['mara_ckpt']}", f"SHA256: {cfg['checkpoint_sha256']}",
        f"Paired images: {len(manifest)}; steps: {cfg['mara_config']['max_steps']}",
        "All arms retain spatial fusion gates. D1/D3 bypass only the gain acceptance gate.",
        "D2/D3 force REFINE at every still-active step, not merely the first step.",
        "D0 matches corrected production STOP behaviour. Historical v1 runs used the old transition.",
        "GT is post-hoc scoring only. Candidate scores describe policy-selected proposals, not a 64-action oracle.",
        "PRO is the existing sampled-threshold definition, accelerated by sorting. Undefined metrics are blank.",
        "Gradient probes are counterfactual at frozen weights, not evidence of historical W_GRPO or optimizer updates.",
        "", "## Final-step category macro metrics", "",
    ]
    for row in macro:
        if row["step"] == cfg["mara_config"]["max_steps"] or row["arm"] == "Base":
            report.append(f"- {row['dataset']} {row['arm']}: PRO={row['PRO']}, P_AUROC={row['P_AUROC']}, "
                          f"delta_PRO={row['delta_PRO']}, quality_gain_hr={row['quality_gain_hr']}")
    (out/"README.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    atomic_json(root/"complete.json", dict(signature=signature, images=len(manifest),
                                         metrics_sha256=sha256_file(out/"metrics.csv")))
    print(f"Complete: {out}; read README.md, stop_summary.csv and macro_metrics.csv first", flush=True)


def parser(stage_choices=("prepare", "export", "diagnose", "collect")):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=stage_choices)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--mara_ckpt", default="")
    p.add_argument("--datasets", default="visa mvtec btad mpdd")
    p.add_argument("--data_roots", default="{}")
    p.add_argument("--categories", default="")
    p.add_argument("--samples_per_category", type=int, default=8)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=0, help="0 preserves checkpoint horizon; >0 explicitly overrides it")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_images", type=int, default=16)
    p.add_argument("--num_shards", type=int, default=2)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dino_repo_dir", default="")
    p.add_argument("--dino_weights", default="")
    p.add_argument("--clip_model_name", default="ViT-L-14-336")
    p.add_argument("--clip_pretrained", default="openai")
    p.add_argument("--text_source", choices=("prompt_learner", "ensemble"), default="prompt_learner")
    p.add_argument("--pro_num_th", type=int, default=1000)
    p.add_argument("--pro_max_fpr", type=float, default=.3)
    p.add_argument("--quality_tolerance", type=float, default=1e-4)
    p.add_argument("--cpu_threads", type=int, default=4)
    return p


def main():
    args = parser().parse_args()
    if args.num_shards < 1 or args.shard_id < 0 or args.steps < 0 or args.gradient_images < 0 or args.samples_per_category < 0:
        raise ValueError("Invalid nonnegative budget/sharding settings")
    if args.cpu_threads < 1 or args.image_size < 1 or args.pro_num_th < 2 or not 0 < args.pro_max_fpr <= 1:
        raise ValueError("Invalid image/metric/thread settings")
    torch.set_num_threads(args.cpu_threads)
    if args.stage in ("export", "diagnose"):
        _, cfg, _, _ = load_work(args.work_dir)
        if args.shard_id >= cfg["num_shards"]:
            raise ValueError("shard_id outside prepared layout")
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
