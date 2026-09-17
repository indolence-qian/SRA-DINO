#!/usr/bin/env python3
"""Frozen dual tower + full-coverage VLM descriptions + supervised dense head.

Stages run in separate processes/environments. VLM inference never receives masks.
The old Base, downstream VLM experiments and MARA entry points are untouched.
"""
import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from tools.vlm_review import atomic_json, digest, file_digest, load_json
from tools.upstream_localization import (
    PROTOCOL, LocalSemanticHead, parse_tile, segmentation_loss,
    source_partition, tile_boxes, tile_prompt,
)

REPO = Path(__file__).resolve().parent
CODE_FILES = ("upstream_localization.py", "tools/upstream_localization.py", "dual_vlm.py",
              "tools/utils_up.py", "tools/semantic_anchor.py", "tools/vlm_decision.py")


def cfg_load(root):
    cfg = load_json(root / "config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong experiment protocol")
    for name, expected in cfg["code"].items():
        if file_digest(REPO / name) != expected:
            raise ValueError(f"Code changed: {name}; use a NEW WORK_DIR")
    if digest(load_json(root / "manifest.json")) != cfg["manifest_sha"]:
        raise ValueError("Manifest was changed")
    return cfg


def records_for(root, partition=None):
    rows = load_json(root / "manifest.json")
    return [r for r in rows if partition is None or r["partition"] == partition]


def source_unchanged(record, include_mask=False):
    for key, sha in (("image_path", "image_sha"), ("mask_path", "mask_sha")):
        if key == "mask_path" and not include_mask:
            continue
        if record[key] and file_digest(record[key]) != record[sha]:
            raise ValueError(f"Changed source file for {record['id']}: {key}")


def prepare(args):
    from dual_vlm import resolve_checkpoint, check_dual_payload, model_signature
    from Datasets import DATASET_REGISTRY, DATASET_CLASSES
    root = Path(args.work_dir)
    supported = {"visa", "mvtec", "btad", "mpdd"}
    targets = args.eval_datasets.split(",")
    if (args.source not in supported or not set(targets) <= supported or
            args.source in targets or len(targets) != len(set(targets))):
        raise ValueError("Use one source and distinct held-out target datasets from visa/mvtec/btad/mpdd")
    if not .05 <= args.val_fraction <= .5 or args.limit_per_category < 0 or args.num_shards < 1:
        raise ValueError("Invalid split/limit/shard settings")
    if args.image_size < 64 or args.image_size % 16:
        raise ValueError("image_size must be >=64 and divisible by DINO patch size 16")
    if not 0 <= args.max_invalid_ratio < 1 or args.retries < 0:
        raise ValueError("Invalid parser settings")
    boxes = tile_boxes(args.tile_grid, args.tile_fraction)
    base = resolve_checkpoint(args.base_ckpt)
    check_dual_payload(torch.load(base, map_location="cpu", weights_only=False))
    names = ("source", "eval_datasets", "val_fraction", "limit_per_category", "seed", "num_shards",
             "image_size", "tile_grid", "tile_fraction", "gpu_memory", "max_model_len", "max_tokens",
             "teacher_image_size", "retries", "max_invalid_ratio", "dino_repo_dir", "dino_model_name",
             "dino_weights", "hfa_setting", "dino_bottleneck", "clip_model_name", "clip_pretrained", "visual_layers")
    cfg = {k: getattr(args, k) for k in names}
    cfg.update(protocol=PROTOCOL, base_ckpt=str(base), base_sha=file_digest(base),
               model=model_signature(args.model_id), boxes=boxes,
               source_root=str(Path(args.source_root).resolve()) if args.source_root else "registry",
               target_root=str(Path(args.target_root).resolve()) if args.target_root else "registry",
               mask_resize="PIL bilinear; VisA >0, other datasets >127 (legacy get_anomaly_map convention)",
               code={p: file_digest(REPO / p) for p in CODE_FILES})
    if args.target_root and len(targets) != 1:
        raise ValueError("--target_root requires a single eval dataset")
    if not 0 < args.max_tokens < args.max_model_len or not 0 < args.gpu_memory < 1 or args.teacher_image_size < 32:
        raise ValueError("Invalid VLM limits")
    if not Path(args.dino_weights).is_file():
        raise FileNotFoundError(f"DINO weights not found: {args.dino_weights}")
    cfg["dino_sha"] = file_digest(args.dino_weights)
    cfg["clip_local_sha"] = file_digest(args.clip_pretrained) if Path(args.clip_pretrained).is_file() else None
    rows = []
    for name in [args.source] + targets:
        cls, splits, default_root = DATASET_REGISTRY[name]
        override = args.source_root if name == args.source else args.target_root
        for cat in sorted(DATASET_CLASSES[name]):
            dataset = cls(source=override or default_root, classname=cat, split=splits.TEST,
                          resize=args.image_size, imagesize=args.image_size)
            entries = list(dataset.data_to_iterate)
            # Subsampling is by path hash, independent of target labels/scores.
            entries.sort(key=lambda e: digest([args.seed, name, cat, str(e[2])]))
            if args.limit_per_category:
                entries = entries[:args.limit_per_category]
            for _, _, image, mask in entries:
                path = str(Path(image).resolve())
                mask = str(Path(mask).resolve()) if mask else None
                rows.append(dict(id=digest([name, cat, path])[:24], dataset=name, category=cat,
                                 image_path=path, image_sha=file_digest(path), mask_path=mask,
                                 mask_sha=file_digest(mask) if mask else None,
                                 partition="source" if name == args.source else "eval"))
    source_partition([r for r in rows if r["partition"] == "source"], args.val_fraction, args.seed)
    seen = {}
    for r in rows:
        prior = seen.setdefault(r["image_sha"], r["partition"])
        if prior != r["partition"]:
            raise ValueError("Identical image content crosses train/val/target partitions; fix data before running")
    counts = Counter(r["partition"] for r in rows)
    if any(counts[p] == 0 for p in ("train", "val", "eval")):
        raise ValueError("An experiment partition is empty")
    cfg["manifest_sha"] = digest(rows)
    if (root / "config.json").exists() and load_json(root / "config.json") != cfg:
        raise ValueError("Settings/data/code changed; use NEW WORK_DIR. Old experiment was not overwritten.")
    if (root / "manifest.json").exists() and load_json(root / "manifest.json") != rows:
        raise ValueError("An existing manifest belongs to another experiment; use NEW WORK_DIR")
    atomic_json(root / "manifest.json", rows)
    atomic_json(root / "config.json", cfg)
    print(f"Prepared {dict(counts)}; {len(rows)*len(boxes)} VLM requests before retries", flush=True)
    print("SUPERVISED SOURCE: source TEST masks train the head; source val may have been seen by old Base.", flush=True)
    print("Only held-out target is a generalization evaluation. VLM never receives labels. NO MARA.", flush=True)


def review(args):
    from tools.vlm_decision import QwenVLLMTeacher
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    check_shard(args, cfg)
    fp = digest(cfg)
    teacher = None
    if (root / "sealed.json").exists():
        # Accepted invalid/unknown observations are also immutable after training starts.
        for r in records_for(root)[args.shard_id::cfg["num_shards"]]:
            _, sha = reviews_for(root, r, cfg)
            meta = load_json(root / "features" / f"{r['id']}.json")
            if meta != dict(fingerprint=fp, review_sha=sha):
                raise ValueError("Sealed semantic cache changed; use NEW WORK_DIR")
        print("Semantic cache already sealed; no regeneration on resume", flush=True)
        return
    for record in records_for(root)[args.shard_id::cfg["num_shards"]]:
        source_unchanged(record)
        with Image.open(record["image_path"]) as im:
            native = im.convert("RGB")
        overview = native.copy()
        overview.thumbnail((cfg["teacher_image_size"], cfg["teacher_image_size"]), Image.Resampling.LANCZOS)
        for tile_id, box in enumerate(cfg["boxes"]):
            path = root / "reviews" / record["id"] / f"{tile_id}.json"
            old = load_json(path) if path.exists() else None
            if old:
                if old["fingerprint"] != fp:
                    raise ValueError("Stale VLM review")
                if old["parsed"]["parse_ok"]:
                    continue
            if teacher is None:
                from dual_vlm import model_signature
                if model_signature(cfg["model"]["path"]) != cfg["model"]:
                    raise ValueError("VLM weights/config changed")
                teacher = QwenVLLMTeacher(model_id=cfg["model"]["path"], gpu_memory_utilization=cfg["gpu_memory"],
                                         max_model_len=cfg["max_model_len"], max_tokens=cfg["max_tokens"],
                                         max_images=2, teacher_image_size=cfg["teacher_image_size"], min_pixels=1024)
            x1, y1, x2, y2 = box
            pixel_box = (int(x1*native.width), int(y1*native.height),
                         int(np.ceil(x2*native.width)), int(np.ceil(y2*native.height)))
            crop = native.crop(pixel_box)
            attempts = old.get("attempts", []) if old else []
            for _ in range(cfg["retries"] + 1):
                trace = root / "traces" / record["id"] / str(tile_id) / str(len(attempts))
                raw = teacher.generate(tile_prompt(record["category"], tile_id), [overview, crop], trace_dir=trace)
                parsed = parse_tile(raw, tile_id)
                attempts.append(dict(raw=raw, parsed=parsed, trace=str(trace.relative_to(root))))
                atomic_json(path, dict(fingerprint=fp, box=box, native_box=pixel_box,
                                       parsed=parsed, attempts=attempts))
                if parsed["parse_ok"]:
                    break
        print(f"review shard={args.shard_id} image={record['id']}", flush=True)


def check_shard(args, cfg):
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard")


def reviews_for(root, record, cfg):
    out, hashes = [], []
    for i in range(len(cfg["boxes"])):
        path = root / "reviews" / record["id"] / f"{i}.json"
        r = load_json(path)
        if r["fingerprint"] != digest(cfg):
            raise ValueError("Review fingerprint mismatch")
        if (not r["attempts"] or r["parsed"] != parse_tile(r["attempts"][-1]["raw"], i)
                or r["box"] != cfg["boxes"][i]):
            raise ValueError("Review no longer agrees with its raw response or tile geometry")
        out.append(r["parsed"])
        hashes.append(file_digest(path))
    return out, digest(hashes)


def audit_reviews(args):
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    stats = defaultdict(Counter)
    for r in records_for(root):
        reviews, _ = reviews_for(root, r, cfg)
        for v in reviews:
            stats[r["partition"]]["total"] += 1
            stats[r["partition"]]["invalid"] += not v["parse_ok"]
            stats[r["partition"]]["semantic_valid"] += v["semantic_valid"]
            stats[r["partition"]][v.get("status", "parse_error")] += 1
    atomic_json(root / "review_audit.json", stats)
    for part, s in stats.items():
        print(part, dict(s), flush=True)
        if s["invalid"] / s["total"] > cfg["max_invalid_ratio"]:
            raise ValueError("Too many malformed responses; resume review (valid abstentions are not invalid)")
        if not s["semantic_valid"]:
            raise ValueError(f"No usable semantics in {part}; do not train an alleged VLM model with zero evidence")


def export(args, probe=False):
    from dual_vlm import atomic_npz, load_dual
    from tools.utils_up import get_anomaly_map
    from tools.semantic_anchor import _transform_text_embeddings
    from CLIP.tokenizer import tokenize
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    check_shard(args, cfg)
    if file_digest(cfg["base_ckpt"]) != cfg["base_sha"]:
        raise ValueError("Base weights changed")
    if file_digest(cfg["dino_weights"]) != cfg["dino_sha"]:
        raise ValueError("DINO weights changed")
    if cfg.get("clip_local_sha") and file_digest(cfg["clip_pretrained"]) != cfg["clip_local_sha"]:
        raise ValueError("CLIP weights changed")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    clip, prompt, dino, adapter, layers = load_dual(cfg, device)
    captured = {}
    hooks = []
    def capture(i):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[i] = F.normalize(value.detach().float(), dim=-1)
        return hook
    for i in range(len(layers)):
        hooks.append(adapter.patch_token_adapter[i].register_forward_hook(capture(i)))
    try:
        selected = records_for(root)[:1] if probe else records_for(root)[args.shard_id::cfg["num_shards"]]
        for record in selected:
            source_unchanged(record)
            reviews, review_sha = ([dict(semantic_valid=False)] * len(cfg["boxes"]), "probe") if probe else reviews_for(root, record, cfg)
            path = root / "features" / f"{record['id']}.npz"
            meta_path = path.with_suffix(".json")
            expected = dict(fingerprint=digest(cfg), review_sha=review_sha)
            if not probe and path.exists() and meta_path.exists() and load_json(meta_path) == expected:
                continue
            if not probe and (root / "sealed.json").exists():
                raise ValueError("Cannot regenerate a sealed feature cache; use NEW WORK_DIR")
            with Image.open(record["image_path"]) as im:
                rgb = np.asarray(im.convert("RGB").resize((cfg["image_size"],)*2, Image.Resampling.BILINEAR)).copy()
            image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
            image = (image - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            info = dict(image=image[None], image_path=[record["image_path"]],
                        mask=torch.zeros(1, 1, cfg["image_size"], cfg["image_size"]))
            captured.clear()
            with torch.inference_mode():
                _, _, base, _, _ = get_anomaly_map(clip, info, device, adapter, dino, prompt, 1,
                                                   visual_layers=layers, return_evidence=True)
                if len(captured) != len(layers):
                    raise ValueError("Failed to capture all frozen adapted DINO feature layers")
                features = torch.stack([captured[i][0] for i in range(len(layers))])
                side = int(features.shape[1] ** .5)
                if side * side != features.shape[1]:
                    raise ValueError("Non-square DINO patch grid")
                features = features.permute(0, 2, 1).reshape(len(layers), -1, side, side)
                texts, valid, truncations = [], [], 0
                for v in reviews:
                    for key in ("observation", "normal_expectation"):
                        text = v.get(key, "") if v["semantic_valid"] else ""
                        texts.append(text)
                        valid.append(bool(text))
                        if text:
                            try:
                                tokenize([text])
                            except RuntimeError:
                                truncations += 1
                ids = tokenize(texts, truncate=True).to(device)
                embeddings = _transform_text_embeddings(clip, clip.token_embedding(ids), ids)
                embeddings = embeddings * torch.tensor(valid, device=device)[:, None]
                embeddings = embeddings.reshape(len(reviews), 2, -1)
            if features.shape[1] != embeddings.shape[-1] or not torch.isfinite(base).all() or not torch.isfinite(features).all():
                raise ValueError("Frozen dual tower returned incompatible/nonfinite features")
            if probe:
                print(f"Preflight passed: frozen Base forward, feature={tuple(features.shape)}, text_dim={embeddings.shape[-1]}", flush=True)
                continue
            atomic_npz(path, visual=features.cpu().numpy().astype(np.float16),
                       base=base[0, 1:2].float().cpu().numpy(),
                       embeddings=embeddings.cpu().numpy().astype(np.float16),
                       valid=np.array(valid, np.float32).reshape(-1, 2),
                       boxes=np.array(cfg["boxes"], np.float32), truncated_texts=np.array(truncations))
            atomic_json(meta_path, expected)
            print(f"export shard={args.shard_id} image={record['id']} feature={tuple(features.shape)}", flush=True)
    finally:
        for h in hooks:
            h.remove()


def seal(args):
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    audit_reviews(args)
    files, shapes, truncations = {}, set(), 0
    for r in records_for(root):
        path = root / "features" / f"{r['id']}.npz"
        _, sha = reviews_for(root, r, cfg)
        if load_json(path.with_suffix(".json")) != dict(fingerprint=digest(cfg), review_sha=sha):
            raise ValueError("Feature export is stale; rerun export before sealing")
        with np.load(path) as data:
            for key in ("visual", "base", "embeddings", "valid", "boxes"):
                if not np.isfinite(data[key]).all():
                    raise ValueError("Nonfinite feature export")
            shape = data["visual"].shape
            shapes.add(shape)
            if (data["base"].shape != (1, cfg["image_size"], cfg["image_size"]) or
                    data["embeddings"].shape != (len(cfg["boxes"]), 2, shape[1])):
                raise ValueError("Feature/text/base dimensions mismatch")
            truncations += int(data["truncated_texts"])
        files[r["id"]] = file_digest(path)
    if len(shapes) != 1:
        raise ValueError("Inconsistent feature shapes")
    value = dict(fingerprint=digest(cfg), features=files, shape=list(next(iter(shapes))), truncated_texts=truncations)
    if (root / "sealed.json").exists() and load_json(root / "sealed.json") != value:
        raise ValueError("Sealed features changed; use NEW WORK_DIR (trained heads cannot reuse modified semantics)")
    atomic_json(root / "sealed.json", value)
    print(f"Sealed {len(files)} images, truncated texts={truncations}", flush=True)


def mask_for(r, size):
    if not r["mask_path"]:
        return torch.zeros(1, size, size)
    with Image.open(r["mask_path"]) as im:
        threshold = 0 if r["dataset"] == "visa" else 127
        arr = np.asarray(im.convert("L").resize((size, size), Image.Resampling.BILINEAR)) > threshold
    return torch.from_numpy(arr.copy()).float()[None]


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, root, cfg, rows, verify=True):
        self.root, self.cfg, self.rows = root, cfg, rows
        sealed = load_json(root / "sealed.json")
        if sealed["fingerprint"] != digest(cfg):
            raise ValueError("Stale cache seal")
        if verify:
            for r in rows:
                source_unchanged(r, include_mask=True)
                if file_digest(root / "features" / f"{r['id']}.npz") != sealed["features"][r["id"]]:
                    raise ValueError("Feature cache modified after sealing")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with np.load(self.root / "features" / f"{r['id']}.npz") as data:
            batch = {k: torch.from_numpy(data[k].astype(np.float32))
                     for k in ("visual", "base", "embeddings", "valid", "boxes")}
        batch["mask"] = mask_for(r, self.cfg["image_size"])
        return batch


def head_forward(head, batch, mode):
    return head(*(batch[k] for k in ("visual", "base", "embeddings", "valid", "boxes")), use_semantics=mode == "vlm")


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def train(args):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.hidden % 8:
        raise ValueError("Invalid head hyperparameters")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        train_data = FeatureDataset(root, cfg, records_for(root, "train"))
        val_data = FeatureDataset(root, cfg, records_for(root, "val")[rank::world])
        if len(train_data) < world:
            raise ValueError("Fewer training images than DDP workers")
        sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=cfg["seed"]) if world > 1 else None
        generator = torch.Generator().manual_seed(cfg["seed"])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                                  num_workers=args.workers, pin_memory=device.type == "cuda", generator=generator)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
        shape = load_json(root / "sealed.json")["shape"]
        model = LocalSemanticHead(shape[0], shape[1], args.hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        out = root / "heads" / args.mode
        settings = dict(cache_sha=file_digest(root / "sealed.json"), mode=args.mode, hidden=args.hidden,
                        epochs=args.epochs, batch_size=args.batch_size, world_size=world, lr=args.lr,
                        loss="BCE+positive-image-Dice", checkpoint_selection="source_val_loss_only")
        start, best, history = 0, float("inf"), []
        if (out / "last.pt").exists():
            saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
            if saved["settings"] != settings:
                raise ValueError("Head settings changed; use a new work directory")
            model.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            start, best, history = saved["epoch"] + 1, saved["best"], saved["history"]
        raw_model = model
        if world > 1:
            # Visual-only keeps identical architecture but does not use text content.
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        for epoch in range(start, args.epochs):
            if sampler:
                sampler.set_epoch(epoch)
            generator.manual_seed(cfg["seed"] + epoch)
            model.train()
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            for batch in train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                losses = segmentation_loss(head_forward(model, batch, args.mode), batch["mask"])
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("Nonfinite training loss")
                losses.mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                totals[0] += losses.detach().sum()
                totals[1] += len(losses)
            raw_model.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = segmentation_loss(head_forward(raw_model, batch, args.mode), batch["mask"])
                    totals[2] += loss.sum()
                    totals[3] += len(loss)
                    totals[4] += segmentation_loss(torch.logit(batch["base"].clamp(1e-5, 1-1e-5)), batch["mask"]).sum()
            if world > 1:
                dist.all_reduce(totals)
            train_loss, val_loss = float(totals[0]/totals[1]), float(totals[2]/totals[3])
            if not np.isfinite(val_loss):
                raise FloatingPointError("Nonfinite validation loss")
            improved = val_loss < best
            best = min(best, val_loss)
            history.append(dict(epoch=epoch + 1, train_loss=train_loss, source_val_loss=val_loss,
                                frozen_base_source_val_loss=float(totals[4]/totals[3])))
            if rank == 0:
                state = dict(state_dict=raw_model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                             best=best, history=history, settings=settings, shape=shape)
                if improved:
                    save_torch(out / "best.pt", state)
                save_torch(out / "last.pt", state)
                atomic_json(out / "history.json", history)
                print(f"{args.mode} epoch={epoch+1}/{args.epochs} train={train_loss:.6f} source_val={val_loss:.6f} best={best:.6f}", flush=True)
            if world > 1:
                dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def evaluate(args):
    from tools.vlm_next_diagnostics import map_metrics
    from torch.utils.data import DataLoader
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    check_shard(args, cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, identities = {}, {}
    for mode in args.head_modes.split(","):
        if mode not in ("visual", "vlm"):
            raise ValueError("Unknown head mode")
        path = root / "heads" / mode / "best.pt"
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["settings"]["cache_sha"] != file_digest(root / "sealed.json"):
            raise ValueError("Head/cache mismatch")
        model = LocalSemanticHead(saved["shape"][0], saved["shape"][1], saved["settings"]["hidden"]).to(device)
        model.load_state_dict(saved["state_dict"])
        models[mode] = model.eval()
        identities[mode] = file_digest(path)
    groups = defaultdict(list)
    for r in records_for(root, "eval"):
        groups[(r["dataset"], r["category"])].append(r)
    for name, cat in sorted(groups)[args.shard_id::cfg["num_shards"]]:
        data = FeatureDataset(root, cfg, groups[(name, cat)])
        masks, predictions = [], defaultdict(list)
        with torch.inference_mode():
            for batch in DataLoader(data, batch_size=args.batch_size, num_workers=args.workers):
                masks.extend(batch["mask"][:, 0].numpy().astype(bool))
                predictions["base"].extend(batch["base"][:, 0].numpy())
                batch = {k: v.to(device) for k, v in batch.items() if k != "mask"}
                for mode, model in models.items():
                    p = head_forward(model, batch, mode).sigmoid()
                    predictions[mode].extend(p[:, 0].cpu().numpy())
                if "vlm" in models:
                    p = head_forward(models["vlm"], batch, "zero_semantics").sigmoid()
                    predictions["vlm_zero_semantics_DIAGNOSTIC"].extend(p[:, 0].cpu().numpy())
        rows = []
        masks = np.asarray(masks)
        base = np.asarray(predictions["base"])
        for mode, maps in predictions.items():
            maps = np.asarray(maps)
            values = map_metrics(masks, maps)
            import cv2
            small_total = small_hit = base_miss = recovered = lost = 0
            for gt, prob, old in zip(masks, maps, base):
                n, cc = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
                for i in range(1, n):
                    region = cc == i
                    if region.sum() > gt.size * .001:
                        continue
                    small_total += 1
                    hit = (prob[region] >= .5).mean() >= .1
                    old_hit = (old[region] >= .5).mean() >= .1
                    small_hit += hit
                    base_miss += not old_hit
                    recovered += hit and not old_hit
                    lost += old_hit and not hit
            bg_count, gt_count = int((~masks).sum()), int(masks.sum())
            rows.append(dict(dataset=name, category=cat, mode=mode, samples=len(data), **values,
                             small_components=int(small_total), small_hits_at_05=int(small_hit),
                             base_small_misses_at_05=int(base_miss), recovered_small_at_05=int(recovered),
                             lost_small_at_05=int(lost),
                             background_FPR_at_05=float(((maps >= .5) & ~masks).sum()/max(1, bg_count)),
                             pixel_recall_at_05=float(((maps >= .5) & masks).sum()/max(1, gt_count)),
                             changed_fraction=float(np.mean(np.abs(maps-base) > 1e-6))))
        atomic_json(root / "results" / f"{name}_{cat}.json",
                    dict(fingerprint=digest(cfg), head_identities=identities, rows=rows))
        print(f"evaluated {name}/{cat}", flush=True)


def report(args):
    root = Path(args.work_dir)
    cfg = cfg_load(root)
    identities = {m: file_digest(root / "heads" / m / "best.pt") for m in args.head_modes.split(",")}
    groups = sorted({(r["dataset"], r["category"]) for r in records_for(root, "eval")})
    rows = []
    for name, cat in groups:
        value = load_json(root / "results" / f"{name}_{cat}.json")
        if value["fingerprint"] != digest(cfg) or value["head_identities"] != identities:
            raise ValueError("Missing/stale target evaluation; rerun evaluate")
        rows.extend(value["rows"])
    for name in sorted({n for n, _ in groups}):
        for mode in sorted({r["mode"] for r in rows if r["dataset"] == name}):
            selected = [r for r in rows if r["dataset"] == name and r["mode"] == mode]
            avg = dict(dataset=name, category="MEAN", mode=mode, samples=sum(r["samples"] for r in selected))
            for k in selected[0]:
                if k in avg:
                    continue
                values = [r[k] for r in selected if r[k] is not None]
                avg[k] = (float(np.mean(values)) if values else None)
                if k.startswith(("small_", "base_small_", "recovered_small_", "lost_small_")):
                    avg[k] = int(sum(values))
            rows.append(avg)
            print(avg, flush=True)
    path = root / "results" / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(root / "results" / "summary.json", rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["prepare", "preflight", "review", "audit", "export", "seal", "train", "evaluate", "report"])
    p.add_argument("--work_dir", required=True)
    for key, default in dict(base_ckpt="", model_id="", source="visa", eval_datasets="mvtec", source_root="", target_root="",
                             dino_repo_dir="./dinov3", dino_model_name="dinov3_vitl16", dino_weights="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                             hfa_setting="hfa3", clip_model_name="ViT-L-14-336", clip_pretrained="openai", visual_layers="5,11,17,23",
                             head_modes="visual,vlm").items():
        p.add_argument("--" + key, default=default)
    for key, default in dict(seed=42, num_shards=2, shard_id=0, image_size=512, tile_grid=3, limit_per_category=0,
                             max_model_len=4096, max_tokens=256, teacher_image_size=512, retries=1, dino_bottleneck=256,
                             epochs=15, batch_size=4, workers=2, hidden=96).items():
        p.add_argument("--" + key, type=int, default=default)
    for key, default in dict(val_fraction=.2, tile_fraction=.4, gpu_memory=.7, max_invalid_ratio=.05, lr=1e-4).items():
        p.add_argument("--" + key, type=float, default=default)
    p.add_argument("--mode", choices=["visual", "vlm"], default="vlm")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    dict(prepare=prepare, preflight=lambda a: export(a, probe=True), review=review, audit=audit_reviews, export=export, seal=seal,
         train=train, evaluate=evaluate, report=report)[arguments.stage](arguments)
