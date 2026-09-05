#!/usr/bin/env python3
"""Distill cached Qwen3-VL decisions into the compact DINO semantic head."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from Datasets import DATASET_REGISTRY
from tools.dino_single_tower import (
    DinoSingleTowerDetector,
    collate_anomaly_batch,
    config_from_checkpoint,
    create_dino_single_tower,
)
from tools.vlm_decision import ACTION_NAMES, VLMDecision, decision_target_map


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(args) -> torch.device:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = args.world_size > 1
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.rank = 0
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("VLM distillation DDP requires CUDA/NCCL.")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.rank = dist.get_rank()
        device = torch.device("cuda", args.local_rank)
    elif torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    args.is_main_process = args.rank == 0
    return device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def load_cache(path: str) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            teacher = value.get("teacher", {})
            if teacher.get("parse_ok") is True:
                records[path_key(str(value["image_path"]))] = value
            elif teacher.get("parse_ok") not in (False, None):
                raise ValueError(f"Invalid parse_ok at {path}:{line_number}")
    return records


class DistillationDataset(Dataset):
    """Attach VLM pseudo-labels while deliberately hiding all GT supervision."""

    def __init__(self, source: Dataset, cache: Mapping[str, Dict], map_size: int) -> None:
        self.source = source
        self.cache = cache
        self.map_size = int(map_size)
        self.indices = []
        for index, item in enumerate(getattr(source, "data_to_iterate", [])):
            if path_key(str(item[2])) in cache:
                self.indices.append(index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict:
        sample = dict(self.source[self.indices[index]])
        record = self.cache[path_key(str(sample["image_path"]))]
        decision = VLMDecision(**record["teacher"])
        region_mask_decision = VLMDecision(
            image_anomaly_probability=decision.image_anomaly_probability,
            action=decision.action,
            regions=[{**region, "defect_probability": 1.0} for region in decision.regions],
            preferred_layer=decision.preferred_layer,
            confidence=decision.confidence,
            parse_ok=decision.parse_ok,
        )
        sample.pop("mask", None)
        sample.pop("is_anomaly", None)
        sample.update(
            {
                "teacher_map": decision_target_map(decision, self.map_size),
                "teacher_region_mask": decision_target_map(
                    region_mask_decision, self.map_size
                ),
                "teacher_action": ACTION_NAMES.index(decision.action),
                "teacher_layer": decision.preferred_layer,
                "teacher_anomaly": decision.image_anomaly_probability,
                "teacher_confidence": decision.confidence,
            }
        )
        return sample


def create_source_dataset(args):
    dataset_name = args.dataset.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset={dataset_name}.")
    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    split_name = args.train_split.upper()
    if not hasattr(split_cls, split_name):
        raise ValueError(f"Dataset {dataset_name} has no split={args.train_split}.")
    return dataset_cls(
        source=root_path,
        split=getattr(split_cls, split_name),
        classname=args.category,
        resize=args.image_size,
        imagesize=args.image_size,
    )


def build_loader(args) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    source = create_source_dataset(args)
    cache = load_cache(args.cache_path)
    dataset = DistillationDataset(source, cache, args.map_size)
    coverage = len(dataset) / max(1, len(source))
    if coverage < args.min_cache_coverage:
        raise RuntimeError(
            f"Valid VLM cache coverage is {coverage:.3f} ({len(dataset)}/{len(source)}), "
            f"below --min_cache_coverage={args.min_cache_coverage:.3f}."
        )
    sampler = None
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_anomaly_batch,
    )
    if args.is_main_process:
        print(
            f"Loaded VLM distillation cache: valid={len(dataset)}/{len(source)} "
            f"({coverage:.2%}), per_gpu_batch={args.batch_size}, "
            f"global_batch={args.batch_size * args.world_size}"
        )
    return loader, sampler


def checkpoint_runtime(payload: Mapping, args) -> Tuple[str, str, str]:
    return (
        args.dino_repo_dir or str(payload.get("dino_repo_dir", "./dinov3")),
        args.dino_model_name or str(payload.get("dino_model_name", "dinov3_vitl16")),
        args.dino_weights
        or str(payload.get("dino_weights", "./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")),
    )


def create_detector(args, device: torch.device, payload: Mapping) -> DinoSingleTowerDetector:
    config = config_from_checkpoint(payload)
    config.semantic_enabled = False
    repo_dir, model_name, weights = checkpoint_runtime(payload, args)
    detector = create_dino_single_tower(config, repo_dir, model_name, weights, device)
    detector.load_checkpoint_fields(payload, strict=True)
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    semantic_head = detector.enable_semantic_head(args.hidden_dim)
    for parameter in semantic_head.parameters():
        parameter.requires_grad_(True)
    return detector


def unwrap(module: torch.nn.Module) -> DinoSingleTowerDetector:
    return module.module if isinstance(module, DDP) else module


def scheduler_for(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


def corrected_map_target(
    base_map: torch.Tensor,
    region_target: torch.Tensor,
    region_mask: torch.Tensor,
    action: torch.Tensor,
    anomaly: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Turn sparse VLM regions/actions into a conservative dense teacher map."""

    base = F.interpolate(
        base_map[:, 1:2].detach(),
        size=region_target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    target = base.clone()
    confidence_map = confidence[:, None, None, None]
    coverage = region_mask > 0
    target = torch.where(
        coverage,
        (1.0 - confidence_map) * base + confidence_map * region_target,
        target,
    )
    suppress = (action == ACTION_NAMES.index("suppress"))[:, None, None, None]
    target_scale = anomaly[:, None, None, None] / base.flatten(1).amax(dim=1).clamp_min(0.05)[
        :, None, None, None
    ]
    suppressed = base * target_scale.clamp(0.0, 1.0)
    target = torch.where(
        suppress,
        (1.0 - confidence_map) * target + confidence_map * suppressed,
        target,
    )
    return target.clamp(0.0, 1.0)


def reduce_metrics(metrics: Dict[str, float], device: torch.device, args) -> Dict[str, float]:
    if not args.distributed:
        return metrics
    keys = list(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= args.world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def train_epoch(detector, loader, optimizer, scheduler, scaler, device, args) -> Dict[str, float]:
    detector.train()
    model = unwrap(detector)
    model.backbone.eval()
    model.head.eval()
    model.semantic_head.train()
    meters = {name: [] for name in ("loss", "map", "action", "layer", "anomaly", "confidence")}
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        region_target = batch["teacher_map"].to(device, non_blocking=True).float()
        region_mask = batch["teacher_region_mask"].to(device, non_blocking=True).float()
        action = batch["teacher_action"].to(device, non_blocking=True).long()
        layer = batch["teacher_layer"].to(device, non_blocking=True).long()
        anomaly = batch["teacher_anomaly"].to(device, non_blocking=True).float()
        confidence = batch["teacher_confidence"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            output = detector(image)
            decision = output["semantic_decision"]
            map_target = corrected_map_target(
                output["prob"], region_target, region_mask, action, anomaly, confidence
            )
            sample_weight = 0.25 + 0.75 * confidence
            map_per_sample = F.binary_cross_entropy_with_logits(
                decision["map_logits"], map_target, reduction="none"
            ).flatten(1).mean(dim=1)
            map_loss = (map_per_sample * sample_weight).mean()
            action_loss = F.cross_entropy(decision["action_logits"], action)
            layer_loss = F.cross_entropy(decision["layer_logits"], layer)
            anomaly_loss = F.binary_cross_entropy_with_logits(
                decision["anomaly_logit"].flatten(), anomaly
            )
            confidence_loss = F.binary_cross_entropy_with_logits(
                decision["confidence_logit"].flatten(), confidence
            )
            total = (
                args.w_map * map_loss
                + args.w_action * action_loss
                + args.w_layer * layer_loss
                + args.w_anomaly * anomaly_loss
                + args.w_confidence * confidence_loss
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.semantic_head.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        values = {
            "loss": total,
            "map": map_loss,
            "action": action_loss,
            "layer": layer_loss,
            "anomaly": anomaly_loss,
            "confidence": confidence_loss,
        }
        for key, value in values.items():
            meters[key].append(float(value.detach().item()))
        if args.is_main_process:
            print(
                f"Batch {batch_index + 1}/{len(loader)} | "
                + " ".join(f"{key}={value.item():.4f}" for key, value in values.items()),
                end="\r",
                flush=True,
            )
    metrics = {key: float(np.mean(value)) if value else 0.0 for key, value in meters.items()}
    return reduce_metrics(metrics, device, args)


def save_checkpoint(detector, optimizer, epoch: int, args, base_payload: Mapping) -> str:
    model = unwrap(detector)
    payload = model.checkpoint_fields()
    payload.update(
        {
            "epoch": epoch,
            "dataset": args.dataset,
            "train_split": args.train_split,
            "image_size": args.image_size,
            "dino_repo_dir": args.dino_repo_dir or base_payload.get("dino_repo_dir"),
            "dino_model_name": args.dino_model_name or base_payload.get("dino_model_name"),
            "dino_weights": args.dino_weights or base_payload.get("dino_weights"),
            "optimizer": optimizer.state_dict(),
            "experiment": "qwen3_vl_fp8_decision_distillation",
            "vlm_teacher_model": args.teacher_model,
            "vlm_cache_path": args.cache_path,
        }
    )
    ckpt_dir = os.path.join(args.result_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"vlm_distilled_epoch_{epoch}.pth")
    torch.save(payload, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="visa")
    parser.add_argument("--category", default="ALL")
    parser.add_argument("--train_split", default="test")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--map_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4, help="per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--min_cache_coverage", type=float, default=0.95)
    parser.add_argument("--teacher_model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dino_repo_dir", default="")
    parser.add_argument("--dino_model_name", default="")
    parser.add_argument("--dino_weights", default="")
    parser.add_argument("--w_map", type=float, default=1.0)
    parser.add_argument("--w_action", type=float, default=0.50)
    parser.add_argument("--w_layer", type=float, default=0.25)
    parser.add_argument("--w_anomaly", type=float, default=0.50)
    parser.add_argument("--w_confidence", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset = args.dataset.lower()
    device = setup_distributed(args)
    set_seed(args.seed + args.rank)
    payload = torch.load(args.base_ckpt, map_location="cpu")
    detector = create_detector(args, device, payload)
    trainable = [parameter for parameter in detector.parameters() if parameter.requires_grad]
    if args.is_main_process:
        os.makedirs(args.result_path, exist_ok=True)
        print(f"Semantic decision-head trainable parameters: {sum(p.numel() for p in trainable):,}")
        print("Ground-truth masks/classes are not used in VLM distillation.")
    if args.distributed:
        detector = DDP(
            detector,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    loader, sampler = build_loader(args)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epoch * len(loader))
    scheduler = scheduler_for(
        optimizer,
        max(1, int(total_steps * args.warmup_ratio)),
        total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    try:
        for epoch in range(args.epoch):
            if sampler is not None:
                sampler.set_epoch(epoch)
            start = time.time()
            metrics = train_epoch(detector, loader, optimizer, scheduler, scaler, device, args)
            if args.is_main_process:
                checkpoint = save_checkpoint(detector, optimizer, epoch, args, payload)
                print()
                print(
                    f"Epoch {epoch + 1}/{args.epoch}: "
                    + " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
                    + f" time={time.time() - start:.1f}s ckpt={checkpoint}"
                )
            if args.distributed:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
