#!/usr/bin/env python3
"""Train the language-free DINOv3 single-tower base detector."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Datasets import DATASET_REGISTRY
from tools.dino_single_tower import (
    DinoSingleTowerConfig,
    DinoSingleTowerDetector,
    collate_anomaly_batch,
    create_dino_single_tower,
)
from tools.loss import BinaryDiceLoss, FocalLoss


HFA_PRESETS = {
    "none": (),
    "l5": (5,),
    "l11": (11,),
    "l17": (17,),
    "l23": (23,),
    "hfa1": (23,),
    "hfa2": (17, 23),
    "hfa3": (11, 17, 23),
    "hfa4": (5, 11, 17, 23),
}


def parse_layers(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        layers = tuple(int(item) for item in value)
    if not layers:
        raise argparse.ArgumentTypeError("At least one DINO layer is required.")
    return layers


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
            raise RuntimeError("DINO single-tower DDP training requires CUDA/NCCL.")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
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


def build_loader(args) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset_name = args.dataset.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset={dataset_name}.")
    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    split_key = args.train_split.upper()
    if not hasattr(split_cls, split_key):
        raise ValueError(f"Dataset {dataset_name} does not provide split={args.train_split}.")
    dataset = dataset_cls(
        source=root_path,
        split=getattr(split_cls, split_key),
        classname=args.category,
        resize=args.image_size,
        imagesize=args.image_size,
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
            f"Loaded [{dataset_name}] split={args.train_split}, category={args.category}, "
            f"samples={len(dataset)}, per_gpu_batch={args.batch_size}, "
            f"global_batch={args.batch_size * args.world_size}"
        )
    return loader, sampler


def unwrap_detector(module: torch.nn.Module) -> DinoSingleTowerDetector:
    return module.module if isinstance(module, DDP) else module


def build_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 0.05 + 0.95 * cosine

    return LambdaLR(optimizer, lr_lambda)


def reduce_metrics(metrics: Dict[str, float], device: torch.device, args) -> Dict[str, float]:
    if not args.distributed:
        return metrics
    keys = list(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= args.world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def train_one_epoch(
    epoch: int,
    detector: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args,
) -> Dict[str, float]:
    detector.train()
    # Keep the large foundation backbone deterministic; only the optional HFA
    # adapters and the visual prototype head are trainable.
    unwrap_detector(detector).backbone.eval()
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    meters = {key: [] for key in ("loss", "seg", "aux", "global", "proto", "fp", "dice", "acc")}

    for batch_idx, image_info in enumerate(loader):
        image = image_info["image"].to(device, non_blocking=True)
        mask = (image_info["mask"].to(device, non_blocking=True) > 0.5).float()
        labels = image_info["is_anomaly"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            output = detector(image)
            prob = output["prob"]
            mask_bhw = mask[:, 0] if mask.dim() == 4 else mask
            seg_loss = loss_focal(prob, mask) + loss_dice(prob[:, 1], mask_bhw)

            auxiliary_losses = []
            for layer_logits in output["layer_logits"]:
                target = F.interpolate(mask.float(), size=layer_logits.shape[-2:], mode="nearest")
                target_bhw = target[:, 0]
                layer_prob = torch.softmax(layer_logits, dim=1)
                auxiliary_losses.append(
                    loss_focal(layer_prob, target) + loss_dice(layer_prob[:, 1], target_bhw)
                )
            auxiliary_loss = torch.stack(auxiliary_losses).mean()
            global_loss = F.cross_entropy(output["global_logits"], labels)
            prototype_loss = output["prototype_regularization"]

            normal = (labels == 0).float()
            false_positive = prob[:, 1].mean(dim=(-2, -1))
            if normal.sum() > 0:
                false_positive_loss = (false_positive * normal).sum() / normal.sum()
            else:
                false_positive_loss = false_positive.mean() * 0.0

            total_loss = (
                args.w_seg * seg_loss
                + args.w_aux * auxiliary_loss
                + args.w_global * global_loss
                + args.w_prototype * prototype_loss
                + args.w_false_positive * false_positive_loss
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in detector.parameters() if parameter.requires_grad),
            args.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            intersection = (prob[:, 1] * mask_bhw).flatten(1).sum(dim=1)
            denominator = prob[:, 1].flatten(1).sum(dim=1) + mask_bhw.flatten(1).sum(dim=1)
            dice = ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
            accuracy = (output["global_logits"].argmax(dim=1) == labels).float().mean()
        values = {
            "loss": total_loss,
            "seg": seg_loss,
            "aux": auxiliary_loss,
            "global": global_loss,
            "proto": prototype_loss,
            "fp": false_positive_loss,
            "dice": dice,
            "acc": accuracy,
        }
        for key, value in values.items():
            meters[key].append(float(value.detach().item()))

        if args.is_main_process:
            print(
                f"Epoch {epoch + 1}/{args.epoch} | Batch {batch_idx + 1}/{len(loader)} "
                f"| loss={values['loss'].item():.4f} seg={values['seg'].item():.4f} "
                f"aux={values['aux'].item():.4f} global={values['global'].item():.4f} "
                f"proto={values['proto'].item():.4f} fp={values['fp'].item():.4f} "
                f"dice={values['dice'].item():.4f} acc={values['acc'].item():.4f}",
                end="\r",
                flush=True,
            )

    metrics = {key: float(np.mean(value)) if value else 0.0 for key, value in meters.items()}
    return reduce_metrics(metrics, device, args)


def save_checkpoint(
    detector: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    result_path: str,
    args,
) -> str:
    model = unwrap_detector(detector)
    payload = model.checkpoint_fields()
    payload.update(
        {
            "epoch": epoch,
            "dataset": args.dataset,
            "train_split": args.train_split,
            "image_size": args.image_size,
            "dino_repo_dir": args.dino_repo_dir,
            "dino_model_name": args.dino_model_name,
            "dino_weights": args.dino_weights,
            "optimizer": optimizer.state_dict(),
            "experiment": "dino_single_tower",
        }
    )
    ckpt_dir = os.path.join(result_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"single_epoch_{epoch}.pth")
    torch.save(payload, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./checkpoint/dino_single")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--category", type=str, default="ALL")
    parser.add_argument("--train_split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4, help="per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--visual_layers", type=parse_layers, default=(5, 11, 17, 23))
    parser.add_argument("--dino_repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitl16")
    parser.add_argument(
        "--dino_weights",
        type=str,
        default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--normal_prototypes", type=int, default=4)
    parser.add_argument("--anomaly_prototypes", type=int, default=8)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--evidence_channels", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--topk_ratio", type=float, default=0.01)
    parser.add_argument("--hfa_setting", choices=tuple(HFA_PRESETS), default="none")
    parser.add_argument("--hfa_bottleneck", type=int, default=256)

    parser.add_argument("--w_seg", type=float, default=1.0)
    parser.add_argument("--w_aux", type=float, default=0.25)
    parser.add_argument("--w_global", type=float, default=0.25)
    parser.add_argument("--w_prototype", type=float, default=0.01)
    parser.add_argument("--w_false_positive", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset = args.dataset.lower()
    device = setup_distributed(args)
    set_seed(args.seed + args.rank)
    if args.is_main_process:
        os.makedirs(args.result_path, exist_ok=True)
        if args.train_split.lower() == "test":
            print(
                "[PROTOCOL] Training with labeled source split=test. This is valid only for "
                "cross-dataset evaluation; do not report results on the same source dataset."
            )

    config = DinoSingleTowerConfig(
        visual_layers=tuple(args.visual_layers),
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        normal_prototypes=args.normal_prototypes,
        anomaly_prototypes=args.anomaly_prototypes,
        attention_heads=args.attention_heads,
        evidence_channels=args.evidence_channels,
        temperature=args.temperature,
        topk_ratio=args.topk_ratio,
        hfa_layers=tuple(HFA_PRESETS[args.hfa_setting]),
        hfa_bottleneck=args.hfa_bottleneck,
    )
    detector = create_dino_single_tower(
        config=config,
        repo_dir=args.dino_repo_dir,
        model_name=args.dino_model_name,
        weights=args.dino_weights,
        device=device,
    )
    trainable = [parameter for parameter in detector.parameters() if parameter.requires_grad]
    if args.is_main_process:
        print("Single-tower config:", config.to_dict())
        print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable):,}")
    if args.distributed:
        detector = DDP(
            detector,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    loader, sampler = build_loader(args)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(1, args.epoch * len(loader))
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    try:
        for epoch in range(args.epoch):
            if sampler is not None:
                sampler.set_epoch(epoch)
            start = time.time()
            metrics = train_one_epoch(
                epoch=epoch,
                detector=detector,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                args=args,
            )
            if args.is_main_process:
                checkpoint = save_checkpoint(detector, optimizer, epoch, args.result_path, args)
                print()
                print(
                    f"Epoch {epoch + 1} summary: loss={metrics['loss']:.5f}, "
                    f"dice={metrics['dice']:.5f}, acc={metrics['acc']:.5f}, "
                    f"time={time.time() - start:.1f}s, ckpt={checkpoint}"
                )
            if args.distributed:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
