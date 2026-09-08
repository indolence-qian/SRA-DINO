#!/usr/bin/env python3
"""Build or merge a sharded Qwen3-VL anomaly-decision teacher cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from PIL import Image

from Datasets import DATASET_REGISTRY
from tools.dino_single_tower import (
    config_from_checkpoint,
    create_dino_single_tower,
)
from tools.vlm_decision import (
    PROMPT_VERSION,
    QwenVLLMTeacher,
    build_vlm_prompt,
    candidate_boxes_from_map,
    crop_regions,
    heatmap_overlay,
    parse_vlm_decision,
)


def shard_path(output_path: str, shard_id: int) -> str:
    path = Path(output_path)
    suffix = path.suffix or ".jsonl"
    return str(path.with_name(f"{path.stem}.shard{shard_id}{suffix}"))


def iter_jsonl(path: str) -> Iterable[Dict]:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(value, dict):
                yield value


def merge_shards(output_path: str, num_shards: int) -> None:
    merged: Dict[str, Dict] = {}
    for shard_id in range(num_shards):
        path = shard_path(output_path, shard_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing VLM cache shard: {path}")
        for record in iter_jsonl(path):
            merged[str(record["image_path"])] = record
    ordered = sorted(merged.values(), key=lambda item: int(item.get("dataset_index", 0)))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    invalid = sum(not bool(item.get("teacher", {}).get("parse_ok")) for item in ordered)
    print(f"Merged {len(ordered)} teacher records -> {output_path} (invalid={invalid})")


def create_dataset(args):
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


def reference_bank(dataset) -> Mapping[str, List[str]]:
    bank: Dict[str, List[str]] = {}
    for item in getattr(dataset, "data_to_iterate", []):
        category, anomaly, image_path = item[:3]
        if str(anomaly).lower() in {"normal", "good", "ok"}:
            bank.setdefault(str(category), []).append(str(image_path))
    return bank


def choose_reference(bank: Mapping[str, List[str]], category: str, image_path: str) -> str:
    candidates = bank.get(category) or [image_path]
    seed = sum((index + 1) * ord(char) for index, char in enumerate(image_path))
    reference = candidates[seed % len(candidates)]
    if reference == image_path and len(candidates) > 1:
        reference = candidates[(seed + 1) % len(candidates)]
    return reference


def limit_image_size(image: Image.Image, longest_edge: int) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= longest_edge:
        return image
    output = image.copy()
    output.thumbnail((longest_edge, longest_edge), Image.Resampling.LANCZOS)
    return output


def checkpoint_runtime(payload: Mapping, args) -> Tuple[str, str, str]:
    repo_dir = args.dino_repo_dir or str(payload.get("dino_repo_dir", "./dinov3"))
    model_name = args.dino_model_name or str(
        payload.get("dino_model_name", "dinov3_vitl16")
    )
    weights = args.dino_weights or str(
        payload.get("dino_weights", "./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    )
    return repo_dir, model_name, weights


def build_shard(args) -> None:
    if not args.base_ckpt or not os.path.isfile(args.base_ckpt):
        raise FileNotFoundError(f"Valid --base_ckpt is required, got {args.base_ckpt!r}.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards).")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    payload = torch.load(args.base_ckpt, map_location="cpu")
    config = config_from_checkpoint(payload)
    # The teacher must observe the unchanged stage-one model, not an older
    # distilled semantic head that may already be present in a resumed checkpoint.
    config.semantic_enabled = False
    repo_dir, model_name, weights = checkpoint_runtime(payload, args)
    detector = create_dino_single_tower(
        config=config,
        repo_dir=repo_dir,
        model_name=model_name,
        weights=weights,
        device=device,
    )
    detector.load_checkpoint_fields(payload, strict=True)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    dataset = create_dataset(args)
    bank = reference_bank(dataset)
    output_path = shard_path(args.output_path, args.shard_id)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    completed = {
        str(item["image_path"])
        for item in iter_jsonl(output_path)
        if bool(item.get("teacher", {}).get("parse_ok"))
    }
    teacher = QwenVLLMTeacher(
        model_id=args.model_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        max_images=3 + args.num_rois,
        teacher_image_size=args.teacher_image_size,
    )

    invalid = 0
    attempted = 0
    assigned = list(range(args.shard_id, len(dataset), args.num_shards))
    with open(output_path, "a", encoding="utf-8", buffering=1) as stream:
        for position, dataset_index in enumerate(assigned, 1):
            sample = dataset[dataset_index]
            image_path = str(sample["image_path"])
            if image_path in completed:
                continue
            raw_item = dataset.data_to_iterate[dataset_index]
            category = str(raw_item[0])
            query = Image.open(image_path).convert("RGB")
            reference_path = choose_reference(bank, category, image_path)
            reference = Image.open(reference_path).convert("RGB")
            image_tensor = sample["image"].unsqueeze(0).to(device)
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                base_output = detector(image_tensor)
            anomaly_map = base_output["prob"][0, 1].float().cpu()
            boxes = candidate_boxes_from_map(
                anomaly_map, num_rois=args.num_rois, roi_fraction=args.roi_fraction
            )
            prompt = build_vlm_prompt(category, boxes, len(config.visual_layers))
            images = [
                limit_image_size(query, args.teacher_image_size),
                limit_image_size(reference, args.teacher_image_size),
                limit_image_size(
                    heatmap_overlay(query, anomaly_map), args.teacher_image_size
                ),
                *[
                    limit_image_size(crop, args.teacher_image_size)
                    for crop in crop_regions(query, boxes)
                ],
            ]
            raw_response = teacher.generate(prompt, images)
            decision = parse_vlm_decision(raw_response, len(config.visual_layers))
            attempted += 1
            invalid += int(not decision.parse_ok)
            record = {
                "cache_version": 1,
                "prompt_version": PROMPT_VERSION,
                "dataset": args.dataset.lower(),
                "dataset_index": dataset_index,
                "image_path": image_path,
                "reference_path": reference_path,
                "category": category,
                "candidate_boxes": boxes,
                "base_max": float(anomaly_map.max().item()),
                "base_mean": float(anomaly_map.mean().item()),
                "teacher_model": args.model_id,
                "teacher": decision.to_dict(),
                "raw_response": raw_response,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"shard={args.shard_id} {position}/{len(assigned)} "
                f"parse={decision.parse_ok} action={decision.action} image={image_path}",
                flush=True,
            )

    latest_records = {
        str(item["image_path"]): item for item in iter_jsonl(output_path)
    }
    total_records = list(latest_records.values())
    total_invalid = sum(
        not bool(item.get("teacher", {}).get("parse_ok")) for item in total_records
    )
    invalid_ratio = total_invalid / max(1, len(total_records))
    print(
        f"Shard {args.shard_id} complete: records={len(total_records)}, "
        f"new={attempted}, invalid_ratio={invalid_ratio:.4f}, path={output_path}"
    )
    if invalid_ratio > args.max_invalid_ratio:
        raise RuntimeError(
            f"Invalid VLM response ratio {invalid_ratio:.3f} exceeds "
            f"--max_invalid_ratio={args.max_invalid_ratio:.3f}."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", type=str, default="")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--merge_shards", action="store_true")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--category", type=str, default="ALL")
    parser.add_argument("--train_split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=192)
    parser.add_argument("--num_rois", type=int, default=3)
    parser.add_argument("--teacher_image_size", type=int, default=512)
    parser.add_argument("--roi_fraction", type=float, default=0.25)
    parser.add_argument("--max_invalid_ratio", type=float, default=0.05)
    parser.add_argument("--dino_repo_dir", type=str, default="")
    parser.add_argument("--dino_model_name", type=str, default="")
    parser.add_argument("--dino_weights", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.merge_shards:
        merge_shards(args.output_path, args.num_shards)
    else:
        build_shard(args)


if __name__ == "__main__":
    main()
