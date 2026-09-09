#!/usr/bin/env python3
"""Frozen CLIP+DINO -> label-free ROI export -> offline VLM -> fixed evaluation.

Each stage is a separate process. Only export loads the dual tower; only review
loads vLLM. There is no base retraining, distillation or MARA in this experiment.
"""
import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from tools.vlm_review import (
    PROTOCOL, atomic_json, calibrate_map, candidate_rois, digest, file_digest,
    load_json, parse_review, review_prompt,
)


def resolve_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        # Only original numeric base epochs, not arbitrary or MARA checkpoints.
        choices = [p for p in path.glob("*.pth") if p.stem.isdigit()]
        if not choices:
            raise ValueError(f"No numeric base checkpoints in {path}; pass an explicit base file")
        path = max(choices, key=lambda p: int(p.stem))
    if not path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {path}")
    return path


def check_dual_payload(payload):
    if payload.get("base_arch", "clip_dino") != "clip_dino" or "dino_single_config" in payload:
        raise ValueError("This pipeline requires a CLIP+DINO base checkpoint")
    if "mara_agent" in payload:
        raise ValueError("Pass the stage-one Base checkpoint, not a MARA checkpoint")
    for name in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner"):
        if name not in payload:
            raise ValueError(f"Base checkpoint is missing {name}")


def model_signature(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("--model_id must be a complete local Qwen FP8 model directory")
    files = sorted(root.glob("*.safetensors"))
    if not files or not (root / "config.json").is_file():
        raise ValueError(f"Missing local model config or safetensors shards: {root}")
    index = root / "model.safetensors.index.json"
    if index.exists():
        expected = set(load_json(index)["weight_map"].values())
        if any(not (root / name).is_file() or (root / name).stat().st_size == 0 for name in expected):
            raise ValueError("A model shard listed in the safetensors index is missing/empty")
    return {"path": str(root), "config_sha256": file_digest(root / "config.json"),
            "weights": [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]}


def prepare(args):
    import torch
    from Datasets import DATASET_CLASSES
    base = resolve_checkpoint(args.base_ckpt)
    check_dual_payload(torch.load(base, map_location="cpu", weights_only=False))
    datasets = args.datasets.split()
    supported = {"visa", "mvtec", "btad", "mpdd"}
    if not datasets or len(set(datasets)) != len(datasets) or any(d not in DATASET_CLASSES or d not in supported for d in datasets):
        raise ValueError(f"Unsupported or duplicate datasets: {datasets}")
    if args.max_per_category < 0 or args.num_shards < 1 or args.batch_size < 1:
        raise ValueError("Invalid sample limit/shard count/batch size")
    if not 0 <= args.alpha <= 2 or not 0 <= args.confidence_threshold <= 1:
        raise ValueError("Invalid calibration alpha/confidence threshold")
    if not 0 <= args.max_invalid_ratio <= 1 or args.retries < 0:
        raise ValueError("Invalid review retry/coverage settings")
    if not 0 < args.roi_fraction <= 0.5 or not 0 < args.max_tokens < args.max_model_len:
        raise ValueError("Invalid ROI/context limits")
    if not 0 < args.gpu_memory < 1 or args.teacher_image_size < 32 or args.image_size < 32:
        raise ValueError("Invalid image size/GPU memory budget")
    if args.pro_num_th < 2 or not 0 < args.pro_max_fpr <= 1:
        raise ValueError("Invalid PRO evaluation settings")
    cfg = {key: getattr(args, key) for key in (
        "datasets", "max_per_category", "seed", "num_shards", "image_size", "batch_size",
        "roi_fraction", "teacher_image_size", "heatmap", "gpu_memory", "max_model_len", "max_tokens",
        "alpha", "confidence_threshold", "max_invalid_ratio", "retries", "pro_num_th", "pro_max_fpr",
        "dino_repo_dir", "dino_model_name", "dino_weights", "hfa_setting", "dino_bottleneck",
        "clip_model_name", "clip_pretrained", "visual_layers",
    )}
    cfg.update(protocol=PROTOCOL, base_ckpt=str(base), base_sha256=file_digest(base),
               model=model_signature(args.model_id), image_score="max_pixel_probability",
               label_policy="evaluation_only", normalization="none")
    path = Path(args.work_dir) / "config.json"
    if path.exists() and load_json(path) != cfg:
        raise ValueError("Work directory belongs to different settings/weights. Use a NEW WORK_DIR; nothing was overwritten.")
    atomic_json(path, cfg)
    print(f"Selected frozen Base checkpoint: {base}", flush=True)
    print(f"Run fingerprint: {digest(cfg)}; datasets={datasets}; limit/category={args.max_per_category}", flush=True)
    print("Calibration settings are fixed before target evaluation; no automatic target tuning.", flush=True)


def config(args):
    cfg = load_json(Path(args.work_dir) / "config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Unknown export protocol")
    return cfg


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npz.tmp")
    with open(temp, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def atomic_image(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".png.tmp")
    Image.fromarray(array).save(temp, format="PNG")
    temp.replace(path)


def load_dual(cfg, device):
    import torch
    from train_mara import apply_base_runtime_config, create_visual_backbone_for_mara
    from train_up import create_clip_model, create_adapter_model, create_prompt_learner
    payload = torch.load(cfg["base_ckpt"], map_location="cpu", weights_only=False)
    check_dual_payload(payload)
    args = argparse.Namespace(**{k: cfg[k] for k in (
        "dino_repo_dir", "dino_model_name", "dino_weights", "hfa_setting", "dino_bottleneck",
        "clip_model_name", "clip_pretrained", "image_size",
    )})
    args.visual_layers = tuple(int(v) for v in cfg["visual_layers"].split(","))
    args.visual_backbone, args.hfa_layers, args.hfa_bottleneck = "dino", "", None
    apply_base_runtime_config(args, payload)
    if args.visual_backbone != "dino":
        raise ValueError("Expected DINO visual backbone with CLIP text branch")
    clip = create_clip_model(args, device)
    prompt = create_prompt_learner(clip, device)
    dino, hfa = create_visual_backbone_for_mara(args, device)
    adapter = create_adapter_model(clip, device, "dino")
    for key in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter"):
        getattr(adapter, key).load_state_dict(payload[key], strict=True)
    prompt.load_state_dict(payload["prompt_learner"], strict=True)
    if hfa is not None:
        if "dino_adapters" not in payload:
            raise ValueError("HFA is enabled but checkpoint has no dino_adapters; specify matching HFA settings")
        hfa.load_state_dict(payload["dino_adapters"], strict=True)
    for module in (clip, prompt, dino, hfa, adapter):
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    print(f"Dual runtime: HFA={args.hfa_layers_runtime}, layers={args.visual_layers}", flush=True)
    return clip, prompt, dino, adapter, args.visual_layers


def selected_indices(dataset, limit, seed):
    indices = np.arange(len(dataset))
    if limit and len(indices) > limit:
        # Fixed random subset; not sorted by test label or anomaly score.
        indices = np.sort(np.random.default_rng(seed).choice(indices, limit, replace=False))
    return indices.tolist()


def export(args):
    import torch
    import torch.nn.functional as F
    from Datasets import DATASET_CLASSES, DATASET_REGISTRY
    from tools.dino_single_tower import collate_anomaly_batch
    from tools.utils_up import get_anomaly_map
    cfg, root = config(args), Path(args.work_dir)
    fingerprint = digest(cfg)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Bad shard index")
    if not torch.cuda.is_available():
        raise RuntimeError("Dual export requires CUDA in the training environment")
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    clip, prompt, dino, adapter, layers = load_dual(cfg, device)
    records = []
    for dataset_name in cfg["datasets"].split():
        cls, splits, source = DATASET_REGISTRY[dataset_name]
        for category in sorted(DATASET_CLASSES[dataset_name]):
            dataset = cls(source=source, split=splits.TEST, classname=category,
                          resize=cfg["image_size"], imagesize=cfg["image_size"])
            pick_seed = int(digest([cfg["seed"], dataset_name, category])[:8], 16)
            indices = selected_indices(dataset, cfg["max_per_category"], pick_seed)
            indices = indices[args.shard_id::cfg["num_shards"]]
            pending = []
            for idx in indices:
                ident = digest([dataset_name, category, idx])[:24]
                # Detect changed/reordered input data on resume without exposing
                # label-bearing source paths to the VLM prompt.
                entry = dataset.data_to_iterate[idx]
                sources = [Path(p).resolve() for p in entry[2:4] if p is not None]
                source_token = digest([[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in sources])
                record_path = root / "records" / f"{ident}.json"
                if record_path.is_file():
                    record = load_json(record_path)
                    if record["fingerprint"] != fingerprint or record.get("source_token") != source_token:
                        raise ValueError("Stale export record or changed source data; use a new WORK_DIR")
                    if all((root / record[k]).is_file() for k in ("query", "evidence", "evaluation")):
                        records.append(record)
                        continue
                pending.append((idx, ident, source_token))
            for start in range(0, len(pending), cfg["batch_size"]):
                group = pending[start:start + cfg["batch_size"]]
                samples = [dataset[idx] for idx, _, _ in group]
                batch = collate_anomaly_batch(samples)
                with torch.inference_mode():
                    _, masks, base, global_logits, evidence = get_anomaly_map(
                        clip, batch, device, adapter, dino, prompt, start,
                        visual_backbone="dino", visual_layers=layers,
                        text_source="prompt_learner", return_evidence=True,
                    )
                    maps = evidence["cross_prob_layers"]
                    maps = [m if m.ndim == 4 else m.unsqueeze(1) for m in maps]
                    maps = [F.interpolate(m, size=base.shape[-2:], mode="bilinear", align_corners=False) for m in maps]
                    stacked = torch.cat(maps, dim=1)
                    disagreement = stacked.std(dim=1, unbiased=False)
                for pos, (idx, ident, source_token) in enumerate(group):
                    prob = base[pos, 1].cpu().numpy().astype(np.float32)
                    dis = disagreement[pos].cpu().numpy().astype(np.float32)
                    rois = candidate_rois(prob, dis, seed=int(ident[:8], 16), fraction=cfg["roi_fraction"])
                    # Reconstruct exactly the tensor geometry seen by the detector.
                    img = batch["image"][pos].cpu().numpy().transpose(1, 2, 0)
                    rgb = np.rint(np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1) * 255).astype(np.uint8)
                    record = dict(id=ident, fingerprint=fingerprint, source_token=source_token, dataset=dataset_name,
                                  category=category, index=idx, rois=rois,
                                  query=f"queries/{ident}.png", evidence=f"evidence/{ident}.npz",
                                  evaluation=f"evaluation_only/{ident}.npz")
                    atomic_image(root / record["query"], rgb)
                    atomic_npz(root / record["evidence"], base=prob, disagreement=dis,
                               layers=stacked[pos].cpu().numpy().astype(np.float32))
                    # GT is never read by the review stage or included in its messages.
                    atomic_npz(root / record["evaluation"], mask=masks[pos, 0].cpu().numpy().astype(np.uint8),
                               label=np.array(int(batch["is_anomaly"][pos])),
                               global_logits=global_logits[pos].cpu().numpy())
                    atomic_json(root / "records" / f"{ident}.json", record)
                    records.append(record)
                print(f"[export shard {args.shard_id}] {dataset_name}/{category} {min(start + len(group), len(pending))}/{len(pending)}", flush=True)
    atomic_json(root / f"export_shard_{args.shard_id}.json", {"fingerprint": fingerprint, "records": records})


def seal(args):
    cfg, root = config(args), Path(args.work_dir)
    records, seen = [], set()
    for shard in range(cfg["num_shards"]):
        payload = load_json(root / f"export_shard_{shard}.json")
        if payload["fingerprint"] != digest(cfg):
            raise ValueError("Export shard settings mismatch")
        for record in payload["records"]:
            if record["id"] in seen:
                raise ValueError("Duplicate exported sample")
            seen.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError("No exported samples")
    records.sort(key=lambda r: (r["dataset"], r["category"], r["index"]))
    atomic_json(root / "manifest.json", {"fingerprint": digest(cfg), "records": records})
    print(f"Sealed {len(records)} samples; dual-tower processes may now be unloaded", flush=True)


def manifest(args):
    cfg, root = config(args), Path(args.work_dir)
    data = load_json(root / "manifest.json")
    if data["fingerprint"] != digest(cfg):
        raise ValueError("Manifest settings mismatch")
    return cfg, root, data["records"]


def build_review_images(root, record, cfg):
    # This function intentionally never opens evaluation_only or source paths.
    with Image.open(root / record["query"]) as image:
        query = image.convert("RGB")
    images = [query.copy()]
    if cfg["heatmap"]:
        with np.load(root / record["evidence"], allow_pickle=False) as data:
            prob = data["base"]
        overlay = np.asarray(query).astype(np.float32)
        color = np.zeros_like(overlay)
        color[..., 0] = 255
        weight = prob[..., None] * 0.45
        images.append(Image.fromarray(np.rint(overlay * (1 - weight) + color * weight).astype(np.uint8)))
    images.extend(query.crop(tuple(roi["box"])) for roi in record["rois"])
    for image in images:
        image.thumbnail((cfg["teacher_image_size"], cfg["teacher_image_size"]), Image.Resampling.LANCZOS)
    return images


def review(args):
    from tools.vlm_decision import QwenVLLMTeacher
    cfg, root, records = manifest(args)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Bad shard index")
    assigned = records[args.shard_id::cfg["num_shards"]]
    pending = []
    for record in assigned:
        path = root / "reviews" / f"{record['id']}.json"
        if path.exists():
            cached = load_json(path)
            if cached["fingerprint"] != digest(cfg):
                raise ValueError("Stale review settings")
            if cached["decision"]["parse_ok"]:
                continue
        pending.append(record)
    if not pending:
        print(f"[review shard {args.shard_id}] already complete", flush=True)
        return
    teacher = QwenVLLMTeacher(model_id=cfg["model"]["path"], gpu_memory_utilization=cfg["gpu_memory"],
                              max_model_len=cfg["max_model_len"], max_tokens=cfg["max_tokens"],
                              max_images=6, teacher_image_size=cfg["teacher_image_size"])
    for index, record in enumerate(pending):
        images = build_review_images(root, record, cfg)
        prompt = review_prompt(record["category"], record["rois"], cfg["heatmap"])
        started = time.monotonic()
        responses = []
        for attempt in range(cfg["retries"] + 1):
            retry_prompt = prompt if attempt == 0 else prompt + " Previous output was invalid. Keep defect_type short and include ALL requested ROI ids."
            raw = teacher.generate(retry_prompt, images)
            responses.append(raw)
            decision = parse_review(raw, record["rois"])
            if decision["parse_ok"]:
                break
        atomic_json(root / "reviews" / f"{record['id']}.json", {
            "id": record["id"], "fingerprint": digest(cfg), "decision": decision,
            "raw_responses": responses, "seconds": time.monotonic() - started,
        })
        print(f"[review shard {args.shard_id}] {index + 1}/{len(pending)} parse_ok={decision['parse_ok']}", flush=True)


def evaluate(args):
    from test2 import compute_best_f1, compute_i_auroc, compute_p_auroc, compute_pro
    cfg, root, records = manifest(args)
    reviews, invalid = {}, 0
    for record in records:
        path = root / "reviews" / f"{record['id']}.json"
        if not path.exists():
            raise ValueError(f"Missing review {record['id']}; resume review stage before evaluation")
        data = load_json(path)
        if data["fingerprint"] != digest(cfg) or data["id"] != record["id"]:
            raise ValueError("Review provenance mismatch")
        reviews[record["id"]] = data
        invalid += not data["decision"]["parse_ok"]
    if invalid / len(records) > cfg["max_invalid_ratio"]:
        raise ValueError(f"Invalid reviews {invalid}/{len(records)} exceed allowed ratio; inspect raw_responses and resume review")
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["category"])].append(record)
    rows = []
    for (dataset, category), group in groups.items():
        masks, bases, updated, controls = [], [], [], []
        roi_gt, roi_votes, coverage = [], [], []
        seconds, valid_count = [], 0
        for record in group:
            with np.load(root / record["evidence"], allow_pickle=False) as data:
                base = data["base"]
            with np.load(root / record["evaluation"], allow_pickle=False) as data:
                mask = data["mask"]
            review_data = reviews[record["id"]]
            decision = review_data["decision"]
            valid_count += bool(decision["parse_ok"])
            seconds.append(review_data["seconds"])
            bases.append(base)
            masks.append(mask)
            updated.append(calibrate_map(base, record["rois"], decision, cfg["alpha"], cfg["confidence_threshold"]))
            controls.append(calibrate_map(base, record["rois"], {}, cfg["alpha"], cfg["confidence_threshold"], control=True))
            union = np.zeros_like(mask, dtype=bool)
            votes = {v["roi_id"]: v for v in decision["regions"]}
            for roi in record["rois"]:
                x1, y1, x2, y2 = roi["box"]
                union[y1:y2, x1:x2] = True
                vote = votes.get(roi["roi_id"])
                if vote and vote["verdict"] != "uncertain" and vote["confidence"] >= cfg["confidence_threshold"]:
                    roi_gt.append(bool(mask[y1:y2, x1:x2].any()))
                    roi_votes.append(vote["verdict"] == "defect")
            if mask.any():
                coverage.append(float(mask[union].sum() / mask.sum()))
        masks, bases, updated, controls = map(np.stack, (masks, bases, updated, controls))
        labels = masks.reshape(len(group), -1).max(axis=1) > 0
        base_scores = bases.reshape(len(group), -1).max(axis=1)
        new_scores = updated.reshape(len(group), -1).max(axis=1)
        metrics = {}
        for name, maps, scores in (
            ("base", bases, base_scores),
            ("vlm_region", updated, new_scores),
            ("control_shrink", controls, controls.reshape(len(group), -1).max(axis=1)),
        ):
            metrics[name] = {"PRO": compute_pro(masks, maps, num_th=cfg["pro_num_th"], max_fpr=cfg["pro_max_fpr"]),
                             "P_AUROC": compute_p_auroc(masks, maps),
                             "I_AUROC": compute_i_auroc(labels, scores),
                             "F1": compute_best_f1(masks, maps)}
        metrics["vlm_image"] = {**metrics["base"], "I_AUROC": metrics["vlm_region"]["I_AUROC"]}
        # Fixed 0.5 is diagnostic only (not selected on target labels).
        fp_base, fp_new = (base_scores >= 0.5) & ~labels, (new_scores >= 0.5) & ~labels
        fn_base, fn_new = (base_scores < 0.5) & labels, (new_scores < 0.5) & labels
        row = {"dataset": dataset, "category": category, "samples": len(group),
               "normal_samples": int((~labels).sum()), "anomaly_samples": int(labels.sum()),
               "valid_ratio": valid_count / len(group), "review_seconds_mean": float(np.mean(seconds)),
               "candidate_pixel_recall": float(np.mean(coverage)) if coverage else None,
               "total_roi_count": sum(len(r["rois"]) for r in group),
               "decisive_roi_count": len(roi_gt),
               "decisive_roi_fraction": len(roi_gt) / max(1, sum(len(r["rois"]) for r in group)),
               "decisive_roi_accuracy": float(np.mean(np.array(roi_gt) == np.array(roi_votes))) if roi_gt else None,
               "removed_fp_at_0_5": int((fp_base & ~fp_new).sum()),
               "new_fn_at_0_5": int((fn_new & ~fn_base).sum()),
               "base_fp_at_0_5": int(fp_base.sum()), "new_fp_at_0_5": int(fp_new.sum()),
               "base_fn_at_0_5": int(fn_base.sum()), "new_total_fn_at_0_5": int(fn_new.sum()),
               "changed_pixel_fraction": float(np.mean(updated != bases))}
        for mode, values in metrics.items():
            row.update({f"{mode}_{metric}": value for metric, value in values.items()})
        rows.append(row)
        print(f"{dataset}/{category}: Base PRO={metrics['base']['PRO']:.5f}, VLM PRO={metrics['vlm_region']['PRO']:.5f}, I-AUC={metrics['base']['I_AUROC']:.5f}->{metrics['vlm_image']['I_AUROC']:.5f}", flush=True)
    means = {}
    for dataset in cfg["datasets"].split():
        selected = [r for r in rows if r["dataset"] == dataset]
        means[dataset] = {key: float(np.mean([r[key] for r in selected])) for key in selected[0]
                          if any(key == f"{mode}_{metric}" for mode in ("base", "vlm_image", "vlm_region", "control_shrink")
                                 for metric in ("PRO", "F1", "P_AUROC", "I_AUROC"))}
    result = {"fingerprint": digest(cfg), "config": cfg, "invalid_reviews": invalid,
              "protocol_note": "Online VLM inference cached for evaluation; no training or MARA. Best-F1 uses test thresholds for reporting only. Fixed 0.5 error counts are diagnostics. Source samples may overlap Base training.",
              "means": means, "categories": rows}
    out = root / "results"
    atomic_json(out / "metrics.json", result)
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [result["protocol_note"], f"Invalid reviews: {invalid}/{len(records)}", ""]
    for dataset, values in means.items():
        lines.append(f"Dataset: {dataset} (macro average over categories)")
        lines.append("Mode                  PRO       P-AUC     I-AUC     F1")
        for mode in ("base", "vlm_image", "vlm_region", "control_shrink"):
            lines.append(f"{mode:<22}" + " ".join(f"{values[f'{mode}_{metric}']:.5f}" for metric in ("PRO", "P_AUROC", "I_AUROC", "F1")))
    (out / "metric_vlm.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Results: {out.resolve()}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "export", "seal", "review", "evaluate"), required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--base_ckpt", default="")
    p.add_argument("--datasets", default="visa")
    p.add_argument("--max_per_category", type=int, default=40)
    p.add_argument("--num_shards", type=int, default=2)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--roi_fraction", type=float, default=0.25)
    p.add_argument("--model_id", default="")
    p.add_argument("--teacher_image_size", type=int, default=512)
    p.add_argument("--heatmap", action="store_true")
    p.add_argument("--gpu_memory", type=float, default=0.70)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--confidence_threshold", type=float, default=0.8)
    p.add_argument("--max_invalid_ratio", type=float, default=0.05)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--pro_num_th", type=int, default=1000)
    p.add_argument("--pro_max_fpr", type=float, default=0.3)
    p.add_argument("--dino_repo_dir", default="./dinov3")
    p.add_argument("--dino_model_name", default="dinov3_vitl16")
    p.add_argument("--dino_weights", default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    p.add_argument("--hfa_setting", default="hfa3")
    p.add_argument("--dino_bottleneck", type=int, default=256)
    p.add_argument("--clip_model_name", default="ViT-L-14-336")
    p.add_argument("--clip_pretrained", default="openai")
    p.add_argument("--visual_layers", default="5,11,17,23")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[args.stage](args)
