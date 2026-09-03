#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Datasets import DATASET_CLASSES, DATASET_REGISTRY
from tools.mara_agent import MARAAgent, MARAConfig, quality_score
from tools.mara_evidence import build_mara_evidence
from tools.dino_single_tower import DinoSingleTowerDetector, forward_dino_single_batch
from tools.utils_up import get_anomaly_map
from tools.visualization import visualization
from train_mara import (
    HFA_CHOICES,
    apply_base_runtime_config,
    create_single_tower_from_checkpoint,
    create_visual_backbone_for_mara,
    normalize_layers_obj,
)
from train_up import (
    create_adapter_model,
    create_clip_model,
    create_prompt_learner,
    parse_visual_layers,
    set_seed,
)
from test2 import (
    compute_best_f1,
    compute_i_auroc,
    compute_image_level_scores,
    compute_p_auroc,
    compute_pro,
    normalize_maps,
)


def sorted_mara_checkpoints(weight_path: str) -> List[str]:
    path = Path(weight_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {weight_path}")

    candidates = list(path.glob("mara_epoch_*.pth"))
    if not candidates:
        candidates = list(path.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No MARA checkpoint found in: {weight_path}")

    def key_fn(p: Path):
        nums = "".join(ch if ch.isdigit() else " " for ch in p.stem).split()
        return int(nums[-1]) if nums else 10**9

    return [str(p) for p in sorted(candidates, key=key_fn)]


def prepare_data(dataset_name: str, category: str, args):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=args.image_size,
        imagesize=args.image_size,
    )
    kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print(f"Loaded [{dataset_name}] ({category}) test set, size: {len(dataset)}")
    return loader


def config_from_payload(payload: Dict, args) -> MARAConfig:
    cfg_dict = dict(payload.get("mara_config", {}))
    if not cfg_dict:
        cfg_dict = {
            "num_layers": len(args.visual_layers),
            "map_size": args.mara_map_size,
            "num_regions": args.mara_regions,
            "roi_size": args.mara_roi_size,
            "hidden_dim": args.mara_hidden_dim,
            "max_steps": args.mara_steps,
            "group_size": 1,
        }
    mara_state = payload.get("mara_agent", {})
    has_gain_regressor = any(str(key).startswith("gain_head.") for key in mara_state.keys())
    has_gain_lower = any(str(key).startswith("gain_lower_head.") for key in mara_state.keys())
    has_gain_classifier = any(str(key).startswith("gain_accept_head.") for key in mara_state.keys())
    if mara_state and not (has_gain_regressor and has_gain_classifier):
        cfg_dict["use_gain_gate"] = False
    if mara_state:
        cfg_dict["use_gain_lower_bound"] = bool(
            cfg_dict.get("use_gain_lower_bound", True) and has_gain_lower
        )
    valid_keys = MARAConfig.__dataclass_fields__.keys()
    cfg = MARAConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    if args.gain_safety_margin is not None:
        cfg.gain_safety_margin = args.gain_safety_margin
    if args.gain_accept_probability is not None:
        cfg.gain_accept_probability = args.gain_accept_probability
    if args.disable_hard_gain_gate:
        cfg.hard_gain_gate = False
    return cfg


def apply_payload_runtime_config(payload: Dict, args) -> None:
    apply_base_runtime_config(args, payload)


def build_models_from_checkpoint(args, device: torch.device, payload: Dict):
    apply_payload_runtime_config(payload, args)
    mara_cfg = config_from_payload(payload, args)
    print(
        "Runtime config: "
        f"base_arch={args.base_arch}, "
        f"visual_backbone={args.visual_backbone}, "
        f"visual_layers={args.visual_layers}, "
        f"hfa_setting={args.hfa_setting_runtime}, "
        f"hfa_layers={args.hfa_layers_runtime}, "
        f"hfa_bottleneck={args.hfa_bottleneck_runtime}, "
        f"feature_channels_per_layer={args.evidence_feature_channels}, "
        f"evidence_channels={mara_cfg.evidence_channels}, "
        f"global_evidence_dim={mara_cfg.global_evidence_dim}"
    )
    single_tower = None
    clip_model = None
    prompt_learner = None
    dino_model = None
    dino_adapters = None
    model = None
    if args.base_arch == "dino_single":
        single_tower = create_single_tower_from_checkpoint(args, device, payload)
        single_tower.eval()
    else:
        clip_model = create_clip_model(args, device)
        prompt_learner = create_prompt_learner(clip_model, device)
        dino_model, dino_adapters = create_visual_backbone_for_mara(args, device)
        model = create_adapter_model(clip_model, device, args.visual_backbone)
    mara_agent = MARAAgent(mara_cfg).to(device)

    if clip_model is not None:
        clip_model.eval()
    if prompt_learner is not None:
        prompt_learner.eval()
    if model is not None:
        model.eval()
    mara_agent.eval()
    if dino_model is not None:
        dino_model.eval()
    if dino_adapters is not None:
        dino_adapters.eval()

    return clip_model, prompt_learner, dino_model, dino_adapters, model, mara_agent, single_tower


def load_mara_checkpoint(
    ckpt_path: str,
    payload: Dict,
    model: Optional[torch.nn.Module],
    prompt_learner: Optional[torch.nn.Module],
    dino_adapters: Optional[torch.nn.Module],
    mara_agent: MARAAgent,
    device: torch.device,
    single_tower: Optional[DinoSingleTowerDetector] = None,
) -> None:
    if single_tower is not None:
        single_tower.load_checkpoint_fields(payload, strict=True)
    else:
        if model is None or prompt_learner is None:
            raise ValueError("Legacy checkpoint requires model and prompt_learner.")
        model.cls_token_adapter.load_state_dict(payload["cls_token_adapter"], strict=False)
        model.patch_token_adapter.load_state_dict(payload["patch_token_adapter"], strict=False)
        model.prompt_adapter.load_state_dict(payload["prompt_adapter"], strict=False)
        if "prompt_learner" in payload:
            prompt_learner.load_state_dict(payload["prompt_learner"], strict=False)
        if dino_adapters is not None and "dino_adapters" in payload:
            dino_adapters.load_state_dict(payload["dino_adapters"], strict=True)
    missing, unexpected = mara_agent.load_state_dict(payload["mara_agent"], strict=False)
    if missing or unexpected:
        print(f"[WARN] MARA state loaded with missing={missing}, unexpected={unexpected}")

    if model is not None:
        model.to(device).eval()
    if prompt_learner is not None:
        prompt_learner.to(device).eval()
    mara_agent.to(device).eval()
    if dino_adapters is not None:
        dino_adapters.to(device).eval()
    if single_tower is not None:
        single_tower.to(device).eval()


def compute_metrics(gt_masks: np.ndarray, pred_eval: np.ndarray, args) -> Dict[str, float]:
    image_gt, image_pred = compute_image_level_scores(gt_masks, pred_eval)
    return {
        "F1": compute_best_f1(gt_masks, pred_eval),
        "I_AUROC": compute_i_auroc(image_gt, image_pred),
        "P_AUROC": compute_p_auroc(gt_masks, pred_eval),
        "PRO": compute_pro(gt_masks, pred_eval, num_th=args.pro_num_th, max_fpr=args.pro_max_fpr),
    }


def evaluate_category(
    args,
    category: str,
    clip_model: Optional[torch.nn.Module],
    prompt_learner: Optional[torch.nn.Module],
    dino_model: Optional[torch.nn.Module],
    model: Optional[torch.nn.Module],
    mara_agent: MARAAgent,
    device: torch.device,
    result_dir: str,
    single_tower: Optional[DinoSingleTowerDetector] = None,
):
    base_maps = []
    mara_maps = []
    oracle_maps = []
    gate_values = []
    gate_roi_values = []
    changed_values = []
    refine_values = []
    attempt_values = []
    reject_values = []
    gain_values = []
    gain_lower_values = []
    accept_values = []
    quality_gain_values = []
    accepted_negative_count = 0
    accepted_count = 0
    identity_mean_values = []
    identity_max_values = []
    gt_masks = []
    img_paths = []

    loader = prepare_data(args.dataset, category, args)
    with torch.no_grad():
        for batch_idx, image_info in enumerate(tqdm(loader)):
            if single_tower is not None:
                mask, base_map, base_logits, stage1_evidence = forward_dino_single_batch(
                    single_tower,
                    image_info,
                    device,
                )
            else:
                _, mask, base_map, base_logits, stage1_evidence = get_anomaly_map(
                    clip_model=clip_model,
                    image_info=image_info,
                    device=device,
                    model=model,
                    Dino_model=dino_model,
                    prompt_learner=prompt_learner,
                    idx=batch_idx,
                    visual_backbone=args.visual_backbone,
                    visual_layers=args.visual_layers,
                    text_source=args.text_source,
                    return_evidence=True,
                )
            mara_evidence = build_mara_evidence(
                evidence=stage1_evidence,
                fallback_prob=base_map,
                num_layers=len(args.visual_layers),
                map_size=mara_agent.cfg.map_size,
                include_full_resolution_oracle=not args.disable_evidence_oracle,
                feature_channels_per_layer=int(getattr(args, "evidence_feature_channels", 0)),
            )
            del stage1_evidence
            layer_maps = mara_evidence["layer_maps"]
            use_evidence_bank = mara_agent.cfg.evidence_channels > 0
            output = mara_agent.infer(
                base_prob=base_map,
                base_logits=base_logits,
                layer_maps=layer_maps,
                evidence_maps=mara_evidence["extra_maps"] if use_evidence_bank else None,
                global_evidence=mara_evidence["global_evidence"] if use_evidence_bank else None,
                sample=args.sample_policy,
            )
            labels = image_info["is_anomaly"].to(device).long()
            base_quality = quality_score(base_map, base_logits, mask, labels, mara_agent.cfg)
            mara_quality = quality_score(
                output["final_prob"], output["final_logits"], mask, labels, mara_agent.cfg
            )
            quality_gain = mara_quality - base_quality
            quality_gain_values.append(quality_gain.detach().cpu().numpy())
            executed = output.get("refine_steps", torch.zeros_like(quality_gain)) > 0
            accepted_negative_count += int(
                ((quality_gain < -args.quality_degradation_tolerance) & executed).sum().item()
            )
            accepted_count += int(executed.sum().item())

            if args.disable_evidence_oracle:
                oracle_map = base_map[:, 1]
            else:
                candidates = mara_evidence["oracle_maps"]
                if candidates.shape[-2:] != base_map.shape[-2:]:
                    candidates = F.interpolate(
                        candidates,
                        size=base_map.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                candidates = torch.cat([base_map[:, 1:2], candidates], dim=1)
                candidate_quality = []
                for candidate_idx in range(candidates.shape[1]):
                    candidate_anomaly = candidates[:, candidate_idx].clamp(0.0, 1.0)
                    candidate_prob = torch.stack([1.0 - candidate_anomaly, candidate_anomaly], dim=1)
                    candidate_quality.append(
                        quality_score(candidate_prob, base_logits, mask, labels, mara_agent.cfg)
                    )
                best_candidate = torch.stack(candidate_quality, dim=1).argmax(dim=1)
                oracle_map = candidates.gather(
                    1,
                    best_candidate.view(-1, 1, 1, 1).expand(
                        -1, 1, candidates.shape[-2], candidates.shape[-1]
                    ),
                )[:, 0]

            base_maps.append(base_map[:, 1].detach().cpu().numpy())
            mara_maps.append(output["final_prob"][:, 1].detach().cpu().numpy())
            oracle_maps.append(oracle_map.detach().cpu().numpy())
            identity_mean_values.append(output["identity_error_mean"].detach().cpu().numpy())
            identity_max_values.append(output["identity_error_max"].detach().cpu().numpy())
            if "gate_map" in output:
                gate_values.append(output["gate_map"].detach().mean(dim=(1, 2, 3)).cpu().numpy())
            if "gate_roi_mean" in output:
                gate_roi_values.append(output["gate_roi_mean"].detach().cpu().numpy())
            if "changed_ratio" in output:
                changed_values.append(output["changed_ratio"].detach().cpu().numpy())
            if "refine_steps" in output:
                refine_values.append(output["refine_steps"].detach().cpu().numpy())
            if "refine_attempts" in output:
                attempt_values.append(output["refine_attempts"].detach().cpu().numpy())
            if "rejected_steps" in output:
                reject_values.append(output["rejected_steps"].detach().cpu().numpy())
            attempted = output.get("refine_attempts", output.get("refine_steps"))
            refined_mask = attempted > 0 if attempted is not None else None
            if "predicted_gain" in output:
                values = output["predicted_gain"]
                if refined_mask is not None:
                    values = values[refined_mask]
                if values.numel() > 0:
                    gain_values.append(values.detach().cpu().numpy())
            if "predicted_gain_lower" in output:
                values = output["predicted_gain_lower"]
                if refined_mask is not None:
                    values = values[refined_mask]
                if values.numel() > 0:
                    gain_lower_values.append(values.detach().cpu().numpy())
            if "accept_score" in output:
                values = output["accept_score"]
                if refined_mask is not None:
                    values = values[refined_mask]
                if values.numel() > 0:
                    accept_values.append(values.detach().cpu().numpy())
            mask_np = mask[:, 0].detach().cpu().numpy() if mask.dim() == 4 else mask.detach().cpu().numpy()
            gt_masks.append((mask_np > 0.5).astype(np.uint8))
            img_paths.extend(list(image_info["image_path"]))

    base_maps = np.concatenate(base_maps, axis=0).astype(np.float32)
    mara_maps = np.concatenate(mara_maps, axis=0).astype(np.float32)
    oracle_maps = np.concatenate(oracle_maps, axis=0).astype(np.float32)
    gt_masks = np.concatenate(gt_masks, axis=0).astype(np.uint8)
    base_eval = normalize_maps(base_maps, mode=args.norm_mode)
    mara_eval = normalize_maps(mara_maps, mode=args.norm_mode)
    oracle_eval = normalize_maps(oracle_maps, mode=args.norm_mode)

    if args.save_vis:
        vis_dir = os.path.join(result_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        visualization(img_paths, normalize_maps(mara_maps, mode="per_class"), gt_masks, category, vis_dir)

    base_metrics = compute_metrics(gt_masks, base_eval, args)
    mara_metrics = compute_metrics(gt_masks, mara_eval, args)
    oracle_metrics = compute_metrics(gt_masks, oracle_eval, args)
    row = {"category": category}
    for metric_name in ("F1", "I_AUROC", "P_AUROC", "PRO"):
        row[f"base_{metric_name}"] = base_metrics[metric_name]
        row[f"mara_{metric_name}"] = mara_metrics[metric_name]
        row[f"delta_{metric_name}"] = mara_metrics[metric_name] - base_metrics[metric_name]
        row[f"oracle_{metric_name}"] = oracle_metrics[metric_name]
        row[f"oracle_delta_{metric_name}"] = oracle_metrics[metric_name] - base_metrics[metric_name]
    row["gate_mean"] = float(np.concatenate(gate_values).mean()) if gate_values else 0.0
    row["gate_roi_mean"] = float(np.concatenate(gate_roi_values).mean()) if gate_roi_values else 0.0
    row["changed_ratio_mean"] = float(np.concatenate(changed_values).mean()) if changed_values else 0.0
    row["refine_steps_mean"] = float(np.concatenate(refine_values).mean()) if refine_values else 0.0
    row["refine_attempts_mean"] = float(np.concatenate(attempt_values).mean()) if attempt_values else 0.0
    row["rejected_steps_mean"] = float(np.concatenate(reject_values).mean()) if reject_values else 0.0
    row["pred_gain_mean"] = float(np.concatenate(gain_values).mean()) if gain_values else 0.0
    row["pred_gain_lower_mean"] = float(np.concatenate(gain_lower_values).mean()) if gain_lower_values else 0.0
    row["accept_mean"] = float(np.concatenate(accept_values).mean()) if accept_values else 0.0
    quality_gains = np.concatenate(quality_gain_values) if quality_gain_values else np.zeros(1, dtype=np.float32)
    row["quality_gain_mean"] = float(quality_gains.mean())
    row["base_degradation_rate"] = float(
        (quality_gains < -args.quality_degradation_tolerance).mean()
    )
    row["negative_gain_accept_rate"] = float(accepted_negative_count / max(1, accepted_count))
    row["identity_error_mean"] = float(np.concatenate(identity_mean_values).mean())
    row["identity_error_max"] = float(np.concatenate(identity_max_values).max())
    return row


def write_results(result_dir: str, epoch_name: str, rows: List[Dict]) -> Dict:
    os.makedirs(result_dir, exist_ok=True)
    metric_names = ("F1", "I_AUROC", "P_AUROC", "PRO")
    means = {"epoch": epoch_name}
    for prefix in ("base", "mara", "delta", "oracle", "oracle_delta"):
        for metric_name in metric_names:
            means[f"mean_{prefix}_{metric_name}"] = float(np.mean([r[f"{prefix}_{metric_name}"] for r in rows]))
    means["mean_gate"] = float(np.mean([r["gate_mean"] for r in rows]))
    means["mean_gate_roi"] = float(np.mean([r["gate_roi_mean"] for r in rows]))
    means["mean_changed_ratio"] = float(np.mean([r["changed_ratio_mean"] for r in rows]))
    means["mean_refine_steps"] = float(np.mean([r["refine_steps_mean"] for r in rows]))
    means["mean_refine_attempts"] = float(np.mean([r["refine_attempts_mean"] for r in rows]))
    means["mean_rejected_steps"] = float(np.mean([r["rejected_steps_mean"] for r in rows]))
    means["mean_pred_gain"] = float(np.mean([r["pred_gain_mean"] for r in rows]))
    means["mean_pred_gain_lower"] = float(np.mean([r["pred_gain_lower_mean"] for r in rows]))
    means["mean_accept"] = float(np.mean([r["accept_mean"] for r in rows]))
    means["mean_quality_gain"] = float(np.mean([r["quality_gain_mean"] for r in rows]))
    means["mean_base_degradation_rate"] = float(np.mean([r["base_degradation_rate"] for r in rows]))
    means["mean_negative_gain_accept_rate"] = float(np.mean([r["negative_gain_accept_rate"] for r in rows]))
    means["mean_identity_error"] = float(np.mean([r["identity_error_mean"] for r in rows]))
    means["max_identity_error"] = float(np.max([r["identity_error_max"] for r in rows]))

    metric_file = os.path.join(result_dir, "metric_mara.txt")
    with open(metric_file, "a", encoding="utf-8") as f:
        f.write(f"----------MARA epoch {epoch_name}----------\n")
        f.write(
            f"{'Classname':<18s}"
            f"{'Base_PRO':>10s}{'MARA_PRO':>10s}{'D_PRO':>10s}"
            f"{'Base_F1':>10s}{'MARA_F1':>10s}{'D_F1':>10s}"
            f"{'Gate':>10s}{'GateROI':>10s}{'Changed':>10s}"
            f"{'Attempt':>10s}{'Refine':>10s}{'Reject':>10s}"
            f"{'GainPred':>10s}{'GainQ10':>10s}{'Accept':>10s}"
            f"{'Base_P-AUC':>12s}{'MARA_P-AUC':>12s}{'D_P-AUC':>10s}"
            f"{'Base_I-AUC':>12s}{'MARA_I-AUC':>12s}{'D_I-AUC':>10s}\n"
        )
        for row in rows:
            f.write(
                f"{row['category']:<18s}"
                f"{row['base_PRO']:>10.5f}{row['mara_PRO']:>10.5f}{row['delta_PRO']:>10.5f}"
                f"{row['base_F1']:>10.5f}{row['mara_F1']:>10.5f}{row['delta_F1']:>10.5f}"
                f"{row['gate_mean']:>10.5f}{row['gate_roi_mean']:>10.5f}"
                f"{row['changed_ratio_mean']:>10.5f}{row['refine_attempts_mean']:>10.5f}"
                f"{row['refine_steps_mean']:>10.5f}{row['rejected_steps_mean']:>10.5f}"
                f"{row['pred_gain_mean']:>10.5f}{row['pred_gain_lower_mean']:>10.5f}{row['accept_mean']:>10.5f}"
                f"{row['base_P_AUROC']:>12.5f}{row['mara_P_AUROC']:>12.5f}{row['delta_P_AUROC']:>10.5f}"
                f"{row['base_I_AUROC']:>12.5f}{row['mara_I_AUROC']:>12.5f}{row['delta_I_AUROC']:>10.5f}\n"
            )
        f.write(
            f"{'Mean':<18s}"
            f"{means['mean_base_PRO']:>10.5f}{means['mean_mara_PRO']:>10.5f}{means['mean_delta_PRO']:>10.5f}"
            f"{means['mean_base_F1']:>10.5f}{means['mean_mara_F1']:>10.5f}{means['mean_delta_F1']:>10.5f}"
            f"{means['mean_gate']:>10.5f}{means['mean_gate_roi']:>10.5f}"
            f"{means['mean_changed_ratio']:>10.5f}{means['mean_refine_attempts']:>10.5f}"
            f"{means['mean_refine_steps']:>10.5f}{means['mean_rejected_steps']:>10.5f}"
            f"{means['mean_pred_gain']:>10.5f}{means['mean_pred_gain_lower']:>10.5f}{means['mean_accept']:>10.5f}"
            f"{means['mean_base_P_AUROC']:>12.5f}{means['mean_mara_P_AUROC']:>12.5f}{means['mean_delta_P_AUROC']:>10.5f}"
            f"{means['mean_base_I_AUROC']:>12.5f}{means['mean_mara_I_AUROC']:>12.5f}{means['mean_delta_I_AUROC']:>10.5f}\n\n"
        )
        f.write(
            f"Safety: quality_gain={means['mean_quality_gain']:.6f}, "
            f"base_degradation_rate={means['mean_base_degradation_rate']:.6f}, "
            f"negative_gain_accept_rate={means['mean_negative_gain_accept_rate']:.6f}, "
            f"identity_error={means['mean_identity_error']:.8f}/{means['max_identity_error']:.8f}\n\n"
        )
        f.write("Evidence Oracle (GT diagnostic only; not a deployable result)\n")
        f.write(
            f"{'Classname':<18s}"
            f"{'Base_PRO':>10s}{'Oracle_PRO':>12s}{'D_PRO':>10s}"
            f"{'Base_P-AUC':>12s}{'Oracle_P-AUC':>14s}{'D_P-AUC':>10s}\n"
        )
        for row in rows:
            f.write(
                f"{row['category']:<18s}"
                f"{row['base_PRO']:>10.5f}{row['oracle_PRO']:>12.5f}{row['oracle_delta_PRO']:>10.5f}"
                f"{row['base_P_AUROC']:>12.5f}{row['oracle_P_AUROC']:>14.5f}"
                f"{row['oracle_delta_P_AUROC']:>10.5f}\n"
            )
        f.write(
            f"{'Mean':<18s}"
            f"{means['mean_base_PRO']:>10.5f}{means['mean_oracle_PRO']:>12.5f}"
            f"{means['mean_oracle_delta_PRO']:>10.5f}"
            f"{means['mean_base_P_AUROC']:>12.5f}{means['mean_oracle_P_AUROC']:>14.5f}"
            f"{means['mean_oracle_delta_P_AUROC']:>10.5f}\n\n"
        )

    csv_file = os.path.join(result_dir, f"{epoch_name}_mara_metrics.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["category"]
        for metric_name in metric_names:
            fieldnames.extend(
                [
                    f"base_{metric_name}",
                    f"mara_{metric_name}",
                    f"delta_{metric_name}",
                    f"oracle_{metric_name}",
                    f"oracle_delta_{metric_name}",
                ]
            )
        fieldnames.append("gate_mean")
        fieldnames.append("gate_roi_mean")
        fieldnames.append("changed_ratio_mean")
        fieldnames.append("refine_steps_mean")
        fieldnames.append("refine_attempts_mean")
        fieldnames.append("rejected_steps_mean")
        fieldnames.append("pred_gain_mean")
        fieldnames.append("pred_gain_lower_mean")
        fieldnames.append("accept_mean")
        fieldnames.append("quality_gain_mean")
        fieldnames.append("base_degradation_rate")
        fieldnames.append("negative_gain_accept_rate")
        fieldnames.append("identity_error_mean")
        fieldnames.append("identity_error_max")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result_MARA_Test")
    parser.add_argument("--weight_path", type=str, required=True, help="MARA checkpoint file or checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--category", type=str, default="ALL")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--visual_backbone", type=str, default="dino", choices=["dino", "clip"])
    parser.add_argument("--visual_layers", type=parse_visual_layers, default=(5, 11, 17, 23))
    parser.add_argument("--text_source", type=str, default="prompt_learner", choices=["prompt_learner", "ensemble"])
    parser.add_argument("--hfa_setting", type=str, default="hfa4", choices=HFA_CHOICES)
    parser.add_argument("--hfa_layers", type=str, default="", help="optional comma-separated HFA adapter layers")

    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--dino_repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--dino_bottleneck", type=int, default=256)
    parser.add_argument("--hfa_bottleneck", type=int, default=None, help="alias for DINO adapter bottleneck fallback")

    parser.add_argument("--mara_steps", type=int, default=3)
    parser.add_argument("--mara_map_size", type=int, default=128)
    parser.add_argument("--mara_regions", type=int, default=16)
    parser.add_argument("--mara_roi_size", type=int, default=32)
    parser.add_argument("--mara_hidden_dim", type=int, default=64)
    parser.add_argument("--gain_safety_margin", type=float, default=None)
    parser.add_argument("--gain_accept_probability", type=float, default=None)
    parser.add_argument("--disable_hard_gain_gate", action="store_true")
    parser.add_argument("--disable_evidence_oracle", action="store_true")
    parser.add_argument("--quality_degradation_tolerance", type=float, default=1e-4)

    parser.add_argument("--norm_mode", type=str, default="none", choices=["none", "per_image", "per_class"])
    parser.add_argument("--pro_num_th", type=int, default=1000)
    parser.add_argument("--pro_max_fpr", type=float, default=0.3)
    parser.add_argument("--eval_latest_only", action="store_true")
    parser.add_argument("--sample_policy", action="store_true", help="sample actions instead of greedy policy during evaluation")
    parser.add_argument("--save_vis", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.dataset = args.dataset.lower()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_result_dir = os.path.join(args.result_path, args.dataset)
    os.makedirs(dataset_result_dir, exist_ok=True)

    ckpts = sorted_mara_checkpoints(args.weight_path)
    if args.eval_latest_only:
        ckpts = [ckpts[-1]]

    first_payload = torch.load(ckpts[0], map_location=device)
    (
        clip_model,
        prompt_learner,
        dino_model,
        dino_adapters,
        model,
        mara_agent,
        single_tower,
    ) = build_models_from_checkpoint(
        args=args,
        device=device,
        payload=first_payload,
    )

    summaries = []
    categories = sorted(DATASET_CLASSES[args.dataset]) if args.category == "ALL" else [args.category]
    for ckpt_path in ckpts:
        epoch_name = Path(ckpt_path).stem
        payload = torch.load(ckpt_path, map_location=device)
        cur_hfa_layers = normalize_layers_obj(payload.get("hfa_layers"))
        if cur_hfa_layers and tuple(cur_hfa_layers) != tuple(args.hfa_layers_runtime):
            raise ValueError(
                f"HFA layer mismatch across MARA checkpoints. "
                f"Runtime layers={args.hfa_layers_runtime}, but {ckpt_path} uses {cur_hfa_layers}."
            )
        load_mara_checkpoint(
            ckpt_path=ckpt_path,
            payload=payload,
            model=model,
            prompt_learner=prompt_learner,
            dino_adapters=dino_adapters,
            mara_agent=mara_agent,
            device=device,
            single_tower=single_tower,
        )
        print(f"===== Evaluating MARA checkpoint: {ckpt_path} =====")
        rows = [
            evaluate_category(
                args=args,
                category=category,
                clip_model=clip_model,
                prompt_learner=prompt_learner,
                dino_model=dino_model,
                model=model,
                mara_agent=mara_agent,
                device=device,
                result_dir=os.path.join(dataset_result_dir, epoch_name),
                single_tower=single_tower,
            )
            for category in categories
        ]
        summary = write_results(dataset_result_dir, epoch_name, rows)
        summary["ckpt_path"] = ckpt_path
        summaries.append(summary)
        print(
            f"MARA {epoch_name}: "
            f"PRO base={summary['mean_base_PRO']:.5f}, "
            f"mara={summary['mean_mara_PRO']:.5f}, "
            f"delta={summary['mean_delta_PRO']:.5f} | "
            f"F1 base={summary['mean_base_F1']:.5f}, "
            f"mara={summary['mean_mara_F1']:.5f}, "
            f"delta={summary['mean_delta_F1']:.5f}, "
            f"gate={summary['mean_gate']:.5f}, "
            f"gate_roi={summary['mean_gate_roi']:.5f}, "
            f"attempt={summary['mean_refine_attempts']:.3f}, "
            f"refine={summary['mean_refine_steps']:.3f}, "
            f"reject={summary['mean_rejected_steps']:.3f}, "
            f"pred_gain={summary['mean_pred_gain']:.5f}, "
            f"accept={summary['mean_accept']:.5f}, "
            f"oracle_dPRO={summary['mean_oracle_delta_PRO']:.5f}, "
            f"identity_max={summary['max_identity_error']:.8f}"
        )

    ranking_file = os.path.join(dataset_result_dir, "mara_epoch_ranking.csv")
    with open(ranking_file, "w", newline="", encoding="utf-8") as f:
        metric_names = ("F1", "I_AUROC", "P_AUROC", "PRO")
        fieldnames = ["epoch"]
        for prefix in ("base", "mara", "delta", "oracle", "oracle_delta"):
            fieldnames.extend([f"mean_{prefix}_{metric_name}" for metric_name in metric_names])
        fieldnames.append("mean_gate")
        fieldnames.append("mean_gate_roi")
        fieldnames.append("mean_changed_ratio")
        fieldnames.append("mean_refine_steps")
        fieldnames.append("mean_refine_attempts")
        fieldnames.append("mean_rejected_steps")
        fieldnames.append("mean_pred_gain")
        fieldnames.append("mean_pred_gain_lower")
        fieldnames.append("mean_accept")
        fieldnames.append("mean_quality_gain")
        fieldnames.append("mean_base_degradation_rate")
        fieldnames.append("mean_negative_gain_accept_rate")
        fieldnames.append("mean_identity_error")
        fieldnames.append("max_identity_error")
        fieldnames.append("ckpt_path")
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summaries)

    best = max(summaries, key=lambda x: x["mean_mara_PRO"])
    print("=" * 60)
    print(
        f"Best MARA checkpoint by mean_PRO: {best['epoch']} | "
        f"Base_PRO={best['mean_base_PRO']:.5f}, "
        f"MARA_PRO={best['mean_mara_PRO']:.5f}, "
        f"Delta_PRO={best['mean_delta_PRO']:.5f}, "
        f"MARA_F1={best['mean_mara_F1']:.5f}, "
        f"Gate={best['mean_gate']:.5f}, "
        f"GateROI={best['mean_gate_roi']:.5f}, "
        f"Attempt={best['mean_refine_attempts']:.3f}, "
        f"Refine={best['mean_refine_steps']:.3f}, "
        f"Reject={best['mean_rejected_steps']:.3f}, "
        f"PredGain={best['mean_pred_gain']:.5f}, "
        f"Accept={best['mean_accept']:.5f}, "
        f"OracleDeltaPRO={best['mean_oracle_delta_PRO']:.5f}, "
        f"IdentityMax={best['max_identity_error']:.8f}"
    )
    print(f"Best checkpoint file: {best['ckpt_path']}")


if __name__ == "__main__":
    main()
