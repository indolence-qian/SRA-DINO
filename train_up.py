import argparse
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

from CLIP.adapter import CLIP_Inplanted as model_adapter
from CLIP.clip import create_model
from Datasets import DATASET_REGISTRY
from tools.bottleneckAdapter import install_bottleneck_adapters_into_dino
from tools.dino_single_tower import collate_anomaly_batch
from tools.loss import BinaryDiceLoss, FocalLoss
from tools.promptLearner import AnomalyCLIP_PromptLearner
from tools.semantic_anchor import build_semantic_anchor_aligner, encode_prompt_features

try:
    from tools.utils_up import (
        compute_reward_and_weight,
        dice_loss_per_sample,
        get_anomaly_map,
        save_comparison_panel,
        save_debug_visuals,
    )
except Exception:
    from utils import (
        compute_reward_and_weight,
        dice_loss_per_sample,
        get_anomaly_map,
        save_comparison_panel,
        save_debug_visuals,
    )


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def grad_norm_stats(named_params: Iterable[Tuple[str, torch.nn.Parameter]]) -> Tuple[float, float, int]:
    total, cnt, max_abs = 0.0, 0, 0.0
    for _, p in named_params:
        if not p.requires_grad or p.grad is None:
            continue
        grad = p.grad.detach()
        total += grad.norm(2).item() ** 2
        max_abs = max(max_abs, grad.abs().max().item())
        cnt += 1
    return (math.sqrt(total) if cnt else 0.0), max_abs, cnt


USE_CUDA = torch.cuda.is_available()
DEFAULT_LOADER_KWARGS = {"num_workers": 0, "pin_memory": True} if USE_CUDA else {}


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def parse_visual_layers(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def prepare_data(
    dataset_name: str,
    category: str,
    batch_size: int,
    split_name: str = "test",
    image_size: int = 512,
    shuffle: bool = True,
    loader_kwargs: Optional[Dict] = None,
):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    split_name = split_name.lower()

    split_map = {
        "train": getattr(split_cls, "TRAIN", None),
        "test": getattr(split_cls, "TEST", None),
        "val": getattr(split_cls, "VAL", None),
        "valid": getattr(split_cls, "VAL", None),
    }
    split_value = split_map.get(split_name)
    if split_value is None:
        available = [name for name, value in split_map.items() if value is not None]
        raise ValueError(f"Split '{split_name}' is unavailable for dataset '{dataset_name}'. Available: {available}")

    dataset = dataset_cls(
        source=root_path,
        split=split_value,
        classname=category,
        resize=image_size,
        imagesize=image_size,
    )

    data_loader_kwargs = dict(loader_kwargs or DEFAULT_LOADER_KWARGS)
    data_loader_kwargs.setdefault("collate_fn", collate_anomaly_batch)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **data_loader_kwargs,
    )

    print(f"Loaded [{dataset_name}] split={split_name} category={category}, size={len(dataset)}")
    return loader


def infer_visual_dim(clip_model: torch.nn.Module, visual_backbone: str) -> int:
    if visual_backbone == "dino":
        return 1024
    visual = clip_model.visual
    width = getattr(getattr(visual, "transformer", None), "width", None)
    if width is not None:
        return int(width)
    output_dim = getattr(visual, "output_dim", None)
    if output_dim is not None:
        return int(output_dim)
    return 1024


# -----------------------------------------------------------------------------
# Model creation / loading
# -----------------------------------------------------------------------------

def create_clip_model(args, device: torch.device) -> torch.nn.Module:
    clip_model = create_model(
        model_name=args.clip_model_name,
        img_size=args.image_size,
        device=device,
        pretrained=args.clip_pretrained,
        require_pretrained=True,
    )
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)
    clip_model.to(device)
    return clip_model


def create_prompt_learner(clip_model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    design_details = {
        "Prompt_length": 20,
        "learnabel_text_embedding_length": 4,
        "learnabel_text_embedding_depth": 1,
    }
    prompt_learner = AnomalyCLIP_PromptLearner(
        clip_model.to("cpu"),
        design_details=design_details,
        classname="object",
    )
    prompt_learner.to(device)
    prompt_learner.train()
    clip_model.to(device)
    return prompt_learner


def create_visual_backbone_for_name(args, device: torch.device, visual_backbone: str):
    dino_model = None
    dino_adapters = None

    if visual_backbone == "dino":
        dino_model = torch.hub.load(
            args.dino_repo_dir,
            args.dino_model_name,
            source="local",
            weights=args.dino_weights,
        )
        dino_model.eval()
        for param in dino_model.parameters():
            param.requires_grad_(False)

        dino_adapters, _ = install_bottleneck_adapters_into_dino(
            dino_model,
            layers=tuple(args.visual_layers),
            bottleneck=args.dino_bottleneck,
        )
        dino_adapters.train()
        dino_model.to(device)

    return dino_model, dino_adapters


def create_adapter_model(clip_model: torch.nn.Module, device: torch.device, visual_backbone: str):
    visual_dim = infer_visual_dim(clip_model, visual_backbone)
    model = model_adapter(c_in=visual_dim, device=device)
    model.to(device)
    model.train()
    return model


def set_trainable_by_name(module: torch.nn.Module, keywords: List[str]) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    for name, param in module.named_parameters():
        if any(keyword in name for keyword in keywords):
            param.requires_grad_(True)


def build_optimizer(
    model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    dino_adapters: Optional[torch.nn.Module],
    semantic_anchor_aligner: Optional[torch.nn.Module],
    args,
) -> torch.optim.Optimizer:
    param_groups = []

    adapter_params = [param for param in model.parameters() if param.requires_grad]
    if adapter_params:
        param_groups.append({
            "params": adapter_params,
            "lr": args.adapter_lr,
            "weight_decay": args.adapter_weight_decay,
        })

    if dino_adapters is not None:
        dino_adapter_params = [param for param in dino_adapters.parameters() if param.requires_grad]
        if dino_adapter_params:
            param_groups.append({
                "params": dino_adapter_params,
                "lr": args.dino_adapter_lr,
                "weight_decay": args.adapter_weight_decay,
            })

    prompt_params = [param for param in prompt_learner.parameters() if param.requires_grad]
    if prompt_params:
        param_groups.append({
            "params": prompt_params,
            "lr": args.prompt_lr,
            "weight_decay": args.prompt_weight_decay,
        })

    if not param_groups:
        raise RuntimeError("No trainable parameters were found for the optimizer.")

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def save_checkpoint(
    epoch: int,
    save_dir: str,
    model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    dino_adapters: Optional[torch.nn.Module],
    semantic_anchor_aligner: Optional[torch.nn.Module],
    semantic_anchor_metadata: Dict,
    args,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "epoch": epoch,
        "visual_backbone": args.visual_backbone,
        "visual_layers": list(args.visual_layers),
        "cls_token_adapter": model.cls_token_adapter.state_dict(),
        "patch_token_adapter": model.patch_token_adapter.state_dict(),
        "prompt_adapter": model.prompt_adapter.state_dict(),
        "prompt_learner": prompt_learner.state_dict(),
        "semantic_anchor_config": {
            "enabled": semantic_anchor_aligner is not None,
            "path": args.semantic_anchor_path,
            "weight": args.semantic_anchor_weight,
            "margin": args.semantic_anchor_margin,
            "separation_weight": args.semantic_anchor_separation_weight,
            "direction_weight": args.semantic_anchor_direction_weight,
            "pair_weight": args.semantic_anchor_pair_weight,
            "bank_weight": args.semantic_anchor_bank_weight,
            "bank_temperature": args.semantic_anchor_bank_temperature,
            "semantic_init": not args.disable_semantic_anchor_init,
            "projector_lr": None,
            "metadata": semantic_anchor_metadata,
        },
    }
    if dino_adapters is not None:
        payload["dino_adapters"] = dino_adapters.state_dict()
    if semantic_anchor_aligner is not None:
        payload["semantic_anchor_aligner"] = semantic_anchor_aligner.state_dict()
    ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_pipeline_from_checkpoint(
    ckpt_path: str,
    visual_backbone: str,
    args,
    device: torch.device,
    clip_model: torch.nn.Module,
):
    model = create_adapter_model(clip_model, device, visual_backbone)
    prompt_learner = create_prompt_learner(clip_model, device)
    dino_model, dino_adapters = create_visual_backbone_for_name(args, device, visual_backbone)

    payload = torch.load(ckpt_path, map_location=device)
    model.cls_token_adapter.load_state_dict(payload["cls_token_adapter"], strict=True)
    model.patch_token_adapter.load_state_dict(payload["patch_token_adapter"], strict=True)
    model.prompt_adapter.load_state_dict(payload["prompt_adapter"], strict=True)
    if "prompt_learner" in payload:
        prompt_learner.load_state_dict(payload["prompt_learner"], strict=False)
    if dino_adapters is not None and "dino_adapters" in payload:
        dino_adapters.load_state_dict(payload["dino_adapters"], strict=False)

    model.eval()
    prompt_learner.eval()
    if dino_adapters is not None:
        dino_adapters.eval()
    if dino_model is not None:
        dino_model.eval()

    return {
        "model": model,
        "prompt_learner": prompt_learner,
        "dino_model": dino_model,
        "dino_adapters": dino_adapters,
        "payload": payload,
    }


# -----------------------------------------------------------------------------
# Export helpers for supplementary figures
# -----------------------------------------------------------------------------

def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _ensure_2d_axes(axes, rows: int, cols: int):
    axes = np.array(axes, dtype=object)
    if rows == 1 and cols == 1:
        return axes.reshape(1, 1)
    if rows == 1:
        return axes.reshape(1, cols)
    if cols == 1:
        return axes.reshape(rows, 1)
    return axes


def _safe_rel_name(path_str: str, keep_parts: int = 4) -> str:
    p = Path(path_str)
    return str(Path(*p.parts[-keep_parts:])) if len(p.parts) >= keep_parts else str(p)


def _vis_image(image_batch, batch_index: int) -> np.ndarray:
    arr = _to_numpy(image_batch[batch_index])
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] not in (1, 3):
        arr = arr[..., :3]

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _vis_mask(mask, batch_index: int) -> np.ndarray:
    arr = _to_numpy(mask)
    if arr.ndim == 4:      # B,1,H,W
        arr = arr[batch_index, 0]
    elif arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = arr[0]
        else:              # B,H,W
            arr = arr[batch_index]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")

    arr = arr.astype(np.float32)
    if arr.max() > 1:
        arr = arr / (arr.max() + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def _to_2d_map(x, batch_index: int = 0) -> Optional[np.ndarray]:
    arr = _to_numpy(x)
    if arr is None:
        return None

    if arr.ndim == 4:  # B,C,H,W
        arr = arr[batch_index]
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] >= 2:
            arr = arr[1]
        else:
            arr = arr.mean(0)

    elif arr.ndim == 3:
        # B,H,W
        if arr.shape[0] > 4 and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = arr[batch_index]
        # C,H,W
        elif arr.shape[0] in (1, 2, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = arr[1] if arr.shape[0] >= 2 else arr[0]
        # H,W,C
        elif arr.shape[-1] in (1, 2, 3, 4) and arr.shape[0] > 8 and arr.shape[1] > 8:
            arr = arr[..., 1] if arr.shape[-1] >= 2 else arr[..., 0]
        else:
            return None

    elif arr.ndim != 2:
        return None

    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _iter_debug_items(obj, path: str = "root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}"
            yield new_path, v
            yield from _iter_debug_items(v, new_path)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            yield new_path, v
            yield from _iter_debug_items(v, new_path)


def _find_debug_value(debug, substrings: Sequence[str]):
    if debug is None:
        return None
    subs = [s.lower() for s in substrings]
    for path, value in _iter_debug_items(debug):
        path_low = path.lower()
        if any(s in path_low for s in subs):
            return value
    return None


def _convert_to_map_list(value, batch_index: int = 0) -> List[np.ndarray]:
    if value is None:
        return []

    maps: List[np.ndarray] = []

    if isinstance(value, dict):
        def _key_sort(k):
            nums = "".join(ch if ch.isdigit() else " " for ch in str(k)).split()
            return int(nums[0]) if nums else 10**9

        for _, v in sorted(value.items(), key=lambda kv: _key_sort(kv[0])):
            m = _to_2d_map(v, batch_index)
            if m is not None:
                maps.append(m)
        return maps

    if isinstance(value, (list, tuple)):
        for v in value:
            m = _to_2d_map(v, batch_index)
            if m is not None:
                maps.append(m)
        return maps

    m = _to_2d_map(value, batch_index)
    return [m] if m is not None else []


def _extract_layer_maps(debug, batch_index: int = 0) -> List[np.ndarray]:
    candidates = [
        "layerwise_map",
        "layerwise_maps",
        "layer_maps",
        "per_layer",
        "token_response",
        "token_responses",
        "token_maps",
        "patch_maps",
        "visual_maps",
        "maps_by_layer",
    ]
    value = _find_debug_value(debug, candidates)
    maps = _convert_to_map_list(value, batch_index)
    if maps:
        return maps

    # fallback: scan first list/dict-like object that can be converted into >=2 maps
    if debug is not None:
        for _, value in _iter_debug_items(debug):
            if isinstance(value, (list, tuple, dict)):
                maps = _convert_to_map_list(value, batch_index)
                if len(maps) >= 2:
                    return maps
    return []


def _extract_final_map(
    debug,
    batch_index: int = 0,
    fallback_list: Optional[List] = None,
) -> Optional[np.ndarray]:
    fallback_list = fallback_list or []

    candidates = [
        "final_anomaly_map",
        "final_map",
        "fused_map",
        "agg_map",
        "aggregated_map",
        "anomaly_map_final",
        "pred_map",
        "score_map",
    ]
    value = _find_debug_value(debug, candidates)
    m = _to_2d_map(value, batch_index)
    if m is not None:
        return m

    for item in fallback_list:
        m = _to_2d_map(item, batch_index)
        if m is not None:
            return m
    return None


def _normalize_for_display(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(m.min()), float(m.max())
    if mx > mn:
        m = (m - mn) / (mx - mn)
    else:
        m = np.zeros_like(m, dtype=np.float32)
    return np.clip(m, 0.0, 1.0)


def _normalize_for_metric(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(m.min()) < 0.0 or float(m.max()) > 1.0:
        mn, mx = float(m.min()), float(m.max())
        if mx > mn:
            m = (m - mn) / (mx - mn)
        else:
            m = np.zeros_like(m, dtype=np.float32)
    return np.clip(m, 0.0, 1.0)


def _shared_norm(maps: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    valid = [m for m in maps if m is not None]
    if not valid:
        return maps

    mn = min(float(m.min()) for m in valid)
    mx = max(float(m.max()) for m in valid)

    if mx - mn < 1e-8:
        return [np.zeros_like(m) if m is not None else None for m in maps]

    out = []
    for m in maps:
        if m is None:
            out.append(None)
        else:
            out.append(np.clip((m - mn) / (mx - mn), 0.0, 1.0))
    return out


def _resize_2d(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr).float()[None, None]
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def _plot_overlay(ax, image_np: np.ndarray, heatmap_np: np.ndarray, title: str):
    ax.imshow(image_np)
    ax.imshow(heatmap_np, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _anomaly_quality(map_2d: np.ndarray, mask_2d: np.ndarray) -> float:
    m = _normalize_for_metric(map_2d)

    gt = mask_2d.astype(np.float32)
    if gt.shape != m.shape:
        gt = _resize_2d(gt, m.shape)
    gt = (gt > 0.5)

    pos_cnt = int(gt.sum())
    neg_cnt = int((~gt).sum())
    if pos_cnt == 0 or neg_cnt == 0:
        return 0.0

    pos_mean = float(m[gt].mean())
    neg_mean = float(m[~gt].mean())
    sep = pos_mean - neg_mean

    pred = (m >= 0.5)
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum()) + 1e-6
    iou = inter / union

    mass_in = float(m[gt].sum() / (m.sum() + 1e-6))
    peak_gap = float(m[gt].max() - m[~gt].max())

    return 0.40 * sep + 0.25 * iou + 0.20 * mass_in + 0.15 * peak_gap


def _normal_fp_quality(map_2d: np.ndarray) -> float:
    m = _normalize_for_metric(map_2d)
    flat = m.reshape(-1)
    if flat.size == 0:
        return 0.0

    q95 = float(np.quantile(flat, 0.95))
    q99 = float(np.quantile(flat, 0.99))
    area50 = float((flat >= 0.5).mean())
    meanv = float(flat.mean())
    return 0.15 * meanv + 0.35 * q95 + 0.30 * q99 + 0.20 * area50


def _layer_refinement_quality(layer_maps: List[np.ndarray], final_map: np.ndarray, mask_2d: np.ndarray) -> float:
    if len(layer_maps) < 2:
        return 0.0
    first_q = _anomaly_quality(layer_maps[0], mask_2d)
    last_q = _anomaly_quality(final_map, mask_2d)
    return last_q - first_q


def _final_case_score(dino_map: np.ndarray, clip_map: np.ndarray, mask_2d: np.ndarray):
    qd = _anomaly_quality(dino_map, mask_2d)
    qc = _anomaly_quality(clip_map, mask_2d)
    score = (qd - qc) + 0.20 * qd
    return score, qd, qc


def _layer_case_score(
    dino_layers: List[np.ndarray],
    clip_layers: List[np.ndarray],
    dino_final: np.ndarray,
    clip_final: np.ndarray,
    mask_2d: np.ndarray,
):
    base, qd, qc = _final_case_score(dino_final, clip_final, mask_2d)
    rd = _layer_refinement_quality(dino_layers, dino_final, mask_2d)
    rc = _layer_refinement_quality(clip_layers, clip_final, mask_2d)
    layer_bonus = (rd - rc)
    layer_count_bonus = 0.02 * min(len(dino_layers), len(clip_layers))
    score = base + 0.35 * layer_bonus + layer_count_bonus
    return score, qd, qc, rd, rc


def _normal_case_score(dino_map: np.ndarray, clip_map: np.ndarray):
    qd = _normal_fp_quality(dino_map)
    qc = _normal_fp_quality(clip_map)
    score = (qc - qd) + 0.20 * qc
    return score, qd, qc


def _push_topk(pool: List[dict], item: dict, max_keep: int, key: str = "rank_score") -> None:
    pool.append(item)
    pool.sort(key=lambda x: x[key], reverse=True)
    if max_keep > 0 and len(pool) > max_keep:
        del pool[max_keep:]


def _token_descriptors_from_maps(
    layer_maps: List[np.ndarray],
    final_map: Optional[np.ndarray],
    mask_np: np.ndarray,
    out_size: int = 32,
):
    maps = [m for m in layer_maps if m is not None]
    if final_map is not None:
        maps = maps + [final_map]

    if len(maps) < 2:
        return None, None

    feats = [_resize_2d(_normalize_for_metric(m), (out_size, out_size)) for m in maps]
    x = np.stack(feats, axis=-1).reshape(-1, len(feats)).astype(np.float32)

    y = (_resize_2d(mask_np.astype(np.float32), (out_size, out_size)) > 0.5).astype(np.int64).reshape(-1)
    return x, y


def _balanced_indices(y: np.ndarray, max_points: int, rng: np.random.RandomState) -> np.ndarray:
    n = len(y)
    if n <= max_points:
        return np.arange(n)

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    chosen = []

    if len(pos) > 0:
        n_pos = min(len(pos), max(1, max_points // 2))
        chosen.append(rng.choice(pos, size=n_pos, replace=False))

    remaining = max_points - sum(len(c) for c in chosen)
    if remaining > 0 and len(neg) > 0:
        n_neg = min(len(neg), remaining)
        chosen.append(rng.choice(neg, size=n_neg, replace=False))

    if not chosen:
        idx = rng.choice(np.arange(n), size=max_points, replace=False)
    else:
        idx = np.concatenate(chosen, axis=0)
        if len(idx) < max_points:
            rest = np.setdiff1d(np.arange(n), idx)
            if len(rest) > 0:
                extra = rng.choice(rest, size=min(len(rest), max_points - len(idx)), replace=False)
                idx = np.concatenate([idx, extra], axis=0)

    rng.shuffle(idx)
    return idx


def _reduce_to_2d(x: np.ndarray, seed: int = 0) -> np.ndarray:
    x = x.astype(np.float32)
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)

    if x.shape[0] < 3:
        z = np.zeros((x.shape[0], 2), dtype=np.float32)
        z[:, 0] = x[:, 0]
        return z

    if TSNE is not None and x.shape[0] >= 10:
        perplexity = max(5, min(30, (x.shape[0] - 1) // 3))
        tsne = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        return tsne.fit_transform(x)

    # PCA fallback
    x0 = x - x.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    z = x0 @ vh[:2].T
    if z.shape[1] == 1:
        z = np.concatenate([z, np.zeros((z.shape[0], 1), dtype=z.dtype)], axis=1)
    return z[:, :2]


def _run_compare_forward(bundle, visual_backbone: str, clip_model, image_info, device, batch_idx, args):
    anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score, debug = get_anomaly_map(
        clip_model=clip_model,
        image_info=image_info,
        device=device,
        model=bundle["model"],
        Dino_model=bundle["dino_model"],
        prompt_learner=bundle["prompt_learner"],
        idx=batch_idx,
        visual_backbone=visual_backbone,
        visual_layers=args.visual_layers,
        text_source=args.text_source,
        return_debug=True,
    )
    return {
        "anomaly_map": anomaly_map,
        "mask": mask,
        "anomaly_map_cross_modal": anomaly_map_cross_modal,
        "global_anomaly_score": global_anomaly_score,
        "debug": debug,
    }


def save_final_grid_figure(cases: List[dict], save_path: str, title: str = "") -> bool:
    if not cases:
        return False

    rows, cols = len(cases), 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = _ensure_2d_axes(axes, rows, cols)

    for r, case in enumerate(cases):
        dino_map, clip_map = _shared_norm([case["dino_final"], case["clip_final"]])
        image_np = case["image_np"]
        mask_np = case["mask_np"]

        axes[r, 0].imshow(image_np)
        axes[r, 0].set_title("Image", fontsize=10)
        axes[r, 0].set_ylabel(f"{r+1}: {case['sample_name']}", fontsize=10)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(mask_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r, 1].set_title("GT", fontsize=10)
        axes[r, 1].axis("off")

        _plot_overlay(axes[r, 2], image_np, dino_map, "DINO-Visual")
        _plot_overlay(axes[r, 3], image_np, clip_map, "CLIP-Visual")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_normal_grid_figure(cases: List[dict], save_path: str, title: str = "") -> bool:
    if not cases:
        return False

    rows, cols = len(cases), 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = _ensure_2d_axes(axes, rows, cols)

    for r, case in enumerate(cases):
        dino_map, clip_map = _shared_norm([case["dino_final"], case["clip_final"]])
        image_np = case["image_np"]

        axes[r, 0].imshow(image_np)
        axes[r, 0].set_title("Image", fontsize=10)
        axes[r, 0].set_ylabel(f"{r+1}: {case['sample_name']}", fontsize=10)
        axes[r, 0].axis("off")

        _plot_overlay(axes[r, 1], image_np, dino_map, "DINO-Visual")
        _plot_overlay(axes[r, 2], image_np, clip_map, "CLIP-Visual")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_layerwise_montage_figure(
    cases: List[dict],
    save_path: str,
    layer_ids: Sequence[int],
    title: str = "",
) -> bool:
    if not cases:
        return False

    n_layers = len(layer_ids)
    cols = 3 + n_layers  # Image | GT | layers... | Final
    rows = 2 * len(cases)

    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.1 * rows))
    axes = _ensure_2d_axes(axes, rows, cols)

    for case_idx, case in enumerate(cases):
        image_np = case["image_np"]
        mask_np = case["mask_np"]

        dino_layers = list(case["dino_layers"])
        clip_layers = list(case["clip_layers"])
        dino_final = case["dino_final"]
        clip_final = case["clip_final"]

        dino_layers = dino_layers[:n_layers] + [None] * max(0, n_layers - len(dino_layers))
        clip_layers = clip_layers[:n_layers] + [None] * max(0, n_layers - len(clip_layers))

        all_maps = [m for m in dino_layers + clip_layers if m is not None] + [dino_final, clip_final]
        all_maps = _shared_norm(all_maps)

        ptr = 0
        dino_layers_norm = []
        for m in dino_layers:
            if m is None:
                dino_layers_norm.append(None)
            else:
                dino_layers_norm.append(all_maps[ptr])
                ptr += 1

        clip_layers_norm = []
        for m in clip_layers:
            if m is None:
                clip_layers_norm.append(None)
            else:
                clip_layers_norm.append(all_maps[ptr])
                ptr += 1

        dino_final_norm = all_maps[ptr]
        clip_final_norm = all_maps[ptr + 1]

        row_dino = 2 * case_idx
        row_clip = 2 * case_idx + 1

        for row_id, row_name, layer_maps, final_map in [
            (row_dino, f"{case_idx+1}-DINO", dino_layers_norm, dino_final_norm),
            (row_clip, f"{case_idx+1}-CLIP", clip_layers_norm, clip_final_norm),
        ]:
            axes[row_id, 0].imshow(image_np)
            axes[row_id, 0].set_title("Image", fontsize=10)
            axes[row_id, 0].set_ylabel(f"{row_name}\n{case['sample_name']}", fontsize=9)
            axes[row_id, 0].axis("off")

            axes[row_id, 1].imshow(mask_np, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_id, 1].set_title("GT", fontsize=10)
            axes[row_id, 1].axis("off")

            for i, lid in enumerate(layer_ids):
                ax = axes[row_id, 2 + i]
                if layer_maps[i] is None:
                    ax.axis("off")
                    continue
                _plot_overlay(ax, image_np, layer_maps[i], f"Layer {lid}")

            _plot_overlay(axes[row_id, 2 + n_layers], image_np, final_map, "Final")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_tsne_compare_figure(
    dino_x: np.ndarray,
    clip_x: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE of token response descriptors",
) -> bool:
    if dino_x is None or clip_x is None or labels is None:
        return False
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        warnings.warn("Skip t-SNE export: not enough positive/negative token descriptors.")
        return False

    feat_dim = min(dino_x.shape[1], clip_x.shape[1])
    dino_x = dino_x[:, :feat_dim]
    clip_x = clip_x[:, :feat_dim]

    dino_2d = _reduce_to_2d(dino_x)
    clip_2d = _reduce_to_2d(clip_x)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes = np.array(axes).reshape(1, 2)

    for ax, emb, name in [
        (axes[0, 0], dino_2d, "DINO-Visual"),
        (axes[0, 1], clip_2d, "CLIP-Visual"),
    ]:
        for lab, lab_name, alpha in [(0, "Normal token", 0.45), (1, "Anomalous token", 0.75)]:
            idx = labels == lab
            if idx.sum() == 0:
                continue
            ax.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=alpha, label=lab_name)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 1].legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_epoch(
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    loss_focal,
    loss_dice,
    epoch: int,
    anomaly_awareness_loss_list: List[float],
    seg_loss_list: List[float],
    global_anomaly_loss_list: List[float],
    semantic_anchor_loss_list: List[float],
    loss_list: List[float],
    clip_model: torch.nn.Module,
    model: torch.nn.Module,
    dino_model: Optional[torch.nn.Module],
    prompt_learner: torch.nn.Module,
    semantic_anchor_aligner: Optional[torch.nn.Module],
    device: torch.device,
    train_data,
    reward_state,
    args,
):
    model.train()
    prompt_learner.train()
    if dino_model is not None:
        dino_model.eval()

    for idx, image_info in enumerate(train_data):
        need_debug = args.save_debug_every > 0 and (idx % args.save_debug_every == 0)

        outputs = get_anomaly_map(
            clip_model=clip_model,
            image_info=image_info,
            device=device,
            model=model,
            Dino_model=dino_model,
            prompt_learner=prompt_learner,
            idx=idx,
            visual_backbone=args.visual_backbone,
            visual_layers=args.visual_layers,
            text_source=args.text_source,
            return_debug=need_debug,
        )

        if need_debug:
            anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score, debug = outputs
        else:
            anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score = outputs
            debug = None

        anomaly_awareness_loss = loss_focal(anomaly_map, mask) + loss_dice(anomaly_map[:, 1, :, :], mask)

        mask_bhw = mask[:, 0] if mask.dim() == 4 else mask
        seg_focal = loss_focal(anomaly_map_cross_modal, mask_bhw)

        if anomaly_map_cross_modal.min() < 0 or anomaly_map_cross_modal.max() > 1:
            cm_prob = torch.softmax(anomaly_map_cross_modal, dim=1)[:, 1]
        else:
            cm_prob = anomaly_map_cross_modal[:, 1].clamp(0, 1)

        seg_dice_per = dice_loss_per_sample(cm_prob, mask_bhw)

        gt_label = image_info["is_anomaly"].to(device).long()
        logits = global_anomaly_score.squeeze(1) if global_anomaly_score.dim() == 3 else global_anomaly_score
        ce_per_sample = F.cross_entropy(logits, gt_label, reduction="none")

        w_cls, w_seg, r_mean, baseline, dice_per, reward_state = compute_reward_and_weight(
            global_logits=logits,
            anomaly_map_cross_modal=anomaly_map_cross_modal,
            mask=mask,
            gt_label=gt_label,
            reward_state=reward_state,
            w_loc=args.reward_w_loc,
            w_conf=args.reward_w_conf,
            alpha_adv_cls=args.alpha_adv_cls,
            seg_k=args.seg_k,
        )

        if args.freeze_seg_reward:
            w_seg = torch.ones_like(w_seg)

        global_anomaly_loss = (ce_per_sample * w_cls).mean()
        if semantic_anchor_aligner is not None:
            learned_prompt_features = encode_prompt_features(
                clip_model=clip_model,
                prompt_learner=prompt_learner,
                device=device,
            )
            anchor_out = semantic_anchor_aligner(learned_prompt_features)
            semantic_anchor_loss = anchor_out["loss"]
        else:
            semantic_anchor_loss = logits.new_zeros(())
            anchor_out = {
                "normal_similarity": logits.new_zeros(()),
                "anomaly_similarity": logits.new_zeros(()),
                "cross_similarity": logits.new_zeros(()),
                "direction_similarity": logits.new_zeros(()),
                "prompt_pair_similarity": logits.new_zeros(()),
                "adaptive_margin": logits.new_zeros(()),
            }
        seg_dice = (seg_dice_per * w_seg).mean()
        seg_loss = seg_focal + seg_dice
        global_objective = global_anomaly_loss + args.semantic_anchor_weight * semantic_anchor_loss
        loss = args.w_awareness * anomaly_awareness_loss + args.w_seg * seg_loss + args.w_global * global_objective

        anomaly_awareness_loss_list.append(float(anomaly_awareness_loss.item()))
        seg_loss_list.append(float(seg_loss.item()))
        global_anomaly_loss_list.append(float(global_anomaly_loss.item()))
        semantic_anchor_loss_list.append(float(semantic_anchor_loss.item()))
        loss_list.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()

        if idx % 50 == 0:
            g_adapter = grad_norm_stats(model.named_parameters())
            g_prompt = grad_norm_stats(prompt_learner.named_parameters())
            print(
                f"\n[grad] adapter norm={g_adapter[0]:.3e} max={g_adapter[1]:.3e} n={g_adapter[2]} | "
                f"prompt norm={g_prompt[0]:.3e} max={g_prompt[1]:.3e} n={g_prompt[2]}"
            )

        optimizer.step()
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epoch} | Batch {idx + 1}/{len(train_data)} "
            f"| loss: {loss.item():.4f} | aw: {anomaly_awareness_loss.item():.4f} "
            f"| seg(focal+dice_w): {seg_focal.item():.4f}+{seg_dice.item():.4f}={seg_loss.item():.4f} "
            f"| global(w): {global_anomaly_loss.item():.4f} "
            f"| anchor: {semantic_anchor_loss.item():.4f} "
            f"| anchor_sim(n/a/x): {anchor_out['normal_similarity'].item():.3f}/"
            f"{anchor_out['anomaly_similarity'].item():.3f}/{anchor_out['cross_similarity'].item():.3f} "
            f"| anchor_dir/pair/margin: {anchor_out['direction_similarity'].item():.3f}/"
            f"{anchor_out['prompt_pair_similarity'].item():.3f}/{anchor_out['adaptive_margin'].item():.3f} "
            f"| reward_mean: {r_mean:.3f} | baseline: {baseline:.3f} "
            f"| dice mean: {dice_per.mean().item():.3f} "
            f"| w_seg mean/min/max: {w_seg.mean().item():.3f}/{w_seg.min().item():.3f}/{w_seg.max().item():.3f}",
            end="\r",
            flush=True,
        )

        if debug is not None:
            debug_dir = os.path.join(args.result_path, "debug_train", f"epoch_{epoch:03d}")
            os.makedirs(debug_dir, exist_ok=True)
            sample_name = Path(image_info["image_path"][0]).stem
            save_path = os.path.join(debug_dir, f"step_{idx:05d}_{sample_name}_{args.visual_backbone}.png")
            save_debug_visuals(
                image=image_info["image"],
                mask=mask.detach(),
                debug=debug,
                save_path=save_path,
                batch_index=0,
                title_prefix=f"{args.visual_backbone.upper()} ",
            )

    return reward_state


def run_train(args) -> None:
    set_seed(args.seed)
    if torch.cuda.is_available():
        device_str = args.device if args.device else f"cuda:{args.cuda_id}"
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    os.makedirs(args.result_path, exist_ok=True)

    clip_model = create_clip_model(args, device)
    prompt_learner = create_prompt_learner(clip_model, device)
    dino_model, dino_adapters = create_visual_backbone_for_name(args, device, args.visual_backbone)
    model = create_adapter_model(clip_model, device, args.visual_backbone)

    semantic_anchor_aligner = None
    semantic_anchor_metadata: Dict = {}
    if args.semantic_anchor_weight > 0:
        if args.text_source != "prompt_learner":
            raise ValueError("Semantic anchor training requires --text_source prompt_learner.")
        semantic_anchor_aligner, semantic_anchor_metadata = build_semantic_anchor_aligner(
            path=args.semantic_anchor_path,
            prompt_learner=prompt_learner,
            clip_model=clip_model,
            device=device,
            margin=args.semantic_anchor_margin,
            separation_weight=args.semantic_anchor_separation_weight,
            direction_weight=args.semantic_anchor_direction_weight,
            pair_weight=args.semantic_anchor_pair_weight,
            bank_weight=args.semantic_anchor_bank_weight,
            bank_temperature=args.semantic_anchor_bank_temperature,
            initialize_context=not args.disable_semantic_anchor_init,
        )
        semantic_anchor_aligner.train()
        print(
            "External semantic anchors enabled: "
            f"path={args.semantic_anchor_path}, "
            f"generator={semantic_anchor_metadata.get('generator_model', 'unknown')}, "
            f"space={semantic_anchor_metadata.get('anchor_space', 'unknown')}, "
            f"banks={semantic_anchor_metadata.get('normal_bank_size', 0)}/"
            f"{semantic_anchor_metadata.get('anomaly_bank_size', 0)}, "
            f"teacher_pair={semantic_anchor_metadata.get('teacher_pair_similarity', 0.0):.4f}, "
            f"adaptive_margin={semantic_anchor_metadata.get('adaptive_margin', 0.0):.4f}, "
            f"semantic_init={semantic_anchor_metadata.get('semantic_context_initialized', False)}, "
            f"weight={args.semantic_anchor_weight}"
        )

    set_trainable_by_name(model, ["patch_token_adapter", "cls_token_adapter", "prompt_adapter"])
    set_trainable_by_name(prompt_learner, ["ctx_pos", "ctx_neg"])

    train_data = prepare_data(
        dataset_name=args.dataset,
        category=args.category,
        batch_size=args.batch_size,
        split_name=args.train_split,
        image_size=args.image_size,
        shuffle=True,
    )

    optimizer = build_optimizer(
        model,
        prompt_learner,
        dino_adapters,
        semantic_anchor_aligner,
        args,
    )
    total_steps = max(1, args.epoch * len(train_data))
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    reward_state = None

    for epoch in range(args.epoch):
        if epoch == args.prompt_stop_epoch:
            prompt_learner.eval()
            for param in prompt_learner.parameters():
                param.requires_grad_(False)
            if semantic_anchor_aligner is not None:
                semantic_anchor_aligner.eval()
                for param in semantic_anchor_aligner.parameters():
                    param.requires_grad_(False)

        start_time = time.time()
        awareness_loss_list, seg_loss_list, loss_list, global_anomaly_loss_list = [], [], [], []
        semantic_anchor_loss_list = []

        reward_state = train_epoch(
            optimizer=optimizer,
            scheduler=scheduler,
            loss_focal=loss_focal,
            loss_dice=loss_dice,
            epoch=epoch,
            anomaly_awareness_loss_list=awareness_loss_list,
            seg_loss_list=seg_loss_list,
            global_anomaly_loss_list=global_anomaly_loss_list,
            semantic_anchor_loss_list=semantic_anchor_loss_list,
            loss_list=loss_list,
            clip_model=clip_model,
            model=model,
            dino_model=dino_model,
            prompt_learner=prompt_learner,
            semantic_anchor_aligner=semantic_anchor_aligner,
            device=device,
            train_data=train_data,
            reward_state=reward_state,
            args=args,
        )
        print()

        ckpt_dir = os.path.join(args.result_path, "ckpt")
        ckpt_path = save_checkpoint(
            epoch,
            ckpt_dir,
            model,
            prompt_learner,
            dino_adapters,
            semantic_anchor_aligner,
            semantic_anchor_metadata,
            args,
        )

        with open(os.path.join(args.result_path, "loss.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"epoch_{epoch}: "
                f"awareness_loss={np.mean(awareness_loss_list):.6f}\t"
                f"seg_loss={np.mean(seg_loss_list):.6f}\t"
                f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}\t"
                f"semantic_anchor_loss={np.mean(semantic_anchor_loss_list):.6f}\t"
                f"total_loss={np.mean(loss_list):.6f}\n"
            )

        print(
            f"epoch_{epoch}: awareness_loss={np.mean(awareness_loss_list):.6f}, "
            f"seg_loss={np.mean(seg_loss_list):.6f}, "
            f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}, "
            f"semantic_anchor_loss={np.mean(semantic_anchor_loss_list):.6f}, "
            f"total_loss={np.mean(loss_list):.6f}, "
            f"time={time.time() - start_time:.2f}s, ckpt={ckpt_path}"
        )


# -----------------------------------------------------------------------------
# One-click DINO vs CLIP export (4 final figures)
# -----------------------------------------------------------------------------

def run_export_compare(args) -> None:
    if not args.dino_ckpt or not args.clip_ckpt:
        raise ValueError("export_compare mode requires both --dino_ckpt and --clip_ckpt.")

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    if torch.cuda.is_available():
        device_str = args.device if args.device else f"cuda:{args.cuda_id}"
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    os.makedirs(args.compare_out_dir, exist_ok=True)

    clip_model = create_clip_model(args, device)

    dino_bundle = load_pipeline_from_checkpoint(
        ckpt_path=args.dino_ckpt,
        visual_backbone="dino",
        args=args,
        device=device,
        clip_model=clip_model,
    )
    clip_bundle = load_pipeline_from_checkpoint(
        ckpt_path=args.clip_ckpt,
        visual_backbone="clip",
        args=args,
        device=device,
        clip_model=clip_model,
    )

    loader = prepare_data(
        dataset_name=args.dataset,
        category=args.category,
        batch_size=args.batch_size,
        split_name=args.export_split,
        image_size=args.image_size,
        shuffle=False,
    )

    abnormal_candidates: List[dict] = []
    layerwise_candidates: List[dict] = []
    normal_candidates: List[dict] = []

    tsne_dino_all = []
    tsne_clip_all = []
    tsne_y_all = []
    tsne_token_count = 0

    with torch.no_grad():
        for batch_idx, image_info in enumerate(loader):
            dino_out = _run_compare_forward(
                bundle=dino_bundle,
                visual_backbone="dino",
                clip_model=clip_model,
                image_info=image_info,
                device=device,
                batch_idx=batch_idx,
                args=args,
            )

            clip_out = _run_compare_forward(
                bundle=clip_bundle,
                visual_backbone="clip",
                clip_model=clip_model,
                image_info=image_info,
                device=device,
                batch_idx=batch_idx,
                args=args,
            )

            batch_size = image_info["image"].shape[0]

            for sample_idx in range(batch_size):
                img_path = image_info["image_path"][sample_idx]
                sample_name = Path(img_path).stem
                rel_name = _safe_rel_name(img_path)

                image_np = _vis_image(image_info["image"], sample_idx)
                mask_np = _vis_mask(dino_out["mask"], sample_idx)
                gt_label = int(image_info["is_anomaly"][sample_idx].item())

                dino_final = _extract_final_map(
                    dino_out["debug"],
                    batch_index=sample_idx,
                    fallback_list=[dino_out["anomaly_map_cross_modal"], dino_out["anomaly_map"]],
                )
                clip_final = _extract_final_map(
                    clip_out["debug"],
                    batch_index=sample_idx,
                    fallback_list=[clip_out["anomaly_map_cross_modal"], clip_out["anomaly_map"]],
                )

                dino_layers = _extract_layer_maps(dino_out["debug"], batch_index=sample_idx)
                clip_layers = _extract_layer_maps(clip_out["debug"], batch_index=sample_idx)

                if dino_final is None or clip_final is None:
                    print(f"[warn] skip {rel_name}: failed to extract final maps.")
                    continue

                if gt_label == 1:
                    final_rank, qd, qc = _final_case_score(dino_final, clip_final, mask_np)
                    case = {
                        "rank_score": final_rank,
                        "qd": qd,
                        "qc": qc,
                        "image_np": image_np,
                        "mask_np": mask_np,
                        "dino_final": dino_final,
                        "clip_final": clip_final,
                        "dino_layers": dino_layers,
                        "clip_layers": clip_layers,
                        "sample_name": sample_name,
                        "rel_name": rel_name,
                    }
                    _push_topk(abnormal_candidates, case, args.max_compare_cases, key="rank_score")

                    if len(dino_layers) >= 2 and len(clip_layers) >= 2:
                        layer_rank, qd2, qc2, rd, rc = _layer_case_score(
                            dino_layers=dino_layers,
                            clip_layers=clip_layers,
                            dino_final=dino_final,
                            clip_final=clip_final,
                            mask_2d=mask_np,
                        )
                        layer_case = {
                            "rank_score": layer_rank,
                            "qd": qd2,
                            "qc": qc2,
                            "rd": rd,
                            "rc": rc,
                            "image_np": image_np,
                            "mask_np": mask_np,
                            "dino_final": dino_final,
                            "clip_final": clip_final,
                            "dino_layers": dino_layers,
                            "clip_layers": clip_layers,
                            "sample_name": sample_name,
                            "rel_name": rel_name,
                        }
                        _push_topk(layerwise_candidates, layer_case, args.max_compare_cases, key="rank_score")

                    if tsne_token_count < args.tsne_token_cap:
                        dino_x, y = _token_descriptors_from_maps(
                            dino_layers,
                            dino_final,
                            mask_np,
                            out_size=args.tsne_grid,
                        )
                        clip_x, y2 = _token_descriptors_from_maps(
                            clip_layers,
                            clip_final,
                            mask_np,
                            out_size=args.tsne_grid,
                        )

                        if dino_x is not None and clip_x is not None and y is not None and y2 is not None:
                            n = min(len(dino_x), len(clip_x), len(y), len(y2))
                            dino_x = dino_x[:n]
                            clip_x = clip_x[:n]
                            y = y[:n]

                            idx = _balanced_indices(
                                y,
                                max_points=min(args.tsne_per_sample, args.tsne_token_cap - tsne_token_count),
                                rng=rng,
                            )
                            if len(idx) > 0:
                                feat_dim = min(dino_x.shape[1], clip_x.shape[1])
                                tsne_dino_all.append(dino_x[idx, :feat_dim])
                                tsne_clip_all.append(clip_x[idx, :feat_dim])
                                tsne_y_all.append(y[idx])
                                tsne_token_count += len(idx)

                else:
                    normal_rank, qd, qc = _normal_case_score(dino_final, clip_final)
                    normal_case = {
                        "rank_score": normal_rank,
                        "qd": qd,
                        "qc": qc,
                        "image_np": image_np,
                        "mask_np": mask_np,
                        "dino_final": dino_final,
                        "clip_final": clip_final,
                        "dino_layers": dino_layers,
                        "clip_layers": clip_layers,
                        "sample_name": sample_name,
                        "rel_name": rel_name,
                    }
                    _push_topk(normal_candidates, normal_case, args.max_normal_cases, key="rank_score")

    abnormal_candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    layerwise_candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    normal_candidates.sort(key=lambda x: x["rank_score"], reverse=True)

    final_cases = abnormal_candidates[:args.final_num_rows]
    layer_cases = layerwise_candidates[:args.layerwise_num_cases]
    normal_cases = normal_candidates[:args.normal_num_rows]

    final_fig_path = os.path.join(args.compare_out_dir, "fig_final_compare.png")
    layer_fig_path = os.path.join(args.compare_out_dir, "fig_layerwise_compare.png")
    normal_fig_path = os.path.join(args.compare_out_dir, "fig_normal_compare.png")
    tsne_fig_path = os.path.join(args.compare_out_dir, "fig_tsne_compare.png")
    summary_path = os.path.join(args.compare_out_dir, "selected_cases.txt")

    ok_final = save_final_grid_figure(
        final_cases,
        final_fig_path,
        title="Final anomaly-map comparison",
    )
    ok_layer = save_layerwise_montage_figure(
        layer_cases,
        layer_fig_path,
        layer_ids=args.visual_layers,
        title="Layer-wise token-response comparison",
    )
    ok_normal = save_normal_grid_figure(
        normal_cases,
        normal_fig_path,
        title="Comparison on normal samples",
    )

    if tsne_dino_all and tsne_clip_all and tsne_y_all:
        dino_x = np.concatenate(tsne_dino_all, axis=0)
        clip_x = np.concatenate(tsne_clip_all, axis=0)
        y = np.concatenate(tsne_y_all, axis=0)
        ok_tsne = save_tsne_compare_figure(
            dino_x=dino_x,
            clip_x=clip_x,
            labels=y,
            save_path=tsne_fig_path,
            title="t-SNE of token response descriptors",
        )
    else:
        ok_tsne = False

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Final comparison cases ===\n")
        for i, c in enumerate(final_cases):
            f.write(
                f"{i+1}. {c['rel_name']} | rank={c['rank_score']:.6f} | "
                f"DINO_q={c['qd']:.6f} | CLIP_q={c['qc']:.6f}\n"
            )

        f.write("\n=== Layer-wise comparison cases ===\n")
        for i, c in enumerate(layer_cases):
            f.write(
                f"{i+1}. {c['rel_name']} | rank={c['rank_score']:.6f} | "
                f"DINO_q={c['qd']:.6f} | CLIP_q={c['qc']:.6f} | "
                f"DINO_refine={c['rd']:.6f} | CLIP_refine={c['rc']:.6f}\n"
            )

        f.write("\n=== Normal comparison cases ===\n")
        for i, c in enumerate(normal_cases):
            f.write(
                f"{i+1}. {c['rel_name']} | rank={c['rank_score']:.6f} | "
                f"DINO_fp={c['qd']:.6f} | CLIP_fp={c['qc']:.6f}\n"
            )

        f.write("\n=== Figure paths ===\n")
        f.write(f"fig_final_compare: {final_fig_path} | saved={ok_final}\n")
        f.write(f"fig_layerwise_compare: {layer_fig_path} | saved={ok_layer}\n")
        f.write(f"fig_normal_compare: {normal_fig_path} | saved={ok_normal}\n")
        f.write(f"fig_tsne_compare: {tsne_fig_path} | saved={ok_tsne}\n")

    print(f"[done] final figure saved: {ok_final} -> {final_fig_path}")
    print(f"[done] layerwise figure saved: {ok_layer} -> {layer_fig_path}")
    print(f"[done] normal figure saved: {ok_normal} -> {normal_fig_path}")
    print(f"[done] tsne figure saved: {ok_tsne} -> {tsne_fig_path}")
    print(f"[done] case summary: {summary_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "export_compare"])
    parser.add_argument("--result_path", type=str, default="./Result", help="path to save training results")
    parser.add_argument("--compare_out_dir", type=str, default="./compare_outputs", help="export directory for supplementary figures")
    parser.add_argument("--device", type=str, default="cuda:1", help="device")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--dataset", type=str, default="visa", help="dataset name")
    parser.add_argument("--category", type=str, default="ALL", help="dataset category")
    parser.add_argument("--image_size", type=int, default=512, help="input image size")
    parser.add_argument("--epoch", type=int, default=100, help="training epochs")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--visual_backbone", type=str, default="dino", choices=["dino", "clip"], help="visual backbone for training mode")
    parser.add_argument("--visual_layers", type=parse_visual_layers, default=(5, 11, 17, 23), help="comma-separated layer ids, e.g. 5,11,17,23")
    parser.add_argument("--text_source", type=str, default="prompt_learner", choices=["prompt_learner", "ensemble"])

    parser.add_argument("--train_split", type=str, default="test", help="split used for training mode")
    parser.add_argument("--export_split", type=str, default="test", help="split used for export_compare mode")
    parser.add_argument("--save_debug_every", type=int, default=0, help="save a debug panel every N batches during training; 0 disables")

    parser.add_argument("--max_compare_cases", type=int, default=80, help="max anomalous candidates kept for ranking; <=0 means keep all")
    parser.add_argument("--max_normal_cases", type=int, default=50, help="max normal candidates kept for ranking; <=0 means keep all")
    parser.add_argument("--final_num_rows", type=int, default=4, help="number of rows in final anomaly-map comparison figure")
    parser.add_argument("--layerwise_num_cases", type=int, default=2, help="number of abnormal cases shown in layer-wise figure")
    parser.add_argument("--normal_num_rows", type=int, default=4, help="number of rows in normal-sample comparison figure")

    parser.add_argument("--tsne_grid", type=int, default=32, help="downsample grid size used to build token descriptors for t-SNE")
    parser.add_argument("--tsne_per_sample", type=int, default=256, help="maximum token descriptors sampled from one image for t-SNE")
    parser.add_argument("--tsne_token_cap", type=int, default=4000, help="maximum total token descriptors used for t-SNE")

    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--dino_repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--dino_bottleneck", type=int, default=256)

    parser.add_argument("--adapter_lr", type=float, default=1e-5)
    parser.add_argument("--dino_adapter_lr", type=float, default=5e-6)
    parser.add_argument("--prompt_lr", type=float, default=5e-5)
    parser.add_argument("--adapter_weight_decay", type=float, default=1e-2)
    parser.add_argument("--prompt_weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--semantic_anchor_path",
        type=str,
        default="./asset/gemini_semantic_anchors.pt",
        help="Gemini description asset; text banks are re-encoded by the frozen CLIP teacher",
    )
    parser.add_argument(
        "--semantic_anchor_weight",
        type=float,
        default=0.0,
        help="anchor loss weight inside the global objective; 0 disables it",
    )
    parser.add_argument("--semantic_anchor_margin", type=float, default=0.20)
    parser.add_argument("--semantic_anchor_separation_weight", type=float, default=0.50)
    parser.add_argument("--semantic_anchor_direction_weight", type=float, default=1.00)
    parser.add_argument("--semantic_anchor_pair_weight", type=float, default=0.10)
    parser.add_argument("--semantic_anchor_bank_weight", type=float, default=0.25)
    parser.add_argument("--semantic_anchor_bank_temperature", type=float, default=0.07)
    parser.add_argument("--disable_semantic_anchor_init", action="store_true")
    parser.add_argument(
        "--semantic_anchor_projector_lr",
        type=float,
        default=1e-6,
        help="deprecated compatibility option; P0/P1 no longer trains a projector",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--prompt_stop_epoch", type=int, default=10)

    parser.add_argument("--w_awareness", type=float, default=0.25)
    parser.add_argument("--w_seg", type=float, default=0.5)
    parser.add_argument("--w_global", type=float, default=0.25)
    parser.add_argument("--reward_w_loc", type=float, default=1.0)
    parser.add_argument("--reward_w_conf", type=float, default=0.2)
    parser.add_argument("--alpha_adv_cls", type=float, default=0.05)
    parser.add_argument("--seg_k", type=float, default=3.0)
    parser.add_argument("--freeze_seg_reward", action="store_true")

    parser.add_argument("--dino_ckpt", type=str, default="", help="checkpoint trained with --visual_backbone dino")
    parser.add_argument("--clip_ckpt", type=str, default="", help="checkpoint trained with --visual_backbone clip")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "export_compare":
        run_export_compare(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
