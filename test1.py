import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train_up import (
    set_seed,
    parse_visual_layers,
    prepare_data,
    create_clip_model,
    load_pipeline_from_checkpoint,
    get_anomaly_map,
)


# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------

def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _resize_2d(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr).float()[None, None]
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def _normalize_score_map(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _softmax_prob_2ch(arr: np.ndarray) -> np.ndarray:
    # arr: [2, H, W]
    x = torch.from_numpy(arr).float().unsqueeze(0)  # [1,2,H,W]
    x = torch.softmax(x, dim=1)[0].cpu().numpy()
    return x


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# -----------------------------------------------------------------------------
# debug parsing helpers
# -----------------------------------------------------------------------------

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


def _to_2d_map(x, batch_index: int = 0, prefer_channel: int = 1) -> Optional[np.ndarray]:
    arr = _to_numpy(x)
    if arr is None:
        return None

    if arr.ndim == 4:
        # 常见情况 1: [B, C, H, W]
        if arr.shape[0] > batch_index and arr.shape[2] > 8 and arr.shape[3] > 8:
            arr_b = arr[batch_index]
            if arr_b.shape[0] == 1:
                arr = arr_b[0]
            elif arr_b.shape[0] >= 2:
                ch = min(prefer_channel, arr_b.shape[0] - 1)
                arr = arr_b[ch]
            else:
                arr = arr_b.mean(0)
        else:
            # 兜底：不把第一维强行视作 batch
            # 尝试把它看成 [N, C, H, W] / [C, N, H, W] 中的某种容器，优先取第一个切片
            # 再从第二维里选 channel
            arr0 = arr[0]
            if arr0.ndim == 3 and arr0.shape[1] > 8 and arr0.shape[2] > 8:
                if arr0.shape[0] == 1:
                    arr = arr0[0]
                elif arr0.shape[0] >= 2:
                    ch = min(prefer_channel, arr0.shape[0] - 1)
                    arr = arr0[ch]
                else:
                    arr = arr0.mean(0)
            else:
                return None

    elif arr.ndim == 3:
        # [B, H, W]
        if arr.shape[0] > batch_index and arr.shape[1] > 8 and arr.shape[2] > 8 and arr.shape[0] > 4:
            arr = arr[batch_index]
        # [C, H, W]
        elif arr.shape[0] in (1, 2, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
            ch = min(prefer_channel, arr.shape[0] - 1)
            arr = arr[ch]
        # [H, W, C]
        elif arr.shape[-1] in (1, 2, 3, 4) and arr.shape[0] > 8 and arr.shape[1] > 8:
            ch = min(prefer_channel, arr.shape[-1] - 1)
            arr = arr[..., ch]
        else:
            return None

    elif arr.ndim != 2:
        return None

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _convert_to_map_list(value, batch_index: int = 0) -> List[np.ndarray]:
    if value is None:
        return []

    maps: List[np.ndarray] = []

    if isinstance(value, dict):
        def _key_sort(k):
            nums = "".join(ch if ch.isdigit() else " " for ch in str(k)).split()
            return int(nums[0]) if nums else 10**9

        for _, v in sorted(value.items(), key=lambda kv: _key_sort(kv[0])):
            m = _to_2d_map(v, batch_index=batch_index, prefer_channel=1)
            if m is not None:
                maps.append(m)
        return maps

    if isinstance(value, (list, tuple)):
        for v in value:
            m = _to_2d_map(v, batch_index=batch_index, prefer_channel=1)
            if m is not None:
                maps.append(m)
        return maps

    m = _to_2d_map(value, batch_index=batch_index, prefer_channel=1)
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
    m = _to_2d_map(value, batch_index=batch_index, prefer_channel=1)
    if m is not None:
        return m

    for item in fallback_list:
        m = _to_2d_map(item, batch_index=batch_index, prefer_channel=1)
        if m is not None:
            return m
    return None


def _extract_mask_2d(mask, batch_index: int = 0) -> np.ndarray:
    arr = _to_numpy(mask)
    if arr.ndim == 4:      # [B,1,H,W]
        arr = arr[batch_index, 0]
    elif arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = arr[0]
        else:              # [B,H,W]
            arr = arr[batch_index]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")

    arr = arr.astype(np.float32)
    if arr.max() > 1:
        arr = arr / (arr.max() + 1e-8)
    return np.clip(arr, 0.0, 1.0)


# -----------------------------------------------------------------------------
# metric extraction
# -----------------------------------------------------------------------------

def _infer_token_hw(layer_maps: List[np.ndarray], default_hw: Tuple[int, int]) -> Tuple[int, int]:
    if layer_maps:
        return tuple(layer_maps[-1].shape)
    return default_hw


def _extract_token_margin_map(
    debug,
    anomaly_map_cross_modal,
    anomaly_map,
    batch_index: int,
    token_hw: Tuple[int, int],
) -> np.ndarray:
    # 1) direct margin-like map in debug
    direct_margin_keys = [
        "margin_map",
        "token_margin",
        "margin_score",
        "token_margin_map",
    ]
    value = _find_debug_value(debug, direct_margin_keys)
    m = _to_2d_map(value, batch_index=batch_index, prefer_channel=1)
    if m is not None:
        if m.shape != token_hw:
            m = _resize_2d(m, token_hw)
        return m.astype(np.float32)

    # 2) explicit abnormal/normal score maps in debug
    abn_keys = [
        "abnormal_score",
        "abn_score",
        "abnormal_map",
        "abn_map",
        "sim_abnormal",
        "match_abnormal",
        "positive_map",
    ]
    norm_keys = [
        "normal_score",
        "norm_score",
        "normal_map",
        "norm_map",
        "sim_normal",
        "match_normal",
        "negative_map",
    ]
    abn_val = _find_debug_value(debug, abn_keys)
    norm_val = _find_debug_value(debug, norm_keys)
    abn_map = _to_2d_map(abn_val, batch_index=batch_index, prefer_channel=1)
    norm_map = _to_2d_map(norm_val, batch_index=batch_index, prefer_channel=0)
    if abn_map is not None and norm_map is not None:
        if abn_map.shape != token_hw:
            abn_map = _resize_2d(abn_map, token_hw)
        if norm_map.shape != token_hw:
            norm_map = _resize_2d(norm_map, token_hw)
        return (abn_map - norm_map).astype(np.float32)

    # 3) fallback: use cross-modal 2ch map -> abnormal prob - normal prob
    cm = _to_numpy(anomaly_map_cross_modal)
    if cm is not None and cm.ndim == 4 and cm.shape[1] >= 2:
        cm_one = cm[batch_index]  # [C,H,W]
        # if not in [0,1], treat as logits
        if float(cm_one.min()) < 0.0 or float(cm_one.max()) > 1.0:
            prob = _softmax_prob_2ch(cm_one[:2])
        else:
            prob = cm_one[:2]
        abn_map = prob[1]
        norm_map = prob[0]
        if abn_map.shape != token_hw:
            abn_map = _resize_2d(abn_map, token_hw)
        if norm_map.shape != token_hw:
            norm_map = _resize_2d(norm_map, token_hw)
        return (abn_map - norm_map).astype(np.float32)

    # 4) final fallback: use final anomaly map probability proxy, margin = 2p-1
    final_map = _extract_final_map(
        debug,
        batch_index=batch_index,
        fallback_list=[anomaly_map_cross_modal, anomaly_map],
    )
    if final_map is None:
        raise RuntimeError("Failed to extract token margin map.")
    final_map = _normalize_score_map(final_map)
    if final_map.shape != token_hw:
        final_map = _resize_2d(final_map, token_hw)
    return (2.0 * final_map - 1.0).astype(np.float32)


def _extract_final_prob_map(
    debug,
    anomaly_map_cross_modal,
    anomaly_map,
    batch_index: int,
) -> np.ndarray:
    final_map = _extract_final_map(
        debug,
        batch_index=batch_index,
        fallback_list=[anomaly_map_cross_modal, anomaly_map],
    )
    if final_map is None:
        raise RuntimeError("Failed to extract final anomaly map.")
    return _normalize_score_map(final_map)


def _extract_global_margin(logits_batch, batch_index: int) -> float:
    logits = _to_numpy(logits_batch)
    if logits.ndim == 3:
        logits = logits.squeeze(1)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise RuntimeError(f"Unexpected global logits shape: {logits.shape}")
    return float(logits[batch_index, 1] - logits[batch_index, 0])


# -----------------------------------------------------------------------------
# evaluation core
# -----------------------------------------------------------------------------

def evaluate_table2_for_one_model(
    *,
    name: str,
    visual_backbone: str,
    bundle,
    clip_model,
    loader,
    device,
    args,
) -> Dict[str, float]:
    token_scores_all: List[np.ndarray] = []
    token_labels_all: List[np.ndarray] = []

    abnormal_margin_sum = 0.0
    abnormal_margin_cnt = 0
    normal_margin_sum = 0.0
    normal_margin_cnt = 0

    normal_alarm_cnt = 0
    normal_sample_cnt = 0

    global_margin_abn: List[float] = []
    global_margin_norm: List[float] = []

    sample_cnt = 0
    abnormal_sample_cnt = 0

    with torch.no_grad():
        for batch_idx, image_info in enumerate(loader):
            outputs = get_anomaly_map(
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

            anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score, debug = outputs
            gt_labels = _to_numpy(image_info["is_anomaly"]).astype(np.int64).reshape(-1)

            batch_size = len(gt_labels)
            for i in range(batch_size):
                sample_cnt += 1
                gt_label = int(gt_labels[i])

                mask_2d = _extract_mask_2d(mask, batch_index=i)
                layer_maps = _extract_layer_maps(debug, batch_index=i)
                token_hw = _infer_token_hw(layer_maps, default_hw=(args.token_grid, args.token_grid))

                final_prob_map = _extract_final_prob_map(
                    debug=debug,
                    anomaly_map_cross_modal=anomaly_map_cross_modal,
                    anomaly_map=anomaly_map,
                    batch_index=i,
                )

                token_margin_map = _extract_token_margin_map(
                    debug=debug,
                    anomaly_map_cross_modal=anomaly_map_cross_modal,
                    anomaly_map=anomaly_map,
                    batch_index=i,
                    token_hw=token_hw,
                )

                token_gt = _resize_2d(mask_2d, token_hw)
                token_gt = (token_gt > 0.5)

                token_scores_all.append(token_margin_map.reshape(-1))
                token_labels_all.append(token_gt.reshape(-1).astype(np.int64))

                if token_gt.any():
                    abnormal_margin_sum += float(token_margin_map[token_gt].sum())
                    abnormal_margin_cnt += int(token_gt.sum())

                if (~token_gt).any():
                    normal_margin_sum += float(token_margin_map[~token_gt].sum())
                    normal_margin_cnt += int((~token_gt).sum())

                gm = _extract_global_margin(global_anomaly_score, batch_index=i)
                if gt_label == 1:
                    abnormal_sample_cnt += 1
                    global_margin_abn.append(gm)
                else:
                    global_margin_norm.append(gm)
                    normal_sample_cnt += 1
                    if float(final_prob_map.max()) > args.far_tau:
                        normal_alarm_cnt += 1

    token_scores_all = np.concatenate(token_scores_all, axis=0) if token_scores_all else np.array([], dtype=np.float32)
    token_labels_all = np.concatenate(token_labels_all, axis=0) if token_labels_all else np.array([], dtype=np.int64)

    token_auroc = _binary_auc(token_labels_all, token_scores_all)
    token_auroc = float("nan") if math.isnan(token_auroc) else 100.0 * token_auroc

    mean_abnormal_margin = abnormal_margin_sum / max(1, abnormal_margin_cnt)
    mean_normal_margin = normal_margin_sum / max(1, normal_margin_cnt)
    margin_gap = mean_abnormal_margin - mean_normal_margin

    normal_far = 100.0 * normal_alarm_cnt / max(1, normal_sample_cnt)

    gm_abn = float(np.mean(global_margin_abn)) if global_margin_abn else float("nan")
    gm_norm = float(np.mean(global_margin_norm)) if global_margin_norm else float("nan")
    global_margin = gm_abn - gm_norm if (not math.isnan(gm_abn) and not math.isnan(gm_norm)) else float("nan")

    return {
        "Model": name,
        "Token-AUROC (%)": token_auroc,
        "Margin Gap": margin_gap,
        f"Normal FAR@{args.far_tau:.2f} (%)": normal_far,
        "Global Margin": global_margin,
        "_num_samples": sample_cnt,
        "_num_abnormal_samples": abnormal_sample_cnt,
        "_num_normal_samples": normal_sample_cnt,
        "_num_tokens": int(token_labels_all.size),
        "_num_abnormal_tokens": int(token_labels_all.sum()) if token_labels_all.size > 0 else 0,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="", help="manual device override, e.g. cuda:1 or cpu")
    parser.add_argument("--cuda_id", type=int, default=0, help="cuda device id when --device is empty")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--category", type=str, default="ALL")
    parser.add_argument("--export_split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--dino_repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--dino_bottleneck", type=int, default=256)

    parser.add_argument("--visual_layers", type=parse_visual_layers, default=(5, 11, 17, 23))
    parser.add_argument("--text_source", type=str, default="prompt_learner", choices=["prompt_learner", "ensemble"])

    parser.add_argument("--clip_ckpt", type=str, default="", help="checkpoint trained with --visual_backbone clip")
    parser.add_argument("--dino_ckpt", type=str, default="", help="checkpoint trained with --visual_backbone dino")

    parser.add_argument("--far_tau", type=float, default=0.5, help="threshold tau for Normal FAR@tau")
    parser.add_argument("--token_grid", type=int, default=32, help="fallback token grid size if layer map size is unavailable")
    parser.add_argument("--out_dir", type=str, default="./table2_eval")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.clip_ckpt and not args.dino_ckpt:
        raise ValueError("Please provide at least one checkpoint: --clip_ckpt and/or --dino_ckpt")

    if torch.cuda.is_available():
        device_str = args.device if args.device else f"cuda:{args.cuda_id}"
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    clip_model = create_clip_model(args, device)

    loader = prepare_data(
        dataset_name=args.dataset,
        category=args.category,
        batch_size=args.batch_size,
        split_name=args.export_split,
        image_size=args.image_size,
        shuffle=False,
    )

    jobs = []
    if args.clip_ckpt:
        jobs.append(("CLIP-Visual", "clip", args.clip_ckpt))
    if args.dino_ckpt:
        jobs.append(("DINO-Visual", "dino", args.dino_ckpt))

    results = []
    for model_name, visual_backbone, ckpt_path in jobs:
        print(f"\n[eval] {model_name} | ckpt={ckpt_path}")
        bundle = load_pipeline_from_checkpoint(
            ckpt_path=ckpt_path,
            visual_backbone=visual_backbone,
            args=args,
            device=device,
            clip_model=clip_model,
        )
        metrics = evaluate_table2_for_one_model(
            name=model_name,
            visual_backbone=visual_backbone,
            bundle=bundle,
            clip_model=clip_model,
            loader=loader,
            device=device,
            args=args,
        )
        results.append(metrics)

    json_path = os.path.join(args.out_dir, "table2_metrics.json")
    tsv_path = os.path.join(args.out_dir, "table2_metrics.tsv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    far_key = f"Normal FAR@{args.far_tau:.2f} (%)"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("Model\tToken-AUROC (%)\tMargin Gap\t" + far_key + "\tGlobal Margin\n")
        for row in results:
            f.write(
                f"{row['Model']}\t"
                f"{row['Token-AUROC (%)']:.4f}\t"
                f"{row['Margin Gap']:.6f}\t"
                f"{row[far_key]:.4f}\t"
                f"{row['Global Margin']:.6f}\n"
            )

    print("\n=== Table 2 metrics ===")
    for row in results:
        print(
            f"{row['Model']}: "
            f"Token-AUROC={row['Token-AUROC (%)']:.4f}, "
            f"Margin Gap={row['Margin Gap']:.6f}, "
            f"{far_key}={row[far_key]:.4f}, "
            f"Global Margin={row['Global Margin']:.6f}"
        )

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TSV : {tsv_path}")


if __name__ == "__main__":
    main()