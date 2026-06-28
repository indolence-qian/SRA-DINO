#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from CLIP.adapter import CLIP_Inplanted as model_adapter
from CLIP.clip import create_model
from Datasets import DATASET_CLASSES, DATASET_REGISTRY
from tools.bottleneckAdapter import install_bottleneck_adapters_into_dino
from tools.promptLearner import AnomalyCLIP_PromptLearner

try:
    import tools.utils_up as utils_mod
except Exception:
    import utils as utils_mod


# -----------------------------------------------------------------------------
# Compatibility patch for CLIP visual forward(x, out_layers)
# -----------------------------------------------------------------------------

def _extract_module_output(out):
    if isinstance(out, dict):
        for key in ["x", "tokens", "hidden_states", "last_hidden_state"]:
            if key in out and torch.is_tensor(out[key]):
                return out[key]
        raise RuntimeError(f"Unexpected dict output keys: {list(out.keys())}")
    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item):
                return item
        return out[0]
    return out


def _normalize_patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens - tokens.mean(dim=1, keepdim=True)) / (tokens.std(dim=1, keepdim=True) + 1e-6)


def patched_get_feature_clip_vit(
    batch_img: torch.Tensor,
    device: torch.device,
    clip_model: torch.nn.Module,
    layers: Sequence[int] = (5, 11, 17, 23),
):
    visual = clip_model.visual
    if not hasattr(visual, "transformer") or not hasattr(visual.transformer, "resblocks"):
        raise RuntimeError("clip_model.visual does not expose transformer.resblocks.")

    resblocks = visual.transformer.resblocks
    tokens_dict: Dict[int, torch.Tensor] = {}
    handles = []
    batch_size = batch_img.shape[0]

    def _to_bld(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == batch_size and x.shape[0] != batch_size:
            x = x.permute(1, 0, 2).contiguous()
        return x

    for layer_idx in layers:
        def _mk_hook(idx: int):
            def _hook(module, inp, out):
                tokens_dict[idx] = _to_bld(_extract_module_output(out))
            return _hook
        handles.append(resblocks[layer_idx].register_forward_hook(_mk_hook(layer_idx)))

    batch_img = batch_img.to(device, non_blocking=True)
    sig = inspect.signature(visual.forward)
    if "out_layers" in sig.parameters:
        _ = visual(batch_img, out_layers=layers)
    else:
        _ = visual(batch_img)

    for handle in handles:
        handle.remove()

    cls_token_list: List[torch.Tensor] = []
    patch_tokens_list: List[torch.Tensor] = []
    ln_post = getattr(visual, "ln_post", None)

    for layer_idx in layers:
        if layer_idx not in tokens_dict:
            raise RuntimeError(f"Hook did not capture CLIP layer {layer_idx}. Captured={sorted(tokens_dict.keys())}")

        toks = tokens_dict[layer_idx]
        if ln_post is not None:
            try:
                toks = ln_post(toks)
            except Exception:
                pass

        cls = toks[:, 0, :].unsqueeze(1)
        patch = _normalize_patch_tokens(toks[:, 1:, :])
        cls_token_list.append(cls)
        patch_tokens_list.append(patch)

    return cls_token_list, patch_tokens_list


if hasattr(utils_mod, "get_feature_clip_vit"):
    utils_mod.get_feature_clip_vit = patched_get_feature_clip_vit

get_anomaly_map = utils_mod.get_anomaly_map


# -----------------------------------------------------------------------------
# Data / model helpers
# -----------------------------------------------------------------------------

def prepare_data(dataset_name: str, category: str, batch_size: int, image_size: int, num_workers: int = 0):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=image_size,
        imagesize=image_size,
    )
    print(f"Loaded [{dataset_name}] split=test category={category}, size={len(dataset)}")
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
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



def build_models(args, device: torch.device):
    clip_model = create_model(
        model_name=args.clip_model_name,
        img_size=args.image_size,
        device=device,
        pretrained=args.clip_pretrained,
        require_pretrained=True,
    )
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    design_details = {
        "Prompt_length": 20,
        "learnabel_text_embedding_length": 4,
        "learnabel_text_embedding_depth": 1,
    }
    prompt_learner = AnomalyCLIP_PromptLearner(
        clip_model.to("cpu"), design_details=design_details, classname="object"
    )
    prompt_learner.to(device)
    prompt_learner.eval()
    clip_model.to(device)

    dino_model = None
    dino_adapters = None
    if args.visual_backbone == "dino":
        dino_model = torch.hub.load(
            args.dino_repo_dir,
            args.dino_model_name,
            source="local",
            weights=args.dino_weights,
        )
        dino_model.eval()
        for p in dino_model.parameters():
            p.requires_grad_(False)
        dino_adapters, _ = install_bottleneck_adapters_into_dino(
            dino_model, layers=tuple(args.visual_layers), bottleneck=args.dino_bottleneck
        )
        dino_model.to(device)
        dino_adapters.to(device)
        dino_adapters.eval()

    model = model_adapter(c_in=infer_visual_dim(clip_model, args.visual_backbone), device=device)
    model.to(device)
    model.eval()

    return clip_model, prompt_learner, dino_model, dino_adapters, model


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def sorted_checkpoints(weight_path: str) -> List[str]:
    p = Path(weight_path)
    if p.is_file():
        return [str(p)]
    if not p.exists():
        raise FileNotFoundError(f"weight_path not found: {weight_path}")

    files = list(p.glob("*.pth")) + list(p.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under: {weight_path}")

    def extract_num(path: Path) -> Tuple[int, str]:
        m = re.findall(r"(\d+)", path.stem)
        num = int(m[-1]) if m else -1
        return num, path.name

    return [str(x) for x in sorted(files, key=lambda x: extract_num(x))]



def load_checkpoint_flexible(
    ckpt_path: str,
    model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    dino_adapters: Optional[torch.nn.Module],
    device: torch.device,
) -> Dict:
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {ckpt_path}")

    if "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"], strict=False)
    else:
        loaded_any = False
        for sub_key, sub_mod in [
            ("patch_token_adapter", getattr(model, "patch_token_adapter", None)),
            ("cls_token_adapter", getattr(model, "cls_token_adapter", None)),
            ("prompt_adapter", getattr(model, "prompt_adapter", None)),
        ]:
            if sub_key in state and sub_mod is not None:
                sub_mod.load_state_dict(state[sub_key], strict=False)
                loaded_any = True
        if not loaded_any:
            try:
                model.load_state_dict(state, strict=False)
            except Exception as exc:
                raise RuntimeError(f"Cannot load adapter weights from checkpoint: {ckpt_path}") from exc

    if "prompt_learner" in state:
        prompt_learner.load_state_dict(state["prompt_learner"], strict=False)

    if dino_adapters is not None and "dino_adapters" in state and state["dino_adapters"] is not None:
        dino_adapters.load_state_dict(state["dino_adapters"], strict=False)

    model.eval()
    prompt_learner.eval()
    if dino_adapters is not None:
        dino_adapters.eval()

    return state


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    amin = arr.min()
    amax = arr.max()
    return (arr - amin) / (amax - amin + 1e-8)



def normalize_maps(amaps: np.ndarray, mode: str = "none") -> np.ndarray:
    amaps = amaps.astype(np.float32)
    if mode == "none":
        return amaps
    if mode == "per_class":
        return normalize_minmax(amaps)
    if mode == "per_image":
        out = np.empty_like(amaps, dtype=np.float32)
        for i in range(len(amaps)):
            out[i] = normalize_minmax(amaps[i])
        return out
    raise ValueError(f"Unsupported norm_mode: {mode}")



def compute_best_f1(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt_flat = gt_mask.flatten().astype(np.uint8)
    pred_flat = pred_mask.flatten().astype(np.float32)
    if len(np.unique(gt_flat)) < 2:
        return 0.0
    precisions, recalls, _ = precision_recall_curve(gt_flat, pred_flat)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    f1_scores = f1_scores[np.isfinite(f1_scores)]
    return float(np.max(f1_scores)) if len(f1_scores) > 0 else 0.0



def compute_image_level_scores(masks: np.ndarray, amaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image_gt = (masks.reshape(masks.shape[0], -1).max(axis=1) > 0).astype(np.uint8)
    image_pred = amaps.reshape(amaps.shape[0], -1).max(axis=1).astype(np.float32)
    image_pred = normalize_minmax(image_pred)
    return image_gt, image_pred



def compute_i_auroc(image_gt: np.ndarray, image_pred: np.ndarray) -> float:
    if len(np.unique(image_gt)) < 2:
        return 0.0
    return float(roc_auc_score(image_gt, image_pred))



def compute_p_auroc(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt_flat = gt_mask.flatten().astype(np.uint8)
    pred_flat = pred_mask.flatten().astype(np.float32)
    if len(np.unique(gt_flat)) < 2:
        return 0.0
    return float(roc_auc_score(gt_flat, pred_flat))



def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 1000, max_fpr: float = 0.3) -> float:
    masks = (masks > 0).astype(np.uint8)
    amaps = amaps.astype(np.float32)
    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th - min_th < 1e-12:
        return 0.0

    thresholds = np.linspace(min_th, max_th, num_th, dtype=np.float32)
    inverse_masks = 1 - masks
    denom = inverse_masks.sum().astype(np.float64)
    if denom <= 0:
        return 0.0

    pros, fprs = [], []
    for th in thresholds:
        binary_amaps = (amaps >= th).astype(np.uint8)
        pro_list = []
        for i in range(len(binary_amaps)):
            gt = masks[i]
            pred = binary_amaps[i]
            num_labels, labels = cv2.connectedComponents(gt)
            for region_id in range(1, num_labels):
                region = labels == region_id
                region_area = region.sum()
                if region_area == 0:
                    continue
                overlap = pred[region].sum() / (region_area + 1e-8)
                pro_list.append(float(overlap))

        fp_pixels = np.logical_and(binary_amaps == 1, inverse_masks == 1).sum().astype(np.float64)
        fpr = fp_pixels / (denom + 1e-12)
        if fpr <= max_fpr:
            fprs.append(float(fpr))
            pros.append(float(np.mean(pro_list)) if len(pro_list) > 0 else 0.0)

    if len(fprs) < 2:
        return 0.0

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    pros = pros[sort_idx]
    uniq_fprs, uniq_indices = np.unique(fprs, return_index=True)
    fprs = uniq_fprs
    pros = pros[uniq_indices]
    if len(fprs) < 2:
        return 0.0
    if fprs[0] > 0.0:
        fprs = np.insert(fprs, 0, 0.0)
        pros = np.insert(pros, 0, pros[0])
    if fprs[-1] < max_fpr:
        fprs = np.append(fprs, max_fpr)
        pros = np.append(pros, pros[-1])
    return float(auc(np.clip(fprs, 0.0, max_fpr) / max_fpr, np.clip(pros, 0.0, 1.0)))


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def _to_heatmap(score_map: np.ndarray) -> np.ndarray:
    x = score_map.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)



def save_visualizations(img_list: List[str], pred_mask_list: np.ndarray, gt_mask_list: np.ndarray, category: str, result_path: str) -> None:
    vis_dir = os.path.join(result_path, category, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for i, img_path in enumerate(img_list):
        try:
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            heat = _to_heatmap(cv2.resize(pred_mask_list[i], (w, h), interpolation=cv2.INTER_CUBIC))
            overlay = cv2.addWeighted(image_rgb, 0.55, heat, 0.45, 0.0)
            gt = cv2.resize(gt_mask_list[i].astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_rgb = np.repeat(gt[..., None], 3, axis=2)
            panel = np.concatenate([image_rgb, gt_rgb, heat, overlay], axis=1)
            save_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
            cv2.imwrite(os.path.join(vis_dir, save_name), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        except Exception:
            continue


# -----------------------------------------------------------------------------
# Forward wrapper
# -----------------------------------------------------------------------------

def call_get_anomaly_map(
    clip_model,
    image_info,
    device,
    model,
    dino_model,
    prompt_learner,
    batch_idx: int,
    visual_backbone: str,
    visual_layers: Sequence[int],
):
    sig = inspect.signature(get_anomaly_map)
    kwargs = {
        "clip_model": clip_model,
        "image_info": image_info,
        "device": device,
        "model": model,
        "Dino_model": dino_model,
        "prompt_learner": prompt_learner,
        "idx": batch_idx,
    }
    if "visual_backbone" in sig.parameters:
        kwargs["visual_backbone"] = visual_backbone
    elif visual_backbone != "dino":
        raise RuntimeError(
            "Current get_anomaly_map() does not support visual_backbone switching. "
            "Please use the updated tools/utils.py that supports DINO vs CLIP testing."
        )
    if "layers" in sig.parameters:
        kwargs["layers"] = tuple(visual_layers)
    return get_anomaly_map(**kwargs)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_one_category(args, category, clip_model, prompt_learner, model, dino_model, device, result_dir):
    pixel_pred = []
    pixel_gt = []
    img_list = []

    test_loader = prepare_data(
        dataset_name=args.dataset,
        category=category,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    for batch_idx, image_info in enumerate(tqdm(test_loader, desc=f"{args.dataset}:{category}")):
        with torch.inference_mode():
            outputs = call_get_anomaly_map(
                clip_model=clip_model,
                image_info=image_info,
                device=device,
                model=model,
                dino_model=dino_model,
                prompt_learner=prompt_learner,
                batch_idx=batch_idx,
                visual_backbone=args.visual_backbone,
                visual_layers=args.visual_layers,
            )

        anomaly_map_cross_modal = outputs[2]
        mask = outputs[1]

        mask_np = (mask[:, 0] if mask.dim() == 4 else mask).cpu().numpy().astype(np.uint8)
        amap_np = anomaly_map_cross_modal[:, 1].cpu().numpy().astype(np.float32)

        pixel_gt.extend(mask_np)
        pixel_pred.extend(amap_np)
        img_list.extend(list(image_info["image_path"]))

    gt_mask_list = np.asarray(pixel_gt, dtype=np.uint8)
    pred_mask_list = np.asarray(pixel_pred, dtype=np.float32)
    pred_mask_eval = normalize_maps(pred_mask_list, mode=args.norm_mode)

    if args.save_vis:
        save_visualizations(
            img_list=img_list,
            pred_mask_list=normalize_maps(pred_mask_list, mode="per_class"),
            gt_mask_list=gt_mask_list,
            category=category,
            result_path=result_dir,
        )

    f1 = compute_best_f1(gt_mask_list, pred_mask_eval)
    image_gt, image_pred = compute_image_level_scores(gt_mask_list, pred_mask_eval)
    i_auroc = compute_i_auroc(image_gt, image_pred)
    p_auroc = compute_p_auroc(gt_mask_list, pred_mask_eval)
    pro = compute_pro(gt_mask_list, pred_mask_eval, num_th=args.pro_num_th, max_fpr=args.pro_max_fpr)

    return {
        "category": category,
        "F1": f1,
        "I_AUROC": i_auroc,
        "P_AUROC": p_auroc,
        "PRO": pro,
    }



def write_category_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "F1", "I_AUROC", "P_AUROC", "PRO"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def export_best_latex_tables(dataset_dir: str, dataset: str, method_name: str, best_result: Dict) -> None:
    paper_dir = os.path.join(dataset_dir, "paper_tables")
    os.makedirs(paper_dir, exist_ok=True)
    nl = "\\\\"
    method_key = method_name.lower().replace(" ", "_").replace("-", "_")

    row_tex = (
        f"{dataset} & {method_name} & {best_result['mean_F1'] * 100:.2f} & "
        f"{best_result['mean_I_AUROC'] * 100:.2f} & {best_result['mean_P_AUROC'] * 100:.2f} & "
        f"{best_result['mean_PRO'] * 100:.2f} {nl}\n"
    )
    with open(os.path.join(paper_dir, "best_mean_row.tex"), "w", encoding="utf-8") as f:
        f.write(row_tex)

    table_tex = "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Best test result on {dataset} using {method_name}.}}",
        f"\\label{{tab:{dataset}_{method_key}_best}}",
        "\\resizebox{0.95\\linewidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        f"Dataset & Method & F1 (\\%) & I-AUROC (\\%) & P-AUROC (\\%) & PRO (\\%) {nl}",
        "\\midrule",
        row_tex.rstrip(),
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])
    with open(os.path.join(paper_dir, "best_mean_table.tex"), "w", encoding="utf-8") as f:
        f.write(table_tex)

    cat_lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Per-category results of the best checkpoint on {dataset} using {method_name}.}}",
        f"\\label{{tab:{dataset}_{method_key}_per_category}}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f"Category & F1 (\\%) & I-AUROC (\\%) & P-AUROC (\\%) & PRO (\\%) {nl}",
        "\\midrule",
    ]
    for row in best_result["rows"]:
        cat_lines.append(
            f"{row['category']} & {row['F1'] * 100:.2f} & {row['I_AUROC'] * 100:.2f} & {row['P_AUROC'] * 100:.2f} & {row['PRO'] * 100:.2f} {nl}"
        )
    cat_lines += [
        "\\midrule",
        f"mean & {best_result['mean_F1'] * 100:.2f} & {best_result['mean_I_AUROC'] * 100:.2f} & {best_result['mean_P_AUROC'] * 100:.2f} & {best_result['mean_PRO'] * 100:.2f} {nl}",
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ]
    with open(os.path.join(paper_dir, "best_category_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(cat_lines))



def save_epoch_ranking(dataset_dir: str, results: List[Dict], selection_metric: str) -> None:
    csv_path = os.path.join(dataset_dir, "epoch_ranking.csv")
    tsv_path = os.path.join(dataset_dir, "epoch_ranking.tsv")
    sorted_results = sorted(results, key=lambda x: x[selection_metric], reverse=True)
    fieldnames = ["epoch_name", "backbone", "dataset", "mean_F1", "mean_I_AUROC", "mean_P_AUROC", "mean_PRO"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_results:
            writer.writerow({k: row[k] for k in fieldnames})

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for row in sorted_results:
            f.write("\t".join(str(row[k]) for k in fieldnames) + "\n")



def save_best_epoch_files(dataset_dir: str, best_result: Dict, selection_metric: str) -> None:
    payload = {
        "selection_metric": selection_metric,
        "epoch_name": best_result["epoch_name"],
        "backbone": best_result["backbone"],
        "dataset": best_result["dataset"],
        "mean_F1": best_result["mean_F1"],
        "mean_I_AUROC": best_result["mean_I_AUROC"],
        "mean_P_AUROC": best_result["mean_P_AUROC"],
        "mean_PRO": best_result["mean_PRO"],
        "ckpt_path": best_result["ckpt_path"],
    }
    with open(os.path.join(dataset_dir, "best_epoch.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(os.path.join(dataset_dir, "best_epoch.txt"), "w", encoding="utf-8") as f:
        f.write(f"selection_metric: {selection_metric}\n")
        f.write(f"epoch_name: {best_result['epoch_name']}\n")
        f.write(f"backbone: {best_result['backbone']}\n")
        f.write(f"dataset: {best_result['dataset']}\n")
        f.write(f"ckpt_path: {best_result['ckpt_path']}\n")
        f.write(f"mean_F1: {best_result['mean_F1']:.6f}\n")
        f.write(f"mean_I_AUROC: {best_result['mean_I_AUROC']:.6f}\n")
        f.write(f"mean_P_AUROC: {best_result['mean_P_AUROC']:.6f}\n")
        f.write(f"mean_PRO: {best_result['mean_PRO']:.6f}\n")



def evaluate_checkpoint(args, ckpt_path, epoch_name, clip_model, prompt_learner, model, dino_model, dino_adapters, device):
    load_checkpoint_flexible(ckpt_path, model, prompt_learner, dino_adapters, device)

    dataset_dir = os.path.join(args.result_path, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    categories = sorted(DATASET_CLASSES[args.dataset])
    if args.category != "ALL":
        categories = [args.category]

    rows = []
    print(f"{'=' * 40} Testing {epoch_name} ({args.visual_backbone}) {'=' * 40}")
    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        row = evaluate_one_category(
            args=args,
            category=category,
            clip_model=clip_model,
            prompt_learner=prompt_learner,
            model=model,
            dino_model=dino_model,
            device=device,
            result_dir=dataset_dir,
        )
        rows.append(row)
        print(
            f"{category}: F1={row['F1']:.5f}\tI-AUROC={row['I_AUROC']:.5f}\t"
            f"P-AUROC={row['P_AUROC']:.5f}\tPRO={row['PRO']:.5f}"
        )

    mean_f1 = float(np.mean([x["F1"] for x in rows])) if rows else 0.0
    mean_i = float(np.mean([x["I_AUROC"] for x in rows])) if rows else 0.0
    mean_p = float(np.mean([x["P_AUROC"] for x in rows])) if rows else 0.0
    mean_pro = float(np.mean([x["PRO"] for x in rows])) if rows else 0.0

    print("-" * 60)
    print(f"mean_F1:      {mean_f1:.5f}")
    print(f"mean_I-AUROC: {mean_i:.5f}")
    print(f"mean_P-AUROC: {mean_p:.5f}")
    print(f"mean_PRO:     {mean_pro:.5f}")

    metric_file = os.path.join(dataset_dir, "metric.txt")
    with open(metric_file, "a", encoding="utf-8") as f:
        f.write(f"----------Dataset: {args.dataset} | {epoch_name} | backbone={args.visual_backbone}----------\n")
        f.write(f"{'Classname':<18s}{'F1':>10s}{'I-AUROC':>12s}{'P-AUROC':>12s}{'PRO':>10s}\n")
        for row in rows:
            f.write(
                f"{row['category']:<18s}"
                f"{row['F1']:>10.5f}"
                f"{row['I_AUROC']:>12.5f}"
                f"{row['P_AUROC']:>12.5f}"
                f"{row['PRO']:>10.5f}\n"
            )
        f.write(
            f"{'mean':<18s}"
            f"{mean_f1:>10.5f}"
            f"{mean_i:>12.5f}"
            f"{mean_p:>12.5f}"
            f"{mean_pro:>10.5f}\n\n"
        )

    summary_file = os.path.join(dataset_dir, "summary.tsv")
    header_needed = not os.path.exists(summary_file)
    with open(summary_file, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("epoch_name\tbackbone\tdataset\tmean_F1\tmean_I_AUROC\tmean_P_AUROC\tmean_PRO\n")
        f.write(f"{epoch_name}\t{args.visual_backbone}\t{args.dataset}\t{mean_f1:.6f}\t{mean_i:.6f}\t{mean_p:.6f}\t{mean_pro:.6f}\n")

    epoch_dir = os.path.join(dataset_dir, "epochs", epoch_name)
    os.makedirs(epoch_dir, exist_ok=True)
    write_category_csv(rows, os.path.join(epoch_dir, "per_category_metrics.csv"))

    return {
        "epoch_name": epoch_name,
        "backbone": args.visual_backbone,
        "dataset": args.dataset,
        "mean_F1": mean_f1,
        "mean_I_AUROC": mean_i,
        "mean_P_AUROC": mean_p,
        "mean_PRO": mean_pro,
        "rows": rows,
        "ckpt_path": ckpt_path,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result_test")
    parser.add_argument("--weight_path", type=str, required=True, help="checkpoint dir or checkpoint file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--category", type=str, default="ALL")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--visual_backbone", type=str, default="dino", choices=["dino", "clip"])
    parser.add_argument("--visual_layers", type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--eval_latest_only", action="store_true")

    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--dino_repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--dino_bottleneck", type=int, default=256)

    parser.add_argument("--norm_mode", type=str, default="none", choices=["none", "per_image", "per_class"])
    parser.add_argument("--pro_num_th", type=int, default=1000)
    parser.add_argument("--pro_max_fpr", type=float, default=0.3)
    parser.add_argument("--save_vis", action="store_true")

    parser.add_argument("--selection_metric", type=str, default="mean_PRO", choices=["mean_F1", "mean_I_AUROC", "mean_P_AUROC", "mean_PRO"])
    parser.add_argument("--paper_method_name", type=str, default="")
    parser.add_argument("--export_paper_tables", action="store_true")

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    if not args.paper_method_name:
        args.paper_method_name = f"{args.visual_backbone.upper()}-Visual"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_dir = os.path.join(args.result_path, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    clip_model, prompt_learner, dino_model, dino_adapters, model = build_models(args, device)

    ckpts = sorted_checkpoints(args.weight_path)
    if args.eval_latest_only:
        ckpts = [ckpts[-1]]

    all_results = []
    for ckpt_path in ckpts:
        epoch_name = Path(ckpt_path).stem
        result = evaluate_checkpoint(
            args=args,
            ckpt_path=ckpt_path,
            epoch_name=epoch_name,
            clip_model=clip_model,
            prompt_learner=prompt_learner,
            model=model,
            dino_model=dino_model,
            dino_adapters=dino_adapters,
            device=device,
        )
        all_results.append(result)

    if not all_results:
        print("No checkpoints evaluated.")
        return

    save_epoch_ranking(dataset_dir, all_results, args.selection_metric)
    best = max(all_results, key=lambda x: x[args.selection_metric])
    save_best_epoch_files(dataset_dir, best, args.selection_metric)

    if args.export_paper_tables:
        export_best_latex_tables(dataset_dir, args.dataset, args.paper_method_name, best)

    print("=" * 60)
    print(
        f"Best checkpoint by {args.selection_metric}: {best['epoch_name']} | "
        f"F1={best['mean_F1']:.5f}, I-AUROC={best['mean_I_AUROC']:.5f}, "
        f"P-AUROC={best['mean_P_AUROC']:.5f}, PRO={best['mean_PRO']:.5f}"
    )
    print(f"Best checkpoint file: {best['ckpt_path']}")


if __name__ == "__main__":
    main()
