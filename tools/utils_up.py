import importlib
import importlib.util
import math
import os
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from utils import encode_text_with_prompt_ensemble, get_text_features_with_prompt_learner

import cv2
import numpy as np
import torch
import torch.nn.functional as F


CURRENT_FILE = Path(__file__).resolve()


# -----------------------------------------------------------------------------
# External text helper resolution
# -----------------------------------------------------------------------------

def _resolve_external_function(func_name: str) -> Callable:
    """Resolve project-level helper functions without recursive self-import.

    Common project layouts place these helpers in a *different* utils-like module.
    This resolver tries a few likely import paths and then falls back to importing
    the top-level ``utils`` only when that file is not this file itself.
    """
    candidate_modules = [
        "prompt_utils",
        "text_utils",
        "utils_text",
        "project_utils",
        "tools.prompt_utils",
        "tools.text_utils",
        "tools.utils_text",
        "tools.project_utils",
    ]

    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, func_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue

    try:
        spec = importlib.util.find_spec("utils")
        if spec is not None and spec.origin is not None:
            origin = Path(spec.origin).resolve()
            if origin != CURRENT_FILE:
                module = importlib.import_module("utils")
                fn = getattr(module, func_name, None)
                if callable(fn):
                    return fn
    except Exception:
        pass

    raise ImportError(
        f"Could not resolve external helper function `{func_name}`. "
        "Please ensure the original helper module is importable."
    )


_GET_TEXT_FEATURES_WITH_PROMPT_LEARNER: Optional[Callable] = None
_ENCODE_TEXT_WITH_PROMPT_ENSEMBLE: Optional[Callable] = None


def get_text_features_with_prompt_learner_safe(*args: Any, **kwargs: Any) -> torch.Tensor:
    global _GET_TEXT_FEATURES_WITH_PROMPT_LEARNER
    if _GET_TEXT_FEATURES_WITH_PROMPT_LEARNER is None:
        _GET_TEXT_FEATURES_WITH_PROMPT_LEARNER = _resolve_external_function(
            "get_text_features_with_prompt_learner"
        )
    return _GET_TEXT_FEATURES_WITH_PROMPT_LEARNER(*args, **kwargs)



def encode_text_with_prompt_ensemble_safe(*args: Any, **kwargs: Any) -> torch.Tensor:
    global _ENCODE_TEXT_WITH_PROMPT_ENSEMBLE
    if _ENCODE_TEXT_WITH_PROMPT_ENSEMBLE is None:
        _ENCODE_TEXT_WITH_PROMPT_ENSEMBLE = _resolve_external_function(
            "encode_text_with_prompt_ensemble"
        )
    return _ENCODE_TEXT_WITH_PROMPT_ENSEMBLE(*args, **kwargs)


# -----------------------------------------------------------------------------
# Generic tensor helpers
# -----------------------------------------------------------------------------

def _extract_module_output(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out



def _first_tensor(x: Any) -> torch.Tensor:
    return x[0] if isinstance(x, (tuple, list)) else x



def _normalize_patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens - tokens.mean(dim=1, keepdim=True)) / (tokens.std(dim=1, keepdim=True) + 1e-6)


# -----------------------------------------------------------------------------
# Visual token extraction
# -----------------------------------------------------------------------------

def get_feature_dinov3(
    image_path: Sequence[str],
    batch_img: torch.Tensor,
    device: torch.device,
    Dino_model: torch.nn.Module,
    layers: Sequence[int] = (5, 11, 17, 23),
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Return multi-layer cls / patch tokens from DINOv3."""
    del image_path  # kept for signature compatibility

    anchor = getattr(Dino_model, "norm", None) or getattr(Dino_model, "fc_norm", None)
    if anchor is None:
        raise RuntimeError("There is no norm/fc_norm module in Dino_model.")

    tokens_dict: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        def _mk_hook(idx: int):
            def _hook(module: torch.nn.Module, inp: Any, out: Any) -> None:
                x = _extract_module_output(out)
                tokens_dict[idx] = anchor(x)
            return _hook
        handles.append(Dino_model.blocks[layer_idx].register_forward_hook(_mk_hook(layer_idx)))

    batch_img = batch_img.to(device, non_blocking=True)
    _ = Dino_model(batch_img)

    for handle in handles:
        handle.remove()

    cls_token_list: List[torch.Tensor] = []
    patch_tokens_list: List[torch.Tensor] = []

    for layer_idx in layers:
        if layer_idx not in tokens_dict:
            raise RuntimeError(f"Hook did not capture DINO layer {layer_idx}.")

        toks = tokens_dict[layer_idx]
        patch = toks[:, 5:, :]
        patch = _normalize_patch_tokens(patch)
        cls = toks[:, 0, :].unsqueeze(1)
        patch_tokens_list.append(patch)
        cls_token_list.append(cls)

    return cls_token_list, patch_tokens_list



def get_feature_clip_vit(batch_img, device, clip_model, layers=(5, 11, 17, 23)):
    """
    兼容两类 CLIP visual forward:
      1) visual(x)
      2) visual(x, out_layers=layers)
    返回:
      cls_token_list:   List[(B,1,C)]
      patch_tokens_list: List[(B,N,C)]
    """
    visual = clip_model.visual
    if not hasattr(visual, "transformer") or not hasattr(visual.transformer, "resblocks"):
        raise RuntimeError("clip_model.visual does not expose transformer.resblocks")

    resblocks = visual.transformer.resblocks
    tokens_dict = {}
    handles = []
    batch_size = batch_img.shape[0]

    def _extract_tensor(out):
        if isinstance(out, dict):
            for k in ["x", "tokens", "hidden_states", "last_hidden_state"]:
                if k in out and torch.is_tensor(out[k]):
                    return out[k]
            raise RuntimeError(f"Unexpected dict output keys: {list(out.keys())}")

        if isinstance(out, (tuple, list)):
            for x in out:
                if torch.is_tensor(x):
                    out = x
                    break
            else:
                out = out[0]

        if not torch.is_tensor(out):
            raise RuntimeError(f"Unexpected hook output type: {type(out)}")
        return out

    def _to_bld(x):
        # 某些实现返回 (L, B, D)，统一转成 (B, L, D)
        if x.dim() == 3 and x.shape[1] == batch_size and x.shape[0] != batch_size:
            x = x.permute(1, 0, 2).contiguous()
        return x

    for layer_idx in layers:
        def _mk_hook(idx):
            def _hook(module, inp, out):
                x = _to_bld(_extract_tensor(out))
                tokens_dict[idx] = x
            return _hook
        handles.append(resblocks[layer_idx].register_forward_hook(_mk_hook(layer_idx)))

    batch_img = batch_img.to(device, non_blocking=True)

    sig = inspect.signature(visual.forward)
    if "out_layers" in sig.parameters:
        _ = visual(batch_img, out_layers=layers)
    else:
        _ = visual(batch_img)

    for h in handles:
        h.remove()

    cls_token_list = []
    patch_tokens_list = []

    ln_post = getattr(visual, "ln_post", None)

    for layer_idx in layers:
        if layer_idx not in tokens_dict:
            raise RuntimeError(
                f"Hook did not capture CLIP layer {layer_idx}. "
                f"Captured layers: {sorted(tokens_dict.keys())}"
            )

        toks = tokens_dict[layer_idx]
        if ln_post is not None:
            toks = ln_post(toks)

        if toks.dim() != 3:
            raise RuntimeError(f"Expected 3D tokens, got shape={tuple(toks.shape)}")

        cls = toks[:, 0, :].unsqueeze(1)
        patch = toks[:, 1:, :]
        patch = (patch - patch.mean(dim=1, keepdim=True)) / (
            patch.std(dim=1, keepdim=True) + 1e-6
        )

        cls_token_list.append(cls)
        patch_tokens_list.append(patch)

    return cls_token_list, patch_tokens_list



def get_visual_tokens(
    image_path: Sequence[str],
    batch_img: torch.Tensor,
    device: torch.device,
    clip_model: torch.nn.Module,
    Dino_model: Optional[torch.nn.Module],
    visual_backbone: str = "dino",
    layers: Sequence[int] = (5, 11, 17, 23),
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    visual_backbone = visual_backbone.lower()
    if visual_backbone == "dino":
        if Dino_model is None:
            raise ValueError("Dino_model is required when visual_backbone='dino'.")
        return get_feature_dinov3(image_path, batch_img, device, Dino_model, layers=layers)
    if visual_backbone == "clip":
        return get_feature_clip_vit(batch_img, device, clip_model, layers=layers)
    raise ValueError(f"Unsupported visual_backbone: {visual_backbone}")


# -----------------------------------------------------------------------------
# Text feature construction
# -----------------------------------------------------------------------------

def _get_text_features(
    clip_model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    image_path: Sequence[str],
    device: torch.device,
    text_source: str = "prompt_learner",
) -> torch.Tensor:
    """Return text features with shape (B, D, 2)."""
    text_source = text_source.lower()
    batch_size = len(image_path)

    if text_source == "prompt_learner":
        sample = get_text_features_with_prompt_learner_safe(
            clip_model=clip_model,
            prompt_learner=prompt_learner,
            device=device,
            adapter=None,
        )
        dim = sample.shape[0]
        text_feature = torch.zeros(batch_size, dim, 2, device=device, dtype=sample.dtype)
        text_feature[0] = sample
        for i in range(1, batch_size):
            text_feature[i] = get_text_features_with_prompt_learner_safe(
                clip_model=clip_model,
                prompt_learner=prompt_learner,
                device=device,
                adapter=None,
            )
        return text_feature

    if text_source == "ensemble":
        sample_cls_name = Path(image_path[0]).parts[-4]
        sample = encode_text_with_prompt_ensemble_safe(clip_model, sample_cls_name, device, "", None)
        dim = sample.shape[0]
        text_feature = torch.zeros(batch_size, dim, 2, device=device, dtype=sample.dtype)
        text_feature[0] = sample
        for i in range(1, batch_size):
            cls_name = Path(image_path[i]).parts[-4]
            text_feature[i] = encode_text_with_prompt_ensemble_safe(clip_model, cls_name, device, "", None)
        return text_feature

    raise ValueError(f"Unsupported text_source: {text_source}")


# -----------------------------------------------------------------------------
# Main anomaly-map forward
# -----------------------------------------------------------------------------

def get_anomaly_map(
    clip_model: torch.nn.Module,
    image_info: Dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
    Dino_model: Optional[torch.nn.Module],
    prompt_learner: torch.nn.Module,
    idx: int,
    visual_backbone: str = "dino",
    visual_layers: Sequence[int] = (5, 11, 17, 23),
    text_source: str = "prompt_learner",
    return_debug: bool = False,
    return_evidence: bool = False,
):
    image = image_info["image"].to(device, non_blocking=True)
    image_path = image_info["image_path"]
    mask = image_info["mask"].to(device, non_blocking=True)
    mask = (mask > 0.5).float()

    batch_size = image.shape[0]
    out_h, out_w = image.shape[-2:]

    text_feature = _get_text_features(
        clip_model=clip_model,
        prompt_learner=prompt_learner,
        image_path=image_path,
        device=device,
        text_source=text_source,
    )

    f0 = _first_tensor(model.prompt_adapter[0](text_feature[:, :, 0]))
    f1 = _first_tensor(model.prompt_adapter[1](text_feature[:, :, 1]))
    adjusted_text_feature = torch.stack([f0, f1], dim=-1)  # (B,D,2)

    if idx % 50 == 0:
        t_norm = adjusted_text_feature[:, :, 0]
        t_abn = adjusted_text_feature[:, :, 1]
        cos_sim = F.cosine_similarity(t_norm, t_abn, dim=1)
        print(
            f"[text/{visual_backbone}] cos_sim mean/min/max: "
            f"{cos_sim.mean().item():.4f}/{cos_sim.min().item():.4f}/{cos_sim.max().item():.4f}"
        )

    cls_token, patch_tokens = get_visual_tokens(
        image_path=image_path,
        batch_img=image,
        device=device,
        clip_model=clip_model,
        Dino_model=Dino_model,
        visual_backbone=visual_backbone,
        layers=visual_layers,
    )

    num_scales = len(cls_token)
    sum_cross_modal = None
    sum_awareness = None
    sum_global_score = None

    debug = None
    if return_debug:
        debug = {
            "visual_backbone": visual_backbone,
            "image_path": list(image_path),
            "token_margin_layers": [],
            "cross_prob_layers": [],
            "aw_layers": [],
            "global_margin_layers": [],
        }
    evidence = None
    if return_evidence:
        evidence = {
            "cross_prob_layers": [],
            "cross_margin_layers": [],
            "awareness_layers": [],
            "normal_similarity_layers": [],
            "anomaly_similarity_layers": [],
            "global_margin_layers": [],
        }

    for layer_i in range(num_scales):
        cls_features = _first_tensor(model.cls_token_adapter[layer_i](cls_token[layer_i]))
        patch_features = _first_tensor(model.patch_token_adapter[layer_i](patch_tokens[layer_i]))

        cls_features = F.normalize(cls_features, dim=-1)
        patch_features = F.normalize(patch_features, dim=-1)

        if cls_features.dim() == 3 and cls_features.shape[1] == 1:
            cls_features = cls_features[:, 0, :]

        num_tokens = patch_features.shape[1]
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise RuntimeError(
                f"Patch token count {num_tokens} is not a perfect square; cannot form 2D map."
            )

        # cross-modal contrastive map
        cross_logits_tokens = 100.0 * torch.bmm(patch_features, adjusted_text_feature)  # (B,N,2)
        token_margin = (cross_logits_tokens[:, :, 1] - cross_logits_tokens[:, :, 0]).reshape(batch_size, side, side)

        cross_logits = cross_logits_tokens.permute(0, 2, 1).reshape(batch_size, 2, side, side)
        cross_logits = F.interpolate(
            cross_logits,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=True,
        )
        cross_prob = torch.softmax(cross_logits, dim=1)

        if return_debug:
            debug["token_margin_layers"].append(token_margin.detach().cpu())
            debug["cross_prob_layers"].append(cross_prob[:, 1].detach().cpu())
        if return_evidence:
            evidence["cross_prob_layers"].append(cross_prob[:, 1].detach())
            evidence["cross_margin_layers"].append(token_margin.detach())
            evidence["normal_similarity_layers"].append(
                cross_logits_tokens[:, :, 0].reshape(batch_size, side, side).detach() / 100.0
            )
            evidence["anomaly_similarity_layers"].append(
                cross_logits_tokens[:, :, 1].reshape(batch_size, side, side).detach() / 100.0
            )

        sum_cross_modal = cross_prob if sum_cross_modal is None else sum_cross_modal + cross_prob

        # anomaly-aware calibration
        cls_vec = cls_features.unsqueeze(-1)  # (B,D,1)
        patch_cls_logits = 10.0 * torch.bmm(patch_features, cls_vec)
        patch_cls_logits = patch_cls_logits.permute(0, 2, 1).reshape(batch_size, 1, side, side)
        patch_cls_logits = F.interpolate(
            patch_cls_logits,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=True,
        )
        patch_cls_prob = torch.sigmoid(patch_cls_logits)
        awareness = torch.cat([1.0 - patch_cls_prob, patch_cls_prob], dim=1)

        if return_debug:
            debug["aw_layers"].append(awareness[:, 1].detach().cpu())
        if return_evidence:
            evidence["awareness_layers"].append(awareness[:, 1].detach())

        sum_awareness = awareness if sum_awareness is None else sum_awareness + awareness

        # global anomaly score
        global_score = 100.0 * torch.bmm(
            cls_features.unsqueeze(1), adjusted_text_feature
        ).squeeze(1)

        if return_debug:
            debug["global_margin_layers"].append(
                (global_score[:, 1] - global_score[:, 0]).detach().cpu()
            )
        if return_evidence:
            evidence["global_margin_layers"].append(
                (global_score[:, 1] - global_score[:, 0]).detach()
            )

        sum_global_score = global_score if sum_global_score is None else sum_global_score + global_score

    anomaly_map_cross_modal = sum_cross_modal / num_scales
    anomaly_awareness = sum_awareness / num_scales
    global_anomaly_score = sum_global_score / num_scales

    if return_debug:
        debug["final_cross_prob"] = anomaly_map_cross_modal[:, 1].detach().cpu()
        debug["final_aw"] = anomaly_awareness[:, 1].detach().cpu()
        debug["final_global_margin"] = (
            global_anomaly_score[:, 1] - global_anomaly_score[:, 0]
        ).detach().cpu()
        if return_evidence:
            return anomaly_awareness, mask, anomaly_map_cross_modal, global_anomaly_score, debug, evidence
        return anomaly_awareness, mask, anomaly_map_cross_modal, global_anomaly_score, debug

    if return_evidence:
        return anomaly_awareness, mask, anomaly_map_cross_modal, global_anomaly_score, evidence

    return anomaly_awareness, mask, anomaly_map_cross_modal, global_anomaly_score


# -----------------------------------------------------------------------------
# Reward weighting and losses
# -----------------------------------------------------------------------------

def compute_reward_and_weight(
    global_logits,
    anomaly_map_cross_modal,
    mask,
    gt_label,
    reward_state=None,
    w_loc=1.0,
    w_conf=0.2,
    alpha_adv_cls=0.05,
    cls_weight_clip=(0.5, 2.0),
    ema_beta=0.95,
    seg_k=3.0,
    seg_weight_clip=(0.5, 5.0),
    eps=1e-6,
):
    """Reward-guided sample weights for global CE and segmentation dice."""
    if reward_state is None:
        reward_state = {"baseline": 0.0}

    with torch.no_grad():
        p = torch.softmax(global_logits, dim=-1)  # (B,2)
        p_correct = p.gather(1, gt_label.unsqueeze(1)).squeeze(1)
        r_cls = 2.0 * p_correct - 1.0

        if mask.dim() == 4:
            mask_bhw = mask[:, 0]
        else:
            mask_bhw = mask
        mask_bhw = mask_bhw.float()

        if anomaly_map_cross_modal.shape[1] == 2:
            if anomaly_map_cross_modal.min() < 0 or anomaly_map_cross_modal.max() > 1:
                prob_map = torch.softmax(anomaly_map_cross_modal, dim=1)[:, 1]
            else:
                prob_map = anomaly_map_cross_modal[:, 1].clamp(0, 1)
        else:
            prob_map = anomaly_map_cross_modal.squeeze(1).sigmoid()

        batch_size = prob_map.shape[0]
        p_flat = prob_map.reshape(batch_size, -1)
        g_flat = mask_bhw.reshape(batch_size, -1)

        inter = (p_flat * g_flat).sum(dim=1)
        denom = p_flat.sum(dim=1) + g_flat.sum(dim=1)
        dice = (2.0 * inter + eps) / (denom + eps)

        r_loc = dice * gt_label.float()

        pred = p.argmax(dim=-1)
        correct = (pred == gt_label).float()
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
        ent_norm = entropy / math.log(2.0)
        r_conf = (1.0 - ent_norm) * correct

        reward = r_cls + w_loc * r_loc + w_conf * r_conf

        batch_mean = reward.mean().item()
        baseline = reward_state["baseline"]
        baseline = ema_beta * baseline + (1.0 - ema_beta) * batch_mean
        reward_state["baseline"] = baseline

        adv_cls = reward - baseline
        w_cls = (1.0 - alpha_adv_cls * adv_cls).clamp(cls_weight_clip[0], cls_weight_clip[1])

        seg_bad = (1.0 - dice).clamp(0, 1) * gt_label.float()
        w_seg = 1.0 + seg_k * seg_bad
        w_seg = w_seg.clamp(seg_weight_clip[0], seg_weight_clip[1])

    return w_cls, w_seg, float(reward.mean().item()), float(baseline), dice, reward_state



def dice_loss_per_sample(prob_map, mask, eps=1e-6):
    """
    prob_map: (B,H,W) in [0,1]
    mask:     (B,H,W) 0/1
    return:   (B,) dice_loss = 1 - dice
    """
    batch_size = prob_map.shape[0]
    p = prob_map.reshape(batch_size, -1)
    g = mask.reshape(batch_size, -1).float()
    inter = (p * g).sum(dim=1)
    denom = p.sum(dim=1) + g.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def _to_numpy_image(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu().float()
    if x.dim() != 3:
        raise ValueError("Expected CHW image tensor.")
    x = x.permute(1, 2, 0).numpy()
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)



def _to_numpy_mask(mask_hw: torch.Tensor) -> np.ndarray:
    x = mask_hw.detach().cpu().float().numpy()
    x = (x > 0.5).astype(np.uint8) * 255
    return x



def _normalize_map(score_hw: torch.Tensor) -> np.ndarray:
    score = score_hw.detach().cpu().float().numpy()
    score = score - score.min()
    denom = score.max() + 1e-6
    return score / denom



def overlay_heatmap(img_uint8: np.ndarray, score_hw: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    score = _normalize_map(score_hw)
    heat = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 1.0 - alpha, heat, alpha, 0.0)



def _score_to_rgb(score_hw: torch.Tensor) -> np.ndarray:
    score = _normalize_map(score_hw)
    heat = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)



def _put_title(img: np.ndarray, text: str) -> np.ndarray:
    canvas = img.copy()
    cv2.putText(
        canvas,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas



def _resize_like(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)



def save_debug_visuals(
    image: torch.Tensor,
    mask: torch.Tensor,
    debug: Dict[str, Any],
    save_path: str,
    batch_index: int = 0,
    title_prefix: str = "",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if mask.dim() == 4:
        mask = mask[:, 0]

    img = _to_numpy_image(image[batch_index])
    gt = cv2.cvtColor(_to_numpy_mask(mask[batch_index]), cv2.COLOR_GRAY2RGB)
    final_cross = overlay_heatmap(img, debug["final_cross_prob"][batch_index])
    final_aw = overlay_heatmap(img, debug["final_aw"][batch_index])

    panels = [
        _put_title(img, f"{title_prefix}Image"),
        _put_title(gt, f"{title_prefix}GT"),
        _put_title(final_cross, f"{title_prefix}Final Cross"),
        _put_title(final_aw, f"{title_prefix}Final Awareness"),
    ]

    for layer_id, token_map in enumerate(debug["token_margin_layers"], start=1):
        vis = _score_to_rgb(token_map[batch_index])
        vis = _resize_like(vis, img.shape[:2])
        panels.append(_put_title(vis, f"{title_prefix}Margin L{layer_id}"))

    rows = []
    cols = 4
    blank = np.zeros_like(img)
    for start in range(0, len(panels), cols):
        row = panels[start:start + cols]
        if len(row) < cols:
            row = row + [blank] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))

    canvas = np.concatenate(rows, axis=0)
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))



def save_comparison_panel(
    image: torch.Tensor,
    mask: torch.Tensor,
    debug_dino: Dict[str, Any],
    debug_clip: Dict[str, Any],
    save_path: str,
    batch_index: int = 0,
    title: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if mask.dim() == 4:
        mask = mask[:, 0]

    img = _to_numpy_image(image[batch_index])
    gt = cv2.cvtColor(_to_numpy_mask(mask[batch_index]), cv2.COLOR_GRAY2RGB)
    dino_final = overlay_heatmap(img, debug_dino["final_cross_prob"][batch_index])
    clip_final = overlay_heatmap(img, debug_clip["final_cross_prob"][batch_index])

    row1 = [
        _put_title(img, "Image"),
        _put_title(gt, "GT"),
        _put_title(dino_final, "DINO Final"),
        _put_title(clip_final, "CLIP Final"),
    ]

    row2 = []
    row3 = []
    num_layers = min(len(debug_dino["token_margin_layers"]), len(debug_clip["token_margin_layers"]))
    for i in range(num_layers):
        d_map = _resize_like(_score_to_rgb(debug_dino["token_margin_layers"][i][batch_index]), img.shape[:2])
        c_map = _resize_like(_score_to_rgb(debug_clip["token_margin_layers"][i][batch_index]), img.shape[:2])
        row2.append(_put_title(d_map, f"DINO L{i + 1}"))
        row3.append(_put_title(c_map, f"CLIP L{i + 1}"))

    while len(row2) < 4:
        row2.append(np.zeros_like(img))
    while len(row3) < 4:
        row3.append(np.zeros_like(img))

    canvas = np.concatenate([
        np.concatenate(row1[:4], axis=1),
        np.concatenate(row2[:4], axis=1),
        np.concatenate(row3[:4], axis=1),
    ], axis=0)

    if title:
        top_bar = np.zeros((48, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(top_bar, title[:160], (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        canvas = np.concatenate([top_bar, canvas], axis=0)

    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
