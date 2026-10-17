"""Frozen MARA A/B diagnostics. GT is used ONLY after action execution.

D0 matches corrected production infer(). Forced actions / bypassed gain
gates are diagnostic. Historical v1 results belong to their original revision.
"""
from dataclasses import replace
import copy
import hashlib
import json
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from tools.mara_agent import (
    MARAAgent, MARAConfig, _compose_high_res_probability, _discounted_returns,
    _resize_map,
)

PROTOCOL = "mara_early_stop_ab_v2"
ARMS = {
    "D0": dict(force_refine=False, bypass_gain=False),
    "D1": dict(force_refine=False, bypass_gain=True),
    "D2": dict(force_refine=True, bypass_gain=False),
    "D3": dict(force_refine=True, bypass_gain=True),
}


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


def select_indices(items, limit, seed):
    """Balanced mask-present/mask-absent metadata sampling, without mask pixels.

    A zero limit means the full category. The image label is checked again
    during export; mask presence is a sampling stratum, not a pseudo-label.
    """
    if limit < 0:
        raise ValueError("samples_per_category must be >= 0")
    if limit == 0 or limit >= len(items):
        return list(range(len(items)))
    buckets = {False: [], True: []}
    for index, item in enumerate(items):
        buckets[item[3] is not None].append(index)
    for bucket in buckets.values():
        bucket.sort(key=lambda i: stable_id(seed, str(items[i][2])))
    selected = []
    while len(selected) < limit:
        progressed = False
        for key in (False, True):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def checkpoint_config(payload):
    """Fail closed instead of evaluating randomly initialized missing heads."""
    if not isinstance(payload, Mapping) or not payload.get("mara_agent"):
        raise ValueError("MARA_CKPT must be a MARA checkpoint, not a Base checkpoint")
    raw = payload.get("mara_config")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Checkpoint must contain mara_config; do not guess architecture")
    unknown = set(raw) - set(MARAConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown checkpoint MARA config keys: {sorted(unknown)}")
    cfg = MARAConfig(**dict(raw))
    if cfg.max_steps < 1 or cfg.num_regions < 1 or cfg.num_layers < 1:
        raise ValueError("Invalid MARA horizon/regions/layers")
    # Strict validation is small and CPU-only; Base is not loaded here.
    probe = MARAAgent(cfg)
    probe.load_state_dict(payload["mara_agent"], strict=True)
    return cfg


def validate_cache(cache, cfg, allow_batch=False):
    base = cache["base_prob"]
    if base.ndim != 4 or base.shape[1] != 2 or base.shape[0] < 1 or (not allow_batch and base.shape[0] != 1):
        raise ValueError("Cache must contain one B=1 two-class Base map")
    if base.shape[-1] != base.shape[-2]:
        raise ValueError("MARA currently requires square maps")
    batch = base.shape[0]
    if cache["base_logits"].shape != (batch, 2):
        raise ValueError("Invalid Base logits")
    layers = cache["layer_maps"]
    if layers is None or layers.shape[:2] != (batch, cfg.num_layers):
        raise ValueError("Layer evidence does not match checkpoint")
    evidence = cache.get("evidence_maps")
    if cfg.evidence_channels and (
        evidence is None or evidence.shape[:2] != (batch, cfg.evidence_channels)
    ):
        raise ValueError("Spatial evidence does not match checkpoint")
    global_evidence = cache.get("global_evidence")
    if cfg.global_evidence_dim and (
        global_evidence is None or global_evidence.shape != (batch, cfg.global_evidence_dim)
    ):
        raise ValueError("Global evidence does not match checkpoint")
    for name in ("base_prob", "base_logits", "layer_maps", "evidence_maps", "global_evidence"):
        value = cache.get(name)
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite input: {name}")
    if bool((base < 0).any() or (base > 1).any()):
        raise ValueError("Base probabilities outside [0,1]")
    if not torch.allclose(base.sum(1), torch.ones_like(base[:, 0]), atol=1e-4):
        raise ValueError("Base class probabilities do not sum to one")


def move_inputs(cache, device):
    return {key: value.to(device=device, dtype=torch.float32)
            for key in ("base_prob", "base_logits", "layer_maps", "evidence_maps", "global_evidence")
            if (value := cache.get(key)) is not None}


@torch.no_grad()
def trace_infer(agent, inputs, arm="D0", intervention=None):
    """No masks, labels, losses or oracle choices are accepted by this function."""
    if intervention is None and arm not in ARMS:
        raise ValueError(f"Unknown diagnostic arm: {arm}")
    if set(inputs)-{"base_prob", "base_logits", "layer_maps", "evidence_maps", "global_evidence"}:
        raise ValueError("Inference inputs must not contain GT or diagnostic scores")
    validate_cache(inputs, agent.cfg)
    original_cfg = agent.cfg
    spec = ARMS[arm] if intervention is None else intervention
    overrides = {k: spec[k] for k in ("gain_safety_margin", "freeze_noop_state") if k in spec}
    if spec["bypass_gain"]:
        overrides["use_gain_gate"] = False
    cfg = replace(original_cfg, **overrides)
    agent.cfg = cfg
    try:
        base_hr = inputs["base_prob"]
        base = _resize_map(base_hr, cfg.map_size)
        layers = _resize_map(inputs["layer_maps"], cfg.map_size)
        evidence = inputs.get("evidence_maps")
        evidence = None if evidence is None else _resize_map(evidence, cfg.map_size)
        global_evidence = inputs.get("global_evidence")
        logits = torch.log(base.clamp(1e-6, 1.0))
        current = base
        global_logits = inputs["base_logits"]
        gate = torch.zeros_like(base[:, :1])
        active = torch.ones(base.shape[0], device=base.device, dtype=torch.bool)
        records = []
        maps = []
        for step in range(cfg.max_steps):
            before = (current, global_logits, gate, active)
            state = agent._make_state(current, base, layers, step, evidence)
            forced = torch.ones_like(active, dtype=torch.long) if spec["force_refine"] else None
            policy = agent._policy(
                state, global_logits, active, step, sample=False,
                forced_op=forced, global_evidence=global_evidence,
            )
            result = agent._refine(
                base, inputs["base_logits"], logits, current, global_logits,
                gate, state, layers, policy, active,
                apply_gain_gate=True, hard_gain_gate=True,
            )
            before_hr = base_hr if step == 0 else maps[-1]
            after_hr, gate_hr = _compose_high_res_probability(
                base_hr, result["next_logits"], result["next_cumulative_gate"])
            # Construct the ungated candidate at ORIGINAL resolution too.
            candidate_hr, _ = _compose_high_res_probability(
                base_hr, result["candidate_logits"], result["candidate_cumulative_gate"])
            op = int(policy["op_action"].item())
            is_active = bool(active.item())
            accepted = bool(result["gain_accepted"].item())
            safety = result["predicted_gain_lower"] if cfg.use_gain_lower_bound else result["predicted_gain"]
            lower_failed = bool(cfg.use_gain_gate and safety.item() <= cfg.gain_safety_margin)
            probability_failed = bool(cfg.use_gain_gate and result["accept_score"].item() < cfg.gain_accept_probability)
            if not is_active:
                reason = "inactive"
            elif op == 0:
                reason = "policy_stop"
            elif not bool(result["next_active"].item()):
                reason = "gain_reject_both" if lower_failed and probability_failed else (
                    "gain_reject_lower" if lower_failed else "gain_reject_probability")
            elif step == cfg.max_steps-1:
                reason = "horizon"
            else:
                reason = "continue"
            region = int(policy["region_action"].item())
            layer = int(policy["layer_action"].item())
            op_prob = policy["op_logits"].softmax(-1)[0]
            records.append(dict(
                step=step, active_before=is_active,
                active_after=bool(result["next_active"].item()), op=op,
                forced_refine=spec["force_refine"], bypass_gain=spec["bypass_gain"],
                region=region, layer=layer, stop_reason=reason,
                p_stop=float(op_prob[0]), p_refine=float(op_prob[1]),
                region_prob=float(policy["region_logits"].softmax(-1)[0, region]),
                layer_prob=float(policy["layer_logits"].softmax(-1)[0, layer]),
                region_probabilities=policy["region_logits"].softmax(-1)[0].cpu().tolist(),
                layer_probabilities=policy["layer_logits"].softmax(-1)[0].cpu().tolist(),
                box=policy["box_features"][0, region].cpu().tolist(),
                predicted_gain=float(result["predicted_gain"].item()),
                predicted_gain_lower=float(result["predicted_gain_lower"].item()),
                accept_probability=float(result["accept_score"].item()),
                gain_accepted=accepted, lower_failed=lower_failed,
                probability_failed=probability_failed,
                execute_weight=float(result["execute_weight"].item()),
                update_gate_mean=float(result["update_gate"].mean()),
                gate_mean=float(gate_hr.mean()),
                pixel_change_max=float((after_hr-before_hr).abs().max()),
                global_change_max=float((result["next_global_logits"]-global_logits).abs().max()),
                global_logits_before=global_logits[0].cpu().tolist(),
                global_logits_after=result["next_global_logits"][0].cpu().tolist(),
                before_prob=before[0], before_global=before[1], before_hr=before_hr,
                after_prob=result["next_prob"], after_global=result["next_global_logits"], after_hr=after_hr,
                candidate_prob=result["candidate_prob"], candidate_global=result["candidate_global_logits"],
                candidate_hr=candidate_hr,
            ))
            maps.append(after_hr)
            logits, current, global_logits, gate, active = (
                result["next_logits"], result["next_prob"], result["next_global_logits"],
                result["next_cumulative_gate"], result["next_active"],
            )
        return dict(records=records, maps=maps, final_prob=maps[-1],
                    final_logits=global_logits, gate_map=gate_hr)
    finally:
        agent.cfg = original_cfg


def quality_parts(prob, global_logits, mask, labels, cfg):
    """Exactly the training quality convention, with decomposed components."""
    mask = mask[:, None] if mask.ndim == 3 else mask
    gt = _resize_map(mask.float(), prob.shape[-1], mode="nearest")[:, 0] > .5
    gt = gt.float().flatten(1)
    p = prob[:, 1].clamp(0, 1).flatten(1)
    q_cls = -F.cross_entropy(global_logits, labels.long(), reduction="none")
    dice = (2*(p*gt).sum(1)+1e-6)/(p.sum(1)+gt.sum(1)+1e-6)
    q_loc = labels.float()*dice + (1-labels.float())*(1-p.mean(1))
    image_prob = global_logits.softmax(-1)
    q_conf = 1+(image_prob*image_prob.clamp(1e-6, 1).log()).sum(1)/np.log(2.)
    bg = 1-gt
    q_bg = (p*bg).sum(1)/bg.sum(1).clamp_min(1e-6)
    total = cfg.reward_cls_weight*q_cls + cfg.reward_loc_weight*q_loc + cfg.reward_conf_weight*q_conf - cfg.reward_fp_weight*q_bg
    return dict(quality=float(total.item()), q_cls=float(q_cls.item()),
                q_loc=float(q_loc.item()), q_conf=float(q_conf.item()),
                q_bg=float(q_bg.item()), gt_pixels=int(gt.sum().item()))


def score_trace(trace, mask, labels, cfg):
    """Post-hoc GT scoring. Never feeds back to inference or selects actions."""
    rows = []
    for record in trace["records"]:
        row = {k: v for k, v in record.items() if not isinstance(v, torch.Tensor)}
        parts = {}
        for prefix, prob_key, global_key in (
            ("before_lr", "before_prob", "before_global"),
            ("after_lr", "after_prob", "after_global"),
            ("candidate_lr", "candidate_prob", "candidate_global"),
            ("before_hr", "before_hr", "before_global"),
            ("after_hr", "after_hr", "after_global"),
            ("candidate_hr", "candidate_hr", "candidate_global"),
        ):
            parts[prefix] = quality_parts(record[prob_key], record[global_key], mask, labels, cfg)
            row.update({f"{prefix}_{k}": v for k, v in parts[prefix].items()})
        raw = parts["after_lr"]["quality"]-parts["before_lr"]["quality"]
        candidate = parts["candidate_lr"]["quality"]-parts["before_lr"]["quality"]
        active = float(record["active_before"])
        step_cost = cfg.step_cost*active
        refine_cost = cfg.refine_cost*active*record["op"]
        row.update(
            delta_quality_lr=raw,
            delta_quality_hr=parts["after_hr"]["quality"]-parts["before_hr"]["quality"],
            candidate_delta_quality_lr=candidate,
            candidate_delta_quality_hr=parts["candidate_hr"]["quality"]-parts["before_hr"]["quality"],
            candidate_valid=record["active_before"],
            candidate_gain_target=candidate-cfg.refine_cost,
            candidate_net_reward=candidate-cfg.step_cost-cfg.refine_cost,
            step_cost=step_cost, refine_cost=refine_cost,
            reward=(raw-step_cost-refine_cost)*active,
            reward_without_cost=raw*active,
        )
        for key in ("q_cls", "q_loc", "q_bg", "q_conf"):
            row[f"delta_{key}"] = parts["after_lr"][key]-parts["before_lr"][key]
        rows.append(row)
    reward = torch.tensor([[r["reward"] for r in rows]], dtype=torch.float32)
    no_cost = torch.tensor([[r["reward_without_cost"] for r in rows]], dtype=torch.float32)
    returns = _discounted_returns(reward, cfg.gamma)[0].tolist()
    no_cost_returns = _discounted_returns(no_cost, cfg.gamma)[0].tolist()
    for row, total, costless in zip(rows, returns, no_cost_returns):
        row["return_to_go"] = total
        row["return_to_go_without_cost"] = costless
        row["cost_flips_reward_sign"] = row["reward_without_cost"] > 0 and row["reward"] <= 0
    return rows


def gradient_probe(agent, inputs, mask, labels, seed=0):
    """Counterfactual post-warmup GRPO gradient, NOT proof of historical updates.

    Uses a separate MARA clone; no optimizer step and no original state change.
    """
    cfg = replace(agent.cfg, group_size=max(2, agent.cfg.group_size))
    probe = copy.deepcopy(agent)
    probe.cfg = cfg
    probe.eval().requires_grad_(True)
    devices = [inputs["base_prob"].device.index] if inputs["base_prob"].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.no_grad():
            behavior = probe.rollout(**inputs, mask=mask, labels=labels,
                                     sample=True, apply_gain_gate=True, force_refine=False)
        with torch.enable_grad():
            replay = probe.rollout(**inputs, mask=mask, labels=labels,
                                  sample=False, trajectory=behavior["trajectory"], apply_gain_gate=True)
            names = ("region_head", "layer_head", "op_head")
            params = [p for name in names for p in getattr(probe, name).parameters()]
            gradients = torch.autograd.grad(replay["policy_loss"], params, allow_unused=True)
            counts = [len(list(getattr(probe, name).parameters())) for name in names]
            stats = {}
            offset = 0
            for name, count in zip(names, counts):
                stats[f"{name}_grad_l1"] = sum(
                    float(g.detach().abs().sum()) for g in gradients[offset:offset+count] if g is not None)
                offset += count
    trajectory = behavior["trajectory"]
    valid = trajectory["policy_mask"] > 0
    advantage = trajectory["advantages"][valid]
    stats.update(
        policy_loss=float(replay["policy_loss"].detach()),
        valid_policy_decisions=int(valid.sum()), group_size=cfg.group_size,
        positive_advantage_fraction=float((advantage > 0).float().mean()) if advantage.numel() else None,
        negative_advantage_fraction=float((advantage < 0).float().mean()) if advantage.numel() else None,
        zero_advantage_fraction=float((advantage == 0).float().mean()) if advantage.numel() else None,
        group_first_returns=trajectory["returns_to_go"][:, 0].cpu().tolist(),
        sampled_ops=trajectory["op_actions"].cpu().tolist(),
        return_to_go=trajectory["returns_to_go"].cpu().tolist(),
        advantages=trajectory["advantages"].cpu().tolist(),
        active_mask=trajectory["active_mask"].cpu().tolist(),
        policy_mask=trajectory["policy_mask"].cpu().tolist(),
        base_trajectory_mask=trajectory["base_trajectory_mask"].cpu().tolist(),
        interpretation="counterfactual_frozen_checkpoint_gradient_not_training_history",
    )
    del probe
    return stats


def binary_metrics(gt, scores):
    """Exact tied-score AUROC and best F1, without requiring sklearn."""
    gt = np.asarray(gt, bool).ravel()
    scores = np.asarray(scores, np.float32).ravel()
    if gt.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid metric arrays")
    positives = int(gt.sum())
    negatives = len(gt)-positives
    if not positives or not negatives:
        return dict(AUROC=None, F1=None)
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_gt = scores[order], gt[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])+1]
    counts = np.diff(np.r_[starts, len(gt)])
    positive_counts = np.add.reduceat(sorted_gt.astype(np.int64), starts)
    average_ranks = starts + (counts+1)/2
    auc = (np.dot(positive_counts, average_ranks)-positives*(positives+1)/2)/(positives*negatives)
    tp = np.cumsum(positive_counts[::-1])
    predicted = np.cumsum(counts[::-1])
    f1 = 2*tp/(predicted+positives)
    return dict(AUROC=float(auc), F1=float(f1.max()))


def sampled_pro(gt, maps, num_th=1000, max_fpr=.3):
    """Same sampled-threshold definition as test2; label components once.

    Uses sorted foreground/background scores instead of relabeling every
    threshold. This is NOT the exact-PRO estimator from other experiments.
    """
    import cv2
    gt = np.asarray(gt, bool)
    maps = np.asarray(maps, np.float32)
    if gt.shape != maps.shape or gt.ndim != 3 or num_th < 2 or not 0 < max_fpr <= 1:
        raise ValueError("Invalid PRO inputs")
    if not np.isfinite(maps).all():
        raise ValueError("Nonfinite metric scores")
    foreground_scores, weights = [], []
    region_count = 0
    for mask, score in zip(gt, maps):
        n, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for component in range(1, n):
            region = labels == component
            area = int(region.sum())
            foreground_scores.append(score[region])
            weights.append(np.full(area, 1/(area+1e-8), np.float64))
            region_count += 1
    bg = maps[~gt]
    if not region_count or not len(bg):
        return None
    lo, hi = float(maps.min()), float(maps.max())
    if hi-lo < 1e-12:
        return 0.
    fg = np.concatenate(foreground_scores)
    weight = np.concatenate(weights)/region_count
    order = np.argsort(fg)
    fg, weight = fg[order], weight[order]
    cumulative = np.r_[0., np.cumsum(weight)]
    bg = np.sort(bg)
    thresholds = np.linspace(lo, hi, num_th, dtype=np.float32)
    fprs = (len(bg)-np.searchsorted(bg, thresholds, side="left"))/len(bg)
    pros = cumulative[-1]-cumulative[np.searchsorted(fg, thresholds, side="left")]
    keep = fprs <= max_fpr
    fprs, pros = fprs[keep], pros[keep]
    if len(fprs) < 2:
        return 0.
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    fprs, indices = np.unique(fprs, return_index=True)
    pros = pros[indices]
    if len(fprs) < 2:
        return 0.
    if fprs[0] > 0:
        fprs, pros = np.r_[0., fprs], np.r_[pros[0], pros]
    if fprs[-1] < max_fpr:
        fprs, pros = np.r_[fprs, max_fpr], np.r_[pros, pros[-1]]
    x, y = np.clip(fprs, 0, max_fpr)/max_fpr, np.clip(pros, 0, 1)
    return float(np.sum(np.diff(x)*(y[1:]+y[:-1])/2))


def map_metrics(masks, maps, labels, num_th=1000, max_fpr=.3):
    pixel = binary_metrics(masks, maps)
    image = binary_metrics(labels, maps.reshape(len(maps), -1).max(1))
    bg = maps[~np.asarray(masks, bool)]
    return dict(P_AUROC=pixel["AUROC"], F1=pixel["F1"],
                I_AUROC=image["AUROC"], PRO=sampled_pro(masks, maps, num_th, max_fpr),
                background_FPR_at_05=float((bg >= .5).mean()) if len(bg) else None)
