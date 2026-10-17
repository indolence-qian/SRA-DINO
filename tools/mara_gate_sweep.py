"""Batched, GT-free gate-sweep trajectories and post-hoc scoring."""
from dataclasses import replace
import math

import numpy as np
import torch
import torch.nn.functional as F

from tools.mara_agent import _compose_high_res_probability, _resize_map
from tools.mara_diagnostics import validate_cache

PROTOCOL = "mara_gate_sweep_v1"
METRICS = ("PRO", "P_AUROC", "I_AUROC", "F1", "background_FPR_at_05")


def make_arms(text):
    margins = [float(x) for x in text.split(",")]
    if not margins or margins[0] != 0 or len(set(margins)) != len(margins):
        raise ValueError("Margins must be unique and begin with 0")
    if any(not math.isfinite(x) or x > 0 for x in margins):
        raise ValueError("Gate sweep margins must be finite and nonpositive")
    arms = {f"G{i}": dict(gain_safety_margin=m, bypass_gain=False, freeze_noop_state=True)
            for i, m in enumerate(margins)}
    arms[f"G{len(margins)}_bypass"] = dict(
        gain_safety_margin=0., bypass_gain=True, freeze_noop_state=True)
    arms["L0_legacy"] = dict(gain_safety_margin=0., bypass_gain=False, freeze_noop_state=False)
    return arms


@torch.no_grad()
def trace_batch(agent, inputs, spec):
    """Same production policy/refiner, batched images; no GT accepted."""
    if set(inputs)-{"base_prob", "base_logits", "layer_maps", "evidence_maps", "global_evidence"}:
        raise ValueError("GT is forbidden in gate-sweep action inputs")
    validate_cache(inputs, agent.cfg, allow_batch=True)
    original = agent.cfg
    cfg = replace(original, gain_safety_margin=spec["gain_safety_margin"],
                  freeze_noop_state=spec["freeze_noop_state"],
                  use_gain_gate=original.use_gain_gate and not spec["bypass_gain"])
    agent.cfg = cfg
    try:
        base_hr = inputs["base_prob"]
        base = _resize_map(base_hr, cfg.map_size)
        current = base
        logits = torch.log(base.clamp(1e-6, 1.))
        global_logits = inputs["base_logits"]
        layers = _resize_map(inputs["layer_maps"], cfg.map_size)
        evidence = inputs.get("evidence_maps")
        evidence = None if evidence is None else _resize_map(evidence, cfg.map_size)
        active = torch.ones(base.shape[0], dtype=torch.bool, device=base.device)
        gate = torch.zeros_like(base[:, :1])
        traces = [dict(records=[], maps=[]) for _ in range(len(base))]
        batch_records = []
        before_hr = base_hr
        for step in range(cfg.max_steps):
            state = agent._make_state(current, base, layers, step, evidence)
            policy = agent._policy(state, global_logits, active, step, sample=False,
                                   global_evidence=inputs.get("global_evidence"))
            out = agent._refine(base, inputs["base_logits"], logits, current, global_logits,
                                gate, state, layers, policy, active, True, True)
            after_hr, gate_hr = _compose_high_res_probability(
                base_hr, out["next_logits"], out["next_cumulative_gate"])
            candidate_hr, _ = _compose_high_res_probability(
                base_hr, out["candidate_logits"], out["candidate_cumulative_gate"])
            tensors = dict(before_prob=current, before_global=global_logits, before_hr=before_hr,
                           after_prob=out["next_prob"], after_global=out["next_global_logits"], after_hr=after_hr,
                           candidate_prob=out["candidate_prob"], candidate_global=out["candidate_global_logits"],
                           candidate_hr=candidate_hr)
            batch_records.append(tensors)
            safety = out["predicted_gain_lower"] if cfg.use_gain_lower_bound else out["predicted_gain"]
            lower = (safety <= cfg.gain_safety_margin) & cfg.use_gain_gate
            probability = (out["accept_score"] < cfg.gain_accept_probability) & cfg.use_gain_gate
            ops = policy["op_logits"].softmax(-1)
            scalars = dict(active_before=active, active_after=out["next_active"], op=policy["op_action"],
                           region=policy["region_action"], layer=policy["layer_action"],
                           p_stop=ops[:, 0], p_refine=ops[:, 1], predicted_gain=out["predicted_gain"],
                           predicted_gain_lower=out["predicted_gain_lower"], accept_probability=out["accept_score"],
                           gain_accepted=out["gain_accepted"], lower_failed=lower, probability_failed=probability,
                           execute_weight=out["execute_weight"], gate_mean=gate_hr.flatten(1).mean(1),
                           update_gate_mean=out["update_gate"].flatten(1).mean(1),
                           pixel_change_max=(after_hr-before_hr).abs().flatten(1).amax(1),
                           global_change_max=(out["next_global_logits"]-global_logits).abs().amax(1))
            numbers = torch.stack([v.float() for v in scalars.values()], 1).cpu().tolist()
            region_probs = policy["region_logits"].softmax(-1).cpu().tolist()
            layer_probs = policy["layer_logits"].softmax(-1).cpu().tolist()
            boxes = policy["box_features"].cpu().tolist()
            global_before = global_logits.cpu().tolist()
            global_after = out["next_global_logits"].cpu().tolist()
            for i, values in enumerate(numbers):
                row = dict(zip(scalars, values))
                for name in ("active_before", "active_after", "gain_accepted", "lower_failed", "probability_failed"):
                    row[name] = bool(row[name])
                for name in ("op", "region", "layer"):
                    row[name] = int(row[name])
                if not row["active_before"]:
                    reason = "inactive"
                elif row["op"] == 0:
                    reason = "policy_stop"
                elif not row["active_after"]:
                    reason = "gain_reject_both" if row["lower_failed"] and row["probability_failed"] else (
                        "gain_reject_lower" if row["lower_failed"] else "gain_reject_probability")
                else:
                    reason = "horizon" if step == cfg.max_steps-1 else "continue"
                row.update(step=step, stop_reason=reason, forced_refine=False,
                           bypass_gain=spec["bypass_gain"], safety_margin=spec["gain_safety_margin"],
                           freeze_noop_state=spec["freeze_noop_state"],
                           region_probabilities=region_probs[i], layer_probabilities=layer_probs[i],
                           region_prob=region_probs[i][row["region"]],
                           layer_prob=layer_probs[i][row["layer"]], box=boxes[i][row["region"]],
                           global_logits_before=global_before[i], global_logits_after=global_after[i])
                traces[i]["records"].append(row)
                traces[i]["maps"].append(after_hr[i:i+1])
            logits, current, global_logits, gate, active = (
                out["next_logits"], out["next_prob"], out["next_global_logits"],
                out["next_cumulative_gate"], out["next_active"])
            before_hr = after_hr
        return dict(traces=traces, records=batch_records, final_prob=after_hr,
                    final_logits=global_logits, gate_map=gate_hr)
    finally:
        agent.cfg = original


def _quality_batch(prob, global_logits, mask, labels, cfg):
    gt = (_resize_map(mask.float(), prob.shape[-1], mode="nearest")[:, 0] > .5).float().flatten(1)
    p = prob[:, 1].clamp(0, 1).flatten(1)
    cls = -F.cross_entropy(global_logits, labels.long(), reduction="none")
    dice = (2*(p*gt).sum(1)+1e-6)/(p.sum(1)+gt.sum(1)+1e-6)
    loc = labels.float()*dice+(1-labels.float())*(1-p.mean(1))
    image_prob = global_logits.softmax(-1)
    conf = 1+(image_prob*image_prob.clamp(1e-6, 1).log()).sum(1)/math.log(2.)
    bg = (p*(1-gt)).sum(1)/(1-gt).sum(1).clamp_min(1e-6)
    quality = cfg.reward_cls_weight*cls+cfg.reward_loc_weight*loc+cfg.reward_conf_weight*conf-cfg.reward_fp_weight*bg
    return torch.stack([quality, cls, loc, conf, bg, gt.sum(1)], 1).cpu().numpy()


@torch.no_grad()
def score_batch(result, mask, labels, cfg):
    """GT enters only after the complete GT-free trajectory has returned."""
    traces = result["traces"]
    for step, tensors in enumerate(result["records"]):
        for prefix, prob_key, global_key in (
            ("before_lr", "before_prob", "before_global"), ("after_lr", "after_prob", "after_global"),
            ("candidate_lr", "candidate_prob", "candidate_global"), ("before_hr", "before_hr", "before_global"),
            ("after_hr", "after_hr", "after_global"), ("candidate_hr", "candidate_hr", "candidate_global"),
        ):
            parts = _quality_batch(tensors[prob_key], tensors[global_key], mask, labels, cfg)
            for trace, values in zip(traces, parts):
                trace["records"][step].update({prefix+"_"+k: int(v) if k == "gt_pixels" else float(v)
                                              for k, v in zip(("quality", "q_cls", "q_loc", "q_conf", "q_bg", "gt_pixels"), values)})
        for trace in traces:
            row = trace["records"][step]
            raw = row["after_lr_quality"]-row["before_lr_quality"]
            candidate = row["candidate_lr_quality"]-row["before_lr_quality"]
            active = float(row["active_before"])
            step_cost, refine_cost = cfg.step_cost*active, cfg.refine_cost*active*row["op"]
            row.update(delta_quality_lr=raw, delta_quality_hr=row["after_hr_quality"]-row["before_hr_quality"],
                       candidate_delta_quality_lr=candidate,
                       candidate_delta_quality_hr=row["candidate_hr_quality"]-row["before_hr_quality"],
                       candidate_valid=row["active_before"], candidate_gain_target=candidate-cfg.refine_cost,
                       candidate_net_reward=candidate-cfg.step_cost-cfg.refine_cost,
                       step_cost=step_cost, refine_cost=refine_cost,
                       reward=(raw-step_cost-refine_cost)*active, reward_without_cost=raw*active)
            row["cost_flips_reward_sign"] = row["reward_without_cost"] > 0 and row["reward"] <= 0
            for key in ("q_cls", "q_loc", "q_bg", "q_conf"):
                row["delta_"+key] = row["after_lr_"+key]-row["before_lr_"+key]
    for trace in traces:
        total = costless = 0.
        for row in reversed(trace["records"]):
            total = row["reward"]+cfg.gamma*total
            costless = row["reward_without_cost"]+cfg.gamma*costless
            row.update(return_to_go=total, return_to_go_without_cost=costless)
    return traces


def summarize_rows(rows, images, tolerance):
    active = [r for r in rows if r["active_before"]]
    attempted = [r for r in active if r["op"] == 1]
    rejected = [r for r in attempted if r["execute_weight"] == 0]
    executed = [r for r in attempted if r["execute_weight"] > 0]
    beneficial = [r for r in attempted if r["candidate_net_reward"] > tolerance]
    stopped = [r for r in rows if not r["active_before"] or r["op"] == 0 or (
        r["active_before"] and r["op"] == 1 and r["execute_weight"] == 0)]
    mean = lambda values: float(np.mean(values)) if values else None
    first = [r for r in rows if r["step"] == 0]
    return dict(images=images, active_decisions=len(active), attempted_refinements=len(attempted),
                executed_refinements=len(executed), mean_refinements_per_image=len(executed)/images,
                policy_stop_images=sum(r["op"] == 0 for r in active), gain_reject_images=len(rejected),
                first_step_policy_stop=sum(r["op"] == 0 for r in first),
                first_step_gain_reject=sum(r["op"] == 1 and r["execute_weight"] == 0 for r in first),
                beneficial_candidates=len(beneficial),
                beneficial_candidates_rejected=sum(r["execute_weight"] == 0 for r in beneficial),
                beneficial_candidate_rejected_fraction=mean([r["execute_weight"] == 0 for r in beneficial]),
                negative_reward_executions=sum(r["reward"] < 0 for r in executed),
                lower_rejections=sum(r["lower_failed"] for r in rejected),
                probability_rejections=sum(r["probability_failed"] for r in rejected),
                return_mean=mean([r["return_to_go"] for r in first]),
                stop_output_drift_records=sum(r["global_change_max"] > tolerance or r["pixel_change_max"] > tolerance for r in stopped))
