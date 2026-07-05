import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class MARAConfig:
    num_layers: int = 4
    map_size: int = 128
    num_regions: int = 16
    roi_size: int = 32
    hidden_dim: int = 64
    max_steps: int = 3
    group_size: int = 4
    gamma: float = 0.95
    grpo_clip_eps: float = 0.2
    entropy_coef: float = 0.01
    step_cost: float = 0.01
    refine_cost: float = 0.02
    reward_cls_weight: float = 0.5
    reward_loc_weight: float = 1.0
    reward_conf_weight: float = 0.1
    reward_fp_weight: float = 0.2
    delta_scale: float = 1.0
    force_first_refine: bool = False
    base_anchor_margin: float = 0.0
    negative_advantage_scale: float = 1.0
    advantage_clip: float = 5.0
    gate_max: float = 0.35
    gate_init_bias: float = -4.0
    use_gain_gate: bool = True
    gain_accept_threshold: float = 0.0
    gain_gate_temperature: float = 0.1
    gain_loss_clip: float = 1.0


def _resize_map(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == (size, size):
        return x
    if mode == "nearest":
        return F.interpolate(x, size=(size, size), mode=mode)
    return F.interpolate(x, size=(size, size), mode=mode, align_corners=False)


def _prob_to_logits(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(prob.clamp(eps, 1.0))


def _entropy_2class(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ent = -(prob.clamp(eps, 1.0) * prob.clamp(eps, 1.0).log()).sum(dim=1, keepdim=True)
    return ent / math.log(2.0)


def _masked_mean(features: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # features: (B,C,H,W), masks: (B,K,H,W) -> (B,K,C)
    denom = masks.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    pooled = torch.einsum("bchw,bkhw->bkc", features, masks)
    return pooled / denom.squeeze(-1)


def _boxes_from_centers(
    centers_y: torch.Tensor,
    centers_x: torch.Tensor,
    height: int,
    width: int,
    roi_size: int,
) -> torch.Tensor:
    half = max(1, roi_size // 2)
    y1 = (centers_y - half).clamp(0, height - 1)
    x1 = (centers_x - half).clamp(0, width - 1)
    y2 = (centers_y + half).clamp(0, height - 1)
    x2 = (centers_x + half).clamp(0, width - 1)
    return torch.stack([y1, x1, y2, x2], dim=-1)


def build_candidate_masks(
    anomaly_map: torch.Tensor,
    uncertainty_map: torch.Tensor,
    num_regions: int,
    roi_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build lightweight ROI candidates from high anomaly and high uncertainty areas."""
    bsz, _, height, width = anomaly_map.shape
    device = anomaly_map.device
    half_k = max(1, num_regions // 2)
    rest_k = num_regions - half_k

    anomaly_flat = anomaly_map.detach().reshape(bsz, -1)
    uncert_flat = uncertainty_map.detach().reshape(bsz, -1)
    _, top_anom = torch.topk(anomaly_flat, k=min(half_k, anomaly_flat.shape[1]), dim=1)
    if rest_k > 0:
        _, top_unc = torch.topk(uncert_flat, k=min(rest_k, uncert_flat.shape[1]), dim=1)
        flat_idx = torch.cat([top_anom, top_unc], dim=1)
    else:
        flat_idx = top_anom

    if flat_idx.shape[1] < num_regions:
        pad = flat_idx[:, :1].expand(-1, num_regions - flat_idx.shape[1])
        flat_idx = torch.cat([flat_idx, pad], dim=1)

    centers_y = flat_idx // width
    centers_x = flat_idx % width
    boxes = _boxes_from_centers(centers_y, centers_x, height, width, roi_size)

    masks = torch.zeros(bsz, num_regions, height, width, device=device, dtype=anomaly_map.dtype)
    for b in range(bsz):
        for k in range(num_regions):
            y1, x1, y2, x2 = boxes[b, k].tolist()
            masks[b, k, y1 : y2 + 1, x1 : x2 + 1] = 1.0

    norm = torch.tensor(
        [max(1, height - 1), max(1, width - 1), max(1, height - 1), max(1, width - 1)],
        device=device,
        dtype=anomaly_map.dtype,
    )
    box_features = boxes.to(anomaly_map.dtype) / norm
    return masks, box_features


def quality_score(
    prob_map: torch.Tensor,
    global_logits: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    cfg: MARAConfig,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Stage quality Q_t used to build step rewards."""
    if mask.dim() == 4:
        mask_bhw = mask[:, 0]
    else:
        mask_bhw = mask
    mask_bhw = _resize_map(mask_bhw.unsqueeze(1).float(), prob_map.shape[-1], mode="nearest")[:, 0]
    mask_bhw = (mask_bhw > 0.5).float()

    anomaly = prob_map[:, 1].clamp(0, 1)
    labels_f = labels.float()
    ce = F.cross_entropy(global_logits, labels, reduction="none")
    q_cls = -ce

    pred_flat = anomaly.reshape(anomaly.shape[0], -1)
    gt_flat = mask_bhw.reshape(mask_bhw.shape[0], -1)
    inter = (pred_flat * gt_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    normal_quality = 1.0 - pred_flat.mean(dim=1)
    q_loc = labels_f * dice + (1.0 - labels_f) * normal_quality

    image_prob = torch.softmax(global_logits, dim=-1)
    q_conf = 1.0 + (image_prob * image_prob.clamp(eps, 1.0).log()).sum(dim=1) / math.log(2.0)

    normal_pixels = 1.0 - gt_flat
    q_fp = (pred_flat * normal_pixels).sum(dim=1) / normal_pixels.sum(dim=1).clamp_min(eps)

    return (
        cfg.reward_cls_weight * q_cls
        + cfg.reward_loc_weight * q_loc
        + cfg.reward_conf_weight * q_conf
        - cfg.reward_fp_weight * q_fp
    )


class MARAAgent(nn.Module):
    """Multi-step Anomaly Refinement Agent with GRPO-style policy training."""

    def __init__(self, cfg: MARAConfig):
        super().__init__()
        self.cfg = cfg
        state_channels = 4 + cfg.num_layers
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_channels, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
        )
        self.region_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 4, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 3, cfg.hidden_dim),
            nn.GELU(),
        )
        self.layer_head = nn.Linear(cfg.hidden_dim, cfg.num_layers)
        self.op_head = nn.Linear(cfg.hidden_dim, 2)  # 0: stop, 1: refine
        self.refiner = nn.Sequential(
            nn.Conv2d(state_channels + 2, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(cfg.hidden_dim, 2, kernel_size=1)
        self.gate_head = nn.Conv2d(cfg.hidden_dim, 1, kernel_size=1)
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.gain_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, cfg.gate_init_bias)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)

    def _make_state(
        self,
        current_prob: torch.Tensor,
        base_prob: torch.Tensor,
        layer_maps: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        bsz, _, height, width = current_prob.shape
        uncertainty = _entropy_2class(current_prob)
        step_plane = torch.full(
            (bsz, 1, height, width),
            float(step_idx) / float(max(1, self.cfg.max_steps)),
            device=current_prob.device,
            dtype=current_prob.dtype,
        )
        return torch.cat(
            [
                current_prob[:, 1:2],
                uncertainty,
                base_prob[:, 1:2],
                step_plane,
                layer_maps,
            ],
            dim=1,
        )

    def _policy(
        self,
        state: torch.Tensor,
        global_logits: torch.Tensor,
        active: torch.Tensor,
        step_idx: int,
        sample: bool,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        encoded = self.state_encoder(state)
        anomaly = state[:, 0:1]
        uncertainty = state[:, 1:2]
        masks, box_features = build_candidate_masks(
            anomaly_map=anomaly,
            uncertainty_map=uncertainty,
            num_regions=cfg.num_regions,
            roi_size=cfg.roi_size,
        )
        region_features = _masked_mean(encoded, masks)
        region_logits = self.region_head(torch.cat([region_features, box_features], dim=-1)).squeeze(-1)

        pooled = encoded.mean(dim=(-2, -1))
        image_prob = torch.softmax(global_logits, dim=-1)[:, 1:2]
        step_feat = torch.full_like(image_prob, float(step_idx) / float(max(1, cfg.max_steps)))
        active_feat = active.float().unsqueeze(1)
        global_features = self.global_head(torch.cat([pooled, image_prob, step_feat, active_feat], dim=1))
        layer_logits = self.layer_head(global_features)
        op_logits = self.op_head(global_features)
        if step_idx == 0 and cfg.force_first_refine:
            op_logits = op_logits.clone()
            op_logits[:, 0] = -1e4

        region_dist = Categorical(logits=region_logits)
        layer_dist = Categorical(logits=layer_logits)
        op_dist = Categorical(logits=op_logits)

        if sample:
            region_action = region_dist.sample()
            layer_action = layer_dist.sample()
            op_action = op_dist.sample()
        else:
            region_action = region_logits.argmax(dim=1)
            layer_action = layer_logits.argmax(dim=1)
            op_action = op_logits.argmax(dim=1)

        logprob = (
            region_dist.log_prob(region_action)
            + layer_dist.log_prob(layer_action)
            + op_dist.log_prob(op_action)
        )
        entropy = region_dist.entropy() + layer_dist.entropy() + op_dist.entropy()
        logprob = logprob * active.float()
        entropy = entropy * active.float()

        return {
            "encoded": encoded,
            "masks": masks,
            "region_action": region_action,
            "layer_action": layer_action,
            "op_action": op_action,
            "logprob": logprob,
            "entropy": entropy,
        }

    def _refine(
        self,
        base_prob: torch.Tensor,
        base_logits: torch.Tensor,
        current_logits: torch.Tensor,
        current_prob: torch.Tensor,
        global_logits: torch.Tensor,
        cumulative_gate: torch.Tensor,
        state: torch.Tensor,
        layer_maps: torch.Tensor,
        policy_out: Dict[str, torch.Tensor],
        active: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = current_prob.shape[0]
        region_idx = policy_out["region_action"].view(bsz, 1, 1, 1).expand(-1, 1, current_prob.shape[-2], current_prob.shape[-1])
        selected_mask = policy_out["masks"].gather(1, region_idx)

        layer_idx = policy_out["layer_action"].view(bsz, 1, 1, 1).expand(-1, 1, layer_maps.shape[-2], layer_maps.shape[-1])
        selected_layer = layer_maps.gather(1, layer_idx)

        refine_gate = (policy_out["op_action"] == 1).float() * active.float()
        gate = refine_gate.view(bsz, 1, 1, 1)

        refiner_in = torch.cat([state, selected_layer, selected_mask], dim=1)
        hidden = self.refiner(refiner_in)
        predicted_gain = self.gain_head(hidden).squeeze(1)
        if self.cfg.use_gain_gate:
            temperature = max(float(self.cfg.gain_gate_temperature), 1e-6)
            accept_score = torch.sigmoid((predicted_gain - self.cfg.gain_accept_threshold) / temperature)
        else:
            accept_score = torch.ones_like(predicted_gain)
        effective_gate = gate * accept_score.view(bsz, 1, 1, 1)

        update_gate = torch.sigmoid(self.gate_head(hidden)) * self.cfg.gate_max
        update_gate = update_gate * selected_mask * effective_gate
        next_cumulative_gate = 1.0 - (1.0 - cumulative_gate) * (1.0 - update_gate)

        delta = self.delta_head(hidden) * selected_mask * effective_gate * self.cfg.delta_scale
        proposal_logits = current_logits + delta
        proposal_prob = torch.softmax(proposal_logits, dim=1)
        next_prob = base_prob * (1.0 - next_cumulative_gate) + proposal_prob * next_cumulative_gate

        gate_scalar = next_cumulative_gate.flatten(1).amax(dim=1)
        delta_score = self.score_head(hidden).squeeze(1) * refine_gate * accept_score * gate_scalar
        proposal_global_logits = global_logits + torch.stack([-delta_score, delta_score], dim=1)
        next_global_logits = base_logits * (1.0 - gate_scalar.unsqueeze(1)) + proposal_global_logits * gate_scalar.unsqueeze(1)
        next_active = active & (policy_out["op_action"] == 1)
        return (
            proposal_logits,
            next_prob,
            next_global_logits,
            next_active,
            next_cumulative_gate,
            update_gate,
            predicted_gain,
            accept_score,
        )

    def rollout(
        self,
        base_prob: torch.Tensor,
        base_logits: torch.Tensor,
        layer_maps: Optional[torch.Tensor],
        mask: torch.Tensor,
        labels: torch.Tensor,
        group_size: Optional[int] = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        group_size = int(group_size or cfg.group_size)
        out_hw = base_prob.shape[-2:]

        base_prob_lr = _resize_map(base_prob, cfg.map_size).detach()
        if layer_maps is None:
            layer_maps_lr = base_prob_lr[:, 1:2].repeat(1, cfg.num_layers, 1, 1)
        else:
            layer_maps_lr = _resize_map(layer_maps, cfg.map_size).detach()
            if layer_maps_lr.shape[1] != cfg.num_layers:
                raise ValueError(f"Expected {cfg.num_layers} layer maps, got {layer_maps_lr.shape[1]}.")

        base_logits = base_logits.detach()
        mask = mask.detach()
        labels = labels.detach()

        base_prob_g = base_prob_lr.repeat_interleave(group_size, dim=0)
        layer_maps_g = layer_maps_lr.repeat_interleave(group_size, dim=0)
        logits_g = base_logits.repeat_interleave(group_size, dim=0)
        mask_g = mask.repeat_interleave(group_size, dim=0)
        labels_g = labels.repeat_interleave(group_size, dim=0)

        current_logits = _prob_to_logits(base_prob_g)
        current_prob = base_prob_g
        global_logits = logits_g
        cumulative_gate = torch.zeros(
            current_prob.shape[0],
            1,
            current_prob.shape[-2],
            current_prob.shape[-1],
            device=current_prob.device,
            dtype=current_prob.dtype,
        )
        active = torch.ones(current_prob.shape[0], device=current_prob.device, dtype=torch.bool)

        logprob_steps = []
        entropy_steps = []
        reward_steps = []
        cost_steps = []
        gate_steps = []
        predicted_gain_steps = []
        accept_steps = []
        value_weight_steps = []
        quality_steps = [quality_score(current_prob, global_logits, mask_g, labels_g, cfg).detach()]

        for step_idx in range(cfg.max_steps):
            q_before = quality_score(current_prob, global_logits, mask_g, labels_g, cfg).detach()
            state = self._make_state(current_prob, base_prob_g, layer_maps_g, step_idx)
            policy_out = self._policy(state, global_logits, active, step_idx, sample=sample)
            active_before = active
            (
                current_logits,
                current_prob,
                global_logits,
                next_active,
                cumulative_gate,
                update_gate,
                predicted_gain,
                accept_score,
            ) = self._refine(
                base_prob=base_prob_g,
                base_logits=logits_g,
                current_logits=current_logits,
                current_prob=current_prob,
                global_logits=global_logits,
                cumulative_gate=cumulative_gate,
                state=state,
                layer_maps=layer_maps_g,
                policy_out=policy_out,
                active=active,
            )
            q_after = quality_score(current_prob, global_logits, mask_g, labels_g, cfg).detach()
            refine_cost = cfg.refine_cost * (policy_out["op_action"] == 1).float()
            step_cost = cfg.step_cost * active.float()
            cost = (refine_cost + step_cost) * active.float()
            reward = (q_after - q_before - cost) * active.float()

            logprob_steps.append(policy_out["logprob"])
            entropy_steps.append(policy_out["entropy"])
            reward_steps.append(reward)
            cost_steps.append(cost)
            gate_steps.append(update_gate.mean(dim=(1, 2, 3)))
            predicted_gain_steps.append(predicted_gain)
            accept_steps.append(accept_score)
            value_weight_steps.append(active_before.float())
            quality_steps.append(q_after)
            active = next_active

        rewards = torch.stack(reward_steps, dim=1)
        costs = torch.stack(cost_steps, dim=1)
        discounts = torch.pow(
            torch.full_like(rewards, cfg.gamma),
            torch.arange(cfg.max_steps, device=rewards.device).view(1, -1),
        )
        base_quality = quality_steps[0]
        final_quality = quality_steps[-1]
        discounted_costs = (costs * discounts).sum(dim=1)
        quality_gain = final_quality - base_quality - cfg.base_anchor_margin
        anchored_score = quality_gain - discounted_costs
        advantage = torch.where(
            anchored_score >= 0,
            anchored_score,
            anchored_score * cfg.negative_advantage_scale,
        )
        advantage_g = advantage.view(-1, group_size)
        advantage_scale = advantage_g.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
        advantage = (advantage_g / advantage_scale).reshape(-1).clamp(
            -cfg.advantage_clip,
            cfg.advantage_clip,
        )

        returns = anchored_score
        logprob_sum = torch.stack(logprob_steps, dim=1).sum(dim=1)
        entropy_mean = torch.stack(entropy_steps, dim=1).mean()
        gate_l1 = torch.stack(gate_steps, dim=1).mean()
        predicted_gain_all = torch.stack(predicted_gain_steps, dim=1)
        accept_all = torch.stack(accept_steps, dim=1)
        value_weights = torch.stack(value_weight_steps, dim=1)
        value_denom = value_weights.sum(dim=1).clamp_min(1.0)
        predicted_gain = (predicted_gain_all * value_weights).sum(dim=1) / value_denom
        accept_mean = (accept_all * value_weights).sum(dim=1) / value_denom
        gain_target = quality_gain.detach().clamp(-cfg.gain_loss_clip, cfg.gain_loss_clip)
        gain_loss = F.smooth_l1_loss(predicted_gain, gain_target)
        old_logprob = logprob_sum.detach()
        ratio = torch.exp(logprob_sum - old_logprob)
        unclipped = ratio * advantage.detach()
        clipped = ratio.clamp(1.0 - cfg.grpo_clip_eps, 1.0 + cfg.grpo_clip_eps) * advantage.detach()
        policy_loss = -torch.min(unclipped, clipped).mean()

        final_prob = _resize_map(current_prob, out_hw[0])
        if final_prob.shape[-2:] != out_hw:
            final_prob = F.interpolate(final_prob, size=out_hw, mode="bilinear", align_corners=False)

        return {
            "final_prob": final_prob,
            "final_logits": global_logits,
            "policy_loss": policy_loss,
            "entropy": entropy_mean,
            "returns": returns.detach(),
            "advantage": advantage.detach(),
            "gate_l1": gate_l1,
            "gain_loss": gain_loss,
            "mean_quality": quality_steps[-1].mean().detach(),
            "mean_base_quality": base_quality.mean().detach(),
            "mean_quality_gain": quality_gain.mean().detach(),
            "mean_predicted_gain": predicted_gain.mean().detach(),
            "mean_accept_score": accept_mean.mean().detach(),
            "mean_reward": rewards.sum(dim=1).mean().detach(),
        }

    def infer(
        self,
        base_prob: torch.Tensor,
        base_logits: torch.Tensor,
        layer_maps: Optional[torch.Tensor] = None,
        sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run deterministic MARA refinement for evaluation without using labels or masks."""
        cfg = self.cfg
        out_hw = base_prob.shape[-2:]

        base_prob_lr = _resize_map(base_prob, cfg.map_size)
        if layer_maps is None:
            layer_maps_lr = base_prob_lr[:, 1:2].repeat(1, cfg.num_layers, 1, 1)
        else:
            layer_maps_lr = _resize_map(layer_maps, cfg.map_size)
            if layer_maps_lr.shape[1] != cfg.num_layers:
                raise ValueError(f"Expected {cfg.num_layers} layer maps, got {layer_maps_lr.shape[1]}.")

        current_logits = _prob_to_logits(base_prob_lr)
        current_prob = base_prob_lr
        global_logits = base_logits
        cumulative_gate = torch.zeros(
            current_prob.shape[0],
            1,
            current_prob.shape[-2],
            current_prob.shape[-1],
            device=current_prob.device,
            dtype=current_prob.dtype,
        )
        active = torch.ones(current_prob.shape[0], device=current_prob.device, dtype=torch.bool)
        predicted_gain_steps = []
        accept_steps = []

        for step_idx in range(cfg.max_steps):
            state = self._make_state(current_prob, base_prob_lr, layer_maps_lr, step_idx)
            policy_out = self._policy(state, global_logits, active, step_idx, sample=sample)
            (
                current_logits,
                current_prob,
                global_logits,
                active,
                cumulative_gate,
                _,
                predicted_gain,
                accept_score,
            ) = self._refine(
                base_prob=base_prob_lr,
                base_logits=base_logits,
                current_logits=current_logits,
                current_prob=current_prob,
                global_logits=global_logits,
                cumulative_gate=cumulative_gate,
                state=state,
                layer_maps=layer_maps_lr,
                policy_out=policy_out,
                active=active,
            )
            predicted_gain_steps.append(predicted_gain)
            accept_steps.append(accept_score)

        final_prob = _resize_map(current_prob, out_hw[0])
        if final_prob.shape[-2:] != out_hw:
            final_prob = F.interpolate(final_prob, size=out_hw, mode="bilinear", align_corners=False)
        gate_map = _resize_map(cumulative_gate, out_hw[0])
        if gate_map.shape[-2:] != out_hw:
            gate_map = F.interpolate(gate_map, size=out_hw, mode="bilinear", align_corners=False)

        return {
            "final_prob": final_prob,
            "final_logits": global_logits,
            "gate_map": gate_map,
            "predicted_gain": torch.stack(predicted_gain_steps, dim=1).mean(dim=1),
            "accept_score": torch.stack(accept_steps, dim=1).mean(dim=1),
        }
