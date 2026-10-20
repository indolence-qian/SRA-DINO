"""Localization/risk ablations of PPO. GT is used only AFTER GT-free execution.

The ranking surrogate is NOT PRO. Oracle search optimizes this surrogate, not a
global PRO bound. P3 is Lagrangian PPO, NOT CPO or a safety guarantee.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import math

import cv2
import numpy as np
import torch

from tools.mara_agent import quality_score
from tools.mara_ppo import MARAPPOAgent, return_targets, score_trajectory

PROTOCOL = "mara_ppo_locsafe_v1"
ARMS = ("P0", "P1", "P2", "P3")


@dataclass
class LocSafeConfig:
    hard_fraction: float = .05
    rank_margin: float = .1
    rank_weight: float = .5
    rank_points: int = 128
    risk_budget: float = .001
    dual_initial: float = 1.
    dual_lr: float = 10.
    normal_fraction: float = .5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Nonfinite LocSafe config")
        if not 0 < self.hard_fraction <= 1 or not 0 < self.normal_fraction < 1:
            raise ValueError("Invalid hard/normal fraction")
        if self.rank_points < 1 or min(self.rank_margin, self.rank_weight, self.risk_budget,
                                      self.dual_initial, self.dual_lr) < 0:
            raise ValueError("Invalid LocSafe reward/dual config")


def evenly(values, limit):
    """Deterministic quadrature, never consumes policy/PPO RNG."""
    if len(values) <= limit:
        return values
    ids = torch.linspace(0, len(values)-1, limit, device=values.device).round().long()
    return values[ids]


def tail_mean(prob, fraction=.05):
    flat = prob[:, 1].flatten(1)
    return flat.topk(max(1, math.ceil(flat.shape[1]*fraction)), dim=1).values.mean(1)


class RegionRank:
    """Small CPU index LRU; no image tensors/GT channels enter Actor/Critic.

    Components are computed at evaluation resolution (8-connectivity), so small
    defects are not erased on the 128-grid. Samples/ordering are deterministic.
    """
    def __init__(self, settings, cache_size=256):
        self.settings, self.cache_size = settings, cache_size
        self.cache = OrderedDict()

    def indices(self, mask):
        native = np.ascontiguousarray(mask > .5, dtype=np.uint8)
        key = (native.shape, hashlib.sha256(native.tobytes()).hexdigest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        count, components = cv2.connectedComponents(native, connectivity=8)
        foreground = []
        for k in range(1, count):
            ids = np.flatnonzero(components.reshape(-1) == k)
            if len(ids) > self.settings.rank_points:
                ids = ids[np.linspace(0, len(ids)-1, self.settings.rank_points).round().astype(int)]
            foreground.append(torch.from_numpy(ids.copy()).long())
        value = (foreground, torch.from_numpy(np.flatnonzero(native.reshape(-1) == 0).copy()).long())
        self.cache[key] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    @torch.no_grad()
    def prepare(self, mask, labels):
        # One bulk transfer per reward batch, not one GPU synchronization per
        # image/component/state. Repeated trajectories share transferred indices.
        native = (mask > .5).detach().cpu().numpy()
        is_anomaly = labels.detach().cpu().tolist()
        transferred, prepared = {}, []
        for i, label in enumerate(is_anomaly):
            if not label:
                prepared.append(None)
                continue
            value = self.indices(native[i, 0])
            key = id(value)
            if key not in transferred:
                regions, bg = value
                transferred[key] = ([r.to(mask.device) for r in regions], bg.to(mask.device))
            prepared.append(transferred[key])
        return prepared

    @torch.no_grad()
    def __call__(self, probability, mask, labels, prepared=None):
        prepared = self.prepare(mask, labels) if prepared is None else prepared
        losses = probability.new_zeros(len(probability))
        for i, indices in enumerate(prepared):
            if indices is None:
                continue
            regions, background = indices
            if not regions:
                raise ValueError("Anomaly without HR GT foreground; run GT audit")
            if not len(background):
                continue  # All-foreground GT has no ranking pairs.
            flat = probability[i, 1].flatten()
            bg = flat[background.to(flat.device)]
            hard = bg.topk(max(1, math.ceil(len(bg)*self.settings.hard_fraction))).values
            hard = evenly(hard, self.settings.rank_points)
            losses[i] = torch.stack([
                (self.settings.rank_margin+hard[None, :]-flat[ids.to(flat.device)][:, None]).clamp_min(0).mean()
                for ids in regions]).mean()
        return losses


class LocSafeAgent(MARAPPOAgent):
    def __init__(self, cfg, ppo, settings, arm="P0"):
        if arm not in ARMS:
            raise ValueError("Unknown LocSafe arm")
        super().__init__(cfg, ppo)
        self.settings, self.arm = settings, arm
        # Same initialized state across ALL arms. Only P3 uses/updates this.
        self.register_buffer("dual", torch.tensor(settings.dual_initial, dtype=torch.float64))
        self.ranker = RegionRank(settings)
        self.reset_epoch_logs()

    def reset_epoch_logs(self):
        self._normal_risks = []
        self._normal_reference = []

    @torch.no_grad()
    def localization_quality(self, prob, logits, mask, labels, ranking=True, prepared=None):
        config = replace(self.cfg, reward_cls_weight=0.)
        score = quality_score(prob, logits, mask, labels, config)
        rank = self.ranker(prob, mask, labels, prepared) if ranking else score.new_zeros(len(score))
        return score-self.settings.rank_weight*rank, rank

    @torch.no_grad()
    def score(self, result, mask, labels):
        # Build the unchanged P0 reference rewards before adding any interventions.
        score_trajectory(result, mask, labels, self.cfg, self.ppo)
        steps = result["steps"]
        for s in steps:
            s["reference_reward"] = s["reward"].clone()
            s["reference_return"] = s["return"].clone()
        quality, ranks = [], []
        prepared = self.ranker.prepare(mask, labels) if self.arm in ("P2", "P3") else None
        states = [(steps[0]["before_hr"], steps[0]["before_logits"])]
        states.extend((s["after_hr"], s["after_logits"]) for s in steps)
        for prob, logits in states:
            if self.arm == "P0":
                q = quality_score(prob, logits, mask, labels, self.cfg)
                rank = q.new_zeros(len(q))
            else:
                q, rank = self.localization_quality(prob, logits, mask, labels, self.arm in ("P2", "P3"), prepared)
            quality.append(q)
            ranks.append(rank)
        initial_tail = tail_mean(states[0][0], self.settings.hard_fraction)
        normal = (labels == 0).to(initial_tail.dtype)
        for t, s in enumerate(steps):
            s["quality_before"], s["quality_after"] = quality[t], quality[t+1]
            s["rank_before"], s["rank_after"] = ranks[t], ranks[t+1]
            s["tail_before"] = tail_mean(s["before_hr"], self.settings.hard_fraction)
            s["tail_after"] = tail_mean(s["after_hr"], self.settings.hard_fraction)
            s["risk_increase"] = normal*(s["tail_after"]-initial_tail).clamp_min(0)
            s["risk_cost"] = s["risk_increase"]*(s["valid"] & s["done"])
            penalty = self.dual.to(initial_tail.dtype)*s["risk_cost"] if self.arm == "P3" else 0.
            s["reward"] = (quality[t+1]-quality[t]-(self.cfg.step_cost+self.cfg.refine_cost)*s["execute"])*s["valid"]-penalty
        stack = lambda key: torch.stack([s[key] for s in steps], 1)
        returns, advantages = return_targets(stack("reward"), stack("done"), stack("valid"), stack("old_value"),
                                             self.ppo.gamma, self.ppo.gae_lambda, self.ppo.use_value)
        for t, s in enumerate(steps):
            s["return"], s["advantage"] = returns[:, t], advantages[:, t]

    def extra_summary(self, result, mask, labels):
        steps, normal = result["steps"], labels == 0
        risk = steps[-1]["risk_increase"][normal].cpu().tolist()
        reference = steps[0]["reference_return"][normal].cpu().tolist()
        self._normal_risks.extend(risk)
        self._normal_reference.extend(reference)
        return dict(reference_return_mean=float(steps[0]["reference_return"].mean()),
                    rank_improvement_mean=float((steps[0]["rank_before"]-steps[-1]["rank_after"]).mean()),
                    normal_risk_mean=sum(risk)/max(1, len(risk)),
                    normal_risk_violation_count=sum(v > self.settings.risk_budget for v in risk),
                    dual=float(self.dual))

    def finish_epoch(self):
        values = sorted(self._normal_risks)
        references = sorted(self._normal_reference)
        mean = sum(values)/max(1, len(values))
        used = float(self.dual)
        next_dual = used
        if self.arm == "P3" and values:
            # E[C] <= rho_N*epsilon: unconditional penalty and matching dual.
            next_dual = max(0., used+self.settings.dual_lr*self.settings.normal_fraction*(mean-self.settings.risk_budget))
        count = max(1, math.ceil(len(values)*.05))
        return dict(dual_used=used, dual_next=next_dual, normal_risk_mean=mean,
                    normal_risk_p95=float(np.quantile(values, .95)) if values else 0.,
                    normal_risk_worst5_mean=sum(values[-count:])/count,
                    normal_risk_max=max(values, default=0.),
                    normal_reference_worst5_mean=sum(references[:count])/count,
                    normal_reference_catastrophes=sum(v < -.5 for v in references))


def nested_split(manifest, audited, historical, seed, fraction=.2):
    """Never put historical holdout/target samples into the new fit/dev pool."""
    from tools.mara_diagnostics import stable_id
    labels = {r["sample_id"]: r["image_label"] for r in audited["rows"] if r["included"]}
    pool = set(historical["train_ids"])
    strata = {}
    for record in manifest:
        if record["id"] in pool:
            strata.setdefault((record["category"], labels[record["id"]]), []).append(record["id"])
    fit, dev = [], []
    for key, ids in sorted(strata.items()):
        if len(ids) < 2:
            raise ValueError(f"Need >=2 fit/development samples per stratum: {key}")
        ids.sort(key=lambda x: stable_id(seed, "locsafe_inner", x))
        count = min(len(ids)-1, max(1, round(len(ids)*fraction)))
        dev.extend(ids[:count])
        fit.extend(ids[count:])
    if not fit or not dev:
        raise ValueError("Empty fit/development pool")
    out = dict(historical, train_ids=sorted(fit), calibration_ids=sorted(dev),
               historical_calibration_ids=historical["calibration_ids"],
               inner_split_seed=seed, development_fraction=fraction,
               note="New MARA fit/dev inside historical source training pool; Base/Refiner independence NOT guaranteed")
    groups = [set(out[k]) for k in ("train_ids", "calibration_ids", "historical_calibration_ids", "target_ids")]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError("Data leakage in nested split")
    return out


def oracle_actions(prefixes, horizon, regions, layers):
    """STOP remains a candidate; completed paths never restart."""
    candidates = []
    for prefix in prefixes:
        if prefix and prefix[-1][0] == 0:
            candidates.append(prefix)
        else:
            candidates.append(prefix+[(0, 0, 0)])
            candidates.extend(prefix+[(1, r, l)] for r in range(regions) for l in range(layers))
    # Stable deduplication, including identical STOP padding.
    return list(dict.fromkeys(tuple(p+[(0, 0, 0)]*(horizon-len(p))) for p in candidates))


@torch.no_grad()
def search_candidates(agent, inputs, mask, labels, prefixes, depth, width=8, batch_size=16):
    """Evaluate a finite candidate pool. Not a PRO oracle/global upper bound."""
    sequences = oracle_actions(prefixes, agent.cfg.max_steps, agent.cfg.num_regions, agent.cfg.num_layers)
    base_quality, _ = agent.localization_quality(inputs["base_prob"], inputs["base_logits"], mask, labels)
    rows, best_maps = [], {}
    for offset in range(0, len(sequences), batch_size):
        chunk = sequences[offset:offset+batch_size]
        array = torch.tensor(chunk, device=mask.device, dtype=torch.long)
        actions = {name: array[:, :, k] for k, name in enumerate(("op", "region", "layer"))}
        repeated = {k: v.repeat_interleave(len(chunk), 0) for k, v in inputs.items()}
        result = agent.trajectory(repeated, sample=False, action_sequence=actions)
        q, ranking = agent.localization_quality(result["final_prob"], result["final_logits"],
            mask.repeat_interleave(len(chunk), 0), labels.repeat_interleave(len(chunk), 0))
        count = array[:, :, 0].sum(1)
        gain = q-base_quality-(agent.cfg.step_cost+agent.cfg.refine_cost)*count
        for i, seq in enumerate(chunk):
            row = dict(candidate=offset+i, depth=depth, path=[list(x) for x in seq[:depth]],
                       proxy_gain=float(gain[i]), rank_loss=float(ranking[i]), refines=int(count[i]),
                       pixel_change=float((result["final_prob"][i]-inputs["base_prob"][0]).abs().max()))
            rows.append(row)
            # Retain only top-width maps, never all K*L HR tensors on CPU/disk.
            current = sorted(rows, key=lambda r: (-r["proxy_gain"], r["refines"], r["candidate"]))[:width]
            keep = {r["candidate"] for r in current}
            if row["candidate"] in keep:
                best_maps[row["candidate"]] = result["final_prob"][i, 1].detach().cpu().numpy().copy()
            best_maps = {k: v for k, v in best_maps.items() if k in keep}
    ranked = sorted(rows, key=lambda r: (-r["proxy_gain"], r["refines"], r["candidate"]))[:width]
    next_prefixes = [[tuple(x) for x in row["path"]] for row in ranked]
    return rows, next_prefixes, ranked[0], best_maps[ranked[0]["candidate"]]
