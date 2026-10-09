"""Cached matching, equivalent vectorized losses and source validation plans."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools.upstream_p2 import MatchingHead
from tools.upstream_repair import corrected_probability, reference_logits, operating_threshold


class CachedMatchingHead(MatchingHead):
    def forward_cached(self, visual, base, matching):
        b, layers, dim, h, w = visual.shape
        if (layers, dim) != (self.layers, self.dim) or matching.shape != (b, 3*layers+2, h, w):
            raise ValueError("Cached head input shape mismatch")
        coverage = matching[:, -2:]
        text = self.semantic(matching) * coverage.amax(1, keepdim=True)
        vis = self.visual(visual.reshape(b, layers*dim, h, w))
        coarse = F.interpolate(base, (h, w), mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([vis, text, coverage, coarse], 1))
        size = (min(base.shape[-2], h*4), min(base.shape[-1], w*4))
        return F.interpolate(self.decode(F.interpolate(x, size, mode="bilinear", align_corners=False)),
                             base.shape[-2:], mode="bilinear", align_corners=False)


def geometry(mask, s):
    """Precompute source GT indices and each component's background ring on CPU."""
    mask = np.asarray(mask, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    ids, gt_indices, areas, weights, bg_indices, offsets = [], [], [], [], [], [0]
    radius = s["near_radius"]
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < s["min_area"]:
            continue
        region = labels == label
        positions = np.flatnonzero(region)
        ids.extend([len(areas)]*len(positions))
        gt_indices.extend(positions)
        areas.append(area)
        weights.append(s["small_weight"] if area <= mask.size*s["small_fraction"] else 1.)
        x, y, w, h = stats[label, :4]
        y0, y1 = max(0, y-radius), min(mask.shape[0], y+h+radius)
        x0, x1 = max(0, x-radius), min(mask.shape[1], x+w+radius)
        ring = cv2.dilate(region[y0:y1, x0:x1].astype(np.uint8), np.ones((2*radius+1,)*2, np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        yy, xx = np.where(ring & ~mask[y0:y1, x0:x1])
        indices = (yy+y0)*mask.shape[1]+xx+x0
        bg_indices.extend(indices)
        offsets.append(offsets[-1]+len(indices))
    result = {"gt_indices": torch.tensor(gt_indices, dtype=torch.long),
              "component_ids": torch.tensor(ids, dtype=torch.long),
              "areas": torch.tensor(areas, dtype=torch.float64),
              "weights": torch.tensor(weights, dtype=torch.float64),
              "bg_indices": torch.tensor(bg_indices, dtype=torch.long),
              "bg_offsets": offsets, "background_pixels": int((~mask).sum())}
    return result


def fast_loss(base, delta, mask, geometries, loss_mode, s):
    """P2 losses with the same reductions; no per-component CUDA scalar reads."""
    delta = delta.float()
    prob = corrected_probability(base, delta)
    logits = reference_logits(base)+delta.double()
    bce = F.binary_cross_entropy_with_logits(logits, mask.double(), reduction="none").flatten(1).mean(1)
    p, gt = prob.flatten(1), mask.flatten(1)
    dice = (1-(2*(p*gt).sum(1)+1)/(p.sum(1)+gt.sum(1)+1))*(gt.sum(1)>0)
    counts = [g["background_pixels"] for g in geometries]
    ks = [max(1, int(np.ceil(n*s["hard_fraction"]))) if n else 0 for n in counts]
    if max(ks):
        increase = torch.where(mask.flatten(1) == 0, (prob-base).flatten(1).relu(), -torch.inf)
        top = increase.topk(max(ks), dim=1).values
        k = torch.tensor(ks, device=base.device)
        take = torch.arange(max(ks), device=base.device)[None] < k[:, None]
        bg = torch.where(take, top, 0).sum(1)/k.clamp_min(1)
    else:
        bg = delta.flatten(1).sum(1)*0
    magnitude = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none").flatten(1).mean(1)
    loss = bce+dice+s["bg_weight"]*bg+s["residual_weight"]*magnitude
    terms = dict(bce=bce, dice=dice, background=bg, magnitude=magnitude)
    components, rankings = [], []
    for i, g in enumerate(geometries):
        zero = delta[i].sum()*0
        if loss_mode == "p2a" or not len(g["areas"]):
            components.append(zero)
            rankings.append(zero)
            continue
        lgt = logits[i, 0].flatten()
        positive = lgt.index_select(0, g["gt_indices"])
        num = len(g["areas"])
        sums = torch.zeros(num, dtype=torch.float64, device=lgt.device)
        positive_mean = sums.scatter_add(0, g["component_ids"], positive)/g["areas"]
        per_component = sums.scatter_add(0, g["component_ids"], F.softplus(-positive))/g["areas"]
        components.append((per_component*g["weights"]).sum()/g["weights"].sum())
        local_terms, local_weights = [], []
        offsets = g["bg_offsets"]
        for j in range(num):
            count = offsets[j+1]-offsets[j]
            if not count:
                continue
            background = lgt.index_select(0, g["bg_indices"][offsets[j]:offsets[j+1]])
            hard = background.topk(min(s["rank_pixels"], count)).values.mean()
            local_terms.append(F.softplus(hard+s["rank_margin"]-positive_mean[j]))
            local_weights.append(g["weights"][j])
        rankings.append((torch.stack(local_terms)*torch.stack(local_weights)).sum()/torch.stack(local_weights).sum()
                        if local_terms else zero)
    terms.update(component=torch.stack(components), ranking=torch.stack(rankings))
    return loss+s["component_weight"]*terms["component"]+s["ranking_weight"]*terms["ranking"], terms


class ValidationPlan:
    """Same draws/8-connected GT/strict thresholds as P1, constructed once."""
    def __init__(self, masks, seed, pixels=8192, fpr=.01):
        self.fpr = fpr
        rng = np.random.default_rng(seed)
        self.sample_indices, self.gt_indices, labels, component_ids, areas = [], [], [], [], []
        self.small = []
        for gt in np.asarray(masks, bool):
            draw = rng.choice(gt.size, min(gt.size, pixels), replace=False)
            self.sample_indices.append(draw)
            labels.extend(gt.ravel()[draw])
            positions = np.flatnonzero(gt)
            self.gt_indices.append(positions)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
            component_ids.extend(cc.ravel()[positions]-1+len(areas))
            for label in range(1, n):
                area = int(stats[label, cv2.CC_STAT_AREA])
                areas.append(area)
                self.small.append(area <= gt.size*.001)
        self.labels = np.asarray(labels, bool)
        self.component_ids = np.asarray(component_ids, np.int64)
        self.areas, self.small = np.asarray(areas), np.asarray(self.small, bool)
        if not self.labels.any() or self.labels.all():
            raise ValueError("Source validation sample has no positives/background; increase val_pixels")

    def gather(self, maps):
        maps = np.asarray(maps)
        sample = np.concatenate([v.ravel()[indices] for v, indices in zip(maps, self.sample_indices)])
        gt = np.concatenate([v.ravel()[indices] for v, indices in zip(maps, self.gt_indices)])
        return sample, gt

    def metrics(self, sample, gt, base_sample):
        from sklearn.metrics import roc_auc_score
        bg = ~self.labels
        threshold = operating_threshold(sample[bg], self.fpr)
        base_threshold = operating_threshold(base_sample[bg], self.fpr)
        recall = np.bincount(self.component_ids, weights=(gt > threshold).astype(float), minlength=len(self.areas))/self.areas
        return dict(pixel_auc_sampled=float(roc_auc_score(self.labels, sample)),
                    region_recall_at_fpr=float(recall.mean()) if len(recall) else None,
                    small_hit_at_fpr=float(np.mean(recall[self.small] >= .1)) if self.small.any() else None,
                    small_components=int(self.small.sum()), threshold=threshold, base_threshold=base_threshold,
                    realized_fpr=float(np.mean(sample[bg] > threshold)),
                    fpr_at_base_threshold=float(np.mean(sample[bg] > base_threshold)),
                    base_fpr=float(np.mean(base_sample[bg] > base_threshold)))


def batch_plan_indices(plan, start, end, image_pixels):
    sample = np.concatenate([indices+(i-start)*image_pixels for i, indices in enumerate(plan.sample_indices[start:end], start)])
    gt = np.concatenate([indices+(i-start)*image_pixels for i, indices in enumerate(plan.gt_indices[start:end], start)])
    return torch.from_numpy(sample.astype(np.int64)), torch.from_numpy(gt.astype(np.int64))
