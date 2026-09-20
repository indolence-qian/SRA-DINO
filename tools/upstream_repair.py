"""P0 identity-preserving arithmetic and P1 source-supervised constraints.

Kept separate from v1: changing v1 code would invalidate its sealed caches.
"""
import numpy as np
import torch
from torch.nn import functional as F

from tools.upstream_localization import LocalSemanticHead, spatial_semantics


def reference_probability(base):
    # Only EXACT endpoints need a finite surrogate for optimization. Interior
    # probabilities, including subnormal float32 values, are never floored.
    p = base.double()
    return torch.where(p == 0, 1e-6, torch.where(p == 1, 1 - 1e-6, p))


def reference_logits(base):
    p = reference_probability(base)
    return p.log() - torch.log1p(-p)


def corrected_probability(base, delta, alpha=1.):
    """Stable odds residual; delta=0 and alpha=0 return Base bit-for-bit.

    Compute the CHANGE using expm1, not sigmoid(logit(p)+d)-p (cancellation).
    Exact 0/1 use finite endpoint surrogates for the change only, with the
    surrogate's initial offset subtracted implicitly. Thus they CAN be corrected
    without shifting any scores at identity. Clamp only the final [0,1] bounds.
    """
    if alpha == 0:
        return base
    p = reference_probability(base)
    d = delta.double() * alpha
    positive = d >= 0
    # torch.where instead of abs: retains a nonzero derivative at d=0.
    neg_magnitude = torch.where(positive, -d, d)
    e = neg_magnitude.exp()
    change_factor = -torch.expm1(neg_magnitude)
    denominator = torch.where(positive, p + (1-p)*e, (1-p) + p*e)
    change = p * (1-p) * change_factor / denominator
    change = torch.where(positive, change, -change)
    return (base.double() + change).clamp(0, 1).to(base.dtype)


class ConstrainedHead(LocalSemanticHead):
    """Same v1 architecture/semantics. Only returns its raw residual instead."""
    def forward(self, visual, base, embeddings, valid, boxes, use_semantics=True):
        b, layers, dim, h, w = visual.shape
        if (layers, dim) != (self.layers, self.dim):
            raise ValueError("Feature/head shape mismatch")
        if not use_semantics:
            valid = torch.zeros_like(valid)
        sem, coverage = spatial_semantics(embeddings, valid, boxes, h, w)
        vis = self.visual(visual.reshape(b, layers*dim, h, w))
        text = self.semantic(sem) * coverage[:, :1]
        coarse = F.interpolate(base, (h, w), mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([vis, text, coverage, coarse], 1))
        size = (min(base.shape[-2], h*4), min(base.shape[-1], w*4))
        x = F.interpolate(x, size, mode="bilinear", align_corners=False)
        return F.interpolate(self.decode(x), base.shape[-2:], mode="bilinear", align_corners=False)


def constrained_loss(base, delta, mask, bg_weight=1., residual_weight=.01, hard_fraction=.01):
    prob = corrected_probability(base, delta)
    # BCE is evaluated in log space, NOT on floored output probabilities.
    # At exact cached endpoints this is the documented finite surrogate loss.
    logits = reference_logits(base) + delta.double()
    bce = F.binary_cross_entropy_with_logits(logits, mask.double(), reduction="none").flatten(1).mean(1)
    p, gt = prob.flatten(1), mask.flatten(1)
    dice = (1-(2*(p*gt).sum(1)+1)/(p.sum(1)+gt.sum(1)+1)) * (gt.sum(1)>0)
    bg_terms = []
    for i in range(len(base)):
        increase = (prob[i]-base[i]).relu()[mask[i] == 0]
        k = max(1, int(np.ceil(increase.numel()*hard_fraction)))
        bg_terms.append(increase.topk(k).values.mean() if increase.numel() else delta[i].sum()*0)
    bg = torch.stack(bg_terms)
    magnitude = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none").flatten(1).mean(1)
    loss = bce + dice + bg_weight*bg + residual_weight*magnitude
    return loss, dict(bce=bce, dice=dice, background=bg, magnitude=magnitude)


def operating_threshold(background, budget):
    """Conservative strict-> threshold; ties can make realized FPR LOWER."""
    return float(np.quantile(background, 1-budget, method="higher"))


def source_metrics(masks, maps, base_maps, seed, sample_pixels=8192, fpr=.01):
    """Bounded validation cost: sampled pixel AUC/background; full GT regions.

    Identical label-independent pixel draws for every alpha/epoch in a category.
    No test data or target-derived threshold enters selection.
    """
    import cv2
    from sklearn.metrics import roc_auc_score
    masks, maps, base_maps = np.asarray(masks, bool), np.asarray(maps), np.asarray(base_maps)
    rng = np.random.default_rng(seed)
    labels, scores, old_scores = [], [], []
    for gt, pred, old in zip(masks, maps, base_maps):
        indices = rng.choice(gt.size, min(gt.size, sample_pixels), replace=False)
        labels.extend(gt.ravel()[indices])
        scores.extend(pred.ravel()[indices])
        old_scores.extend(old.ravel()[indices])
    labels, scores, old_scores = np.asarray(labels), np.asarray(scores), np.asarray(old_scores)
    bg = ~labels
    if not bg.any() or not labels.any():
        raise ValueError("Source validation pixel sample has no positives/background; increase --val_pixels")
    threshold = operating_threshold(scores[bg], fpr)
    base_threshold = operating_threshold(old_scores[bg], fpr)
    regions, small = [], []
    for gt, pred in zip(masks, maps):
        n, cc = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        for index in range(1, n):
            region = cc == index
            recall = float(np.mean(pred[region] > threshold))
            regions.append(recall)
            if region.sum() <= gt.size*.001:
                small.append(float(recall >= .1))
    return dict(pixel_auc_sampled=float(roc_auc_score(labels, scores)),
                region_recall_at_fpr=float(np.mean(regions)) if regions else None,
                small_hit_at_fpr=float(np.mean(small)) if small else None,
                small_components=len(small), threshold=threshold, base_threshold=base_threshold,
                realized_fpr=float(np.mean(scores[bg] > threshold)),
                fpr_at_base_threshold=float(np.mean(scores[bg] > base_threshold)),
                base_fpr=float(np.mean(old_scores[bg] > base_threshold)))


def aggregate_source(categories):
    result = {}
    for key in ("pixel_auc_sampled", "region_recall_at_fpr", "small_hit_at_fpr", "fpr_at_base_threshold", "base_fpr"):
        values = [r[key] for r in categories if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def selection_score(candidate, baseline, categories, base_categories, auc_tolerance=.002, fpr_slack=.002):
    """No degradation in matched-FPR regional/small recall; per-category FPR guard.

    This is a SOURCE validation rule, not a promise of target-domain safety.
    Return None to keep the explicit Base fallback.
    """
    if candidate["pixel_auc_sampled"] < baseline["pixel_auc_sampled"]-auc_tolerance:
        return None
    if any(c["fpr_at_base_threshold"] > b["base_fpr"] + fpr_slack for c, b in zip(categories, base_categories)):
        return None
    for key in ("region_recall_at_fpr", "small_hit_at_fpr"):
        if baseline[key] is not None and (candidate[key] is None or candidate[key] < baseline[key]-1e-12):
            return None
    score = candidate["region_recall_at_fpr"]
    if candidate["small_hit_at_fpr"] is not None:
        score = .5*(score + candidate["small_hit_at_fpr"])
    return score
