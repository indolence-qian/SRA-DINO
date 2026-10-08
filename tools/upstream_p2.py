"""Explicit local patch/text matching and component-balanced source supervision."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.upstream_repair import constrained_loss, reference_logits


def spatial_matching(visual, embeddings, valid, boxes):
    """Cosines in the frozen adapter/CLIP space, BEFORE the hidden projection.

    Return observation, expectation and paired difference for each visual layer,
    plus two coverage channels. No labels, Base scores or verdict pseudo-labels.
    Overlapping tiles are averaged; unknown roles contribute neither bias nor
    difference. Matching is confined to feature cells touched by each tile.
    """
    b, layers, dim, h, w = visual.shape
    if (embeddings.shape[0] != b or embeddings.shape[2:] != (2, dim) or
            valid.shape != embeddings.shape[:3] or boxes.shape != (*valid.shape[:2], 4)):
        raise ValueError("Misaligned patch/text features")
    xs = torch.arange(w, device=visual.device)[None, None, None, :]
    ys = torch.arange(h, device=visual.device)[None, None, :, None]
    support = ((xs >= (boxes[..., 0]*w).floor()[..., None, None]) &
               (xs < (boxes[..., 2]*w).ceil()[..., None, None]) &
               (ys >= (boxes[..., 1]*h).floor()[..., None, None]) &
               (ys < (boxes[..., 3]*h).ceil()[..., None, None]))
    cosines = torch.einsum("bldhw,btkd->bltkhw", F.normalize(visual.float(), dim=2),
                          F.normalize(embeddings.float(), dim=-1))
    weights = support[:, :, None].float() * (valid > 0)[..., None, None]
    counts = weights.sum(1)
    role_maps = (cosines * weights[:, None]).sum(2) / counts.clamp_min(1)[:, None]
    paired = support.float() * (valid > 0).all(2)[..., None, None]
    difference = ((cosines[:, :, :, 0]-cosines[:, :, :, 1])*paired[:, None]).sum(2)
    difference = difference / paired.sum(1).clamp_min(1)[:, None]
    coverage = (counts > 0).float()
    return torch.cat([role_maps[:, :, 0], role_maps[:, :, 1], difference, coverage], 1)


class MatchingHead(nn.Module):
    def __init__(self, layers, dim, hidden=96):
        super().__init__()
        self.layers, self.dim = layers, dim
        self.visual = nn.Sequential(nn.Conv2d(layers*dim, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU())
        self.semantic = nn.Sequential(nn.Conv2d(3*layers+2, 32, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(hidden+32+2+1, hidden, 3, padding=1),
                                  nn.GroupNorm(8, hidden), nn.GELU())
        self.decode = nn.Sequential(nn.Conv2d(hidden, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(self, visual, base, embeddings, valid, boxes, use_semantics=True):
        b, layers, dim, h, w = visual.shape
        if (layers, dim) != (self.layers, self.dim):
            raise ValueError("Feature/head shape mismatch")
        if not use_semantics:
            valid = torch.zeros_like(valid)
        match = spatial_matching(visual, embeddings, valid, boxes)
        coverage = match[:, -2:]
        text = self.semantic(match) * coverage.amax(1, keepdim=True)
        vis = self.visual(visual.reshape(b, layers*dim, h, w))
        coarse = F.interpolate(base, (h, w), mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([vis, text, coverage, coarse], 1))
        size = (min(base.shape[-2], h*4), min(base.shape[-1], w*4))
        x = F.interpolate(x, size, mode="bilinear", align_corners=False)
        return F.interpolate(self.decode(x), base.shape[-2:], mode="bilinear", align_corners=False)


def component_supervision(mask, min_area=2):
    """CPU source-mask preprocessing; preserve retained component identities.

    Rings can overlap: store per-component bounding boxes rather than a single
    ring-label map that could silently overwrite a smaller component's ring.
    """
    mask = np.asarray(mask, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(labels)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == label] = label
    return torch.from_numpy(kept.astype(np.int64))


def component_loss(base, delta, mask, labels, component_weight=.1, ranking_weight=.1,
                   small_fraction=.001, small_weight=2., near_radius=8, rank_pixels=64,
                   rank_margin=1., **p1):
    """P1 + equal-component BCE and local hard-background ranking, per image.

    Ranking uses logits to preserve gradients for very low Base probabilities.
    Small regions receive a bounded multiplier; normal images have zero extras.
    The original global BCE/Dice/background/magnitude losses remain in force.
    """
    loss, terms = constrained_loss(base, delta, mask, **p1)
    logits = reference_logits(base)+delta.double()
    cb, rb = [], []
    for i in range(len(base)):
        lgt, lab, gt = logits[i, 0], labels[i], mask[i, 0] > 0
        positive_terms, ranking_terms, weights, rank_weights = [], [], [], []
        for index in torch.unique(lab).tolist():
            if index == 0:
                continue
            region = lab == index
            area = int(region.sum())
            weight = small_weight if area <= lab.numel()*small_fraction else 1.
            positive_terms.append(F.softplus(-lgt[region]).mean())
            weights.append(weight)
            # A bounding rectangle around this component bounds memory and CPU
            # preprocessing; exclude other GT pixels from the near-background.
            yy, xx = torch.where(region)
            y0, y1 = max(0, int(yy.min())-near_radius), min(lab.shape[0], int(yy.max())+near_radius+1)
            x0, x1 = max(0, int(xx.min())-near_radius), min(lab.shape[1], int(xx.max())+near_radius+1)
            local = region[y0:y1, x0:x1].float()[None, None]
            ring = F.max_pool2d(local, 2*near_radius+1, stride=1, padding=near_radius)[0, 0].bool()
            background = lgt[y0:y1, x0:x1][ring & ~gt[y0:y1, x0:x1]]
            if background.numel():
                hard = background.topk(min(rank_pixels, background.numel())).values
                ranking_terms.append(F.softplus(hard.mean()+rank_margin-lgt[region].mean()))
                rank_weights.append(weight)
        zero = delta[i].sum()*0
        cb.append(sum(t*w for t, w in zip(positive_terms, weights))/sum(weights) if weights else zero)
        rb.append(sum(t*w for t, w in zip(ranking_terms, rank_weights))/sum(rank_weights) if rank_weights else zero)
    terms.update(component=torch.stack(cb), ranking=torch.stack(rb))
    return loss+component_weight*terms["component"]+ranking_weight*terms["ranking"], terms
