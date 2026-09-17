"""Full-coverage semantic conditioning; no Base candidate gate or MARA actions."""
import json
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROTOCOL = "upstream_local_semantics_v1"


def tile_boxes(grid=3, fraction=.4):
    if grid < 2 or not 1 / grid <= fraction < 1:
        raise ValueError("Tiles must cover the entire image: grid>=2, 1/grid<=fraction<1")
    starts = np.linspace(0, 1 - fraction, grid)
    return [[float(x), float(y), float(x + fraction), float(y + fraction)]
            for y in starts for x in starts]


def tile_prompt(category, tile_id):
    # No image paths, labels, Base heatmaps, candidate selection or GT enter this prompt.
    return (
        f"Inspect industrial {category.replace('_', ' ')}. Image 1 is the entire query for context. "
        f"Image 2 is native-resolution local tile {tile_id} from the SAME query, not a normal reference. "
        "Describe Image 2 only. Look closely for tiny cracks, scratches, missing material, contamination, "
        "broken continuity and unusual texture. A defect may occupy only a few pixels; do not assume "
        "the tile is normal because most of it is intact. Distinguish reflections, regular edges and "
        "repeating structures from physical damage. Do not invent an unseen normal reference. "
        "Normal expectation is a generic material/structure expectation, not an observed reference. "
        "If blurred or ambiguous, use visibility=insufficient or status=uncertain. "
        "Your description is contextual evidence, not a segmentation mask or a ground-truth label. "
        "Return exactly one JSON object, no markdown: "
        f'{{"tile_id":{tile_id},"visibility":"sufficient|insufficient",'
        '"status":"possible_defect|apparently_normal|uncertain",'
        '"observation":"At most 30 English words describing visible local material and irregularities",'
        '"normal_expectation":"At most 25 English words about expected intact local structure"}. '
        "For insufficient visibility, descriptions may be empty; never guess."
    )


def parse_tile(raw, tile_id):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate key")
            result[k] = v
        return result
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        d = json.loads(text, object_pairs_hook=unique)
        if not isinstance(d, dict) or set(d) != {
            "tile_id", "visibility", "status", "observation", "normal_expectation"
        }:
            raise ValueError("schema")
        if type(d["tile_id"]) is not int or d["tile_id"] != tile_id:
            raise ValueError("tile_id")
        if d["visibility"] not in ("sufficient", "insufficient"):
            raise ValueError("visibility")
        if d["status"] not in ("possible_defect", "apparently_normal", "uncertain"):
            raise ValueError("status")
        for k in ("observation", "normal_expectation"):
            if not isinstance(d[k], str) or len(d[k]) > 1500:
                raise ValueError(k)
            d[k] = d[k].strip()
        if d["visibility"] == "sufficient" and not d["observation"]:
            raise ValueError("missing visible observation")
        # Abstention is VALID, not a normal label and not a parser error.
        d.update(parse_ok=True, semantic_valid=d["visibility"] == "sufficient" and bool(d["observation"]))
        return d
    except (ValueError, TypeError, AttributeError) as exc:
        return dict(tile_id=tile_id, parse_ok=False, semantic_valid=False, error=str(exc))


def spatial_semantics(embeddings, valid, boxes, height, width):
    """[B,T,2,D] -> [B,2D,H,W], with separate observation/expectation coverage."""
    b, t, kinds, dim = embeddings.shape
    if kinds != 2 or valid.shape != (b, t, 2) or boxes.shape != (b, t, 4):
        raise ValueError("Misaligned semantic tiles")
    # Include every feature cell touched by a tile; avoids quantization gaps at boundaries.
    xs = torch.arange(width, device=boxes.device)[None, None, None, :]
    ys = torch.arange(height, device=boxes.device)[None, None, :, None]
    support = ((xs >= (boxes[..., 0] * width).floor()[..., None, None]) &
               (xs < (boxes[..., 2] * width).ceil()[..., None, None]) &
               (ys >= (boxes[..., 1] * height).floor()[..., None, None]) &
               (ys < (boxes[..., 3] * height).ceil()[..., None, None]))
    weights = support[:, :, None].to(embeddings.dtype) * valid[..., None, None]
    counts = weights.sum(1)
    value = torch.einsum("btkd,btkhw->bkdhw", embeddings, weights)
    value = value / counts.clamp_min(1)[:, :, None]
    return value.reshape(b, 2 * dim, height, width), (counts > 0).to(value.dtype)


class LocalSemanticHead(nn.Module):
    """Frozen multi-layer DINO/CLIP features + local text -> dense logit residual.

    No thresholded support: the trainable head may recover Base misses anywhere.
    Zero final initialization reproduces Base (apart from probability clamping).
    """
    def __init__(self, layers, dim, hidden=96):
        super().__init__()
        self.layers, self.dim = layers, dim
        self.visual = nn.Sequential(nn.Conv2d(layers * dim, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU())
        self.semantic = nn.Sequential(nn.Conv2d(2 * dim, 32, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(hidden + 32 + 2 + 1, hidden, 3, padding=1),
                                  nn.GroupNorm(8, hidden), nn.GELU())
        self.decode = nn.Sequential(nn.Conv2d(hidden, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(self, visual, base, embeddings, valid, boxes, use_semantics=True):
        b, layers, dim, h, w = visual.shape
        if (layers, dim) != (self.layers, self.dim):
            raise ValueError("Feature cache/head dimensions differ")
        if not use_semantics:
            valid = torch.zeros_like(valid)
        sem, coverage = spatial_semantics(embeddings, valid, boxes, h, w)
        vis = self.visual(visual.reshape(b, layers * dim, h, w))
        text = self.semantic(sem) * coverage[:, :1]  # unknown has no semantic bias
        coarse = F.interpolate(base, (h, w), mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([vis, text, coverage, coarse], dim=1))
        size = (min(base.shape[-2], h * 4), min(base.shape[-1], w * 4))
        x = F.interpolate(x, size, mode="bilinear", align_corners=False)
        delta = self.decode(x)
        delta = F.interpolate(delta, base.shape[-2:], mode="bilinear", align_corners=False)
        # No hard residual bound: even an extremely low Base score must be recoverable.
        # Stable BCE-with-logits and training gradient clipping handle numerics.
        return torch.logit(base.clamp(1e-5, 1 - 1e-5)) + delta


def segmentation_loss(logits, mask):
    """Per-image BCE + positive-image Dice; normal images are supervised by BCE."""
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none").flatten(1).mean(1)
    prob = logits.sigmoid().flatten(1)
    target = mask.flatten(1)
    dice = 1 - (2 * (prob * target).sum(1) + 1) / (prob.sum(1) + target.sum(1) + 1)
    return bce + dice * (target.sum(1) > 0).to(dice.dtype)


def source_partition(records, val_fraction, seed):
    """Stratify SOURCE only. Target labels never control selection or partitions."""
    from collections import defaultdict
    from tools.vlm_review import digest
    groups = defaultdict(list)
    for r in records:
        groups[(r["category"], bool(r["mask_path"]))].append(r)
    for rows in groups.values():
        if len(rows) < 2:
            raise ValueError("Need >=2 source examples in each category/normal-anomaly stratum")
        rows.sort(key=lambda r: digest([seed, r["id"]]))
        n = max(1, min(len(rows) - 1, round(len(rows) * val_fraction)))
        for i, r in enumerate(rows):
            r["partition"] = "val" if i < n else "train"
    return records
