"""Global/local normal-reference evidence for a frozen CLIP+DINO detector.

No dataset labels or masks are accepted by the extractor or correspondence code.
The random projection is a fixed cache compression, NOT pretrained text alignment.
Only EvidenceHead learns an interface to the frozen normal/defect text anchors.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("A_global", "B_highres", "C_reference", "D_highres_reference")


def arm_options(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown experimental arm: {arm}")
    return arm in (ARMS[1], ARMS[3]), arm in (ARMS[2], ARMS[3])


def covering_boxes(width, height, grid=2, fraction=.6):
    """Pixel-coordinate crops of the ORIGINAL image, including every border."""
    if min(width, height) < 1 or grid < 2 or not 1 / grid <= fraction < 1:
        raise ValueError("Require positive image size, grid>=2 and 1/grid<=fraction<1")
    bw, bh = max(1, math.ceil(width * fraction)), max(1, math.ceil(height * fraction))
    xs = sorted(set(round(x) for x in np.linspace(0, width - bw, grid)))
    ys = sorted(set(round(y) for y in np.linspace(0, height - bh, grid)))
    return [(x, y, x + bw, y + bh) for y in ys for x in xs]


def patch_coordinates(side, box=(0., 0., 1., 1.), device=None):
    axis = (torch.arange(side, device=device, dtype=torch.float32) + .5) / side
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x1, y1, x2, y2 = box
    return torch.stack([x1 + xx * (x2 - x1), y1 + yy * (y2 - y1)], -1).reshape(-1, 2)


def normalized_box(box, width, height):
    x1, y1, x2, y2 = box
    return x1 / width, y1 / height, x2 / width, y2 / height


def stitch_tiles(values: Sequence[torch.Tensor], boxes, size):
    """Blend [C,h,w] tile maps onto one grid using positive Hann tapers.

    Pixel-center sampling respects fractional crop boundaries. The boundary floor
    guarantees complete coverage, including the original image's outside edges.
    """
    if not values or len(values) != len(boxes) or size < 1:
        raise ValueError("Need aligned tiles/boxes and positive output size")
    first = values[0]
    total = first.new_zeros(first.shape[0], size, size, dtype=torch.float32)
    weight = total[:1].clone()
    axis = (torch.arange(size, device=first.device, dtype=torch.float32) + .5) / size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    for value, (x1, y1, x2, y2) in zip(values, boxes):
        if value.ndim != 3 or value.shape[0] != total.shape[0] or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Invalid tile shape/coordinates")
        gx, gy = (xx - x1) / (x2 - x1), (yy - y1) / (y2 - y1)
        # A cell intersecting the crop is covered; clamp sampling at crop edges.
        support = ((xx + .5 / size > x1) & (xx - .5 / size < x2) &
                   (yy + .5 / size > y1) & (yy - .5 / size < y2))
        window = (.05 + .95 * torch.sin(math.pi * gx.clamp(0, 1)).square()) * (
            .05 + .95 * torch.sin(math.pi * gy.clamp(0, 1)).square()) * support
        grid = torch.stack([2 * gx - 1, 2 * gy - 1], -1)[None]
        sampled = F.grid_sample(value.float()[None], grid, mode="bilinear", padding_mode="border",
                                align_corners=False)[0]
        total += sampled * window
        weight += window
    if bool((weight <= 0).any()):
        raise ValueError("Tile grid has uncovered output cells")
    return total / weight


def fixed_projection(input_dim, output_dim, seed=42):
    if not 1 <= output_dim <= input_dim:
        raise ValueError("Compression dimension must be in [1,input_dim]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced")[0] * math.sqrt(input_dim / output_dim)


def match_reference(query, reference, query_xy, reference_xy, topk=3, temperature=.07,
                    spatial_weight=.15, radius=.35, chunk_size=128):
    """Return signed residual, L2 distance and top-k ambiguity in raw DINO space.

    Inputs are [L,N,C] L2-normalized tokens; only neighbor indices/weights are
    chosen with a spatial penalty. Appearance distance excludes that penalty.
    Similarity matrices are chunked rather than allocating Nquery*Nbank*L.
    """
    if query.ndim != 3 or reference.ndim != 3 or query.shape[::2] != reference.shape[::2]:
        raise ValueError("Expected matching [layers,patches,channels] tensors")
    if (query_xy.shape != (query.shape[1], 2) or reference_xy.shape != (reference.shape[1], 2)
            or reference.shape[1] < topk or topk < 1 or chunk_size < 1 or temperature <= 0
            or spatial_weight < 0 or radius < 0):
        raise ValueError("Invalid correspondence geometry/budget")
    if not all(bool(torch.isfinite(x).all()) for x in (query, reference, query_xy, reference_xy)):
        raise ValueError("Nonfinite matching evidence")
    residuals, distances, entropies = [], [], []
    for layer in range(query.shape[0]):
        residual, distance, entropy = [], [], []
        bank = reference[layer].float()
        for start in range(0, query.shape[1], chunk_size):
            q = query[layer, start:start + chunk_size].float()
            xy = query_xy[start:start + chunk_size]
            displacement = (xy[:, None] - reference_xy[None]).square().sum(-1)
            score = q @ bank.T - spatial_weight * displacement
            if radius:
                allowed = displacement <= radius * radius
                # An impossible locality constraint must be explicit, not global fallback.
                if bool((allowed.sum(1) < topk).any()):
                    raise ValueError("Too few reference patches inside MATCH_RADIUS; enlarge radius or use 0")
                score = score.masked_fill(~allowed, -torch.inf)
            values, indices = score.topk(topk, dim=1)
            weights = (values / temperature).softmax(1)
            matched = (bank[indices] * weights[..., None]).sum(1)
            delta = q - matched
            residual.append(delta)
            distance.append(delta.square().sum(-1).sqrt())
            entropy.append(-(weights * weights.clamp_min(1e-12).log()).sum(1) / math.log(topk)
                           if topk > 1 else weights[:, 0] * 0)
        residuals.append(torch.cat(residual))
        distances.append(torch.cat(distance))
        entropies.append(torch.cat(entropy))
    return torch.stack(residuals), torch.stack(distances), torch.stack(entropies)


class FrozenExtractor:
    """Exact historical Base readout plus pre-standardization correspondence tokens."""
    def __init__(self, cfg, device):
        from dual_vlm import load_dual
        from utils import get_text_features_with_prompt_learner
        self.cfg, self.device = cfg, device
        self.clip, prompt, self.dino, self.adapter, self.layers = load_dual(cfg, device)
        with torch.inference_mode():
            text = get_text_features_with_prompt_learner(self.clip, prompt, device)
            self.text = torch.stack([self.adapter.prompt_adapter[i](text[:, i][None])[0][0]
                                     for i in range(2)])
        self.project = fixed_projection(1024, cfg["feature_dim"], cfg["seed"]).to(device)
        self.captured, self.handles = {}, []
        anchor = getattr(self.dino, "norm", None) or getattr(self.dino, "fc_norm", None)
        registers = int(getattr(self.dino, "n_storage_tokens", 4))
        for layer in self.layers:
            def capture(_module, _inputs, output, layer=layer):
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.captured[layer] = anchor(value)[:, 1 + registers:].float()
            self.handles.append(self.dino.blocks[layer].register_forward_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.inference_mode()
    def view(self, image):
        size = self.cfg["image_size"]
        rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).copy()
        x = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).float() / 255
        mean, std = x.new_tensor([.485, .456, .406]), x.new_tensor([.229, .224, .225])
        self.captured.clear()
        self.dino(((x - mean[:, None, None]) / std[:, None, None])[None])
        if set(self.captured) != set(self.layers):
            raise ValueError("DINO hooks did not capture every checkpoint layer")
        raw, layer_prob, full_prob = [], [], []
        side = size // 16
        for index, layer in enumerate(self.layers):
            patches = self.captured[layer]
            if patches.shape != (1, side * side, 1024):
                raise ValueError(f"Unexpected DINO patch shape: {tuple(patches.shape)}")
            raw.append(F.normalize(patches[0], dim=-1))
            # Preserve the existing detector's per-image standardization ONLY here.
            standardized = (patches - patches.mean(1, keepdim=True)) / (patches.std(1, keepdim=True) + 1e-6)
            adapted = F.normalize(self.adapter.patch_token_adapter[index](standardized)[0], dim=-1)
            logits = (100 * adapted @ self.text.T).transpose(1, 2).reshape(1, 2, side, side)
            layer_prob.append(logits.softmax(1)[0, 1])
            full_prob.append(F.interpolate(logits, (size, size), mode="bilinear",
                                           align_corners=True).softmax(1)[0, 1])
        return dict(raw=torch.stack(raw), prior=torch.stack(layer_prob),
                    base=torch.stack(full_prob).mean(0)[None])

    @torch.inference_mode()
    def image_views(self, image):
        boxes = covering_boxes(image.width, image.height, self.cfg["tile_grid"], self.cfg["tile_fraction"])
        global_view = self.view(image)
        tiles = [self.view(image.crop(box)) for box in boxes]
        return global_view, tiles, [normalized_box(b, image.width, image.height) for b in boxes]


def pack_view(view, reference, query_xy, reference_xy, project, cfg):
    raw = view["raw"]
    delta, distance, uncertainty = match_reference(
        raw, reference, query_xy, reference_xy, cfg["match_topk"], cfg["match_temperature"],
        cfg["spatial_weight"], cfg["match_radius"], cfg["match_chunk"])
    side = int(math.isqrt(raw.shape[1]))
    def spatial(tokens):
        return tokens.permute(0, 2, 1).reshape(raw.shape[0], -1, side, side)
    return dict(query=spatial(raw @ project), residual=spatial(delta @ project),
                distance=distance.reshape(-1, side, side), uncertainty=uncertainty.reshape(-1, side, side),
                prior=view["prior"], base=view["base"])


def assemble_tiles(tiles, boxes, grid_size, image_size):
    result = {}
    for key in tiles[0]:
        shape = tiles[0][key].shape
        values = [tile[key].reshape(-1, *tile[key].shape[-2:]) for tile in tiles]
        size = image_size if key == "base" else grid_size
        result[key] = stitch_tiles(values, boxes, size).reshape(*shape[:-2], size, size)
    return result


def arm_inputs(arrays, arm, grid_size, beta):
    """Same head shape and fusion weight for every arm; absent reference is explicit."""
    high, reference = arm_options(arm)
    result = {}
    for key in ("query", "residual", "distance", "uncertainty", "prior", "base"):
        value = arrays["global_" + key].float()
        if key != "base":
            shape = value.shape
            value = F.interpolate(value.reshape(shape[0], -1, *shape[-2:]), (grid_size, grid_size),
                                  mode="bilinear", align_corners=False).reshape(*shape[:-2], grid_size, grid_size)
        if high:
            value = (1 - beta) * value + beta * arrays["high_" + key].float()
        if key in ("residual", "distance", "uncertainty") and not reference:
            value = torch.zeros_like(value)
        result[key] = value
    result["reference_valid"] = torch.full_like(result["prior"][:, :1], float(reference))
    return result


class EvidenceHead(nn.Module):
    """Learn evidence-to-text alignment; no native alignment of differences assumed."""
    def __init__(self, layers, feature_dim, anchors, hidden=64, temperature=20.):
        super().__init__()
        if hidden % 8 or anchors.ndim != 2 or anchors.shape[0] != 2:
            raise ValueError("Need hidden divisible by 8 and two text anchors")
        self.layers, self.feature_dim, self.temperature = layers, feature_dim, temperature
        self.register_buffer("anchors", F.normalize(anchors.float(), dim=-1))
        self.visual = nn.Conv2d(layers * feature_dim, hidden, 1)
        self.reference = nn.Conv2d(layers * (3 * feature_dim + 2), hidden, 1, bias=False)
        self.fuse = nn.Sequential(nn.Conv2d(2 * hidden + layers + 4, hidden, 3, padding=1),
                                  nn.GroupNorm(8, hidden), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.to_text = nn.Conv2d(hidden, anchors.shape[1], 1)
        self.delta = nn.Conv2d(hidden + 1, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, inputs):
        q, residual = inputs["query"], inputs["residual"]
        b, layers, dim, h, w = q.shape
        if (layers, dim) != (self.layers, self.feature_dim):
            raise ValueError("Cache/head feature dimensions differ")
        valid = inputs["reference_valid"]
        matched = (q - residual) * valid[:, None]
        evidence = torch.cat([matched, residual, residual.abs(), inputs["distance"][:, :, None],
                              inputs["uncertainty"][:, :, None]], 2).reshape(b, -1, h, w)
        visual = self.visual(q.reshape(b, -1, h, w))
        reference = self.reference(evidence) * valid
        base = F.interpolate(inputs["base"], (h, w), mode="bilinear", align_corners=False)
        xy = patch_coordinates(h, device=q.device).T.reshape(1, 2, h, w).expand(b, -1, -1, -1)
        hidden = self.fuse(torch.cat([visual, reference, inputs["prior"], base, valid, xy], 1))
        aligned = F.normalize(self.to_text(hidden).float(), dim=1)
        logits = self.temperature * torch.einsum("bdhw,kd->bkhw", aligned, self.anchors)
        margin = logits[:, 1:2] - logits[:, :1]
        delta = self.delta(torch.cat([hidden, margin.to(hidden.dtype)], 1)).float()
        size = inputs["base"].shape[-2:]
        return dict(delta=F.interpolate(delta, size, mode="bilinear", align_corners=False),
                    semantic_logits=F.interpolate(margin, size, mode="bilinear", align_corners=False))


def supervision_loss(base, output, mask, semantic_weight=.1, background_weight=1., residual_weight=.01):
    from tools.upstream_repair import constrained_loss
    loss, _ = constrained_loss(base, output["delta"], mask, background_weight, residual_weight)
    semantic = F.binary_cross_entropy_with_logits(output["semantic_logits"], mask, reduction="none").flatten(1).mean(1)
    return loss + semantic_weight * semantic
