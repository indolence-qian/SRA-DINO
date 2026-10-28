"""Residual-only detection and language interfaces (no images, labels or Base bypass).

This is a ResAD-inspired implementation, not an exact reproduction. The shallow
constraintor has a signed identity skip; affine coupling flows model its output.
Text adapters see pre-constraint residuals and frozen paired-text directions.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tools.reference_highres import covering_boxes, normalized_box, stitch_tiles


def nearest_residual(query, bank, query_chunk=128, bank_chunk=4096):
    """Exact cosine NN on normalized [L,N,C] tokens; subtract BEFORE compression.

    Ties keep the earliest bank index, including across bank chunks. No spatial
    restriction or query-dependent normal-bank adaptation is used.
    """
    if (query.ndim != 3 or bank.ndim != 3 or query.shape[0] != bank.shape[0]
            or query.shape[2] != bank.shape[2] or not bank.shape[1]
            or min(query_chunk, bank_chunk) < 1):
        raise ValueError("Expected matching nonempty [layers,tokens,channels] arrays")
    if not torch.isfinite(query).all() or not torch.isfinite(bank).all():
        raise ValueError("Nonfinite matching features")
    residuals, indices = [], []
    for q_layer, n_layer in zip(query.float(), bank.float()):
        layer_r, layer_i = [], []
        for start in range(0, len(q_layer), query_chunk):
            q = q_layer[start:start + query_chunk]
            best = q.new_full((len(q),), -torch.inf)
            index = torch.zeros(len(q), dtype=torch.long, device=q.device)
            for offset in range(0, len(n_layer), bank_chunk):
                score, local = (q @ n_layer[offset:offset + bank_chunk].T).max(1)
                improve = score > best
                index = torch.where(improve, local + offset, index)
                best = torch.maximum(best, score)
            layer_r.append(q - n_layer[index])
            layer_i.append(index)
        residuals.append(torch.cat(layer_r))
        indices.append(torch.cat(layer_i))
    residual = torch.stack(residuals)
    return residual, residual.norm(dim=-1), torch.stack(indices)


def image_geometry(width, height, fraction=.6):
    crops = covering_boxes(width, height, 2, fraction)
    if len(crops) != 4:
        raise ValueError("Need four distinct native crops; image is too small")
    return [(0, 0, width, height)] + crops


def coverage_maps(mask, pixel_boxes, grid):
    """Any small positive survives pooling; mixed cells are NEVER normal samples."""
    if mask.ndim != 2 or grid < 1:
        raise ValueError("Expected a 2D native-resolution mask")
    result = []
    for x1, y1, x2, y2 in pixel_boxes:
        crop = mask[y1:y2, x1:x2]
        if crop.numel() == 0:
            raise ValueError("Empty supervision crop")
        result.append(F.adaptive_avg_pool2d(crop.float()[None, None], (grid, grid))[0, 0])
    return torch.stack(result)


def fuse_views(maps, boxes, size, high=True, beta=.25):
    """[B,V,C,h,w] -> [B,C,size,size]; score fusion, never residual averaging."""
    if maps.ndim != 5 or maps.shape[1] not in (1, 5) or not 0 <= beta <= 1:
        raise ValueError("Expected one global and optionally four local score maps")
    global_map = F.interpolate(maps[:, 0].float(), (size, size), mode="bilinear", align_corners=False)
    if not high:
        return global_map
    if maps.shape[1] != 5 or tuple(boxes.shape[:2]) != tuple(maps.shape[:2]):
        raise ValueError("High resolution requires every local view and its coordinates")
    local = torch.stack([stitch_tiles(list(maps[b, 1:]), boxes[b, 1:].tolist(), size)
                         for b in range(len(maps))])
    return (1 - beta) * global_map + beta * local


class SignedConstraintor(nn.Module):
    """Shallow, same-dimensional, zero-preserving residual map; no target BN."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.down = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden, permutation):
        super().__init__()
        self.half = dim // 2
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.net = nn.Sequential(nn.Linear(self.half, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * (dim - self.half)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def parameters_for(self, first):
        scale, shift = self.net(first).chunk(2, -1)
        return 1.5 * torch.tanh(scale / 1.5), shift

    def forward(self, x):
        x = x[..., self.permutation]
        first, second = x[..., :self.half], x[..., self.half:]
        scale, shift = self.parameters_for(first)
        return torch.cat([first, second * scale.exp() + shift], -1), scale.sum(-1)

    def inverse(self, z):
        first, second = z[..., :self.half], z[..., self.half:]
        scale, shift = self.parameters_for(first)
        x = torch.cat([first, (second - shift) * (-scale).exp()], -1)
        return x[..., self.inverse_permutation], -scale.sum(-1)


class ResidualFlow(nn.Module):
    def __init__(self, dim, hidden=256, blocks=8, seed=42):
        super().__init__()
        if dim < 2 or blocks < 1:
            raise ValueError("Flow needs dimension>=2 and at least one block")
        generator = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleList([AffineCoupling(dim, hidden, torch.randperm(dim, generator=generator))
                                     for _ in range(blocks)])

    def forward(self, x):
        determinant = x.new_zeros(x.shape[:-1])
        for block in self.blocks:
            x, change = block(x)
            determinant = determinant + change
        return x, determinant

    def inverse(self, z):
        determinant = z.new_zeros(z.shape[:-1])
        for block in reversed(self.blocks):
            z, change = block.inverse(z)
            determinant = determinant + change
        return z, determinant

    def nll(self, x):
        z, logdet = self(x.float())
        return (.5 * (z.square() + math.log(2 * math.pi)).sum(-1) - logdet) / x.shape[-1]


class ResidualDensity(nn.Module):
    def __init__(self, layers, dim, hidden=256, blocks=8, seed=42):
        super().__init__()
        self.constraints = nn.ModuleList([SignedConstraintor(dim, hidden) for _ in range(layers)])
        self.flows = nn.ModuleList([ResidualFlow(dim, hidden, blocks, seed + i) for i in range(layers)])

    def layer(self, x, index):
        constrained = self.constraints[index](x)
        return self.flows[index].nll(constrained), constrained

    def forward(self, residual, chunk=1024):
        """Only [B,V,L,D,h,w] residuals are accepted, never raw image features."""
        b, v, layers, dim, h, w = residual.shape
        maps = []
        for index in range(layers):
            flat = residual[:, :, index].permute(0, 1, 3, 4, 2).reshape(-1, dim)
            scores = [self.layer(x, index)[0] for x in flat.split(chunk)]
            maps.append(torch.cat(scores).reshape(b, v, h, w))
        return torch.stack(maps, 2)


def sampled_indices(coverage, budget):
    """Sample both strata; a small anomaly cannot be missed by uniform sampling."""
    normal = torch.where(coverage == 0)[0]
    abnormal = torch.where(coverage > 0)[0]
    def take(index):
        if len(index) <= budget:
            return index
        return index[torch.randperm(len(index), device=index.device)[:budget]]
    return take(normal), take(abnormal)


def view_weights(views, beta=.25):
    if views == 1:
        return [1.]
    if views != 5:
        raise ValueError("Expected one or five views")
    return [1 - beta] + [beta / 4] * 4


def density_loss(model, residual, coverage, tokens=256, beta=.25, margin=.5, occ_weight=.1):
    """Normal NLL + normal compaction + anomaly preservation/boundary separation.

    Boundary is the detached normal 95th percentile per layer/view mini-batch;
    there is no target adaptation or anomaly-likelihood maximization shortcut.
    """
    b, views, layers, dim, h, w = residual.shape
    total = residual.new_zeros(())
    for view, weight in enumerate(view_weights(views, beta)):
        y = coverage[:, view].reshape(-1)
        normal, abnormal = sampled_indices(y, tokens)
        for layer in range(layers):
            x = residual[:, view, layer].permute(0, 2, 3, 1).reshape(-1, dim)
            loss = x.sum() * 0
            boundary = x.new_tensor(2.)
            if len(normal):
                nll, z = model.layer(x[normal], layer)
                boundary = torch.quantile(nll.detach(), .95)
                compact = (torch.sqrt(1 + z.square().mean(-1)) - 1).mean()
                loss = nll.mean() + occ_weight * compact
            if len(abnormal):
                nll, z = model.layer(x[abnormal], layer)
                preserve = (z - x[abnormal]).square().mean(-1)
                # Coverage weighting preserves tiny positives without relabeling
                # mixed cells as normal or equating them to full defect patches.
                weights = y[abnormal].clamp_min(1e-3)
                anomaly = F.relu(boundary + margin - nll) + occ_weight * preserve
                loss = loss + (anomaly * weights).sum() / weights.sum()
            total = total + weight * loss / layers
    return total


def paired_directions(normal, abnormal):
    """[types,templates,text_dim] -> averaged, normalized state differences."""
    if normal.shape != abnormal.shape or normal.ndim != 3:
        raise ValueError("Expected equally shaped paired text features")
    delta = (F.normalize(abnormal.float(), dim=-1) - F.normalize(normal.float(), dim=-1)).mean(1)
    if not torch.isfinite(delta).all() or (delta.norm(dim=-1) < 1e-6).any():
        raise ValueError("Degenerate paired text directions")
    return F.normalize(delta, dim=-1)


class ResidualText(nn.Module):
    def __init__(self, layers, dim, directions, hidden=256):
        super().__init__()
        if directions.ndim != 2 or not torch.isfinite(directions).all():
            raise ValueError("Expected finite [directions,text_dim] anchors")
        self.register_buffer("directions", F.normalize(directions.float(), dim=-1))
        self.projectors = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.GELU(),
                                                       nn.Linear(hidden, directions.shape[-1], bias=False))
                                        for _ in range(layers)])

    def forward(self, residual):
        b, views, layers, dim, h, w = residual.shape
        score, amplitude = [], []
        for layer, projector in enumerate(self.projectors):
            x = residual[:, :, layer].permute(0, 1, 3, 4, 2)
            z = projector(x)
            # Do NOT unit-normalize z: its magnitude is anomaly evidence.
            score.append((z @ self.directions.T).permute(0, 1, 4, 2, 3))
            amplitude.append(z.norm(dim=-1))
        return torch.stack(score).mean(0), torch.stack(amplitude, 2)


def multi_positive_loss(logits, positive):
    """A region may match multiple valid states; those are not false negatives."""
    selected = positive > 0
    valid = selected.any(-1)
    if not valid.any():
        return logits.sum() * 0
    logits, selected = logits[valid], selected[valid]
    return (torch.logsumexp(logits, -1)
            - torch.logsumexp(logits.masked_fill(~selected, -torch.inf), -1)).mean()


def text_loss(model, residual, coverage, type_coverage=None, beta=.25, temperature=4., offset=2.):
    score, amplitude = model(residual)
    total = residual.new_zeros(())
    for view, weight in enumerate(view_weights(residual.shape[1], beta)):
        y = coverage[:, view]
        # Soft coverage retains small anomalies while binary support protects
        # mixed cells from being used for normal compaction.
        logits = temperature * score[:, view, 0] - offset
        bce = F.binary_cross_entropy_with_logits(logits, y)
        p = logits.sigmoid().flatten(1)
        target = y.flatten(1)
        dice = ((1 - (2 * (p * target).sum(1) + 1) / (p.sum(1) + target.sum(1) + 1))
                * (target.sum(1) > 0)).mean()
        normal = (y == 0)[:, None].expand_as(amplitude[:, view])
        abnormal = (y > 0)[:, None].expand_as(amplitude[:, view])
        norm = amplitude[:, view]
        penalty = norm[normal].square().mean() if normal.any() else norm.sum() * 0
        if abnormal.any():
            desired = residual[:, view].square().mean(2).sqrt().detach()
            penalty = penalty + (norm[abnormal] - desired[abnormal]).square().mean()
        loss = bce + dice + .01 * penalty
        if type_coverage is not None and type_coverage.shape[2]:
            typed = temperature * score[:, view, 1:].permute(0, 2, 3, 1)
            positive = type_coverage[:, view].permute(0, 2, 3, 1)
            loss = loss + multi_positive_loss(typed.reshape(-1, typed.shape[-1]),
                                               positive.reshape(-1, positive.shape[-1]))
        total = total + weight * loss
    return total


def positive_text_fusion(density, text, strength):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Text fusion strength must be finite and nonnegative")
    if strength == 0:
        return density  # exact fallback; no multiplication by a possibly NaN map
    return density + strength * text.clamp_min(0)
