"""Label-free model interfaces for the three-stage alignment ablation.

The legacy extractor/head are deliberately not modified: v1 seals remain valid.
Masks are accepted only by the separate supervision/teacher-loss functions.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tools.reference_highres import (
    EvidenceHead, FrozenExtractor, arm_inputs, patch_coordinates, stitch_tiles,
)
from tools.upstream_repair import corrected_probability, constrained_loss, reference_logits

HEAD_MODES = ("H1", "H2", "H3", "H1_aux0", "T10", "T01", "T11")
CORE_HEADS = ("H1", "H2", "H3", "T10", "T01", "T11")
LATE = ("T01", "T11")
KEEP = ("T10", "T11")


class AlignmentHead(EvidenceHead):
    """H1 exactly retains v1 parameter names and initialization order."""
    def __init__(self, mode, layers, feature_dim, anchors, hidden=64):
        if mode not in HEAD_MODES:
            raise ValueError(f"Unknown head: {mode}")
        super().__init__(layers, feature_dim, anchors, hidden)
        self.mode = mode
        if mode == "H2":
            del self.to_text
            del self.anchors
            self.delta = nn.Conv2d(hidden, 1, 1)
        elif mode in ("H3", "T10", "T01", "T11"):
            self.delta = nn.Conv2d(1, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, inputs):
        q, residual = inputs["query"], inputs["residual"]
        b, layers, dim, h, w = q.shape
        if (layers, dim) != (self.layers, self.feature_dim):
            raise ValueError("Cache/head dimensions differ")
        valid = inputs["reference_valid"]
        matched = (q - residual) * valid[:, None]
        evidence = torch.cat([matched, residual, residual.abs(), inputs["distance"][:, :, None],
                              inputs["uncertainty"][:, :, None]], 2).reshape(b, -1, h, w)
        visual = self.visual(q.reshape(b, -1, h, w))
        reference = self.reference(evidence) * valid
        base = F.interpolate(inputs["base"], (h, w), mode="bilinear", align_corners=False)
        xy = inputs.get("coords")
        if xy is None:
            xy = patch_coordinates(h, device=q.device).T.reshape(1, 2, h, w).expand(b, -1, -1, -1)
        hidden = self.fuse(torch.cat([visual, reference, inputs["prior"], base, valid, xy], 1))
        size = inputs["base"].shape[-2:]
        resize = lambda x: F.interpolate(x.float(), size, mode="bilinear", align_corners=False)
        if self.mode == "H2":
            delta = self.delta(hidden)
            return dict(delta=resize(delta), hidden_term=resize(delta - self.delta.bias[None, :, None, None]))
        aligned = F.normalize(self.to_text(hidden).float(), dim=1)
        logits = self.temperature * torch.einsum("bdhw,kd->bkhw", aligned, self.anchors)
        margin = logits[:, 1:2] - logits[:, :1]
        if self.mode in ("H1", "H1_aux0"):
            delta = self.delta(torch.cat([hidden, margin.to(hidden.dtype)], 1))
            hidden_term = F.conv2d(hidden, self.delta.weight[:, :-1])
            semantic_term = margin * self.delta.weight[0, -1, 0, 0].float()
        else:
            delta = self.delta(margin.to(hidden.dtype))
            hidden_term = torch.zeros_like(margin)
            semantic_term = margin * self.delta.weight[0, 0, 0, 0].float()
        return dict(delta=resize(delta), semantic_logits=resize(margin), aligned=aligned,
                    hidden_term=resize(hidden_term), semantic_term=resize(semantic_term))


class ResidualHead(nn.Module):
    def __init__(self, mode, layers, feature_dim, hidden=64):
        super().__init__()
        if mode not in ("R2", "R3") or hidden % 8:
            raise ValueError("Invalid residual head")
        self.mode = mode
        self.branch = nn.Sequential(
            nn.Conv2d(layers * (3 * feature_dim + 2), hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        self.delta = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        if mode == "R3":
            self.gate = nn.Sequential(nn.Conv2d(2 * layers + 1, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, inputs):
        q, residual = inputs["query"], inputs["residual"]
        b, layers, dim, h, w = q.shape
        features = torch.cat([q - residual, residual, residual.abs(), inputs["distance"][:, :, None],
                              inputs["uncertainty"][:, :, None]], 2).reshape(b, -1, h, w)
        raw = self.delta(self.branch(features)).float()
        if self.mode == "R3":
            similarity = inputs["reference_similarity"].reshape(b, 1, 1, 1).expand(b, 1, h, w)
            gate = self.gate(torch.cat([inputs["distance"], inputs["uncertainty"], similarity], 1)).float().sigmoid()
        else:
            gate = torch.full_like(raw, .5)
        size = inputs["base"].shape[-2:]
        resize = lambda x: F.interpolate(x, size, mode="bilinear", align_corners=False)
        return dict(delta=resize(raw * gate), gate=resize(gate), raw_delta=resize(raw))


class TeacherExtractor(FrozenExtractor):
    """Frozen teacher is reconstructed from captured, pre-standardized tokens."""
    @torch.inference_mode()
    def view(self, image):
        result = super().view(image)
        if getattr(self, "include_teacher", True):
            aligned = []
            for i, layer in enumerate(self.layers):
                patches = self.captured[layer]
                standardized = (patches - patches.mean(1, keepdim=True)) / (patches.std(1, keepdim=True) + 1e-6)
                aligned.append(F.normalize(self.adapter.patch_token_adapter[i](standardized)[0][0].float(), dim=-1))
            teacher = F.normalize(torch.stack(aligned).mean(0), dim=-1)
            side = math.isqrt(teacher.shape[0])
            result["teacher"] = teacher.T.reshape(-1, side, side)
        return result


def batch_stitch(values, boxes, size):
    """Each item in a batch has its own native-image crop geometry."""
    stitch = double_stitch if values[0].dtype == torch.float64 else stitch_tiles
    return torch.stack([stitch([v[i] for v in values], boxes[i], size) for i in range(values[0].shape[0])])


def double_stitch(values, boxes, size):
    """Same Hann geometry in float64 for finite-endpoint training surrogates only."""
    first=values[0]
    total=first.new_zeros(first.shape[0],size,size)
    weight=first.new_zeros(1,size,size)
    axis=(torch.arange(size,device=first.device,dtype=torch.float64)+.5)/size
    yy,xx=torch.meshgrid(axis,axis,indexing="ij")
    for value,(x1,y1,x2,y2) in zip(values,boxes):
        if not (0<=x1<x2<=1 and 0<=y1<y2<=1):
            raise ValueError("Invalid surrogate crop")
        gx,gy=(xx-x1)/(x2-x1),(yy-y1)/(y2-y1)
        support=((xx+.5/size>x1)&(xx-.5/size<x2)&(yy+.5/size>y1)&(yy-.5/size<y2))
        window=(.05+.95*torch.sin(math.pi*gx.clamp(0,1)).square())*(.05+.95*torch.sin(math.pi*gy.clamp(0,1)).square())*support
        grid=torch.stack([2*gx-1,2*gy-1],-1)[None]
        sampled=F.grid_sample(value[None],grid,mode="bilinear",padding_mode="border",align_corners=False)[0]
        total=total+sampled*window
        weight=weight+window
    if bool((weight<=0).any()):
        raise ValueError("Surrogate crop grid has uncovered cells")
    return total/weight


def frozen_high(batch, cfg):
    arrays = batch["old"]
    return (1 - cfg["high_weight"]) * arrays["global_base"] + cfg["high_weight"] * arrays["high_base"]


def early_inputs(batch, cfg, reference=False):
    result = arm_inputs(batch["old"], "D_highres_reference" if reference else "B_highres",
                        cfg["feature_grid"], cfg["high_weight"])
    if reference:
        result["reference_similarity"] = batch["similarity"]
    return result


def tile_inputs(batch, index):
    view = batch["views"][index]
    result = {k: view[k] for k in ("query", "prior", "base")}
    result.update(residual=torch.zeros_like(view["query"]), distance=torch.zeros_like(view["prior"]),
                  uncertainty=torch.zeros_like(view["prior"]), reference_valid=torch.zeros_like(view["prior"][:, :1]))
    side = view["query"].shape[-1]
    result["coords"] = torch.stack([patch_coordinates(side, box, view["query"].device).T.reshape(2, side, side)
                                    for box in batch["boxes"][:, index].detach().cpu().tolist()])
    return result


def head_bundle(model, mode, batch, cfg):
    """Prediction accepts feature-only batches; neither masks nor categories are used."""
    if mode in LATE:
        views = [tile_inputs(batch, i) for i in range(len(batch["views"]))]
    else:
        views = [early_inputs(batch, cfg)]
    return dict(outputs=[model(v) for v in views], views=views, base=frozen_high(batch, cfg),
                boxes=batch.get("boxes"), mode=mode)


def bundle_prediction(bundle, alpha, cfg):
    if alpha == 0:
        return bundle["base"]
    outputs, views = bundle["outputs"], bundle["views"]
    if bundle["mode"] not in LATE:
        return corrected_probability(bundle["base"], outputs[0]["delta"], alpha)
    predicted = [corrected_probability(v["base"], o["delta"], alpha) for v, o in zip(views, outputs)]
    high = batch_stitch(predicted[1:], bundle["boxes"][:, 1:].detach().cpu().tolist(), cfg["image_size"])
    return (1 - cfg["high_weight"]) * predicted[0] + cfg["high_weight"] * high


def bundle_training_logits(bundle,alpha,cfg):
    """Loss-only finite endpoint proxy; never changes actual inference probabilities.

    Subtract the surrogate's initial logit offset so a zero residual has exactly
    the Base loss while retaining a nonzero gradient at cached exact 0/1 pixels.
    """
    base_logits=reference_logits(bundle["base"])
    if alpha==0:
        return base_logits
    if bundle["mode"] not in LATE:
        return base_logits+alpha*bundle["outputs"][0]["delta"].double()
    def mixture(change):
        values=[torch.sigmoid(reference_logits(view["base"])+change*output["delta"].double())
                for view,output in zip(bundle["views"],bundle["outputs"])]
        high=batch_stitch(values[1:],bundle["boxes"][:,1:].detach().cpu().tolist(),cfg["image_size"])
        return (1-cfg["high_weight"])*values[0]+cfg["high_weight"]*high
    if "initial_surrogate_logits" not in bundle:
        with torch.no_grad():
            bundle["initial_surrogate_logits"]=reference_logits(mixture(0))
    return base_logits+reference_logits(mixture(alpha))-bundle["initial_surrogate_logits"]


def normal_cells(mask, side, boxes=None):
    """Conservative max-pooling: a one-pixel defect cannot disappear at downsampling."""
    if boxes is None:
        return F.adaptive_max_pool2d(mask.float(), (side, side)) == 0
    result = []
    h, w = mask.shape[-2:]
    for value, (x1, y1, x2, y2) in zip(mask, boxes.detach().cpu().tolist()):
        xa, ya = max(0, math.floor(x1 * w)), max(0, math.floor(y1 * h))
        xb, yb = min(w, math.ceil(x2 * w)), min(h, math.ceil(y2 * h))
        if xa >= xb or ya >= yb:
            raise ValueError("Empty supervision crop")
        result.append(F.adaptive_max_pool2d(value[None, :, ya:yb, xa:xb].float(), (side, side))[0] == 0)
    return torch.stack(result)


def teacher_loss(bundle, batch, mask, cfg):
    if bundle["mode"] not in KEEP:
        return mask.flatten(1).sum(1) * 0
    teachers = [v["teacher"].detach() for v in batch["views"]]
    if bundle["mode"] not in LATE:
        side = cfg["feature_grid"]
        low = F.interpolate(teachers[0], (side, side), mode="bilinear", align_corners=False)
        high = batch_stitch(teachers[1:], batch["boxes"][:, 1:].detach().cpu().tolist(), side)
        target = F.normalize((1 - cfg["high_weight"]) * low + cfg["high_weight"] * high, dim=1)
        teachers = [target]
    losses = []
    for index, (output, target) in enumerate(zip(bundle["outputs"], teachers)):
        aligned = output["aligned"]
        keep = normal_cells(mask, aligned.shape[-1], batch["boxes"][:, index] if bundle["mode"] in LATE else None)
        error = (1 - (aligned * F.normalize(target.float(), dim=1)).sum(1, keepdim=True)) * keep
        losses.append(error.flatten(1).sum(1) / keep.flatten(1).sum(1).clamp_min(1))
    if len(losses) == 1:
        return losses[0]
    return (1 - cfg["high_weight"]) * losses[0] + cfg["high_weight"] * torch.stack(losses[1:]).mean(0)


def probability_loss(base, probability, mask, cfg, logits=None):
    logits = reference_logits(probability) if logits is None else logits
    bce = F.binary_cross_entropy_with_logits(logits, mask.double(), reduction="none").flatten(1).mean(1)
    p, y = probability.flatten(1), mask.flatten(1)
    dice = (1 - (2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)) * (y.sum(1) > 0)
    bg = []
    for i in range(len(mask)):
        increase = (probability[i] - base[i]).relu()[mask[i] == 0]
        count = max(1, math.ceil(increase.numel() * .01))
        bg.append(increase.topk(count).values.mean() if increase.numel() else probability[i].sum() * 0)
    delta = logits - reference_logits(base)
    magnitude = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none").flatten(1).mean(1)
    return bce + dice + cfg["background_weight"] * torch.stack(bg) + cfg["residual_weight"] * magnitude


def head_loss(bundle, batch, mask, cfg):
    if bundle["mode"] in LATE:
        main = probability_loss(bundle["base"], bundle_prediction(bundle, 1., cfg), mask, cfg,
                                bundle_training_logits(bundle,1.,cfg))
    else:
        main, _ = constrained_loss(bundle["base"], bundle["outputs"][0]["delta"], mask,
                                   cfg["background_weight"], cfg["residual_weight"])
    sem = []
    for i, output in enumerate(bundle["outputs"]):
        if "semantic_logits" not in output or bundle["mode"] == "H1_aux0":
            continue
        target = mask
        if bundle["mode"] in LATE and i:
            # Area-preserving binary support; never nearest-resize a tiny defect away.
            side = output["semantic_logits"].shape[-1]
            target = (~normal_cells(mask, side, batch["boxes"][:, i])).float()
        sem.append(F.binary_cross_entropy_with_logits(output["semantic_logits"], target, reduction="none").flatten(1).mean(1))
    if sem:
        semantic = sem[0] if len(sem) == 1 else ((1 - cfg["high_weight"]) * sem[0] + cfg["high_weight"] * torch.stack(sem[1:]).mean(0))
        main = main + cfg["semantic_weight"] * semantic
    if bundle["mode"] in KEEP:
        main = main + cfg["keep_weight"] * teacher_loss(bundle, batch, mask, cfg)
    return main


def eligible(candidate, baseline):
    keys = ("PRO_exact", "P_AP")
    return (all(candidate[k] is not None and baseline[k] is not None and math.isfinite(candidate[k]) for k in keys)
            and all(candidate[k] >= baseline[k] for k in keys)
            and any(candidate[k] > baseline[k] for k in keys))


def selection_key(candidate):
    return (-candidate["PRO_exact"], -candidate["P_AP"], candidate["alpha"], candidate["epoch"])


def best_source(candidates, baseline):
    valid = [c for c in candidates if eligible(c, baseline)]
    return min(valid, key=selection_key) if valid else baseline


def binary_selection_loss(probability, mask, logits=None):
    logits = reference_logits(probability) if logits is None else logits
    bce = F.binary_cross_entropy_with_logits(logits, mask.double(), reduction="none").flatten(1).mean(1)
    p, y = probability.flatten(1), mask.flatten(1)
    return bce + (1 - (2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)) * (y.sum(1) > 0)
