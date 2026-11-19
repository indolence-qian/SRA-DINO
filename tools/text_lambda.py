"""Fixed-map lambda ablations; all calibration/heads remain frozen."""
from __future__ import annotations

import base64
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from tools.residual_text import fuse_views
import residual_text_experiment as base

ARMS = ("T1", "T2", "T3")
DEFAULT_GRID = (0., .1, .25, .5, 1., 2., 4.)


def parse_grid(value):
    values = [float(v) for v in value.split(",")]
    if (len(set(values)) != len(values) or not all(math.isfinite(v) and v >= 0 for v in values)
            or set(values) != set(DEFAULT_GRID)):
        raise ValueError("The preregistered grid is 0,0.1,0.25,0.5,1,2,4; 8 is source-gated")
    return sorted(values)


def readouts(grid):
    result = {"V":dict(arm="V", strength=0., kind="visual")}
    for arm in ARMS:
        for value in grid:
            if value:
                key = arm+"_l"+format(value,"g").replace(".","p")
                result[key] = dict(arm=arm,strength=float(value),kind="fusion")
        result[arm+"_increment"] = dict(arm=arm,strength=None,kind="increment")
    return result


def name_for(arm, value):
    return "V" if value == 0 else arm+"_l"+format(value,"g").replace(".","p")


def score(basis, spec):
    visual = basis["V"]
    if spec["kind"] == "visual" or spec["strength"] == 0:
        return visual  # Exact lambda-zero baseline, even if an unused A is invalid.
    increment = basis["A_"+spec["arm"]]
    if spec["kind"] == "increment": return increment
    return visual + spec["strength"]*increment


@torch.inference_mode()
def basis_maps(visual, texts, selected, batch, scale, cfg, size):
    p = cfg["settings"]
    v = base.calibrated_map(visual,"density",batch,scale,"HR",cfg["visual_calibration"])
    result = {"V":fuse_views(v,batch["boxes"],size,True,p["high_weight"])[0,0].cpu().numpy()}
    for arm,model in texts.items():
        t = base.calibrated_map(model,"text",batch,scale,"HR",selected["text"][arm]["calibration"])
        # Clamp BEFORE view stitching. relu(stitch(t)) is NOT the same operation.
        result["A_"+arm] = fuse_views(t.clamp_min(0),batch["boxes"],size,True,p["high_weight"])[0,0].cpu().numpy()
    return result


def quality(point):
    values = [point.get(k) for k in ("P_AP","PRO_exact")]
    return sum(values)/2 if all(v is not None and math.isfinite(v) for v in values) else -math.inf


def best_lambda(points, grid):
    chosen = min(grid)
    for value in sorted(grid):
        if quality(points[value]) > quality(points[chosen])+1e-12: chosen=value
    if quality(points[chosen]) == -math.inf: raise ValueError("No valid source validation metric")
    return chosen


def needs_extension(points, grid):
    return best_lambda(points,grid)==4 and quality(points[4]) > quality(points[2])+1e-12


def diagnostics(values, truth, label, threshold):
    predicted = values > threshold
    tp,fp,fn = int((predicted&truth).sum()),int((predicted&~truth).sum()),int((~predicted&truth).sum())
    image_score = float(values.max()); triggered = image_score > threshold
    annotated = bool(label and truth.any())
    return dict(score=image_score,threshold=threshold,outcome=("TP" if label else "FP") if triggered else ("FN" if label else "TN"),
        tp_pixels=tp,fp_pixels=fp,fn_pixels=fn,
        dice=2*tp/(2*tp+fp+fn) if annotated else None,
        iou=tp/(tp+fp+fn) if annotated else None,
        recall=tp/(tp+fn) if annotated else None,
        background_fpr=fp/int((~truth).sum()) if (~truth).any() else None)


def pool_mask(mask, side):
    return F.adaptive_max_pool2d(torch.from_numpy(mask.copy()).float()[None,None],(side,side))[0,0].numpy()>0


def html_asset(root,row,basis,truth,specs,thresholds,side,key):
    """Display-only quantization. All numerical metrics use original fp32 maps."""
    folder=Path(root)/"html"/"assets"; folder.mkdir(parents=True,exist_ok=True)
    original=folder/(row["id"]+".jpg"); gt=folder/(row["id"]+"_gt.png")
    if not original.exists():
        with Image.open(row["image_path"]) as image:
            image=image.convert("RGB"); image.thumbnail((384,384)); image.save(original,quality=90)
    Image.fromarray(pool_mask(truth,256).astype(np.uint8)*255).save(gt)
    maps={}
    for name,values in basis.items():
        thumb=F.interpolate(torch.from_numpy(values)[None,None],(side,side),mode="bilinear",align_corners=False)[0,0].numpy()
        lo,hi=float(thumb.min()),float(thumb.max())
        q=np.round((thumb-lo)/max(hi-lo,1e-20)*65535).astype('<u2')
        maps[name]=dict(lo=lo,hi=hi,data=base64.b64encode(q.tobytes()).decode())
    binary={name:base64.b64encode(np.packbits(pool_mask(score(basis,spec)>thresholds[name],side)).tobytes()).decode()
            for name,spec in specs.items()}
    data=dict(side=side,maps=maps,binary=binary,gt=base64.b64encode(np.packbits(pool_mask(truth,side)).tobytes()).decode())
    return data,dict(original="assets/"+original.name,gt="assets/"+gt.name,payload="assets/"+key+".js")
