"""Normal--normal residual pools. No query labels or trainable parameters.

Keep a real pool of signed residual vectors, not just a scalar threshold. A
support image is matched only to OTHER support images, including local views.
The first experiment uses their uncentered channel-wise second moments. This
keeps zero residuals at zero and exposes a simple, testable normal-variation
baseline before changing Flow or language heads.
"""
from __future__ import annotations

import math
import numpy as np

from tools.residual_text_runtime import export_residuals


def normal_pool(extractor, views, references, cfg):
    if len(views) != len(references) or len(views) < 2:
        raise ValueError("Normal mutual matching requires at least two normal references")
    if any(r.get("mask_path") or r.get("partition") != "reference" for r in references):
        raise ValueError("Only known-normal reference records may build the normal pool")
    if len({r["rgb_sha"] for r in references}) != len(references):
        raise ValueError("Duplicate normal image content in the support pool")
    if len({(r["dataset"], r["category"]) for r in references}) != 1:
        raise ValueError("Normal supports must come from the same category")
    residuals, distances, indices, correspondence = [], [], [], []
    for index, query in enumerate(views):
        others = [j for j in range(len(views)) if j != index]
        # The entire held-out image (global AND every crop) is excluded.
        arrays = export_residuals(extractor, query, [views[j] for j in others], cfg)
        residuals.append(arrays["residual"])
        distances.append(arrays["distance"])
        indices.append(arrays["match_index"])
        correspondence.append(dict(query_id=references[index]["id"],
                                   bank_ids=[references[j]["id"] for j in others]))
    vectors = np.stack(residuals)
    return dict(residual=vectors, distance=np.stack(distances),
                match_index=np.stack(indices), second_moment=second_moment(vectors)), correspondence


def second_moment(vectors):
    """[K,5,L,D,h,w] -> [L,2,D], equal weight per image and local view."""
    vectors = np.asarray(vectors, np.float32)
    if vectors.ndim != 6 or vectors.shape[0] < 2 or vectors.shape[1] != 5:
        raise ValueError("Expected normal residuals [K,5,L,D,h,w], K>=2")
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite normal residual pool")
    per_view = np.square(vectors.astype(np.float64)).mean(axis=(-1, -2))
    return np.stack([per_view[:, :1].mean(axis=(0, 1)),
                     per_view[:, 1:].mean(axis=(0, 1))], axis=1)


def variation_scale(source, support, rho=.5, floor_ratio=.01):
    """Shrink support second moments to a source-class-balanced normal prior.

    rho=1 is the source-only control. Floors bound amplification when a support
    set happens to have a near-zero channel. No per-query/image normalization.
    """
    source, support = np.asarray(source, np.float64), np.asarray(support, np.float64)
    if (source.shape != support.shape or source.ndim != 3 or source.shape[1] != 2
            or not math.isfinite(rho) or not 0 <= rho <= 1
            or not math.isfinite(floor_ratio) or not 0 < floor_ratio <= 1
            or not np.isfinite(source).all() or not np.isfinite(support).all()
            or (source < 0).any() or (support < 0).any()):
        raise ValueError("Invalid normal variation statistics or shrinkage")
    average = source.mean(-1, keepdims=True)
    if (average < 1e-16).any():
        raise ValueError("Degenerate source normal pool; inspect normal support diversity")
    floor = np.maximum(np.maximum(floor_ratio * source, 1e-4 * average), 1e-12)
    return np.sqrt(np.maximum(rho * source + (1 - rho) * support, floor)).astype(np.float32)


def raw_maps(residual, distance, source, support, rho=.5, floor_ratio=.01):
    """Four score families at every level/view; inputs contain NO appearance bypass.

    D: original full-dimensional distance; P: projected residual magnitude;
    S: source-only channel scale; N: normal-support-pool + source channel scale.
    """
    residual, distance = np.asarray(residual, np.float32), np.asarray(distance, np.float32)
    if (residual.ndim != 5 or residual.shape[0] != 5
            or distance.shape != residual.shape[:2] + residual.shape[-2:]
            or np.shape(source) != (residual.shape[1], 2, residual.shape[2])
            or not np.isfinite(residual).all() or not np.isfinite(distance).all()):
        raise ValueError("Expected finite per-view residual/distance arrays")
    square = np.square(residual)
    result = dict(D=distance, P=np.sqrt(square.mean(2)))
    for name, value in (("S", 1.), ("N", rho)):
        scale = variation_scale(source, support, value, floor_ratio)
        denominator = scale[:, [0, 1, 1, 1, 1], :].transpose(1, 0, 2)
        result[name] = np.sqrt((square / denominator[..., None, None] ** 2).mean(2))
    return result


def calibrate_views(scores, calibration):
    center, scale = np.asarray(calibration["center"]), np.asarray(calibration["scale"])
    indices = [0, 1, 1, 1, 1]
    center, scale = center[:, indices].T, scale[:, indices].T
    if (center.shape != scores.shape[:2] or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid source-only score calibration")
    return ((scores - center[..., None, None]) / scale[..., None, None]).mean(1, keepdims=True)


def image_scores(maps, fraction=.01):
    """Report BOTH maximum and top-fraction mean; never select using target GT."""
    maps = np.asarray(maps)
    if maps.ndim != 3 or not np.isfinite(maps).all() or not 0 < fraction <= 1:
        raise ValueError("Expected finite [images,h,w] maps and a valid top fraction")
    flat = maps.reshape(len(maps), -1)
    count = max(1, math.ceil(flat.shape[1] * fraction))
    return flat.max(1), np.partition(flat, flat.shape[1] - count, axis=1)[:, -count:].mean(1)
