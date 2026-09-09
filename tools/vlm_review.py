"""Label-free ROI review, strict parsing and bounded calibration for dual towers."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

PROTOCOL = "clip_dino_roi_review_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_rois(base, disagreement, seed=0, fraction=0.25):
    """Two peaks, one layer-disagreement ROI, one coverage ROI; no GT access."""
    base = np.asarray(base, dtype=np.float32)
    disagreement = np.asarray(disagreement, dtype=np.float32)
    if base.ndim != 2 or base.shape != disagreement.shape:
        raise ValueError("Expected equally shaped 2D maps")
    if not np.isfinite(base).all() or not np.isfinite(disagreement).all():
        raise ValueError("Non-finite evidence map")
    if not 0 < fraction <= 0.5:
        raise ValueError("ROI fraction must be in (0, 0.5]")
    h, w = base.shape
    bh, bw = max(2, round(h * fraction)), max(2, round(w * fraction))
    bh, bw = min(h, bh), min(w, bw)
    rng = np.random.default_rng(seed)
    boxes = []
    result = []
    for kind, score in (("peak", base), ("peak", base),
                        ("disagreement", disagreement), ("coverage", rng.random(base.shape))):
        for index in np.argsort(-score.ravel(), kind="stable"):
            y, x = divmod(int(index), w)
            x1, y1 = max(0, min(w - bw, x - bw // 2)), max(0, min(h - bh, y - bh // 2))
            box = [x1, y1, x1 + bw, y1 + bh]
            def overlap(other):
                ix = max(0, min(box[2], other[2]) - max(box[0], other[0]))
                iy = max(0, min(box[3], other[3]) - max(box[1], other[1]))
                inter = ix * iy
                return inter / max(1, 2 * bw * bh - inter)
            if any(overlap(other) > 0.35 for other in boxes):
                continue
            boxes.append(box)
            result.append({"roi_id": len(result) + 1, "box": box, "kind": kind})
            break
    return result


def review_prompt(category, rois, heatmap=False):
    # Paths, labels, masks, base scores and internal DINO layer identities are
    # deliberately excluded. A query is never presented as a normal reference.
    order = "Image 1 is the query; the following images are ROI crops in listed order."
    if heatmap:
        order = "Image 1 is the query, image 2 is a fallible proposal heatmap; the following images are ROI crops in listed order."
    ids = [int(r["roi_id"]) for r in rois]
    return (
        f"Inspect visible industrial defects in category {category}. {order} "
        "No known-normal reference is available. Do not assume the query is normal. "
        "Distinguish visible damage from expected texture, reflections and edges. "
        "If visual evidence is insufficient use uncertain. Do not invent pixel masks or feature-layer choices. "
        f"Return one entry for each ROI id {ids}. Return JSON only: "
        '{"regions":[{"roi_id":1,"verdict":"defect",'
        '"confidence":0.8,"defect_type":"scratch"}]}. '
        'verdict must be "defect", "normal", or "uncertain"; confidence must be a number in [0,1].'
    )


def parse_review(text, rois):
    """Malformed/incomplete responses fail closed, rather than guessed repairs."""
    fallback = {"parse_ok": False, "regions": []}
    try:
        value = str(text).strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.S | re.I)
        if match:
            value = match.group(1)
        data = json.loads(value)
        regions = data["regions"]
        expected = {int(r["roi_id"]) for r in rois}
        if not isinstance(regions, list) or len(regions) != len(expected):
            return fallback
        found, clean = set(), []
        for region in regions:
            idx, verdict, confidence = region["roi_id"], region["verdict"], region["confidence"]
            if type(idx) is not int or idx not in expected or idx in found:
                return fallback
            if verdict not in {"defect", "normal", "uncertain"}:
                return fallback
            if type(confidence) not in (int, float) or not np.isfinite(confidence) or not 0 <= confidence <= 1:
                return fallback
            defect_type = region.get("defect_type", "")
            if not isinstance(defect_type, str):
                return fallback
            found.add(idx)
            clean.append({"roi_id": idx, "verdict": verdict, "confidence": float(confidence),
                          "defect_type": defect_type[:160]})
        return {"parse_ok": True, "regions": sorted(clean, key=lambda r: r["roi_id"])}
    except (ValueError, TypeError, KeyError):
        return fallback


def calibrate_map(base, rois, review, alpha=0.5, confidence_threshold=0.8, control=False):
    """Bounded log-odds correction with tapered ROI edges and overlap averaging.

    Confidence is only an uncalibrated eligibility filter, not a probability
    target. Unknown/rejected/missing reviews preserve the base exactly.
    """
    if not 0 <= alpha <= 2 or not 0 <= confidence_threshold <= 1:
        raise ValueError("Require alpha in [0,2] and confidence_threshold in [0,1]")
    base = np.asarray(base, dtype=np.float32)
    if base.ndim != 2 or not np.isfinite(base).all() or np.any((base < 0) | (base > 1)):
        raise ValueError("Base must be a finite 2D probability map")
    votes = {r["roi_id"]: r for r in review.get("regions", [])} if review.get("parse_ok") else {}
    accum, weights = np.zeros_like(base), np.zeros_like(base)
    h, w = base.shape
    for roi in rois:
        vote = votes.get(roi["roi_id"])
        if not control and (vote is None or vote["confidence"] < confidence_threshold or vote["verdict"] == "uncertain"):
            continue
        sign = -1 if control or vote["verdict"] == "normal" else 1
        x1, y1, x2, y2 = roi["box"]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("ROI outside probability map")
        taper = np.outer(np.hanning(y2 - y1 + 2)[1:-1], np.hanning(x2 - x1 + 2)[1:-1]).astype(np.float32)
        accum[y1:y2, x1:x2] += sign * taper
        weights[y1:y2, x1:x2] += taper
    delta = alpha * accum / np.maximum(weights, 1.0)
    result = base.copy()
    touched = delta != 0
    p = np.clip(base[touched], 1e-6, 1 - 1e-6)
    result[touched] = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + delta[touched])))
    return result
