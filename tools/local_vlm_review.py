"""Small-defect proposals and mask-restricted decisions. No evaluation labels."""
import json
import re

import numpy as np
from PIL import Image

PROTOCOL = "clip_dino_small_defect_v1"


def small_candidates(base, disagreement, max_candidates=6, max_area_fraction=0.005,
                     total_area_fraction=0.02, min_probability=0.1):
    """Seeded local connected components, not rectangular correction supports.

    Keep even single pixels and thin components. Oversized/broad plateaus are
    left to Base, never arbitrarily cut down to fit the area budget. All limits
    refer to detector-map geometry, not the raw image. No GT is accessed.
    """
    import cv2
    base, disagreement = np.asarray(base, np.float32), np.asarray(disagreement, np.float32)
    if base.ndim != 2 or base.shape != disagreement.shape or not np.isfinite(base).all() or not np.isfinite(disagreement).all():
        raise ValueError("Expected finite, equally shaped 2D evidence")
    if np.any((base < 0) | (base > 1)):
        raise ValueError("Base evidence must be probabilities")
    if not 0 < max_area_fraction <= total_area_fraction <= 0.1 or max_candidates < 1 or not 0 <= min_probability <= 1:
        raise ValueError("Invalid small-candidate limits")
    h, w = base.shape
    cap, budget = max(1, int(base.size * max_area_fraction)), max(1, int(base.size * total_area_fraction))
    used = np.zeros_like(base, dtype=bool)
    result, supports = [], []
    # Alternate peak and disagreement seeds; local growth has two scales.
    scores = [base.copy(), disagreement.copy()]
    floor = float(np.quantile(base, 0.5))
    for step in range(max_candidates * 12):
        branch = step % 2
        score = scores[branch]
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        if not np.isfinite(score[y, x]) or score[y, x] <= 0:
            if all(not np.isfinite(s).any() or np.max(s) <= 0 for s in scores):
                break
            continue
        # Prevent repeated seeds in the same immediate neighborhood.
        radius = max(1, min(h, w) // 128)
        for s in scores:
            s[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -np.inf
        if base[y, x] < min_probability or used[y, x] or base[y, x] <= floor + 1e-6:
            continue
        selected = None
        for scale in (0.125, 0.25):
            rad = max(2, round(min(h, w) * scale / 2))
            x1, x2, y1, y2 = max(0, x-rad), min(w, x+rad+1), max(0, y-rad), min(h, y+rad+1)
            local = base[y1:y2, x1:x2]
            for ratio in (0.6, 0.8, 0.95):
                threshold = max(min_probability, floor + ratio * (float(base[y, x]) - floor))
                _, labels = cv2.connectedComponents((local >= threshold).astype(np.uint8), connectivity=8)
                label = labels[y-y1, x-x1]
                if not label:
                    continue
                component = labels == label
                # Grow at the wider scale if a component was truncated by the window.
                truncated = ((x1 > 0 and component[:, 0].any()) or (x2 < w and component[:, -1].any()) or
                             (y1 > 0 and component[0].any()) or (y2 < h and component[-1].any()))
                area = int(component.sum())
                if truncated or area > cap or area + used.sum() > budget:
                    continue
                support = np.zeros_like(used)
                support[y1:y2, x1:x2] = component
                if np.any(support & used):
                    continue  # no duplicate/repeated correction of a component
                selected = support
                break
            if selected is not None:
                break
        if selected is None:
            continue
        yy, xx = np.where(selected)
        result.append({"roi_id": len(result)+1, "box": [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                       "kind": "peak" if branch == 0 else "disagreement", "area": int(selected.sum())})
        supports.append(selected)
        used |= selected
        for s in scores:
            s[selected] = -np.inf
        if len(result) >= max_candidates or used.sum() >= budget:
            break
    return result, np.stack(supports) if supports else np.zeros((0, h, w), dtype=bool)


def raw_boxes(box, detector_shape, raw_size):
    """Dataset export uses direct square resize with same-size center crop.

    Map pixel *edges* using independent x/y scales, then crop native RGB before
    any VLM resizing. Tight crop has a small halo; context contains more anatomy.
    """
    h, w = detector_shape
    rw, rh = raw_size
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Candidate box outside detector geometry")
    def expand(factor, minimum):
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw, bh = max((x2-x1)*factor, minimum), max((y2-y1)*factor, minimum)
        return [max(0, int(np.floor((cx-bw/2)*rw/w))), max(0, int(np.floor((cy-bh/2)*rh/h))),
                min(rw, int(np.ceil((cx+bw/2)*rw/w))), min(rh, int(np.ceil((cy+bh/2)*rh/h)))]
    return expand(1.5, 8), expand(4.0, 32)


def native_crops(raw, box, detector_shape, image_limit=512, resized_ablation=False):
    if resized_ablation:
        raw = raw.resize((detector_shape[1], detector_shape[0]), Image.Resampling.BICUBIC)
    tight, context = raw_boxes(box, detector_shape, raw.size)
    images = [raw.crop(tuple(context)), raw.crop(tuple(tight))]
    for image in images:
        image.thumbnail((image_limit, image_limit), Image.Resampling.LANCZOS)
    return images, {"detail_box": tight, "context_box": context, "source_size": list(raw.size)}


def local_prompt(category, candidate_id, reference=False, generic=False):
    order = ("Image 1 is context around ONE candidate, image 2 is a tight detail crop of that SAME candidate. "
             "The candidate is near the center of the detail crop, except where cropped at the image boundary. ")
    order += ("Image 3 is a possible matching patch from a separate known-normal TRAINING image. "
              "First check if its structure/location is comparable; an unmatched reference is not evidence of normality. "
              if reference else "No known-normal reference is available. ")
    specific = (
        "The defect may occupy only a few pixels. Do not dismiss weak contrast, a tiny area, or a mostly normal object. "
        "Inspect local material continuity, thin cracks, scratches, missing edges, foreign material and texture discontinuities. "
        "These are inspection possibilities, NOT claims that a defect exists. Distinguish damage from highlights, shadows, "
        "normal seams and manufacturing texture. Not seeing a defect does NOT establish normality. "
        "Use normal_supported only with positive visual evidence of normal structure. If detail is unresolved, the reference "
        "is misleading, or evidence is ambiguous, use insufficient_evidence. "
    ) if not generic else "Inspect visible industrial damage versus normal appearance; abstain if evidence is unclear. "
    return (f"Review industrial category {category}, candidate {candidate_id}. " + order + specific +
            "Judge ONLY this candidate, not the whole product. Do not draw a mask or choose feature layers. "
            "Return JSON only, with exactly these fields: "
            f'{{"candidate_id":{candidate_id},"verdict":"insufficient_evidence","visibility":"insufficient",'
            '"reference_match":"unavailable","defect_type":"unknown","evidence":"short visible reason"}. '
            "verdict: defect_supported/normal_supported/insufficient_evidence; visibility: sufficient/insufficient; "
            "reference_match: matched/unmatched/unavailable. Do not invent numeric confidence.")


def parse_local(text, candidate_id, reference=False):
    fallback = {"parse_ok": False, "candidate_id": candidate_id, "verdict": "insufficient_evidence",
                "visibility": "insufficient", "reference_match": "unavailable", "evidence": "", "defect_type": "unknown"}
    try:
        raw = str(text).strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.S | re.I)
        value = json.loads(match.group(1) if match else raw)
        if set(value) != set(fallback) - {"parse_ok"}:
            return fallback
        if type(value["candidate_id"]) is not int or value["candidate_id"] != candidate_id:
            return fallback
        if value["verdict"] not in {"defect_supported", "normal_supported", "insufficient_evidence"}:
            return fallback
        if value["visibility"] not in {"sufficient", "insufficient"} or value["reference_match"] not in {"matched", "unmatched", "unavailable"}:
            return fallback
        if not reference and value["reference_match"] != "unavailable":
            return fallback
        if any(not isinstance(value[k], str) or not value[k].strip() or len(value[k]) > 400 for k in ("evidence", "defect_type")):
            return fallback
        return {"parse_ok": True, **value}
    except (ValueError, TypeError, KeyError):
        return fallback


def action(vote):
    if not vote.get("parse_ok") or vote.get("visibility") != "sufficient":
        return "keep"
    return {"defect_supported": "enhance", "normal_supported": "suppress"}.get(vote.get("verdict"), "keep")


def calibrate_local(base, supports, votes, mode="enhance", alpha=0.25, suppress_alpha=0.1, conflict_threshold=0.7):
    """No rectangle editing; exact identity outside eligible disjoint supports.

    Suppression is experimental: require a matched known-normal reference and
    veto it when the candidate contains a high Base score. Enhancement and
    suppression use separate strengths. Controls have no semantic decisions.
    """
    base, supports = np.asarray(base, np.float32), np.asarray(supports, bool)
    if base.ndim != 2 or not np.isfinite(base).all() or np.any((base < 0) | (base > 1)):
        raise ValueError("Invalid Base probabilities")
    if supports.shape != (len(votes), *base.shape) or np.any(supports.sum(axis=0) > 1):
        raise ValueError("Require one disjoint support per decision")
    if mode not in {"enhance", "suppress", "bidirectional", "control_enhance", "control_suppress"}:
        raise ValueError("Unknown local correction mode")
    if not 0 <= alpha <= 2 or not 0 <= suppress_alpha <= 2 or not 0 <= conflict_threshold <= 1:
        raise ValueError("Invalid correction limits")
    delta = np.zeros_like(base)
    for support, vote in zip(supports, votes):
        if not support.any():
            continue
        choice = action(vote)
        if mode == "control_enhance" or (choice == "enhance" and mode in {"enhance", "bidirectional"}):
            delta[support] = alpha
        elif mode == "control_suppress" or (choice == "suppress" and mode in {"suppress", "bidirectional"}
                and vote.get("reference_match") == "matched" and base[support].max() < conflict_threshold):
            delta[support] = -suppress_alpha
    out = base.copy()
    # Saturated endpoints stay saturated: clipping 1 before a positive residual
    # would otherwise LOWER it and violate the enhance-only guarantee.
    touched = (delta != 0) & (base > 0) & (base < 1)
    p = np.clip(base[touched], 1e-6, 1-1e-6)
    out[touched] = 1 / (1 + np.exp(-(np.log(p / (1-p)) + delta[touched])))
    return out


def component_counts(mask, candidate_union, base, updated, threshold=0.5):
    """Evaluation-only small/medium/large GT component diagnostics.

    Small <=0.1% image area, medium <=1%; detected = >=10% of the
    component exceeds the fixed threshold. These are diagnostic, not PRO.
    """
    import cv2
    n, labels = cv2.connectedComponents(np.asarray(mask, np.uint8), connectivity=8)
    counts = {f"{size}_{key}": 0 for size in ("small", "medium", "large")
              for key in ("count", "candidate_hit", "base_hit", "updated_hit", "new_miss")}
    for i in range(1, n):
        support = labels == i
        fraction = support.mean()
        size = "small" if fraction <= 0.001 else "medium" if fraction <= 0.01 else "large"
        hit = lambda values: float(np.mean(values[support])) >= 0.1
        b, u = hit(base >= threshold), hit(updated >= threshold)
        for key, value in (("count", 1), ("candidate_hit", hit(candidate_union)), ("base_hit", b),
                           ("updated_hit", u), ("new_miss", b and not u)):
            counts[f"{size}_{key}"] += int(value)
    return counts
