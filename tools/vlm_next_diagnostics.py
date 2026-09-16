"""CPU-only, label-isolated helpers for the post-R2 diagnostic experiment."""
import json
import re

import numpy as np

from tools.local_vlm_protocol import parse_repaired
from tools.local_vlm_review import action, calibrate_local
from tools.vlm_review import digest

ARMS = ('A_no_reference', 'B_retrieved_reference', 'C_shuffled_reference')
OBSERVATIONS = ('query_observation', 'reference_observation', 'local_difference')


def audit_prompt(category, candidate_id, reference):
    role = (
        'Exactly THREE images are supplied in order. Image 1: QUERY CONTEXT. '
        'Image 2: QUERY DETAIL of the same candidate. Image 3: NORMAL TRAIN REFERENCE DETAIL. '
        'Image 3 is present, but correspondence is NOT guaranteed. Describe its actual content separately. '
        'Use reference_match=matched only if the local material, part and viewpoint correspond; '
        'use unmatched if different; use unavailable only if correspondence cannot be assessed, and explain why. '
        if reference else
        'Exactly TWO images are supplied in order. Image 1: QUERY CONTEXT. Image 2: QUERY DETAIL. '
        'No third image is supplied. Set reference_match=unavailable and reference_observation=not_provided. '
    )
    return (
        f'Inspect industrial category {category}, candidate {candidate_id}. ' + role +
        'Inspect the small candidate near the center of Image 2, not the overall object. '
        'First describe Image 2 in query_observation, then Image 3 in reference_observation if present. '
        'In local_difference describe corresponding edges, texture or material continuity, or why comparison is impossible. '
        'A normal-looking component may still have a tiny crack, missing material or contamination. '
        'Do not invent defects or dismiss a region merely because it is small. '
        'Do not infer a defect just because the reference is different. '
        'Visibility is insufficient when query detail is blurred or unresolved. Clear but ambiguous normality '
        'means sufficient visibility with insufficient_evidence. Do not force a verdict or a match. '
        'Return JSON only with exactly NINE fields: '
        f'candidate_id: integer {candidate_id}; '
        'verdict: defect_supported/normal_supported/insufficient_evidence; '
        'visibility: sufficient/insufficient; reference_match: matched/unmatched/unavailable; '
        'defect_type: short string (unknown if undetermined); evidence: brief justification; '
        'query_observation: short string; reference_observation: short string; local_difference: short string. '
        'All strings must be nonempty. Prefer one short sentence per observation. No confidence or masks.'
    )


def parse_audit(raw, candidate_id, reference):
    """Strict JSON core; semantic contradictions are reported, not silently relabeled."""
    observations, warnings = {}, []
    try:
        def unique(pairs):
            obj = {}
            for k, v in pairs:
                if k in obj:
                    raise ValueError('duplicate_key')
                obj[k] = v
            return obj
        text = str(raw).strip()
        if len(text) > 65536:
            raise ValueError('response_too_large')
        fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.S | re.I)
        obj = json.loads(fenced.group(1) if fenced else text, object_pairs_hook=unique)
        if not isinstance(obj, dict):
            raise ValueError('not_object')
        for name in OBSERVATIONS:
            value = obj.pop(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError('missing_observation')
            observations[name] = value
        decision, audit = parse_repaired(json.dumps(obj), candidate_id, reference)
        if not reference and observations['reference_observation'] != 'not_provided':
            warnings.append('describes_reference_without_image')
        if reference and decision['reference_match'] == 'unavailable':
            warnings.append('provided_reference_not_assessed')
        if reference and re.search(r'not.provided|no reference', observations['reference_observation'], re.I):
            warnings.append('denies_provided_reference')
        query_text = observations['query_observation'] + ' ' + decision['evidence']
        if decision['visibility'] == 'sufficient' and re.search(
                r'out of focus|too blurr|unresolved|no discernible|cannot be resolved', query_text, re.I):
            warnings.append('visibility_text_conflict')
    except (ValueError, KeyError, TypeError) as exc:
        decision, audit = parse_repaired('', candidate_id, reference)
        audit['parse_errors'] = [str(exc)]
    return decision, dict(**audit, observations=observations, semantic_warnings=warnings)


def paired_selection(pool, count, seed):
    """GT-free area/category round robin. Shuffle references are distinct normal crops.

    Donors must be another query's distinct crop of the SAME normal source.
    That source was already verified not to be the current query by R2 export.
    Shuffling locations within it avoids introducing a query-as-reference leak.
    This is a correspondence control, not a guaranteed mismatched/anomaly label.
    """
    eligible = []
    for item in pool:
        donors = [x for x in pool if x['dataset'] == item['dataset'] and x['category'] == item['category']
                  and x['sample_id'] != item['sample_id']
                  and x['reference_sha256'] != item['reference_sha256']
                  and x['reference_source_sha256'] == item['reference_source_sha256']]
        if donors:
            donor = min(donors, key=lambda x: (abs(np.log(max(1,x['area'])/max(1,item['area']))),
                                               digest([seed, item['key'], x['key']])))
            eligible.append(dict(item, shuffled_reference=donor['reference'], donor_key=donor['key'],
                                 donor_source_sha256=donor['reference_source_sha256']))
    buckets = {}
    for x in eligible:
        area_bin = 0 if x['area'] <= 16 else 1 if x['area'] <= 64 else 2 if x['area'] <= 256 else 3
        buckets.setdefault((x['dataset'], x['category'], area_bin), []).append(x)
    for values in buckets.values():
        values.sort(key=lambda x: digest([seed, x['key']]), reverse=True)
    target = len(eligible) if count == 0 else min(count, len(eligible))
    selected = []
    while len(selected) < target:
        for bucket in sorted(buckets):
            if buckets[bucket]:
                selected.append(buckets[bucket].pop())
                if len(selected) == target:
                    break
    return selected, len(eligible)


def correction(base, supports, votes, gt, mode, alpha):
    """Called ONLY from offline evaluation. GT choices never become VLM caches."""
    if mode == 'base':
        return base.copy()
    if mode == 'vlm_cached':
        return calibrate_local(base, supports, votes, 'enhance', alpha)
    if mode == 'all_candidates':
        return calibrate_local(base, supports, votes, 'control_enhance', alpha)
    if mode == 'gt_candidate_DIAGNOSTIC_ONLY':
        flags = [bool(gt[s].any()) for s in supports]
    elif mode == 'gt_pixel_DIAGNOSTIC_ONLY':
        supports = supports & gt[None]
        flags = [True] * len(supports)
    else:
        raise ValueError('Unknown diagnostic correction')
    oracle_votes = [dict(parse_ok=True, visibility='sufficient', verdict='defect_supported' if f else
                        'insufficient_evidence') for f in flags]
    return calibrate_local(base, supports, oracle_votes, 'enhance', alpha)


def exact_pro(masks, maps, max_fpr=.3):
    """Equal-component weighted ROC integral, all distinct score thresholds.

    Unlike the historical sampled PRO, no per-mode min/max threshold grid or
    arbitrary duplicate-FPR point is used. Scores tied at a threshold are grouped.
    This is a separately named diagnostic metric, NOT a silent legacy replacement.
    """
    import cv2
    from sklearn.metrics import roc_curve, auc
    masks = np.asarray(masks, bool)
    maps = np.asarray(maps)
    if masks.shape != maps.shape or masks.ndim != 3 or not np.isfinite(maps).all():
        raise ValueError('Expected finite aligned [N,H,W] arrays')
    if not 0 < max_fpr <= 1:
        raise ValueError('Invalid max_fpr')
    if not masks.any() or masks.all():
        return None
    weights = np.ones(masks.shape, np.float64)
    for i, mask in enumerate(masks):
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        counts = np.bincount(labels.ravel(), minlength=n)
        region_weights = np.zeros(n, np.float64)
        region_weights[1:] = 1.0 / counts[1:]
        weights[i][mask] = region_weights[labels[mask]]
    fpr, pro, _ = roc_curve(masks.ravel(), maps.ravel(), sample_weight=weights.ravel(), drop_intermediate=True)
    # Keep both endpoints of vertical segments (zero area), avoiding a spurious
    # diagonal formed by collapsing duplicate FPR coordinates to their maxima.
    end = int(np.searchsorted(fpr, max_fpr, side='right'))
    x, y = fpr[:end], pro[:end]
    if x[-1] < max_fpr:
        value = y[-1] + (pro[end]-y[-1]) * (max_fpr-x[-1]) / (fpr[end]-x[-1])
        x, y = np.r_[x, max_fpr], np.r_[y, value]
    return float(auc(x / max_fpr, y))


def map_metrics(masks, maps, max_fpr=.3):
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    masks, maps = np.asarray(masks, bool), np.asarray(maps)
    labels = masks.reshape(len(masks), -1).any(axis=1)
    image_scores = maps.reshape(len(maps), -1).max(axis=1)
    precision, recall, _ = precision_recall_curve(masks.ravel(), maps.ravel())
    return dict(PRO_exact=exact_pro(masks, maps, max_fpr),
                P_AUROC=float(roc_auc_score(masks.ravel(), maps.ravel())) if masks.any() and not masks.all() else None,
                I_AUROC=float(roc_auc_score(labels, image_scores)) if labels.any() and not labels.all() else None,
                F1_best=float(np.max(2*precision*recall / np.maximum(precision+recall, 1e-12))))
