"""Frozen text readouts and explicit evaluation for the second ablation.

This module deliberately does not change any sealed first-experiment code.
"""
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from tools.residual_text import paired_directions
from tools.residual_text_runtime import GENERIC_PROMPT

MODES = {"T1": "single_delta", "T2": "normal_abnormal", "T3": "mean_paired_delta"}
EVALUATION = "manifest_image_labels__nonzero_native_mask__maxpool_down_nearest_up_v1"


def validate_prompts(prompts):
    if not isinstance(prompts, list) or len(prompts) < 2 or prompts[0] != GENERIC_PROMPT:
        raise ValueError("Prompt bank must start with the original GENERIC_PROMPT and contain >=2 pairs")
    ids, pairs = set(), set()
    for p in prompts:
        if (not isinstance(p, dict) or set(p) != {"id", "normal", "abnormal"}
                or not all(isinstance(v, str) and v.strip() == v and v for v in p.values())
                or p["id"] in ids or p["normal"] == p["abnormal"]
                or (p["normal"], p["abnormal"]) in pairs):
            raise ValueError("Need unique IDs and distinct, nonempty paired state descriptions")
        ids.add(p["id"]); pairs.add((p["normal"], p["abnormal"]))
    return prompts


def text_geometry(states):
    x = torch.as_tensor(states, dtype=torch.float32)
    if (x.ndim != 4 or x.shape[1] != 2 or min(x.shape) < 1
            or not torch.isfinite(x).all() or (x.norm(dim=-1) < 1e-6).any()):
        raise ValueError("Expected finite nonzero [pairs,2,templates,text_dim] states")
    x = F.normalize(x, dim=-1)
    directions = paired_directions(x[:, 0], x[:, 1])
    normal, abnormal = x[0, 0].mean(0), x[0, 1].mean(0)
    return x, directions, normal, abnormal


class TextMatcher(nn.Module):
    """Matched bias-free MLPs; only frozen text readout changes between arms.

    T2 averages the two state logits over templates, then uses their log odds.
    T3 averages signed similarities to individually normalized paired deltas.
    No z normalization, absolute delta, top-k, max-prompt or new trainable anchor.
    """
    def __init__(self, layers, dim, states, mode, hidden=256):
        super().__init__()
        if mode not in MODES.values():
            raise ValueError("Unknown text readout")
        x, directions, normal, abnormal = text_geometry(states)
        self.mode = mode
        self.register_buffer("directions", directions if mode == "mean_paired_delta" else directions[:1])
        self.register_buffer("normal", normal)
        self.register_buffer("abnormal", abnormal)
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.GELU(),
                          nn.Linear(hidden, x.shape[-1], bias=False)) for _ in range(layers)])

    def state_logits(self, z):
        return torch.stack((z @ self.normal, z @ self.abnormal), -1)

    def readout(self, z):
        if self.mode == "normal_abnormal":
            logits = self.state_logits(z)
            return (logits[..., 1] - logits[..., 0])[..., None]
        # Averaging pair scores preserves their sign and magnitude; normalizing
        # the mean again would implement a different prompt ensemble.
        return (z @ self.directions.T).mean(-1, keepdim=True)

    def forward(self, residual):
        scores, amplitude = [], []
        for layer, projector in enumerate(self.projectors):
            z = projector(residual[:, :, layer].permute(0, 1, 3, 4, 2))
            scores.append(self.readout(z).permute(0, 1, 4, 2, 3))
            amplitude.append(z.norm(dim=-1))
        return torch.stack(scores).mean(0), torch.stack(amplitude, 2)


def bank_audit(states):
    x, directions, normal, abnormal = text_geometry(states)
    delta = abnormal - normal
    average = directions.mean(0)
    if average.norm() < 1e-6:
        raise ValueError("Paired prompt directions cancel; do not use a zero ensemble")
    return dict(pairs=len(x), templates=x.shape[2], text_dim=x.shape[-1],
        ordinary_margin_norm=float(delta.norm()),
        single_equivalence_max_error=float((delta - delta.norm() * directions[0]).abs().max()),
        mean_pair_direction_norm=float(average.norm()),
        single_vs_ensemble_cosine=float(F.cosine_similarity(directions[0], average, dim=0)),
        pair_cosines=(directions @ directions.T).tolist(),
        equivalence="T2(z)=||mean_t(a_t-n_t)|| * T1(z), for the same projector z. "
                    "Independent optimization can differ; no new representational capacity is proved.")


def evaluation_truth(row, size):
    """Nonzero annotation IDs/colors; preserve tiny positives when reducing size.

    Called only after target predictions lock. A palette is read as annotation
    indices; alpha is ignored. Original values/counts are retained for auditing.
    """
    width, height = row["native_size"]
    if not row["mask_path"]:
        return np.zeros((size, size), bool), dict(id=row["id"], image_label=0,
            native_positive=0, resized_positive=0, mode=None, unique_values=[])
    with Image.open(row["mask_path"]) as image:
        if image.size != (width, height):
            raise ValueError(f"Mask/native geometry differs: {row['mask_path']}")
        if image.mode not in ("1", "L", "P", "I", "I;16", "I;16B", "RGB", "RGBA"):
            raise ValueError(f"Unsupported mask mode {image.mode}: {row['mask_path']}")
        mode, raw = image.mode, np.asarray(image)
    values = np.unique(raw)
    mask = (raw[..., :3] > 0).any(-1) if raw.ndim == 3 else raw > 0
    tensor = torch.from_numpy(mask.copy()).float()[None, None]
    reduced = (min(height, size), min(width, size))
    if reduced != (height, width):
        tensor = F.adaptive_max_pool2d(tensor, reduced)
    if reduced != (size, size):
        tensor = F.interpolate(tensor, (size, size), mode="nearest")
    result = tensor[0, 0].numpy().astype(bool)
    audit = dict(id=row["id"], image_label=1, native_positive=int(mask.sum()),
                 resized_positive=int(result.sum()), mode=mode,
                 unique_values=values[:32].tolist(), unique_count=len(values))
    return result, audit


def corrected_metrics(masks, maps, labels, threshold):
    from residual_text_experiment import metric_row
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, bool)
    if labels.shape != (len(masks),):
        raise ValueError("Explicit dataset image labels required")
    point = metric_row(masks, maps, threshold)
    scores = maps.reshape(len(maps), -1).max(1)
    point.update(I_AUROC=float(roc_auc_score(labels, scores)) if labels.any() and not labels.all() else None,
        I_AP=float(average_precision_score(labels, scores)) if labels.any() else None,
        abnormal_images=int(labels.sum()), image_label_source="dataset_manifest_mask_path_presence",
        evaluation_protocol=EVALUATION)
    return point
