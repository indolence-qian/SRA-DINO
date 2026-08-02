from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def _as_map(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.unsqueeze(1)
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        return tensor
    raise ValueError(f"Expected a BxHxW or Bx1xHxW evidence map, got {tuple(tensor.shape)}.")


def _resize_map(tensor: torch.Tensor, map_size: int) -> torch.Tensor:
    if tensor.shape[-2:] == (map_size, map_size):
        return tensor
    return F.interpolate(tensor, size=(map_size, map_size), mode="bilinear", align_corners=False)


def _stack_map_layers(
    evidence: Optional[Dict[str, Any]],
    key: str,
    fallback: torch.Tensor,
    num_layers: int,
    map_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values: List[torch.Tensor] = [] if evidence is None else list(evidence.get(key, []))
    layers: List[torch.Tensor] = []
    for value in values[:num_layers]:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = _as_map(value.to(device=device, dtype=dtype))
        layers.append(_resize_map(value, map_size))

    fallback = _resize_map(_as_map(fallback.to(device=device, dtype=dtype)), map_size)
    if not layers:
        layers.append(fallback)
    while len(layers) < num_layers:
        layers.append(layers[-1])
    return torch.cat(layers, dim=1)


def _stack_global_layers(
    evidence: Optional[Dict[str, Any]],
    key: str,
    batch_size: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values: List[torch.Tensor] = [] if evidence is None else list(evidence.get(key, []))
    layers: List[torch.Tensor] = []
    for value in values[:num_layers]:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=device, dtype=dtype).reshape(batch_size, -1)[:, 0]
        layers.append(value)
    if not layers:
        layers.append(torch.zeros(batch_size, device=device, dtype=dtype))
    while len(layers) < num_layers:
        layers.append(layers[-1])
    return torch.stack(layers, dim=1)


def _normalize_spatial_margin(margin: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = margin.mean(dim=(-2, -1), keepdim=True)
    std = margin.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    return torch.tanh((margin - mean) / (2.0 * std))


def build_mara_evidence(
    evidence: Optional[Dict[str, Any]],
    fallback_prob: torch.Tensor,
    num_layers: int,
    map_size: int,
    include_full_resolution_oracle: bool = False,
) -> Dict[str, Any]:
    """Pack frozen stage-one evidence into compact MARA decision tensors.

    The returned tensors stay on the current device. Raw 768-D patch features
    are intentionally excluded in P1; this bank only contains compact score
    maps and per-layer global margins.
    """
    device = fallback_prob.device
    dtype = fallback_prob.dtype
    batch_size = fallback_prob.shape[0]
    fallback_anomaly = fallback_prob[:, 1:2].detach()
    zero_fallback = torch.zeros_like(fallback_anomaly)

    cross_prob = _stack_map_layers(
        evidence, "cross_prob_layers", fallback_anomaly, num_layers, map_size, device, dtype
    )
    cross_margin = _stack_map_layers(
        evidence, "cross_margin_layers", zero_fallback, num_layers, map_size, device, dtype
    )
    awareness = _stack_map_layers(
        evidence, "awareness_layers", fallback_anomaly, num_layers, map_size, device, dtype
    )
    normal_similarity = _stack_map_layers(
        evidence, "normal_similarity_layers", zero_fallback, num_layers, map_size, device, dtype
    )
    anomaly_similarity = _stack_map_layers(
        evidence, "anomaly_similarity_layers", zero_fallback, num_layers, map_size, device, dtype
    )
    global_margin = _stack_global_layers(
        evidence, "global_margin_layers", batch_size, num_layers, device, dtype
    )

    cross_margin = _normalize_spatial_margin(cross_margin)
    global_margin = torch.tanh(global_margin / 10.0)
    cross_disagreement = cross_prob.std(dim=1, keepdim=True, unbiased=False)
    awareness_disagreement = awareness.std(dim=1, keepdim=True, unbiased=False)

    oracle_maps = torch.cat([cross_prob, awareness], dim=1)
    if include_full_resolution_oracle:
        oracle_size = int(fallback_prob.shape[-1])
        oracle_cross = _stack_map_layers(
            evidence,
            "cross_prob_layers",
            fallback_anomaly,
            num_layers,
            oracle_size,
            device,
            dtype,
        )
        oracle_awareness = _stack_map_layers(
            evidence,
            "awareness_layers",
            fallback_anomaly,
            num_layers,
            oracle_size,
            device,
            dtype,
        )
        oracle_maps = torch.cat([oracle_cross, oracle_awareness], dim=1)

    extra_maps = torch.cat(
        [
            cross_margin,
            awareness,
            normal_similarity,
            anomaly_similarity,
            cross_disagreement,
            awareness_disagreement,
        ],
        dim=1,
    ).detach()

    return {
        "layer_maps": cross_prob.detach(),
        "extra_maps": extra_maps,
        "global_evidence": global_margin.detach(),
        "oracle_maps": oracle_maps.detach(),
        "oracle_names": [
            *[f"cross_l{idx}" for idx in range(num_layers)],
            *[f"awareness_l{idx}" for idx in range(num_layers)],
        ],
    }
