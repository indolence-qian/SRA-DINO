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


def _stack_feature_layers(
    evidence: Optional[Dict[str, Any]],
    key: str,
    batch_size: int,
    channels_per_layer: int,
    num_layers: int,
    map_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels_per_layer <= 0:
        return torch.zeros(batch_size, 0, map_size, map_size, device=device, dtype=dtype)

    values: List[torch.Tensor] = [] if evidence is None else list(evidence.get(key, []))
    layers: List[torch.Tensor] = []
    for value in values[:num_layers]:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=device, dtype=dtype)
        if value.dim() != 4:
            raise ValueError(
                f"Expected BxCxHxW feature evidence for {key}, got {tuple(value.shape)}."
            )
        if value.shape[1] != channels_per_layer:
            raise ValueError(
                f"Expected {channels_per_layer} feature channels per layer, "
                f"got {value.shape[1]}."
            )
        layers.append(_resize_map(value, map_size))

    zero = torch.zeros(
        batch_size,
        channels_per_layer,
        map_size,
        map_size,
        device=device,
        dtype=dtype,
    )
    if not layers:
        layers.append(zero)
    while len(layers) < num_layers:
        layers.append(zero)
    return torch.cat(layers, dim=1)


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
    feature_channels_per_layer: int = 0,
    semantic_spatial_channels: int = 0,
    semantic_global_dim: int = 0,
) -> Dict[str, Any]:
    """Pack frozen stage-one evidence into compact MARA decision tensors.

    The returned tensors stay on the current device. The legacy CLIP/DINO path
    uses only compact score maps. A DINO single-tower checkpoint can additionally
    provide a small learned projection of every DINO feature layer.
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
    feature_maps = _stack_feature_layers(
        evidence,
        "feature_layers",
        batch_size,
        feature_channels_per_layer,
        num_layers,
        map_size,
        device,
        dtype,
    )

    semantic_maps = torch.zeros(
        batch_size, 0, map_size, map_size, device=device, dtype=dtype
    )
    if semantic_spatial_channels > 0:
        semantic_keys = (
            "semantic_prior_map",
            "semantic_confidence_map",
            "semantic_disagreement_map",
        )
        if semantic_spatial_channels != len(semantic_keys):
            raise ValueError(
                f"Expected {len(semantic_keys)} semantic spatial channels, "
                f"got {semantic_spatial_channels}."
            )
        values = []
        for key in semantic_keys:
            value = None if evidence is None else evidence.get(key)
            if value is None:
                value = torch.zeros_like(fallback_anomaly)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = _as_map(value.to(device=device, dtype=dtype))
            values.append(_resize_map(value, map_size))
        semantic_maps = torch.cat(values, dim=1)

    semantic_global = torch.zeros(
        batch_size, 0, device=device, dtype=dtype
    )
    if semantic_global_dim > 0:
        value = None if evidence is None else evidence.get("semantic_global")
        if value is None:
            semantic_global = torch.zeros(
                batch_size, semantic_global_dim, device=device, dtype=dtype
            )
        else:
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            semantic_global = value.to(device=device, dtype=dtype).reshape(batch_size, -1)
            if semantic_global.shape[1] != semantic_global_dim:
                raise ValueError(
                    f"Expected {semantic_global_dim} semantic global values, "
                    f"got {semantic_global.shape[1]}."
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
            feature_maps,
            semantic_maps,
        ],
        dim=1,
    ).detach()

    return {
        "layer_maps": cross_prob.detach(),
        "extra_maps": extra_maps,
        "global_evidence": torch.cat([global_margin, semantic_global], dim=1).detach(),
        "oracle_maps": oracle_maps.detach(),
        "oracle_names": [
            *[f"cross_l{idx}" for idx in range(num_layers)],
            *[f"awareness_l{idx}" for idx in range(num_layers)],
        ],
    }
