"""Language-free DINOv3 detector used by the single-tower experiment.

The detector keeps the DINOv3 backbone frozen by default and learns a compact
visual head.  Normal/anomaly prototypes are refined by image patch tokens and
matched directly in the DINO feature space, so no CLIP/text projection is
required.  The returned evidence dictionary intentionally follows the keys
consumed by :mod:`tools.mara_evidence`.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate

from tools.bottleneckAdapter import install_bottleneck_adapters_into_dino


def collate_anomaly_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate anomaly samples after normalizing mixed mask dtypes.

    Industrial datasets in this repository return boolean masks for anomalous
    test images but float zero masks for normal images.  Multiprocess PyTorch
    collation preallocates shared storage using the first tensor's dtype, so a
    mixed batch can fail before it reaches the training loop.  Normalizing here
    keeps the dataset implementations and all downstream losses consistent.
    """

    normalized = []
    for sample in batch:
        item = dict(sample)
        mask = item.get("mask")
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            item["mask"] = mask.to(dtype=torch.float32)
        normalized.append(item)
    return default_collate(normalized)


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Unexpected DINO block output type: {type(output)}")
    return output


def extract_dino_features(
    image: torch.Tensor,
    backbone: nn.Module,
    layers: Sequence[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Capture CLS/Patch tokens without importing the legacy CLIP utilities."""

    anchor = getattr(backbone, "norm", None) or getattr(backbone, "fc_norm", None)
    if anchor is None:
        raise RuntimeError("DINO backbone does not expose norm/fc_norm.")

    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layers:
        def make_hook(index: int):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                captured[index] = anchor(_extract_tensor(output))

            return hook

        handles.append(backbone.blocks[layer_index].register_forward_hook(make_hook(layer_index)))

    try:
        backbone(image)
    finally:
        for handle in handles:
            handle.remove()

    cls_tokens: List[torch.Tensor] = []
    patch_tokens: List[torch.Tensor] = []
    for layer_index in layers:
        if layer_index not in captured:
            raise RuntimeError(f"DINO hook did not capture layer {layer_index}.")
        tokens = captured[layer_index]
        # DINOv3 ViTs may use storage/register tokens between CLS and patches.
        patch_start = 1 + int(getattr(backbone, "n_storage_tokens", 4))
        patch = tokens[:, patch_start:, :]
        patch = (patch - patch.mean(dim=1, keepdim=True)) / (
            patch.std(dim=1, keepdim=True) + 1e-6
        )
        cls_tokens.append(tokens[:, 0:1, :])
        patch_tokens.append(patch)
    return cls_tokens, patch_tokens


@dataclass
class DinoSingleTowerConfig:
    visual_layers: Tuple[int, ...] = (5, 11, 17, 23)
    input_dim: int = 1024
    embed_dim: int = 256
    normal_prototypes: int = 4
    anomaly_prototypes: int = 8
    attention_heads: int = 8
    evidence_channels: int = 8
    temperature: float = 0.07
    topk_ratio: float = 0.01
    hfa_layers: Tuple[int, ...] = field(default_factory=tuple)
    hfa_bottleneck: int = 256

    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["visual_layers"] = list(self.visual_layers)
        output["hfa_layers"] = list(self.hfa_layers)
        return output

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DinoSingleTowerConfig":
        valid = cls.__dataclass_fields__.keys()
        kwargs = {key: value[key] for key in valid if key in value}
        if "visual_layers" in kwargs:
            kwargs["visual_layers"] = tuple(int(x) for x in kwargs["visual_layers"])
        if "hfa_layers" in kwargs:
            kwargs["hfa_layers"] = tuple(int(x) for x in kwargs["hfa_layers"])
        return cls(**kwargs)


class DinoLayerAdapter(nn.Module):
    """Project one DINO layer while preserving its spatial layout."""

    def __init__(self, input_dim: int, embed_dim: int, evidence_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, embed_dim)
        self.depthwise = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.spatial_norm = nn.GroupNorm(1, embed_dim)
        self.spatial_scale = nn.Parameter(torch.tensor(0.1))
        self.evidence_projector = nn.Sequential(
            nn.Conv2d(embed_dim, evidence_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, evidence_channels),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cls_token.dim() == 3:
            cls_token = cls_token[:, 0]
        patch = self.project(self.norm(patch_tokens))
        cls = self.project(self.norm(cls_token))

        side = int(math.sqrt(patch.shape[1]))
        if side * side != patch.shape[1]:
            raise ValueError(f"DINO patch count must form a square map, got {patch.shape[1]}.")
        patch_map = patch.transpose(1, 2).reshape(patch.shape[0], patch.shape[2], side, side)
        spatial = self.pointwise(F.gelu(self.depthwise(patch_map)))
        patch_map = patch_map + torch.tanh(self.spatial_scale) * self.spatial_norm(spatial)
        evidence = torch.tanh(self.evidence_projector(patch_map))
        patch = patch_map.flatten(2).transpose(1, 2)
        return cls, patch, evidence


class ImageConditionedPrototypeBlock(nn.Module):
    """Let visual prototypes read the current image without a text tower."""

    def __init__(self, embed_dim: int, attention_heads: int) -> None:
        super().__init__()
        if embed_dim % attention_heads != 0:
            raise ValueError("embed_dim must be divisible by attention_heads.")
        self.prototype_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.attention_scale = nn.Parameter(torch.tensor(-2.0))
        self.ffn_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, prototypes: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        batch_size = patches.shape[0]
        queries = prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(
            self.prototype_norm(queries),
            self.patch_norm(patches),
            self.patch_norm(patches),
            need_weights=False,
        )
        queries = queries + torch.sigmoid(self.attention_scale) * attended
        queries = queries + torch.sigmoid(self.ffn_scale) * self.ffn(self.ffn_norm(queries))
        return queries


def _multi_prototype_score(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Smooth maximum cosine similarity over a bank of prototypes."""

    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    similarity = torch.einsum("bnd,bkd->bnk", features, prototypes)
    normalizer = math.log(max(1, prototypes.shape[1]))
    return temperature * (torch.logsumexp(similarity / temperature, dim=-1) - normalizer)


class DinoVisualPrototypeHead(nn.Module):
    """Multi-layer normal/anomaly prototype head in the DINO feature space."""

    def __init__(self, config: DinoSingleTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_adapters = nn.ModuleList(
            [
                DinoLayerAdapter(
                    input_dim=config.input_dim,
                    embed_dim=config.embed_dim,
                    evidence_channels=config.evidence_channels,
                )
                for _ in config.visual_layers
            ]
        )
        self.prototype_blocks = nn.ModuleList(
            [
                ImageConditionedPrototypeBlock(config.embed_dim, config.attention_heads)
                for _ in config.visual_layers
            ]
        )
        self.normal_bank = nn.Parameter(
            torch.empty(config.normal_prototypes, config.embed_dim)
        )
        self.anomaly_bank = nn.Parameter(
            torch.empty(config.anomaly_prototypes, config.embed_dim)
        )
        nn.init.trunc_normal_(self.normal_bank, std=0.02)
        nn.init.trunc_normal_(self.anomaly_bank, std=0.02)

        self.layer_weights = nn.Parameter(torch.zeros(len(config.visual_layers)))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.topk_weight_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def prototype_count(self) -> int:
        return self.config.normal_prototypes + self.config.anomaly_prototypes

    def prototype_regularization(self) -> torch.Tensor:
        """Encourage diversity within banks and separation across banks."""

        normal = F.normalize(self.normal_bank, dim=-1)
        anomaly = F.normalize(self.anomaly_bank, dim=-1)

        def off_diagonal_square(bank: torch.Tensor) -> torch.Tensor:
            if bank.shape[0] <= 1:
                return bank.new_zeros(())
            cosine = bank @ bank.t()
            mask = ~torch.eye(bank.shape[0], dtype=torch.bool, device=bank.device)
            return cosine[mask].square().mean()

        diversity = 0.5 * (off_diagonal_square(normal) + off_diagonal_square(anomaly))
        cross = normal @ anomaly.t()
        separation = F.relu(cross - 0.20).mean()
        return diversity + separation

    def forward(
        self,
        cls_tokens: Sequence[torch.Tensor],
        patch_tokens: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        if len(cls_tokens) != len(self.layer_adapters) or len(patch_tokens) != len(self.layer_adapters):
            raise ValueError(
                f"Expected {len(self.layer_adapters)} DINO layers, "
                f"got cls={len(cls_tokens)}, patch={len(patch_tokens)}."
            )

        static_prototypes = torch.cat([self.normal_bank, self.anomaly_bank], dim=0)
        layer_logits_lowres: List[torch.Tensor] = []
        layer_logits_fullres: List[torch.Tensor] = []
        topk_margins: List[torch.Tensor] = []
        global_margins: List[torch.Tensor] = []
        evidence: Dict[str, List[torch.Tensor]] = {
            "cross_prob_layers": [],
            "cross_margin_layers": [],
            "awareness_layers": [],
            "normal_similarity_layers": [],
            "anomaly_similarity_layers": [],
            "global_margin_layers": [],
            "feature_layers": [],
        }

        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        for adapter, prototype_block, cls_token, patches in zip(
            self.layer_adapters,
            self.prototype_blocks,
            cls_tokens,
            patch_tokens,
        ):
            cls_feature, patch_feature, compact_feature = adapter(cls_token, patches)
            conditioned = prototype_block(static_prototypes, patch_feature)
            normal = conditioned[:, : self.config.normal_prototypes]
            anomaly = conditioned[:, self.config.normal_prototypes :]

            normal_score = _multi_prototype_score(
                patch_feature, normal, self.config.temperature
            )
            anomaly_score = _multi_prototype_score(
                patch_feature, anomaly, self.config.temperature
            )
            token_logits = scale * torch.stack([normal_score, anomaly_score], dim=-1)

            side = int(math.sqrt(patch_feature.shape[1]))
            lowres_logits = token_logits.transpose(1, 2).reshape(
                patch_feature.shape[0], 2, side, side
            )
            fullres_logits = F.interpolate(
                lowres_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            layer_logits_lowres.append(lowres_logits)
            layer_logits_fullres.append(fullres_logits)

            cls_normal = _multi_prototype_score(
                cls_feature.unsqueeze(1), normal, self.config.temperature
            )[:, 0]
            cls_anomaly = _multi_prototype_score(
                cls_feature.unsqueeze(1), anomaly, self.config.temperature
            )[:, 0]
            cls_margin = scale * (cls_anomaly - cls_normal)
            global_margins.append(cls_margin)

            margin = token_logits[..., 1] - token_logits[..., 0]
            topk = max(1, int(round(margin.shape[1] * self.config.topk_ratio)))
            topk_margins.append(margin.topk(topk, dim=1).values.mean(dim=1))

            cls_unit = F.normalize(cls_feature, dim=-1).unsqueeze(1)
            patch_unit = F.normalize(patch_feature, dim=-1)
            deviation = 1.0 - (patch_unit * cls_unit).sum(dim=-1)
            deviation_mean = deviation.mean(dim=1, keepdim=True)
            deviation_std = deviation.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
            awareness = torch.sigmoid((deviation - deviation_mean) / deviation_std)

            evidence["cross_prob_layers"].append(
                torch.softmax(lowres_logits, dim=1)[:, 1]
            )
            evidence["cross_margin_layers"].append(margin.reshape(-1, side, side))
            evidence["awareness_layers"].append(awareness.reshape(-1, side, side))
            evidence["normal_similarity_layers"].append(
                normal_score.reshape(-1, side, side)
            )
            evidence["anomaly_similarity_layers"].append(
                anomaly_score.reshape(-1, side, side)
            )
            evidence["global_margin_layers"].append(cls_margin)
            evidence["feature_layers"].append(compact_feature)

        weights = torch.softmax(self.layer_weights, dim=0)
        fused_logits = sum(
            weight * layer_logits
            for weight, layer_logits in zip(weights, layer_logits_fullres)
        )
        base_prob = torch.softmax(fused_logits, dim=1)

        cls_margin = sum(
            weight * margin for weight, margin in zip(weights, global_margins)
        )
        topk_margin = sum(
            weight * margin for weight, margin in zip(weights, topk_margins)
        )
        global_margin = cls_margin + torch.sigmoid(self.topk_weight_logit) * topk_margin
        global_logits = torch.stack([-0.5 * global_margin, 0.5 * global_margin], dim=1)

        return {
            "prob": base_prob,
            "pixel_logits": fused_logits,
            "global_logits": global_logits,
            "layer_logits": layer_logits_lowres,
            "layer_weights": weights,
            "evidence": evidence,
            "prototype_regularization": self.prototype_regularization(),
        }


class DinoSingleTowerDetector(nn.Module):
    """Frozen DINOv3 plus the trainable visual prototype head."""

    def __init__(self, backbone: nn.Module, config: DinoSingleTowerConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        if config.hfa_layers:
            install_bottleneck_adapters_into_dino(
                self.backbone,
                layers=config.hfa_layers,
                dim=config.input_dim,
                bottleneck=config.hfa_bottleneck,
            )
        self.head = DinoVisualPrototypeHead(config)

    @property
    def dino_adapters(self) -> nn.Module | None:
        return getattr(self.backbone, "_peft_adapters", None)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        has_trainable_backbone_adapter = self.dino_adapters is not None and any(
            parameter.requires_grad for parameter in self.dino_adapters.parameters()
        )
        context = nullcontext() if has_trainable_backbone_adapter else torch.no_grad()
        with context:
            cls_tokens, patch_tokens = extract_dino_features(
                image=image,
                backbone=self.backbone,
                layers=self.config.visual_layers,
            )
        return self.head(cls_tokens, patch_tokens, output_size=tuple(image.shape[-2:]))

    def checkpoint_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "base_arch": "dino_single",
            "visual_backbone": "dino",
            "visual_layers": list(self.config.visual_layers),
            "hfa_setting": "custom" if self.config.hfa_layers else "none",
            "hfa_layers": list(self.config.hfa_layers),
            "hfa_bottleneck": self.config.hfa_bottleneck,
            "dino_single_config": self.config.to_dict(),
            "dino_single_head": self.head.state_dict(),
        }
        if self.dino_adapters is not None:
            fields["dino_adapters"] = self.dino_adapters.state_dict()
        return fields

    def load_checkpoint_fields(self, payload: Mapping[str, Any], strict: bool = True) -> None:
        self.head.load_state_dict(payload["dino_single_head"], strict=strict)
        if self.dino_adapters is not None:
            if "dino_adapters" not in payload:
                raise KeyError("Single-tower checkpoint is missing dino_adapters.")
            self.dino_adapters.load_state_dict(payload["dino_adapters"], strict=strict)


def create_dino_single_tower(
    config: DinoSingleTowerConfig,
    repo_dir: str,
    model_name: str,
    weights: str,
    device: torch.device,
) -> DinoSingleTowerDetector:
    backbone = torch.hub.load(
        repo_dir,
        model_name,
        source="local",
        weights=weights,
    )
    detector = DinoSingleTowerDetector(backbone=backbone, config=config)
    return detector.to(device)


def config_from_checkpoint(payload: Mapping[str, Any]) -> DinoSingleTowerConfig:
    if payload.get("base_arch") != "dino_single":
        raise ValueError("Checkpoint is not a DINO single-tower checkpoint.")
    config = payload.get("dino_single_config")
    if not isinstance(config, Mapping):
        raise KeyError("Checkpoint is missing dino_single_config.")
    return DinoSingleTowerConfig.from_dict(config)


def forward_dino_single_batch(
    detector: DinoSingleTowerDetector,
    image_info: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    image = image_info["image"].to(device, non_blocking=True)
    mask = (image_info["mask"].to(device, non_blocking=True) > 0.5).float()
    output = detector(image)
    return mask, output["prob"], output["global_logits"], output["evidence"]
