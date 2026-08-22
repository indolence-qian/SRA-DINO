import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CLIP.tokenizer import tokenize


ANCHOR_LABELS: Tuple[str, str] = ("normal", "anomaly")


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_text_list(value: Any, field: str) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        raise ValueError(f"Semantic anchor `{field}` must be a string or a sequence of strings.")
    values = [item.strip() for item in values if item.strip()]
    if not values:
        raise ValueError(f"Semantic anchor `{field}` is empty.")
    return values


def load_external_semantic_anchors(path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load the legacy Gemini embedding pair for audit or old checkpoints only.

    P0/P1 training no longer compares CLIP features with these coordinates. The
    external vectors remain loadable so existing assets and checkpoints stay
    auditable and backward compatible.
    """
    anchor_path = Path(path)
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Semantic anchor file not found: {anchor_path}.")
    payload = _torch_load(str(anchor_path))
    if not isinstance(payload, Mapping) or "anchors" not in payload:
        raise ValueError("Semantic anchor payload must be a mapping containing `anchors`.")
    anchors = torch.as_tensor(payload["anchors"]).detach().float()
    if anchors.dim() != 2:
        raise ValueError(f"Semantic anchors must be 2D, got shape={tuple(anchors.shape)}.")
    if anchors.shape[0] != 2 and anchors.shape[1] == 2:
        anchors = anchors.transpose(0, 1)
    if anchors.shape[0] != 2 or not torch.isfinite(anchors).all():
        raise ValueError("Semantic anchors must contain two finite normal/anomaly rows.")
    metadata = {str(key): value for key, value in payload.items() if key != "anchors"}
    anchors = F.normalize(anchors, dim=-1)
    metadata["path"] = str(anchor_path.resolve())
    metadata["dimension"] = int(anchors.shape[1])
    metadata["legacy_embedding_used_for_training"] = False
    return anchors, metadata


def load_semantic_anchor_descriptions(path: str) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Load text banks while treating external embeddings as provenance only."""
    anchor_path = Path(path)
    if not anchor_path.is_file():
        raise FileNotFoundError(
            f"Semantic anchor file not found: {anchor_path}. "
            "Generate it with tools/build_gemini_semantic_anchors.py first."
        )
    payload = _torch_load(str(anchor_path))
    if not isinstance(payload, Mapping):
        raise ValueError("Semantic anchor payload must be an auditable mapping.")

    merged: Dict[str, Any] = dict(payload)
    merged.pop("anchors", None)
    sidecar_path = anchor_path.with_suffix(".json")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, Mapping):
            raise ValueError(f"Semantic anchor sidecar must contain a JSON object: {sidecar_path}")
        merged.update(sidecar)

    descriptions = merged.get("descriptions")
    if not isinstance(descriptions, Mapping):
        raise ValueError("Semantic anchor payload must contain normal/anomaly `descriptions`.")
    banks = merged.get("description_banks", descriptions)
    if not isinstance(banks, Mapping):
        raise ValueError("`description_banks` must contain normal/anomaly text lists.")
    text_banks = {
        "normal": _as_text_list(banks.get("normal", descriptions.get("normal")), "normal"),
        "anomaly": _as_text_list(banks.get("anomaly", descriptions.get("anomaly")), "anomaly"),
    }
    metadata = {str(key): value for key, value in merged.items()}
    metadata["path"] = str(anchor_path.resolve())
    metadata["sidecar_path"] = str(sidecar_path.resolve()) if sidecar_path.is_file() else ""
    metadata["anchor_space"] = "frozen_clip_text"
    metadata["external_embedding_used_for_training"] = False
    metadata["normal_bank_size"] = len(text_banks["normal"])
    metadata["anomaly_bank_size"] = len(text_banks["anomaly"])
    return text_banks, metadata


def _transform_text_embeddings(
    clip_model: nn.Module,
    token_embeddings: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    cast_dtype = clip_model.transformer.get_cast_dtype()
    x = token_embeddings.to(dtype=cast_dtype)
    x = x + clip_model.positional_embedding.to(device=x.device, dtype=cast_dtype)
    transformed = clip_model.transformer(
        x.permute(1, 0, 2),
        attn_mask=clip_model.attn_mask,
    )
    if isinstance(transformed, (tuple, list)):
        transformed = transformed[0]
    x = clip_model.ln_final(transformed.permute(1, 0, 2))
    eos = token_ids.to(x.device).argmax(dim=-1)
    features = x[torch.arange(x.shape[0], device=x.device), eos] @ clip_model.text_projection
    return F.normalize(features.float(), dim=-1)


def encode_prompt_features(
    clip_model: nn.Module,
    prompt_learner: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Return the two final CLIP prompt features actually used before adaptation."""
    prompts, tokenized_prompts = prompt_learner()
    features = _transform_text_embeddings(
        clip_model,
        prompts.to(device),
        tokenized_prompts.to(device),
    )
    normal_count = int(getattr(prompt_learner, "n_cls", 1)) * int(
        getattr(prompt_learner, "normal_num", 1)
    )
    anomaly_count = int(getattr(prompt_learner, "n_cls", 1)) * int(
        getattr(prompt_learner, "anomaly_num", 1)
    )
    if normal_count + anomaly_count != features.shape[0]:
        normal_count = features.shape[0] // 2
        anomaly_count = features.shape[0] - normal_count
    if normal_count <= 0 or anomaly_count <= 0:
        raise ValueError("Prompt learner must produce both normal and anomaly prompts.")
    normal = F.normalize(features[:normal_count].mean(dim=0), dim=0)
    anomaly = F.normalize(features[normal_count : normal_count + anomaly_count].mean(dim=0), dim=0)
    return torch.stack([normal, anomaly], dim=0)


@torch.no_grad()
def encode_description_banks(
    clip_model: nn.Module,
    text_banks: Mapping[str, Sequence[str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normal_texts = list(text_banks["normal"])
    anomaly_texts = list(text_banks["anomaly"])
    all_texts = normal_texts + anomaly_texts
    token_ids = tokenize(all_texts).to(device)
    token_embeddings = clip_model.token_embedding(token_ids)
    features = _transform_text_embeddings(clip_model, token_embeddings, token_ids)
    normal_bank = features[: len(normal_texts)].detach()
    anomaly_bank = features[len(normal_texts) :].detach()
    return normal_bank, anomaly_bank


def _interleaved_context_seed(
    clip_model: nn.Module,
    descriptions: Sequence[str],
    n_ctx: int,
    device: torch.device,
) -> torch.Tensor:
    token_ids = tokenize(list(descriptions)).to(device)
    with torch.no_grad():
        embeddings = clip_model.token_embedding(token_ids).float()
    sequences: List[torch.Tensor] = []
    for row_idx in range(token_ids.shape[0]):
        eos_idx = int(token_ids[row_idx].argmax().item())
        content = embeddings[row_idx, 1:eos_idx]
        if content.numel() > 0:
            sequences.append(content)
    if not sequences:
        raise ValueError("Semantic anchor descriptions did not produce any CLIP content tokens.")
    selected: List[torch.Tensor] = []
    token_idx = 0
    while len(selected) < n_ctx:
        added = False
        for sequence in sequences:
            if token_idx < sequence.shape[0]:
                selected.append(sequence[token_idx])
                added = True
                if len(selected) == n_ctx:
                    break
        if not added:
            token_idx = 0
            continue
        token_idx += 1
    return torch.stack(selected, dim=0)


@torch.no_grad()
def initialize_prompt_context_from_descriptions(
    prompt_learner: nn.Module,
    clip_model: nn.Module,
    text_banks: Mapping[str, Sequence[str]],
    device: torch.device,
) -> Dict[str, float]:
    """Initialize all context positions with interleaved CLIP token semantics."""
    stats: Dict[str, float] = {}
    for label, parameter_name in (("normal", "ctx_pos"), ("anomaly", "ctx_neg")):
        parameter = getattr(prompt_learner, parameter_name)
        n_ctx = int(parameter.shape[-2])
        seed = _interleaved_context_seed(
            clip_model=clip_model,
            descriptions=text_banks[label],
            n_ctx=n_ctx,
            device=device,
        ).to(device=parameter.device, dtype=parameter.dtype)
        view_shape = (1,) * (parameter.dim() - 2) + tuple(seed.shape)
        parameter.copy_(seed.view(view_shape).expand_as(parameter))
        stats[f"{label}_context_norm"] = float(parameter.float().norm(dim=-1).mean().item())
    return stats


class SemanticAnchorAligner(nn.Module):
    """Directly align final learned CLIP prompts to frozen CLIP text banks."""

    def __init__(
        self,
        normal_bank: torch.Tensor,
        anomaly_bank: torch.Tensor,
        margin: float = 0.20,
        separation_weight: float = 0.50,
        direction_weight: float = 1.00,
        pair_weight: float = 0.10,
        bank_weight: float = 0.25,
        bank_temperature: float = 0.07,
        reference_prompt_gap: float = 0.0,
    ):
        super().__init__()
        if normal_bank.dim() != 2 or anomaly_bank.dim() != 2:
            raise ValueError("Teacher banks must have shape KxD.")
        if normal_bank.shape[1] != anomaly_bank.shape[1]:
            raise ValueError("Normal and anomaly teacher banks must share one feature dimension.")
        self.margin = float(margin)
        self.separation_weight = float(separation_weight)
        self.direction_weight = float(direction_weight)
        self.pair_weight = float(pair_weight)
        self.bank_weight = float(bank_weight)
        self.bank_temperature = max(float(bank_temperature), 1e-4)
        normal_bank = F.normalize(normal_bank.detach().float(), dim=-1)
        anomaly_bank = F.normalize(anomaly_bank.detach().float(), dim=-1)
        normal_center = F.normalize(normal_bank.mean(dim=0), dim=0)
        anomaly_center = F.normalize(anomaly_bank.mean(dim=0), dim=0)
        teacher_pair_similarity = normal_center @ anomaly_center
        teacher_gap = (1.0 - teacher_pair_similarity).clamp_min(0.0)
        reference_gap = max(float(reference_prompt_gap), 0.0)
        adaptive_margin = min(
            max(self.margin, 0.0),
            0.9 * max(float(teacher_gap.item()), reference_gap),
        )
        self.register_buffer("normal_bank", normal_bank)
        self.register_buffer("anomaly_bank", anomaly_bank)
        self.register_buffer("normal_center", normal_center)
        self.register_buffer("anomaly_center", anomaly_center)
        self.register_buffer("teacher_pair_similarity", teacher_pair_similarity.reshape(()))
        self.register_buffer("adaptive_margin", torch.tensor(adaptive_margin, dtype=torch.float32))

    def _multi_positive_loss(
        self,
        feature: torch.Tensor,
        positive_bank: torch.Tensor,
        negative_bank: torch.Tensor,
    ) -> torch.Tensor:
        positive_logits = feature @ positive_bank.transpose(0, 1) / self.bank_temperature
        negative_logits = feature @ negative_bank.transpose(0, 1) / self.bank_temperature
        numerator = torch.logsumexp(positive_logits, dim=-1)
        denominator = torch.logsumexp(torch.cat([positive_logits, negative_logits], dim=-1), dim=-1)
        return (denominator - numerator).mean()

    def forward(self, prompt_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if prompt_features.dim() != 2 or prompt_features.shape[0] != 2:
            raise ValueError(
                "Expected final learned prompt features with shape (2,D), "
                f"got {tuple(prompt_features.shape)}."
            )
        prompt_features = F.normalize(prompt_features.float(), dim=-1)
        normal_feature, anomaly_feature = prompt_features[0], prompt_features[1]
        normal_center = self.normal_center.to(prompt_features)
        anomaly_center = self.anomaly_center.to(prompt_features)
        normal_bank = self.normal_bank.to(prompt_features)
        anomaly_bank = self.anomaly_bank.to(prompt_features)

        normal_similarity = normal_feature @ normal_center
        anomaly_similarity = anomaly_feature @ anomaly_center
        cross_normal = normal_feature @ anomaly_center
        cross_anomaly = anomaly_feature @ normal_center
        cross_similarity = 0.5 * (cross_normal + cross_anomaly)
        alignment_loss = 0.5 * (1.0 - normal_similarity + 1.0 - anomaly_similarity)

        teacher_direction = F.normalize(anomaly_center - normal_center, dim=0)
        prompt_direction = F.normalize(anomaly_feature - normal_feature, dim=0)
        direction_similarity = prompt_direction @ teacher_direction
        direction_loss = 1.0 - direction_similarity

        prompt_pair_similarity = normal_feature @ anomaly_feature
        prompt_gap = 1.0 - prompt_pair_similarity
        separation_loss = F.relu(self.adaptive_margin.to(prompt_features) - prompt_gap)

        bank_loss = 0.5 * (
            self._multi_positive_loss(normal_feature.unsqueeze(0), normal_bank, anomaly_bank)
            + self._multi_positive_loss(anomaly_feature.unsqueeze(0), anomaly_bank, normal_bank)
        )
        loss = (
            self.direction_weight * direction_loss
            + self.pair_weight * alignment_loss
            + self.separation_weight * separation_loss
            + self.bank_weight * bank_loss
        )
        return {
            "loss": loss,
            "alignment_loss": alignment_loss,
            "direction_loss": direction_loss,
            "separation_loss": separation_loss,
            "bank_loss": bank_loss,
            "normal_similarity": normal_similarity,
            "anomaly_similarity": anomaly_similarity,
            "cross_similarity": cross_similarity,
            "direction_similarity": direction_similarity,
            "prompt_pair_similarity": prompt_pair_similarity,
            "teacher_pair_similarity": self.teacher_pair_similarity.to(prompt_features),
            "adaptive_margin": self.adaptive_margin.to(prompt_features),
            "prompt_features": prompt_features,
        }


def build_semantic_anchor_aligner(
    path: str,
    prompt_learner: nn.Module,
    clip_model: nn.Module,
    device: torch.device,
    margin: float,
    separation_weight: float,
    direction_weight: float = 1.00,
    pair_weight: float = 0.10,
    bank_weight: float = 0.25,
    bank_temperature: float = 0.07,
    initialize_context: bool = True,
) -> Tuple[SemanticAnchorAligner, Dict[str, Any]]:
    text_banks, metadata = load_semantic_anchor_descriptions(path)
    initialization_stats: Dict[str, float] = {}
    if initialize_context:
        initialization_stats = initialize_prompt_context_from_descriptions(
            prompt_learner=prompt_learner,
            clip_model=clip_model,
            text_banks=text_banks,
            device=device,
        )
    normal_bank, anomaly_bank = encode_description_banks(
        clip_model=clip_model,
        text_banks=text_banks,
        device=device,
    )
    with torch.no_grad():
        initial_prompt_features = encode_prompt_features(
            clip_model=clip_model,
            prompt_learner=prompt_learner,
            device=device,
        )
        reference_prompt_gap = float(
            1.0 - (initial_prompt_features[0] @ initial_prompt_features[1]).item()
        )
    aligner = SemanticAnchorAligner(
        normal_bank=normal_bank,
        anomaly_bank=anomaly_bank,
        margin=margin,
        separation_weight=separation_weight,
        direction_weight=direction_weight,
        pair_weight=pair_weight,
        bank_weight=bank_weight,
        bank_temperature=bank_temperature,
        reference_prompt_gap=reference_prompt_gap,
    ).to(device)
    metadata["semantic_context_initialized"] = bool(initialize_context)
    metadata["initialization_stats"] = initialization_stats
    metadata["teacher_pair_similarity"] = float(aligner.teacher_pair_similarity.item())
    metadata["adaptive_margin"] = float(aligner.adaptive_margin.item())
    metadata["initial_prompt_gap"] = reference_prompt_gap
    metadata["direction_weight"] = float(direction_weight)
    metadata["pair_weight"] = float(pair_weight)
    metadata["bank_weight"] = float(bank_weight)
    metadata["bank_temperature"] = float(bank_temperature)
    return aligner, metadata
