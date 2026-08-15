from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ANCHOR_LABELS: Tuple[str, str] = ("normal", "anomaly")


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_external_semantic_anchors(path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a fixed normal/anomaly anchor pair produced outside CLIP.

    The payload must be a dict containing ``anchors`` and explicit non-CLIP
    provenance metadata. This prevents accidentally feeding CLIP features into
    a training run that is configured for external LLM anchors.
    """
    anchor_path = Path(path)
    if not anchor_path.is_file():
        raise FileNotFoundError(
            f"Semantic anchor file not found: {anchor_path}. "
            "Generate it with tools/build_gemini_semantic_anchors.py first."
        )

    payload = _torch_load(str(anchor_path))
    metadata: Dict[str, Any] = {}
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Semantic anchor payload must be an auditable dict with `anchors`, "
            "`embedding_model`, and `clip_encoded=False`."
        )
    if "anchors" not in payload:
        raise ValueError(f"Semantic anchor payload {anchor_path} has no `anchors` field.")
    anchors = payload["anchors"]
    metadata = {str(key): value for key, value in payload.items() if key != "anchors"}
    if metadata.get("clip_encoded") is not False:
        raise ValueError(
            "Semantic anchor payload must explicitly set `clip_encoded=False`; "
            "CLIP-encoded anchors are not accepted."
        )
    if not str(metadata.get("embedding_model", "")).strip():
        raise ValueError("Semantic anchor payload must record its external `embedding_model`.")

    if not torch.is_tensor(anchors):
        anchors = torch.as_tensor(anchors)
    anchors = anchors.detach().float()
    if anchors.dim() != 2:
        raise ValueError(f"Semantic anchors must be a 2D tensor, got shape={tuple(anchors.shape)}.")
    if anchors.shape[0] != 2 and anchors.shape[1] == 2:
        anchors = anchors.transpose(0, 1)
    if anchors.shape[0] != 2:
        raise ValueError(
            "Semantic anchors must contain exactly two rows ordered as normal/anomaly; "
            f"got shape={tuple(anchors.shape)}."
        )
    if not torch.isfinite(anchors).all():
        raise ValueError("Semantic anchors contain NaN or Inf values.")

    labels = tuple(metadata.get("labels", ANCHOR_LABELS))
    if labels != ANCHOR_LABELS:
        raise ValueError(
            f"Semantic anchor labels must be {ANCHOR_LABELS}, got {labels}."
        )
    anchors = F.normalize(anchors, dim=-1)
    metadata["labels"] = list(ANCHOR_LABELS)
    metadata["path"] = str(anchor_path.resolve())
    metadata["dimension"] = int(anchors.shape[1])
    return anchors, metadata


class SemanticAnchorAligner(nn.Module):
    """Align learnable normal/anomaly context-token prototypes to fixed LLM anchors.

    The prompt tokens live in CLIP's token-input space while the anchors live in
    an external embedding space. A small trainable projector bridges the spaces;
    it should use a substantially lower learning rate than the prompt tokens.
    """

    def __init__(
        self,
        prompt_dim: int,
        anchors: torch.Tensor,
        margin: float = 0.20,
        separation_weight: float = 0.50,
    ):
        super().__init__()
        if anchors.dim() != 2 or anchors.shape[0] != 2:
            raise ValueError(f"Expected anchors with shape (2,D), got {tuple(anchors.shape)}.")
        if prompt_dim <= 0:
            raise ValueError(f"prompt_dim must be positive, got {prompt_dim}.")

        anchor_dim = int(anchors.shape[1])
        self.prompt_dim = int(prompt_dim)
        self.anchor_dim = anchor_dim
        self.margin = float(margin)
        self.separation_weight = float(separation_weight)
        self.projector = nn.Linear(self.prompt_dim, self.anchor_dim, bias=False)
        if self.prompt_dim == self.anchor_dim:
            nn.init.eye_(self.projector.weight)
        else:
            nn.init.orthogonal_(self.projector.weight)
        self.register_buffer("anchors", F.normalize(anchors.detach().float(), dim=-1))

    @staticmethod
    def _token_prototype(context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.dim() < 2:
            raise ValueError(
                "Prompt context tokens must have at least two dimensions, "
                f"got shape={tuple(context_tokens.shape)}."
            )
        reduce_dims = tuple(range(context_tokens.dim() - 1))
        return context_tokens.float().mean(dim=reduce_dims)

    def forward(
        self,
        normal_context_tokens: torch.Tensor,
        anomaly_context_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        normal_proto = self._token_prototype(normal_context_tokens)
        anomaly_proto = self._token_prototype(anomaly_context_tokens)
        prompt_prototypes = torch.stack([normal_proto, anomaly_proto], dim=0)
        if prompt_prototypes.shape[1] != self.prompt_dim:
            raise ValueError(
                f"Expected prompt dimension {self.prompt_dim}, got {prompt_prototypes.shape[1]}."
            )

        projected = F.normalize(self.projector(prompt_prototypes), dim=-1)
        anchors = self.anchors.to(device=projected.device, dtype=projected.dtype)
        similarities = projected @ anchors.transpose(0, 1)
        positive = similarities.diagonal()
        negative = similarities.flip(dims=(1,)).diagonal()

        alignment_loss = (1.0 - positive).mean()
        separation_loss = F.relu(self.margin + negative - positive).mean()
        loss = alignment_loss + self.separation_weight * separation_loss

        return {
            "loss": loss,
            "alignment_loss": alignment_loss,
            "separation_loss": separation_loss,
            "normal_similarity": positive[0],
            "anomaly_similarity": positive[1],
            "cross_similarity": negative.mean(),
            "projected_prompts": projected,
        }


def build_semantic_anchor_aligner(
    path: str,
    prompt_learner: nn.Module,
    device: torch.device,
    margin: float,
    separation_weight: float,
) -> Tuple[SemanticAnchorAligner, Dict[str, Any]]:
    if not hasattr(prompt_learner, "ctx_pos") or not hasattr(prompt_learner, "ctx_neg"):
        raise AttributeError("Prompt learner must expose `ctx_pos` and `ctx_neg` parameters.")
    anchors, metadata = load_external_semantic_anchors(path)
    prompt_dim = int(prompt_learner.ctx_pos.shape[-1])
    aligner = SemanticAnchorAligner(
        prompt_dim=prompt_dim,
        anchors=anchors,
        margin=margin,
        separation_weight=separation_weight,
    ).to(device)
    return aligner, metadata
