"""Qwen3-VL teacher helpers for offline anomaly-decision distillation.

The VLM is deliberately isolated from the training graph.  It sees an original
image, one normal reference, the stage-one heatmap, and a few proposed crops,
then returns a small JSON decision.  A compact DINO-side head learns these
decisions so deployment and MARA training do not need the 8B model.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ACTION_NAMES = ("stop", "refine", "suppress")
PROMPT_VERSION = "sra-dino-vlm-teacher-v1"


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    if not math.isfinite(number):
        number = low
    return min(high, max(low, number))


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class VLMDecision:
    image_anomaly_probability: float
    action: str
    regions: List[Dict[str, Any]]
    preferred_layer: int
    confidence: float
    parse_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fallback_decision(num_layers: int) -> VLMDecision:
    return VLMDecision(
        image_anomaly_probability=0.5,
        action="stop",
        regions=[],
        preferred_layer=max(0, num_layers - 1),
        confidence=0.0,
        parse_ok=False,
    )


def _extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, Mapping):
            return value
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("VLM response contains no JSON object.")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, Mapping):
        raise ValueError("VLM JSON response must be an object.")
    return value


def parse_vlm_decision(text: str, num_layers: int) -> VLMDecision:
    """Parse and constrain an untrusted VLM response without using GT labels."""

    try:
        value = _extract_json_object(text)
        action = str(value.get("action", "stop")).strip().lower()
        if action not in ACTION_NAMES:
            action = "stop"
        preferred_layer = min(
            max(0, _integer(value.get("preferred_layer"), num_layers - 1)),
            max(0, num_layers - 1),
        )
        regions: List[Dict[str, Any]] = []
        raw_regions = value.get("regions", [])
        if isinstance(raw_regions, Sequence) and not isinstance(raw_regions, (str, bytes)):
            for item in raw_regions[:5]:
                if not isinstance(item, Mapping):
                    continue
                raw_box = item.get("box", item.get("bbox", []))
                if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
                    continue
                x1, y1, x2, y2 = [_clamp(coord, 0.0, 1000.0) for coord in raw_box]
                if max(x1, y1, x2, y2) <= 1.0:
                    x1, y1, x2, y2 = [coord * 1000.0 for coord in (x1, y1, x2, y2)]
                if x2 <= x1 or y2 <= y1:
                    continue
                regions.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "defect_probability": _clamp(
                            item.get("defect_probability", item.get("probability", 0.5))
                        ),
                    }
                )
        return VLMDecision(
            image_anomaly_probability=_clamp(
                value.get("image_anomaly_probability", value.get("anomaly_probability", 0.5))
            ),
            action=action,
            regions=regions,
            preferred_layer=preferred_layer,
            confidence=_clamp(value.get("confidence", 0.5)),
            parse_ok=True,
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return fallback_decision(num_layers)


def build_vlm_prompt(
    category: str,
    candidate_boxes: Sequence[Sequence[float]],
    num_layers: int,
) -> str:
    roi_text = ", ".join(
        f"ROI-{index + 1}={list(map(lambda x: round(float(x), 1), box))}"
        for index, box in enumerate(candidate_boxes)
    ) or "none"
    return f"""You are an industrial visual-inspection teacher. Evaluate only visible evidence.
The product category is {category}. Images are ordered as: (1) query image, (2) known-normal
reference of the same category, (3) SRA-DINO anomaly heatmap, followed by proposed ROI crops.
Compare query and normal reference. Treat the heatmap only as a fallible proposal: reject
texture, edge, reflection, alignment, or background false positives. Candidate boxes use
normalized [x1,y1,x2,y2] coordinates in 0..1000: {roi_text}.

Choose action "refine" when localized evidence should change the heatmap, "suppress" when the
proposal is likely a false positive, or "stop" when it is already reliable. preferred_layer is
an index from 0 to {max(0, num_layers - 1)} (early=fine detail, late=semantic structure).
Return JSON only, with this exact schema:
{{"image_anomaly_probability":0.0,"action":"stop","regions":[{{"box":[0,0,1,1],"defect_probability":0.0}}],"preferred_layer":0,"confidence":0.0}}
Use at most five regions. Never mention ground-truth labels or masks."""


def candidate_boxes_from_map(
    anomaly_map: torch.Tensor,
    num_rois: int = 3,
    roi_fraction: float = 0.25,
) -> List[List[float]]:
    """Pick deterministic peak-centered boxes in normalized 0..1000 coordinates."""

    value = anomaly_map.detach().float()
    while value.dim() > 2:
        value = value[0]
    if value.dim() != 2:
        raise ValueError(f"Expected a 2D anomaly map, got shape={tuple(value.shape)}")
    height, width = value.shape
    suppressed = value.clone()
    box_h = max(2, int(round(height * roi_fraction)))
    box_w = max(2, int(round(width * roi_fraction)))
    boxes: List[List[float]] = []
    for _ in range(max(0, int(num_rois))):
        flat_index = int(suppressed.argmax().item())
        y, x = divmod(flat_index, width)
        y1 = max(0, min(height - box_h, y - box_h // 2))
        x1 = max(0, min(width - box_w, x - box_w // 2))
        y2 = min(height, y1 + box_h)
        x2 = min(width, x1 + box_w)
        boxes.append(
            [
                1000.0 * x1 / width,
                1000.0 * y1 / height,
                1000.0 * x2 / width,
                1000.0 * y2 / height,
            ]
        )
        suppressed[y1:y2, x1:x2] = torch.finfo(suppressed.dtype).min
    return boxes


def heatmap_overlay(image: Image.Image, anomaly_map: torch.Tensor) -> Image.Image:
    value = anomaly_map.detach().float().cpu()
    while value.dim() > 2:
        value = value[0]
    value = value - value.min()
    value = value / value.max().clamp_min(1e-6)
    value = F.interpolate(
        value[None, None], size=(image.height, image.width), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    # A compact blue-to-yellow-to-red map without a matplotlib dependency.
    red = np.clip(2.0 * value, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * np.abs(value - 0.5), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - value), 0.0, 1.0)
    color = np.stack([red, green, blue], axis=-1)
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    overlay = np.clip(0.55 * base + 0.45 * color, 0.0, 1.0)
    return Image.fromarray((overlay * 255.0).astype(np.uint8), mode="RGB")


def crop_regions(image: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Image.Image]:
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        pixel_box = (
            int(round(x1 * image.width / 1000.0)),
            int(round(y1 * image.height / 1000.0)),
            int(round(x2 * image.width / 1000.0)),
            int(round(y2 * image.height / 1000.0)),
        )
        crops.append(image.crop(pixel_box).convert("RGB"))
    return crops


def decision_target_map(decision: VLMDecision | Mapping[str, Any], map_size: int) -> torch.Tensor:
    if not isinstance(decision, VLMDecision):
        decision = VLMDecision(**dict(decision))
    target = torch.zeros(1, map_size, map_size, dtype=torch.float32)
    for region in decision.regions:
        x1, y1, x2, y2 = region["box"]
        left = max(0, min(map_size - 1, int(math.floor(x1 * map_size / 1000.0))))
        top = max(0, min(map_size - 1, int(math.floor(y1 * map_size / 1000.0))))
        right = max(left + 1, min(map_size, int(math.ceil(x2 * map_size / 1000.0))))
        bottom = max(top + 1, min(map_size, int(math.ceil(y2 * map_size / 1000.0))))
        probability = _clamp(region.get("defect_probability", 0.5))
        target[:, top:bottom, left:right] = torch.maximum(
            target[:, top:bottom, left:right],
            torch.tensor(probability, dtype=target.dtype),
        )
    return target


class QwenVLLMTeacher:
    """Lazy vLLM wrapper so cache parsing/tests do not require VLM dependencies."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct-FP8",
        gpu_memory_utilization: float = 0.70,
        max_model_len: int = 4096,
        max_tokens: int = 192,
        max_images: int = 8,
    ) -> None:
        # Qwen's official offline-vLLM example uses spawn.  Cache generation
        # already initialized CUDA for DINO before the vLLM engine is created.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from qwen_vl_utils import process_vision_info
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "VLM cache generation requires vLLM. Install requirements-vlm.txt first."
            ) from exc

        self._sampling = SamplingParams(
            temperature=0.0,
            max_tokens=int(max_tokens),
            stop_token_ids=None,
        )
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._process_vision_info = process_vision_info
        self._llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            limit_mm_per_prompt={"image": int(max_images)},
            seed=0,
        )
        self.model_id = model_id

    @staticmethod
    def _messages(prompt: str, images: Sequence[Image.Image]) -> List[Dict[str, Any]]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def generate(self, prompt: str, images: Sequence[Image.Image]) -> str:
        messages = self._messages(prompt, images)
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        multi_modal_data: Dict[str, Any] = {}
        if image_inputs is not None:
            multi_modal_data["image"] = image_inputs
        if video_inputs is not None:
            multi_modal_data["video"] = video_inputs
        request = {
            "prompt": prompt_text,
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": video_kwargs,
        }
        output = self._llm.generate([request], sampling_params=self._sampling, use_tqdm=False)
        return output[0].outputs[0].text
