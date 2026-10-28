"""Frozen pretrained encoders and audited I/O for residual-first experiments."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from tools.reference_highres import fixed_projection, normalized_box
from tools.residual_text import coverage_maps, image_geometry, nearest_residual, paired_directions
from tools.vlm_review import atomic_json, digest, file_digest, load_json


GENERIC_PROMPT = dict(id="generic", normal="a normal intact surface", abnormal="an abnormal damaged surface")
TEMPLATES = ("a photo of {}.", "a close-up photo of {}.", "an industrial inspection image of {}.")


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_torch(path, device="cpu"):
    # Only task-owned, locally produced checkpoints are accepted by this driver.
    return torch.load(path, map_location=device, weights_only=False)


def image_identity(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(), list(rgb.size)


def make_record(dataset, category, image, mask, partition):
    image = str(Path(image).resolve())
    sha, size = image_identity(image)
    mask = str(Path(mask).resolve()) if mask else None
    return dict(id=digest([dataset, category, image])[:24], dataset=dataset, category=category,
                image_path=image, image_sha=file_digest(image), rgb_sha=sha, native_size=size,
                mask_path=mask, partition=partition,
                # Target masks are not opened, even for hashing, before predictions lock.
                mask_sha=file_digest(mask) if mask and partition != "eval" else None)


def unchanged(row, masks=False):
    if file_digest(row["image_path"]) != row["image_sha"]:
        raise ValueError(f"Image changed after prepare: {row['id']}")
    if masks and row["mask_path"] and row["partition"] != "eval":
        if file_digest(row["mask_path"]) != row["mask_sha"]:
            raise ValueError(f"Source mask changed: {row['id']}")


def normalized_boxes(pixel_boxes, size):
    return np.asarray([normalized_box(b, *size) for b in pixel_boxes], np.float32)


def native_mask(row, path=None):
    width, height = row["native_size"]
    path = path or row["mask_path"]
    if not path:
        return torch.zeros(height, width)
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError(f"Mask/native-image geometry differs: {path}")
        array = np.asarray(image.convert("L"))
    return torch.from_numpy((array > (0 if row["dataset"] == "visa" else 127)).copy()).float()


def evaluation_mask(row, size):
    mask = native_mask(row)
    return F.interpolate(mask[None, None], (size, size), mode="nearest")[0, 0].numpy().astype(bool)


def source_labels(row, pixel_boxes, grid, annotations, type_ids):
    if row["partition"] not in ("train", "val"):
        raise ValueError("Target labels must not enter training/selection datasets")
    unchanged(row, masks=True)
    normal_mask = native_mask(row)
    coverage = coverage_maps(normal_mask, pixel_boxes, grid)
    typed = torch.zeros(len(pixel_boxes), len(type_ids), grid, grid)
    for annotation in annotations:
        if annotation["image_id"] != row["id"]:
            continue
        if file_digest(annotation["mask_path"]) != annotation["mask_sha"]:
            raise ValueError("Source semantic region changed")
        region = native_mask(row, annotation["mask_path"])
        if ((region > 0) & (normal_mask == 0)).any():
            raise ValueError("Semantic region extends outside the source anomaly mask")
        index = type_ids.index(annotation["type_id"])
        typed[:, index] = torch.maximum(typed[:, index], coverage_maps(region, pixel_boxes, grid))
    return coverage, typed


def load_annotations(path, rows):
    if not path:
        return [], []
    data = load_json(path)
    if data.get("version") != 1 or not isinstance(data.get("types"), list) or not data["types"]:
        raise ValueError("Semantic annotations require version=1 and nonempty types/regions")
    prompts, seen = [], {"generic"}
    for value in data["types"]:
        if (not isinstance(value, dict) or set(value) != {"id", "normal", "abnormal"}
                or not all(isinstance(v, str) and v.strip() for v in value.values())
                or value["id"] in seen or value["normal"] == value["abnormal"]):
            raise ValueError("Each semantic type needs unique id and distinct paired state descriptions")
        prompts.append(value)
        seen.add(value["id"])
    by_path = {r["image_path"]: r for r in rows if r["partition"] in ("train", "val")}
    regions = []
    trained = set()
    for value in data.get("regions", []):
        row = by_path.get(str(Path(value["image_path"]).resolve()))
        if row is None or not row["mask_path"] or value["type_id"] not in seen - {"generic"}:
            raise ValueError("Semantic regions must refer only to source anomalous query images")
        mask_path = str(Path(value["mask_path"]).resolve())
        region = native_mask(row, mask_path)
        if not region.any() or ((region > 0) & (native_mask(row) == 0)).any():
            raise ValueError("Semantic masks must be nonempty subsets of source anomaly masks")
        regions.append(dict(image_id=row["id"], mask_path=mask_path, mask_sha=file_digest(mask_path),
                            type_id=value["type_id"]))
        if row["partition"] == "train":
            trained.add(value["type_id"])
    if trained != seen - {"generic"}:
        raise ValueError("Every semantic type needs at least one source TRAIN region; binary masks alone are insufficient")
    if len(prompts) < 2:
        raise ValueError("Fine-grained contrastive training needs at least two labeled change types")
    return prompts, regions


class FrozenDinoExtractor:
    """Direct pretrained DINO. Does NOT call load_dual or install HFA/adapters."""
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        if file_digest(cfg["dino_weights"]) != cfg["dino_sha"]:
            raise ValueError("Pretrained DINO weights changed")
        self.model = torch.hub.load(cfg["dino_repo_dir"], cfg["dino_model_name"], source="local",
                                    weights=cfg["dino_weights"]).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.layers = cfg["layers"]
        if max(self.layers) >= len(self.model.blocks):
            raise ValueError("Requested layers exceed backbone depth")
        self.anchor = getattr(self.model, "norm", None) or getattr(self.model, "fc_norm", None)
        if self.anchor is None:
            raise ValueError("Backbone has no native token normalization")
        self.tokens, self.handles = {}, []
        for layer in self.layers:
            def capture(module, inputs, output, layer=layer):
                value = output[0] if isinstance(output, (tuple, list)) else output
                start = 1 + int(getattr(self.model, "n_storage_tokens", 0))
                self.tokens[layer] = self.anchor(value)[:, start:].float()
            self.handles.append(self.model.blocks[layer].register_forward_hook(capture))
        self.projection = fixed_projection(cfg["input_dim"], cfg["feature_dim"], cfg["data_seed"]).to(device)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.inference_mode()
    def view(self, image):
        size = self.cfg["image_size"]
        pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).copy()
        x = torch.from_numpy(pixels).permute(2, 0, 1).to(self.device).float() / 255
        mean = x.new_tensor([.485, .456, .406])[:, None, None]
        std = x.new_tensor([.229, .224, .225])[:, None, None]
        self.tokens.clear()
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if self.device.type == "cuda" and self.cfg["precision"] == "bf16" else nullcontext())
        with context:
            self.model(((x - mean) / std)[None])
        side = size // self.cfg["patch_size"]
        expected = (1, side * side, self.cfg["input_dim"])
        if set(self.tokens) != set(self.layers) or any(tuple(t.shape) != expected for t in self.tokens.values()):
            raise ValueError("Unexpected backbone token shapes; verify patch size/input dimension/layers")
        return torch.stack([F.normalize(self.tokens[layer][0], dim=-1) for layer in self.layers])

    def views(self, image):
        boxes = image_geometry(*image.size, self.cfg["tile_fraction"])
        return [self.view(image if i == 0 else image.crop(box)) for i, box in enumerate(boxes)], boxes


@torch.inference_mode()
def encode_text_bank(cfg, prompts, device):
    """Load the explicit local OpenAI-format CLIP checkpoint, never auto-download.

    This repository's CLIP.encode_text has a nonstandard signature. Use its
    image-independent path directly; no Prompt Learner/Prompt Adapter is loaded.
    """
    from CLIP.openai import load_openai_model
    from CLIP.tokenizer import tokenize
    if file_digest(cfg["clip_weights"]) != cfg["clip_sha"]:
        raise ValueError("Frozen CLIP weights changed")
    model = load_openai_model(cfg["clip_weights"], precision="fp32", device="cpu", jit=False)
    model.visual = nn.Identity()  # discard unused visual tower before transfer
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    states = [[template.format(prompt[state]) for template in TEMPLATES]
              for prompt in prompts for state in ("normal", "abnormal")]
    features = []
    for texts in states:
        tokens = tokenize(texts).to(device)
        features.append(model.encode_text(tokens, "", device, None, normalize=True).float().cpu())
    features = torch.stack(features).reshape(len(prompts), 2, len(TEMPLATES), -1)
    return paired_directions(features[:, 0], features[:, 1]).numpy(), features.numpy()


@torch.inference_mode()
def export_residuals(extractor, query_views, normal_views, cfg):
    pools = [torch.cat([views[0] for views in normal_views], 1),
             torch.cat([tile for views in normal_views for tile in views[1:]], 1)]
    residual, distances, indices = [], [], []
    side = cfg["image_size"] // cfg["patch_size"]
    for view, query in enumerate(query_views):
        delta, distance, index = nearest_residual(query, pools[int(view > 0)].to(query.device),
                                                   cfg["match_chunk"], cfg["bank_chunk"])
        projected = delta @ extractor.projection
        residual.append(projected.permute(0, 2, 1).reshape(len(cfg["layers"]), -1, side, side).cpu())
        distances.append(distance.reshape(len(cfg["layers"]), side, side).cpu())
        indices.append(index.reshape(len(cfg["layers"]), side, side).cpu())
    return dict(residual=torch.stack(residual).numpy().astype(np.float16),
                distance=torch.stack(distances).numpy().astype(np.float32),
                match_index=torch.stack(indices).numpy().astype(np.int32))
