"""Build fixed normal/anomaly semantic anchors with the Gemini free tier.

Default pipeline:
  1. Gemini 3.5 Flash generates two category-independent descriptions.
  2. The descriptions are expanded with auditable normal/anomaly subtype banks.
  3. Legacy Gemini embeddings are retained for provenance only. P0/P1 training
     re-encodes the text banks with the frozen project CLIP model.

The API key is read only from ``GEMINI_API_KEY``. Training never calls Gemini;
it only loads the generated ``.pt`` file.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import httpx
import torch
import torch.nn.functional as F


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

NORMAL_SUBTYPE_DESCRIPTIONS = [
    "An intact and complete object with regular geometry and correctly aligned components.",
    "A clean continuous surface with uniform texture, color, and material appearance.",
    "A defect-free object without cracks, scratches, holes, chips, or deformation.",
    "All expected regions are present with no missing, extra, or displaced parts.",
    "The object is free from stains, residue, contamination, and foreign matter.",
]

ANOMALY_SUBTYPE_DESCRIPTIONS = [
    "A cracked, scratched, chipped, broken, or otherwise damaged surface.",
    "A deformed, distorted, misaligned, or structurally incomplete object.",
    "An object containing missing, extra, loose, or displaced components.",
    "Irregular texture, localized discoloration, holes, or material discontinuity.",
    "Visible stain, residue, contamination, embedded debris, or foreign matter.",
]


def _raise_for_api_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json()
    except Exception:
        detail = response.text[:1000]
    raise RuntimeError(f"Gemini API request failed ({response.status_code}): {detail}")


def _extract_generated_text(payload: Dict[str, Any]) -> str:
    texts = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if part.get("text"):
                texts.append(str(part["text"]))
    if not texts:
        raise RuntimeError(f"Gemini response did not contain generated text: {payload}")
    return "\n".join(texts).strip()


def _parse_descriptions(text: str) -> Tuple[str, str]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
        normal = str(payload["normal"]).strip()
        anomaly = str(payload["anomaly"]).strip()
    except Exception as exc:
        raise RuntimeError(
            "Could not parse Gemini output as JSON with `normal` and `anomaly` fields. "
            f"Raw output: {text[:1000]}"
        ) from exc
    if not normal or not anomaly or normal == anomaly:
        raise RuntimeError("Gemini returned empty or identical semantic descriptions.")
    return normal, anomaly


def generate_descriptions(
    client: httpx.Client,
    api_key: str,
    model: str,
) -> Tuple[str, str, str]:
    prompt = """
Create two fixed, category-independent semantic anchors for visual anomaly detection.
The NORMAL anchor must describe the observable visual state of an intact, expected,
defect-free manufactured or natural object. The ANOMALY anchor must describe the
observable visual state of a defective, damaged, contaminated, deformed, incomplete,
or texturally inconsistent object. Cover structure, geometry, surface, texture, color,
material continuity, missing/extra regions, and foreign matter without naming any
dataset or object category. Make the two English descriptions concise, visually
grounded, mutually discriminative, and approximately equal in length.
Return JSON only: {"normal":"one sentence","anomaly":"one sentence"}
""".strip()
    response = client.post(
        f"{GEMINI_API_BASE}/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
    )
    _raise_for_api_error(response)
    raw_text = _extract_generated_text(response.json())
    normal, anomaly = _parse_descriptions(raw_text)
    return normal, anomaly, raw_text


def _extract_embedding(payload: Dict[str, Any]) -> torch.Tensor:
    if isinstance(payload.get("embedding"), dict):
        values = payload["embedding"].get("values")
    else:
        embeddings = payload.get("embeddings", [])
        values = embeddings[0].get("values") if embeddings else None
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Gemini response did not contain an embedding vector: {payload}")
    vector = torch.tensor(values, dtype=torch.float32)
    if vector.dim() != 1 or not torch.isfinite(vector).all():
        raise RuntimeError(f"Invalid Gemini embedding shape/content: {tuple(vector.shape)}")
    return F.normalize(vector, dim=0)


def embed_description(
    client: httpx.Client,
    api_key: str,
    model: str,
    description: str,
    dimensions: int,
) -> torch.Tensor:
    response = client.post(
        f"{GEMINI_API_BASE}/models/{model}:embedContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "model": f"models/{model}",
            "content": {"parts": [{"text": description}]},
            "output_dimensionality": int(dimensions),
        },
    )
    _raise_for_api_error(response)
    return _extract_embedding(response.json())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="./asset/gemini_semantic_anchors.pt",
        help="output .pt file loaded by stage-one training",
    )
    parser.add_argument("--generator_model", type=str, default="gemini-3.5-flash")
    parser.add_argument("--embedding_model", type=str, default="gemini-embedding-2")
    parser.add_argument("--embedding_dimensions", type=int, default=768)
    parser.add_argument(
        "--normal_text",
        type=str,
        default="",
        help="optional fixed normal description; requires --anomaly_text",
    )
    parser.add_argument(
        "--anomaly_text",
        type=str,
        default="",
        help="optional fixed anomaly description; requires --normal_text",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to build Gemini semantic anchors.")
    if args.embedding_dimensions < 128 or args.embedding_dimensions > 3072:
        raise ValueError("Gemini embedding dimensions must be between 128 and 3072.")
    if bool(args.normal_text) != bool(args.anomaly_text):
        raise ValueError("--normal_text and --anomaly_text must be supplied together.")

    with httpx.Client(timeout=args.timeout) as client:
        if args.normal_text:
            normal_text = args.normal_text.strip()
            anomaly_text = args.anomaly_text.strip()
            raw_generation = "manual descriptions supplied on the command line"
            generator_model = "manual"
        else:
            normal_text, anomaly_text, raw_generation = generate_descriptions(
                client=client,
                api_key=api_key,
                model=args.generator_model,
            )
            generator_model = args.generator_model

        anchors = torch.stack(
            [
                embed_description(
                    client, api_key, args.embedding_model, normal_text, args.embedding_dimensions
                ),
                embed_description(
                    client, api_key, args.embedding_model, anomaly_text, args.embedding_dimensions
                ),
            ],
            dim=0,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "labels": ["normal", "anomaly"],
        "anchors": anchors.cpu(),
        "descriptions": {"normal": normal_text, "anomaly": anomaly_text},
        "description_banks": {
            "normal": [normal_text, *NORMAL_SUBTYPE_DESCRIPTIONS],
            "anomaly": [anomaly_text, *ANOMALY_SUBTYPE_DESCRIPTIONS],
        },
        "provider": "google-gemini",
        "generator_model": generator_model,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": int(anchors.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clip_encoded": False,
    }
    torch.save(payload, output_path)

    audit_path = output_path.with_suffix(".json")
    audit_payload = {key: value for key, value in payload.items() if key != "anchors"}
    audit_payload["raw_generation"] = raw_generation
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pair_similarity = float((anchors[0] @ anchors[1]).detach())
    print(f"Saved semantic anchors: {output_path.resolve()}")
    print(f"Saved audit metadata: {audit_path.resolve()}")
    print(f"Anchor shape: {tuple(anchors.shape)}")
    print(f"Normal/anomaly cosine similarity: {pair_similarity:.6f}")
    print(f"Normal: {normal_text}")
    print(f"Anomaly: {anomaly_text}")


if __name__ == "__main__":
    main()
