"""Versioned decision validation: explanation warnings do not erase actions."""
import json
import re

POLICY = "repair_v2"


def parse_repaired(text, candidate_id, reference=False):
    fallback = dict(parse_ok=False, candidate_id=candidate_id, verdict="insufficient_evidence",
                    visibility="insufficient", reference_match="unavailable", evidence="", defect_type="unknown")
    errors, warnings = [], []
    def unique_object(pairs):
        obj = {}
        for k,v in pairs:
            if k in obj:
                raise ValueError("duplicate_key")
            obj[k] = v
        return obj
    try:
        raw = str(text).strip()
        if len(raw) > 65536:
            raise ValueError("response_too_large")
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.S | re.I)
        value = json.loads(fenced.group(1) if fenced else raw, object_pairs_hook=unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        if not isinstance(value, dict):
            raise ValueError("not_object")
        if set(value) != set(fallback)-{"parse_ok"}:
            errors.append("fields_mismatch")
        if type(value.get("candidate_id")) is not int or value["candidate_id"] != candidate_id:
            errors.append("candidate_id_mismatch")
        if value.get("verdict") not in ("defect_supported", "normal_supported", "insufficient_evidence"):
            errors.append("invalid_verdict")
        if value.get("visibility") not in ("sufficient", "insufficient"):
            errors.append("invalid_visibility")
        if value.get("reference_match") not in ("matched", "unmatched", "unavailable"):
            errors.append("invalid_reference_match")
        elif not reference and value["reference_match"] != "unavailable":
            errors.append("reference_without_image")
        # A decisive judgment must still have an actual textual justification.
        evidence = value.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append("missing_evidence")
        elif len(evidence) > 400:
            warnings.append("evidence_over_400_preserved")
        kind = value.get("defect_type")
        if not isinstance(kind, str):
            errors.append("invalid_defect_type_type")
        elif not kind.strip():
            if value.get("verdict") == "defect_supported":
                errors.append("missing_defect_type_for_defect")
            else:
                value["defect_type"] = "unknown"
                warnings.append("empty_defect_type_normalized")
        elif len(kind) > 400:
            warnings.append("defect_type_over_400_preserved")
    except (ValueError, TypeError) as exc:
        errors.append(str(exc) if str(exc) in ("duplicate_key", "response_too_large", "not_object", "nonfinite_json") else "invalid_json")
    audit = dict(parser_policy=POLICY, parse_errors=errors, parse_warnings=warnings)
    return (fallback if errors else {"parse_ok": True, **value}), audit


def repaired_prompt(category, candidate_id, reference=False):
    ref = ("Image 3 is a candidate normal TRAIN reference, not a guaranteed registered match. "
           "First check corresponding structure. If it does not match, do not use it to justify normality. "
           if reference else "No normal reference was provided; reference_match must be unavailable. ")
    return (f"Inspect industrial category {category}, candidate {candidate_id}. "
            "Image 1 is surrounding context and image 2 is detail of the SAME candidate near its center. " + ref +
            "Assess only the candidate. Describe observable local structure, then compare damage against "
            "plausible seams, reflections and normal texture. Tiny area alone is not grounds for dismissal. "
            "Set visibility to insufficient if edges or texture cannot be resolved; do not claim sufficient "
            "visibility while describing the image as unresolved or out of focus. If the structure is clear "
            "but its normality is uncertain, use sufficient visibility with insufficient_evidence. "
            "Do not force a defect or normal verdict. Return JSON only with exactly six fields: "
            f"candidate_id: integer {candidate_id}; verdict: defect_supported, normal_supported or insufficient_evidence; "
            "visibility: sufficient or insufficient; reference_match: matched, unmatched or unavailable; "
            "defect_type: a short string, use unknown when undetermined; evidence: a short nonempty observation "
            "and justification (prefer 1-2 sentences). No numerical confidence, masks or additional fields.")
