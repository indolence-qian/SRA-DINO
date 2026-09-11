"""Local-review integration; export, VLM review and GT evaluation stay separate."""
from collections import defaultdict
import csv
from pathlib import Path
import time

import numpy as np
from PIL import Image

from tools.local_vlm_review import (PROTOCOL, action, calibrate_local, component_counts,
                                    local_prompt, native_crops, parse_local, small_candidates)
from tools.vlm_review import atomic_json, digest, file_digest, load_json


def add_config(args, cfg):
    if not args.local_review:
        return
    if args.heatmap:
        raise ValueError("Local review uses clean native detail crops; VLM_HEATMAP must be 0")
    if not 0 < args.local_max_area <= args.local_total_area <= 0.1 or args.local_candidates < 1:
        raise ValueError("Invalid local candidate area/count limits")
    if not 0 <= args.local_min_probability <= 1 or not 0 <= args.suppress_alpha <= 2 or not 0 <= args.conflict_threshold <= 1:
        raise ValueError("Invalid local correction settings")
    if args.reference_pool < 1 or not 0 <= args.reference_distance <= 1:
        raise ValueError("Invalid normal reference settings")
    cfg.update({key: getattr(args, key) for key in (
        "local_review", "local_candidates", "local_max_area", "local_total_area", "local_min_probability",
        "local_input", "local_prompt", "normal_reference", "reference_pool", "reference_distance",
        "include_suppression", "suppress_alpha", "conflict_threshold")})
    cfg.update(protocol=PROTOCOL, correction_default="enhance_only", confidence_policy="no_self_reported_confidence",
               geometry="direct_resize_same_size_center_crop", component_area_bins=[0.001, 0.01],
               diagnostic_pixel_threshold=0.5, diagnostic_component_hit_fraction=0.1)
    cfg["reference_bank"] = reference_bank(cfg) if args.normal_reference else {}


def reference_bank(cfg):
    """Opt-in normal TRAIN images only. Never search evaluation images for refs."""
    from Datasets import DATASET_CLASSES, DATASET_REGISTRY
    bank = {}
    for name in cfg["datasets"].split():
        cls, splits, source = DATASET_REGISTRY[name]
        for category in sorted(DATASET_CLASSES[name]):
            dataset = cls(source=source, split=splits.TRAIN, classname=category,
                          resize=cfg["image_size"], imagesize=cfg["image_size"])
            normal_name = "Normal" if name == "visa" else "ok" if name == "btad" else "good"
            paths = sorted({str(Path(entry[2]).resolve()) for entry in dataset.data_to_iterate if entry[1] == normal_name})
            if not paths:
                raise ValueError(f"No normal TRAIN reference for {name}/{category}; disable NORMAL_REFERENCE")
            rng = np.random.default_rng(int(digest([cfg["seed"], name, category, "reference"])[:8], 16))
            chosen = sorted(rng.choice(paths, min(len(paths), cfg["reference_pool"]), replace=False).tolist())
            bank[f"{name}/{category}"] = [{"path": p, "sha256": file_digest(p)} for p in chosen]
    return bank


def descriptor(image):
    # Conservative appearance prefilter, NOT learned correspondence/registration.
    return np.asarray(image.resize((24, 24), Image.Resampling.BILINEAR), np.float32) / 255


def select_reference(cfg, record, candidate, query_source, context, shape):
    best = None
    query_hash = None
    for entry in cfg["reference_bank"].get(f"{record['dataset']}/{record['category']}", []):
        path = Path(entry["path"])
        if path.resolve() == Path(query_source).resolve():
            continue
        if query_hash is None:
            query_hash = file_digest(query_source)
        if entry["sha256"] == query_hash:
            continue  # no duplicated query presented as a known-normal reference
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Normal reference changed since prepare; use a new WORK_DIR")
        with Image.open(path) as image:
            crops, _ = native_crops(image.convert("RGB"), candidate["box"], shape, cfg["teacher_image_size"],
                                    cfg["local_input"] == "resized")
        distance = float(np.abs(descriptor(context) - descriptor(crops[0])).mean())
        if distance <= cfg["reference_distance"] and (best is None or distance < best[0]):
            best = distance, crops[1], entry
    return best


def export_local(root, record, base, disagreement, source, cfg):
    from dual_vlm import atomic_image, atomic_npz
    candidates, supports = small_candidates(base, disagreement, cfg["local_candidates"], cfg["local_max_area"],
                                             cfg["local_total_area"], cfg["local_min_probability"])
    record["rois"] = candidates
    record["supports"] = f"supports/{record['id']}.npz"
    atomic_npz(root / record["supports"], masks=supports)
    with Image.open(source) as image:
        raw = image.convert("RGB")
    for candidate in candidates:
        images, geometry = native_crops(raw, candidate["box"], base.shape, cfg["teacher_image_size"],
                                        cfg["local_input"] == "resized")
        prefix = f"local_crops/{record['id']}_{candidate['roi_id']}"
        candidate.update(context=prefix+"_context.png", detail=prefix+"_detail.png", geometry=geometry)
        atomic_image(root / candidate["context"], np.asarray(images[0]))
        atomic_image(root / candidate["detail"], np.asarray(images[1]))
        if cfg["normal_reference"]:
            ref = select_reference(cfg, record, candidate, source, images[0], base.shape)
            if ref is not None:
                distance, image, entry = ref
                candidate.update(reference=prefix+"_reference.png", reference_distance=distance,
                                 reference_source_sha256=entry["sha256"])
                atomic_image(root / candidate["reference"], np.asarray(image))
    record["local_assets"] = [record["supports"]] + [r[k] for r in candidates for k in ("context", "detail", "reference") if k in r]


def review_path(root, record, candidate):
    return root / "local_reviews" / f"{record['id']}_{candidate['roi_id']}.json"


def check_cached(payload, cfg, record, candidate):
    if payload.get("fingerprint") != digest(cfg) or payload.get("id") != record["id"] or payload.get("candidate_id") != candidate["roi_id"]:
        raise ValueError("Stale/mismatched local review; do not mix caches")
    # Reparse structured content, not just an untrusted parse_ok flag.
    value = dict(payload["decision"])
    expected = value.pop("parse_ok")
    import json
    parsed = parse_local(json.dumps(value), candidate["roi_id"], "reference" in candidate)
    if bool(expected) != parsed["parse_ok"]:
        raise ValueError("Local cached decision schema is inconsistent")
    return payload


def review(args):
    from dual_vlm import manifest
    from tools.vlm_decision import QwenVLLMTeacher
    cfg, root, records = manifest(args)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Bad shard index")
    pending = []
    for record in records[args.shard_id::cfg["num_shards"]]:
        for candidate in record["rois"]:
            path = review_path(root, record, candidate)
            if path.exists() and check_cached(load_json(path), cfg, record, candidate)["decision"]["parse_ok"]:
                continue
            pending.append((record, candidate))
    if not pending:
        print(f"[local review {args.shard_id}] complete or no eligible small candidates", flush=True)
        return
    teacher = QwenVLLMTeacher(model_id=cfg["model"]["path"], gpu_memory_utilization=cfg["gpu_memory"],
                             max_model_len=cfg["max_model_len"], max_tokens=cfg["max_tokens"],
                             max_images=3, teacher_image_size=cfg["teacher_image_size"])
    for i, (record, candidate) in enumerate(pending):
        images = []
        for key in ("context", "detail", "reference"):
            if key in candidate:
                with Image.open(root / candidate[key]) as image:
                    images.append(image.convert("RGB"))
        prompt = local_prompt(record["category"], candidate["roi_id"], "reference" in candidate,
                              cfg["local_prompt"] == "generic")
        started, responses = time.monotonic(), []
        for attempt in range(cfg["retries"]+1):
            raw = teacher.generate(prompt + (" Previous JSON was invalid; return all required fields." if attempt else ""), images)
            responses.append(raw)
            decision = parse_local(raw, candidate["roi_id"], "reference" in candidate)
            if decision["parse_ok"]:
                break
        atomic_json(review_path(root, record, candidate), {"fingerprint": digest(cfg), "id": record["id"],
                    "candidate_id": candidate["roi_id"], "decision": decision, "raw_responses": responses,
                    "seconds": time.monotonic()-started})
        print(f"[local review {args.shard_id}] {i+1}/{len(pending)} {decision['verdict']} valid={decision['parse_ok']}", flush=True)


def evaluate(args):
    """P0 decision audit + P1 support-restricted comparisons, without new calls."""
    from dual_vlm import manifest
    from test2 import compute_best_f1, compute_i_auroc, compute_p_auroc, compute_pro
    cfg, root, records = manifest(args)
    cached, invalid, total = {}, 0, 0
    for record in records:
        cached[record["id"]] = []
        for candidate in record["rois"]:
            path = review_path(root, record, candidate)
            if not path.exists():
                raise ValueError(f"Missing candidate review {path.name}; resume review")
            value = check_cached(load_json(path), cfg, record, candidate)
            cached[record["id"]].append(value)
            invalid += not value["decision"]["parse_ok"]
            total += 1
    if invalid / max(1, total) > cfg["max_invalid_ratio"]:
        raise ValueError(f"Invalid local reviews {invalid}/{total}; inspect raw output and resume")
    modes = ["base", "enhance", "control_enhance"]
    if cfg["include_suppression"]:
        modes += ["suppress", "bidirectional", "control_suppress"]
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["category"])].append(record)
    rows, diagnostics, decisions = [], [], []
    for (dataset, category), group in groups.items():
        ground_truth, predictions = [], {m: [] for m in modes}
        stats = defaultdict(float)
        for record in group:
            with np.load(root / record["evidence"], allow_pickle=False) as data:
                base = data["base"]
            with np.load(root / record["supports"], allow_pickle=False) as data:
                supports = data["masks"].astype(bool)
            # Evaluation labels are opened ONLY here, after all decisions exist.
            with np.load(root / record["evaluation"], allow_pickle=False) as data:
                mask = data["mask"] > 0
            entries = cached[record["id"]]
            votes = [v["decision"] for v in entries]
            union = supports.any(axis=0)
            ground_truth.append(mask)
            stats["samples"] += 1
            stats["anomaly_samples"] += bool(mask.any())
            stats["candidates"] += len(votes)
            stats["review_seconds"] += sum(e["seconds"] for e in entries)
            stats["no_candidate_images"] += len(votes) == 0
            stats["candidate_pixels"] += union.sum()
            stats["defect_pixels"] += mask.sum()
            stats["covered_defect_pixels"] += (mask & union).sum()
            for support, candidate, vote in zip(supports, record["rois"], votes):
                gt = bool(mask[support].any())
                choice = action(vote)
                stats["abstain"] += choice == "keep"
                if choice != "keep":
                    stats["tp" if gt and choice == "enhance" else "fn" if gt else "fp" if choice == "enhance" else "tn"] += 1
                # Include abstentions in defect-candidate recall denominator.
                stats["gt_positive_candidates"] += gt
                decisions.append({"dataset": dataset, "category": category, "sample_id": record["id"],
                                  "candidate_id": candidate["roi_id"], "area": candidate["area"],
                                  "has_reference": "reference" in candidate, "parse_ok": vote["parse_ok"],
                                  "verdict": vote["verdict"], "visibility": vote["visibility"], "proposed_action": choice,
                                  "reference_match": vote["reference_match"], "gt_contains_defect": gt,
                                  "context_crop": candidate["context"], "detail_crop": candidate["detail"],
                                  "evidence": vote["evidence"]})
            for mode in modes:
                updated = base if mode == "base" else calibrate_local(base, supports, votes, mode, cfg["alpha"],
                                            cfg["suppress_alpha"], cfg["conflict_threshold"])
                predictions[mode].append(updated)
                changed = updated != base
                # Exact outside-support identity is an invariant, not a metric target.
                if np.any(changed & ~union):
                    raise AssertionError("Local correction escaped candidate support")
                stats[f"{mode}_changed_pixels"] += changed.sum()
                stats[f"{mode}_changed_gt_pixels"] += (changed & mask).sum()
                stats[f"{mode}_new_fn_pixels_at_0_5"] += ((base >= .5) & (updated < .5) & mask).sum()
                stats[f"{mode}_removed_fp_pixels_at_0_5"] += ((base >= .5) & (updated < .5) & ~mask).sum()
                for key, value in component_counts(mask, union, base, updated).items():
                    stats[f"{mode}_{key}"] += value
            stats["pixels"] += base.size
        masks = np.stack(ground_truth)
        for key in ("tp", "fp", "tn", "fn", "gt_positive_candidates", "abstain"):
            stats[key] += 0
        labels = masks.reshape(len(group), -1).any(axis=1)
        row = {"dataset": dataset, "category": category, "samples": len(group)}
        for mode in modes:
            maps = np.stack(predictions[mode])
            values = {"PRO": compute_pro(masks, maps, cfg["pro_num_th"], cfg["pro_max_fpr"]),
                      "P_AUROC": compute_p_auroc(masks, maps), "I_AUROC": compute_i_auroc(labels, maps.reshape(len(group), -1).max(axis=1)),
                      "F1": compute_best_f1(masks, maps)}
            row.update({f"{mode}_{k}": v for k, v in values.items()})
        rows.append(row)
        def ratio(a, b):
            return stats[a] / stats[b] if stats[b] else None
        diag = {"dataset": dataset, "category": category, **dict(stats)}
        diag.update(candidate_pixel_recall=ratio("covered_defect_pixels", "defect_pixels"),
                    candidate_area_fraction=ratio("candidate_pixels", "pixels"),
                    abstain_fraction=ratio("abstain", "candidates"),
                    defect_candidate_recall=ratio("tp", "gt_positive_candidates"),
                    defect_precision=stats["tp"]/(stats["tp"]+stats["fp"]) if stats["tp"]+stats["fp"] else None,
                    normal_verdict_defect_fraction=stats["fn"]/(stats["tn"]+stats["fn"]) if stats["tn"]+stats["fn"] else None)
        decisive = stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"]
        diag["decisive_accuracy"] = (stats["tp"] + stats["tn"]) / decisive if decisive else None
        for mode in modes:
            diag[f"{mode}_changed_pixel_fraction"] = ratio(f"{mode}_changed_pixels", "pixels")
            diag[f"{mode}_changed_pixel_gt_fraction"] = ratio(f"{mode}_changed_gt_pixels", f"{mode}_changed_pixels")
            for size in ("small", "medium", "large"):
                for name in ("candidate", "base", "updated"):
                    diag[f"{mode}_{size}_{name}_recall"] = ratio(f"{mode}_{size}_{name}_hit", f"{mode}_{size}_count")
        diagnostics.append(diag)
        print(f"[local eval] {dataset}/{category} PRO {row['base_PRO']:.5f}->{row['enhance_PRO']:.5f}", flush=True)
    means = {}
    for dataset in cfg["datasets"].split():
        selected = [r for r in rows if r["dataset"] == dataset]
        means[dataset] = {f"{mode}_{key}": float(np.mean([r[f"{mode}_{key}"] for r in selected]))
                          for mode in modes for key in ("PRO", "P_AUROC", "I_AUROC", "F1")}
    note = ("Frozen dual tower + single-candidate local VLM; NO training/MARA. Default enhance only. "
            "Controls use the SAME masks without VLM. GT used only in evaluation. Source pilot may overlap Base training. "
            "F1 is best test-threshold reporting, not a deployment threshold. Zero candidates preserve Base.")
    lines = [note, f"Invalid candidate reviews: {invalid}/{total}", ""]
    for dataset, values in means.items():
        lines += [f"Dataset: {dataset} (category macro mean)", "Mode                  PRO       P-AUC     I-AUC     F1"]
        for mode in modes:
            lines.append(f"{mode:<22}" + " ".join(f"{values[f'{mode}_{key}']:.5f}" for key in ("PRO", "P_AUROC", "I_AUROC", "F1")))
    out = root / "results"
    atomic_json(out / "metrics.json", {"config": cfg, "fingerprint": digest(cfg), "means": means, "categories": rows,
                "diagnostics": diagnostics, "invalid_reviews": invalid, "candidate_reviews": total, "protocol_note": note})
    for name, data in (("metrics.csv", rows), ("diagnostics.csv", diagnostics), ("candidate_decisions.csv", decisions)):
        with open(out / name, "w", newline="", encoding="utf-8") as stream:
            if data:
                fields = list(dict.fromkeys(k for r in data for k in r))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
    (out / "metric_vlm.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
