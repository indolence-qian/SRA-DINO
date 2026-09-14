#!/usr/bin/env python3
"""Paired VLM input/prompt diagnosis using an immutable frozen-Base export.

prepare/review never read evaluation labels. Only evaluate opens GT. This is
selected-candidate diagnosis, not a new training run or an unbiased benchmark.
"""
import argparse
from collections import Counter, defaultdict
import csv
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from tools.local_vlm_review import action, calibrate_local, component_counts, local_prompt, parse_local
from tools.vlm_review import atomic_json, digest, file_digest, load_json

PROTOCOL = "local_vlm_diagnosis_v1"
VARIANTS = {"A_original": (False, False), "B_prompt": (True, False),
            "C_pixels": (False, True), "D_prompt_pixels": (True, True)}
CODE_FILES = ("vlm_diagnose.py", "tools/vlm_decision.py", "tools/local_vlm_review.py", "tools/vlm_review.py", "tools/local_vlm_protocol.py")


def neutral_prompt(category, candidate_id, reference=False):
    ref = ("Image 3 is a possible normal TRAIN reference. Use it only if location and structure match. "
           if reference else "There is no normal reference image. This alone does not imply poor visibility. ")
    return (
        f"Inspect industrial category {category}, candidate {candidate_id}. "
        "Image 1 is local context; image 2 is detail of the SAME candidate near its center. " + ref +
        "First describe visible material, edges and texture. Then judge this candidate only. "
        "Visibility means whether local structure can be resolved, NOT whether its normality is certain. "
        "A visible but ambiguous structure can have sufficient visibility and insufficient_evidence. "
        "Inspect tiny cracks, scratches, missing material and foreign matter without assuming they exist. "
        "Compare damage against plausible seams, texture, reflections and shadows. "
        "Use defect_supported for visible evidence of damage, normal_supported for positive evidence "
        "of normal structure, or insufficient_evidence when neither is supported. Do not force a defect verdict. "
        "Return one JSON object only, with exactly six fields. "
        f"candidate_id: integer {candidate_id}; "
        "verdict: one of defect_supported, normal_supported, insufficient_evidence; "
        "visibility: sufficient or insufficient; reference_match: matched, unmatched or unavailable "
        "(unavailable when there is no reference); defect_type: nonempty short string; "
        "evidence: nonempty string of at most 400 characters stating observations and the reason. "
        "Do not output numeric confidence or a mask."
    )


def original_prompt(record, candidate, config):
    return local_prompt(record["category"], candidate["roi_id"], "reference" in candidate,
                        generic=config.get("local_prompt") == "generic")


def key(record, candidate):
    return f"{record['id']}_{candidate['roi_id']}"


def safe_asset(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f"Missing/out-of-root source asset: {relative}")
    return path


def code_hashes():
    root = Path(__file__).resolve().parent
    return {p: file_digest(root / p) for p in CODE_FILES}


def select_candidates(records, count, seed, visible_keys=()):
    """Label-free round-robin over category and mask-area strata; include sentinels.

    Existing visible candidates are deliberately oversampled for diagnosis.
    No GT or label-bearing file paths are used to choose candidates.
    """
    pool = []
    for r in records:
        for c in r["rois"]:
            pool.append({"key": key(r, c), "sample_id": r["id"], "candidate_id": c["roi_id"],
                         "dataset": r["dataset"], "category": r["category"], "area": c["area"]})
    if len({x["key"] for x in pool}) != len(pool):
        raise ValueError("Duplicate candidate identifiers")
    if not pool:
        raise ValueError("No candidates in source export")
    order = lambda x: digest([seed, x["key"]])
    if count == 0 or count >= len(pool):
        return sorted(pool, key=order)
    sentinels = sorted([x for x in pool if x["key"] in visible_keys], key=order)[:count]
    chosen = {x["key"] for x in sentinels}
    buckets = defaultdict(list)
    for x in pool:
        if x["key"] not in chosen:
            area_bin = 0 if x["area"] <= 16 else 1 if x["area"] <= 64 else 2 if x["area"] <= 256 else 3
            buckets[(x["dataset"], x["category"], area_bin)].append(x)
    for values in buckets.values():
        values.sort(key=order, reverse=True)
    result = list(sentinels)
    while len(result) < count:
        for bucket in sorted(buckets):
            if buckets[bucket]:
                result.append(buckets[bucket].pop())
                if len(result) == count:
                    break
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(path, images, captions, heading):
    # Diagnostic evidence montage, never supplied to the teacher.
    canvas = Image.new("RGB", (320 * len(images), 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), heading, fill="black")
    for i, (im, caption) in enumerate(zip(images, captions)):
        preview = ImageOps.contain(im, (304, 292))
        canvas.paste(preview, (i*320+8, 35))
        draw.text((i*320+8, 332), caption, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def prepare(args):
    from dual_vlm import model_signature
    src, out = Path(args.source_dir).resolve(), Path(args.work_dir).resolve()
    if src == out or src.is_relative_to(out) or out.is_relative_to(src):
        raise ValueError("Diagnostic WORK_DIR must be separate from SOURCE_WORK_DIR")
    original = load_json(src / "config.json")
    manifest = load_json(src / "manifest.json")
    if not original.get("local_review") or manifest["fingerprint"] != digest(original):
        raise ValueError("Source must be a sealed local-VLM export with matching config")
    records = manifest["records"]
    visible, previous_hashes = set(), {}
    for r in records:
        if r["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Mixed source record fingerprints")
        for c in r["rois"]:
            p = src / "local_reviews" / f"{key(r, c)}.json"
            if p.exists():
                cached = load_json(p)
                if cached.get("fingerprint") != manifest["fingerprint"]:
                    raise ValueError(f"Stale source review: {p}")
                previous_hashes[str(p.relative_to(src))] = file_digest(p)
                d = cached.get("decision", {})
                if d.get("parse_ok") and d.get("visibility") == "sufficient":
                    visible.add(key(r, c))
    selected = select_candidates(records, args.candidates, args.seed, visible)
    keys = {x["key"] for x in selected}
    ids = {x["sample_id"] for x in selected}
    selected_records = [r for r in records if r["id"] in ids]
    assets = {"config.json": file_digest(src / "config.json"),
              "manifest.json": file_digest(src / "manifest.json"), **previous_hashes}
    for r in selected_records:
        for name in ("query", "evidence", "supports"):
            assets[r[name]] = file_digest(safe_asset(src, r[name]))
        for c in r["rois"]:
            if key(r, c) in keys:
                for name in ("context", "detail", "reference"):
                    if name in c:
                        assets[c[name]] = file_digest(safe_asset(src, c[name]))
    if not 1024 <= args.min_pixels <= original["teacher_image_size"] ** 2:
        raise ValueError("Diagnostic min_pixels must be between 1024 and the source max pixel budget")
    if args.num_shards < 1 or args.candidates < 0 or args.sanity_samples < 0:
        raise ValueError("Invalid diagnostic sample/shard count")
    cfg = dict(protocol=PROTOCOL, source_dir=str(src), original_config=original,
               model=model_signature(args.model_id or original["model"]["path"]),
               seed=args.seed, num_shards=args.num_shards, requested_candidates=args.candidates,
               min_pixels=args.min_pixels, sanity_samples=args.sanity_samples,
               code_hashes=code_hashes(), assets=assets, selected=selected, records=selected_records,
               selection="Label-free category/area stratification plus previous visible sentinels; biased diagnosis only",
               label_policy="GT opened only in evaluate; no GT in requests or selection")
    cfg["prompt_hashes"] = {}
    for r in selected_records:
        for c in r["rois"]:
            if key(r, c) in keys:
                cfg["prompt_hashes"][key(r, c)] = {
                    "original": digest(original_prompt(r, c, original)),
                    "neutral": digest(neutral_prompt(r["category"], c["roi_id"], "reference" in c))}
    path = out / "diagnosis_config.json"
    if path.exists() and load_json(path) != cfg:
        raise ValueError("Settings/code/source changed. Choose NEW WORK_DIR; old evidence is preserved.")
    atomic_json(path, cfg)
    write_csv(out / "selection.csv", selected)
    geometry_rows = []
    for r, c in entries(cfg):
        images = open_images(src, c)
        with Image.open(safe_asset(src, r["query"])) as im:
            query = im.convert("RGB")
        ImageDraw.Draw(query).rectangle(c["box"], outline="red", width=2)
        geom = c.get("geometry", {})
        geometry_rows.append(dict(key=key(r, c), dataset=r["dataset"], category=r["category"],
                                  mask_area=c["area"], detector_box=str(c["box"]),
                                  source_size=str(geom.get("source_size")), detail_box=str(geom.get("detail_box")),
                                  context_box=str(geom.get("context_box")),
                                  context_size=str(images[0].size), detail_size=str(images[1].size),
                                  context_sha256=assets[c["context"]], detail_sha256=assets[c["detail"]]))
        contact_sheet(out / "input_audit" / f"{key(r,c)}.png", [query, *images[:2]],
                      ["Detector query + candidate box", f"Context {images[0].size}", f"Detail {images[1].size}"], key(r,c))
    write_csv(out / "input_geometry.csv", geometry_rows)
    print(f"Prepared {len(selected)} paired candidates across {len(selected_records)} images. NO training.", flush=True)


def load_config(args):
    cfg = load_json(Path(args.work_dir) / "diagnosis_config.json")
    if cfg["protocol"] != PROTOCOL or cfg["code_hashes"] != code_hashes():
        raise ValueError("Diagnostic code changed: use NEW WORK_DIR")
    src = Path(cfg["source_dir"])
    for relative, expected in cfg["assets"].items():
        if file_digest(safe_asset(src, relative)) != expected:
            raise ValueError(f"Source changed: {relative}; do not mix experiments")
    return cfg


def entries(cfg):
    lookup = {key(r, c): (r, c) for r in cfg["records"] for c in r["rois"]}
    return [lookup[x["key"]] for x in cfg["selected"]]


def open_images(src, c):
    images = []
    for name in ("context", "detail", "reference"):
        if name in c:
            with Image.open(safe_asset(src, c[name])) as im:
                images.append(im.convert("RGB"))
    return images


def validate_review(path, cfg):
    payload = load_json(path)
    if payload.get("fingerprint") != digest(cfg):
        raise ValueError(f"Stale diagnostic review {path}; use NEW WORK_DIR")
    if "decision" in payload:
        if payload.get("key") != path.stem or payload.get("variant") != path.parent.name:
            raise ValueError(f"Mismatched candidate/arm: {path}")
        candidate = next(c for r,c in entries(cfg) if key(r,c) == path.stem)
        reparsed = parse_local(payload["raw_responses"][-1], candidate["roi_id"], "reference" in candidate,
                               policy=cfg.get("parser_policy", "legacy"))
        if reparsed != payload["decision"]:
            raise ValueError(f"Stored decision differs from raw response: {path}")
    for relative, expected in payload["trace_hashes"].items():
        if file_digest(safe_asset(path.parent, relative)) != expected:
            raise ValueError(f"Changed trace: {relative}")
    return payload


def review(args):
    from tools.vlm_decision import QwenVLLMTeacher
    from dual_vlm import model_signature
    cfg, out = load_config(args), Path(args.work_dir)
    if cfg.get("reparse_source"):
        raise ValueError("Offline replay outputs cannot start new reviews; use a new diagnostic experiment")
    if model_signature(cfg["model"]["path"]) != cfg["model"]:
        raise ValueError("Model files changed; use NEW WORK_DIR")
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard")
    neutral, high = VARIANTS[args.variant]
    selected = entries(cfg)[args.shard_id::cfg["num_shards"]]
    pending = []
    for r, c in selected:
        path = out / args.variant / f"{key(r,c)}.json"
        if path.exists() and validate_review(path, cfg)["decision"]["parse_ok"]:
            continue
        pending.append((r, c, path))
    if not pending:
        print(f"{args.variant} shard {args.shard_id}: cached/complete", flush=True)
        return
    original = cfg["original_config"]
    teacher = QwenVLLMTeacher(model_id=cfg["model"]["path"], gpu_memory_utilization=original["gpu_memory"],
                             max_model_len=original["max_model_len"], max_tokens=original["max_tokens"],
                             max_images=3, teacher_image_size=original["teacher_image_size"],
                             min_pixels=cfg["min_pixels"] if high else 1024)
    versions = {}
    for package in ("torch", "vllm", "transformers", "qwen-vl-utils"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    for i, (r, c, path) in enumerate(pending):
        images = open_images(cfg["source_dir"], c)
        prompt = (neutral_prompt(r["category"], c["roi_id"], "reference" in c) if neutral
                  else original_prompt(r, c, original))
        started, responses, trace_hashes = time.monotonic(), [], {}
        for attempt in range(original["retries"]+1):
            trace_dir = path.parent / "traces" / key(r,c) / str(attempt)
            raw = teacher.generate(prompt + (" Previous JSON was invalid; return all required fields." if attempt else ""),
                                   images, trace_dir=trace_dir)
            responses.append(raw)
            for asset in trace_dir.iterdir():
                if asset.is_file():
                    trace_hashes[str(asset.relative_to(path.parent))] = file_digest(asset)
            decision = parse_local(raw, c["roi_id"], "reference" in c)
            if decision["parse_ok"]:
                break
        atomic_json(path, dict(fingerprint=digest(cfg), key=key(r,c), variant=args.variant,
                              decision=decision, raw_responses=responses, prompt_sha256=digest(prompt),
                              versions=versions, seconds=time.monotonic()-started, trace_hashes=trace_hashes,
                              final_trace=str((trace_dir / "trace.json").relative_to(path.parent))))
        print(f"{args.variant} shard {args.shard_id} {i+1}/{len(pending)}: {decision['verdict']} / {decision['visibility']}", flush=True)


def sanity(args):
    """Image dependence probes, saved separately and NEVER used for calibration."""
    from tools.vlm_decision import QwenVLLMTeacher
    from dual_vlm import model_signature
    cfg, out = load_config(args), Path(args.work_dir)
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard")
    if model_signature(cfg["model"]["path"]) != cfg["model"]:
        raise ValueError("Model files changed; use NEW WORK_DIR")
    all_entries = sorted(entries(cfg), key=lambda rc: (-rc[1]["area"], key(*rc)))
    chosen = all_entries[:cfg["sanity_samples"]][args.shard_id::cfg["num_shards"]]
    if not chosen:
        return
    original, teacher = cfg["original_config"], None
    prompt = ("Describe the provided two images, not a hypothetical image. Image 1 is context and image 2 is detail. "
              "Return JSON with three short string fields: material_and_color, center_structure, visible_marks. "
              "If the images are blank say blank. Do not decide anomaly or invent invisible details.")
    for r, c in chosen:
        # Cross-category donor where possible; no correctness assumption is made.
        donor = next((rc for rc in all_entries if rc[0]["category"] != r["category"]), None)
        real = open_images(cfg["source_dir"], c)[:2]
        variants = {"correct": real, "blank": [Image.new("RGB", im.size, "gray") for im in real]}
        if donor:
            variants["swapped"] = open_images(cfg["source_dir"], donor[1])[:2]
        for mode, images in variants.items():
            path = out / "sanity" / f"{key(r,c)}_{mode}.json"
            if path.exists():
                validate_review(path, cfg)
                continue
            if teacher is None:
                teacher = QwenVLLMTeacher(model_id=cfg["model"]["path"], gpu_memory_utilization=original["gpu_memory"],
                                         max_model_len=original["max_model_len"], max_tokens=original["max_tokens"],
                                         max_images=3, teacher_image_size=original["teacher_image_size"], min_pixels=cfg["min_pixels"])
            trace_dir = path.parent / "traces" / path.stem
            raw = teacher.generate(prompt, images, trace_dir=trace_dir)
            atomic_json(path, dict(fingerprint=digest(cfg), key=key(r,c), mode=mode, raw_response=raw,
                                  donor=key(*donor) if mode == "swapped" else None,
                                  trace_hashes={str(p.relative_to(path.parent)): file_digest(p) for p in trace_dir.iterdir() if p.is_file()},
                                  note="Manual image-dependence audit only; not an anomaly accuracy result"))
            print(f"Sanity {path.stem} complete", flush=True)


def ratio(a, b):
    return a / b if b else None


def pixel_effect(variant, ident, base, gt, masks, votes, alpha):
    updated = calibrate_local(base, masks, votes, alpha=alpha)
    union, changed = masks.any(axis=0), updated != base
    if np.any(changed & ~union):
        raise AssertionError("Calibration escaped selected supports")
    return dict(variant=variant, sample_id=ident, support_pixels=int(union.sum()),
                enhanced_support_pixels=int(sum(m.sum() for m,v in zip(masks,votes) if action(v)=="enhance")),
                changed_pixels=int(changed.sum()), changed_gt_pixels=int((changed & gt).sum()),
                changed_background_pixels=int((changed & ~gt).sum()),
                newly_positive_gt_pixels=int(((base < .5) & (updated >= .5) & gt).sum()),
                newly_positive_background_pixels=int(((base < .5) & (updated >= .5) & ~gt).sum()),
                **component_counts(gt, union, base, updated))


def control_vote(enhance):
    return dict(parse_ok=True, visibility="sufficient", verdict="defect_supported" if enhance else "insufficient_evidence")


def matched_control(masks, votes, seed):
    """Match action count exactly and mask areas greedily, without reading GT.

    Exact area matching may be impossible; actual areas are always reported.
    One-candidate images can produce identical controls. No claim of an oracle.
    """
    rng = np.random.default_rng(seed)
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    targets = sorted([int(a) for a,v in zip(areas,votes) if action(v)=="enhance"])
    available, selected = list(rng.permutation(len(masks))), set()
    for target in targets:
        idx = min(available, key=lambda j: abs(int(areas[j])-target))
        selected.add(idx)
        available.remove(idx)
    return [control_vote(i in selected) for i in range(len(masks))]


def evaluate(args):
    cfg, out = load_config(args), Path(args.work_dir)
    src, chosen = Path(cfg["source_dir"]), entries(cfg)
    summaries, details, changes, strata = [], [], [], []
    # Require all four paired arms before exposing GT or emitting comparisons.
    cached = {variant: {key(r,c): validate_review(out / variant / f"{key(r,c)}.json", cfg)
                        for r,c in chosen} for variant in VARIANTS}
    labels, bases, supports_by_id, eval_hashes = {}, {}, {}, {}
    for r in cfg["records"]:
        gt_path = safe_asset(src, r["evaluation"])
        eval_hashes[r["evaluation"]] = file_digest(gt_path)
        with np.load(gt_path, allow_pickle=False) as data:
            labels[r["id"]] = data["mask"] > 0
        with np.load(safe_asset(src, r["evidence"]), allow_pickle=False) as data:
            bases[r["id"]] = data["base"]
        with np.load(safe_asset(src, r["supports"]), allow_pickle=False) as data:
            supports_by_id[r["id"]] = data["masks"].astype(bool)
    for variant in VARIANTS:
        counts, grouped, stratified, detail_tokens = Counter(), defaultdict(list), defaultdict(Counter), []
        for r,c in chosen:
            payload = cached[variant][key(r,c)]
            d, trace = payload["decision"], load_json(out / variant / payload["final_trace"])
            idx = next(i for i, x in enumerate(r["rois"]) if x["roi_id"] == c["roi_id"])
            support = supports_by_id[r["id"]][idx]
            gt = bool(labels[r["id"]][support].any())
            choice = action(d)
            counts["candidates"] += 1
            counts["parse_ok"] += bool(d["parse_ok"])
            counts["visible"] += d["visibility"] == "sufficient"
            counts[choice] += 1
            counts["gt_positive"] += gt
            counts["tp"] += gt and choice == "enhance"
            counts["fp"] += not gt and choice == "enhance"
            counts["tn"] += not gt and choice == "suppress"
            counts["fn"] += gt and choice == "suppress"
            counts["defect_blocked_by_visibility"] += d["verdict"] == "defect_supported" and choice == "keep"
            counts["length_terminated"] += trace["finish_reason"] == "length"
            detail_tokens.append(trace["processor_probe_tokens"][1])
            area_bin = "1-16" if c["area"] <= 16 else "17-64" if c["area"] <= 64 else "65-256" if c["area"] <= 256 else ">256"
            sc = stratified[(r["dataset"], r["category"], area_bin)]
            sc["candidates"] += 1
            sc["gt_positive"] += gt
            sc["visible"] += d["visibility"] == "sufficient"
            sc["enhance"] += choice == "enhance"
            sc["tp"] += gt and choice == "enhance"
            sc["fp"] += not gt and choice == "enhance"
            grouped[r["id"]].append((support, d))
            detail = dict(variant=variant, key=key(r,c), dataset=r["dataset"], category=r["category"],
                          area=c["area"], gt_positive=gt, **d, proposed_action=choice,
                          input_sizes=str(trace["input_sizes"]), post_vision_sizes=str(trace["post_vision_sizes"]),
                          probe_tokens=str(trace["processor_probe_tokens"]), finish_reason=trace["finish_reason"],
                          output_tokens=trace["output_tokens"], seconds=payload["seconds"])
            detail.update(parse_errors=";".join(payload.get("parse_errors", [])),
                          parse_warnings=";".join(payload.get("parse_warnings", [])))
            details.append(detail)
        n = counts["candidates"]
        for name in ("enhance", "suppress", "keep"):
            counts[name] += 0
        for (dataset, category, area_bin), sc in sorted(stratified.items()):
            strata.append(dict(variant=variant, dataset=dataset, category=category, mask_area_bin=area_bin, **dict(sc)))
        summaries.append(dict(variant=variant, **dict(counts),
                              invalid=n-counts["parse_ok"], median_detail_probe_tokens=float(np.median(detail_tokens)),
                              parse_rate=ratio(counts["parse_ok"], n), visibility_rate=ratio(counts["visible"], n),
                              keep_rate=ratio(counts["keep"], n), defect_recall=ratio(counts["tp"], counts["gt_positive"]),
                              defect_precision=ratio(counts["tp"], counts["tp"]+counts["fp"]),
                              false_enhance_rate=ratio(counts["fp"], n-counts["gt_positive"])))
        for ident, rows in grouped.items():
            base, gt = bases[ident], labels[ident]
            masks, votes = np.stack([x[0] for x in rows]), [x[1] for x in rows]
            alpha = cfg["original_config"]["alpha"]
            changes.append(pixel_effect(variant, ident, base, gt, masks, votes, alpha))
            control_seed = int(digest([cfg["seed"], ident, "area_matched_control"])[:8], 16)
            changes.append(pixel_effect(variant+"_area_matched_control", ident, base, gt, masks,
                                        matched_control(masks, votes, control_seed), alpha))
            if variant == "A_original":
                changes.append(pixel_effect("control_all_candidates", ident, base, gt, masks,
                                            [control_vote(True) for _ in masks], alpha))
                # GT-positive supports still contain background. This is a
                # diagnostic intervention, NOT a guaranteed performance bound.
                changes.append(pixel_effect("GT_overlap_diagnostic_ONLY", ident, base, gt, masks,
                                            [control_vote(bool(gt[m].any())) for m in masks], alpha))
    # Paired transitions expose interactions without selecting a winning arm automatically.
    paired = []
    for r,c in chosen:
        row = {"key": key(r,c), "category": r["category"], "area": c["area"]}
        for variant in VARIANTS:
            d = cached[variant][key(r,c)]["decision"]
            row[f"{variant}_verdict"] = d["verdict"]
            row[f"{variant}_visibility"] = d["visibility"]
            row[f"{variant}_action"] = action(d)
        paired.append(row)
    results = out / "results"
    write_csv(results / "summary.csv", summaries)
    write_csv(results / "candidate_audit.csv", details)
    write_csv(results / "paired_transitions.csv", paired)
    write_csv(results / "pixel_effects.csv", changes)
    write_csv(results / "area_category_audit.csv", strata)
    pixel_totals = defaultdict(Counter)
    for row in changes:
        for k,v in row.items():
            if k not in ("variant", "sample_id"):
                pixel_totals[row["variant"]][k] += v
    write_csv(results / "pixel_summary.csv", [dict(variant=k, **dict(v)) for k,v in pixel_totals.items()])
    note = ("Selected-candidate DIAGNOSIS, not full-dataset PRO/AUC or training. "
            "Existing-visible sentinels are oversampled. GT only used after all four review arms complete. "
            "Unselected candidates remain unchanged; normal verdicts are not applied in enhance-only mode. "
            "CPU processor token counts are probes, not vLLM internal telemetry. "
            "Area-matched controls match count exactly, area approximately; inspect actual area columns. "
            "GT_overlap_diagnostic_ONLY is non-deployable and not a guaranteed upper bound. "
            "Higher decision coverage alone is not improved detection accuracy.")
    atomic_json(results / "summary.json", dict(note=note, fingerprint=digest(cfg),
                summaries=summaries, pixel_totals={k: dict(v) for k,v in pixel_totals.items()}, evaluation_assets=eval_hashes))
    lines = [note, "", "variant,candidates,valid,visible,enhance,keep,tp,fp"]
    for row in summaries:
        lines.append(",".join(str(row.get(k, 0)) for k in ("variant", "candidates", "parse_ok", "visible", "enhance", "keep", "tp", "fp")))
    (results / "diagnosis.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "review", "sanity", "evaluate"), required=True)
    parser.add_argument("--source_dir", default="")
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--model_id", default="")
    parser.add_argument("--candidates", type=int, default=96, help="0 selects all; no labels used")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=256*256)
    parser.add_argument("--sanity_samples", type=int, default=6)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="A_original")
    args = parser.parse_args()
    {"prepare": prepare, "review": review, "sanity": sanity, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
