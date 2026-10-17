#!/usr/bin/env python3
"""Experiment G: frozen threshold sweep, audited GT, two independent GPU workers."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import hashlib
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
import torch

import mara_diagnose as shared
from tools.mara_agent import MARAAgent, MARAConfig, _resize_map
from tools.mara_diagnostics import checkpoint_config, map_metrics, move_inputs, validate_cache
from tools.mara_gate_sweep import PROTOCOL, METRICS, make_arms, score_batch, summarize_rows, trace_batch


def load_work(path):
    return shared.load_work(path, protocol=PROTOCOL)


def prepare(args):
    if args.gradient_images:
        raise ValueError("G is a frozen inference experiment; gradient_images must be 0")
    payload = shared.load_torch(args.mara_ckpt)
    cfg = checkpoint_config(payload)
    if not (cfg.use_gain_gate and cfg.hard_gain_gate and cfg.use_gain_lower_bound and cfg.stop_on_gain_reject):
        raise ValueError("G requires the checkpoint's hard lower-bound gate and stop_on_gain_reject")
    source = None
    if args.source_work_dir:
        root = Path(args.source_work_dir).resolve()
        if root == Path(args.work_dir).resolve():
            raise ValueError("SOURCE_WORK_DIR must differ from WORK_DIR")
        old = shared.load_json(root/"config.json")
        manifest = shared.load_json(root/"manifest.json")
        if old.get("protocol") not in ("mara_early_stop_ab_v1", "mara_early_stop_ab_v2", PROTOCOL):
            raise ValueError("Unsupported source cache protocol")
        if old["manifest_sha256"] != shared.digest(manifest):
            raise ValueError("Source manifest is corrupted")
        if old["checkpoint_sha256"] != shared.sha256_file(args.mara_ckpt):
            raise ValueError("Source cache uses a different MARA/Base checkpoint")
        source = dict(path=str(root), config_sha256=shared.sha256_file(root/"config.json"),
                      signature=shared.digest(old), manifest_sha256=old["manifest_sha256"])
    shared.prepare(args, protocol=PROTOCOL, arms=make_arms(args.gate_margins), extra_config=dict(
        category_shards=True, source_cache=source, export_batch_size=args.export_batch_size,
        sweep_batch_size=args.sweep_batch_size, decode_workers=args.decode_workers,
        metric_workers=args.metric_workers, empty_gt_policy=args.empty_gt_policy,
        precision="fp32_no_autocast_tf32_off", transition="freeze_noop_state_v1",
        experiment="G: vary safety margin only; no training, no target-based winner selection",
    ))


def evidence(root, cfg, record, signature):
    if not shared.cache_ready(root, record, signature):
        raise ValueError(f"Missing Base cache {record['id']}; run export first")
    path = root/"evidence"/f"{record['id']}.pt"
    metadata = shared.load_json(path.with_suffix(".json"))
    cache = shared.load_torch(path)
    if cache.get("signature") != metadata.get("payload_signature", signature) or cache.get("record_id") != record["id"]:
        raise ValueError("Evidence payload identity mismatch")
    validate_cache(cache, MARAConfig(**cfg["mara_config"]))
    expected = (1, 1, cfg["image_size"], cfg["image_size"])
    if cache["mask"].shape != expected or cache["labels"].shape != (1,) or int(cache["labels"][0]) not in (0, 1):
        raise ValueError("Invalid GT dimensions/label")
    return cache


def import_caches(root, cfg, manifest, signature, shard):
    source = cfg["source_cache"]
    if not source:
        return 0
    old_root = Path(source["path"])
    if shared.sha256_file(old_root/"config.json") != source["config_sha256"]:
        raise ValueError("Source configuration changed")
    old_cfg = shared.load_json(old_root/"config.json")
    old_manifest = shared.load_json(old_root/"manifest.json")
    if shared.digest(old_manifest) != source["manifest_sha256"]:
        raise ValueError("Source manifest changed")
    for key in ("image_size", "visual_backbone", "visual_layers", "text_source", "clip_model_name", "clip_pretrained"):
        if old_cfg["runtime"].get(key) != cfg["runtime"].get(key):
            raise ValueError(f"Incompatible source Base runtime: {key}")
    if (old_cfg.get("backbone_artifact") or {}).get("sha256") != (cfg.get("backbone_artifact") or {}).get("sha256"):
        raise ValueError("Source DINO weights differ")
    lookup = {r["id"]: r for r in old_manifest}
    reused = 0
    for record in manifest:
        if record["shard"] != shard or shared.cache_ready(root, record, signature):
            continue
        old_record = lookup.get(record["id"])
        if old_record is None:
            continue
        for key in ("dataset", "category", "image_path", "image_sha256", "mask_path", "mask_sha256"):
            if record[key] != old_record[key]:
                raise ValueError("Source image/mask identity differs")
        shared.verify_source(record)
        old_path = old_root/"evidence"/f"{record['id']}.pt"
        if not old_path.exists() or not old_path.with_suffix(".json").exists():
            continue
        old_meta = shared.load_json(old_path.with_suffix(".json"))
        if old_meta["signature"] != source["signature"] or shared.sha256_file(old_path) != old_meta["sha256"]:
            raise ValueError("Source evidence hash/signature mismatch")
        old_cache = shared.load_torch(old_path)
        if old_cache.get("record_id") != record["id"] or old_cache.get("signature") != old_meta.get("payload_signature", source["signature"]):
            raise ValueError("Source evidence payload mismatch")
        validate_cache(old_cache, MARAConfig(**cfg["mara_config"]))
        target = root/"evidence"/old_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+f".{os.getpid()}.tmp")
        try:
            try:
                os.link(old_path, temporary)
            except OSError:
                shutil.copyfile(old_path, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        shared.atomic_json(target.with_suffix(".json"), dict(
            signature=signature, payload_signature=old_cache["signature"], sha256=old_meta["sha256"],
            reused_from=str(old_path)))
        reused += 1
    print(f"Export shard {shard}: reused {reused} verified Base caches; source unchanged", flush=True)
    return reused


def export(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    import_caches(root, cfg, manifest, signature, args.shard_id)
    shared.export(args, protocol=PROTOCOL)


def audit(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    rows = []
    for record in manifest:
        shared.verify_source(record)
        cache = evidence(root, cfg, record, signature)
        label = int(cache["labels"][0])
        mask = cache["mask"].float()
        hr = int((mask > .5).sum())
        lr = int((_resize_map(mask, cfg["mara_config"]["map_size"], mode="nearest") > .5).sum())
        native_nonzero, native_max = None, None
        if record["mask_path"]:
            with Image.open(record["mask_path"]) as image:
                raw = np.asarray(image)
                native_nonzero, native_max = int(np.count_nonzero(raw)), int(raw.max())
        issue = ""
        if label and not hr:
            issue = ("missing_annotation" if record["mask_path"] is None else
                     "empty_native_annotation" if native_nonzero == 0 else "foreground_lost_during_preprocess_or_threshold")
        elif not label and hr:
            raise ValueError(f"Normal-labelled image has foreground GT: {record['image_path']}")
        included = not issue or cfg["empty_gt_policy"] == "flag"
        rows.append(dict(sample_id=record["id"], dataset=record["dataset"], category=record["category"],
                         image_path=record["image_path"], mask_path=record["mask_path"], image_label=label,
                         hr_gt_pixels=hr, lr_gt_pixels=lr, native_nonzero_pixels=native_nonzero,
                         native_max=native_max, issue=issue, included=included,
                         cache_sha256=shared.load_json(root/"evidence"/f"{record['id']}.json")["sha256"]))
    if not any(r["included"] for r in rows):
        raise ValueError("No valid images remain after the shared GT audit")
    result = dict(signature=signature, policy=cfg["empty_gt_policy"], rows=rows,
                  included=sum(r["included"] for r in rows), excluded=sum(not r["included"] for r in rows))
    path = root/"audit.json"
    if path.exists() and shared.load_json(path) != result:
        raise ValueError("GT audit changed; use a new WORK_DIR")
    shared.atomic_json(path, result)
    shared.write_csv(root/"results"/"data_audit.csv", rows)
    print(f"GT audit: included={result['included']} excluded={result['excluded']}; "
          "same sample set for every arm; annotations were NOT edited", flush=True)


def category_path(root, key):
    return root/"categories"/(shared.digest(key)[:24]+".json")


def read_category(path, signature, audit_hash, ids):
    if not path.exists() or not path.with_suffix(".sha256.json").exists():
        return None
    metadata = shared.load_json(path.with_suffix(".sha256.json"))
    if metadata["sha256"] != shared.sha256_file(path):
        raise ValueError(f"Corrupted category result: {path}")
    result = shared.load_json(path)
    if result["signature"] != signature or result["audit_hash"] != audit_hash or result["sample_ids"] != ids:
        raise ValueError("Category result belongs to another protocol/audit")
    return result


def finish_category(root, cfg, signature, audit_hash, key, records, traces, maps, masks, labels, base):
    """CPU metrics overlap the next GPU category; no huge per-image NPZ output."""
    dataset, category = key
    baseline = map_metrics(masks, base, labels, cfg["pro_num_th"], cfg["pro_max_fpr"])
    metrics = [dict(dataset=dataset, category=category, arm="Base", step=0, images=len(records), **baseline)]
    memo = {hashlib.sha256(base.tobytes()).hexdigest(): baseline}
    for arm in cfg["arms"]:
        for step in range(cfg["mara_config"]["max_steps"]):
            pred = maps[arm][:, step]
            checksum = hashlib.sha256(pred.tobytes()).hexdigest()
            if checksum not in memo:
                memo[checksum] = map_metrics(masks, pred, labels, cfg["pro_num_th"], cfg["pro_max_fpr"])
            score = memo[checksum]
            current = [r for r in traces if r["arm"] == arm and r["step"] == step]
            starts = {r["sample_id"]: r["before_hr_quality"] for r in traces if r["arm"] == arm and r["step"] == 0}
            metrics.append(dict(dataset=dataset, category=category, arm=arm, step=step+1, images=len(records),
                                mean_step_reward=shared.mean([r["reward"] for r in current]),
                                quality_gain_hr=shared.mean([r["after_hr_quality"]-starts[r["sample_id"]] for r in current]),
                                **score, **{"delta_"+k: score[k]-baseline[k] if score[k] is not None and baseline[k] is not None else None for k in METRICS}))
    value = dict(signature=signature, audit_hash=audit_hash, sample_ids=[r["id"] for r in records],
                 dataset=dataset, category=category, rows=traces, metrics=metrics)
    path = category_path(root, key)
    shared.atomic_json(path, value)
    shared.atomic_json(path.with_suffix(".sha256.json"), dict(sha256=shared.sha256_file(path)))
    print(f"Metrics complete {dataset}/{category}: images={len(records)}", flush=True)


def sweep(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    audit_data = shared.load_json(root/"audit.json")
    if audit_data["signature"] != signature:
        raise ValueError("Audit signature mismatch")
    audit_hash = shared.digest(audit_data)
    included = {r["sample_id"] for r in audit_data["rows"] if r["included"]}
    categories = defaultdict(list)
    for r in manifest:
        if r["shard"] == args.shard_id and r["id"] in included:
            categories[(r["dataset"], r["category"])].append(r)
    device = torch.device(args.device)
    agent = MARAAgent(MARAConfig(**cfg["mara_config"])).to(device).eval().requires_grad_(False)
    payload = shared.load_torch(cfg["mara_ckpt"])
    agent.load_state_dict(payload["mara_agent"], strict=True)
    del payload
    start = time.monotonic()
    batch_size = cfg["sweep_batch_size"]
    futures = []
    with ThreadPoolExecutor(max_workers=cfg["metric_workers"]) as pool:
        for key, records in sorted(categories.items()):
            if read_category(category_path(root, key), signature, audit_hash, [r["id"] for r in records]):
                print(f"Sweep reuse {key[0]}/{key[1]}", flush=True)
                continue
            if len(futures) >= cfg["metric_workers"]:
                futures.pop(0).result()
            caches = [evidence(root, cfg, r, signature) for r in records]
            base = np.stack([c["base_prob"][0, 1].numpy() for c in caches])
            masks = np.stack([c["mask"][0, 0].numpy() for c in caches]).astype(bool)
            labels = np.array([int(c["labels"][0]) for c in caches])
            maps = {arm: [] for arm in cfg["arms"]}
            rows = []
            position = 0
            while position < len(records):
                chunk = caches[position:position+batch_size]
                succeeded = False
                try:
                    packed = {k: torch.cat([c[k] for c in chunk], 0).to(device) for k in (
                        "base_prob", "base_logits", "layer_maps", "evidence_maps", "global_evidence")
                        if chunk[0].get(k) is not None}
                    mask = torch.cat([c["mask"] for c in chunk], 0).to(device)
                    image_labels = torch.cat([c["labels"] for c in chunk], 0).to(device)
                    chunk_rows, chunk_maps = [], {}
                    for arm, spec in cfg["arms"].items():
                        result = trace_batch(agent, packed, spec)
                        scored = score_batch(result, mask, image_labels, agent.cfg)
                        chunk_maps[arm] = np.stack([
                            np.stack([p[0, 1].cpu().numpy() for p in trace["maps"]]) for trace in scored])
                        for i, trace in enumerate(scored):
                            record = records[position+i]
                            for row in trace["records"]:
                                row.update(sample_id=record["id"], dataset=record["dataset"], category=record["category"],
                                           arm=arm, image_label=int(labels[position+i]))
                                if spec["freeze_noop_state"] and row["execute_weight"] == 0 and (
                                    row["global_change_max"] > cfg["quality_tolerance"] or row["pixel_change_max"] > cfg["quality_tolerance"]):
                                    raise RuntimeError("Unified no-op state invariant failed")
                                chunk_rows.append(row)
                        del result, scored
                    succeeded = True
                except torch.cuda.OutOfMemoryError:
                    if len(chunk) == 1:
                        raise
                    batch_size = max(1, len(chunk)//2)
                    print(f"Sweep OOM backoff batch={batch_size}; FP32 preserved", flush=True)
                if not succeeded:
                    packed = mask = image_labels = result = scored = trace = None
                    chunk_maps = chunk_rows = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                rows.extend(chunk_rows)
                for arm in maps:
                    maps[arm].append(chunk_maps[arm])
                position += len(chunk)
                del packed, mask, image_labels
            maps = {arm: np.concatenate(values, 0) for arm, values in maps.items()}
            futures.append(pool.submit(finish_category, root, cfg, signature, audit_hash, key, records,
                                       rows, maps, masks, labels, base))
            peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.
            print(f"Sweep GPU {args.shard_id} {key[0]}/{key[1]} batch<={batch_size} peak={peak:.2f}GiB", flush=True)
        for future in futures:
            future.result()
    shared.atomic_json(root/f"resources_shard_{args.shard_id}.json", dict(
        signature=signature, shard=args.shard_id, seconds=time.monotonic()-start,
        effective_batch_cap=batch_size, requested_batch=cfg["sweep_batch_size"],
        peak_allocated_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"))


def collect(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    audit_data = shared.load_json(root/"audit.json")
    if audit_data["signature"] != signature:
        raise ValueError("Audit signature mismatch")
    audit_hash = shared.digest(audit_data)
    included = {r["sample_id"] for r in audit_data["rows"] if r["included"]}
    audited_hashes = {r["sample_id"]: r["cache_sha256"] for r in audit_data["rows"]}
    categories = defaultdict(list)
    for r in manifest:
        if r["id"] in included:
            if not shared.cache_ready(root, r, signature) or shared.load_json(
                root/"evidence"/f"{r['id']}.json")["sha256"] != audited_hashes[r["id"]]:
                raise ValueError("Evidence changed after the audit")
            categories[(r["dataset"], r["category"])].append(r)
    rows, metrics = [], []
    for key, records in sorted(categories.items()):
        result = read_category(category_path(root, key), signature, audit_hash, [r["id"] for r in records])
        if result is None:
            raise ValueError(f"Missing category result {key}; refusing partial collection")
        expected = len(records)*len(cfg["arms"])*cfg["mara_config"]["max_steps"]
        if len(result["rows"]) != expected or len({(r["sample_id"], r["arm"], r["step"]) for r in result["rows"]}) != expected:
            raise ValueError("Incomplete/duplicate trajectory rows")
        rows.extend(result["rows"])
        metrics.extend(result["metrics"])
    groups = defaultdict(list)
    for r in metrics:
        groups[(r["dataset"], r["arm"], r["step"])].append(r)
    macro = []
    for (dataset, arm, step), values in sorted(groups.items()):
        cols = (*METRICS, *("delta_"+k for k in METRICS), "quality_gain_hr", "mean_step_reward")
        macro.append(dict(dataset=dataset, arm=arm, step=step, categories=len(values),
                          images=sum(r["images"] for r in values),
                          **{k: shared.mean([r.get(k) for r in values]) for k in cols},
                          **{"valid_categories_"+k: sum(r.get(k) is not None for r in values) for k in METRICS}))
    summaries = []
    for dataset in cfg["datasets"]:
        for arm, spec in cfg["arms"].items():
            for label, label_name in ((None, "all"), (0, "normal"), (1, "anomaly")):
                selected = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm and (label is None or r["image_label"] == label)]
                images = len({r["sample_id"] for r in selected})
                if images:
                    summaries.append(dict(dataset=dataset, arm=arm, stratum=label_name, **spec,
                                          **summarize_rows(selected, images, cfg["quality_tolerance"])))
    comparisons = []
    reference = {(r["dataset"], r["category"], r["step"]): r for r in metrics if r["arm"] == "G0"}
    for r in metrics:
        if r["arm"] in ("G0", "Base"):
            continue
        baseline = reference[(r["dataset"], r["category"], r["step"])]
        comparisons.append(dict(dataset=r["dataset"], category=r["category"], arm=r["arm"], step=r["step"],
                                **{k+"_minus_G0": r[k]-baseline[k] if r[k] is not None and baseline[k] is not None else None for k in METRICS}))
    out = root/"results"
    for filename, data in (("step_trace", rows), ("metrics", metrics), ("macro_metrics", macro),
                           ("gate_summary", summaries), ("comparisons", comparisons),
                           ("foundation_comparison", [r for r in comparisons if r["arm"] == "L0_legacy"]),
                           ("data_audit", audit_data["rows"])):
        shared.write_csv(out/(filename+".csv"), data)
    report = ["# MARA experiment G", "", "Frozen weights: NO TRAINING, NO VLM, NO target-based winner selection.",
              f"Checkpoint SHA256: {cfg['checkpoint_sha256']}",
              f"Selected={len(manifest)}; analyzed={len(included)}; excluded={audit_data['excluded']}; steps={cfg['mara_config']['max_steps']}",
              "G0 is the corrected production baseline. L0_legacy changes only the old no-op transition.",
              "Every G arm uses the same audited images and Base evidence. Policy STOP and spatial gates remain enabled.",
              "Only the predicted gain safety margin changes. Gain accept probability, reward/cost, ROI/layer policy stay fixed.",
              "GT enters only audit/post-hoc scoring. Empty-GT exclusions are protocol-wide, not selected by outcomes.",
              "PRO uses the existing sampled-threshold definition. Undefined metrics stay blank.",
              "Old caches, when reused, keep their original FP32 values; new export/trajectories use FP32 without autocast/TF32.",
              "The source checkpoint can have seen VisA test images. This is mechanism diagnosis, not independent validation.",
              "Do not choose a threshold/checkpoint on target-domain results. Report the complete predeclared curve.", "", "## Arms"]
    report.extend(f"- {name}: {spec}" for name, spec in cfg["arms"].items())
    report.extend(["", "## Final-step category macro metrics"])
    for r in macro:
        if r["arm"] == "Base" or r["step"] == cfg["mara_config"]["max_steps"]:
            report.append(f"- {r['dataset']} {r['arm']}: PRO={r['PRO']} P_AUROC={r['P_AUROC']} delta_PRO={r['delta_PRO']}")
    (out/"README.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    shared.atomic_json(root/"complete.json", dict(protocol=PROTOCOL, signature=signature, images=len(included),
        excluded=audit_data["excluded"], audit_sha256=audit_hash, metrics_sha256=shared.sha256_file(out/"metrics.csv")))
    target = root/"gate_report.zip"
    temporary = target.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.iterdir()):
                if path.is_file():
                    archive.write(path, "results/"+path.name)
            for name in ("config.json", "manifest.json", "audit.json", "complete.json"):
                archive.write(root/name, name)
            for path in sorted(root.glob("resources_shard_*.json")):
                archive.write(path, path.name)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Complete: {target} ({target.stat().st_size/1024**2:.2f}MiB); no large NPZ files required", flush=True)


def parser():
    p = shared.parser(stage_choices=("prepare", "export", "audit", "sweep", "collect"))
    p.description = __doc__
    p.set_defaults(gradient_images=0)
    p.add_argument("--gate_margins", default="0,-0.01,-0.03,-0.05,-0.10")
    p.add_argument("--source_work_dir", default="")
    p.add_argument("--export_batch_size", type=int, default=8)
    p.add_argument("--sweep_batch_size", type=int, default=8)
    p.add_argument("--decode_workers", type=int, default=2)
    p.add_argument("--metric_workers", type=int, default=2)
    p.add_argument("--empty_gt_policy", choices=("exclude", "flag"), default="exclude")
    return p


def main():
    args = parser().parse_args()
    if min(args.export_batch_size, args.sweep_batch_size, args.decode_workers, args.metric_workers,
           args.cpu_threads, args.num_shards, args.image_size) < 1 or args.shard_id < 0 or args.steps < 0 or args.samples_per_category < 0:
        raise ValueError("Invalid budget/batching/sharding arguments")
    if args.pro_num_th < 2 or not 0 < args.pro_max_fpr <= 1:
        raise ValueError("Invalid PRO parameters")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage != "prepare":
        _, cfg, _, _ = load_work(args.work_dir)
        if args.shard_id >= cfg["num_shards"]:
            raise ValueError("Shard outside prepared layout")
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
