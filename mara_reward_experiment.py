#!/usr/bin/env python3
"""R: paired, source-only MARA retraining with a classification-reward ablation."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch

import mara_diagnose as shared
import mara_gate_sweep as gate
from tools.mara_agent import MARAConfig
from tools.mara_diagnostics import checkpoint_config
from tools.mara_gate_sweep import METRICS, score_batch, summarize_rows, trace_batch
from tools.mara_reward import (
    ARMS, INPUTS, PROTOCOL, CachedBatches, EvidenceStore, arm_config, initial_agent,
    job_list, options, scheduler_for, split_source, state_hash,
)


def load_work(path):
    return shared.load_work(path, protocol=PROTOCOL)


def prepare(args):
    seeds = [int(x) for x in args.train_seeds.replace(",", " ").split()]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("TRAIN_SEEDS must contain unique nonnegative integers")
    template = checkpoint_config(shared.load_torch(args.mara_ckpt))
    if not (template.use_gain_gate and template.hard_gain_gate and template.use_gain_lower_bound
            and template.stop_on_gain_reject and template.include_base_trajectory):
        raise ValueError("R requires gain/lower/hard gates, stop-on-reject and a Base anchor")
    if args.steps or args.gradient_images or args.source_dataset not in args.datasets.split():
        raise ValueError("R preserves the checkpoint horizon, uses no gradient probe, and needs source_dataset in DATASETS")
    if args.epochs <= args.warmup_epochs:
        raise ValueError("EPOCHS must exceed WARMUP_EPOCHS so policy learning actually runs")
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("Invalid calibration fraction")
    source = None
    if args.source_work_dir:
        path = Path(args.source_work_dir).resolve()
        old = shared.load_json(path/"config.json")
        old_manifest = shared.load_json(path/"manifest.json")
        if path == Path(args.work_dir).resolve():
            raise ValueError("SOURCE_WORK_DIR must differ from WORK_DIR")
        if old.get("protocol") not in ("mara_gate_sweep_v1", "mara_early_stop_ab_v1", "mara_early_stop_ab_v2", PROTOCOL):
            raise ValueError("Unsupported source cache protocol")
        if old["manifest_sha256"] != shared.digest(old_manifest) or old["checkpoint_sha256"] != shared.sha256_file(args.mara_ckpt):
            raise ValueError("Source cache manifest/checkpoint mismatch")
        source = dict(path=str(path), signature=shared.digest(old),
                      config_sha256=shared.sha256_file(path/"config.json"), manifest_sha256=old["manifest_sha256"])
    train_args = options(args.epochs, template, args.train_batch_size)
    train_args.gain_warmup_epochs = args.warmup_epochs
    initial_hashes = {str(seed): state_hash(initial_agent(arm_config(template, "R0"), seed).state_dict()) for seed in seeds}
    shared.prepare(args, protocol=PROTOCOL, arms={k: dict(reward_cls_weight=v) for k, v in ARMS.items()}, extra_config=dict(
        source_cache=source, dataset_sample_limits={args.source_dataset: args.source_samples_per_category},
        source_dataset=args.source_dataset, calibration_fraction=args.calibration_fraction,
        source_split_seed=args.source_split_seed, train_seeds=seeds, jobs=job_list(seeds),
        training=vars(train_args), initial_state_hashes=initial_hashes,
        arm_configs={arm: asdict(arm_config(template, arm)) for arm in ARMS},
        category_shards=False, empty_gt_policy="exclude", export_batch_size=args.export_batch_size,
        decode_workers=args.decode_workers, eval_batch_size=args.eval_batch_size, metric_workers=args.metric_workers,
        cpu_cache_gb=args.cpu_cache_gb, gpu_cache_gb=args.gpu_cache_gb,
        precision="fp32_no_autocast_tf32_off", transition="freeze_noop_state_v1",
        selection_note="Source-only stratified split; fixed final epoch; no target checkpoint/threshold selection",
        gradient_note="Fresh matched initialization; canonical train_mara rollout/replay losses; no VLM",
        sampling_note="Full source by default; paired metadata-stratified target subset; diagnostic, not final benchmark",
    ))
    print(f"R jobs={len(seeds)*3}; cls rewards={ARMS}; fresh matched initialization; no VLM", flush=True)


def export(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    gate.import_caches(root, cfg, manifest, signature, args.shard_id)
    shared.export(args, protocol=PROTOCOL)


def audit(args):
    gate.audit(args, protocol=PROTOCOL)
    root, cfg, manifest, signature = load_work(args.work_dir)
    audited = shared.load_json(root/"audit.json")
    split = split_source(manifest, audited, cfg["source_dataset"], cfg["source_split_seed"], cfg["calibration_fraction"])
    split.update(signature=signature, audit_sha256=shared.digest(audited))
    path = root/"split.json"
    if path.exists() and shared.load_json(path) != split:
        raise ValueError("Source split changed; use a new WORK_DIR")
    shared.atomic_json(path, split)
    print(f"Source train={len(split['train_ids'])} calibration={len(split['calibration_ids'])}; "
          f"targets={len(split['target_ids'])}; target GT is never used for training", flush=True)


def checked_split(root, signature):
    split = shared.load_json(root/"split.json")
    audited = shared.load_json(root/"audit.json")
    if split["signature"] != signature or split["audit_sha256"] != shared.digest(audited):
        raise ValueError("Split/audit signature mismatch")
    return split, audited


def resume_payload(directory, identity):
    pointer = directory/"resume.json"
    if not pointer.exists():
        return None
    meta = shared.load_json(pointer)
    path = directory/"resume"/meta["filename"]
    if path.parent.resolve() != (directory/"resume").resolve() or shared.sha256_file(path) != meta["sha256"]:
        raise ValueError("Resume checkpoint path/hash mismatch")
    payload = shared.load_torch(path)
    if payload["identity"] != identity or payload["epoch"] != meta["epoch"]:
        raise ValueError("Resume checkpoint belongs to another experiment/job/split")
    return payload


def save_resume(directory, identity, epoch, agent, optimizer, scheduler, history, device):
    payload = dict(identity=identity, epoch=epoch, mara_agent=agent.state_dict(),
                   optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    path = directory/"resume"/f"epoch_{epoch:03d}.pth"
    shared.atomic_torch(path, payload)
    # The atomic pointer is committed LAST: interrupted writes retain the previous
    # complete epoch, including optimizer, scheduler and stochastic policy RNG.
    shared.atomic_json(directory/"resume.json", dict(filename=path.name, sha256=shared.sha256_file(path), epoch=epoch))


def train_job(root, cfg, manifest, signature, split, store, job, device):
    from train_mara import train_one_epoch
    from types import SimpleNamespace
    directory = root/"jobs"/job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(signature=signature, job=job, split_sha256=shared.digest(split),
                    initial_state_sha256=cfg["initial_state_hashes"][str(job["seed"])])
    agent_cfg = MARAConfig(**cfg["arm_configs"][job["arm"]])
    agent = initial_agent(agent_cfg, job["seed"])
    if state_hash(agent.state_dict()) != identity["initial_state_sha256"]:
        raise ValueError("Matched initial MARA state changed")
    agent = agent.to(device)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(job["seed"])
        torch.cuda.reset_peak_memory_stats(device)
    train_args = SimpleNamespace(**cfg["training"])
    ids = set(split["train_ids"])
    records = sorted((r for r in manifest if r["id"] in ids), key=lambda r: r["id"])
    optimizer = torch.optim.AdamW(agent.parameters(), lr=train_args.mara_lr,
                                 weight_decay=train_args.mara_weight_decay, betas=(.9, .999))
    batches = math.ceil(len(records)/train_args.batch_size)
    scheduler = scheduler_for(optimizer, train_args.epoch, batches, train_args)
    saved = resume_payload(directory, identity)
    history, first_epoch = [], 0
    if saved:
        if not 0 <= saved["epoch"] < train_args.epoch or len(saved["history"]) != saved["epoch"]+1:
            raise ValueError("Resume epoch/history is incomplete or outside the sealed budget")
        agent.load_state_dict(saved["mara_agent"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, first_epoch = saved["history"], saved["epoch"]+1
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        print(f"Resume {job['id']} at epoch {first_epoch+1}; full optimizer/scheduler/RNG restored", flush=True)
        del saved
    start = time.monotonic()
    for epoch in range(first_epoch, train_args.epoch):
        loader = CachedBatches(store, records, train_args.batch_size, job["seed"], epoch)
        try:
            meters = train_one_epoch(epoch, agent, optimizer, scheduler, None, None, None, None, None,
                                     loader, device, train_args)
        except torch.cuda.OutOfMemoryError:
            # Replaying part of an already-updated batch would alter the protocol.
            # Fail rather than silently halve batch/group or skip examples.
            raise RuntimeError("R training OOM: no batch/group reduction or sample skipping. "
                               "Use --stage train with --gpu_cache_gb 0 on this sealed directory to resume the last committed epoch.") from None
        row = dict(job=job["id"], arm=job["arm"], seed=job["seed"], epoch=epoch+1,
                   optimizer_steps=(epoch+1)*batches*train_args.grpo_update_epochs, **meters)
        if not all(math.isfinite(v) for v in meters.values()):
            raise ValueError(f"Nonfinite epoch metrics: {job['id']} epoch {epoch+1}")
        history.append(row)
        save_resume(directory, identity, epoch, agent, optimizer, scheduler, history, device)
        shared.write_csv(directory/"training.csv", history)
        print(f"\n{job['id']} committed epoch={epoch+1}/{train_args.epoch} loss={meters['loss']:.6f}", flush=True)
    final = directory/"final.pth"
    training_path = directory/"training.json"
    if not final.exists() or not training_path.exists():
        # A self-contained final checkpoint remains usable by the ordinary tester.
        # The old trained policy/optimizer are replaced, not used as initialization.
        payload = shared.load_torch(cfg["mara_ckpt"])
        payload.update(epoch=train_args.epoch-1, mara_agent={k: v.detach().cpu() for k, v in agent.state_dict().items()},
                       mara_config=asdict(agent_cfg), optimizer=optimizer.state_dict(),
                       reward_experiment=identity)
        shared.atomic_torch(final, payload)
        del payload
        shared.atomic_json(training_path, dict(**identity, final_sha256=shared.sha256_file(final),
            final_state_sha256=state_hash(agent.state_dict()),
            epochs=train_args.epoch, history=history, seconds_this_invocation=time.monotonic()-start,
            effective_batch=train_args.batch_size, group_size=agent_cfg.group_size,
            cpu_cache_gb=store.cpu_bytes/2**30, gpu_cache_gb=store.gpu_bytes/2**30,
            peak_allocated_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.))
    trained = shared.load_json(training_path)
    if (any(trained.get(k) != v for k, v in identity.items()) or shared.sha256_file(final) != trained["final_sha256"]
            or trained["final_state_sha256"] != state_hash(agent.state_dict())):
        raise ValueError("Final checkpoint identity/hash mismatch")
    return directory, agent.eval().requires_grad_(False), trained


def evaluate_job(directory, agent, trained, root, cfg, manifest, signature, split, store, job):
    eval_signature = shared.digest(dict(training=trained["final_sha256"], split=shared.digest(split), signature=signature))
    audit_hash = split["audit_sha256"]
    calibration, targets = set(split["calibration_ids"]), set(split["target_ids"])
    categories = defaultdict(list)
    for record in manifest:
        if record["id"] in calibration | targets:
            categories[(record["dataset"], record["category"])].append(record)
    spec = dict(gain_safety_margin=0., bypass_gain=False, freeze_noop_state=True)
    eval_cfg = {**cfg, "arms": {job["arm"]: spec}, "mara_config": asdict(agent.cfg)}
    # Asynchronous category scoring, bounded to metric_workers categories in RAM.
    futures = []
    with ThreadPoolExecutor(max_workers=cfg["metric_workers"]) as pool:
        for key, records in sorted(categories.items()):
            if gate.read_category(gate.category_path(directory, key), eval_signature, audit_hash, [r["id"] for r in records]):
                continue
            if len(futures) >= cfg["metric_workers"]:
                futures.pop(0).result()
            rows, maps, masks, labels, bases = [], [], [], [], []
            for offset in range(0, len(records), cfg["eval_batch_size"]):
                chunk = records[offset:offset+cfg["eval_batch_size"]]
                batch = store.batch(chunk)
                # This function accepts no GT, and returns the full trajectory first.
                result = trace_batch(agent, {k: batch[k] for k in INPUTS if k in batch}, spec)
                scored = score_batch(result, batch["mask"], batch["labels"], agent.cfg)
                chunk_labels = batch["labels"].cpu().tolist()
                for index, (record, trace) in enumerate(zip(chunk, scored)):
                    maps.append(np.stack([p[0, 1].cpu().numpy() for p in trace["maps"]]))
                    for row in trace["records"]:
                        row.update(sample_id=record["id"], dataset=record["dataset"], category=record["category"],
                                   arm=job["arm"], image_label=int(chunk_labels[index]),
                                   role="source_calibration" if record["id"] in calibration else "target")
                        reference = cfg["arm_configs"]["R0"]
                        row["reward_reference_R0"] = (reference["reward_cls_weight"]*row["delta_q_cls"]
                            +reference["reward_loc_weight"]*row["delta_q_loc"]
                            +reference["reward_conf_weight"]*row["delta_q_conf"]
                            -reference["reward_fp_weight"]*row["delta_q_bg"]-row["step_cost"]-row["refine_cost"])
                        if row["execute_weight"] == 0 and (
                            row["global_change_max"] > cfg["quality_tolerance"] or row["pixel_change_max"] > cfg["quality_tolerance"]):
                            raise RuntimeError("R no-op state invariant failed")
                        rows.append(row)
                masks.extend(batch["mask"][:, 0].cpu().numpy().astype(bool))
                labels.extend(chunk_labels)
                bases.extend(batch["base_prob"][:, 1].cpu().numpy())
                del result, scored, batch
            futures.append(pool.submit(gate.finish_category, directory, eval_cfg, eval_signature, audit_hash, key, records,
                                       rows, {job["arm"]: np.stack(maps)}, np.stack(masks), np.array(labels), np.stack(bases)))
        for future in futures:
            future.result()
    shared.atomic_json(directory/"evaluated.json", dict(signature=eval_signature, job=job,
        final_sha256=trained["final_sha256"], split_sha256=shared.digest(split)))


def train(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, audited = checked_split(root, signature)
    hashes = {r["sample_id"]: r["cache_sha256"] for r in audited["rows"]}
    verified = set()
    def loader(record):
        # Verify full cache/native image hashes once per process; LRU reloads only
        # need identity checks because this run never writes evidence files.
        if record["id"] not in verified:
            shared.verify_source(record)
            value = gate.evidence(root, cfg, record, signature)
            meta = shared.load_json(root/"evidence"/f"{record['id']}.json")
            if meta["sha256"] != hashes[record["id"]]:
                raise ValueError("Evidence changed after GT audit")
            verified.add(record["id"])
            return value
        return shared.load_torch(root/"evidence"/f"{record['id']}.pt")
    device = torch.device(args.device)
    # Cache size is an execution-only override. Scientific batches/config stay sealed.
    gpu_gb = cfg["gpu_cache_gb"] if args.gpu_cache_gb is None else args.gpu_cache_gb
    store = EvidenceStore(loader, device, cfg["cpu_cache_gb"], gpu_gb)
    for index, job in enumerate(cfg["jobs"]):
        if index % cfg["num_shards"] != args.shard_id:
            continue
        print(f"Start {job['id']} device={device}; cls={ARMS[job['arm']]}; same Base/split/init/budget", flush=True)
        directory, agent, trained = train_job(root, cfg, manifest, signature, split, store, job, device)
        store.clear_gpu()
        evaluate_job(directory, agent, trained, root, cfg, manifest, signature, split, store, job)
        del agent
        store.clear_gpu()


def collect(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, audited = checked_split(root, signature)
    ids = set(split["calibration_ids"]) | set(split["target_ids"])
    categories = defaultdict(list)
    for record in manifest:
        if record["id"] in ids:
            if not shared.cache_ready(root, record, signature):
                raise ValueError("Evidence missing/changed during collection")
            categories[(record["dataset"], record["category"])].append(record)
    metrics, rows, histories, summaries, checkpoints = [], [], [], [], []
    for job in cfg["jobs"]:
        directory = root/"jobs"/job["id"]
        trained = shared.load_json(directory/"training.json")
        if (trained["signature"] != signature or trained["split_sha256"] != shared.digest(split)
                or trained["job"] != job or trained["epochs"] != cfg["training"]["epoch"]):
            raise ValueError("Training signature mismatch")
        if shared.sha256_file(directory/"final.pth") != trained["final_sha256"]:
            raise ValueError("Final checkpoint changed")
        eval_signature = shared.digest(dict(training=trained["final_sha256"], split=shared.digest(split), signature=signature))
        if shared.load_json(directory/"evaluated.json")["signature"] != eval_signature:
            raise ValueError("Evaluation incomplete or stale")
        histories.extend(trained["history"])
        checkpoints.append({**trained, "job": job["id"], "arm": job["arm"], "seed": job["seed"]})
        job_rows = []
        for key, records in sorted(categories.items()):
            result = gate.read_category(gate.category_path(directory, key), eval_signature, split["audit_sha256"], [r["id"] for r in records])
            if result is None or len(result["rows"]) != len(records)*cfg["mara_config"]["max_steps"]:
                raise ValueError("Refusing partial category collection")
            for row in result["metrics"]:
                metrics.append(dict(row, job=job["id"], seed=job["seed"],
                    role="source_calibration" if key[0] == cfg["source_dataset"] else "target"))
            job_rows.extend(dict(row, job=job["id"], seed=job["seed"]) for row in result["rows"])
        rows.extend(job_rows)
        for dataset in cfg["datasets"]:
            for label, name in ((None, "all"), (0, "normal"), (1, "anomaly")):
                selected = [r for r in job_rows if r["dataset"] == dataset and (label is None or r["image_label"] == label)]
                count = len({r["sample_id"] for r in selected})
                if count:
                    summaries.append(dict(job=job["id"], arm=job["arm"], seed=job["seed"], dataset=dataset,
                        stratum=name, **summarize_rows(selected, count, cfg["quality_tolerance"])))
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["job"], row["dataset"], row["arm"], row["step"])].append(row)
    macro = []
    for (job, dataset, arm, step), values in sorted(groups.items()):
        macro.append(dict(job=job, dataset=dataset, arm=arm, step=step, seed=values[0]["seed"], role=values[0]["role"],
            categories=len(values), images=sum(r["images"] for r in values),
            **{k: shared.mean([r.get(k) for r in values]) for k in METRICS},
            **{"valid_categories_"+k: sum(r.get(k) is not None for r in values) for k in METRICS}))
    reference = {(r["seed"], r["dataset"], r["category"], r["step"]): r for r in metrics if r["arm"] == "R0"}
    comparisons = []
    for row in metrics:
        if row["arm"] in ("R0", "Base"):
            continue
        ref = reference[(row["seed"], row["dataset"], row["category"], row["step"])]
        comparisons.append(dict(job=row["job"], seed=row["seed"], dataset=row["dataset"], category=row["category"],
            arm=row["arm"], step=row["step"], **{k+"_minus_R0": row[k]-ref[k] if row[k] is not None and ref[k] is not None else None for k in METRICS}))
    out = root/"results"
    for name, values in (("metrics", metrics), ("macro_metrics", macro), ("comparisons", comparisons),
                         ("step_trace", rows), ("gate_summary", summaries), ("training", histories)):
        shared.write_csv(out/f"{name}.csv", values)
    shared.atomic_json(out/"checkpoints.json", checkpoints)
    (out/"README.md").write_text(
        "# MARA experiment R\n\n"
        "R0/R1/R2: classification reward weights 0.5/0.1/0; localization/FP weights and all losses fixed.\n"
        "Fresh identical per-seed initialization; frozen Base; no VLM; margin=0, probability=0.5.\n"
        "Fixed final epoch, no target-based checkpoint/threshold selection.\n"
        "Source calibration is MARA-only holdout; the frozen Base may already have seen those source images.\n"
        "Target subset defaults to 8 images/category: mechanism diagnosis, not a final benchmark.\n"
        "Compare PRO/P-AUROC/FP with Base and same-seed retrained R0, not the historical checkpoint.\n"
        "Raw rewards use DIFFERENT quality coefficients across arms; do not directly rank their magnitudes.\n"
        "step_trace includes unweighted delta_q_* and a common reward_reference_R0 for post-hoc comparison.\n"
        "I_AUROC follows the existing map-max image score; q_cls instead uses the global-logit head.\n"
        "One seed is a pilot; repeat predeclared complete arms with >=3 seeds before a robust claim.\n"
        "Report all groups; no automatic winner selection. Checkpoints/caches are intentionally excluded.\n",
        encoding="utf-8")
    shared.atomic_json(root/"complete.json", dict(protocol=PROTOCOL, signature=signature,
        split_sha256=shared.digest(split), jobs=len(cfg["jobs"]), final_epoch=cfg["training"]["epoch"],
        metrics_sha256=shared.sha256_file(out/"metrics.csv"), source_train=len(split["train_ids"]),
        source_calibration=len(split["calibration_ids"]), target_images=len(split["target_ids"])))
    target = root/"reward_report.zip"
    temporary = target.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.iterdir()):
                if path.is_file():
                    archive.write(path, "results/"+path.name)
            for name in ("config.json", "manifest.json", "audit.json", "split.json", "complete.json"):
                archive.write(root/name, name)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Complete: {target} ({target.stat().st_size/2**20:.2f}MiB); upload only this report", flush=True)


def parser():
    p = shared.parser(stage_choices=("prepare", "export", "audit", "train", "collect"))
    p.description = __doc__
    p.set_defaults(gradient_images=0)
    p.add_argument("--source_work_dir", default="")
    p.add_argument("--source_dataset", default="visa")
    p.add_argument("--source_samples_per_category", type=int, default=0)
    p.add_argument("--source_split_seed", type=int, default=20261018)
    p.add_argument("--calibration_fraction", type=float, default=.2)
    p.add_argument("--train_seeds", default="0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--export_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--decode_workers", type=int, default=2)
    p.add_argument("--metric_workers", type=int, default=2)
    p.add_argument("--cpu_cache_gb", type=float, default=4.)
    p.add_argument("--gpu_cache_gb", type=float, default=None)
    return p


def main():
    args = parser().parse_args()
    if args.stage == "prepare" and args.gpu_cache_gb is None:
        args.gpu_cache_gb = 12.
    positive = (args.epochs, args.train_batch_size, args.export_batch_size, args.eval_batch_size,
                args.num_shards, args.cpu_threads, args.image_size, args.decode_workers, args.metric_workers)
    if min(positive) < 1 or min(args.shard_id, args.source_samples_per_category, args.samples_per_category,
                               args.warmup_epochs, args.source_split_seed) < 0:
        raise ValueError("Invalid R budget/split arguments")
    if not math.isfinite(args.cpu_cache_gb) or args.cpu_cache_gb < 0 or (
        args.gpu_cache_gb is not None and (not math.isfinite(args.gpu_cache_gb) or args.gpu_cache_gb < 0)):
        raise ValueError("Cache budgets must be finite and nonnegative")
    if args.pro_num_th < 2 or not 0 < args.pro_max_fpr <= 1:
        raise ValueError("Invalid PRO parameters")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage != "prepare" and args.shard_id >= load_work(args.work_dir)[1]["num_shards"]:
        raise ValueError("Shard outside prepared layout")
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
