#!/usr/bin/env python3
"""PPO-LocSafe: source-only action diagnosis O and paired P0/P1/P2/P3."""
from collections import defaultdict
from dataclasses import asdict, replace
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
import mara_ppo_experiment as legacy
from mara_reward_experiment import checked_split, resume_payload, save_resume
from tools.mara_agent import MARAConfig
from tools.mara_diagnostics import checkpoint_config, map_metrics, stable_id
from tools.mara_gate_sweep import METRICS
from tools.mara_reward import split_source, state_hash
from tools.mara_ppo import INPUTS, PPOConfig, environment_config, optimizers, agent_from_payload
from tools.mara_locsafe import PROTOCOL, ARMS, LocSafeConfig, LocSafeAgent, nested_split, search_candidates
from tools.mara_ppo_monitor import (EpochTelemetry, diagnostic_context, monitor_epoch, refresh_job,
    refresh_overview, monitoring_selection, panels, pyplot, atomic_figure, atomic_html, page, visual_files)


def load_work(path, verify_checkpoint=True):
    return shared.load_work(path, verify_checkpoint=verify_checkpoint, protocol=PROTOCOL)


def new_agent(cfg, seed, refiner, arm="P0", normal_fraction=None):
    settings = LocSafeConfig(**cfg["locsafe_config"])
    if normal_fraction is not None:
        settings = replace(settings, normal_fraction=normal_fraction)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        agent = LocSafeAgent(MARAConfig(**cfg["mara_config"]), PPOConfig(**cfg["ppo_config"]), settings, arm)
        agent.initialize_refiner(refiner)
    return agent


def prepare(args):
    payload = shared.load_torch(args.mara_ckpt)
    if payload.get("mara_algorithm"):
        raise ValueError("MARA_CKPT must be original complete trained legacy MARA, not PPO")
    if all(torch.count_nonzero(v) == 0 for k, v in payload["mara_agent"].items()
           if k.startswith(("delta_head.", "score_head."))):
        raise ValueError("Identity Refiner is not a trained template")
    seeds = [int(x) for x in args.train_seeds.replace(",", " ").split()]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Need unique nonnegative train seeds")
    if args.source_dataset not in args.datasets.split() or args.steps < 2 or args.gradient_images:
        raise ValueError("Need source included, H>=2, no legacy gradient probe")
    local = dict(mara_config=asdict(environment_config(checkpoint_config(payload), args.steps)),
        ppo_config=asdict(PPOConfig(trajectories=args.trajectories, update_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size, actor_lr=args.actor_lr, critic_lr=args.critic_lr)),
        locsafe_config=asdict(LocSafeConfig(hard_fraction=args.hard_fraction, rank_margin=args.rank_margin,
            rank_weight=args.rank_weight, rank_points=args.rank_points, risk_budget=args.risk_budget,
            dual_initial=args.dual_initial, dual_lr=args.dual_lr)))
    source, reference = None, None
    reference_path = args.reference_ppo_ckpt
    if args.source_work_dir:
        old_root = Path(args.source_work_dir).resolve()
        old, manifest = shared.load_json(old_root/"config.json"), shared.load_json(old_root/"manifest.json")
        if old_root == Path(args.work_dir).resolve() or old.get("protocol") not in (
            "mara_early_stop_ab_v1", "mara_early_stop_ab_v2", "mara_gate_sweep_v1", "mara_reward_r_v1", "mara_ppo_v1", PROTOCOL):
            raise ValueError("Invalid cache source protocol/directory")
        if old["checkpoint_sha256"] != shared.sha256_file(args.mara_ckpt) or old["manifest_sha256"] != shared.digest(manifest):
            raise ValueError("Cache source checkpoint/manifest mismatch")
        source = dict(path=str(old_root), signature=shared.digest(old), config_sha256=shared.sha256_file(old_root/"config.json"),
                      manifest_sha256=old["manifest_sha256"])
        if not reference_path and old["protocol"] == "mara_ppo_v1" and (old_root/"jobs/V_seed0/final.pth").is_file():
            reference_path = str(old_root/"jobs/V_seed0/final.pth")
    if reference_path:
        path = Path(reference_path).resolve()
        prior = shared.load_torch(path)
        if prior.get("mara_algorithm") != "mara_ppo_v1":
            raise ValueError("Reference must be the prior PPO V/MC checkpoint")
        prior_agent = agent_from_payload(prior)
        if state_hash(prior_agent.transition.state_dict()) != state_hash(payload["mara_agent"]):
            raise ValueError("Reference PPO uses another Refiner")
        if asdict(prior_agent.cfg) != local["mara_config"]:
            raise ValueError("Reference PPO transition config/horizon differs")
        reference = dict(path=str(path), sha256=shared.sha256_file(path))
    initial = {str(s): state_hash(new_agent(local, s, payload["mara_agent"]).state_dict()) for s in seeds}
    shared.prepare(args, protocol=PROTOCOL, arms={a: {} for a in ARMS}, extra_config=dict(
        **local, source_cache=source, reference_ppo=reference,
        dataset_sample_limits={args.source_dataset: args.source_samples_per_category},
        source_dataset=args.source_dataset, source_split_seed=args.source_split_seed,
        calibration_fraction=args.calibration_fraction, development_fraction=args.development_fraction,
        inner_split_seed=args.inner_split_seed, train_seeds=seeds,
        jobs=[dict(id=f"{a}_seed{s}", arm=a, seed=s) for s in seeds for a in ARMS],
        initial_state_hashes=initial, refiner_state_sha256=state_hash(payload["mara_agent"]),
        epochs=args.epochs, train_batch_size=args.train_batch_size, export_batch_size=args.export_batch_size,
        eval_batch_size=args.eval_batch_size, decode_workers=args.decode_workers, metric_workers=args.metric_workers,
        cpu_cache_gb=args.cpu_cache_gb, gpu_cache_gb=args.gpu_cache_gb, value_probe_trajectories=args.value_probe_trajectories,
        category_shards=False, empty_gt_policy="exclude", precision="fp32_no_autocast_tf32_off",
        oracle_per_category=args.oracle_per_category, beam_per_category=args.beam_per_category,
        beam_width=args.beam_width, oracle_batch_size=args.oracle_batch_size,
        validation_every=args.validation_every, preview_samples=args.preview_samples,
        selection_bg_tolerance=args.selection_bg_tolerance, selection_auc_tolerance=args.selection_auc_tolerance,
        transition="frozen_trained_refiner_no_gain_veto_v1",
        selection_note="Source DEV only: max PRO among registered epochs passing Base FP/AUC guardrails; no eligible epoch => FINAL diagnostic only, NOT success",
        gradient_note="Only Actor/Critic train. No GT in actions/state. No VLM/LoRA/Refiner updates.",
        sampling_note="Nested source fit/dev; historical source holdout excluded from fit/dev; targets only after ALL jobs train. Source system-level independence NOT guaranteed.",
    ))
    print("LocSafe: O first; P0 original, P1 cls=0, P2 +region/hard-BG rank, P3 +terminal normal-risk constraint", flush=True)


def export(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    gate.import_caches(root, cfg, manifest, signature, args.shard_id)
    shared.export(args, protocol=PROTOCOL)


def audit(args):
    gate.audit(args, protocol=PROTOCOL)
    root, cfg, manifest, signature = load_work(args.work_dir)
    audited = shared.load_json(root/"audit.json")
    historical = split_source(manifest, audited, cfg["source_dataset"], cfg["source_split_seed"], cfg["calibration_fraction"])
    split = nested_split(manifest, audited, historical, cfg["inner_split_seed"], cfg["development_fraction"])
    if cfg["source_cache"]:
        prior_manifest = shared.load_json(Path(cfg["source_cache"]["path"])/"manifest.json")
        seen = {r["id"] for r in prior_manifest if r["dataset"] != cfg["source_dataset"]}
        split["historical_target_ids"] = sorted(set(split["target_ids"]) & seen)
        split["target_history_tracking"] = "IDs present in cache-source manifest; not a claim of never-seen categories"
    else:
        split["historical_target_ids"] = []
        split["target_history_tracking"] = "unavailable; target independence cannot be certified"
    split.update(signature=signature, audit_sha256=shared.digest(audited))
    path = root/"split.json"
    if path.exists() and shared.load_json(path) != split:
        raise ValueError("Nested split changed; use a new WORK_DIR")
    shared.atomic_json(path, split)
    print(f"LocSafe split: fit={len(split['train_ids'])} DEV={len(split['calibration_ids'])} historical_source={len(split['historical_calibration_ids'])} targets={len(split['target_ids'])}", flush=True)


def oracle_path(root, category):
    return root/"oracle"/"categories"/(shared.digest(category)[:24]+".json")


def oracle_preview(path, batch, maps, record):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2+len(maps), figsize=(3*(2+len(maps)), 3.4))
    axes[0].imshow(batch["mask"][0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("GT: diagnosis only")
    axes[1].imshow(batch["base_prob"][0, 1].cpu(), cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Base")
    for ax, (name, probability) in zip(axes[2:], maps.items()):
        ax.imshow(probability, cmap="magma", vmin=0, vmax=1)
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(record["category"]+" | "+record["id"]+" | fixed [0,1]; surrogate oracle, NOT PRO upper bound")
    fig.tight_layout()
    atomic_figure(fig, path)


@torch.no_grad()
def oracle(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, audited = checked_split(root, signature)
    selected, _ = monitoring_selection(manifest, split, audited, cfg["oracle_per_category"], 0)
    beam_records, _ = monitoring_selection(manifest, split, audited, cfg["beam_per_category"], 0)
    beam_ids = {r["id"] for r in beam_records}
    groups = defaultdict(list)
    for r in selected:
        groups[r["category"]].append(r)
    device = torch.device(args.device)
    store = legacy.evidence_store(root, cfg, signature, audited, device, args.gpu_cache_gb)
    original = shared.load_torch(cfg["mara_ckpt"])
    agent = new_agent(cfg, cfg["train_seeds"][0], original["mara_agent"], "P2").to(device).eval()
    reference = None
    if cfg["reference_ppo"]:
        ref = cfg["reference_ppo"]
        if shared.sha256_file(ref["path"]) != ref["sha256"]:
            raise ValueError("Reference PPO changed")
        reference = agent_from_payload(shared.load_torch(ref["path"])).to(device).eval()
    for index, (category, records) in enumerate(sorted(groups.items())):
        if index % cfg["num_shards"] != args.shard_id:
            continue
        path = oracle_path(root, category)
        identity = dict(signature=signature, split=shared.digest(split), ids=[r["id"] for r in records])
        if path.exists():
            if shared.load_json(path)["identity"] != identity:
                raise ValueError("Stale oracle category")
            print(f"Oracle reuse {category}", flush=True)
            continue
        traces, arrays, masks, labels, bases, beam_arrays = [], defaultdict(list), [], [], [], defaultdict(list)
        beam_masks, beam_labels, beam_bases, summaries = [], [], [], []
        for record in records:
            batch = store.batch([record])
            inputs = {k: batch[k] for k in INPUTS if k in batch}
            rows, prefixes, best, best_map = search_candidates(agent, inputs, batch["mask"], batch["labels"],
                [[]], 1, cfg["beam_width"], cfg["oracle_batch_size"])
            traces.extend(dict(r, sample_id=record["id"], category=category, image_label=int(batch["labels"][0])) for r in rows)
            arrays["O1_loc"].append(best_map)
            maps = {"O1_loc": best_map}
            if reference:
                ref_map = reference.trajectory(inputs, sample=False)["final_prob"][0, 1].cpu().numpy()
                arrays["prior_PPO"].append(ref_map)
                maps["prior_PPO"] = ref_map
            summary = dict(sample_id=record["id"], category=category, image_label=int(batch["labels"][0]),
                           best_one_step_gain=best["proxy_gain"], best_one_step_path=best["path"],
                           candidate_count=len(rows), changing_candidates=sum(r["pixel_change"] > 1e-7 for r in rows))
            if record["id"] in beam_ids:
                beam_arrays["O1_loc"].append(best_map)
                beam_bases.append(batch["base_prob"][0, 1].cpu().numpy())
                beam_masks.append(batch["mask"][0, 0].cpu().numpy().astype(bool))
                beam_labels.append(int(batch["labels"][0]))
                first_by_action = {tuple(r["path"][0]): r["proxy_gain"] for r in rows}
                for depth in range(2, agent.cfg.max_steps+1):
                    rows, prefixes, best, best_map = search_candidates(agent, inputs, batch["mask"], batch["labels"],
                        prefixes, depth, cfg["beam_width"], cfg["oracle_batch_size"])
                    traces.extend(dict(r, sample_id=record["id"], category=category, image_label=int(batch["labels"][0])) for r in rows)
                    beam_arrays[f"O{depth}_loc"].append(best_map)
                    maps[f"O{depth}_loc"] = best_map
                summary.update(best_beam_gain=best["proxy_gain"], best_beam_path=best["path"],
                    negative_first_positive_final=first_by_action[tuple(best["path"][0])] < 0 < best["proxy_gain"])
                oracle_preview(root/"oracle"/"previews"/(record["id"]+".png"), batch, maps, record)
            summaries.append(summary)
            masks.append(batch["mask"][0, 0].cpu().numpy().astype(bool))
            labels.append(int(batch["labels"][0]))
            bases.append(batch["base_prob"][0, 1].cpu().numpy())
            print(f"O {category} {len(summaries)}/{len(records)} best_1={summary['best_one_step_gain']:+.5f}", flush=True)
        metrics = []
        for population, source_masks, source_labels, source_bases, source_arrays in (
            ("one_step_panel", masks, labels, bases, arrays), ("beam_panel", beam_masks, beam_labels, beam_bases, beam_arrays)):
            if not source_masks:
                continue
            for arm, probabilities in {"Base": source_bases, **source_arrays}.items():
                metrics.append(dict(category=category, population=population, arm=arm, images=len(source_labels),
                    **map_metrics(np.stack(source_masks), np.stack(probabilities), np.array(source_labels), cfg["pro_num_th"], cfg["pro_max_fpr"])))
        shared.atomic_json(path, dict(identity=identity, metrics=metrics, rows=traces, summaries=summaries,
            note="Source DEV GT-assisted finite surrogate search, not deployment/PRO upper bound. Prior PPO may have trained on these IDs."))
        print(f"Oracle committed {category}", flush=True)
    shared.atomic_json(root/"oracle"/f"shard_{args.shard_id}.json", dict(signature=signature, split_sha256=shared.digest(split)))


def oracle_report(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, audited = checked_split(root, signature)
    records, _ = monitoring_selection(manifest, split, audited, cfg["oracle_per_category"], 0)
    metrics, traces, summaries = [], [], []
    for shard in range(cfg["num_shards"]):
        if shared.load_json(root/"oracle"/f"shard_{shard}.json") != dict(signature=signature, split_sha256=shared.digest(split)):
            raise ValueError("Oracle worker incomplete")
    for category in sorted({r["category"] for r in records}):
        result = shared.load_json(oracle_path(root, category))
        if result["identity"]["signature"] != signature or result["identity"]["split"] != shared.digest(split):
            raise ValueError("Stale oracle report")
        metrics.extend(result["metrics"])
        traces.extend(result["rows"])
        summaries.extend(result["summaries"])
    groups = defaultdict(list)
    for r in metrics:
        groups[(r["population"], r["arm"])].append(r)
    macro = [dict(population=p, arm=a, categories=len(rs), images=sum(r["images"] for r in rs),
                  **{m: shared.mean([r.get(m) for r in rs]) for m in METRICS}) for (p, a), rs in sorted(groups.items())]
    for name, rows in (("metrics", metrics), ("macro_metrics", macro), ("candidate_trace", traces), ("sample_summary", summaries)):
        shared.write_csv(root/"oracle"/(name+".csv"), rows)
    positives = sum(r["best_one_step_gain"] > 1e-6 for r in summaries if r["image_label"])
    atomic_html(root/"oracle"/"index.html", page("O: source-only candidate diagnosis",
        "<p>Finite GT-assisted surrogate search; NOT deployment and NOT a global PRO upper bound. "
        "A poor beam result does not prove no useful trajectory exists. Prior PPO comparison may be in-sample.</p>"
        +"<pre>"+__import__("html").escape(str(macro))+"</pre>"
        +"".join(f'<img loading="lazy" src="previews/{p.name}">' for p in sorted((root/"oracle"/"previews").glob("*.png")))))
    shared.atomic_json(root/"oracle"/"complete.json", dict(signature=signature, split_sha256=shared.digest(split),
        anomalous_positive_one_step_images=positives, images=len(summaries),
        recommendation="Inspect PRO, FP and maps. No automatic go/no-go based only on this surrogate/small panel."))
    package(root, cfg, "oracle_report.zip", final=False)
    print(f"O complete: {root/'oracle_report.zip'}; training will not be silently disabled by surrogate results", flush=True)


def select_epoch(directory, cfg):
    candidates = []
    for path in sorted((directory/"monitor/validation").glob("epoch_*.json")):
        value = shared.load_json(path)
        if value["epoch"] == 0:
            continue
        m = value["macro"]
        eligible = (m["PRO"] is not None and m["PRO"] > m["base_PRO"]
            and m["background_FPR_at_05"] <= m["base_background_FPR_at_05"]+cfg["selection_bg_tolerance"]
            and m["P_AUROC"] >= m["base_P_AUROC"]-cfg["selection_auc_tolerance"])
        candidates.append(dict(epoch=value["epoch"], eligible=eligible, **m))
    eligible = [r for r in candidates if r["eligible"]]
    selected = min(eligible, key=lambda r: (-r["PRO"], r["epoch"]))["epoch"] if eligible else cfg["epochs"]
    return dict(epoch=selected, status="source_guardrails_passed" if eligible else "FAILED_guardrails_FINAL_diagnostic_only",
                candidates=candidates, note="No target metrics used. No eligible checkpoint does NOT silently turn into Base success.")


def train_job(root, cfg, manifest, signature, split, audited, store, job, device):
    directory = root/"jobs"/job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(signature=signature, job=job, split_sha256=shared.digest(split),
                    initial_state_sha256=cfg["initial_state_hashes"][str(job["seed"])])
    fit_ids = set(split["train_ids"])
    records = sorted((r for r in manifest if r["id"] in fit_ids), key=lambda r: r["id"])
    labels = {r["sample_id"]: r["image_label"] for r in audited["rows"] if r["included"]}
    rho = sum(labels[r["id"]] == 0 for r in records)/len(records)
    original = shared.load_torch(cfg["mara_ckpt"])
    agent = new_agent(cfg, job["seed"], original["mara_agent"], job["arm"], rho)
    if state_hash(agent.state_dict()) != identity["initial_state_sha256"]:
        raise ValueError("Matched initial parameters changed")
    agent.to(device)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(job["seed"])
        torch.cuda.reset_peak_memory_stats(device)
    pair = legacy.OptimizerPair(*optimizers(agent))
    scheduler = torch.optim.lr_scheduler.LambdaLR(pair.actor, lambda _: 1.)
    history, first = [], 0
    saved = resume_payload(directory, identity)
    if saved:
        if len(saved["history"]) != saved["epoch"]+1 or not 0 <= saved["epoch"] < cfg["epochs"]:
            raise ValueError("Bad resume history/budget")
        agent.load_state_dict(saved["mara_agent"], strict=True)
        pair.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, first = saved["history"], saved["epoch"]+1
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
        print(f"Resume {job['id']} at epoch {first+1}", flush=True)
        shared.write_csv(directory/"training.csv", history)
    start = time.monotonic()
    settings = dict(every=cfg["validation_every"], per_category=0, previews=cfg["preview_samples"])
    if first == 0:
        monitor_epoch(directory, agent, cfg, manifest, split, audited, store, job, 0, **settings)
    # If interrupted after committing weights but before validation, backfill the
    # committed scheduled epoch before the next update / source-only selection.
    if first and (first == 1 or first % cfg["validation_every"] == 0 or first == cfg["epochs"]):
        monitor_epoch(directory, agent, cfg, manifest, split, audited, store, job, first, **settings)
    for epoch in range(first, cfg["epochs"]):
        if history and job["arm"] == "P3":
            agent.dual.fill_(history[-1]["dual_next"])
        agent.reset_epoch_logs()
        telemetry = EpochTelemetry(directory, job, epoch+1)
        try:
            meters = legacy.train_epoch(agent, store, records, cfg, job, epoch, pair, scheduler, telemetry)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("OOM: do not silently change batches. Rerun with EXEC_GPU_CACHE_GB=0 to resume committed epoch.") from None
        finally:
            telemetry.close()
        meters.update(agent.finish_epoch())
        if not all(math.isfinite(v) for v in meters.values()):
            raise ValueError("Nonfinite LocSafe training metrics")
        if state_hash(agent.transition.state_dict()) != cfg["refiner_state_sha256"]:
            raise ValueError("Frozen Refiner changed")
        history.append(dict(job=job["id"], arm=job["arm"], seed=job["seed"], epoch=epoch+1, **meters))
        save_resume(directory, identity, epoch, agent, pair, scheduler, history, device)
        telemetry.commit()
        shared.write_csv(directory/"training.csv", history)
        if epoch == 0 or (epoch+1) % cfg["validation_every"] == 0 or epoch+1 == cfg["epochs"]:
            monitor_epoch(directory, agent, cfg, manifest, split, audited, store, job, epoch+1, **settings)
        with diagnostic_context(agent):
            refresh_job(directory, history)
        print(f"Committed {job['id']} {epoch+1}/{cfg['epochs']} risk={meters['normal_risk_mean']:.6f} dual={meters['dual_used']:.4f}", flush=True)
    final, selected, metadata = directory/"final.pth", directory/"selected.pth", directory/"training.json"
    if not metadata.exists():
        payload = {k: v for k, v in original.items() if k not in ("mara_agent", "optimizer", "scheduler")}
        payload.update(mara_algorithm=PROTOCOL, mara_config=asdict(agent.cfg), ppo_config=asdict(agent.ppo),
                       locsafe_config=asdict(agent.settings), locsafe_arm=job["arm"], locsafe_experiment=identity)
        final_payload = dict(payload, mara_ppo_agent={k: v.detach().cpu() for k, v in agent.state_dict().items()}, epoch=cfg["epochs"]-1)
        shared.atomic_torch(final, final_payload)
        selection = select_epoch(directory, cfg)
        state = shared.load_torch(directory/"resume"/f"epoch_{selection['epoch']-1:03d}.pth")["mara_agent"]
        chosen = dict(payload, mara_ppo_agent=state, epoch=selection["epoch"]-1, source_selection=selection)
        shared.atomic_torch(selected, chosen)
        shared.atomic_json(directory/"selection.json", selection)
        shared.atomic_json(metadata, dict(**identity, final_sha256=shared.sha256_file(final),
            selected_sha256=shared.sha256_file(selected), selected_epoch=selection["epoch"], selection=selection,
            final_state_sha256=state_hash(agent.state_dict()), refiner_state_sha256=state_hash(agent.transition.state_dict()),
            epochs=cfg["epochs"], history=history, seconds_this_invocation=time.monotonic()-start,
            peak_allocated_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.,
            gpu_cache_gb=store.gpu_bytes/2**30, cpu_cache_gb=store.cpu_bytes/2**30))
    trained = shared.load_json(metadata)
    if any(trained.get(k) != v for k, v in identity.items()) or trained["final_sha256"] != shared.sha256_file(final) or trained["selected_sha256"] != shared.sha256_file(selected):
        raise ValueError("Checkpoint/identity mismatch")
    return directory, trained


def train(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, audited = checked_split(root, signature)
    if shared.load_json(root/"oracle/complete.json")["signature"] != signature:
        raise ValueError("O must complete first")
    device = torch.device(args.device)
    store = legacy.evidence_store(root, cfg, signature, audited, device, args.gpu_cache_gb)
    for index, job in enumerate(cfg["jobs"]):
        if index % cfg["num_shards"] == args.shard_id:
            train_job(root, cfg, manifest, signature, split, audited, store, job, device)
            store.clear_gpu()


def evaluation_split(split):
    return dict(split, target_ids=sorted(split["target_ids"]+split["historical_calibration_ids"]))


def populations(split, dataset):
    if dataset == split["source"]:
        return dict(source_dev=set(split["calibration_ids"]), source_historical=set(split["historical_calibration_ids"]))
    all_targets, historical = set(split["target_ids"]), set(split["historical_target_ids"])
    return dict(target_full=all_targets, target_historical=historical, target_additional=all_targets-historical)


def finish_populations(split, directory, cfg, signature, audit_hash, key, records, traces, maps, masks, labels, base):
    # Reuse the SAME GPU outputs; compute source DEV/historical and target
    # full/historical/additional separately, rather than pooling distinct roles.
    for role, selected in populations(split, key[0]).items():
        positions = [i for i, r in enumerate(records) if r["id"] in selected]
        if positions:
            subset = [records[i] for i in positions]
            keep = {r["id"] for r in subset}
            gate.finish_category(directory/"populations"/role, cfg, signature, audit_hash, key, subset,
                [r for r in traces if r["sample_id"] in keep], {a: p[positions] for a, p in maps.items()},
                masks[positions], labels[positions], base[positions])
    # Parent completion is committed LAST: a partial population computation must
    # not cause the shared evaluator to skip unfinished work on restart.
    path = gate.category_path(directory, key)
    shared.atomic_json(path, dict(signature=signature, audit_hash=audit_hash, sample_ids=[r["id"] for r in records],
        dataset=key[0], category=key[1], rows=traces, metrics=[]))
    shared.atomic_json(path.with_suffix(".sha256.json"), dict(sha256=shared.sha256_file(path)))


def evaluate(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, audited = checked_split(root, signature)
    # ALL source selection completes before ANY target evaluation.
    for job in cfg["jobs"]:
        meta = shared.load_json(root/"jobs"/job["id"]/"training.json")
        if meta["signature"] != signature or meta["epochs"] != cfg["epochs"]:
            raise ValueError("All training jobs must complete before targets")
    store = legacy.evidence_store(root, cfg, signature, audited, args.device, args.gpu_cache_gb)
    for index, job in enumerate(cfg["jobs"]):
        if index % cfg["num_shards"] != args.shard_id:
            continue
        directory = root/"jobs"/job["id"]
        trained = shared.load_json(directory/"training.json")
        if shared.sha256_file(directory/"selected.pth") != trained["selected_sha256"]:
            raise ValueError("Selected checkpoint changed")
        agent = agent_from_payload(shared.load_torch(directory/"selected.pth")).to(args.device).eval()
        callback = lambda *values: finish_populations(split, *values)
        legacy.evaluate_job(directory, agent, trained, root, cfg, manifest, signature, evaluation_split(split), store, job, callback)
        del agent
        store.clear_gpu()


def package(root, cfg, name, final=True):
    target = root/name
    temporary = target.with_suffix(".zip.tmp")
    paths = set()
    for folder in ((root/"results", root/"oracle") if final else (root/"oracle",)):
        paths.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".csv", ".json", ".html", ".png", ".md")
                     and "categories" not in p.relative_to(folder).parts)
    if final:
        paths.update(visual_files(root, cfg))
        for job in cfg["jobs"]:
            paths.update(p for p in (root/"jobs"/job["id"]).glob("*.json") if p.name in ("training.json", "selection.json", "value_probe.json"))
    names = ("config.json", "manifest.json", "audit.json", "split.json")+(('complete.json',) if final else ())
    paths.update(root/n for n in names if (root/n).exists())
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(paths):
                archive.write(p, p.relative_to(root).as_posix())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def calibration_summary(probes):
    grouped = defaultdict(list)
    for r in probes:
        grouped[(r["job"], r["image_label"])].append(r)
    rows = []
    for (job, label), rs in sorted(grouped.items()):
        actual = np.array([r["mc_return_mean"] for r in rs])
        predicted = np.array([r["v_prediction"] for r in rs])
        error = predicted-actual
        variance = float(((actual-actual.mean())**2).sum())
        rows.append(dict(job=job, image_label=label, states=len(rs), MAE=float(np.abs(error).mean()),
            RMSE=float(np.sqrt((error**2).mean())), R2=1-float((error**2).sum())/variance if variance > 1e-12 else None,
            max_absolute_error=float(np.abs(error).max()),
            optimistic_negative_states=sum(r["v_prediction"] > 0 and r["mc_return_mean"] < -2*r["mc_standard_error"] for r in rs),
            note="2*MC_SE used only to flag clear negative sampled means, not a formal multiple-testing confidence statement"))
    return rows


def final_dashboard(root, cfg, probes, checkpoints):
    import html
    directory = root/"visual_report"
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
    values = [r[k] for r in probes for k in ("v_prediction", "mc_return_mean")]
    lo, hi = (min(values), max(values)) if values else (-1., 1.)
    for ax, arm in zip(axes.flat, ARMS):
        for label, marker in ((0, "x"), (1, "o")):
            rs = [r for r in probes if r["arm"] == arm and r["image_label"] == label]
            if rs:
                ax.errorbar([r["v_prediction"] for r in rs], [r["mc_return_mean"] for r in rs],
                    yerr=[r["mc_standard_error"] for r in rs], fmt=marker, markersize=4,
                    alpha=.7, label="normal" if label == 0 else "anomaly")
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=.8)
        ax.set(title=arm, xlabel="V prediction", ylabel="independent stochastic mean G", xlim=(lo-.02, hi+.02), ylim=(lo-.02, hi+.02))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Source DEV V calibration by label | bars: one MC standard error; rewards differ across arms")
    fig.tight_layout()
    atomic_figure(fig, directory/"value_calibration_by_label.png")
    rows = "".join(f'<tr><td>{html.escape(r["job"]["id"])}</td><td>{r["selected_epoch"]}</td>'
        f'<td>{html.escape(r["selection"]["status"])}</td><td><a href="../jobs/{r["job"]["id"]}/monitor/index.html">curves/maps</a></td></tr>' for r in checkpoints)
    figures = "".join(f'<h2>{html.escape(p.stem)}</h2><img src="{p.name}">' for p in sorted(directory.glob("*.png")))
    atomic_html(directory/"index.html", page("PPO-LocSafe: O + P0/P1/P2/P3",
        '<p><a href="../oracle/index.html">O candidate diagnosis</a> | '
        '<a href="../results/macro_metrics.csv">population-separated metrics</a> | '
        '<a href="../results/value_calibration_summary.csv">V errors by label</a></p>'
        '<p>FAILED guardrails means diagnostic only, NOT successful deployment. '
        'Source DEV selects weights; historical source is separate. No target selection. '
        'Raw rewards differ across arms: compare PRO/FP and the fixed reference reward, not reward magnitudes.</p>'
        '<table><tr><th>Job</th><th>Selected epoch</th><th>Source guardrail status</th><th>Process</th></tr>'+rows+'</table>'+figures))


def collect(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, _ = checked_split(root, signature)
    esplit = evaluation_split(split)
    ids = set(esplit["calibration_ids"]+esplit["target_ids"])
    historical = set(split["historical_calibration_ids"])
    old_targets = set(split["historical_target_ids"])
    groups = defaultdict(list)
    for r in manifest:
        if r["id"] in ids:
            groups[(r["dataset"], r["category"])].append(r)
    metrics, traces, histories, probes, checkpoints = [], [], [], [], []
    for job in cfg["jobs"]:
        directory = root/"jobs"/job["id"]
        trained = shared.load_json(directory/"training.json")
        expected = shared.digest(dict(training=trained["selected_sha256"], split=shared.digest(esplit), signature=signature))
        if (trained["signature"] != signature or trained["split_sha256"] != shared.digest(split)
            or trained["job"] != job or trained["epochs"] != cfg["epochs"]
            or trained["refiner_state_sha256"] != cfg["refiner_state_sha256"]
            or shared.sha256_file(directory/"final.pth") != trained["final_sha256"]
            or shared.sha256_file(directory/"selected.pth") != trained["selected_sha256"]
            or shared.load_json(directory/"evaluated.json")["signature"] != expected):
            raise ValueError("Incomplete/corrupt job evaluation")
        histories.extend(trained["history"])
        checkpoints.append(trained)
        probe = shared.load_json(directory/"value_probe.json")
        if probe["signature"] != expected:
            raise ValueError("Stale independent value probe")
        probes.extend(probe["rows"])
        for key, records in sorted(groups.items()):
            value = gate.read_category(gate.category_path(directory, key), expected, split["audit_sha256"], [r["id"] for r in records])
            if value is None or len(value["rows"]) != len(records)*cfg["mara_config"]["max_steps"]:
                raise ValueError("Partial category")
            for role, selected_ids in populations(split, key[0]).items():
                subset = [r for r in records if r["id"] in selected_ids]
                if not subset:
                    continue
                population = gate.read_category(gate.category_path(directory/"populations"/role, key), expected,
                    split["audit_sha256"], [r["id"] for r in subset])
                if population is None:
                    raise ValueError("Partial population metrics")
                metrics.extend(dict(r, job=job["id"], seed=job["seed"], selected_epoch=trained["selected_epoch"],
                    role=role) for r in population["metrics"])
            for r in value["rows"]:
                role = ("source_historical" if r["sample_id"] in historical else "source_dev") if key[0] == cfg["source_dataset"] else (
                    "target_historical" if r["sample_id"] in old_targets else "target_additional_not_certified_unseen")
                traces.append(dict(r, job=job["id"], seed=job["seed"], role=role))
    grouped = defaultdict(list)
    for r in metrics:
        grouped[(r["job"], r["dataset"], r["role"], r["arm"], r["step"])].append(r)
    macro = [dict(job=j, seed=rs[0]["seed"], dataset=d, role=role, arm=a, step=t, categories=len(rs), images=sum(r["images"] for r in rs),
                  **{m: shared.mean([r.get(m) for r in rs]) for m in METRICS}) for (j, d, role, a, t), rs in sorted(grouped.items())]
    reference = {(r["seed"], r["dataset"], r["role"], r["category"], r["step"]): r for r in metrics if r["arm"] == "P0"}
    comparisons = []
    for r in metrics:
        if r["arm"] not in ("P1", "P2", "P3"):
            continue
        ref = reference[(r["seed"], r["dataset"], r["role"], r["category"], r["step"])]
        comparisons.append(dict(job=r["job"], arm=r["arm"], seed=r["seed"], dataset=r["dataset"], role=r["role"], category=r["category"], step=r["step"],
            **{m+"_minus_P0": r[m]-ref[m] if r[m] is not None and ref[m] is not None else None for m in METRICS}))
    risk_groups = defaultdict(list)
    for r in traces:
        if r["step"] == cfg["mara_config"]["max_steps"]-1:
            risk_groups[(r["job"], r["dataset"], r["role"], r["image_label"])].append(r)
    risks = []
    for (job, dataset, role, label), rows in sorted(risk_groups.items()):
        values = sorted(r["risk_increase"] for r in rows)
        n = max(1, math.ceil(len(values)*.05))
        risks.append(dict(job=job, dataset=dataset, role=role, image_label=label, images=len(rows),
            normal_risk_mean=shared.mean(values), normal_risk_max=max(values), normal_risk_p95=float(np.quantile(values, .95)),
            normal_risk_worst5_mean=shared.mean(values[-n:]),
            risk_violations=sum(v > cfg["locsafe_config"]["risk_budget"] for v in values)))
    for name, rows in (("metrics", metrics), ("macro_metrics", macro), ("comparisons", comparisons), ("step_trace", traces),
                       ("training", histories), ("value_probe", probes), ("risk_summary", risks),
                       ("value_calibration_summary", calibration_summary(probes))):
        shared.write_csv(root/"results"/(name+".csv"), rows)
    shared.atomic_json(root/"results/checkpoints.json", checkpoints)
    shared.atomic_json(root/"complete.json", dict(protocol=PROTOCOL, signature=signature, split_sha256=shared.digest(split),
        jobs=len(cfg["jobs"]), metrics_sha256=shared.sha256_file(root/"results/metrics.csv"),
        source_fit=len(split["train_ids"]), source_dev=len(split["calibration_ids"]),
        source_historical=len(historical), targets=len(split["target_ids"]),
        note="Source DEV/historical and target full/historical/additional reported separately. Targets never select weights. Three seeds/full sets needed for stable claims."))
    for job in cfg["jobs"]:
        refresh_job(root/"jobs"/job["id"])
    refresh_overview(root, cfg)
    final_dashboard(root, cfg, probes, checkpoints)
    target = package(root, cfg, "locsafe_report.zip")
    print(f"Complete: {target} ({target.stat().st_size/2**20:.2f} MiB). Upload ONLY this report.", flush=True)


def parser():
    p = legacy.parser()
    p.description = __doc__
    next(a for a in p._actions if a.dest == "stage").choices = ("prepare", "export", "audit", "oracle", "oracle_report", "train", "evaluate", "collect")
    p.set_defaults(samples_per_category=0)
    p.add_argument("--reference_ppo_ckpt", default="")
    p.add_argument("--development_fraction", type=float, default=.2)
    p.add_argument("--inner_split_seed", type=int, default=20261020)
    for name, default in (("oracle_per_category", 8), ("beam_per_category", 2), ("beam_width", 8),
                          ("oracle_batch_size", 16), ("rank_points", 128), ("validation_every", 5)):
        p.add_argument("--"+name, type=int, default=default)
    for name, default in (("hard_fraction", .05), ("rank_margin", .1), ("rank_weight", .5),
                          ("risk_budget", .001), ("dual_initial", 1.), ("dual_lr", 10.),
                          ("selection_bg_tolerance", .0002), ("selection_auc_tolerance", .002)):
        p.add_argument("--"+name, type=float, default=default)
    return p


def main():
    args = parser().parse_args()
    if args.stage == "prepare" and args.gpu_cache_gb is None:
        args.gpu_cache_gb = 12.
    for name in ("epochs", "train_batch_size", "trajectories", "ppo_epochs", "minibatch_size", "num_shards",
                 "export_batch_size", "eval_batch_size", "decode_workers", "metric_workers", "cpu_threads", "image_size",
                 "oracle_per_category", "beam_width", "oracle_batch_size", "rank_points", "validation_every"):
        if getattr(args, name) < 1:
            raise ValueError("Nonpositive budget: "+name)
    if not 0 <= args.beam_per_category <= args.oracle_per_category or args.value_probe_trajectories < 2:
        raise ValueError("Invalid beam/probe sample counts")
    if min(args.shard_id, args.samples_per_category, args.source_samples_per_category, args.preview_samples,
           args.source_split_seed, args.inner_split_seed) < 0 or not 0 < args.development_fraction < 1 or not 0 < args.calibration_fraction < 1:
        raise ValueError("Invalid split/sampling")
    if any(not math.isfinite(v) or v < 0 for v in (args.cpu_cache_gb, args.selection_bg_tolerance, args.selection_auc_tolerance)):
        raise ValueError("Invalid cache/selection tolerance")
    if args.gpu_cache_gb is not None and (not math.isfinite(args.gpu_cache_gb) or args.gpu_cache_gb < 0):
        raise ValueError("Invalid GPU cache")
    if args.pro_num_th < 2 or not 0 < args.pro_max_fpr <= 1:
        raise ValueError("Invalid PRO settings")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage != "prepare" and args.shard_id >= load_work(args.work_dir)[1]["num_shards"]:
        raise ValueError("Shard outside sealed layout")
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
