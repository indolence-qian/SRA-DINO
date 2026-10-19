#!/usr/bin/env python3
"""Paired PPO+V / PPO+MC pilot with a shared frozen, trained MARA Refiner."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
from mara_reward_experiment import checked_split, resume_payload, save_resume
from tools.mara_agent import MARAConfig
from tools.mara_diagnostics import checkpoint_config, stable_id
from tools.mara_gate_sweep import METRICS, _quality_batch
from tools.mara_reward import EvidenceStore, CachedBatches, split_source, state_hash
from tools.mara_ppo import (
    PROTOCOL, ARMS, INPUTS, PPOConfig, MARAPPOAgent, environment_config,
    optimizers, update, score_trajectory,
)


def load_work(path):
    return shared.load_work(path, protocol=PROTOCOL)


def new_agent(cfg, seed, refiner_state):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        agent = MARAPPOAgent(MARAConfig(**cfg["mara_config"]), PPOConfig(**cfg["ppo_config"]))
        agent.initialize_refiner(refiner_state)
    return agent


def prepare(args):
    seeds = [int(x) for x in args.train_seeds.replace(",", " ").split()]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("TRAIN_SEEDS must be unique nonnegative integers")
    payload = shared.load_torch(args.mara_ckpt)
    if payload.get("mara_algorithm"):
        raise ValueError("MARA_CKPT must be the original trained legacy MARA Refiner, not a PPO output")
    template = checkpoint_config(payload)
    env = environment_config(template, args.steps or 3)
    if env.max_steps < 2 or args.source_dataset not in args.datasets.split() or args.gradient_images:
        raise ValueError("Need horizon>=2, source dataset included and no legacy gradient probe")
    # A zero delta AND score head is the legacy identity initialization, not an
    # actionable trained Refiner. Do not spend a full RL experiment on it.
    if all(torch.count_nonzero(v) == 0 for k, v in payload["mara_agent"].items()
           if k.startswith(("delta_head.", "score_head."))):
        raise ValueError("Refiner appears identity-initialized; supply a trained MARA checkpoint")
    ppo = PPOConfig(trajectories=args.trajectories, update_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr)
    local = dict(mara_config=asdict(env), ppo_config=asdict(ppo))
    initial = {str(seed): state_hash(new_agent(local, seed, payload["mara_agent"]).state_dict()) for seed in seeds}
    source = None
    if args.source_work_dir:
        path = Path(args.source_work_dir).resolve()
        old, manifest = shared.load_json(path/"config.json"), shared.load_json(path/"manifest.json")
        if path == Path(args.work_dir).resolve() or old.get("protocol") not in (
            "mara_gate_sweep_v1", "mara_early_stop_ab_v1", "mara_early_stop_ab_v2", "mara_reward_r_v1", PROTOCOL):
            raise ValueError("Invalid source cache directory/protocol")
        if old["checkpoint_sha256"] != shared.sha256_file(args.mara_ckpt) or old["manifest_sha256"] != shared.digest(manifest):
            raise ValueError("Source cache checkpoint/manifest mismatch")
        source = dict(path=str(path), signature=shared.digest(old), config_sha256=shared.sha256_file(path/"config.json"),
                      manifest_sha256=old["manifest_sha256"])
    shared.prepare(args, protocol=PROTOCOL, arms=ARMS, extra_config=dict(
        **local, source_cache=source, dataset_sample_limits={args.source_dataset: args.source_samples_per_category},
        source_dataset=args.source_dataset, source_split_seed=args.source_split_seed,
        calibration_fraction=args.calibration_fraction, train_seeds=seeds,
        jobs=[dict(id=f"{arm}_seed{seed}", arm=arm, seed=seed) for seed in seeds for arm in ARMS],
        initial_state_hashes=initial, refiner_state_sha256=state_hash(payload["mara_agent"]),
        epochs=args.epochs, train_batch_size=args.train_batch_size,
        export_batch_size=args.export_batch_size, eval_batch_size=args.eval_batch_size,
        decode_workers=args.decode_workers, metric_workers=args.metric_workers,
        cpu_cache_gb=args.cpu_cache_gb, gpu_cache_gb=args.gpu_cache_gb,
        value_probe_trajectories=args.value_probe_trajectories, category_shards=False, empty_gt_policy="exclude",
        precision="fp32_no_autocast_tf32_off", transition="frozen_trained_refiner_no_gain_veto_v1",
        selection_note="Source-only MARA split; fixed final epoch; no target selection; deterministic hierarchical greedy evaluation",
        gradient_note="Only Actor/Critic train; immutable trained Refiner/Base; fresh on-policy complete episodes; no VLM",
        sampling_note="Same source split seed as R; full source; paired small target subset is a mechanism pilot, not final benchmark",
    ))
    print("PPO: V versus zero-baseline MC; gamma=lambda=1; HR rewards; STOP=0; REFINE cost=.003; no gain veto", flush=True)


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
    print(f"PPO source train={len(split['train_ids'])} holdout={len(split['calibration_ids'])}; targets={len(split['target_ids'])}", flush=True)


class OptimizerPair:
    """Small adapter for the existing atomic epoch+RNG resume implementation."""
    def __init__(self, actor, critic):
        self.actor, self.critic = actor, critic

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict() if self.critic else None)

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        if self.critic:
            self.critic.load_state_dict(state["critic"])


def train_epoch(agent, store, records, cfg, job, epoch, pair, scheduler):
    rows = []
    agent.train()
    loader = CachedBatches(store, records, cfg["train_batch_size"], job["seed"], epoch)
    entropy = agent.ppo.entropy_coef*(1-epoch/max(1, cfg["epochs"]-1))
    for index, batch in enumerate(loader):
        samples, summary = agent.collect(batch["_mara_cached"], batch["mask"], batch["is_anomaly"])
        metrics = update(agent, samples, pair.actor, pair.critic, entropy_coef=entropy)
        rows.append(dict(**summary, **metrics))
        scheduler.step()
        if index % 20 == 0 or index+1 == len(loader):
            print(f"{job['id']} epoch={epoch+1}/{cfg['epochs']} batch={index+1}/{len(loader)} "
                  f"G={summary['return_mean']:.5f} Vloss={metrics['value_loss']:.6f} "
                  f"refines={summary['refine_mean']:.2f} KL={metrics['approx_kl']:.5f}", flush=True)
        del samples
    total = sum(r["trajectories"] for r in rows)
    meters = {k: sum(r[k]*r["trajectories"] for r in rows)/total for k in rows[0]
              if k not in ("trajectories", "decisions", "negative_refines", "optimizer_steps")}
    meters.update({k: sum(r[k] for r in rows) for k in ("trajectories", "decisions", "negative_refines", "optimizer_steps")})
    return meters


def train_job(root, cfg, manifest, signature, split, store, job, device):
    directory = root/"jobs"/job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(signature=signature, job=job, split_sha256=shared.digest(split),
                    initial_state_sha256=cfg["initial_state_hashes"][str(job["seed"])])
    original = shared.load_torch(cfg["mara_ckpt"])
    agent = new_agent(cfg, job["seed"], original["mara_agent"])
    if state_hash(agent.state_dict()) != identity["initial_state_sha256"]:
        raise ValueError("Paired initial state changed")
    agent.ppo = replace(agent.ppo, use_value=ARMS[job["arm"]])
    agent.critic.requires_grad_(agent.ppo.use_value)
    agent.to(device)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(job["seed"])
        torch.cuda.reset_peak_memory_stats(device)
    pair = OptimizerPair(*optimizers(agent))
    # Constant, separately configured actor/critic learning rates. Persist the
    # actor scheduler too, for the common atomic resume schema.
    scheduler = torch.optim.lr_scheduler.LambdaLR(pair.actor, lambda _: 1.)
    train_ids = set(split["train_ids"])
    records = sorted((r for r in manifest if r["id"] in train_ids), key=lambda r: r["id"])
    history, first = [], 0
    saved = resume_payload(directory, identity)
    if saved:
        if not 0 <= saved["epoch"] < cfg["epochs"] or len(saved["history"]) != saved["epoch"]+1:
            raise ValueError("Invalid resume epoch/history")
        agent.load_state_dict(saved["mara_agent"], strict=True)
        pair.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, first = saved["history"], saved["epoch"]+1
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
        print(f"Resume {job['id']} epoch={first+1}, optimizer/RNG restored", flush=True)
    start = time.monotonic()
    for epoch in range(first, cfg["epochs"]):
        try:
            meters = train_epoch(agent, store, records, cfg, job, epoch, pair, scheduler)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("PPO OOM; no silent sample/batch changes. Rerun same script with EXEC_GPU_CACHE_GB=0 "
                               "to resume the last complete epoch. Changing scientific batches needs a new WORK_DIR.") from None
        if not all(math.isfinite(v) for v in meters.values()):
            raise ValueError("Nonfinite PPO epoch metrics")
        if state_hash(agent.transition.state_dict()) != cfg["refiner_state_sha256"]:
            raise ValueError("Frozen Refiner changed")
        history.append(dict(job=job["id"], arm=job["arm"], seed=job["seed"], epoch=epoch+1, **meters))
        save_resume(directory, identity, epoch, agent, pair, scheduler, history, device)
        shared.write_csv(directory/"training.csv", history)
        print(f"Committed {job['id']} epoch {epoch+1}/{cfg['epochs']}", flush=True)
    final, meta = directory/"final.pth", directory/"training.json"
    if not final.exists() or not meta.exists():
        # Full Base fields + new schema; deliberately omit the old mara_agent so
        # old evaluators fail instead of silently running the historical policy.
        payload = {k: v for k, v in original.items() if k not in ("mara_agent", "optimizer", "scheduler")}
        payload.update(mara_algorithm=PROTOCOL, mara_config=asdict(agent.cfg), ppo_config=asdict(agent.ppo),
                       mara_ppo_agent={k: v.detach().cpu() for k, v in agent.state_dict().items()},
                       epoch=cfg["epochs"]-1, ppo_experiment=identity)
        shared.atomic_torch(final, payload)
        shared.atomic_json(meta, dict(**identity, final_sha256=shared.sha256_file(final),
            final_state_sha256=state_hash(agent.state_dict()), refiner_state_sha256=state_hash(agent.transition.state_dict()),
            epochs=cfg["epochs"], history=history, seconds_this_invocation=time.monotonic()-start,
            peak_allocated_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.,
            gpu_cache_gb=store.gpu_bytes/2**30, cpu_cache_gb=store.cpu_bytes/2**30))
    trained = shared.load_json(meta)
    if (any(trained.get(k) != v for k, v in identity.items()) or shared.sha256_file(final) != trained["final_sha256"]
            or trained["final_state_sha256"] != state_hash(agent.state_dict())):
        raise ValueError("Final PPO checkpoint identity/hash mismatch")
    return directory, agent.eval(), trained


def trace_rows(result, chunk, labels, mask, agent, arm, source):
    rows = []
    for t, step in enumerate(result["steps"]):
        numbers = {k: step[k].cpu().tolist() for k in ("valid", "done", "old_value", "return", "advantage", "reward",
                   "quality_before", "quality_after", "execute", "p_stop", "pixel_change", "global_change", "box")}
        actions = {k: v.cpu().tolist() for k, v in step["actions"].items()}
        parts = [_quality_batch(step[k], step[g], mask, labels, agent.cfg)
                 for k, g in (("before_hr", "before_logits"), ("after_hr", "after_logits"))]
        for i, record in enumerate(chunk):
            row = {k: v[i] for k, v in numbers.items()}
            row.update({k: v[i] for k, v in actions.items()})
            for j, name in enumerate(("quality", "q_cls", "q_loc", "q_conf", "q_bg", "gt_pixels")):
                row["before_hr_"+name], row["after_hr_"+name] = float(parts[0][i, j]), float(parts[1][i, j])
                if name.startswith("q_"):
                    row["delta_"+name] = float(parts[1][i, j]-parts[0][i, j])
            row.update(sample_id=record["id"], dataset=record["dataset"], category=record["category"], arm=arm,
                       image_label=int(labels[i]), step=t, role="source_calibration" if record["dataset"] == source else "target",
                       before_hr_quality=row["quality_before"], after_hr_quality=row["quality_after"],
                       # V estimates stochastic-policy returns, not necessarily this greedy return.
                       evaluation_policy="hierarchical_greedy",
                       stop_reason="inactive" if not row["valid"] else "policy_stop" if row["op"] == 0 else
                                   "horizon" if row["done"] else "continue")
            if row["execute"] == 0 and (row["pixel_change"] != 0 or row["global_change"] != 0):
                raise ValueError("PPO STOP/inactive identity invariant failed")
            rows.append(row)
    return rows


def evaluate_job(directory, agent, trained, root, cfg, manifest, signature, split, store, job):
    eval_signature = shared.digest(dict(training=trained["final_sha256"], split=shared.digest(split), signature=signature))
    categories = defaultdict(list)
    ids = set(split["calibration_ids"]+split["target_ids"])
    for record in manifest:
        if record["id"] in ids:
            categories[(record["dataset"], record["category"])].append(record)
    eval_cfg = {**cfg, "arms": {job["arm"]: {}}}
    futures = []
    with ThreadPoolExecutor(max_workers=cfg["metric_workers"]) as pool:
        for key, records in sorted(categories.items()):
            if gate.read_category(gate.category_path(directory, key), eval_signature, split["audit_sha256"], [r["id"] for r in records]):
                continue
            if len(futures) >= cfg["metric_workers"]:
                futures.pop(0).result()
            rows, maps, masks, labels, bases = [], [], [], [], []
            for offset in range(0, len(records), cfg["eval_batch_size"]):
                chunk = records[offset:offset+cfg["eval_batch_size"]]
                batch = store.batch(chunk)
                result = agent.trajectory({k: batch[k] for k in INPUTS if k in batch}, sample=False)
                score_trajectory(result, batch["mask"], batch["labels"], agent.cfg, agent.ppo)
                rows.extend(trace_rows(result, chunk, batch["labels"], batch["mask"], agent, job["arm"], cfg["source_dataset"]))
                maps.extend(torch.stack([s["after_hr"][:, 1] for s in result["steps"]], 1).cpu().numpy())
                masks.extend(batch["mask"][:, 0].cpu().numpy().astype(bool))
                labels.extend(batch["labels"].cpu().tolist())
                bases.extend(batch["base_prob"][:, 1].cpu().numpy())
            futures.append(pool.submit(gate.finish_category, directory, eval_cfg, eval_signature, split["audit_sha256"],
                                       key, records, rows, {job["arm"]: np.stack(maps)}, np.stack(masks), np.array(labels), np.stack(bases)))
        for future in futures:
            future.result()
    value_probe(directory, agent, cfg, manifest, split, store, job, eval_signature)
    shared.atomic_json(directory/"evaluated.json", dict(signature=eval_signature, job=job, final_sha256=trained["final_sha256"]))


@torch.no_grad()
def value_probe(directory, agent, cfg, manifest, split, store, job, signature):
    """Independent MC sanity check at fixed held-out initial states, no fitting."""
    path = directory/"value_probe.json"
    if path.exists():
        if shared.load_json(path)["signature"] != signature:
            raise ValueError("Stale value probe")
        return
    candidates = defaultdict(list)
    for record in manifest:
        if record["id"] in set(split["calibration_ids"]):
            candidates[(record["category"], record["sampling_stratum"])].append(record)
    records = [min(rs, key=lambda r: stable_id("value_probe", r["id"])) for rs in candidates.values()]
    torch.manual_seed(700001+job["seed"])
    count, rows = cfg["value_probe_trajectories"], []
    for record in records:
        batch = store.batch([record])
        inputs = {k: batch[k].repeat_interleave(count, 0) for k in INPUTS if k in batch}
        result = agent.trajectory(inputs, sample=True)
        score_trajectory(result, batch["mask"].repeat_interleave(count, 0), batch["labels"].repeat_interleave(count, 0), agent.cfg, agent.ppo)
        first = result["steps"][0]
        returns = first["return"]
        mean, value = float(returns.mean()), float(first["old_value"][0])
        rows.append(dict(job=job["id"], arm=job["arm"], seed=job["seed"], sample_id=record["id"],
                         category=record["category"], image_label=int(batch["labels"][0]), trajectories=count,
                         v_prediction=value, mc_return_mean=mean, mc_standard_error=float(returns.std(unbiased=True)/math.sqrt(count)),
                         prediction_minus_mc=value-mean, first_stop_fraction=float((first["actions"]["op"] == 0).float().mean())))
    shared.atomic_json(path, dict(signature=signature, rows=rows))


def train(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, audited = checked_split(root, signature)
    hashes = {r["sample_id"]: r["cache_sha256"] for r in audited["rows"]}
    verified = set()
    def loader(record):
        if record["id"] not in verified:
            shared.verify_source(record)
            cache = gate.evidence(root, cfg, record, signature)
            if shared.load_json(root/"evidence"/f"{record['id']}.json")["sha256"] != hashes[record["id"]]:
                raise ValueError("Evidence changed after audit")
            verified.add(record["id"])
            return cache
        return shared.load_torch(root/"evidence"/f"{record['id']}.pt")
    device = torch.device(args.device)
    store = EvidenceStore(loader, device, cfg["cpu_cache_gb"], cfg["gpu_cache_gb"] if args.gpu_cache_gb is None else args.gpu_cache_gb)
    for index, job in enumerate(cfg["jobs"]):
        if index % cfg["num_shards"] != args.shard_id:
            continue
        directory, agent, trained = train_job(root, cfg, manifest, signature, split, store, job, device)
        store.clear_gpu()
        evaluate_job(directory, agent, trained, root, cfg, manifest, signature, split, store, job)
        del agent
        store.clear_gpu()


def collect(args):
    root, cfg, manifest, signature = load_work(args.work_dir)
    split, _ = checked_split(root, signature)
    categories = defaultdict(list)
    for record in manifest:
        if record["id"] in set(split["calibration_ids"]+split["target_ids"]):
            if not shared.cache_ready(root, record, signature):
                raise ValueError("Evidence missing/changed")
            categories[(record["dataset"], record["category"])].append(record)
    metrics, traces, histories, checkpoints, probes = [], [], [], [], []
    for job in cfg["jobs"]:
        directory = root/"jobs"/job["id"]
        trained = shared.load_json(directory/"training.json")
        if (trained["signature"] != signature or trained["split_sha256"] != shared.digest(split) or trained["job"] != job
                or trained["epochs"] != cfg["epochs"] or trained["refiner_state_sha256"] != cfg["refiner_state_sha256"]
                or shared.sha256_file(directory/"final.pth") != trained["final_sha256"]):
            raise ValueError("Training/frozen Refiner/checkpoint mismatch")
        eval_signature = shared.digest(dict(training=trained["final_sha256"], split=shared.digest(split), signature=signature))
        if shared.load_json(directory/"evaluated.json")["signature"] != eval_signature:
            raise ValueError("Evaluation incomplete")
        histories.extend(trained["history"])
        checkpoints.append(trained)
        probe = shared.load_json(directory/"value_probe.json")
        if probe["signature"] != eval_signature:
            raise ValueError("Value probe mismatch")
        probes.extend(probe["rows"])
        for key, records in sorted(categories.items()):
            result = gate.read_category(gate.category_path(directory, key), eval_signature, split["audit_sha256"], [r["id"] for r in records])
            if result is None or len(result["rows"]) != len(records)*cfg["mara_config"]["max_steps"]:
                raise ValueError("Partial category results")
            metrics.extend(dict(row, job=job["id"], seed=job["seed"], role="source_calibration" if key[0] == cfg["source_dataset"] else "target") for row in result["metrics"])
            traces.extend(dict(row, job=job["id"], seed=job["seed"]) for row in result["rows"])
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["job"], row["dataset"], row["arm"], row["step"])].append(row)
    macro = [dict(job=job, dataset=dataset, arm=arm, step=step, seed=rs[0]["seed"], role=rs[0]["role"],
                  categories=len(rs), images=sum(r["images"] for r in rs),
                  **{k: shared.mean([r.get(k) for r in rs]) for k in (*METRICS, "quality_gain_hr", "mean_step_reward")},
                  **{"valid_categories_"+k: sum(r.get(k) is not None for r in rs) for k in METRICS})
             for (job, dataset, arm, step), rs in sorted(groups.items())]
    reference = {(r["seed"], r["dataset"], r["category"], r["step"]): r for r in metrics if r["arm"] == "MC"}
    comparisons = []
    for row in metrics:
        if row["arm"] != "V":
            continue
        ref = reference[(row["seed"], row["dataset"], row["category"], row["step"])]
        comparisons.append(dict(job=row["job"], seed=row["seed"], dataset=row["dataset"], category=row["category"], step=row["step"],
            **{k+"_V_minus_MC": row[k]-ref[k] if row[k] is not None and ref[k] is not None else None for k in METRICS}))
    summaries = []
    for job in cfg["jobs"]:
        for dataset in cfg["datasets"]:
            for label in (None, 0, 1):
                rs = [r for r in traces if r["job"] == job["id"] and r["dataset"] == dataset and (label is None or r["image_label"] == label)]
                starts = [r for r in rs if r["step"] == 0]
                active = [r for r in rs if r["valid"]]
                if not starts:
                    continue
                totals = defaultdict(float)
                for row in rs:
                    totals[row["sample_id"]] += row["execute"]
                summaries.append(dict(job=job["id"], arm=job["arm"], dataset=dataset, seed=job["seed"],
                    stratum="all" if label is None else "normal" if label == 0 else "anomaly", images=len(starts),
                    first_stop_fraction=shared.mean([r["op"] == 0 for r in starts]),
                    mean_refines=shared.mean(list(totals.values())), multi_step_images=sum(v >= 2 for v in totals.values()),
                    mean_return=shared.mean([r["return"] for r in starts]),
                    negative_refines=sum(r["execute"] > 0 and r["reward"] < 0 for r in active)))
    out = root/"results"
    for name, values in (("metrics", metrics), ("macro_metrics", macro), ("comparisons", comparisons), ("step_trace", traces),
                         ("training", histories), ("value_probe", probes), ("rollout_summary", summaries)):
        shared.write_csv(out/f"{name}.csv", values)
    shared.atomic_json(out/"checkpoints.json", checkpoints)
    (out/"README.md").write_text(
        "# MARA PPO pilot\n\nV: PPO with learned V; MC: same PPO with zero baseline, no value fitting.\n"
        "Same per-seed initialization, trained frozen Refiner/Base, split, reward, budget. No VLM.\n"
        "gamma=lambda=1, full MC return targets; STOP=0; REFINE cost=.003. HR utility weights cls=.5, loc=1, bg=.2.\n"
        "No gain veto, forced Base anchor, immediate STOP auxiliary or supervised Refiner updates. Spatial fusion retained.\n"
        "PPO uses actual stored observations, old log-probabilities and old V; terminal values zero.\n"
        "Primary metrics: fixed final epoch, hierarchical greedy policy. This is not joint-action argmax.\n"
        "value_probe is independent stochastic-policy MC at held-out initial states; do not compare V with one greedy return.\n"
        "MC probe V is zero by design. Explained variance in training is the pre-update value diagnostic.\n"
        "Source holdout is MARA-only: Base/Refiner may have seen it previously. Targets are a small diagnostic subset.\n"
        "Compare V with retrained MC to isolate the baseline. Historical G/R also differ in MDP/training.\n"
        "Check PRO/P-AUROC/normal FP, not just Dice/return/value loss. One seed is only a pilot.\n"
        "Checkpoints and large evidence tensors intentionally excluded.\n", encoding="utf-8")
    shared.atomic_json(root/"complete.json", dict(protocol=PROTOCOL, signature=signature,
        split_sha256=shared.digest(split), jobs=len(cfg["jobs"]), final_epoch=cfg["epochs"],
        metrics_sha256=shared.sha256_file(out/"metrics.csv"), refiner_state_sha256=cfg["refiner_state_sha256"],
        source_train=len(split["train_ids"]), source_calibration=len(split["calibration_ids"]), target_images=len(split["target_ids"])))
    target = root/"ppo_report.zip"
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
    print(f"Complete: {target} ({target.stat().st_size/2**20:.2f} MiB); upload only this report", flush=True)


def parser():
    p = shared.parser(stage_choices=("prepare", "export", "audit", "train", "collect"))
    p.description = __doc__
    p.set_defaults(gradient_images=0, steps=3)
    p.add_argument("--source_work_dir", default="")
    p.add_argument("--source_dataset", default="visa")
    p.add_argument("--source_samples_per_category", type=int, default=0)
    p.add_argument("--source_split_seed", type=int, default=20261018)
    p.add_argument("--calibration_fraction", type=float, default=.2)
    p.add_argument("--train_seeds", default="0")
    for flag, default in (("epochs", 30), ("train_batch_size", 8), ("trajectories", 4), ("ppo_epochs", 4),
                          ("minibatch_size", 16), ("export_batch_size", 8), ("eval_batch_size", 8),
                          ("decode_workers", 2), ("metric_workers", 2), ("value_probe_trajectories", 32)):
        p.add_argument("--"+flag, type=int, default=default)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--cpu_cache_gb", type=float, default=4.)
    p.add_argument("--gpu_cache_gb", type=float, default=None)
    return p


def main():
    args = parser().parse_args()
    if args.stage == "prepare" and args.gpu_cache_gb is None:
        args.gpu_cache_gb = 12.
    if min(args.epochs, args.train_batch_size, args.trajectories, args.ppo_epochs, args.minibatch_size,
           args.export_batch_size, args.eval_batch_size, args.decode_workers, args.metric_workers,
           args.cpu_threads, args.num_shards, args.image_size) < 1 or args.value_probe_trajectories < 2:
        raise ValueError("Invalid PPO budgets")
    if min(args.shard_id, args.source_samples_per_category, args.samples_per_category, args.source_split_seed) < 0:
        raise ValueError("Invalid sampling/shard")
    if not 0 < args.calibration_fraction < 1 or args.pro_num_th < 2 or not 0 < args.pro_max_fpr <= 1:
        raise ValueError("Invalid split/PRO config")
    if not math.isfinite(args.cpu_cache_gb) or args.cpu_cache_gb < 0 or (args.gpu_cache_gb is not None and
            (not math.isfinite(args.gpu_cache_gb) or args.gpu_cache_gb < 0)):
        raise ValueError("Invalid cache budgets")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage != "prepare" and args.shard_id >= load_work(args.work_dir)[1]["num_shards"]:
        raise ValueError("Shard outside prepared layout")
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
