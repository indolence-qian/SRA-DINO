#!/usr/bin/env python3
"""Read-only v1 cache reuse -> P0 diagnostic -> P1 constrained heads -> evaluation."""
import argparse
from collections import defaultdict
import csv
from datetime import timedelta
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import upstream_localization as old
from tools.vlm_review import atomic_json, digest, file_digest, load_json
from tools.upstream_repair import (ConstrainedHead, corrected_probability, constrained_loss,
                                  source_metrics, aggregate_source, selection_score)

FILES = ("upstream_repair.py", "tools/upstream_repair.py")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def prepare(a):
    if not a.source_work_dir:
        raise ValueError("--source_work_dir must name the sealed v1 experiment")
    root, source = Path(a.work_dir).resolve(), Path(a.source_work_dir).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source cache and repair output must be separate, non-nested directories")
    cfg = old.cfg_load(source)
    sealed = load_json(source / "sealed.json")
    if sealed["fingerprint"] != digest(cfg):
        raise ValueError("Source config/seal mismatch")
    alphas = [float(x) for x in a.alphas.split(",")]
    if not alphas or len(set(alphas)) != len(alphas) or any(not 0 < x <= 1 for x in alphas):
        raise ValueError("alphas must be unique in (0,1]; Base alpha=0 is always included")
    if (a.epochs < 1 or a.batch_size < 1 or a.hidden < 8 or a.hidden % 8 or a.lr <= 0 or
            a.num_shards < 1 or a.val_pixels < 128 or not 0 < a.fpr < .5 or
            not 0 < a.hard_fraction <= 1 or min(a.bg_weight, a.residual_weight, a.auc_tolerance, a.fpr_slack) < 0):
        raise ValueError("Invalid training/selection settings")
    names = ("epochs", "batch_size", "hidden", "lr", "num_shards", "val_pixels", "fpr", "hard_fraction",
             "bg_weight", "residual_weight", "auc_tolerance", "fpr_slack")
    settings = {k: getattr(a, k) for k in names}
    settings.update(protocol="upstream_P0_P1_v1", source=str(source), source_config_sha=file_digest(source/"config.json"),
                    seal_sha=file_digest(source/"sealed.json"), alphas=sorted(alphas),
                    code={name: file_digest(old.REPO/name) for name in FILES})
    path = root / "repair_config.json"
    if path.exists() and load_json(path) != settings:
        raise ValueError("Repair settings changed; use NEW WORK_DIR")
    if not path.exists() and (root/"config.json").exists():
        raise ValueError("Output contains another experiment; use NEW WORK_DIR")
    # Verify old assets without altering config, seal, features, reviews or heads.
    old.FeatureDataset(source, cfg, old.records_for(source), verify=True)
    atomic_json(path, settings)
    print(f"Reusing sealed cache READ ONLY: {source}; writing P0/P1 to {root}", flush=True)
    print("Only source train/val supervise/select heads. Target P0 is diagnostic only. No VLM/DINO reload, no MARA.", flush=True)


def context(a):
    root = Path(a.work_dir).resolve()
    settings = load_json(root/"repair_config.json")
    for name, sha in settings["code"].items():
        if file_digest(old.REPO/name) != sha:
            raise ValueError("Repair code changed; use NEW WORK_DIR")
    source = Path(settings["source"])
    if file_digest(source/"config.json") != settings["source_config_sha"] or file_digest(source/"sealed.json") != settings["seal_sha"]:
        raise ValueError("Source cache/config changed")
    return root, source, old.cfg_load(source), settings


def device_for(rank=0):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def loader(source, cfg, rows, settings, workers, verify=True):
    return DataLoader(old.FeatureDataset(source, cfg, rows, verify=verify),
                      batch_size=settings["batch_size"], num_workers=workers)


def infer_group(model, batches, device, mode):
    base, delta, masks = [], [], []
    with torch.inference_mode():
        for batch in batches:
            masks.extend(batch["mask"][:, 0].numpy().astype(bool))
            batch = {k: v.to(device) for k, v in batch.items() if k != "mask"}
            residual = old.head_forward(model, batch, mode)
            base.extend(batch["base"][:, 0].cpu().numpy())
            delta.extend(residual[:, 0].cpu().numpy())
    return np.asarray(base), np.asarray(delta), np.asarray(masks)


def adjust_numpy(base, delta, alpha):
    return corrected_probability(torch.from_numpy(base), torch.from_numpy(delta), alpha).numpy()


def groups_for(source, partitions):
    groups = defaultdict(list)
    for r in old.records_for(source):
        if r["partition"] in partitions:
            groups[(r["partition"], r["dataset"], r["category"])].append(r)
    return dict(sorted(groups.items()))


def validate(model, source, cfg, settings, device, mode, workers, baseline_only=False):
    per_alpha = {str(x): [] for x in [0.] + settings["alphas"]}
    for key, rows in groups_for(source, {"val"}).items():
        base, delta, masks = infer_group(model, loader(source, cfg, rows, settings, workers, verify=False), device, mode)
        seed = int(digest([cfg["seed"], list(key)])[:8], 16)
        base_value = source_metrics(masks, base, base, seed, settings["val_pixels"], settings["fpr"])
        per_alpha["0.0"].append(base_value)
        if not baseline_only:
            for alpha in settings["alphas"]:
                metrics = source_metrics(masks, adjust_numpy(base, delta, alpha), base, seed,
                                         settings["val_pixels"], settings["fpr"])
                per_alpha[str(alpha)].append(metrics)
        print(f"source validation {mode} {key[1]}/{key[2]} samples={len(rows)}", flush=True)
    return {k: dict(mean=aggregate_source(v), categories=v) for k, v in per_alpha.items() if v}


def train(a):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DistributedSampler
    root, source, cfg, s = context(a)
    world, rank, local_rank = (int(os.environ.get(k, "1" if k == "WORLD_SIZE" else "0"))
                               for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if world != s["num_shards"]:
        raise ValueError("DDP WORLD_SIZE must match prepared num_shards")
    device = device_for(local_rank)
    if world > 1:
        # Rank 0 performs CPU regional validation while other ranks wait.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2))
    try:
        torch.manual_seed(cfg["seed"])
        dataset = old.FeatureDataset(source, cfg, old.records_for(source, "train"))
        if len(dataset) < world:
            raise ValueError("Not enough source train samples for DDP")
        # Verify only SOURCE validation before optimization; target masks never loaded here.
        if rank == 0:
            old.FeatureDataset(source, cfg, old.records_for(source, "val"))
        sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=cfg["seed"]) if world > 1 else None
        gen = torch.Generator()
        batches = DataLoader(dataset, batch_size=s["batch_size"], sampler=sampler, shuffle=sampler is None,
                             num_workers=a.workers, generator=gen)
        shape = load_json(source/"sealed.json")["shape"]
        raw = ConstrainedHead(shape[0], shape[1], s["hidden"]).to(device)
        optimizer = torch.optim.AdamW(raw.parameters(), lr=s["lr"], weight_decay=1e-4)
        out = root/"heads"/a.mode
        fingerprint = digest([s, a.mode])
        start, best_score, history = 0, None, []
        if (out/"last.pt").exists():
            last = torch.load(out/"last.pt", map_location=device, weights_only=False)
            if last["fingerprint"] != fingerprint:
                raise ValueError("Checkpoint settings mismatch")
            raw.load_state_dict(last["state_dict"])
            optimizer.load_state_dict(last["optimizer"])
            start, best_score, history = last["epoch"]+1, last["best_score"], last["history"]
            selected = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
            if selected["fingerprint"] != fingerprint:
                raise ValueError("Selected checkpoint settings mismatch")
            # If interrupted after saving best but before saving last, never
            # overwrite that already validated best with a worse candidate.
            best_score = max(best_score, selected["score"])
        model = DistributedDataParallel(raw, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else raw
        if rank == 0 and start == 0:
            raw.eval()
            baseline = validate(raw, source, cfg, s, device, a.mode, a.workers, baseline_only=True)["0.0"]
            best_score = selection_score(baseline["mean"], baseline["mean"], baseline["categories"], baseline["categories"], s["auc_tolerance"], s["fpr_slack"])
            old.save_torch(out/"best.pt", dict(state_dict=raw.state_dict(), alpha=0., epoch=-1, shape=shape,
                                              fingerprint=fingerprint, source_validation=baseline,
                                              selection="BASE_FALLBACK", score=best_score))
        if world > 1:
            dist.barrier()
        for epoch in range(start, s["epochs"]):
            if sampler:
                sampler.set_epoch(epoch)
            gen.manual_seed(cfg["seed"]+epoch)
            model.train()
            sums = torch.zeros(6, dtype=torch.float64, device=device)
            for batch in batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                delta = old.head_forward(model, batch, a.mode)
                loss, terms = constrained_loss(batch["base"], delta, batch["mask"], s["bg_weight"], s["residual_weight"], s["hard_fraction"])
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite constrained loss")
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                sums[0] += loss.detach().sum()
                sums[1] += len(loss)
                for i, key in enumerate(("bce", "dice", "background", "magnitude"), 2):
                    sums[i] += terms[key].detach().sum()
            if world > 1:
                dist.all_reduce(sums)
            if rank == 0:
                raw.eval()
                metrics = validate(raw, source, cfg, s, device, a.mode, a.workers)
                baseline = metrics["0.0"]
                decisions = []
                for alpha in s["alphas"]:
                    candidate = metrics[str(alpha)]
                    score = selection_score(candidate["mean"], baseline["mean"], candidate["categories"], baseline["categories"], s["auc_tolerance"], s["fpr_slack"])
                    decisions.append(dict(alpha=alpha, eligible=score is not None, score=score))
                    if score is not None and score > best_score + 1e-8:
                        best_score = score
                        old.save_torch(out/"best.pt", dict(state_dict=raw.state_dict(), alpha=alpha, epoch=epoch,
                                                          shape=shape, fingerprint=fingerprint, source_validation=candidate,
                                                          selection="SOURCE_VALIDATED", score=score))
                history.append(dict(epoch=epoch+1, train_loss=float(sums[0]/sums[1]),
                                    loss_terms={k: float(sums[i]/sums[1]) for i, k in enumerate(("bce", "dice", "background", "magnitude"), 2)},
                                    validation=metrics, decisions=decisions, best_score=best_score))
                old.save_torch(out/"last.pt", dict(state_dict=raw.state_dict(), optimizer=optimizer.state_dict(),
                                                  epoch=epoch, fingerprint=fingerprint, best_score=best_score, history=history))
                atomic_json(out/"history.json", history)
                best = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
                print(f"{a.mode} epoch={epoch+1}/{s['epochs']} loss={history[-1]['train_loss']:.5f} selected_alpha={best['alpha']} selected_epoch={best['epoch']+1} {best['selection']}", flush=True)
            if world > 1:
                dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def evaluate(a, p0=False):
    from tools.vlm_next_diagnostics import map_metrics
    root, source, cfg, s = context(a)
    if not 0 <= a.shard_id < s["num_shards"]:
        raise ValueError("Bad shard id")
    device = device_for()
    shape = load_json(source/"sealed.json")["shape"]
    models = {}
    identities = {}
    if p0:
        models["zero_residual"] = (ConstrainedHead(shape[0], shape[1], s["hidden"]).to(device).eval(), 1., "IDENTITY", 0)
        # Optional old trained heads: decompose their residuals without retraining
        # or interpreting these target diagnostics as a checkpoint selection rule.
        for mode in ("visual", "vlm"):
            path = source/"heads"/mode/"best.pt"
            if path.exists():
                saved = torch.load(path, map_location=device, weights_only=False)
                if saved["settings"]["cache_sha"] != s["seal_sha"]:
                    raise ValueError("Old head does not match source feature cache")
                head = ConstrainedHead(shape[0], shape[1], saved["settings"]["hidden"]).to(device).eval()
                head.load_state_dict(saved["state_dict"])
                models["old_"+mode] = (head, 1., "V1_DIAGNOSTIC_ONLY", saved["epoch"]+1)
                identities[mode] = file_digest(path)
    else:
        for mode in ("visual", "vlm"):
            saved = torch.load(root/"heads"/mode/"best.pt", map_location=device, weights_only=False)
            if saved["fingerprint"] != digest([s, mode]):
                raise ValueError("Selected head fingerprint mismatch")
            head = ConstrainedHead(shape[0], shape[1], s["hidden"]).to(device).eval()
            head.load_state_dict(saved["state_dict"])
            models[mode] = (head, saved["alpha"], saved["selection"], saved["epoch"]+1)
    if not p0:
        identities = {m: file_digest(root/"heads"/m/"best.pt") for m in models}
    groups = list(groups_for(source, {"val", "eval"} if p0 else {"eval"}).items())
    for key, records in groups[a.shard_id::s["num_shards"]]:
        # Category-level batches bound memory. No target selection/tuning here.
        maps, masks, residuals = defaultdict(list), [], defaultdict(list)
        with torch.inference_mode():
            for batch in loader(source, cfg, records, s, a.workers):
                masks.extend(batch["mask"][:, 0].numpy().astype(bool))
                maps["base"].extend(batch["base"][:, 0].numpy())
                if p0:
                    maps["legacy_clamp"].extend(batch["base"][:, 0].clamp(1e-5, 1-1e-5).numpy())
                batch = {k: v.to(device) for k, v in batch.items() if k != "mask"}
                for mode, (model, alpha, _, _) in models.items():
                    head_mode = mode.removeprefix("old_") if mode != "zero_residual" else "vlm"
                    delta = old.head_forward(model, batch, head_mode)
                    if mode.startswith("old_"):
                        prob = (torch.logit(batch["base"].clamp(1e-5, 1-1e-5))+delta).sigmoid()
                    else:
                        prob = corrected_probability(batch["base"], delta, alpha)
                    if mode == "zero_residual" and not torch.equal(prob, batch["base"]):
                        raise AssertionError("P0 FAILED: zero head differs from Base")
                    maps[mode].extend(prob[:, 0].cpu().numpy())
                    residuals[mode].extend((delta*alpha)[:, 0].cpu().numpy())
        masks = np.asarray(masks)
        base = np.asarray(maps["base"])
        rows = []
        for mode, values in maps.items():
            pred = np.asarray(values)
            metric = map_metrics(masks, pred)
            operating = source_metrics(masks, pred, base, int(digest(list(key))[:8], 16), s["val_pixels"], s["fpr"])
            small, hit, recovered, lost = small_counts(masks, pred, base)
            residual = np.asarray(residuals[mode]) if mode in residuals else np.zeros_like(base)
            rows.append(dict(partition=key[0], dataset=key[1], category=key[2], mode=mode, samples=len(records),
                             **metric, small_components=small, small_hits_at_05=hit, recovered_small_at_05=recovered,
                             region_recall_at_target_fpr_DIAGNOSTIC=operating["region_recall_at_fpr"],
                             small_hit_at_target_fpr_DIAGNOSTIC=operating["small_hit_at_fpr"],
                             realized_target_fpr_DIAGNOSTIC=operating["realized_fpr"],
                             lost_small_at_05=lost, background_FPR_at_05=float(np.mean(pred[~masks]>=.5)),
                             pixel_recall_at_05=float(np.mean(pred[masks]>=.5)) if masks.any() else None,
                             mean_gt_logit_residual=float(residual[masks].mean()) if masks.any() else None,
                             mean_bg_logit_residual=float(residual[~masks].mean()),
                             mean_bg_probability_increase=float(np.maximum(pred-base, 0)[~masks].mean()),
                             changed_fraction=float(np.mean(np.abs(pred-base)>1e-6)),
                             alpha=models[mode][1] if mode in models else 0.,
                             selection=models[mode][2] if mode in models else "REFERENCE",
                             selected_epoch=models[mode][3] if mode in models else 0))
        counts = dict(pixels=int(base.size), below_floor=int((base<1e-5).sum()), above_ceiling=int((base>1-1e-5).sum()),
                      gt_pixels=int(masks.sum()), gt_below_floor=int(((base<1e-5)&masks).sum()),
                      bg_pixels=int((~masks).sum()), bg_below_floor=int(((base<1e-5)&~masks).sum()),
                      exact_zero=int((base==0).sum()), exact_one=int((base==1).sum()))
        folder = root/("p0" if p0 else "results")
        atomic_json(folder/("_".join(key)+".json"), dict(fingerprint=digest(s), heads=identities, rows=rows, counts=counts))
        print(f"{'P0' if p0 else 'P1'} evaluated {'/'.join(key)}", flush=True)


def small_counts(masks, pred, base):
    import cv2
    total = hits = recovered = lost = 0
    for gt, p, b in zip(masks, pred, base):
        n, cc = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            region = cc == i
            if region.sum() > gt.size*.001:
                continue
            hit, old_hit = np.mean(p[region]>=.5)>=.1, np.mean(b[region]>=.5)>=.1
            total += 1
            hits += int(hit)
            recovered += int(hit and not old_hit)
            lost += int(old_hit and not hit)
    return total, hits, recovered, lost


def report(a, p0=False):
    root, source, _, s = context(a)
    folder = root/("p0" if p0 else "results")
    rows, counts = [], []
    head_root = source if p0 else root
    expected = {m: file_digest(head_root/"heads"/m/"best.pt") for m in ("visual", "vlm")
                if not p0 or (head_root/"heads"/m/"best.pt").exists()}
    for key in groups_for(source, {"val", "eval"} if p0 else {"eval"}):
        value = load_json(folder/("_".join(key)+".json"))
        if value["fingerprint"] != digest(s) or value["heads"] != expected:
            raise ValueError("Stale evaluation")
        rows.extend(value["rows"])
        counts.append(dict(partition=key[0], dataset=key[1], category=key[2], **value["counts"]))
    for part, name, mode in sorted({(r["partition"], r["dataset"], r["mode"]) for r in rows}):
        group = [r for r in rows if (r["partition"],r["dataset"],r["mode"]) == (part,name,mode)]
        mean = dict(group[0], category="MEAN")
        for k in group[0]:
            if k in ("partition", "dataset", "category", "mode", "selection", "alpha", "selected_epoch"):
                continue
            values = [r[k] for r in group if r[k] is not None]
            mean[k] = sum(values) if k in ("samples", "small_components", "small_hits_at_05", "recovered_small_at_05", "lost_small_at_05") else float(np.mean(values)) if values else None
        rows.append(mean)
        print(mean, flush=True)
    write_csv(folder/"metrics.csv", rows)
    if p0:
        write_csv(folder/"probability_audit.csv", counts)
    atomic_json(folder/"summary.json", rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["prepare", "p0", "p0_report", "train", "evaluate", "report"], required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--source_work_dir", default="")
    p.add_argument("--mode", choices=["visual", "vlm"], default="vlm")
    p.add_argument("--alphas", default="0.1,0.25,0.5,1")
    for k, v in dict(epochs=15, batch_size=4, hidden=96, num_shards=2, shard_id=0, workers=2, val_pixels=8192).items():
        p.add_argument("--"+k, type=int, default=v)
    for k, v in dict(lr=1e-4, fpr=.01, hard_fraction=.01, bg_weight=1., residual_weight=.01, auc_tolerance=.002, fpr_slack=.002).items():
        p.add_argument("--"+k, type=float, default=v)
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    dict(prepare=prepare, p0=lambda a: evaluate(a, True), p0_report=lambda a: report(a, True),
         train=train, evaluate=evaluate, report=report)[a.stage](a)
