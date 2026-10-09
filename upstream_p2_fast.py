#!/usr/bin/env python3
"""Two independent GPU workers, resident frozen data, resumable P2 screening."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import gc
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
import traceback

import cv2
import numpy as np
import torch

import upstream_p2 as p2
from tools.upstream_p2 import spatial_matching
from tools.upstream_p2_fast import CachedMatchingHead, ValidationPlan, batch_plan_indices, fast_loss, geometry
from tools.upstream_repair import aggregate_source, corrected_probability, selection_score
from tools.upstream_semantic_audit import shuffled_semantics
from tools.vlm_review import atomic_json, digest, file_digest, load_json

FILES = ("upstream_p2_fast.py", "tools/upstream_p2_fast.py", "run_exp_upstream_p2_fast.sh")
JOBS = (("p2a", "real"), ("p2a", "shuffled"), ("p2a", "visual"),
        ("p2b", "visual"), ("p2b", "real"), ("p2b", "shuffled"))
TERMS = ("bce", "dice", "background", "magnitude", "component", "ranking")


def prepare(a):
    root = Path(a.work_dir).resolve()
    if (root/"p2_config.json").exists() and not (root/"fast_config.json").exists():
        raise ValueError("Fast output must use a NEW WORK_DIR, not an existing P2 experiment")
    if min(a.micro_batch, a.cpu_threads, a.pack_workers, a.until_epoch) < 1 or a.until_epoch > a.epochs:
        raise ValueError("Invalid runtime limits")
    if not np.isfinite([a.reserve_gb, a.gpu_cache_gb]).all() or min(a.reserve_gb, a.gpu_cache_gb) < 0:
        raise ValueError("Invalid cache budget")
    if a.cpu and a.precision != "fp32":
        raise ValueError("CPU fixtures require --precision fp32")
    p2.prepare(a)
    s = load_json(root/"p2_config.json")
    fast = dict(protocol="upstream_P2_fast_v1", p2_sha=digest(s), precision=a.precision,
                effective_batch=s["batch_size"], order="seed_plus_epoch_randperm", loss="equivalent_vectorized_P2",
                code={name: file_digest(p2.old.REPO/name) for name in FILES})
    path = root/"fast_config.json"
    if path.exists() and load_json(path) != fast:
        raise ValueError("Fast protocol/precision/code changed; use NEW WORK_DIR")
    atomic_json(path, fast)
    print(f"Fast: independent workers; effective batch={s['batch_size']}; planned epochs={s['epochs']}; this run until={a.until_epoch}", flush=True)


def context(a):
    root, source, cfg, s = p2.context(a)
    fast = load_json(root/"fast_config.json")
    if fast["p2_sha"] != digest(s):
        raise ValueError("Fast/P2 config mismatch")
    for name, sha in fast["code"].items():
        if file_digest(p2.old.REPO/name) != sha:
            raise ValueError("Fast code changed; use NEW WORK_DIR")
    return root, source, cfg, s, fast


def diagnostics(a):
    root, source, _, s, _ = context(a)
    donor = Path(a.diagnostic_work_dir).resolve() if a.diagnostic_work_dir else None
    if donor and (donor/"p2_config.json").exists():
        _, old_source, _, old_s = p2.context(type("Args", (), {"work_dir": str(donor)})())
        same = old_source == source and all(old_s[k] == s[k] for k in
               ("source_config_sha", "seal_sha", "val_pixels", "small_fraction", "review_per_category", "match_panels"))
        if same and all((donor/stage/"complete.json").exists() for stage in ("quality", "matching")):
            for stage in ("quality", "matching"):
                p2.require_stage(donor, stage, old_s)
            atomic_json(root/"diagnostics.json", dict(source_config_sha=s["source_config_sha"], seal_sha=s["seal_sha"],
                        directory=str(donor), config_sha=file_digest(donor/"p2_config.json"),
                        complete={stage: file_digest(donor/stage/"complete.json") for stage in ("quality", "matching")}))
            print(f"Reuse existing quality/matching READ ONLY: {donor}", flush=True)
            return
    p2.quality(a)
    p2.matching(a)
    atomic_json(root/"diagnostics.json", dict(source_config_sha=s["source_config_sha"], seal_sha=s["seal_sha"],
                directory=str(root), config_sha=file_digest(root/"p2_config.json"),
                complete={stage: file_digest(root/stage/"complete.json") for stage in ("quality", "matching")}))


def verify_diagnostics(root, s):
    d = load_json(root/"diagnostics.json")
    donor = Path(d["directory"])
    if (d["source_config_sha"] != s["source_config_sha"] or d["seal_sha"] != s["seal_sha"] or
            file_digest(donor/"p2_config.json") != d["config_sha"] or
            any(file_digest(donor/stage/"complete.json") != sha for stage, sha in d["complete"].items())):
        raise ValueError("Diagnostic provenance changed")


def verify_pack(root, fast):
    value = load_json(root/"packed/complete.json")
    if value["fingerprint"] != digest(fast):
        raise ValueError("Packed source fingerprint mismatch")
    for name, sha in value["files"].items():
        if file_digest(root/"packed"/name) != sha:
            raise ValueError("Packed source modified")
    return value


def pack(a):
    root, source, cfg, s, fast = context(a)
    if (root/"packed/complete.json").exists():
        verify_pack(root, fast)
        print("Resume: source tensor pack already complete", flush=True)
        return
    rows = [r for r in p2.old.records_for(source) if r["partition"] in ("train", "val")]
    p2.old.FeatureDataset(source, cfg, rows)
    p2.verify_reviews(source, cfg, rows)
    shape = load_json(source/"sealed.json")["shape"]
    size, tiles = cfg["image_size"], len(cfg["boxes"])
    specs = dict(visual=(np.float16, shape), base=(np.float32, [1, size, size]), mask=(np.uint8, [1, size, size]),
                 embeddings=(np.float32, [tiles, 2, shape[1]]), valid=(np.float32, [tiles, 2]), boxes=(np.float32, [tiles, 4]))
    folder = root/"packed"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.lib.format.open_memmap(folder/f"{key}.building.npy", mode="w+", dtype=dtype, shape=(len(rows), *dimensions))
              for key, (dtype, dimensions) in specs.items()}
    geometries = [None]*len(rows)
    cv2.setNumThreads(1)
    def read(i):
        r = rows[i]
        with np.load(source/"features"/f"{r['id']}.npz") as data:
            values = {k: data[k].astype(specs[k][0]) for k in specs if k != "mask"}
        values["mask"] = p2.old.mask_for(r, size).numpy().astype(np.uint8)
        g = geometry(values["mask"][0], s) if r["partition"] == "train" else {"background_pixels": int((values["mask"] == 0).sum())}
        return i, values, g
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.pack_workers) as pool:
        pending, next_i, completed = {}, 0, 0
        while next_i < len(rows) or pending:
            while next_i < len(rows) and len(pending) < 2*a.pack_workers:
                future = pool.submit(read, next_i)
                pending[future] = next_i
                next_i += 1
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                i, values, g = future.result()
                del pending[future]
                for key, value in values.items():
                    arrays[key][i] = value
                geometries[i] = g
                completed += 1
                if completed % 100 == 0 or completed == len(rows):
                    print(f"pack SOURCE only {completed}/{len(rows)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    for key in specs:
        arrays[key].flush()
    arrays.clear()
    gc.collect()
    for key in specs:
        (folder/f"{key}.building.npy").replace(folder/f"{key}.npy")
    p2.old.save_torch(folder/"geometry.pt", geometries)
    files = {name: file_digest(folder/name) for name in [*(f"{k}.npy" for k in specs), "geometry.pt"]}
    atomic_json(folder/"complete.json", dict(fingerprint=digest(fast), files=files, rows=rows,
                                            bytes=sum((folder/name).stat().st_size for name in files), source_only=True))
    print("Derived tensor pack complete; original sealed NPZ cache unchanged", flush=True)


def tensor_to(value, device):
    t = torch.from_numpy(np.array(value, copy=True)) if isinstance(value, np.ndarray) else value
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t


class Store:
    """Read-only mmap backing + automatically bounded CUDA resident prefix."""
    def __init__(self, root, cfg, s, device, a):
        self.root, self.device, self.s = root, device, s
        meta = load_json(root/"packed/complete.json")
        self.rows = meta["rows"]
        self.arrays = {key: np.load(root/"packed"/f"{key}.npy", mmap_mode="r")
                       for key in ("visual", "base", "mask", "embeddings", "valid", "boxes")}
        self.geometries = torch.load(root/"packed/geometry.pt", weights_only=True)
        self.resident, self.geometry_cache = {}, OrderedDict()
        self.geometry_bytes = 0
        budget = 0
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            budget = max(0, free-a.reserve_gb*2**30)*.85
            if a.gpu_cache_gb:
                budget = min(budget, a.gpu_cache_gb*2**30)
        used = 0
        for key in ("base", "mask", "embeddings", "valid", "boxes", "visual"):
            arr = self.arrays[key]
            item = arr[0].nbytes
            count = min(len(arr), int(max(0, budget-used)//item)) if budget else 0
            if count:
                resident = torch.empty((count, *arr.shape[1:]), dtype=torch.from_numpy(np.array(arr[0])).dtype, device=device)
                for start in range(0, count, 32):
                    resident[start:start+32].copy_(tensor_to(arr[start:start+32], device))
                self.resident[key] = resident
                used += count*item
        self.matching = {}
        shuffled = self.arrays["embeddings"].copy()
        groups, audit, donors = {}, [], []
        for i, r in enumerate(self.rows):
            groups.setdefault((r["partition"], r["dataset"], r["category"]), []).append(i)
        for key, indices in sorted(groups.items()):
            seed = int(digest([cfg["seed"], list(key), "P2_donor"])[:8], 16)
            changed, info, links = shuffled_semantics(self.arrays["embeddings"][indices], self.arrays["valid"][indices],
                                                     [self.rows[i]["id"] for i in indices], seed)
            shuffled[indices] = changed
            audit.extend(dict(partition=key[0], dataset=key[1], category=key[2], image_id=self.rows[i]["id"], **v) for i, v in zip(indices, info))
            donors.extend(dict(partition=key[0], dataset=key[1], category=key[2], **v) for v in links)
        self.shuffle_audit, self.donors = audit, donors
        for mode in ("real", "shuffled"):
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(self.rows), 8):
                    indices = np.arange(start, min(start+8, len(self.rows)))
                    embeddings = self.get("embeddings", indices) if mode == "real" else tensor_to(shuffled[indices], device)
                    match = spatial_matching(self.get("visual", indices).float(), embeddings,
                                             self.get("valid", indices), self.get("boxes", indices))
                    chunks.append(match.cpu())
            cached = torch.cat(chunks)
            size = cached.numel()*cached.element_size()
            if used+size <= budget:
                cached = cached.to(device)
                used += size
            self.matching[mode] = cached
        self.geometry_budget = min(512*2**20, max(0, budget-used))
        self.cache_gb = used/2**30
        self.train_indices = np.array([i for i, r in enumerate(self.rows) if r["partition"] == "train"])
        self.plans = {}
        for key, indices in sorted(groups.items()):
            if key[0] != "val":
                continue
            seed = int(digest([cfg["seed"], list(key)])[:8], 16)
            plan = ValidationPlan(self.arrays["mask"][indices, 0], seed, s["val_pixels"], s["fpr"])
            base_sample, base_gt = plan.gather(self.arrays["base"][indices, 0])
            self.plans[key] = dict(indices=indices, plan=plan, base_sample=base_sample, base_gt=base_gt)
        print(f"worker device={device} resident={self.cache_gb:.2f}GiB; matches precomputed; source plans ready", flush=True)

    def get(self, key, indices):
        indices = np.asarray(indices)
        resident = self.resident.get(key)
        if resident is not None and (indices < len(resident)).all():
            return resident.index_select(0, torch.tensor(indices, device=self.device))
        result = tensor_to(self.arrays[key][indices], self.device)
        if resident is not None:
            take = np.flatnonzero(indices < len(resident))
            if len(take):
                result[torch.tensor(take, device=self.device)] = resident.index_select(0, torch.tensor(indices[take], device=self.device))
        return result

    def batch(self, indices, mode, supervised=False):
        visual, base = self.get("visual", indices).float(), self.get("base", indices)
        if mode == "visual":
            match = torch.zeros(len(indices), 3*visual.shape[1]+2, *visual.shape[-2:], device=self.device)
        else:
            cached = self.matching[mode]
            match = cached.index_select(0, torch.tensor(indices, device=cached.device)).to(self.device, non_blocking=True)
        batch = dict(visual=visual, base=base, matching=match)
        if supervised:
            batch["mask"] = self.get("mask", indices).float()
        return batch

    def get_geometry(self, index):
        if index in self.geometry_cache:
            self.geometry_cache.move_to_end(index)
            return self.geometry_cache[index][0]
        original = self.geometries[index]
        value = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in original.items()}
        size = sum(v.numel()*v.element_size() for v in value.values() if isinstance(v, torch.Tensor))
        if self.device.type == "cuda" and size <= self.geometry_budget:
            while self.geometry_bytes+size > self.geometry_budget and self.geometry_cache:
                _, (_, removed) = self.geometry_cache.popitem(last=False)
                self.geometry_bytes -= removed
            self.geometry_cache[index] = (value, size)
            self.geometry_bytes += size
        return value


def autocast(device, fast):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda" and fast["precision"] == "bf16")


def baseline(store):
    values = [v["plan"].metrics(v["base_sample"], v["base_gt"], v["base_sample"]) for v in store.plans.values()]
    return dict(mean=aggregate_source(values), categories=values)


def validate(model, store, s, fast, micro):
    per_alpha = {str(x): [] for x in [0.]+s["alphas"]}
    mode = model.training_mode
    with torch.inference_mode():
        for info in store.plans.values():
            sample, gt = [], []
            indices, plan = info["indices"], info["plan"]
            for start in range(0, len(indices), micro):
                end = min(start+micro, len(indices))
                batch = store.batch(indices[start:end], mode)
                with autocast(store.device, fast):
                    delta = model.forward_cached(**batch).float().flatten()
                draw, positive = batch_plan_indices(plan, start, end, batch["base"][0].numel())
                sample.append(delta.index_select(0, draw.to(store.device)).cpu().numpy())
                gt.append(delta.index_select(0, positive.to(store.device)).cpu().numpy())
            sample, gt = np.concatenate(sample), np.concatenate(gt)
            for alpha in [0.]+s["alphas"]:
                values = []
                for b, d in ((info["base_sample"], sample), (info["base_gt"], gt)):
                    values.append(p2.repair.adjust_numpy(b, d, alpha))
                per_alpha[str(alpha)].append(plan.metrics(*values, info["base_sample"]))
    return {k: dict(mean=aggregate_source(v), categories=v) for k, v in per_alpha.items()}


def effective_step(model, optimizer, store, indices, mode, loss_mode, s, fast, micro):
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(7, dtype=torch.float64, device=store.device)
    for start in range(0, len(indices), micro):
        chosen = indices[start:start+micro]
        batch = store.batch(chosen, mode, supervised=True)
        gs = [store.get_geometry(int(i)) for i in chosen]
        with autocast(store.device, fast):
            delta = model.forward_cached(batch["visual"], batch["base"], batch["matching"]).float()
        loss, terms = fast_loss(batch["base"], delta, batch["mask"], gs, loss_mode, s)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite fast loss")
        (loss.sum()/len(indices)).backward()
        sums[0] += loss.detach().sum()
        for i, key in enumerate(TERMS, 1):
            sums[i] += terms[key].detach().sum()
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
    except torch.cuda.OutOfMemoryError as exc:
        # An optimizer step may have partially changed state: never retry it.
        raise RuntimeError("Optimizer OOM: stop and resume the last checkpoint with less GPU cache") from exc
    return sums


def train_head(a, store, loss_mode, mode):
    root, _, cfg, s, fast = context(a)
    torch.manual_seed(cfg["seed"])
    shape = store.arrays["visual"].shape[1:]
    model = CachedMatchingHead(shape[0], shape[1], s["hidden"]).to(store.device)
    model.training_mode = mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=1e-4)
    out = root/"heads"/loss_mode/mode
    fingerprint = digest([fast, loss_mode, mode])
    start, history = 0, []
    base = baseline(store)
    best_score = selection_score(base["mean"], base["mean"], base["categories"], base["categories"], s["auc_tolerance"], s["fpr_slack"])
    if (out/"last.pt").exists():
        last = torch.load(out/"last.pt", map_location=store.device, weights_only=False)
        if last["fingerprint"] != fingerprint:
            raise ValueError("Fast checkpoint mismatch")
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        start, history = last["epoch"]+1, last["history"]
        best_score = max(best_score, last["best_score"])
    if (out/"best.pt").exists():
        saved = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Selected fast checkpoint mismatch")
        best_score = max(best_score, saved["score"])
    else:
        p2.old.save_torch(out/"best.pt", dict(state_dict=model.state_dict(), alpha=0., epoch=-1, shape=shape,
                          fingerprint=fingerprint, selection="BASE_FALLBACK", source_validation=base, score=best_score))
    if mode == "shuffled":
        train_audit = [v for v in store.shuffle_audit if v["partition"] == "train"]
        p2.write_csv(out/"shuffle_audit.csv", train_audit)
        p2.write_csv(out/"donors.csv", [v for v in store.donors if v["partition"] == "train"],
                     ["partition", "dataset", "category", "image_id", "tile", "role", "donor_id", "donor_tile"])
        if sum(v["valid_slots"] for v in train_audit) and not sum(v["changed_slots"] for v in train_audit):
            raise ValueError("Shuffled control has no effective changes")
    until = min(a.until_epoch, s["epochs"])
    micro = min(a.micro_batch, s["batch_size"])
    if start >= until:
        print(f"Resume: {loss_mode}/{mode} complete through {until} epochs", flush=True)
        return
    for epoch in range(start, until):
        model.train()
        gen = torch.Generator().manual_seed(cfg["seed"]+epoch)
        order = store.train_indices[torch.randperm(len(store.train_indices), generator=gen).numpy()]
        sums = torch.zeros(7, dtype=torch.float64, device=store.device)
        if store.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(store.device)
        begin = time.perf_counter()
        for offset in range(0, len(order), s["batch_size"]):
            indices = order[offset:offset+s["batch_size"]]
            while True:
                try:
                    result = effective_step(model, optimizer, store, indices, mode, loss_mode, s, fast, micro)
                    sums += result
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro = max(1, micro//2)
                    print(f"OOM backoff {loss_mode}/{mode}: micro={micro}, effective batch remains {s['batch_size']}", flush=True)
        if store.device.type == "cuda":
            torch.cuda.synchronize(store.device)
        train_seconds = max(1e-9, time.perf_counter()-begin)
        model.eval()
        begin = time.perf_counter()
        metrics = validate(model, store, s, fast, micro)
        val_seconds = time.perf_counter()-begin
        decisions = []
        for alpha in s["alphas"]:
            candidate = metrics[str(alpha)]
            score = selection_score(candidate["mean"], base["mean"], candidate["categories"], base["categories"], s["auc_tolerance"], s["fpr_slack"])
            decisions.append(dict(alpha=alpha, eligible=score is not None, score=score))
            if score is not None and score > best_score+1e-8:
                best_score = score
                p2.old.save_torch(out/"best.pt", dict(state_dict=model.state_dict(), alpha=alpha, epoch=epoch, shape=shape,
                                  fingerprint=fingerprint, selection="SOURCE_VALIDATED", source_validation=candidate, score=score))
        count = len(order)
        history.append(dict(epoch=epoch+1, train_loss=float(sums[0]/count),
                            loss_terms={key: float(sums[i]/count) for i, key in enumerate(TERMS, 1)},
                            validation=metrics, decisions=decisions, best_score=best_score,
                            train_seconds=train_seconds, validation_seconds=val_seconds, images_per_second=count/train_seconds,
                            micro_batch=micro, effective_batch=s["batch_size"], resident_cache_gb=store.cache_gb,
                            peak_allocated_gb=torch.cuda.max_memory_allocated(store.device)/2**30 if store.device.type == "cuda" else 0))
        selected = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
        p2.old.save_torch(out/"last.pt", dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                          fingerprint=fingerprint, history=history, best_score=best_score, alpha=selected["alpha"]))
        atomic_json(out/"history.json", history)
        print(f"{store.device} {loss_mode}/{mode} epoch={epoch+1}/{s['epochs']} train={train_seconds:.1f}s val={val_seconds:.1f}s "
              f"img/s={count/train_seconds:.1f} cache={store.cache_gb:.2f}GiB peak={history[-1]['peak_allocated_gb']:.2f}GiB alpha={selected['alpha']}", flush=True)


def head_hashes(root):
    return {f"{loss}/{mode}": file_digest(root/"heads"/loss/mode/"best.pt") for loss, mode in JOBS}


def infer_batch(arrays, shuffled, start, end, device, models, fast):
    batch = {k: tensor_to(v[start:end], device).float() for k, v in arrays.items()}
    matching = {"real": spatial_matching(*(batch[k] for k in ("visual", "embeddings", "valid", "boxes")))}
    matching["shuffled"] = spatial_matching(batch["visual"], tensor_to(shuffled[start:end], device), batch["valid"], batch["boxes"])
    matching["visual"] = torch.zeros_like(matching["real"])
    output = {}
    for name, (model, saved) in models.items():
        if saved["alpha"] == 0:
            output[name] = arrays["base"][start:end, 0]
            continue
        with autocast(device, fast):
            delta = model.forward_cached(batch["visual"], batch["base"], matching[name.split("/")[1]]).float()
        pred = corrected_probability(batch["base"], delta, saved["alpha"])
        output[name] = pred[:, 0].cpu().numpy()
    return output


def evaluate_group(a, key, device, models, hashes):
    from tools.vlm_next_diagnostics import map_metrics
    root, source, cfg, s, fast = context(a)
    path = root/"results"/("_".join(key)+".json")
    if path.exists():
        saved = load_json(path)
        if saved["fingerprint"] == digest(fast) and saved["heads"] == hashes:
            print(f"Resume: evaluation {key} complete", flush=True)
            return
    rows = p2.repair.groups_for(source, {key[0]})[key]
    dataset = p2.old.FeatureDataset(source, cfg, rows)
    visual, base, embeddings, valid, boxes, masks = [], [], [], [], [], []
    for i in range(len(rows)):
        batch = dataset[i]
        visual.append(batch["visual"].numpy().astype(np.float16))
        base.append(batch["base"].numpy())
        embeddings.append(batch["embeddings"].numpy())
        valid.append(batch["valid"].numpy())
        boxes.append(batch["boxes"].numpy())
        masks.append(batch["mask"][0].numpy().astype(bool))
    arrays = {"visual": np.stack(visual), "base": np.stack(base), "embeddings": np.stack(embeddings),
              "valid": np.stack(valid), "boxes": np.stack(boxes)}
    masks = np.stack(masks)
    shuffled, _, _ = shuffled_semantics(arrays["embeddings"], arrays["valid"], [r["id"] for r in rows],
                        int(digest([cfg["seed"], list(key), "P2_donor"])[:8], 16))
    maps = {"base": arrays["base"][:, 0]}
    for name in models:
        maps[name] = np.empty_like(maps["base"])
    with torch.inference_mode():
        start, micro = 0, a.micro_batch
        while start < len(rows):
            end = min(start+micro, len(rows))
            retry = False
            try:
                predictions = infer_batch(arrays, shuffled, start, end, device, models, fast)
            except torch.cuda.OutOfMemoryError:
                if micro == 1:
                    raise
                micro = max(1, micro//2)
                retry = True
            if retry:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Evaluation OOM backoff {key}: micro={micro}; full dataset retained", flush=True)
                continue
            for name, pred in predictions.items():
                maps[name][start:end] = pred
            start = end
    plan = ValidationPlan(masks, int(digest([cfg["seed"], list(key)])[:8], 16), s["val_pixels"], s["fpr"])
    base_sample, _ = plan.gather(maps["base"])
    output = []
    for name, pred in maps.items():
        sample, gt = plan.gather(pred)
        operating = plan.metrics(sample, gt, base_sample)
        small, hit, recovered, lost = p2.repair.small_counts(masks, pred, maps["base"])
        checkpoint = models[name][1] if name in models else dict(alpha=0., epoch=-1, selection="REFERENCE")
        loss_mode, mode = name.split("/") if name != "base" else ("reference", "base")
        output.append(dict(partition=key[0], dataset=key[1], category=key[2], loss=loss_mode, training_mode=mode,
                           samples=len(rows), alpha=checkpoint["alpha"], epoch=checkpoint["epoch"]+1, selection=checkpoint["selection"],
                           **map_metrics(masks, pred), region_recall_at_fpr_DIAGNOSTIC=operating["region_recall_at_fpr"],
                           small_hit_at_fpr_DIAGNOSTIC=operating["small_hit_at_fpr"],
                           small_components=small, small_hits_at_05=hit, recovered_small_at_05=recovered, lost_small_at_05=lost,
                           background_FPR_at_05=float(np.mean(pred[~masks] >= .5)),
                           changed_fraction=float(np.mean(np.abs(pred-maps["base"]) > 1e-6))))
    atomic_json(path, dict(fingerprint=digest(fast), heads=hashes, rows=output, evaluation="best_only_full_dataset"))
    print(f"evaluated {key}; 6 source-selected best heads + Base; full dataset", flush=True)


def worker(a, index, tasks, events, stage):
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        torch.set_num_threads(a.cpu_threads)
        cv2.setNumThreads(1)
        device = torch.device("cpu") if a.cpu else p2.repair.device_for(index)
        root, _, cfg, s, fast = context(a)
        if device.type == "cuda" and fast["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is unsupported; use a new fp32 experiment")
        events.put(("startup", index, stage))
        if stage == "train":
            store = Store(root, cfg, s, device, a)
        else:
            models = {}
            hashes = head_hashes(root)
            shape = load_json(Path(s["source"])/"sealed.json")["shape"]
            for loss_mode, mode in JOBS:
                saved = torch.load(root/"heads"/loss_mode/mode/"best.pt", map_location=device, weights_only=False)
                if saved["fingerprint"] != digest([fast, loss_mode, mode]):
                    raise ValueError("Evaluation checkpoint mismatch")
                model = CachedMatchingHead(shape[0], shape[1], s["hidden"]).to(device).eval()
                model.load_state_dict(saved["state_dict"])
                models[f"{loss_mode}/{mode}"] = (model, saved)
        while True:
            task = tasks.get()
            if task is None:
                break
            if stage == "train":
                train_head(a, store, *task)
            else:
                evaluate_group(a, tuple(task), device, models, hashes)
            events.put(("complete", index, task))
    except BaseException:
        events.put(("error", index, traceback.format_exc()))
        raise


def run_pool(a, stage):
    root, source, _, s, fast = context(a)
    verify_diagnostics(root, s)
    if not 1 <= a.until_epoch <= s["epochs"]:
        raise ValueError("until_epoch must be within the prepared epoch plan")
    verify_pack(root, fast)
    jobs = list(JOBS) if stage == "train" else list(p2.repair.groups_for(source, {"val", "eval"}))
    ctx = mp.get_context("spawn")
    tasks, events = ctx.Queue(), ctx.Queue()
    for job in jobs:
        tasks.put(job)
    for _ in range(s["num_shards"]):
        tasks.put(None)
    processes = [ctx.Process(target=worker, args=(a, i, tasks, events, stage)) for i in range(s["num_shards"])]
    begin, last_update, completed = time.monotonic(), time.monotonic(), 0
    prior_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            try:
                kind, index, detail = events.get(timeout=1)
                if kind == "error":
                    raise RuntimeError(f"worker {index} failed:\n{detail}")
                if kind == "complete":
                    completed += 1
                    print(f"{stage} worker={index} complete={detail} jobs={completed}/{len(jobs)}", flush=True)
                    if stage == "train":
                        screen_report(a)
                if kind == "startup":
                    print(f"{stage} worker={index} starting", flush=True)
            except queue.Empty:
                pass
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError("GPU worker failed; stopping this run's other workers")
            if time.monotonic()-last_update >= 30:
                print(f"{stage}: elapsed={time.monotonic()-begin:.0f}s; jobs reported={completed}/{len(jobs)}", flush=True)
                last_update = time.monotonic()
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("Worker failed")
        if stage == "train":
            screen_report(a)
    finally:
        signal.signal(signal.SIGTERM, prior_handler)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
        tasks.close()
        events.close()


def screen_report(a):
    root, _, _, s, fast = context(a)
    rows = []
    for loss_mode, mode in JOBS:
        path = root/"heads"/loss_mode/mode/"last.pt"
        if not path.exists():
            continue
        last = torch.load(path, map_location="cpu", weights_only=False)
        selected = torch.load(path.with_name("best.pt"), map_location="cpu", weights_only=False)
        if last["fingerprint"] != digest([fast, loss_mode, mode]) or selected["fingerprint"] != last["fingerprint"]:
            raise ValueError("Source screen checkpoint mismatch")
        if selected["epoch"] > last["epoch"]:
            continue  # Another worker is between saving best and last.
        stats = selected["source_validation"]["mean"]
        rows.append(dict(loss=loss_mode, training_mode=mode, completed_epochs=last["epoch"]+1,
                         planned_epochs=s["epochs"], selected_epoch=selected["epoch"]+1, alpha=selected["alpha"],
                         selection=selected["selection"], **stats, **{k: last["history"][-1][k] for k in
                         ("images_per_second", "train_seconds", "validation_seconds", "micro_batch", "effective_batch", "resident_cache_gb", "peak_allocated_gb")}))
    if rows:
        p2.write_csv(root/"source_screen.csv", rows)
        atomic_json(root/"source_screen.json", rows)


def report(a):
    root, source, _, s, fast = context(a)
    expected = head_hashes(root)
    rows = []
    for key in p2.repair.groups_for(source, {"val", "eval"}):
        value = load_json(root/"results"/("_".join(key)+".json"))
        if value["fingerprint"] != digest(fast) or value["heads"] != expected:
            raise ValueError("Stale best evaluation; run evaluate again")
        rows.extend(value["rows"])
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in ("partition", "dataset", "loss", "training_mode")), []).append(row)
    sums = {"samples", "small_components", "small_hits_at_05", "recovered_small_at_05", "lost_small_at_05"}
    for _, values in sorted(groups.items()):
        mean = dict(values[0], category="MEAN")
        for k in mean:
            if k in ("partition", "dataset", "category", "loss", "training_mode", "alpha", "epoch", "selection"):
                continue
            vals = [r[k] for r in values if r[k] is not None]
            mean[k] = sum(vals) if k in sums else float(np.mean(vals)) if vals else None
        rows.append(mean)
    p2.write_csv(root/"results/metrics.csv", rows)
    atomic_json(root/"results/summary.json", rows)
    indexed = {(r["partition"], r["dataset"], r["category"], r["loss"], r["training_mode"]): r for r in rows}
    contrasts = []
    for r in rows:
        if r["training_mode"] == "base":
            continue
        key = r["partition"], r["dataset"], r["category"]
        controls = [("head_vs_base", indexed[(*key, "reference", "base")])]
        if r["training_mode"] == "real":
            controls.extend(("real_vs_"+m, indexed[(*key, r["loss"], m)]) for m in ("visual", "shuffled"))
        if r["loss"] == "p2b":
            controls.append(("p2b_vs_p2a", indexed[(*key, "p2a", r["training_mode"])]))
        for contrast, reference in controls:
            delta = {"delta_"+k+"_pp": 100*(r[k]-reference[k]) if r[k] is not None and reference[k] is not None else None
                     for k in ("PRO_exact", "P_AUROC", "background_FPR_at_05", "small_hit_at_fpr_DIAGNOSTIC")}
            contrasts.append(dict(partition=key[0], dataset=key[1], category=key[2], loss=r["loss"],
                                  training_mode=r["training_mode"], contrast=contrast, **delta))
    p2.write_csv(root/"results/comparisons.csv", contrasts)
    screen_report(a)
    progress = load_json(root/"source_screen.json")
    epochs = min(r["completed_epochs"] for r in progress)
    lines = ["# P2 加速实验", "", f"本报告六组均至少完成 {epochs}/{s['epochs']} 轮。精度协议：{fast['precision']}；有效 batch={s['batch_size']}。",
             "best 仅由完整源验证选择；目标全量评估，不使用目标标签选参。当前报告只评估 best，last 与推理干预省略。", "",
             "| partition/dataset | loss | mode | PRO % | small hit at matched FPR % (diagnostic) | alpha |",
             "|---|---|---|---:|---:|---:|"]
    for r in rows:
        if r["category"] == "MEAN":
            hit = "NA" if r["small_hit_at_fpr_DIAGNOSTIC"] is None else f"{100*r['small_hit_at_fpr_DIAGNOSTIC']:.4f}"
            lines.append(f"| {r['partition']}/{r['dataset']} | {r['loss']} | {r['training_mode']} | {100*r['PRO_exact']:.4f} | {hit} | {r['alpha']} |")
    (root/"results/README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Report: {root/'results/README.md'}; source timing/VRAM: {root/'source_screen.csv'}", flush=True)


def parser():
    p = p2.parser()
    p.description = __doc__
    p._option_string_actions["--stage"].choices = ("prepare", "diagnostics", "pack", "train", "evaluate", "report")
    p.set_defaults(batch_size=8, workers=0)
    p.add_argument("--diagnostic_work_dir", default="")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--cpu", action="store_true", help="Synthetic fixtures only")
    for key, default in dict(micro_batch=8, until_epoch=5, cpu_threads=4, pack_workers=4).items():
        p.add_argument("--"+key, type=int, default=default)
    for key, default in dict(reserve_gb=4., gpu_cache_gb=0.).items():
        p.add_argument("--"+key, type=float, default=default)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"prepare": prepare, "diagnostics": diagnostics, "pack": pack, "train": lambda a: run_pool(a, "train"),
     "evaluate": lambda a: run_pool(a, "evaluate"), "report": report}[args.stage](args)
