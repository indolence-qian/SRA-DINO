#!/usr/bin/env python3
"""Three matched text readouts on one frozen factorial-D visual residual model.

The first experiment is read-only. No DINO export, Flow retraining or target
adaptation occurs here. All text heads use source labels and source selection.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

import factorial_ablation_experiment as f
import residual_text_experiment as base
from tools.residual_text import fuse_views, positive_text_fusion, text_loss
from tools.residual_text_runtime import TEMPLATES, atomic_npz, encode_text_bank, read_torch, save_torch, unchanged
from tools.text_ablation import (EVALUATION, MODES, TextMatcher, bank_audit, corrected_metrics,
                                 evaluation_truth, validate_prompts)
from tools.vlm_review import atomic_json, digest, file_digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "frozen_visual_text_readouts_v1"
CODE = ("text_alignment_ablation.py", "tools/text_ablation.py", "run_exp_text_alignment_ablation.sh")
MAPS = ("V", *MODES, *(a+"_text" for a in MODES))
METRICS = f.METRICS
NAMES = {"V": "纯视觉共同参照", "T1": "单对文本差分", "T2": "普通正常/异常文本", "T3": "多对文本平均差分"}


def parent(cfg):
    root = Path(cfg["base_work_dir"])
    pcfg = f.config(root)
    if digest(pcfg) != cfg["base_fingerprint"]:
        raise ValueError("Parent configuration changed; use NEW WORK_DIR")
    for name, sha in cfg["parent_artifacts"].items():
        if file_digest(root/name) != sha:
            raise ValueError(f"Frozen parent artifact changed: {name}")
    return root, pcfg


def config(root):
    cfg = load_json(Path(root)/"config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong experiment; use NEW WORK_DIR")
    for name, sha in cfg["code"].items():
        if file_digest(REPO/name) != sha:
            raise ValueError(f"Code changed: {name}; use NEW WORK_DIR")
    if file_digest(cfg["prompt_file"]) != cfg["prompt_sha"]:
        raise ValueError("Prompt definitions changed; use NEW WORK_DIR")
    parent(cfg)
    return cfg


def horizon(args, cfg):
    if not 1 <= args.until_epoch <= cfg["settings"]["text_epochs"]:
        raise ValueError("Require 1 <= until_epoch <= planned text epochs")
    return args.until_epoch


def prepare(args):
    root, source = Path(args.work_dir).resolve(), Path(args.base_work_dir).resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Use separate non-nested text and base WORK_DIRs")
    if not (source/"config.json").is_file():
        raise FileNotFoundError("BASE_WORK_DIR must contain the completed factorial experiment, including residual features and D/Flow snapshots")
    pcfg = f.config(source)
    settings = dict(pcfg, text_epochs=args.epochs)
    for name in ("batch_size", "micro_batch", "workers", "text_lr"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if (not 1 <= args.until_epoch <= args.epochs or args.num_shards < 1 or args.base_train_seed < 0
            or min(settings["batch_size"], settings["micro_batch"]) < 1
            or settings["micro_batch"] > settings["batch_size"] or settings["workers"] < 0
            or not np.isfinite(settings["text_lr"]) or settings["text_lr"] <= 0
            or not 0 < args.text_strength <= 1):
        raise ValueError("Invalid text experiment settings")
    prompts = validate_prompts(load_json(args.prompt_file))
    visual = f.head_dir(source, args.base_train_seed, "D", "density")/f"through{args.base_until_epoch}.pt"
    _, meta, sha = f.load_snapshot(source, pcfg, "D", "density", args.base_train_seed, args.base_until_epoch, "cpu")
    tracked = ("config.json", "manifest.json", "references.json", "normal_stats.json", "sealed_source.json", "sealed_eval.json")
    artifacts = {name: file_digest(source/name) for name in tracked}
    artifacts[visual.relative_to(source).as_posix()] = sha
    cfg = dict(protocol=PROTOCOL, base_work_dir=str(source), base_fingerprint=digest(pcfg),
        parent_artifacts=artifacts, base_train_seed=args.base_train_seed, base_until_epoch=args.base_until_epoch,
        visual_sha=sha, visual_epoch=meta["epoch"], visual_calibration=meta["calibration"],
        settings=settings, prompts=prompts, templates=list(TEMPLATES),
        prompt_file=str(Path(args.prompt_file).resolve()), prompt_sha=file_digest(args.prompt_file),
        text_strength=args.text_strength, num_shards=args.num_shards, evaluation_protocol=EVALUATION,
        selection="source class-macro mean(AUPRO,P-AP), trained epochs only, earliest tie",
        code={name:file_digest(REPO/name) for name in CODE}, torch_version=str(torch.__version__))
    # No parent files (including its old result tables/locks) are written.
    f.immutable(root/"config.json", cfg)
    for name in tracked:
        f.immutable(root/"provenance"/name, load_json(source/name))
    print(f"Shared frozen D/Flow seed={args.base_train_seed}, epoch={meta['epoch']}, snapshot through{args.base_until_epoch}; "
          f"3 text heads, {len(prompts)} paired descriptions x {len(TEMPLATES)} templates", flush=True)
    return cfg


def directions(root, cfg):
    path = Path(root)/"text_bank.npz"
    meta = load_json(path.with_suffix(".json"))
    if meta["fingerprint"] != digest(cfg) or meta["sha"] != file_digest(path):
        raise ValueError("Text bank changed")
    with np.load(path, allow_pickle=False) as data:
        states = data["paired_states"].copy()
    expected = (len(cfg["prompts"]), 2, len(TEMPLATES))
    if states.shape[:3] != expected:
        raise ValueError("Text bank shape mismatch")
    bank_audit(states)
    return states


def text_bank(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    path = root/"text_bank.npz"
    if path.exists() and path.with_suffix(".json").exists():
        directions(root, cfg); return
    _, pcfg = parent(cfg)
    vectors, states = encode_text_bank(pcfg, cfg["prompts"], base.device_for(args, pcfg))
    atomic_npz(path, directions=vectors, paired_states=states)
    f.immutable(path.with_suffix(".json"), dict(fingerprint=digest(cfg), sha=file_digest(path),
                                              geometry=bank_audit(states)))


def new_model(root, cfg, arm, device):
    p = cfg["settings"]
    return TextMatcher(len(p["layers"]), p["feature_dim"], directions(root, cfg), MODES[arm], p["hidden"]).to(device)


def head_dir(root, seed, arm):
    return Path(root)/"heads"/f"seed{seed}"/arm


def training_identity(root, cfg, seed, arm):
    return dict(fingerprint=digest(cfg), text_bank=file_digest(Path(root)/"text_bank.npz"), seed=seed, arm=arm)


def datasets(cfg, labels=True):
    source, pcfg = parent(cfg)
    ids = set(pcfg["calibration_ids"])
    return tuple(f.Features(source, pcfg, records, "D", labels) for records in
        ([r for r in f.rows(source, "train") if r["id"] not in ids],
         f.rows(source, "val"), f.calibration_rows(source, pcfg)))


def train(args):
    root, cfg = Path(args.work_dir), config(args.work_dir)
    until = horizon(args, cfg); p = cfg["settings"]
    source, pcfg = parent(cfg); device = base.device_for(args, pcfg)
    seed, arm = args.train_seed, args.arm
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = new_model(root, cfg, arm, device)
    data, val, cal = datasets(cfg)
    scale = f.scale_for(source, pcfg, "D", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["text_lr"], weight_decay=.01)
    steps = int(np.ceil(len(data)/p["batch_size"]))*p["text_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:.05+.95*(1+np.cos(np.pi*min(s,steps)/steps))/2)
    out = head_dir(root, seed, arm); identity = training_identity(root, cfg, seed, arm)
    start, history, best = 0, [], None
    if (out/"last.pt").exists():
        saved = read_torch(out/"last.pt", device)
        if saved["identity"] != identity or saved["best_sha"] != file_digest(out/"best.pt"):
            raise ValueError("Training resume mismatch")
        model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, history, best = saved["epoch"], saved["history"], read_torch(out/"best.pt")
        if start >= until:
            load_snapshot(root, cfg, arm, seed, until, "cpu"); return
    for epoch in range(start, until):
        torch.manual_seed(seed+100003*(epoch+1)); model.train()
        loader = torch.utils.data.DataLoader(data, batch_size=p["batch_size"], shuffle=True,
            generator=torch.Generator().manual_seed(p["data_seed"]+epoch), num_workers=p["workers"])
        total, samples = 0., 0
        for batch in loader:
            count = len(batch["residual"]); optimizer.zero_grad(set_to_none=True)
            for offset in range(0, count, p["micro_batch"]):
                part = {k:v[offset:offset+p["micro_batch"]].to(device) for k,v in batch.items() if k != "index"}
                x = base.scaled_residual(part, scale, "HR")
                # T2 uses binary-softmax log odds. BCE(4*(sA-sN)-2,y) is
                # exactly soft-target CE([4*sN,4*sA-2],y); same loss budget.
                loss = text_loss(model, x, part["coverage"], part["typed"], p["high_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite text loss")
                (loss*len(x)/count).backward(); total += float(loss.detach())*len(x)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); samples += count
        calibration = base.fit_calibration(model, "text", cal, scale, "HR", p, device)
        point = base.validate_scorer(model, "text", val, scale, "HR", calibration, p, device)
        if best is None or f.quality(point) > f.quality(best["point"])+1e-12:
            save_torch(out/"best.pt", dict(identity=identity, state_dict=model.state_dict(), epoch=epoch+1,
                                           calibration=calibration, point=point))
            best = read_torch(out/"best.pt")
        history.append(dict(epoch=epoch+1, loss=total/max(1,samples), point=point, selected_epoch=best["epoch"]))
        save_torch(out/"last.pt", dict(identity=identity, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch+1, history=history, best_sha=file_digest(out/"best.pt")))
        atomic_json(out/"history.json", dict(identity=identity, history=history))
        print(f"{arm} epoch={epoch+1}/{p['text_epochs']} loss={history[-1]['loss']:.5f} selected={best['epoch']}", flush=True)
    snapshot = out/f"through{until}.pt"
    if snapshot.exists() and file_digest(snapshot) != file_digest(out/"best.pt"):
        raise ValueError("Immutable snapshot mismatch")
    if not snapshot.exists():
        snapshot.write_bytes((out/"best.pt").read_bytes())


def load_snapshot(root, cfg, arm, seed, until, device):
    path = head_dir(root, seed, arm)/f"through{until}.pt"
    meta = read_torch(path, device)
    if meta["identity"] != training_identity(root, cfg, seed, arm) or not 1 <= meta["epoch"] <= until:
        raise ValueError("Wrong/untrained selected text head")
    model = new_model(root, cfg, arm, device); model.load_state_dict(meta["state_dict"]); model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, meta, file_digest(path)


def decision_path(root, seed, until):
    return Path(root)/"decisions"/f"seed{seed}"/f"through{until}"/"selection.json"


def visual_model(cfg, device):
    source, pcfg = parent(cfg)
    return f.load_snapshot(source, pcfg, "D", "density", cfg["base_train_seed"], cfg["base_until_epoch"], device)[0]


def score_maps(visual, texts, state, batch, scale, cfg):
    p = cfg["settings"]
    v = base.calibrated_map(visual, "density", batch, scale, "HR", cfg["visual_calibration"])
    result = {"V":fuse_views(v, batch["boxes"], p["eval_size"], True, p["high_weight"])[0,0].cpu().numpy()}
    for arm, model in texts.items():
        t = base.calibrated_map(model, "text", batch, scale, "HR", state[arm]["calibration"])
        full = positive_text_fusion(v, t, cfg["text_strength"])
        for name, values in ((arm,full), (arm+"_text",t)):
            result[name] = fuse_views(values, batch["boxes"], p["eval_size"], True, p["high_weight"])[0,0].cpu().numpy()
    return result


@torch.inference_mode()
def select(args):
    root, cfg = Path(args.work_dir), config(args.work_dir); until = horizon(args,cfg)
    source, pcfg = parent(cfg); device = base.device_for(args,pcfg)
    texts, state = {}, {}
    for arm in MODES:
        texts[arm], meta, sha = load_snapshot(root,cfg,arm,args.train_seed,until,device)
        state[arm] = dict(sha=sha, epoch=meta["epoch"], calibration=meta["calibration"], source_point=meta["point"])
    visual = visual_model(cfg,device); scale = f.scale_for(source,pcfg,"D",device)
    background = {name:[] for name in MAPS}
    for item in f.Features(source,pcfg,f.calibration_rows(source,pcfg),"D",False):
        values = score_maps(visual,texts,state,base.batch_to(item,device),scale,cfg)
        for name, value in values.items():
            background[name].append(value.ravel()[::max(1,value.size//8192)])
    f.immutable(decision_path(root,args.train_seed,until),dict(fingerprint=digest(cfg),seed=args.train_seed,
        horizon=until,visual_sha=cfg["visual_sha"],text=state,thresholds={
            k:float(np.quantile(np.concatenate(v),1-pcfg["val_fpr"],method="higher")) for k,v in background.items()}))


def prediction_dir(root, seed, until):
    return Path(root)/"predictions"/f"seed{seed}"/f"through{until}"


def prediction_identity(root,cfg,seed,until):
    return dict(fingerprint=digest(cfg),selection=file_digest(decision_path(root,seed,until)),seed=seed,horizon=until)


def decision(root,cfg,seed,until):
    value = load_json(decision_path(root,seed,until))
    if (value["fingerprint"] != digest(cfg) or value["seed"] != seed or value["horizon"] != until
            or value["visual_sha"] != cfg["visual_sha"] or set(value["text"]) != set(MODES)):
        raise ValueError("Stale/wrong source selection")
    for arm in MODES:
        if file_digest(head_dir(root,seed,arm)/f"through{until}.pt") != value["text"][arm]["sha"]:
            raise ValueError("Selected text head changed")
    return value


def increment_diagnostics(maps):
    result = {}
    for arm in MODES:
        increment = maps[arm]-maps["V"]
        result[arm] = dict(positive_fraction=float((increment>1e-6).mean()),
            mean=float(increment.mean()), p95=float(np.quantile(increment,.95)), maximum=float(increment.max()))
    return result


def valid_shard(args,cfg):
    if not 0 <= args.shard_id < cfg["num_shards"]:
        raise ValueError("Invalid shard")


@torch.inference_mode()
def predict(args):
    root, cfg = Path(args.work_dir), config(args.work_dir); until = horizon(args,cfg); valid_shard(args,cfg)
    source, pcfg = parent(cfg); device = base.device_for(args,pcfg)
    selected = decision(root,cfg,args.train_seed,until)
    visual = visual_model(cfg,device)
    texts = {a:load_snapshot(root,cfg,a,args.train_seed,until,device)[0] for a in MODES}
    scale = f.scale_for(source,pcfg,"D",device); identity = prediction_identity(root,cfg,args.train_seed,until)
    out = prediction_dir(root,args.train_seed,until)
    for i,(key,queries) in enumerate(f.groups(f.rows(source,"eval")).items()):
        if i%cfg["num_shards"] != args.shard_id: continue
        pending = []
        for row in queries:
            path = out/(row["id"]+".npz")
            if path.exists() and path.with_suffix(".json").exists():
                meta = load_json(path.with_suffix(".json"))
                if meta["identity"] != identity or meta["sha"] != file_digest(path):
                    raise ValueError("Prediction changed")
            else:
                if (out/"locked.json").exists(): raise ValueError("Cannot regenerate locked predictions")
                pending.append(row)
        for row,item in zip(pending,f.Features(source,pcfg,pending,"D",False)):
            maps = score_maps(visual,texts,selected["text"],base.batch_to(item,device),scale,cfg)
            path = out/(row["id"]+".npz"); atomic_npz(path,**maps)
            f.immutable(path.with_suffix(".json"),dict(identity=identity,sha=file_digest(path),increments=increment_diagnostics(maps)))
        print(f"predicted {key} text seed={args.train_seed}",flush=True)


def lock_predictions(args):
    root,cfg = Path(args.work_dir),config(args.work_dir); until = horizon(args,cfg)
    source,_ = parent(cfg); out = prediction_dir(root,args.train_seed,until)
    decision(root,cfg,args.train_seed,until)
    identity = prediction_identity(root,cfg,args.train_seed,until); hashes = {}
    for row in f.rows(source,"eval"):
        path = out/(row["id"]+".npz"); meta = load_json(path.with_suffix(".json"))
        if meta["identity"] != identity or meta["sha"] != file_digest(path):
            raise ValueError("Invalid prediction before lock")
        with np.load(path,allow_pickle=False) as data:
            if set(data.files) != set(MAPS) or any(data[n].shape != (cfg["settings"]["eval_size"],)*2
                    or not np.isfinite(data[n]).all() for n in MAPS):
                raise ValueError("Invalid prediction arrays")
        hashes[row["id"]] = dict(sha=file_digest(path),metadata=file_digest(path.with_suffix(".json")))
    f.immutable(out/"locked.json",dict(identity=identity,predictions=hashes))


def panels(root,cfg,queries,out,seed,until,truth):
    size = min(256,cfg["settings"]["eval_size"])
    masks = {r["id"]:m for r,m in zip(queries,truth)}
    for row in sorted(queries,key=lambda r:digest([cfg["settings"]["data_seed"],"text_panel",r["id"]]))[:2]:
        with Image.open(row["image_path"]) as image: original = image.convert("RGB").resize((size,size))
        gt = Image.fromarray((masks[row["id"]]*255).astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST).convert("RGB")
        tiles = [("query",original),("GT (eval only)",gt)]
        with np.load(out/(row["id"]+".npz"),allow_pickle=False) as data:
            for name in ("V",*MODES):
                t = torch.from_numpy(data[name])[None,None]
                p = torch.nn.functional.interpolate(t,(size,size),mode="bilinear",align_corners=False)[0,0].sigmoid().numpy()
                color = np.stack([p,.15*(1-p),1-p],-1)*255
                tiles.append((name,Image.fromarray((.55*np.asarray(original)+.45*color).clip(0,255).astype(np.uint8))))
        canvas = Image.new("RGB",(3*size,2*(size+24)),"white"); draw = ImageDraw.Draw(canvas)
        for i,(name,tile) in enumerate(tiles):
            x,y = i%3*size,i//3*(size+24); draw.text((x+4,y+4),name,fill="black"); canvas.paste(tile,(x,y+24))
        path = Path(root)/"panels"/f"seed{seed}"/f"through{until}"/(row["dataset"]+"__"+row["category"])/(row["id"]+".png")
        path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def evaluate(args):
    root,cfg = Path(args.work_dir),config(args.work_dir); until = horizon(args,cfg); valid_shard(args,cfg)
    source,pcfg = parent(cfg); out = prediction_dir(root,args.train_seed,until)
    lock = load_json(out/"locked.json")
    if lock["identity"] != prediction_identity(root,cfg,args.train_seed,until):
        raise ValueError("Stale prediction lock")
    selected = decision(root,cfg,args.train_seed,until)
    for i,(key,queries) in enumerate(f.groups(f.rows(source,"eval")).items()):
        if i%cfg["num_shards"] != args.shard_id: continue
        paths = [out/(r["id"]+".npz") for r in queries]
        for row,path in zip(queries,paths):
            unchanged(row)
            if lock["predictions"][row["id"]] != dict(sha=file_digest(path),metadata=file_digest(path.with_suffix(".json"))):
                raise ValueError("Prediction changed after lock")
        # First target mask content access, after every target prediction locked.
        pairs = [evaluation_truth(r,pcfg["eval_size"]) for r in queries]
        truth = np.stack([p[0] for p in pairs]); labels = np.array([bool(r["mask_path"]) for r in queries])
        mask_hashes = {r["id"]:file_digest(r["mask_path"]) if r["mask_path"] else None for r in queries}
        audit = dict(protocol=EVALUATION,mask_hashes=mask_hashes,images=[p[1] for p in pairs])
        f.immutable(root/"mask_audit"/("__".join(key)+".json"),audit)
        if any(a["image_label"] and not a["native_positive"] for a in audit["images"]):
            raise ValueError(f"Empty original anomaly mask in {key}; inspect mask_audit before reporting pixel metrics")
        records = []
        for name in MAPS:
            values = []
            for path in paths:
                with np.load(path,allow_pickle=False) as data: values.append(data[name].copy())
            maps = np.stack(values)
            point = corrected_metrics(truth,maps,labels,selected["thresholds"][name])
            arm = name.split("_")[0]
            point.update(dataset=key[0],category=key[1],arm=arm,score="text" if name.endswith("_text") else "full",
                seed=args.train_seed,text_epoch=selected["text"][arm]["epoch"] if arm in MODES else None,
                visual_sha=cfg["visual_sha"],visual_epoch=cfg["visual_epoch"],strength=cfg["text_strength"] if name in MODES else 0.)
            if name in MODES:
                inc = [load_json(p.with_suffix(".json"))["increments"][name] for p in paths]
                point["increment_positive_fraction"] = float(np.mean([v["positive_fraction"] for v in inc]))
                point["mean_increment"] = float(np.mean([v["mean"] for v in inc]))
            records.append(point)
        f.immutable(root/"metrics"/f"seed{args.train_seed}"/f"through{until}"/("__".join(key)+".json"),
            dict(prediction_lock_sha=file_digest(out/"locked.json"),evaluation_mask_hashes=mask_hashes,metrics=records))
        panels(root,cfg,queries,out,args.train_seed,until,truth)
        print(f"evaluated {key}, dataset anomaly count={labels.sum()} (independent of resized masks)",flush=True)


def summarize(records,keys,metrics=METRICS):
    grouped = {}
    for row in records: grouped.setdefault(tuple(row[k] for k in keys),[]).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        point = dict(zip(keys,key))
        for metric in metrics:
            numbers = [r[metric] for r in values if r.get(metric) is not None]
            point[metric] = float(np.mean(numbers)) if numbers else None
            point[metric+"_std"] = float(np.std(numbers)) if numbers else None
        result.append(point)
    return result


def table(rows, category=False):
    lines = ["| 数据集"+(" / 类别" if category else "")+" | 组别 | 评分 | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO | F1-best* | 背景FPR↓ | 像素召回 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        cells = [r["dataset"]+(" / "+r["category"] if category else ""),r["arm"],r["score"]]
        for m in METRICS:
            value,std = r.get(m),r.get(m+"_std",0)
            cells.append("—" if value is None else f"{value*100:.2f}"+(f" ± {std*100:.2f}" if std and not category else ""))
        lines.append("| "+" | ".join(cells)+" |")
    return lines


def report(args):
    root,cfg = Path(args.work_dir),config(args.work_dir); until = horizon(args,cfg)
    source,_ = parent(cfg); seeds = [int(v) for v in args.train_seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds): raise ValueError("Invalid report seeds")
    records, selections = [], []
    expected_groups = f.groups(f.rows(source,"eval"))
    for seed in seeds:
        out = prediction_dir(root,seed,until); lock = load_json(out/"locked.json")
        if lock["identity"] != prediction_identity(root,cfg,seed,until): raise ValueError("Stale prediction lock")
        chosen = decision(root,cfg,seed,until)
        for arm,s in chosen["text"].items():
            selections.append(dict(seed=seed,arm=arm,epoch=s["epoch"],**s["source_point"]))
        for key,queries in expected_groups.items():
            value = load_json(root/"metrics"/f"seed{seed}"/f"through{until}"/("__".join(key)+".json"))
            expected = {r["id"]:file_digest(r["mask_path"]) if r["mask_path"] else None for r in queries}
            if value["prediction_lock_sha"] != file_digest(out/"locked.json") or value["evaluation_mask_hashes"] != expected:
                raise ValueError("Stale evaluation or changed masks")
            for row in queries:
                path = out/(row["id"]+".npz")
                if lock["predictions"][row["id"]] != dict(sha=file_digest(path),metadata=file_digest(path.with_suffix(".json"))):
                    raise ValueError("Predictions changed after evaluation")
            points = value["metrics"]
            actual = {r["arm"]+("_text" if r["score"] == "text" else "") for r in points}
            if (len(points) != len(MAPS) or actual != set(MAPS) or any(r["seed"] != seed
                    or (r["dataset"],r["category"]) != key or r["visual_sha"] != cfg["visual_sha"]
                    or r["evaluation_protocol"] != EVALUATION or r["images"] != len(queries)
                    or r["abnormal_images"] != sum(bool(q["mask_path"]) for q in queries) for r in points)):
                raise ValueError("Missing/wrong arms, visual model, evaluation protocol or image labels")
            records.extend(points)
    all_metrics = METRICS+("increment_positive_fraction","mean_increment")
    per_seed = summarize(records,("dataset","arm","score","seed"),all_metrics)
    summary = summarize(per_seed,("dataset","arm","score"),all_metrics)
    category = summarize(records,("dataset","category","arm","score"),all_metrics)
    paired = []
    for dataset in sorted({r["dataset"] for r in per_seed}):
        for seed in seeds:
            lookup = {r["arm"]:r for r in per_seed if r["dataset"] == dataset and r["seed"] == seed and r["score"] == "full"}
            for name,a,b in (("差分减普通_T1-T2","T1","T2"),("多描述减单描述_T3-T1","T3","T1"),
                             *((arm+"减纯视觉",arm,"V") for arm in MODES)):
                paired.append(dict(dataset=dataset,comparison=name,seed=seed,**{m+"_pp":
                    100*(lookup[a][m]-lookup[b][m]) if lookup[a][m] is not None and lookup[b][m] is not None else None for m in METRICS}))
    contrasts = summarize(paired,("dataset","comparison"),tuple(m+"_pp" for m in METRICS))
    target = root/"reports"/f"through{until}"
    for name,values in (("metrics",records),("summary",summary),("per_category",category),("per_seed",per_seed),
                        ("contrasts",contrasts),("paired_contrasts",paired),("selected_epochs",selections)):
        base.write_csv(target/(name+".csv"),values)
    lines = ["# 文本对比实验结果","",f"协议：{PROTOCOL}；文本训练至{until}/{cfg['settings']['text_epochs']}轮；文本种子：{seeds}。",
        f"所有组共用原D组Flow：种子{cfg['base_train_seed']}、through{cfg['base_until_epoch']}选中第{cfg['visual_epoch']}轮；λ={cfg['text_strength']}。",
        f"T1单对差分；T2同一对普通状态文本；T3为{len(cfg['prompts'])}对差分匹配分数平均；每对均用{len(TEMPLATES)}个模板。",
        "V为共同纯视觉参照；text为未融合的文本单支路诊断，不是额外的训练组。",
        "百分数，先类别宏平均再跨文本种子平均；种子标准差不包含视觉主干重训的不确定性。",
        f"评价版本：`{EVALUATION}`。图像标签独立于mask；非零标注、下采样max-pool保留小缺陷。",
        "本轮V也重新评价，不能与上一轮旧GT口径的表格直接相减；原生异常mask为空会停止报告。",
        "F1-best*使用目标最优阈值仅作诊断；背景FPR/召回使用源CAL固定阈值。","",
        "## 完整模型与共同视觉参照","",*table([r for r in summary if r['score']=='full']),"",
        "## 文本单支路诊断","",*table([r for r in summary if r['score']=='text']),"",
        "## 配对增益（百分点）","","| 数据集 | 对照 | I-AUROC | P-AP | AUPRO |","|---|---|---:|---:|---:|"]
    for r in contrasts:
        lines.append("| "+" | ".join([r['dataset'],r['comparison']]+["—" if r[m+'_pp'] is None
            else f"{r[m+'_pp']:+.2f}"+(f" ± {r[m+'_pp_std']:.2f}" if r[m+'_pp_std'] else "") for m in ('I_AUROC','P_AP','PRO_exact')])+" |")
    lines += ["","## 文本正增量覆盖","","| 数据集 | 组别 | 正增量像素比例(%) | 平均分数增量 |","|---|---|---:|---:|"]
    for r in summary:
        if r['score']=='full' and r['arm'] in MODES:
            lines.append(f"| {r['dataset']} | {r['arm']} | {100*r['increment_positive_fraction']:.2f} | {r['mean_increment']:.4f} |")
    lines += ["","## 源验证选择轮次","","| 文本种子 | 组别 | 轮次 | 源P-AP | 源AUPRO |","|---|---|---:|---:|---:|"]
    for r in selections:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['epoch']} | {100*r['P_AP']:.2f} | {100*r['PRO_exact']:.2f} |")
    lines += ["","## 逐类别结果","",*table(category,True),"",
        "T2的两状态logit差与T1方向只差固定正尺度；同一投影器下严格等价。训练后的差异应结合尺度、优化和校准分析，不能直接当作独立表示能力证明。",
        "T3的均值也可合并成一个平均方向；本实验检验提示词集成，不证明细粒度缺陷分类能力。"]
    text = "\n".join(lines)+"\n"; (target/"对比实验.md").write_text(text,encoding="utf-8")
    if args.publish_report: f.publish_markdown(REPO/"对比实验.md",text)
    print(f"Results: {target/'对比实验.md'}",flush=True)


def package(args):
    root,cfg = Path(args.work_dir),config(args.work_dir); until = horizon(args,cfg)
    path = root/("text_comparison_analysis_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+".zip")
    candidates = [root/"config.json",root/"text_bank.json"]
    for folder in ("provenance","heads","decisions","metrics","reports","panels","mask_audit"):
        candidates.extend(p for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".json",".csv",".md",".png"))
    candidates.extend((root/"predictions").glob(f"seed*/through{until}/locked.json")); candidates.extend(root.glob("run_*.log"))
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(set(candidates)):
            if item.is_file(): archive.write(item,item.relative_to(root).as_posix())
    print(f"Analysis ZIP (no large features/checkpoints/prediction tensors): {path}",flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=("prepare","text_bank","train","select","predict","lock_predictions","evaluate","report","package"))
    p.add_argument("--work_dir",default="./checkpoint/text_comparison_v1")
    p.add_argument("--base_work_dir",default="./checkpoint/factorial_hr_residual_v1")
    p.add_argument("--prompt_file",default=str(REPO/"configs/text_ablation_prompts_v1.json"))
    p.add_argument("--base_train_seed",type=int,default=42); p.add_argument("--base_until_epoch",type=int,default=5)
    p.add_argument("--epochs",type=int,default=15); p.add_argument("--until_epoch",type=int,default=5)
    p.add_argument("--text_strength",type=float,default=.5); p.add_argument("--text_lr",type=float)
    p.add_argument("--batch_size",type=int); p.add_argument("--micro_batch",type=int); p.add_argument("--workers",type=int)
    p.add_argument("--train_seed",type=int,default=42); p.add_argument("--train_seeds",default="42")
    p.add_argument("--num_shards",type=int,default=2); p.add_argument("--shard_id",type=int,default=0)
    p.add_argument("--arm",choices=tuple(MODES),default="T1"); p.add_argument("--device",default="cuda:0")
    p.add_argument("--publish_report",action="store_true")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    globals()[arguments.stage](arguments)
