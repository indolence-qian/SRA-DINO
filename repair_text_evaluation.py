#!/usr/bin/env python3
"""Versioned, CPU-only reevaluation of the original locked text predictions.

No training code, masks, checkpoints, source decisions or old outputs are changed.
Empty decoded anomaly annotations remain anomalous for image metrics. Pixel
metrics are unavailable on the full set; valid-mask subset results are separate.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

import text_alignment_ablation as app
from tools.vlm_review import atomic_json, digest, file_digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "locked_text_reevaluation_empty_mask_audit_v2"
CODE = ("repair_text_evaluation.py", "run_repair_text_evaluation.sh")
PIXEL = tuple(m for m in app.METRICS if m not in ("I_AUROC", "I_AP"))
EXTRA = ("increment_positive_fraction", "mean_increment")
SCOPES = ("full_set", "valid_masks_diagnostic")


def source_context(root, cfg):
    text_root = Path(cfg["text_work_dir"])
    text_cfg = app.config(text_root)  # Keep original code/artifact checks intact.
    if file_digest(text_root/"config.json") != cfg["source_config_sha"]:
        raise ValueError("Original text configuration changed")
    source, _ = app.parent(text_cfg)
    for seed in cfg["seeds"]:
        out = app.prediction_dir(text_root, seed, cfg["until_epoch"])
        pinned = cfg["source_runs"][str(seed)]
        if (file_digest(out/"locked.json") != pinned["lock_sha"]
                or file_digest(app.decision_path(text_root, seed, cfg["until_epoch"])) != pinned["selection_sha"]):
            raise ValueError("Original prediction lock/source decision changed")
        app.decision(text_root, text_cfg, seed, cfg["until_epoch"])
    return text_root, text_cfg, source


def config(root):
    cfg = load_json(Path(root)/"config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong reevaluation directory")
    if any(file_digest(REPO/n) != sha for n,sha in cfg["code"].items()):
        raise ValueError("Reevaluation code changed; use a new REEVAL_DIR")
    source_context(root, cfg)
    return cfg


def verify_prediction(path, row, lock):
    meta = load_json(path.with_suffix(".json"))
    sha = file_digest(path)
    if (lock["predictions"][row["id"]] != dict(sha=sha, metadata=file_digest(path.with_suffix(".json")))
            or meta["identity"] != lock["identity"] or meta["sha"] != sha):
        raise ValueError(f"Prediction changed after lock: {row['id']}")
    return meta


def verify_all_predictions(root,cfg):
    text_root,_,source = source_context(root,cfg)
    queries = app.f.rows(source,"eval")
    for seed in cfg["seeds"]:
        out = app.prediction_dir(text_root,seed,cfg["until_epoch"])
        lock = load_json(out/"locked.json")
        if set(lock["predictions"]) != {r["id"] for r in queries}:
            raise ValueError("Incomplete prediction lock")
        for row in queries: verify_prediction(out/(row["id"]+".npz"),row,lock)


def prepare(args):
    root, text_root = Path(args.reeval_dir).resolve(), Path(args.work_dir).resolve()
    if root == text_root or root in text_root.parents or text_root in root.parents:
        raise ValueError("Use separate, non-nested WORK_DIR and REEVAL_DIR")
    text_cfg = app.config(text_root)
    if not 1 <= args.until_epoch <= text_cfg["settings"]["text_epochs"] or args.workers < 1:
        raise ValueError("Invalid horizon/workers")
    seeds = sorted(int(v) for v in args.train_seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Need unique nonnegative training seeds")
    source, _ = app.parent(text_cfg)
    queries = app.f.rows(source,"eval")
    source_runs = {}
    # Verify ALL selected seeds and ALL locked predictions before ANY target GT
    # content is read, including during audit-only operation.
    for seed in seeds:
        app.decision(text_root,text_cfg,seed,args.until_epoch)
        out = app.prediction_dir(text_root,seed,args.until_epoch)
        lock = load_json(out/"locked.json")
        if (lock["identity"] != app.prediction_identity(text_root,text_cfg,seed,args.until_epoch)
                or set(lock["predictions"]) != {r["id"] for r in queries}):
            raise ValueError("Missing/incomplete/stale prediction lock; finish original prediction stage first")
        for row in queries:
            verify_prediction(out/(row["id"]+".npz"),row,lock)
        source_runs[str(seed)] = dict(lock_sha=file_digest(out/"locked.json"),
            selection_sha=file_digest(app.decision_path(text_root,seed,args.until_epoch)))
    cfg = dict(protocol=PROTOCOL,text_work_dir=str(text_root),until_epoch=args.until_epoch,seeds=seeds,
        workers=args.workers,source_config_sha=file_digest(text_root/"config.json"),source_runs=source_runs,
        decoder_protocol=app.EVALUATION,
        empty_mask_policy="image_label_retained;full_pixel_unavailable;separate_valid_mask_diagnostic",
        code={n:file_digest(REPO/n) for n in CODE})
    app.f.immutable(root/"config.json",cfg)
    for name,path in (("text_config.json",text_root/"config.json"),
                      ("manifest.json",source/"manifest.json"),("references.json",source/"references.json"),
                      ("normal_stats.json",source/"normal_stats.json"),("text_bank.json",text_root/"text_bank.json")):
        app.f.immutable(root/"provenance"/name,load_json(path))
    for seed in seeds:
        app.f.immutable(root/"provenance"/f"seed{seed}"/"selection.json",load_json(app.decision_path(text_root,seed,args.until_epoch)))
        for arm in app.MODES:
            path = app.head_dir(text_root,seed,arm)/"history.json"
            app.f.immutable(root/"provenance"/f"seed{seed}"/(arm+"_history.json"),load_json(path))
    print("Verified locked cached predictions. No training, feature export or prediction generation.",flush=True)


def decode_audit(row, size):
    truth, info = app.evaluation_truth(row,size)
    empty = bool(info["image_label"] and not info["native_positive"])
    info.update(dataset=row["dataset"],category=row["category"],image_path=row["image_path"],
        mask_path=row["mask_path"],mask_sha=file_digest(row["mask_path"]) if row["mask_path"] else None,
        pixel_valid=not empty,reason="empty_under_frozen_decoder" if empty else "valid",
        palette_zero_rgb=None)
    if empty and info["mode"] == "P":
        with Image.open(row["mask_path"]) as image:
            colors = image.getpalette()
            info["palette_zero_rgb"] = colors[:3] if colors else None
    return truth, info


def preview(root,row,truth):
    size = 256
    with Image.open(row["image_path"]) as image: original = image.convert("RGB").resize((size,size))
    with Image.open(row["mask_path"]) as image: raw = image.convert("RGB").resize((size,size),Image.Resampling.NEAREST)
    decoded = Image.fromarray((truth*255).astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST).convert("RGB")
    canvas = Image.new("RGB",(3*size,size+48),"white"); draw = ImageDraw.Draw(canvas)
    draw.text((4,4),f"{row['dataset']}/{row['category']} {row['id']}",fill="black")
    for i,(title,tile) in enumerate((("query: anomaly label retained",original),("original mask rendered",raw),("decoded: no pixel GT",decoded))):
        draw.text((i*size+4,25),title,fill="black"); canvas.paste(tile,(i*size,48))
    path = Path(root)/"empty_mask_previews"/(row["id"]+".png"); path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def audit(args):
    root = Path(args.reeval_dir); cfg = config(root)
    _,text_cfg,source = source_context(root,cfg)
    verify_all_predictions(root,cfg)
    rows,counts,hashes = [],[],{}
    for key,queries in app.f.groups(app.f.rows(source,"eval")).items():
        infos = []
        for row in queries:
            app.unchanged(row)
            truth,info = decode_audit(row,text_cfg["settings"]["eval_size"])
            infos.append(info); hashes[row["id"]] = info["mask_sha"]
            if not info["pixel_valid"] and sum(not a["pixel_valid"] for a in infos) <= 4:
                preview(root,row,truth)
        value = dict(fingerprint=digest(cfg),images=infos)
        app.f.immutable(root/"mask_audit"/("__".join(key)+".json"),value)
        rows.extend(infos)
        counts.append(dict(dataset=key[0],category=key[1],images=len(infos),
            abnormal_images=sum(a["image_label"] for a in infos),
            empty_anomaly_masks=sum(not a["pixel_valid"] for a in infos),
            pixel_images_valid=sum(a["pixel_valid"] for a in infos)))
    files = {p.name:file_digest(p) for p in sorted((root/"mask_audit").glob("*.json"))}
    app.f.immutable(root/"mask_audit_locked.json",dict(fingerprint=digest(cfg),files=files,mask_hashes=hashes))
    app.base.write_csv(root/"evaluation_counts.csv",counts)
    app.base.write_csv(root/"empty_masks.csv",[r for r in rows if not r["pixel_valid"]])
    print(f"Mask audit: {sum(not r['pixel_valid'] for r in rows)} empty anomaly masks. "
          "Image labels retained; pixel subset results will be labeled diagnostic.",flush=True)


def image_metrics(maps, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    scores = maps.reshape(len(maps),-1).max(1)
    return dict(I_AUROC=float(roc_auc_score(labels,scores)) if labels.any() and not labels.all() else None,
                I_AP=float(average_precision_score(labels,scores)) if labels.any() else None)


def metric_scopes(truth, maps, labels, keep, threshold):
    if (truth.shape != maps.shape or labels.shape != (len(maps),) or keep.shape != labels.shape
            or not np.isfinite(maps).all()):
        raise ValueError("Invalid maps/GT/validity array")
    pixel = {m:None for m in PIXEL}
    if keep.any():
        point = app.corrected_metrics(truth[keep],maps[keep],labels[keep],threshold)
        pixel = {m:point[m] for m in PIXEL}
    common = dict(**image_metrics(maps,labels),images=len(maps),abnormal_images=int(labels.sum()),
        pixel_images_valid=int(keep.sum()),empty_anomaly_masks=int((~keep).sum()),threshold=threshold)
    subset = dict(common,**pixel,scope="valid_masks_diagnostic")
    full = dict(common,**(pixel if keep.all() else {m:None for m in PIXEL}),scope="full_set")
    return [full,subset]


def audit_record(root,cfg,key,queries):
    lock = load_json(Path(root)/"mask_audit_locked.json")
    path = Path(root)/"mask_audit"/("__".join(key)+".json")
    if lock["fingerprint"] != digest(cfg) or lock["files"][path.name] != file_digest(path):
        raise ValueError("Mask audit changed")
    infos = load_json(path)["images"]
    if [a["id"] for a in infos] != [r["id"] for r in queries]:
        raise ValueError("Mask audit query mismatch")
    for row,info in zip(queries,infos):
        sha = file_digest(row["mask_path"]) if row["mask_path"] else None
        if info["mask_sha"] != sha or lock["mask_hashes"][row["id"]] != sha:
            raise ValueError("Target mask changed; use new REEVAL_DIR for corrected annotations")
    return infos


def evaluate(args):
    root = Path(args.reeval_dir); cfg = config(root)
    text_root,text_cfg,source = source_context(root,cfg)
    if not 0 <= args.shard_id < cfg["workers"]: raise ValueError("Invalid evaluation shard")
    for i,(key,queries) in enumerate(app.f.groups(app.f.rows(source,"eval")).items()):
        if i%cfg["workers"] != args.shard_id: continue
        infos = audit_record(root,cfg,key,queries)
        truth = np.stack([app.evaluation_truth(r,text_cfg["settings"]["eval_size"])[0] for r in queries])
        keep = np.array([r["pixel_valid"] for r in infos],bool)
        labels = np.array([bool(r["mask_path"]) for r in queries])
        for seed in cfg["seeds"]:
            out = app.prediction_dir(text_root,seed,cfg["until_epoch"]); lock = load_json(out/"locked.json")
            selected = app.decision(text_root,text_cfg,seed,cfg["until_epoch"])
            metadata = []
            for row in queries:
                app.unchanged(row); metadata.append(verify_prediction(out/(row["id"]+".npz"),row,lock))
            records = []
            for name in app.MAPS:
                values = []
                for row in queries:
                    with np.load(out/(row["id"]+".npz"),allow_pickle=False) as data:
                        if set(data.files) != set(app.MAPS): raise ValueError("Wrong prediction arms")
                        values.append(data[name].copy())
                maps = np.stack(values)
                for point in metric_scopes(truth,maps,labels,keep,selected["thresholds"][name]):
                    arm = name.split("_")[0]
                    point.update(dataset=key[0],category=key[1],seed=seed,arm=arm,
                        score="text" if name.endswith("_text") else "full",visual_sha=text_cfg["visual_sha"],
                        text_epoch=selected["text"][arm]["epoch"] if arm in app.MODES else None)
                    if name in app.MODES:
                        point.update(increment_positive_fraction=float(np.mean([m["increments"][name]["positive_fraction"] for m in metadata])),
                                     mean_increment=float(np.mean([m["increments"][name]["mean"] for m in metadata])))
                    records.append(point)
            app.f.immutable(root/"metrics"/f"seed{seed}"/("__".join(key)+".json"),
                dict(fingerprint=digest(cfg),audit_sha=file_digest(root/"mask_audit_locked.json"),records=records))
        print(f"evaluated {key}: images={len(labels)}, anomaly={labels.sum()}, "
              f"pixel-valid={keep.sum()}, empty-anomaly={int((~keep).sum())}",flush=True)


def aggregate(rows, keys, metrics=app.METRICS+EXTRA):
    # An unavailable class metric must NOT disappear from the macro denominator.
    result = app.summarize(rows,keys,metrics)
    for value in result:
        members = [r for r in rows if all(r[k] == value[k] for k in keys)]
        for m in metrics:
            if any(r.get(m) is None for r in members):
                value[m] = value[m+"_std"] = None
    return result


def report(args):
    root = Path(args.reeval_dir); cfg = config(root)
    text_root,text_cfg,source = source_context(root,cfg)
    records = []
    for key,queries in app.f.groups(app.f.rows(source,"eval")).items():
        infos = audit_record(root,cfg,key,queries)
        for seed in cfg["seeds"]:
            out = app.prediction_dir(text_root,seed,cfg["until_epoch"]); lock = load_json(out/"locked.json")
            for row in queries: verify_prediction(out/(row["id"]+".npz"),row,lock)
            data = load_json(root/"metrics"/f"seed{seed}"/("__".join(key)+".json"))
            points = data["records"]
            actual = {(r["scope"],r["arm"]+("_text" if r["score"]=="text" else "")) for r in points}
            if (data["fingerprint"] != digest(cfg) or data["audit_sha"] != file_digest(root/"mask_audit_locked.json")
                    or len(points) != len(SCOPES)*len(app.MAPS) or actual != {(s,n) for s in SCOPES for n in app.MAPS}
                    or any(r["seed"] != seed or (r["dataset"],r["category"]) != key or r["images"] != len(queries)
                        or r["pixel_images_valid"] != sum(a["pixel_valid"] for a in infos)
                        or r["abnormal_images"] != sum(bool(q["mask_path"]) for q in queries)
                        or r["visual_sha"] != text_cfg["visual_sha"] for r in points)):
                raise ValueError("Wrong/stale/incomplete reevaluation metrics")
            records.extend(points)
    per_seed = aggregate(records,("scope","dataset","arm","score","seed"))
    summary = aggregate(per_seed,("scope","dataset","arm","score"))
    category = aggregate(records,("scope","dataset","category","arm","score"))
    paired = []
    for scope in SCOPES:
        for dataset in sorted({r["dataset"] for r in summary}):
            for seed in cfg["seeds"]:
                lookup = {r["arm"]:r for r in per_seed if r["scope"]==scope and r["dataset"]==dataset and r["seed"]==seed and r["score"]=="full"}
                for name,a,b in (("T1-T2","T1","T2"),("T3-T1","T3","T1"),*((a+"-V",a,"V") for a in app.MODES)):
                    paired.append(dict(scope=scope,dataset=dataset,seed=seed,comparison=name,**{m+"_pp":
                        100*(lookup[a][m]-lookup[b][m]) if lookup[a][m] is not None and lookup[b][m] is not None else None for m in app.METRICS}))
    contrasts = aggregate(paired,("scope","dataset","comparison"),tuple(m+"_pp" for m in app.METRICS))
    target = root/"reports"; target.mkdir(parents=True,exist_ok=True)
    for name,values in (("metrics",records),("per_seed",per_seed),("summary",summary),("per_category",category),
                        ("paired_contrasts",paired),("contrasts",contrasts)):
        app.base.write_csv(target/(name+".csv"),values)
    counts = []
    for path in sorted((root/"mask_audit").glob("*.json")):
        infos = load_json(path)["images"]
        counts.append(dict(dataset=infos[0]["dataset"],category=infos[0]["category"],images=len(infos),
            abnormal_images=sum(a["image_label"] for a in infos),pixel_images_valid=sum(a["pixel_valid"] for a in infos),
            empty_anomaly_masks=sum(not a["pixel_valid"] for a in infos)))
    lines = ["# 文本对比实验：空mask审计与缓存复评","",f"协议：`{PROTOCOL}`；原文本快照through{cfg['until_epoch']}；种子{cfg['seeds']}。",
        "全部复用原来已锁定的V/T1/T2/T3及文本单支路预测，不重训、不重新预测、不修改原mask。",
        "**原始空mask的原因尚需人工核查。下表不把空mask当正常图，也不伪造像素标注。**",
        "所有图像均参与I-AUROC/I-AP，包括空mask异常图。完整集像素GT不全的类别/数据集显示—，缺失类别不从宏平均中悄悄移除。",
        "另列有效mask子集的像素诊断：只排除原生异常mask解码为空的图像，正常图保留，七种读出共用完全相同的样本集。",
        "**有效mask子集是诊断结果，不能标作标准完整BTAD结果或直接与其他论文的完整数据集分数比较。**",
        f"mask解码仍为原协议`{cfg['decoder_protocol']}`，没有自动改变前景定义或根据预测补标注。",
        "数值为%，类别宏平均后跨文本种子平均；F1-best为目标最优阈值诊断，FPR/召回沿用源CAL阈值。","",
        "## 样本数与缺失标注","","| 数据集 | 类别 | 图像总数 | 异常图数 | 像素有效图数 | 空异常mask数 |","|---|---|---:|---:|---:|---:|"]
    for r in counts:
        lines.append("| "+" | ".join(str(r[k]) for k in ("dataset","category","images","abnormal_images","pixel_images_valid","empty_anomaly_masks"))+" |")
    for scope,title in (("full_set","完整图像集；像素标注不全时不报告完整分割分数"),
                        ("valid_masks_diagnostic","有效mask子集诊断；不是标准完整数据集分数")):
        lines += ["",f"## {title}","",*app.table([r for r in summary if r['scope']==scope and r['score']=='full']),
                  "","### 文本单支路","",*app.table([r for r in summary if r['scope']==scope and r['score']=='text'])]
    lines += ["","## 配对增益（百分点）","","| 范围 | 数据集 | 对比 | I-AUROC | P-AP | AUPRO |","|---|---|---|---:|---:|---:|"]
    for r in contrasts:
        lines.append("| "+" | ".join([r['scope'],r['dataset'],r['comparison']]+[
            "—" if r[m+'_pp'] is None else f"{r[m+'_pp']:+.2f}"+(f" ± {r[m+'_pp_std']:.2f}" if r[m+'_pp_std'] else "") for m in ('I_AUROC','P_AP','PRO_exact')])+" |")
    lines += ["","## 文本正增量覆盖（完整图像集，与GT有效性无关）","",
              "| 数据集 | 组别 | 正增量像素比例(%) | 平均分数增量 |","|---|---|---:|---:|"]
    for r in summary:
        if r['scope']=='full_set' and r['score']=='full' and r['arm'] in app.MODES:
            lines.append(f"| {r['dataset']} | {r['arm']} | {100*r['increment_positive_fraction']:.2f} | {r['mean_increment']:.4f} |")
    for scope in SCOPES:
        lines += ["",f"## 逐类别：{scope}","",*app.table([r for r in category if r['scope']==scope],True)]
    text = "\n".join(lines)+"\n"; (target/"对比实验.md").write_text(text,encoding="utf-8")
    if args.publish_report: app.f.publish_markdown(REPO/"对比实验.md",text)
    print(f"Reevaluation report: {target/'对比实验.md'}",flush=True)


def package(args):
    root = Path(args.reeval_dir); config(root)
    path = root/("text_reevaluation_analysis_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+".zip")
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".json",".csv",".md",".png",".log")]
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(candidates): archive.write(item,item.relative_to(root).as_posix())
    print(f"Reevaluation analysis ZIP: {path}",flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=("prepare","audit","evaluate","report","package"))
    p.add_argument("--work_dir",default="./checkpoint/text_comparison_v1")
    p.add_argument("--reeval_dir",default="./checkpoint/text_comparison_v1_reeval_v2")
    p.add_argument("--until_epoch",type=int,default=5); p.add_argument("--train_seeds",default="42")
    p.add_argument("--workers",type=int,default=2); p.add_argument("--shard_id",type=int,default=0)
    p.add_argument("--publish_report",action="store_true")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    globals()[arguments.stage](arguments)
