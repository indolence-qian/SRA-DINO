#!/usr/bin/env python3
"""Versioned, CPU-only reevaluation of the original locked text predictions.

No training code, masks, checkpoints, source decisions or old outputs are changed.
Image labels and pixel annotations are independent. Official BTAD contains
anomalous images with all-zero masks: include them exactly as distributed.
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
PROTOCOL = "locked_text_reevaluation_as_provided_v3"
CODE = ("repair_text_evaluation.py", "run_repair_text_evaluation.sh", "docs/btad_official_zero_masks.json")
PIXEL = tuple(m for m in app.METRICS if m not in ("I_AUROC", "I_AP"))
EXTRA = ("increment_positive_fraction", "mean_increment")
SCOPES = ("full_set",)


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
        empty_mask_policy="image_label_retained;all_images_and_as_provided_pixel_masks_included;no_exclusions",
        code={n:file_digest(REPO/n) for n in CODE})
    app.f.immutable(root/"config.json",cfg)
    for name,path in (("text_config.json",text_root/"config.json"),
                      ("manifest.json",source/"manifest.json"),("references.json",source/"references.json"),
                      ("normal_stats.json",source/"normal_stats.json"),("text_bank.json",text_root/"text_bank.json")):
        app.f.immutable(root/"provenance"/name,load_json(path))
    app.f.immutable(root/"provenance"/"official_zero_mask_reference.json",load_json(REPO/"docs/btad_official_zero_masks.json"))
    for seed in seeds:
        app.f.immutable(root/"provenance"/f"seed{seed}"/"selection.json",load_json(app.decision_path(text_root,seed,args.until_epoch)))
        for arm in app.MODES:
            path = app.head_dir(text_root,seed,arm)/"history.json"
            app.f.immutable(root/"provenance"/f"seed{seed}"/(arm+"_history.json"),load_json(path))
    print("Verified locked cached predictions. No training, feature export or prediction generation.",flush=True)


def dataset_label(row):
    label = bool(row["mask_path"])
    if row["dataset"] == "btad":
        state = Path(row["image_path"]).parent.name.lower()
        if state not in ("ok", "ko") or label != (state == "ko"):
            raise ValueError(f"BTAD directory/image-label/mask-path contradiction: {row['image_path']}")
        return state == "ko"
    return label


def decode_audit(row, size):
    truth, info = app.evaluation_truth(row,size)
    empty = bool(info["image_label"] and not info["native_positive"])
    info.update(dataset=row["dataset"],category=row["category"],image_path=row["image_path"],
        mask_path=row["mask_path"],mask_sha=file_digest(row["mask_path"]) if row["mask_path"] else None,
        pixel_included=True,empty_anomaly_mask=empty,
        reason="empty_as_provided_included" if empty else "as_provided_included",
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
    for i,(title,tile) in enumerate((("query: anomaly label retained",original),("original mask rendered",raw),("decoded: zero GT retained",decoded))):
        draw.text((i*size+4,25),title,fill="black"); canvas.paste(tile,(i*size,48))
    path = Path(root)/"empty_mask_previews"/(row["id"]+".png"); path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def audit(args):
    root = Path(args.reeval_dir); cfg = config(root)
    _,text_cfg,source = source_context(root,cfg)
    verify_all_predictions(root,cfg)
    references = load_json(REPO/"docs/btad_official_zero_masks.json")["groups"]
    official = {g["directory_in_zip"].split("/",1)[1]+"/"+name:g["sha256_each"]
                for g in references for name in g["filenames"]}
    rows,counts,hashes = [],[],{}
    for key,queries in app.f.groups(app.f.rows(source,"eval")).items():
        infos = []
        for row in queries:
            app.unchanged(row)
            dataset_label(row)
            truth,info = decode_audit(row,text_cfg["settings"]["eval_size"])
            suffix = f"{row['category']}/ground_truth/ko/{Path(row['mask_path']).name}" if row['mask_path'] else None
            expected_sha = official.get(suffix) if row['dataset']=='btad' else None
            info.update(official_reference_sha=expected_sha,
                        official_bytes_match=(info['mask_sha']==expected_sha) if expected_sha else None)
            infos.append(info); hashes[row["id"]] = info["mask_sha"]
            if info["empty_anomaly_mask"] and sum(a["empty_anomaly_mask"] for a in infos) <= 4:
                preview(root,row,truth)
        value = dict(fingerprint=digest(cfg),images=infos)
        app.f.immutable(root/"mask_audit"/("__".join(key)+".json"),value)
        rows.extend(infos)
        counts.append(dict(dataset=key[0],category=key[1],images=len(infos),
            abnormal_images=sum(a["image_label"] for a in infos),
            empty_anomaly_masks=sum(a["empty_anomaly_mask"] for a in infos),
            pixel_images_included=len(infos)))
    files = {p.name:file_digest(p) for p in sorted((root/"mask_audit").glob("*.json"))}
    app.f.immutable(root/"mask_audit_locked.json",dict(fingerprint=digest(cfg),files=files,mask_hashes=hashes))
    app.base.write_csv(root/"evaluation_counts.csv",counts)
    app.base.write_csv(root/"empty_masks.csv",[r for r in rows if r["empty_anomaly_mask"]])
    print(f"Mask audit: {sum(r['empty_anomaly_mask'] for r in rows)} empty anomaly masks. "
          "All image labels and pixel masks retained as provided; NO sample exclusions.",flush=True)


def metric_scopes(truth, maps, labels, threshold):
    if (truth.shape != maps.shape or labels.shape != (len(maps),)
            or not np.isfinite(maps).all()):
        raise ValueError("Invalid maps/GT/labels array")
    point = app.corrected_metrics(truth,maps,labels,threshold)
    empty = labels & ~truth.reshape(len(truth),-1).any(1)
    point.update(images=len(maps),abnormal_images=int(labels.sum()),pixel_images_included=len(maps),
                 empty_anomaly_masks=int(empty.sum()),scope="full_set")
    return [point]


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
        labels = np.array([dataset_label(r) for r in queries],bool)
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
                for point in metric_scopes(truth,maps,labels,selected["thresholds"][name]):
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
            app.panels(root,text_cfg,queries,out,seed,cfg["until_epoch"],truth)
        print(f"evaluated {key}: images={len(labels)}, anomaly={labels.sum()}, "
              f"pixel-included={len(labels)}, empty-anomaly={sum(r['empty_anomaly_mask'] for r in infos)} (all retained)",flush=True)


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
                        or r["pixel_images_included"] != len(infos)
                        or r["abnormal_images"] != sum(dataset_label(q) for q in queries)
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
    selections = []
    for seed in cfg["seeds"]:
        for arm,value in app.decision(text_root,text_cfg,seed,cfg["until_epoch"])["text"].items():
            selections.append(dict(seed=seed,arm=arm,epoch=value["epoch"],**value["source_point"]))
    app.base.write_csv(target/"selected_epochs.csv",selections)
    counts = []
    for path in sorted((root/"mask_audit").glob("*.json")):
        infos = load_json(path)["images"]
        counts.append(dict(dataset=infos[0]["dataset"],category=infos[0]["category"],images=len(infos),
            abnormal_images=sum(a["image_label"] for a in infos),pixel_images_included=len(infos),
            empty_anomaly_masks=sum(a["empty_anomaly_mask"] for a in infos)))
    lines = ["# 文本对比实验：空mask审计与缓存复评","",f"协议：`{PROTOCOL}`；原文本快照through{cfg['until_epoch']}；种子{cfg['seeds']}。",
        "全部复用原来已锁定的V/T1/T2/T3及文本单支路预测，不重训、不重新预测、不修改原mask。",
        "**保留全部样本。图像标签与像素标注独立：空mask的ko图像仍是异常图，像素级使用发布的全零mask。**",
        "已从作者官方ZIP独立读取02/0145.png及03/0028.bmp—0037.bmp；11个mask的SHA-256与用户审计记录全部一致，原件均为灰度L、像素仅0。",
        "官方原件核对记录见provenance/official_zero_mask_reference.json；不把全零mask自动判为损坏或缺失，不删除图像、不补造GT。",
        "图像级I-AUROC/I-AP使用完整图像标签；P-AUROC/P-AP/AUPRO/F1/FPR/召回使用完整集合的原样mask。所有七种读出使用相同样本和GT。",
        f"mask解码仍为原协议`{cfg['decoder_protocol']}`，没有自动改变前景定义或根据预测补标注。",
        "数值为%，类别宏平均后跨文本种子平均；F1-best为目标最优阈值诊断，FPR/召回沿用源CAL阈值。","",
        "## 全量样本数与全零mask统计","","| 数据集 | 类别 | 图像总数 | 异常图数 | 像素评价图数 | 全零异常mask数 |","|---|---|---:|---:|---:|---:|"]
    for r in counts:
        lines.append("| "+" | ".join(str(r[k]) for k in ("dataset","category","images","abnormal_images","pixel_images_included","empty_anomaly_masks"))+" |")
    for scope,title in (("full_set","全部图像、全部发布mask的统一评价"),):
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
    lines += ["","## 原源验证选择轮次（保持原样）","","| 文本种子 | 组别 | 轮次 | 源P-AP | 源AUPRO |","|---|---|---:|---:|---:|"]
    for r in selections:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['epoch']} | {100*r['P_AP']:.2f} | {100*r['PRO_exact']:.2f} |")
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
    p.add_argument("--reeval_dir",default="./checkpoint/text_comparison_v1_reeval_v3")
    p.add_argument("--until_epoch",type=int,default=5); p.add_argument("--train_seeds",default="42")
    p.add_argument("--workers",type=int,default=2); p.add_argument("--shard_id",type=int,default=0)
    p.add_argument("--publish_report",action="store_true")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    globals()[arguments.stage](arguments)
