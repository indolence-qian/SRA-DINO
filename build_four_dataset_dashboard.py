#!/usr/bin/env python3
"""Offline HTML error analysis for the current four-dataset experiment.

Reuses locked predictions from the selected trained weights. No model training
or forward pass is performed. NumPy/Pillow are the only external dependencies.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

import render_four_dataset_heatmaps as heat

TEMPLATE = Path(__file__).resolve().parent / "asset" / "four_dataset_dashboard.html"
VERSION = 1


def write_html(output, data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__DASHBOARD_DATA__") != 1:
        raise ValueError("Dashboard template must contain one data placeholder")
    (Path(output) / "index.html").write_text(template.replace("__DASHBOARD_DATA__", payload), encoding="utf-8")


def evaluation_truth(row, side):
    """Match adaptive-max-pool reduction + PyTorch nearest upsampling in NumPy."""
    if not row["mask_path"]:
        return np.zeros((side, side), bool), None
    with Image.open(row["mask_path"]) as image:
        if image.size != tuple(row["native_size"]):
            raise ValueError(f"Mask geometry differs: {row['mask_path']}")
        raw = np.asarray(image)
        mask = (raw[..., :3] > 0).any(-1) if raw.ndim == 3 else raw > 0
    h, w = mask.shape
    oh, ow = min(h, side), min(w, side)
    if (h, w) != (oh, ow):
        integral = np.zeros((h+1, w+1), np.int32)
        np.cumsum(mask, axis=0, dtype=np.int32, out=integral[1:, 1:])
        np.cumsum(integral[1:, 1:], axis=1, dtype=np.int32, out=integral[1:, 1:])
        ys, xs = np.arange(oh)*h//oh, np.arange(ow)*w//ow
        ye, xe = ((np.arange(oh)+1)*h+oh-1)//oh, ((np.arange(ow)+1)*w+ow-1)//ow
        mask = (integral[ye[:, None], xe]-integral[ys[:, None], xe]
                -integral[ye[:, None], xs]+integral[ys[:, None], xs]) > 0
    mask = mask[(np.arange(side)*oh//side)[:, None], np.arange(side)*ow//side]
    return mask, heat.sha256(row["mask_path"])


def binary_ranking(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positives = int(labels.sum())
    negatives = len(labels)-positives
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite image scores")
    auc = None
    if positives and negatives:
        order = np.argsort(scores, kind="stable")
        x, y = scores[order], labels[order]
        starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1])+1]
        counts = np.diff(np.r_[starts, len(x)])
        pos = np.add.reduceat(y.astype(np.int64), starts)
        auc = float((np.dot(pos, starts+(counts+1)/2)-positives*(positives+1)/2)/(positives*negatives))
    ap = None
    if positives:
        order = np.argsort(-scores, kind="stable")
        x, y = scores[order], labels[order]
        ends = np.r_[np.flatnonzero(x[1:] != x[:-1]), len(x)-1]
        tp = np.cumsum(y)[ends]
        ap = float(np.sum(tp/(ends+1)*np.diff(np.r_[0, tp])/positives))
    return auc, ap


def diagnose(maps, truth, label, thresholds):
    result = {}
    foreground, background = int(truth.sum()), int((~truth).sum())
    for name, values in maps.items():
        predicted = values >= thresholds[name]
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = foreground-tp
        union = foreground+fp
        score = float(values.max())
        triggered = score >= thresholds[name]
        outcome = ("TP" if label else "FP") if triggered else ("FN" if label else "TN")
        annotated = label and foreground > 0
        result[name] = dict(score=score, margin=score-thresholds[name], threshold=thresholds[name],
            outcome=outcome, image_correct=bool(triggered == label), tp_pixels=tp, fp_pixels=fp, fn_pixels=fn,
            iou=tp/union if annotated else None,
            dice=2*tp/(foreground+tp+fp) if annotated else None,
            recall=tp/foreground if annotated else None,
            precision=tp/(tp+fp) if annotated and tp+fp else (0. if annotated else None),
            background_fpr=fp/background if background else None,
            predicted_fraction=float(predicted.mean()))
    return dict(branches=result, gt_positive_pixels=foreground, gt_fraction=foreground/truth.size,
                empty_anomaly_mask=bool(label and foreground == 0),
                dice_delta=(result["full"]["dice"]-result["visual"]["dice"]) if label and foreground else None)


def export_details(row, truth, maps, thresholds, output, max_side):
    folder = output / heat.component(row["dataset"]) / heat.component(row["category"])
    with Image.open(row["image_path"]) as image:
        size = heat.thumbnail_size(image.size, max_side)
        original = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    predicted = maps["full"] >= thresholds["full"]
    color = np.zeros((*truth.shape, 3), np.uint8)
    color[predicted & truth] = [30, 180, 105]
    color[predicted & ~truth] = [239, 68, 68]
    color[~predicted & truth] = [55, 120, 245]
    resized = Image.fromarray(color).resize(size, Image.Resampling.NEAREST)
    colors = np.asarray(resized)
    pixels = np.asarray(original).copy()
    active = colors.any(-1)
    pixels[active] = (.35*pixels[active]+.65*colors[active]).astype(np.uint8)
    files = {"original": row["id"]+"_original.jpg", "gt": row["id"]+"_gt.png", "errors": row["id"]+"_errors.jpg"}
    original.save(folder / files["original"], quality=90)
    Image.fromarray(truth.astype(np.uint8)*255).resize(size, Image.Resampling.NEAREST).save(folder / files["gt"])
    Image.fromarray(pixels).save(folder / files["errors"], quality=92)
    return {key: (folder / value).relative_to(output).as_posix() for key, value in files.items()}


def suggestions(case):
    point = case["full"]
    notes = []
    if case["empty_anomaly_mask"]:
        notes.append("图像为异常标签，但像素标注全零：仅评价图像检测，不按 IoU/Dice 判断定位。")
    if point["outcome"] == "FP":
        notes.append("正常图触发源像素阈值：请检查高分区是否为反光、纹理、边缘或未标注区域。")
    if point["outcome"] == "FN":
        notes.append("异常图未触发源阈值：请对照缺陷尺寸、分数范围及热图响应检查。")
    if point["dice"] is not None and point["dice"] < .3 and point["outcome"] == "TP":
        notes.append("图像虽检出，但与标注区域重叠较低；请对照红色误报与蓝色漏检区。")
    if 0 < case["gt_fraction"] <= .001 and point["recall"] is not None and point["recall"] < .5:
        notes.append("标注区域占比≤0.1%，源阈值下召回较低：重点检查微小缺陷。")
    if point["background_fpr"] is not None and point["background_fpr"] > .05:
        notes.append("背景激活超过5%，请检查响应是否扩散到正常物体区域。")
    if case["annotation_kind"] == "bounding_box":
        notes.append("LD 使用框级代理标注，定位指标反映与矩形框的重叠。")
    return notes or ["可结合原图与热图复核；当前统计不能直接证明具体缺陷原因。"]


def aggregate(cases, official=None):
    official = official or {}
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["dataset"], case["category"]].append(case)
    categories = []
    for (dataset, category), values in sorted(grouped.items()):
        row = dict(dataset=dataset, category=category, images=len(values))
        for branch in ("visual", "full"):
            auc, ap = binary_ranking([r["label"] for r in values], [r[branch]["score"] for r in values])
            counts = {name: sum(r[branch]["outcome"] == name for r in values) for name in ("TP", "TN", "FP", "FN")}
            dices = [r[branch]["dice"] for r in values if r[branch]["dice"] is not None]
            row[branch] = dict(**counts, image_auc=auc, image_ap=ap,
                dice=float(np.mean(dices)) if dices else None, localized_images=len(dices),
                pixel_ap=official.get((dataset, category, branch), {}).get("P_AP"),
                pixel_auc=official.get((dataset, category, branch), {}).get("P_AUROC"),
                best_f1=official.get((dataset, category, branch), {}).get("F1_best"),
                pro=official.get((dataset, category, branch), {}).get("PRO_exact"))
        categories.append(row)
    datasets = []
    for dataset in heat.TARGETS:
        values = [r for r in cases if r["dataset"] == dataset]
        if not values:
            continue
        row = dict(dataset=dataset, images=len(values), normal=sum(not r["label"] for r in values),
                   abnormal=sum(r["label"] for r in values), empty_masks=sum(r["empty_anomaly_mask"] for r in values))
        cats = [r for r in categories if r["dataset"] == dataset]
        for branch in ("visual", "full"):
            point = {name: sum(r[branch][name] for r in cats) for name in ("TP", "TN", "FP", "FN")}
            point["accuracy"] = (point["TP"]+point["TN"])/len(values)
            for metric in ("image_auc", "image_ap", "pixel_auc", "pixel_ap", "pro", "best_f1", "dice"):
                valid = [r[branch][metric] for r in cats if r[branch][metric] is not None]
                point[metric] = float(np.mean(valid)) if valid else None
                point[metric+"_valid_categories"] = len(valid)
            point["pooled_image_auc"] = binary_ranking([r["label"] for r in values], [r[branch]["score"] for r in values])[0]
            row[branch] = point
        datasets.append(row)
    return datasets, categories


def run(args):
    cfg, manifest, predictions, lock, thresholds, until = heat.load_run(args.work_dir, args.seed, args.until_epoch)
    if args.no_gt:
        raise ValueError("Error analysis needs GT; use the heatmap-only renderer when GT is unavailable")
    work = Path(args.work_dir).resolve()
    output = args.output_dir or work / "analysis" / f"seed{args.seed}" / f"through{until}" / args.normalization
    render_args = copy_args(args, output)
    output, rendered = heat.run(render_args)
    shutil.copyfile(output / "index.html", output / "gallery.html")
    by_id = {r["id"]: r for r in manifest}
    expected_groups = defaultdict(set)
    for row in manifest:
        expected_groups[row["dataset"], row["category"]].add(row["id"])
    selected_groups = defaultdict(set)
    for row in rendered:
        selected_groups[row["dataset"], row["category"]].add(row["id"])
    official, audit = {}, {}
    for key, ids in selected_groups.items():
        path = work / "metrics" / f"seed{args.seed}" / f"through{until}" / ("__".join(key)+".json")
        if not path.is_file():
            continue
        data = heat.read_json(path)
        if data["prediction_lock_sha"] != heat.sha256(predictions / "locked.json"):
            raise ValueError("Official metrics do not match prediction lock")
        audit.update(data["evaluation_mask_hashes"])
        if ids == expected_groups[key]:
            for point in data["metrics"]:
                official[key[0], key[1], point["score"]] = point
    cases = []
    for index, record in enumerate(rendered, 1):
        row = by_id[record["id"]]
        maps = heat.load_scores(row, cfg, predictions, lock)
        truth, mask_sha = evaluation_truth(row, cfg["eval_size"])
        if row["id"] in audit and audit[row["id"]] != mask_sha:
            raise ValueError(f"GT changed after evaluation: {row['mask_path']}")
        metrics = diagnose(maps, truth, bool(row["mask_path"]), thresholds)
        branches = metrics.pop("branches")
        case = dict(id=row["id"], dataset=row["dataset"], category=row["category"],
            filename=Path(row["image_path"]).name, image_path=row["image_path"], label=int(bool(row["mask_path"])),
            annotation_kind=record["annotation_kind"], visual=branches["visual"], full=branches["full"],
            **metrics, files=dict(record["files"]))
        case["files"].update(export_details(row, truth, maps, thresholds, output, args.max_side))
        case["notes"] = suggestions(case)
        cases.append(case)
        if index % 25 == 0 or index == len(rendered):
            print(f"Analysis {index}/{len(rendered)}", flush=True)
    datasets, categories = aggregate(cases, official)
    data = dict(version=VERSION, seed=args.seed, horizon=until, normalization=args.normalization,
        experiment=str(work), source_thresholds=thresholds, datasets=datasets, categories=categories, cases=cases,
        coverage=dict(selected=len(cases), total=len(manifest)),
        provenance=dict(prediction_lock_sha256=heat.sha256(predictions / "locked.json"), builder_sha256=heat.sha256(__file__),
                        template_sha256=heat.sha256(TEMPLATE)))
    heat.write_json(output / "analysis.json", data)
    write_html(output, data)
    fields = ["id", "dataset", "category", "filename", "label", "annotation_kind", "gt_fraction", "empty_anomaly_mask", "dice_delta"]
    flat = []
    for row in cases:
        item = {key: row[key] for key in fields}
        for branch in ("visual", "full"):
            item.update({branch+"_"+key: value for key, value in row[branch].items()})
        flat.append(item)
    with (output / "sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if args.pack:
        files = {output / name for name in ("index.html", "gallery.html", "analysis.json", "sample_metrics.csv", "render_config.json", "index.csv", "index.json")}
        for row in cases:
            files.update(output / path for path in row["files"].values())
        with zipfile.ZipFile(output / "analysis_report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):
                archive.write(path, path.relative_to(output).as_posix())
        print(f"Download HTML + assets: {output / 'analysis_report.zip'}", flush=True)
    print(f"Interactive analysis: {output / 'index.html'}", flush=True)
    return output, data


def copy_args(args, output):
    import argparse
    values = vars(args).copy()
    values.update(output_dir=output, pack=False)
    return argparse.Namespace(**values)


def parser():
    p = heat.parser()
    p.description = __doc__
    p.set_defaults(pack=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
