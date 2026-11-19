#!/usr/bin/env python3
"""Render sealed four-dataset predictions using only NumPy and Pillow.

Reads existing scores; does not import training code, load models, or change
weights, prediction arrays, metrics, or sealed code hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROTOCOL = "fresh_residual_hr_text_four_targets_v1"
RENDERER_VERSION = 2
TARGETS = ("mvtec", "btad", "mpdd", "ld")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    # Match the immutable experiment's canonical JSON hashing exactly.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid dataset/category/id for output paths: {value!r}")
    return value


def output_variant(args):
    fraction = format(args.sample_fraction * 100, ".10g").replace(".", "p")
    return f"{args.normalization}_sample{fraction}_v2"


def select_samples(rows, args):
    """Systematic sampling within dataset/category/image-label strata.

    Midpoints of equal-width filename bins cover the whole stratum, without
    looking at scores, localization quality, or mask foreground. Small strata
    round up to one, so the actual fraction can be slightly above the request.
    """
    if not math.isfinite(args.sample_fraction) or not 0 < args.sample_fraction <= 1:
        raise ValueError("sample-fraction must be in (0, 1]")
    datasets = args.datasets.replace(",", " ").split()
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(TARGETS):
        raise ValueError("DATASETS must contain unique choices from mvtec,btad,mpdd,ld")
    categories = args.categories.replace(",", " ").split() if args.categories else []
    eligible = [r for r in rows if r["dataset"] in datasets and (not categories or r["category"] in categories)]
    if not eligible or (categories and set(categories)-{r["category"] for r in eligible}):
        raise ValueError("No matching predictions or unknown category filter")
    grouped = {}
    for row in eligible:
        key = component(row["dataset"]), component(row["category"]), int(bool(row["mask_path"]))
        grouped.setdefault(key, []).append(row)
    selected = []
    for key in sorted(grouped):
        samples = sorted(grouped[key], key=lambda r: (Path(r["image_path"]).name, r["id"]))
        count = max(1, math.ceil(len(samples) * args.sample_fraction))
        selected.extend(samples[(2*i+1)*len(samples)//(2*count)] for i in range(count))
    if args.limit_per_category:
        capped = {}
        for row in selected:
            capped.setdefault((row["dataset"], row["category"]), []).append(row)
        selected = []
        for samples in capped.values():
            samples.sort(key=lambda r: fingerprint([args.seed, "heatmap_sample", r["id"]]))
            selected.extend(samples[:args.limit_per_category])
    selected_ids = {r["id"] for r in selected}
    sampling = dict(method="systematic_midpoints_by_dataset_category_label", fraction=args.sample_fraction,
        limit_per_category=args.limit_per_category, eligible=len(eligible), selected=len(selected),
        actual_fraction=len(selected)/len(eligible), selected_ids=[r["id"] for r in selected],
        strata=[dict(dataset=key[0], category=key[1], label=key[2], total=len(samples),
                     selected=sum(r["id"] in selected_ids for r in samples)) for key, samples in sorted(grouped.items())])
    return eligible, selected, sampling


def pack_assets(output, archive_name, files):
    """Package an explicit resource closure, never leftovers from older renders."""
    output = Path(output).resolve()
    paths = sorted({Path(path) for path in files})
    hashes = {}
    for path in paths:
        if not path.resolve().is_relative_to(output) or not path.is_file():
            raise ValueError(f"Missing or nonportable report resource: {path}")
        hashes[path.relative_to(output).as_posix()] = sha256(path)
    manifest = output / "asset_manifest.json"
    write_json(manifest, dict(version=1, entry="index.html", files=hashes))
    temporary = output / (archive_name+".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [manifest]:
            archive.write(path, path.relative_to(output).as_posix())
    archive_path = output / archive_name
    temporary.replace(archive_path)
    return archive_path


def load_run(work, seed, until=None):
    work = Path(work).resolve()
    cfg = read_json(work / "config.json")
    if cfg.get("protocol") != PROTOCOL:
        raise ValueError("WORK_DIR must be the current four-dataset residual/Flow/HR/text run")
    manifest = work / "manifest.json"
    if sha256(manifest) != cfg["manifest_sha"]:
        raise ValueError("Data manifest changed after prepare")
    if until is None:
        candidates = []
        for path in (work / "predictions" / f"seed{seed}").glob("through*/locked.json"):
            match = re.fullmatch(r"through([0-9]+)", path.parent.name)
            if match:
                candidates.append(int(match[1]))
        if not candidates:
            raise FileNotFoundError("No locked predictions found. Finish prediction/lock stages before rendering heatmaps.")
        until = max(candidates)
    prediction_dir = work / "predictions" / f"seed{seed}" / f"through{until}"
    lock_path = prediction_dir / "locked.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"No completed prediction lock: {lock_path}")
    decision_path = work / "decisions" / f"seed{seed}" / f"through{until}" / "selection.json"
    decision = read_json(decision_path)
    if decision["fingerprint"] != fingerprint(cfg) or decision["seed"] != seed or decision["horizon"] != until:
        raise ValueError("Source selection does not match this experiment/seed/horizon")
    expected_identity = dict(fingerprint=fingerprint(cfg), selection=sha256(decision_path),
                             eval_seal=sha256(work / "sealed_eval.json"), seed=seed, horizon=until)
    lock = read_json(lock_path)
    if lock["identity"] != expected_identity:
        raise ValueError("Prediction lock does not match source selection/configuration")
    rows = [r for r in read_json(manifest) if r["partition"] == "eval"]
    if len({r["id"] for r in rows}) != len(rows) or set(lock["predictions"]) != {r["id"] for r in rows}:
        raise ValueError("Prediction lock does not cover exactly the target manifest")
    thresholds = decision["arms"]["D"]["thresholds"]
    if not all(math.isfinite(thresholds[name]) for name in ("visual", "full")):
        raise ValueError("Invalid source thresholds")
    return cfg, rows, prediction_dir, lock, thresholds, until


def load_scores(row, cfg, folder, lock):
    path = folder / (component(row["id"])+".npz")
    if sha256(path) != lock["predictions"][row["id"]]:
        raise ValueError(f"Prediction changed after locking: {path}")
    sidecar = read_json(path.with_suffix(".json"))
    if sidecar != {"identity": lock["identity"], "sha": lock["predictions"][row["id"]]}:
        raise ValueError(f"Prediction metadata changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"D_visual", "D_full"}:
            raise ValueError(f"Expected visual/full maps from the current D network: {path}")
        maps = {name: archive["D_"+name].astype(np.float32) for name in ("visual", "full")}
    if any(value.shape != (cfg["eval_size"], cfg["eval_size"]) or not np.isfinite(value).all() for value in maps.values()):
        raise ValueError(f"Invalid anomaly-score arrays: {path}")
    return maps


def display_maps(maps, thresholds, mode="source", temperature=1., low_percentile=1., high_percentile=99.):
    if mode == "source":
        result = {name: (1/(1+np.exp(-np.clip((value-thresholds[name])/temperature, -60, 60)))).astype(np.float32)
                  for name, value in maps.items()}
        return result, dict(mode=mode, temperature=temperature, thresholds=thresholds,
                            note="Color midpoint is the branch's fixed source CAL threshold; colors are not probabilities.")
    # Same per-image bounds for visual/full; GT never enters display normalization.
    values = np.concatenate([value.ravel() for value in maps.values()])
    low, high = map(float, np.percentile(values, [low_percentile, high_percentile]))
    if high-low <= 1e-12:
        result = {name: np.zeros_like(value) for name, value in maps.items()}
    else:
        result = {name: np.clip((value-low)/(high-low), 0, 1) for name, value in maps.items()}
    return result, dict(mode=mode, low=low, high=high,
                        percentiles=[low_percentile, high_percentile],
                        note="Display-only per-image contrast. Colors cannot be compared between different images.")


def jet(values):
    """Classic blue/cyan/yellow/red heatmap, RGB output without OpenCV/Matplotlib."""
    x = np.asarray(values, np.float32).clip(0, 1)
    return (np.stack([np.clip(1.5-np.abs(4*x-offset), 0, 1) for offset in (3, 2, 1)], -1)*255).round().astype(np.uint8)


def thumbnail_size(size, max_side):
    scale = min(1., max_side/max(size))
    return tuple(max(1, round(d*scale)) for d in size)


def load_truth(row, size, show_gt):
    if not show_gt:
        return None, None
    path = row["mask_path"]
    if not path:
        return np.zeros((size[1], size[0]), bool), None
    with Image.open(path) as image:
        if image.size != tuple(row["native_size"]):
            raise ValueError(f"GT/native image geometry differs: {path}")
        # Read palette indices and binary 0/1 masks directly. Ignore RGBA alpha.
        array = np.asarray(image)
        truth = (array[..., :3] > 0).any(-1) if array.ndim == 3 else array > 0
    # Positive coverage avoids losing a subpixel annotation in a thumbnail.
    truth = Image.fromarray(truth.astype(np.float32)).resize(size, Image.Resampling.BOX)
    return np.asarray(truth) > 0, sha256(path)


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()


def render_sample(row, maps, thresholds, folder, args, expected_mask_sha=None):
    if sha256(row["image_path"]) != row["image_sha"]:
        raise ValueError(f"Original image changed: {row['image_path']}")
    with Image.open(row["image_path"]) as image:
        if image.size != tuple(row["native_size"]):
            raise ValueError("Image geometry differs from manifest")
        size = thumbnail_size(image.size, args.max_side)
        original = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    colors, scale = display_maps(maps, thresholds, args.normalization, args.temperature,
                                  args.low_percentile, args.high_percentile)
    heatmaps = {}
    for name, values in colors.items():
        resized = np.asarray(Image.fromarray(values).resize(size, Image.Resampling.BILINEAR))
        heatmaps[name] = Image.fromarray(jet(resized))
    overlay = Image.blend(original, heatmaps["full"], args.alpha)
    prediction = Image.fromarray((maps["full"] >= thresholds["full"]).astype(np.uint8)*255).resize(size, Image.Resampling.NEAREST)
    truth, mask_sha = load_truth(row, size, not args.no_gt)
    if expected_mask_sha is not None and mask_sha != expected_mask_sha:
        raise ValueError(f"GT changed after evaluation: {row['mask_path']}")
    gt_image = original.copy()
    if truth is not None:
        pixels = np.asarray(original).copy()
        pixels[truth] = (.4*pixels[truth]+.6*np.array([255, 40, 40])).astype(np.uint8)
        gt_image = Image.fromarray(pixels)
    titles = ["Original", "GT (red annotation)" if truth is not None else "GT hidden",
              "Prediction (source threshold)", "Visual heatmap", "Full model heatmap", "Full model overlay"]
    tiles = [original, gt_image, prediction.convert("RGB"), heatmaps["visual"], heatmaps["full"], overlay]
    gap, header, footer = 10, 32, 100
    tile_width = max(size[0], 260)
    width, height = tile_width*3+gap*4, (size[1]+header)*2+gap*3+footer
    canvas = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    for i, (title, tile) in enumerate(zip(titles, tiles)):
        x, y = gap+(i % 3)*(tile_width+gap), gap+(i // 3)*(size[1]+header+gap)
        draw.text((x+4, y+6), title, fill=(20, 25, 35), font=font(15))
        canvas.paste(tile, (x+(tile_width-size[0])//2, y+header))
    y = height-footer+8
    gradient = np.repeat(np.linspace(0, 1, width-2*gap, dtype=np.float32)[None], 16, axis=0)
    canvas.paste(Image.fromarray(jet(gradient)), (gap, y))
    draw.text((gap, y+20), "LOW anomaly score", fill="black", font=font(14))
    draw.text((width-170, y+20), "HIGH anomaly score", fill="black", font=font(14))
    scale_note = ("Source CAL threshold = color midpoint; fixed across images; colors are not probabilities."
                  if args.normalization == "source" else "Per-image percentile contrast ONLY; do not compare colors across images.")
    draw.text((gap, y+43), scale_note, fill=(35, 45, 65), font=font(13))
    raw = maps["full"]
    draw.text((gap, y+64), f"{row['dataset']}/{row['category']} | full score {raw.min():.4g} .. {raw.max():.4g} | threshold {thresholds['full']:.4g}",
              fill=(35, 45, 65), font=font(13))
    folder.mkdir(parents=True, exist_ok=True)
    stem = component(row["id"])
    outputs = {"panel": stem+"_panel.jpg", "full_heatmap": stem+"_full_heatmap.png",
               "visual_heatmap": stem+"_visual_heatmap.png", "overlay": stem+"_overlay.png", "prediction": stem+"_prediction.png"}
    canvas.save(folder / outputs["panel"], quality=92)
    for name, image in (("full_heatmap", heatmaps["full"]), ("visual_heatmap", heatmaps["visual"]),
                        ("overlay", overlay), ("prediction", prediction)):
        image.save(folder / outputs[name])
    return dict(id=row["id"], dataset=row["dataset"], category=row["category"],
        image_path=row["image_path"], is_anomaly=int(bool(row["mask_path"])),
        native_size=row["native_size"], rendered_size=list(size), scale=scale,
        annotation_kind="bounding_box" if row["dataset"] == "ld" else "as_provided_mask",
        mask_sha256=mask_sha, full_min=float(raw.min()), full_max=float(raw.max()),
        full_threshold=thresholds["full"], predicted_pixels=int((raw >= thresholds["full"]).sum()), files=outputs)


def make_gallery(output, records):
    datasets = [name for name in TARGETS if any(r["dataset"] == name for r in records)]
    content = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Four-dataset heatmaps</title>',
        '<style>body{font:16px system-ui;margin:28px;background:#f5f7fa;color:#192333}article{background:white;margin:24px 0;padding:16px;border-radius:12px}img{max-width:100%;height:auto}a{margin-right:18px}select{font:inherit;padding:8px}[hidden]{display:none!important}</style>',
        '<h1>Four-dataset anomaly heatmaps</h1>',
        '<p>Existing locked predictions. Heatmap colors are display values, not anomaly probabilities. LD ground truth is box-derived.</p>',
        '<label>Dataset / 数据集 <select id="galleryDataset">'+''.join(f'<option value="{name}">{name.upper()}</option>' for name in datasets)+'</select></label>',
        '<p id="galleryCount"></p>']
    for row in records:
        files = row["files"]
        title = html.escape(f"{row['dataset']}/{row['category']} | {Path(row['image_path']).name}")
        hidden = '' if row["dataset"] == datasets[0] else ' hidden'
        content += [f'<article data-dataset="{row["dataset"]}"{hidden}><h2>{title}</h2><img loading="lazy" src="{quote(files["panel"])}" alt="{title}"><p>']
        for key in ("full_heatmap", "visual_heatmap", "overlay", "prediction"):
            content.append(f'<a href="{quote(files[key])}">{key}</a>')
        content.append('</p></article>')
    content.append('''<script>
const selector=document.getElementById('galleryDataset');
if([...selector.options].some(option=>option.value===location.hash.slice(1)))selector.value=location.hash.slice(1);
function showDataset(){let count=0;document.querySelectorAll('article[data-dataset]').forEach(el=>{
el.hidden=el.dataset.dataset!==selector.value;if(!el.hidden)count++;});
document.getElementById('galleryCount').textContent=selector.value.toUpperCase()+' · '+count+' sampled images / 抽样图片';}
selector.addEventListener('change',showDataset);showDataset();
</script></html>''')
    (output / "index.html").write_text("\n".join(content), encoding="utf-8")


def run(args):
    if args.seed < 0 or (args.until_epoch is not None and args.until_epoch < 1):
        raise ValueError("Invalid seed/horizon")
    if args.max_side < 64 or not 0 < args.alpha <= 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Invalid image size, overlay alpha or temperature")
    if not 0 <= args.low_percentile < args.high_percentile <= 100 or args.limit_per_category < 0:
        raise ValueError("Invalid percentile range or image limit")
    cfg, rows, predictions, lock, thresholds, until = load_run(args.work_dir, args.seed, args.until_epoch)
    eligible, selected, sampling = select_samples(rows, args)
    output = Path(args.output_dir).resolve() if args.output_dir else Path(args.work_dir).resolve() / "heatmaps" / f"seed{args.seed}" / f"through{until}" / output_variant(args)
    # The renderer only writes inside heatmaps, never to sealed model/data folders.
    if output in (Path(args.work_dir).resolve(), predictions.resolve()) or output.is_relative_to(predictions.resolve()):
        raise ValueError("Use a separate heatmap output directory")
    settings = dict(renderer_version=RENDERER_VERSION, renderer_sha256=sha256(__file__),
        prediction_lock_sha256=sha256(predictions / "locked.json"), normalization=args.normalization,
        temperature=args.temperature, percentiles=[args.low_percentile, args.high_percentile],
        alpha=args.alpha, max_side=args.max_side, show_gt=not args.no_gt)
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "render_config.json"
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError("Heatmap style/predictions changed; choose a new OUTPUT_DIR")
    write_json(settings_path, settings)
    print(f"Uniform heatmap sample: {len(selected)}/{len(eligible)} eligible images (requested {args.sample_fraction:.0%}; small strata round up).", flush=True)
    records = []
    audit_cache = {}
    for index, row in enumerate(selected, 1):
        maps = load_scores(row, cfg, predictions, lock)
        if sha256(row["image_path"]) != row["image_sha"]:
            raise ValueError(f"Original image changed: {row['image_path']}")
        key = row["dataset"], row["category"]
        if key not in audit_cache:
            metrics = Path(args.work_dir) / "metrics" / f"seed{args.seed}" / f"through{until}" / ("__".join(key)+".json")
            audit_cache[key] = read_json(metrics).get("evaluation_mask_hashes", {}) if metrics.is_file() else {}
        folder = output / key[0] / key[1]
        metadata = folder / (component(row["id"])+".json")
        if metadata.is_file():
            record = read_json(metadata)
            if not args.no_gt and row["mask_path"] and sha256(row["mask_path"]) != record["mask_sha256"]:
                raise ValueError(f"GT changed since the previous heatmap render: {row['mask_path']}")
            if record["prediction_sha256"] != lock["predictions"][row["id"]] or record["render_fingerprint"] != fingerprint(settings):
                raise ValueError("Previously rendered image belongs to different predictions/style")
            if any(not (output / name).resolve().is_relative_to(output) or not (output / name).is_file() or sha256(output / name) != expected
                   for name, expected in record["output_sha256"].items()):
                raise ValueError("Rendered image changed or is missing; use a new OUTPUT_DIR")
        else:
            record = render_sample(row, maps, thresholds, folder, args,
                                   audit_cache[key].get(row["id"]) if not args.no_gt else None)
            record["files"] = {key: (folder / name).relative_to(output).as_posix() for key, name in record["files"].items()}
            record.update(prediction_sha256=lock["predictions"][row["id"]], render_fingerprint=fingerprint(settings))
            record["output_sha256"] = {name: sha256(output / name) for name in record["files"].values()}
            write_json(metadata, record)
        records.append(record)
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"Heatmaps {index}/{len(selected)}: {row['dataset']}/{row['category']}", flush=True)
    make_gallery(output, records)
    write_json(output / "sampling.json", sampling)
    write_json(output / "index.json", records)
    with (output / "index.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["id", "dataset", "category", "image_path", "is_anomaly", "full_min", "full_max", "full_threshold", "predicted_pixels"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    if args.pack:
        files = [output / name for name in ("render_config.json", "index.html", "index.json", "index.csv", "sampling.json")]
        for row in records:
            files += [output / name for name in row["files"].values()]
            files.append(output / row["dataset"] / row["category"] / (row["id"]+".json"))
        archive_path = pack_assets(output, "heatmaps.zip", files)
        print(f"Download ZIP: {archive_path}", flush=True)
    print(f"Generated {len(records)} image sets. Gallery: {output / 'index.html'}", flush=True)
    return output, records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, default=Path("checkpoint/residual_hr_text_four_datasets_v1"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--until-epoch", type=int)
    p.add_argument("--datasets", default=",".join(TARGETS))
    p.add_argument("--categories", default="")
    p.add_argument("--sample-fraction", type=float, default=.1,
                   help="Fraction rendered per dataset/category/image-label stratum (default: 0.1)")
    p.add_argument("--limit-per-category", type=int, default=0)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--alpha", type=float, default=.45)
    p.add_argument("--normalization", choices=("source", "percentile"), default="source")
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--low-percentile", type=float, default=1.)
    p.add_argument("--high-percentile", type=float, default=99.)
    p.add_argument("--no-gt", action="store_true")
    p.add_argument("--pack", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
