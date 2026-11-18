#!/usr/bin/env python3
"""Fresh residual/Flow/HR/text model, VisA source -> MVTec/BTAD/MPDD/LD.

Reuses the existing D-arm losses, models, source-only selection and inference.
The legacy sealed experiments are never edited. Only residual features are kept.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

import factorial_ablation_experiment as f
import residual_text_experiment as base
from tools.residual_text_runtime import (
    FrozenDinoExtractor, atomic_npz, export_residuals, make_record, normalized_boxes, unchanged,
)
from tools.text_ablation import TextMatcher, MODES, corrected_metrics, evaluation_truth
from tools.vlm_review import atomic_json, digest, file_digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "fresh_residual_hr_text_four_targets_v1"
TARGETS = ("mvtec", "btad", "mpdd", "ld")
CODE = tuple(dict.fromkeys(f.CODE + ("train_test_four_datasets.py", "run_train_test_four_datasets.sh",
                                    "tools/text_ablation.py", "Datasets/__init__.py", "Datasets/ld.py", "evaluate_ld.py")))
NATIVE_NEW_MODEL = f.new_model


def new_model(root, cfg, kind, seed, device):
    if kind == "density":
        return NATIVE_NEW_MODEL(root, cfg, kind, seed, device)
    path = Path(root) / "text_bank.npz"
    if load_json(path.with_suffix(".json")) != {"fingerprint": digest(cfg), "sha": file_digest(path)}:
        raise ValueError("Frozen text bank changed")
    with np.load(path, allow_pickle=False) as archive:
        states = archive["paired_states"].copy()
    return TextMatcher(len(cfg["layers"]), cfg["feature_dim"], states,
                       MODES[cfg["text_readout"]], cfg["hidden"]).to(device)


@contextmanager
def runtime():
    """Scoped single-D execution; old files and module state remain unchanged."""
    overrides = dict(PROTOCOL=PROTOCOL, ARMS={"D": ("residual", "HR")},
                     MAPS=("D_visual", "D_full"), CODE=CODE, new_model=new_model,
                     evaluation_mask=lambda row, size: evaluation_truth(row, size)[0])
    previous = {key: getattr(f, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(f, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(f, key, value)


def catalog(args):
    from Datasets import DATASET_CLASSES, DATASET_REGISTRY
    targets = args.eval_datasets.replace(",", " ").split()
    if args.source != "visa" or len(targets) != 4 or set(targets) != set(TARGETS):
        raise ValueError("This run trains on VisA and tests exactly mvtec,btad,mpdd,ld")
    roots = json.loads(args.data_roots)
    if not isinstance(roots, dict) or not set(roots) <= set(DATASET_REGISTRY):
        raise ValueError("DATA_ROOTS_JSON must map registered datasets to directories")
    if args.target_root:
        raise ValueError("Use DATA_ROOTS_JSON / LD_DATA_ROOT for four different target roots")
    if args.source_root:
        roots["visa"] = args.source_root
    queries, candidates = [], []
    for name in ("visa", *TARGETS):
        cls, splits, default = DATASET_REGISTRY[name]
        for category in sorted(DATASET_CLASSES[name]):
            for split, records in ((splits.TEST, queries), (splits.TRAIN, candidates)):
                dataset = cls(source=roots.get(name, default), classname=category, split=split,
                              resize=args.image_size, imagesize=args.image_size)
                entries = sorted(dataset.data_to_iterate,
                                 key=lambda e: digest([args.data_seed, name, category, str(e[2])]))
                if records is queries and args.limit_per_category:
                    entries = entries[:args.limit_per_category]
                for _, state, image, mask in entries:
                    if records is candidates and (mask or str(state).lower() not in ("normal", "good", "ok")):
                        continue
                    if mask and not Path(mask).is_file():
                        raise FileNotFoundError(f"Annotation file missing: {mask}")
                    partition = "reference" if records is candidates else "source" if name == "visa" else "eval"
                    records.append(make_record(name, category, image, mask, partition))
            print(f"Catalog: {name}/{category}", flush=True)
    return queries, candidates


def prepare(args, supplied_catalog=None):
    if args.arm != "D" or args.semantic_annotations:
        raise ValueError("Use the existing residual/HR D structure without additional semantic annotations")
    records = copy.deepcopy(supplied_catalog if supplied_catalog is not None else catalog(args))
    if {r["dataset"] for r in records[0] if r["partition"] == "eval"} != set(TARGETS):
        raise ValueError("Four complete target datasets are required")
    if {r["dataset"] for r in records[0] if r["partition"] == "source"} != {"visa"}:
        raise ValueError("Only VisA may enter source training/selection")
    with runtime(), redirect_stdout(io.StringIO()):
        cfg = f.prepare(args, records)
    size = len(records[0])*5*len(cfg["layers"])*cfg["feature_dim"]*(cfg["image_size"]//cfg["patch_size"])**2*2/2**30
    print(f"One fresh residual/Flow/HR/{cfg['text_readout']} model; K={cfg['reference_count']}; "
          f"source holdout={cfg['val_categories']}; target datasets={TARGETS}", flush=True)
    print(f"Residual-only feature estimate {size:.1f} GiB, plus model/prediction storage", flush=True)
    return cfg


@torch.inference_mode()
def export(args):
    with runtime():
        root, cfg = Path(args.work_dir), f.config(args.work_dir)
        if not 0 <= args.shard_id < cfg["num_shards"]:
            raise ValueError("Invalid export shard")
        supports = f.groups(load_json(root / "references.json"))
        extractor = None
        try:
            for index, (key, queries) in enumerate(f.groups(f.rows(root, args.partition)).items()):
                if index % cfg["num_shards"] != args.shard_id:
                    continue
                pending = [r for r in queries if not f.feature_path(root, r, "residual").with_suffix(".json").exists()]
                if not pending:
                    continue
                if (root / ("sealed_"+args.partition+".json")).exists():
                    raise ValueError("Cannot change a sealed feature cache")
                if extractor is None:
                    code = digest({str(p.relative_to(cfg["dino_repo_dir"])): file_digest(p)
                                   for p in sorted(Path(cfg["dino_repo_dir"]).rglob("*.py"))})
                    if code != cfg["backbone_code"]:
                        raise ValueError("Frozen DINO code changed")
                    extractor = FrozenDinoExtractor(cfg, base.device_for(args, cfg))
                normal_views = []
                for row in supports[key]:
                    unchanged(row)
                    with Image.open(row["image_path"]) as image:
                        views, _ = extractor.views(image.convert("RGB"))
                    normal_views.append([v.cpu() for v in views])
                for row in pending:
                    unchanged(row)
                    with Image.open(row["image_path"]) as image:
                        views, boxes = extractor.views(image.convert("RGB"))
                    values = export_residuals(extractor, views, normal_views, cfg)
                    path = f.feature_path(root, row, "residual")
                    atomic_npz(path, features=values["residual"], match_index=values["match_index"],
                               boxes=normalized_boxes(boxes, row["native_size"]), pixel_boxes=np.asarray(boxes, np.int32))
                    f.immutable(path.with_suffix(".json"), dict(id=row["id"], fingerprint=digest(cfg),
                        representation="residual", references=[r["id"] for r in supports[key]], sha=file_digest(path)))
                    print(f"export {args.partition}: {key}/{row['id']}", flush=True)
        finally:
            if extractor is not None:
                extractor.close()


def seal(args):
    with runtime():
        root, cfg = Path(args.work_dir), f.config(args.work_dir)
        hashes = {}
        for row in f.rows(root, args.partition):
            f.load_feature(root, cfg, row, "residual")
            path = f.feature_path(root, row, "residual")
            hashes["residual/"+row["id"]] = dict(sha=file_digest(path), metadata=file_digest(path.with_suffix(".json")))
        f.immutable(root / ("sealed_"+args.partition+".json"), dict(fingerprint=digest(cfg), features=hashes))


def fit_stats(args):
    with runtime():
        root, cfg = Path(args.work_dir), f.config(args.work_dir)
        sums = np.zeros((len(cfg["layers"]), 2))
        counts = sums.copy()
        for item in f.Features(root, cfg, f.calibration_rows(root, cfg), "D", True):
            x = item["residual"].numpy().astype(np.float64)
            for view in range(5):
                sums[:, int(view > 0)] += np.square(x[view]).sum(axis=(1, 2, 3))
                counts[:, int(view > 0)] += np.prod(x[view].shape[1:])
        scale = np.sqrt(sums/counts)
        if not np.isfinite(scale).all() or (scale < 1e-8).any():
            raise ValueError("Degenerate source normal RMS")
        f.immutable(root / "normal_stats.json", dict(fingerprint=digest(cfg),
            source_seal=file_digest(root / "sealed_source.json"), rms={"residual": scale.tolist()},
            calibration_ids=cfg["calibration_ids"]))


def delegate(args, stage):
    with runtime():
        if args.arm != "D":
            raise ValueError("Only the complete residual/HR D model is trained")
        return getattr(f, stage)(args)


def evaluate(args):
    with runtime():
        root, cfg = Path(args.work_dir), f.config(args.work_dir)
        until = f.horizon(args, cfg)
        out = f.prediction_dir(root, args.train_seed, until)
        lock = load_json(out / "locked.json")
        if lock["identity"] != f.prediction_identity(root, cfg, args.train_seed, until):
            raise ValueError("Stale prediction lock")
        selected = load_json(f.decision_path(root, args.train_seed, until))["arms"]["D"]
        for index, (key, queries) in enumerate(f.groups(f.rows(root, "eval")).items()):
            if index % cfg["num_shards"] != args.shard_id:
                continue
            paths = [out / (r["id"]+".npz") for r in queries]
            if any(file_digest(path) != lock["predictions"][r["id"]] for r, path in zip(queries, paths)):
                raise ValueError("Predictions changed after locking")
            for row in queries:
                unchanged(row)
            # First target-mask decoding, after every dataset prediction is locked.
            truths = [evaluation_truth(r, cfg["eval_size"]) for r in queries]
            masks = np.stack([value[0] for value in truths])
            labels = np.asarray([bool(r["mask_path"]) for r in queries])
            points = []
            for score in ("visual", "full"):
                values = []
                for path in paths:
                    with np.load(path, allow_pickle=False) as archive:
                        values.append(archive["D_"+score].copy())
                maps = np.stack(values)
                point = corrected_metrics(masks, maps, labels, selected["thresholds"][score])
                point.update(dataset=key[0], category=key[1], score=score, seed=args.train_seed,
                             normal_images=int((~labels).sum()), images=len(labels),
                             density_epoch=selected["density_epoch"], text_epoch=selected["text_epoch"],
                             annotation_kind="bounding_box" if key[0] == "ld" else "as_provided_mask")
                points.append(point)
            info = dict(prediction_lock_sha=file_digest(out / "locked.json"), metrics=points,
                        evaluation_mask_hashes={r["id"]: file_digest(r["mask_path"]) if r["mask_path"] else None for r in queries},
                        mask_audit=[value[1] for value in truths])
            f.immutable(root / "metrics" / f"seed{args.train_seed}" / f"through{until}" / ("__".join(key)+".json"), info)
            f.panels(root, cfg, queries, out, args.train_seed, until)
            print(f"evaluated {key}: images={len(labels)}, abnormal={int(labels.sum())}", flush=True)


def report(args):
    with runtime():
        root, cfg = Path(args.work_dir), f.config(args.work_dir)
        until = f.horizon(args, cfg)
        lock = f.prediction_dir(root, args.train_seed, until) / "locked.json"
        all_points = []
        for key, queries in f.groups(f.rows(root, "eval")).items():
            info = load_json(root / "metrics" / f"seed{args.train_seed}" / f"through{until}" / ("__".join(key)+".json"))
            if info["prediction_lock_sha"] != file_digest(lock):
                raise ValueError("Metrics have a different prediction lock")
            expected = {r["id"]: file_digest(r["mask_path"]) if r["mask_path"] else None for r in queries}
            if info["evaluation_mask_hashes"] != expected:
                raise ValueError("Evaluation masks changed")
            all_points.extend(info["metrics"])
        result = f.summarize(all_points, ("dataset", "score"))
        for row in result:
            points = [r for r in all_points if (r["dataset"], r["score"]) == (row["dataset"], row["score"])]
            row.update(images=sum(r["images"] for r in points), categories=len(points),
                       I_AUROC_valid_categories=sum(r["I_AUROC"] is not None for r in points),
                       annotation_kind="bounding_box" if row["dataset"] == "ld" else "as_provided_mask")
        if {r["dataset"] for r in result} != set(TARGETS):
            raise ValueError("Four-dataset evaluation is incomplete")
        folder = root / "reports"
        folder.mkdir(exist_ok=True)
        base.write_csv(folder / "summary.csv", result)
        base.write_csv(folder / "per_category.csv", all_points)
        atomic_json(folder / "summary.json", dict(protocol=PROTOCOL, text_readout=cfg["text_readout"],
            source="visa", seed=args.train_seed, horizon=until, datasets=result))
        lines = ["# One model, four target datasets", "",
            f"Fresh residual + Flow + HR + {cfg['text_readout']} text; source=VisA; epochs={until}; seed={args.train_seed}.",
            f"K={cfg['reference_count']}; local weight={cfg['high_weight']}; text fusion={cfg['text_strength']}.", "",
            "| Dataset | Branch | Images | Categories | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO | Best pixel F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(result, key=lambda r: (r["score"] != "full", TARGETS.index(r["dataset"]))):
            scores = ["N/A" if row[m] is None else f"{100*row[m]:.2f}" for m in
                      ("I_AUROC", "I_AP", "P_AUROC", "P_AP", "PRO_exact", "F1_best")]
            lines.append("| "+" | ".join([row["dataset"], row["score"], str(row["images"]), str(row["categories"])]+scores)+" |")
        lines += ["", "Values are category-macro percentages; the four datasets are not pooled into one score.",
            "LD pixel metrics use box-derived proxy masks and are not directly comparable to precise segmentation masks.",
            "Cam5 image AUROC is N/A and excluded from the LD I-AUROC macro; valid-category counts are in summary.csv.",
            "BTAD original zero masks are retained. Image labels come from the manifest, not resized foreground.",
            "Checkpoint selection and operating thresholds use source data only. Best pixel F1 is descriptive, not a deployment threshold.",
            "The three established targets have already informed development; LD is the added target, not source training data."]
        (folder / "FOUR_DATASETS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        files = [root / n for n in ("config.json", "manifest.json", "references.json", "normal_stats.json", "sealed_source.json", "sealed_eval.json")]
        for name in ("heads", "decisions", "metrics", "reports", "panels"):
            files += [p for p in (root / name).rglob("*") if p.is_file() and p.suffix in (".json", ".csv", ".md", ".png")]
        files += [lock] + list(root.glob("run*.log"))
        path = root / "four_datasets_report.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(set(files)):
                if item.is_file():
                    archive.write(item, item.relative_to(root).as_posix())
        print(f"Four-dataset table: {folder / 'FOUR_DATASETS.md'}\nReport ZIP: {path}", flush=True)
        return result


def parser():
    with runtime():
        p = f.parser()
    p.add_argument("--text_readout", choices=("T1", "T2"), default="T2")
    p.set_defaults(arm="D", source="visa", eval_datasets=",".join(TARGETS),
                   until_epoch=15, work_dir="./checkpoint/residual_hr_text_four_datasets_v1")
    for action in p._actions:
        if action.dest == "stage":
            action.choices = ("prepare", "text_bank", "export", "seal", "fit_stats", "train", "select",
                              "predict", "lock_predictions", "evaluate", "report")
    return p


def main():
    args = parser().parse_args()
    if args.stage in ("text_bank", "train", "select", "predict", "lock_predictions"):
        delegate(args, args.stage)
    else:
        globals()[args.stage](args)


if __name__ == "__main__":
    main()
