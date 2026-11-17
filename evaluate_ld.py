#!/usr/bin/env python3
"""Frozen LD evaluation using the existing CLIP+DINO and MARA implementations.

No training, architecture changes, target checkpoint selection, or VLM calls.
Image labels remain independent of resized masks. Pixel truth is box-derived.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from tools.prepare_ld_dataset import TASKS, sha256, validate
from tools.vlm_review import atomic_json, digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "ld_frozen_clip_dino_mara_v1"
METRICS = ("I_AUROC", "P_AUROC", "PRO", "F1", "background_FPR_at_05")


def checkpoint_mode(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected an existing model checkpoint dictionary")
    if payload.get("base_arch", "clip_dino") != "clip_dino" or "dino_single_config" in payload:
        raise ValueError("This entry preserves CLIP+DINO; a single-tower checkpoint is incompatible")
    if payload.get("visual_backbone", "dino") != "dino":
        raise ValueError("Expected the current DINO backbone with CLIP text")
    for name in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter", "prompt_learner"):
        if name not in payload:
            raise ValueError(f"Checkpoint is missing {name}; Flow/text-only heads are not Base/MARA checkpoints")
    return "mara" if "mara_agent" in payload or "mara_ppo_agent" in payload else "base"


def resolve_checkpoint(value, mode="auto"):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        choices = []
        for candidate in path.glob("*.pth"):
            match = re.fullmatch(r"(?:(mara_epoch|epoch)_)?(\d+)", candidate.stem)
            if match:
                family = "mara" if match[1] == "mara_epoch" else "base"
                if mode == "auto" or family == mode:
                    choices.append((family, int(match[2]), candidate))
        if not choices:
            raise ValueError(f"No numeric checkpoint epochs for mode={mode}: {path}; pass an explicit file")
        if len({family for family, _, _ in choices}) > 1:
            raise ValueError("Mixed Base/MARA checkpoint directory; specify MODE or an explicit file")
        epoch = max(item[1] for item in choices)
        latest = [p for _, number, p in choices if number == epoch]
        if len(latest) != 1:
            raise ValueError("Ambiguous epoch filenames; pass the exact checkpoint file")
        path = latest[0]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def ensure_data(root, archive):
    """Optionally extract only the data tree from the previously supplied bundle."""
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if (root / "manifest.json").is_file():
        return
    if root.exists():
        raise ValueError(f"Partial/unrecognized data directory: {root}; do not overwrite it")
    if not archive.is_file():
        raise FileNotFoundError(f"Upload/extract the LD bundle to {root.parent}, or set LD_DATA_ROOT")
    sidecar = archive.with_name(archive.name + ".sha256")
    if sidecar.is_file() and sha256(archive) != sidecar.read_text(encoding="ascii").split()[0]:
        raise ValueError("Dataset archive SHA256 mismatch; upload it again")
    root.parent.mkdir(parents=True, exist_ok=True)
    prefix = "LD_AD_20261116"
    with tarfile.open(archive, "r:gz") as stream:
        members = []
        for member in stream.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != prefix:
                continue
            relative = Path(*parts[1:])
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError("Invalid data archive entry")
            members.append((member, target))
        if not any(target == root / "manifest.json" for _, target in members):
            raise ValueError("Archive has no LD manifest")
        for member, target in members:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("wb") as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
    print(f"Extracted LD data: {root}", flush=True)


def prepare(args):
    root, work = Path(args.data_root).resolve(), Path(args.work_dir).resolve()
    ensure_data(root, args.archive)
    checkpoint = resolve_checkpoint(args.weight_path, args.mode)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mode = checkpoint_mode(payload)
    if args.mode != "auto" and args.mode != mode:
        raise ValueError(f"Requested mode={args.mode}, but checkpoint contains {mode}")
    for path in (args.dino_weights, args.clip_weights):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Existing local pretrained weights required: {path}")
    if not Path(args.dino_repo_dir, "hubconf.py").is_file():
        raise FileNotFoundError("DINO_REPO_DIR must contain hubconf.py")
    rows = load_json(root / "manifest.json") if args.skip_disk_audit else validate(root)
    selected = sorted((r for r in rows if r["split"] == "test"),
                      key=lambda r: (r["category"], r["is_anomaly"], r["image_path"]))
    if {r["category"] for r in selected} != set(TASKS.values()):
        raise ValueError("All four LD TEST categories are required")
    if args.image_size < 32 or args.image_size % 16 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid image size, batch size, or workers")
    if not 0 < args.threshold < 1 or args.pro_num_th < 2 or args.vis_per_category < 0:
        raise ValueError("Invalid metric/visualization settings")
    # Separate file locations from architecture metadata; CLI paths are relocatable.
    settings = {k: str(getattr(args, k)) if isinstance(getattr(args, k), Path) else getattr(args, k)
                for k in ("image_size", "hfa_setting", "dino_model_name", "threshold", "pro_num_th",
                          "no_vis", "vis_per_category", "seed")}
    settings.update(dino_repo_dir=str(Path(args.dino_repo_dir).resolve()),
                    dino_weights=str(Path(args.dino_weights).resolve()),
                    clip_weights=str(Path(args.clip_weights).resolve()))
    cfg = dict(protocol=PROTOCOL, mode=mode, base_arch="clip_dino", data_root=str(root),
               weight_path=str(checkpoint), weight_sha256=sha256(checkpoint),
               dino_sha256=sha256(args.dino_weights), clip_sha256=sha256(args.clip_weights),
               manifest_sha256=sha256(root / "manifest.json"), rows=selected, settings=settings,
               code={name: sha256(REPO / name) for name in (
                   "evaluate_ld.py", "run_test_ld.sh", "Datasets/ld.py", "test_mara.py", "test2.py",
                   "dual_vlm.py", "train_mara.py", "train_up.py", "tools/utils_up.py", "tools/mara_agent.py")},
               annotation_kind="bounding_box", normalization="none", image_score="maximum_pixel_probability")
    work.mkdir(parents=True, exist_ok=True)
    path = work / "config.json"
    if path.exists() and load_json(path) != cfg:
        raise ValueError("WORK_DIR settings/checkpoint/data changed; use a new WORK_DIR")
    atomic_json(path, cfg)
    print(f"Frozen {mode} checkpoint: {checkpoint}; TEST images={len(selected)}", flush=True)
    return cfg, payload


class ExistingNetwork:
    """Delegate model construction and inference to the existing implementation."""
    def __init__(self, cfg, payload, device):
        self.device, self.cfg = device, cfg
        from tools.utils_up import get_anomaly_map
        self.get_map = get_anomaly_map
        s = cfg["settings"]
        self.args = argparse.Namespace(
            image_size=s["image_size"], dino_repo_dir=s["dino_repo_dir"],
            dino_model_name=s["dino_model_name"], dino_weights=s["dino_weights"],
            clip_model_name="ViT-L-14-336", clip_pretrained=s["clip_weights"],
            visual_layers=(5, 11, 17, 23), visual_backbone="dino", text_source="prompt_learner",
            hfa_setting=s["hfa_setting"], hfa_layers="", hfa_bottleneck=None, dino_bottleneck=256,
            gain_safety_margin=None, gain_accept_probability=None, disable_hard_gain_gate=False,
            mara_map_size=128, mara_regions=16, mara_roi_size=32, mara_hidden_dim=64,
            mara_steps=3,
        )
        relocated = dict(payload, dino_repo_dir=s["dino_repo_dir"], dino_weights=s["dino_weights"])
        if cfg["mode"] == "base":
            from train_mara import apply_base_runtime_config, create_visual_backbone_for_mara
            from train_up import create_clip_model, create_prompt_learner, create_adapter_model
            apply_base_runtime_config(self.args, relocated)
            self.clip = create_clip_model(self.args, device)
            self.prompt = create_prompt_learner(self.clip, device)
            self.dino, hfa = create_visual_backbone_for_mara(self.args, device)
            self.adapter = create_adapter_model(self.clip, device, "dino")
            for key in ("cls_token_adapter", "patch_token_adapter", "prompt_adapter"):
                getattr(self.adapter, key).load_state_dict(payload[key], strict=True)
            self.prompt.load_state_dict(payload["prompt_learner"], strict=True)
            if hfa is not None:
                hfa.load_state_dict(payload["dino_adapters"], strict=True)
            self.agent = None
        else:
            from test_mara import build_models_from_checkpoint, load_mara_checkpoint
            self.clip, self.prompt, self.dino, hfa, self.adapter, self.agent, single = build_models_from_checkpoint(
                self.args, device, relocated)
            if single is not None:
                raise ValueError("Checkpoint unexpectedly changed to single-tower architecture")
            # Existing loaders use relaxed legacy loads: explicitly require all supplied
            # trained modules to match before reusing that loader.
            for key, module in (("cls_token_adapter", self.adapter.cls_token_adapter),
                                ("patch_token_adapter", self.adapter.patch_token_adapter),
                                ("prompt_adapter", self.adapter.prompt_adapter), ("prompt_learner", self.prompt)):
                module.load_state_dict(payload[key], strict=True)
            if hfa is not None:
                hfa.load_state_dict(payload["dino_adapters"], strict=True)
            agent_key = "mara_ppo_agent" if payload.get("mara_algorithm") in ("mara_ppo_v1", "mara_ppo_locsafe_v1") else "mara_agent"
            self.agent.load_state_dict(payload[agent_key], strict=True)
            load_mara_checkpoint(cfg["weight_path"], relocated, self.adapter, self.prompt,
                                 hfa, self.agent, device)
        for module in (self.clip, self.prompt, self.dino, hfa, self.adapter, self.agent):
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.runtime = dict(visual_layers=list(self.args.visual_layers),
                            hfa_layers=list(getattr(self.args, "hfa_layers_runtime", [])),
                            mara_algorithm=payload.get("mara_algorithm", "legacy" if self.agent is not None else None))
        if self.agent is not None:
            from dataclasses import asdict
            self.runtime["mara_config"] = asdict(self.agent.cfg)

    @torch.no_grad()
    def __call__(self, batch, index):
        # The existing function expects mask/label fields, but they are output-only.
        # Supply zeros so actual TEST labels cannot enter either model branch.
        model_input = dict(image=batch["image"], image_path=batch["image_path"],
                           mask=torch.zeros_like(batch["mask"]), is_anomaly=torch.zeros_like(batch["is_anomaly"]))
        values = self.get_map(self.clip, model_input, self.device, self.adapter, self.dino, self.prompt,
                              index, visual_backbone="dino", visual_layers=self.args.visual_layers,
                              text_source="prompt_learner", return_evidence=self.agent is not None)
        _, _, probability, logits, *extra = values
        result = {"base": probability[:, 1].detach().cpu().numpy()}
        if self.agent is not None:
            from tools.mara_evidence import build_mara_evidence
            evidence = build_mara_evidence(
                evidence=extra[0], fallback_prob=probability, num_layers=len(self.args.visual_layers),
                map_size=self.agent.cfg.map_size, include_full_resolution_oracle=False,
                feature_channels_per_layer=self.args.evidence_feature_channels,
                semantic_spatial_channels=self.args.evidence_semantic_channels,
                semantic_global_dim=self.args.evidence_semantic_global_dim)
            bank = self.agent.cfg.evidence_channels > 0
            output = self.agent.infer(base_prob=probability, base_logits=logits,
                layer_maps=evidence["layer_maps"], evidence_maps=evidence["extra_maps"] if bank else None,
                global_evidence=evidence["global_evidence"] if bank else None, sample=False)
            result["mara"] = output["final_prob"][:, 1].detach().cpu().numpy()
        return result


def read_prediction(path, cfg, rows):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data["fingerprint"]) != digest(cfg) or list(data["paths"]) != [r["image_path"] for r in rows]:
        raise ValueError(f"Prediction identity differs: {path}")
    if not np.array_equal(data["labels"], [r["is_anomaly"] for r in rows]):
        raise ValueError(f"Prediction labels differ from manifest: {path}")
    shape = (len(rows), cfg["settings"]["image_size"], cfg["settings"]["image_size"])
    if data["masks"].shape != shape:
        raise ValueError(f"Unexpected cached mask shape: {path}")
    for name in ("base", "mara") if cfg["mode"] == "mara" else ("base",):
        if data[name].shape != shape or not np.isfinite(data[name]).all() or data[name].min() < 0 or data[name].max() > 1:
            raise ValueError(f"Invalid probability maps: {path}/{name}")
    return data


def infer(cfg, args, payload, predictor=None):
    from Datasets.ld import Dataset, DatasetSplit
    device = torch.device(args.device)
    if predictor is None and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the original GPU training environment")
    torch.manual_seed(cfg["settings"]["seed"])
    work, root = Path(args.work_dir), Path(cfg["data_root"])
    out = work / "predictions"
    out.mkdir(exist_ok=True)
    if predictor is None:
        predictor = ExistingNetwork(cfg, payload, device)
        atomic_json(work / "runtime.json", predictor.runtime)
    hashes = {}
    start = time.monotonic()
    for category in sorted(TASKS.values()):
        rows = [r for r in cfg["rows"] if r["category"] == category]
        path = out / f"{category}.npz"
        if path.exists():
            read_prediction(path, cfg, rows)
            print(f"Resume {category}: verified completed predictions", flush=True)
        else:
            if (work / "prediction_lock.json").exists():
                raise ValueError("Locked predictions are incomplete; use a new WORK_DIR")
            dataset = Dataset(root, category, resize=cfg["settings"]["image_size"],
                              imagesize=cfg["settings"]["image_size"], split=DatasetSplit.TEST)
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                                 shuffle=False, pin_memory=device.type == "cuda")
            maps, masks, labels, paths = {}, [], [], []
            for index, batch in enumerate(loader):
                result = predictor(batch, index)
                for name, values in result.items():
                    maps.setdefault(name, []).append(np.asarray(values, np.float32))
                masks.append(batch["mask"][:, 0].numpy() > .5)
                labels.extend(batch["is_anomaly"].tolist())
                paths.extend(Path(p).relative_to(root).as_posix() for p in batch["image_path"])
                print(f"{category}: {len(paths)}/{len(dataset)}", flush=True)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, fingerprint=np.asarray(digest(cfg)),
                    paths=np.asarray(paths), labels=np.asarray(labels, np.uint8), masks=np.concatenate(masks),
                    **{name: np.concatenate(values) for name, values in maps.items()})
            temporary.replace(path)
            read_prediction(path, cfg, rows)
        hashes[category] = sha256(path)
    lock = dict(fingerprint=digest(cfg), predictions=hashes)
    lock_path = work / "prediction_lock.json"
    if lock_path.exists() and load_json(lock_path) != lock:
        raise ValueError("Completed prediction lock changed")
    atomic_json(lock_path, lock)
    print(f"All predictions locked; inference stage elapsed {time.monotonic()-start:.1f}s", flush=True)


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def visualize(work, root, category, rows, data, settings):
    if settings["no_vis"]:
        return
    folder = work / "visualizations" / category
    folder.mkdir(parents=True, exist_ok=True)
    selected = sorted(range(len(rows)), key=lambda i: digest([settings["seed"], rows[i]["image_path"]]))
    size = 256
    for i in selected[:settings["vis_per_category"]]:
        with Image.open(root / rows[i]["image_path"]) as image:
            rgb = image.convert("RGB").resize((size, size))
        tiles = [("image", rgb), ("box-derived GT", Image.fromarray(data["masks"][i].astype(np.uint8)*255).resize(
            (size, size), Image.Resampling.NEAREST).convert("RGB"))]
        for name in ("base", "mara"):
            if name not in data:
                continue
            score = np.asarray(Image.fromarray(data[name][i]).resize((size, size), Image.Resampling.BILINEAR))
            color = np.stack([score, .15*(1-score), 1-score], -1)*255
            overlay = (.6*np.asarray(rgb)+.4*color).clip(0, 255).astype(np.uint8)
            tiles.append((name + " (fixed 0..1)", Image.fromarray(overlay)))
        canvas = Image.new("RGB", (size*len(tiles), size+25), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (title, tile) in enumerate(tiles):
            draw.text((col*size+4, 4), title, fill="black")
            canvas.paste(tile, (col*size, 25))
        canvas.save(folder / (digest(rows[i]["image_path"])[:16]+".png"))


def report(cfg, args):
    from tools.mara_diagnostics import map_metrics, binary_metrics
    work, root = Path(args.work_dir), Path(cfg["data_root"])
    lock = load_json(work / "prediction_lock.json")
    if lock["fingerprint"] != digest(cfg):
        raise ValueError("Prediction lock does not match config")
    rows_out, images_out, pooled = [], [], {}
    for category in sorted(TASKS.values()):
        rows = [r for r in cfg["rows"] if r["category"] == category]
        path = work / "predictions" / f"{category}.npz"
        if sha256(path) != lock["predictions"][category]:
            raise ValueError(f"Locked predictions were modified: {category}")
        data = read_prediction(path, cfg, rows)
        visualize(work, root, category, rows, data, cfg["settings"])
        for name in ("base", "mara") if cfg["mode"] == "mara" else ("base",):
            labels, maps = data["labels"], data[name]
            point = map_metrics(data["masks"], maps, labels, cfg["settings"]["pro_num_th"], .3)
            counts = dict(images=len(rows), normal_images=int((labels == 0).sum()), abnormal_images=int(labels.sum()))
            rows_out.append(dict(category=category, model=name, **counts, **point))
            scores = maps.reshape(len(rows), -1).max(1)
            pooled.setdefault(name, []).append((labels, scores))
            for row, label, score in zip(rows, labels, scores):
                images_out.append(dict(category=category, model=name, image_path=row["image_path"],
                    is_anomaly=int(label), image_score=float(score),
                    predicted_anomaly=int(score >= cfg["settings"]["threshold"])))
            print(f"{category}/{name}: {point}", flush=True)
    summary = {}
    for name, chunks in pooled.items():
        records = [r for r in rows_out if r["model"] == name]
        macro = {}
        for metric in METRICS:
            valid = [r[metric] for r in records if r[metric] is not None]
            macro[metric] = dict(value=float(np.mean(valid)) if valid else None, valid_categories=len(valid))
        labels, scores = (np.concatenate([c[i] for c in chunks]) for i in (0, 1))
        summary[name] = dict(macro=macro, pooled_image_AUROC=binary_metrics(labels, scores)["AUROC"], images=len(labels))
    value = dict(protocol=PROTOCOL, checkpoint=cfg["weight_path"], checkpoint_sha256=cfg["weight_sha256"],
                 annotation_kind="bounding_box", image_labels="dataset manifest, independent of resized mask",
                 threshold=cfg["settings"]["threshold"],
                 notes=["Pixel metrics use box-derived proxy masks, not exact segmentation truth.",
                        "Single-label image AUROC is null and excluded from valid-category macro averages.",
                        "F1 is best pixel F1 on the test set, not a deployment threshold.",
                        "A fixed 0.5 background FPR is diagnostic; no target threshold tuning or per-image normalization."],
                 categories=rows_out, summary=summary)
    atomic_json(work / "metrics.json", value)
    write_csv(work / "metrics.csv", rows_out)
    write_csv(work / "image_scores.csv", images_out)
    lines = ["# LD frozen-model evaluation", "", f"Checkpoint: `{cfg['weight_path']}`", "",
             "Pixel metrics use box-derived masks. N/A image AUROC is excluded from the macro mean.", "",
             "| Category | Model | Normal | Abnormal | I-AUROC | P-AUROC | PRO | Best pixel F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows_out:
        values = ["N/A" if row[key] is None else f"{row[key]:.5f}" for key in ("I_AUROC", "P_AUROC", "PRO", "F1")]
        lines.append("| "+" | ".join([row["category"],row["model"],str(row["normal_images"]),str(row["abnormal_images"])]+values)+" |")
    for name, result in summary.items():
        lines += ["", f"## {name} summary", "", f"Pooled image AUROC: {result['pooled_image_AUROC']}"]
        for metric, item in result["macro"].items():
            lines.append(f"- Macro {metric}: {item['value']} ({item['valid_categories']} valid categories)")
    (work / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    archive = work / "ld_test_report.zip"
    files = [work / name for name in ("config.json", "runtime.json", "prediction_lock.json", "metrics.json",
                                     "metrics.csv", "image_scores.csv", "REPORT.md", "run.log")]
    files += sorted((work / "visualizations").rglob("*.png"))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            if path.is_file():
                stream.write(path, path.relative_to(work).as_posix())
    print(f"Report: {work / 'REPORT.md'}\nSmall report ZIP: {archive}", flush=True)
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weight-path", required=True)
    p.add_argument("--mode", choices=("auto", "base", "mara"), default="auto")
    p.add_argument("--data-root", type=Path, default=Path("/mnt/qfg/Tate_qfg/data/LD_AD_20261116"))
    p.add_argument("--work-dir", type=Path, default=Path("results") / ("ld_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    p.add_argument("--archive", type=Path, default=Path("/mnt/qfg/Tate_qfg/data/LD_AD_20261116_bundle.tar.gz"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--hfa-setting", default="hfa3")
    p.add_argument("--dino-repo-dir", type=Path, default=Path("dinov3"))
    p.add_argument("--dino-model-name", default="dinov3_vitl16")
    p.add_argument("--dino-weights", type=Path, default=Path("dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    p.add_argument("--clip-weights", type=Path, default=Path("CLIP/ckpt/ViT-L-14-336px.pt"))
    p.add_argument("--threshold", type=float, default=.5)
    p.add_argument("--pro-num-th", type=int, default=1000)
    p.add_argument("--vis-per-category", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--skip-disk-audit", action="store_true")
    p.add_argument("--check-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    cfg, payload = prepare(args)
    if args.check_only:
        print("Preflight passed. Checkpoint, local weights and dataset validated; model inference was not run.")
        return
    infer(cfg, args, payload)
    report(cfg, args)


if __name__ == "__main__":
    main()
