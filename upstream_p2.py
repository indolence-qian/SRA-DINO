#!/usr/bin/env python3
"""Read-only semantic quality audit -> spatial matching -> P2a/P2b controls."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import timedelta
import html
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import upstream_localization as old
import upstream_repair as repair
from tools.upstream_p2 import MatchingHead, component_loss, component_supervision, spatial_matching
from tools.upstream_repair import corrected_probability, constrained_loss, selection_score, source_metrics, aggregate_source
from tools.upstream_semantic_audit import shuffled_semantics, difference
from tools.vlm_review import atomic_json, digest, file_digest, load_json

MODES = ("visual", "real", "shuffled")
LOSSES = ("p2a", "p2b")
FILES = ("upstream_p2.py", "tools/upstream_p2.py", "run_exp_upstream_p2.sh",
         "upstream_repair.py", "tools/upstream_repair.py", "tools/upstream_semantic_audit.py",
         "tools/vlm_next_diagnostics.py")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def prepare(a):
    if not a.source_work_dir:
        raise ValueError("--source_work_dir must name the original sealed localization cache")
    root, source = Path(a.work_dir).resolve(), Path(a.source_work_dir).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source/output must be separate non-nested directories")
    cfg = old.cfg_load(source)
    seal = load_json(source/"sealed.json")
    if seal["fingerprint"] != digest(cfg):
        raise ValueError("Cache/config mismatch")
    alphas = sorted(float(v) for v in a.alphas.split(","))
    if not alphas or len(set(alphas)) != len(alphas) or any(not 0 < v <= 1 for v in alphas):
        raise ValueError("Unique finite alphas in (0,1] required; alpha=0 always included")
    positive_ints = (a.epochs, a.batch_size, a.num_shards, a.review_per_category, a.min_area,
                     a.near_radius, a.rank_pixels, a.match_panels)
    finite = (a.lr, a.fpr, a.hard_fraction, a.bg_weight, a.residual_weight, a.auc_tolerance,
              a.fpr_slack, a.component_weight, a.ranking_weight, a.small_fraction, a.small_weight, a.rank_margin)
    if (min(positive_ints) < 1 or a.hidden < 8 or a.hidden % 8 or a.val_pixels < 128 or
            not all(np.isfinite(v) for v in finite) or a.lr <= 0 or not 0 < a.fpr < .5 or
            not 0 < a.hard_fraction <= 1 or not 0 < a.small_fraction <= 1 or a.small_weight < 1 or
            min(a.bg_weight, a.residual_weight, a.auc_tolerance, a.fpr_slack, a.component_weight,
                a.ranking_weight, a.rank_margin) < 0):
        raise ValueError("Invalid training/audit settings")
    names = ("epochs", "batch_size", "hidden", "num_shards", "lr", "val_pixels", "fpr", "hard_fraction",
             "bg_weight", "residual_weight", "auc_tolerance", "fpr_slack", "review_per_category", "min_area",
             "near_radius", "rank_pixels", "component_weight", "ranking_weight", "small_fraction",
             "small_weight", "rank_margin", "match_panels")
    s = {k: getattr(a, k) for k in names}
    s.update(protocol="upstream_P2_matching_components_v1", source=str(source),
             source_config_sha=file_digest(source/"config.json"), seal_sha=file_digest(source/"sealed.json"),
             alphas=alphas, code={p: file_digest(old.REPO/p) for p in FILES})
    path = root/"p2_config.json"
    if path.exists() and load_json(path) != s:
        raise ValueError("Settings/code changed; use NEW WORK_DIR")
    if not path.exists() and any((root/p).exists() for p in ("config.json", "repair_config.json", "audit_config.json")):
        raise ValueError("Output belongs to another experiment; use NEW WORK_DIR")
    old.FeatureDataset(source, cfg, old.records_for(source))
    atomic_json(path, s)
    print(f"READ ONLY cache: {source}; new output: {root}", flush=True)
    print("Source TEST masks supervise train/val; old Base may have seen source val. Target is final evaluation only.", flush=True)


def context(a):
    root = Path(a.work_dir).resolve()
    s = load_json(root/"p2_config.json")
    for p, sha in s["code"].items():
        if file_digest(old.REPO/p) != sha:
            raise ValueError("P2 code changed; use NEW WORK_DIR")
    source = Path(s["source"])
    if (file_digest(source/"config.json") != s["source_config_sha"] or
            file_digest(source/"sealed.json") != s["seal_sha"]):
        raise ValueError("Source cache/config changed")
    return root, source, old.cfg_load(source), s


def verify_reviews(source, cfg, rows):
    for r in rows:
        _, sha = old.reviews_for(source, r, cfg)
        if load_json(source/"features"/f"{r['id']}.json") != dict(fingerprint=digest(cfg), review_sha=sha):
            raise ValueError("Sealed review/text cache changed")


def quality(a):
    root, source, cfg, s = context(a)
    rows = [r for r in old.records_for(source) if r["partition"] in ("train", "val")]
    old.FeatureDataset(source, cfg, rows)
    verify_reviews(source, cfg, rows)
    details, candidates = [], defaultdict(list)
    for r in rows:
        reviews, _ = old.reviews_for(source, r, cfg)
        gt = old.mask_for(r, cfg["image_size"])[0].numpy().astype(bool)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
        small = np.zeros_like(gt)
        for label in range(1, n):
            if stats[label, cv2.CC_STAT_AREA] <= gt.size*s["small_fraction"]:
                small |= labels == label
        with np.load(source/"features"/f"{r['id']}.npz") as data:
            embeddings, valid = data["embeddings"].astype(float), data["valid"] > 0
            truncations = int(data["truncated_texts"])
        for tile, parsed in enumerate(reviews):
            obs, exp = parsed.get("observation", ""), parsed.get("normal_expectation", "")
            x, y = embeddings[tile]
            similarity = float(np.dot(x, y)/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-12)) if valid[tile].all() else None
            x1, y1, x2, y2 = cfg["boxes"][tile]
            region = np.s_[int(y1*gt.shape[0]):int(np.ceil(y2*gt.shape[0])), int(x1*gt.shape[1]):int(np.ceil(x2*gt.shape[1]))]
            gt_pixels, small_pixels = int(gt[region].sum()), int(small[region].sum())
            kind = "small_defect" if small_pixels else "other_defect" if gt_pixels else "background_tile" if gt.any() else "normal_image"
            value = dict(partition=r["partition"], dataset=r["dataset"], category=r["category"], image_id=r["id"],
                         tile=tile, parse_ok=parsed["parse_ok"], semantic_valid=parsed.get("semantic_valid", False),
                         visibility=parsed.get("visibility", ""), status=parsed.get("status", ""),
                         observation=obs, normal_expectation=exp, observation_words=len(obs.split()),
                         expectation_words=len(exp.split()), observation_valid=bool(valid[tile, 0]),
                         expectation_valid=bool(valid[tile, 1]), observation_expectation_cosine=similarity,
                         image_truncated_texts=truncations, source_gt_pixels_DIAGNOSTIC=gt_pixels,
                         source_small_gt_pixels_DIAGNOSTIC=small_pixels, source_tile_kind_DIAGNOSTIC=kind)
            details.append(value)
            candidates[(r["dataset"], r["category"])].append((r, value))
    folder = root/"quality"
    write_csv(folder/"tiles.csv", details)
    summaries, selected = [], []
    for key, values in sorted(candidates.items()):
        observations = Counter(v["observation"].strip().casefold() for _, v in values if v["observation_valid"])
        expectations = Counter(v["normal_expectation"].strip().casefold() for _, v in values if v["expectation_valid"])
        count = sum(observations.values())
        sims = [v["observation_expectation_cosine"] for _, v in values if v["observation_expectation_cosine"] is not None]
        summaries.append(dict(dataset=key[0], category=key[1], tiles=len(values),
                              parse_ok_fraction=float(np.mean([v["parse_ok"] for _, v in values])),
                              semantic_valid_fraction=float(np.mean([v["semantic_valid"] for _, v in values])),
                              observation_unique_fraction=len(observations)/count if count else None,
                              most_common_observation_fraction=max(observations.values())/count if count else None,
                              expectation_unique_count=len(expectations), mean_role_cosine=float(np.mean(sims)) if sims else None,
                              high_role_cosine_fraction=float(np.mean(np.asarray(sims) > .95)) if sims else None,
                              truncated_texts=sum(v["image_truncated_texts"] for _, v in values if v["tile"] == 0)))
        # SOURCE GT stratifies human inspection, never VLM input/pseudo-labels.
        # Ensure small defects and normal tiles can appear alongside abstentions.
        strata = defaultdict(list)
        for item in values:
            strata[(item[1]["source_tile_kind_DIAGNOSTIC"], item[1]["status"], item[1]["semantic_valid"])].append(item)
        for group in strata.values():
            group.sort(key=lambda item: digest([cfg["seed"], item[0]["id"], item[1]["tile"]]))
        priority = {"small_defect": 0, "normal_image": 1, "other_defect": 2, "background_tile": 3}
        order = sorted(strata, key=lambda k: (priority[k[0]], k[2], k[1]))
        ordered = [item for j in range(max(map(len, strata.values()))) for group in order for item in strata[group][j:j+1]]
        selected.extend(ordered[:s["review_per_category"]])
    write_csv(folder/"summary.csv", summaries)
    annotations, cards = [], []
    for r, value in selected:
        tile = value["tile"]
        review = load_json(source/"reviews"/r["id"]/f"{tile}.json")
        sample = folder/"samples"/f"{r['id']}_{tile}"
        sample.mkdir(parents=True, exist_ok=True)
        trace_dir = source/review["attempts"][-1]["trace"]
        if not trace_dir.resolve().is_relative_to(source):
            raise ValueError("Teacher trace must stay inside source cache")
        trace = load_json(trace_dir/"trace.json") if (trace_dir/"trace.json").exists() else {}
        images = trace.get("processed_images", [])
        exact = len(images) == 2
        if exact:
            for index, item in enumerate(images):
                path = trace_dir/item["file"]
                if not path.resolve().is_relative_to(trace_dir.resolve()):
                    raise ValueError("Teacher input must stay inside trace directory")
                if file_digest(path) != item["sha256"]:
                    raise ValueError("Teacher trace image changed")
                shutil.copyfile(path, sample/f"input_{index}.png")
        else:
            with Image.open(r["image_path"]) as im:
                native = im.convert("RGB")
                overview = native.copy()
                overview.thumbnail((cfg["teacher_image_size"],)*2, Image.Resampling.LANCZOS)
                overview.save(sample/"input_0.png")
                native.crop(review["native_box"]).save(sample/"input_1.png")
        gt = old.mask_for(r, cfg["image_size"])[0].numpy()
        x1, y1, x2, y2 = cfg["boxes"][tile]
        crop_gt = gt[int(y1*gt.shape[0]):int(np.ceil(y2*gt.shape[0])), int(x1*gt.shape[1]):int(np.ceil(x2*gt.shape[1]))]
        with Image.open(sample/"input_1.png") as im:
            Image.fromarray((crop_gt*255).astype(np.uint8)).resize(im.size, Image.Resampling.NEAREST).save(sample/"source_gt_crop.png")
        atomic_json(sample/"review.json", dict(record=r, review=review, teacher_trace=trace, exact_post_vision_inputs=exact))
        # Do not overwrite a user's completed manual annotations on resume.
        rel = sample.relative_to(folder).as_posix()
        annotations.append(dict(image_id=r["id"], tile=tile, category=r["category"], sample=rel,
                                local_defect_visible="", observation_grounded="", expectation_useful="", notes=""))
        cards.append(f'<article><h3>{html.escape(r["category"])} / {html.escape(r["id"])} / tile {tile}</h3>'
                     f'<p>{"实际 post-vision 输入" if exact else "重建输入；缺少实际 post-vision trace"}</p>'
                     f'<img src="{rel}/input_0.png"><img src="{rel}/input_1.png">'
                     f'<p>源域 GT，仅供人工复查：{html.escape(value["source_tile_kind_DIAGNOSTIC"])}</p><img src="{rel}/source_gt_crop.png">'
                     f'<p>status: {html.escape(value["status"])}</p>'
                     f'<p>observation: {html.escape(value["observation"])}</p>'
                     f'<p>normal expectation: {html.escape(value["normal_expectation"])}</p>'
                     f'<details><summary>原始回答</summary><pre>{html.escape(review["attempts"][-1]["raw"])}</pre></details></article>')
    if not (folder/"manual_review.csv").exists():
        write_csv(folder/"manual_review.csv", annotations)
    (folder/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>源域语义复查</title>'
                                    '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}article{border-top:1px solid #ccc;padding:20px}img{max-width:45%;max-height:380px;margin:10px}pre{white-space:pre-wrap}</style>'
                                    '<h1>源域局部语义复查</h1><p>解析成功、重复率和余弦统计不能证明描述正确；请在 manual_review.csv 记录可见性、描述是否贴合、正常预期是否有用。统计不自动改写语义或弃权。</p>'+"".join(cards), encoding="utf-8")
    atomic_json(folder/"complete.json", dict(fingerprint=digest(s), source_only=True, reviewed_tiles=len(details),
                                             exported_samples=len(selected), human_accuracy="NOT_MEASURED"))
    print(f"quality exported: {folder/'index.html'}; accuracy requires human inspection", flush=True)


class P2Dataset(old.FeatureDataset):
    def __init__(self, source, cfg, rows, s, mode="real", components=False, verify=True):
        super().__init__(source, cfg, rows, verify=verify)
        self.mode, self.s, self.components = mode, s, components
        self.bank, self.audit, self.donors = {}, [], []
        if mode == "shuffled":
            groups = defaultdict(list)
            for r in rows:
                groups[(r["partition"], r["dataset"], r["category"])].append(r)
            for key, group in sorted(groups.items()):
                emb, valid = [], []
                for r in group:
                    with np.load(source/"features"/f"{r['id']}.npz") as data:
                        emb.append(data["embeddings"].astype(np.float32))
                        valid.append(data["valid"].astype(np.float32))
                seed = int(digest([cfg["seed"], list(key), "P2_donor"])[:8], 16)
                bank, audit, donors = shuffled_semantics(np.asarray(emb), np.asarray(valid), [r["id"] for r in group], seed)
                self.bank.update({r["id"]: torch.from_numpy(e) for r, e in zip(group, bank)})
                self.audit.extend(dict(partition=key[0], dataset=key[1], category=key[2], image_id=r["id"], **v) for r, v in zip(group, audit))
                self.donors.extend(dict(partition=key[0], dataset=key[1], category=key[2], **v) for v in donors)
        self.component_cache = {}

    def __getitem__(self, i):
        batch = super().__getitem__(i)
        if self.mode == "shuffled":
            batch["embeddings"] = self.bank[self.rows[i]["id"]]
        if self.components:
            if i not in self.component_cache:
                self.component_cache[i] = component_supervision(batch["mask"][0].numpy(), self.s["min_area"])
            batch["components"] = self.component_cache[i]
        return batch


def batches_for(source, cfg, rows, s, mode, workers, verify=True):
    return DataLoader(P2Dataset(source, cfg, rows, s, mode, verify=verify), batch_size=s["batch_size"], num_workers=workers)


def forward(head, batch, mode):
    return head(*(batch[k] for k in ("visual", "base", "embeddings", "valid", "boxes")), use_semantics=mode != "visual")


def require_stage(root, name, s):
    if load_json(root/name/"complete.json")["fingerprint"] != digest(s):
        raise ValueError(f"Stale {name} stage")


def matching(a):
    from sklearn.metrics import roc_auc_score
    root, source, cfg, s = context(a)
    require_stage(root, "quality", s)
    device = repair.device_for()
    summaries, audit, donors = [], [], []
    groups = repair.groups_for(source, {"val"})
    for key, rows in groups.items():
        verify_reviews(source, cfg, rows)
        real_ds = P2Dataset(source, cfg, rows, s)
        shuffle_ds = P2Dataset(source, cfg, rows, s, "shuffled")
        audit.extend(shuffle_ds.audit)
        donors.extend(shuffle_ds.donors)
        values = defaultdict(list)
        for index, r in enumerate(rows):
            batch = {k: v[None].to(device) for k, v in real_ds[index].items()}
            shuffled = dict(batch, embeddings=shuffle_ds.bank[r["id"]][None].to(device))
            with torch.inference_mode():
                real_map = spatial_matching(*(batch[k] for k in ("visual", "embeddings", "valid", "boxes")))
                shuffled_map = spatial_matching(*(shuffled[k] for k in ("visual", "embeddings", "valid", "boxes")))
                real_map = torch.nn.functional.interpolate(real_map, batch["base"].shape[-2:], mode="bilinear", align_corners=False)[0].cpu().numpy()
                shuffled_map = torch.nn.functional.interpolate(shuffled_map, batch["base"].shape[-2:], mode="bilinear", align_corners=False)[0].cpu().numpy()
            gt = batch["mask"][0, 0].cpu().numpy().astype(bool)
            rng = np.random.default_rng(int(digest([cfg["seed"], r["id"], "matching_pixels"])[:8], 16))
            draw = rng.choice(gt.size, min(s["val_pixels"], gt.size), replace=False)
            layers = batch["visual"].shape[1]
            for arm, maps in (("real", real_map), ("shuffled", shuffled_map)):
                for role_index, role in enumerate(("observation", "expectation", "difference")):
                    coverage = maps[-2+role_index] > .999 if role_index < 2 else (maps[-2:] > .999).all(0)
                    # Paired support must come from one tile, not two unrelated
                    # tiles each offering only one role.
                    if role_index == 2:
                        paired_batch = dict(batch, valid=batch["valid"].min(-1, keepdim=True).values.expand_as(batch["valid"]))
                        with torch.inference_mode():
                            paired = spatial_matching(*(paired_batch[k] for k in ("visual", "embeddings", "valid", "boxes")))[:, -1:]
                            coverage = torch.nn.functional.interpolate(paired, gt.shape, mode="nearest")[0, 0].cpu().numpy() > 0
                    take = draw[coverage.ravel()[draw]]
                    for layer in range(layers):
                        scores = maps[role_index*layers+layer].ravel()[take]
                        values[(arm, role, layer)].append((gt.ravel()[take], scores))
            if index < s["match_panels"]:
                save_matching_panel(root/"matching/panels"/f"{r['id']}.png", r["image_path"], gt, real_map, shuffled_map, layers)
        for (arm, role, layer), chunks in sorted(values.items()):
            labels = np.concatenate([v[0] for v in chunks])
            scores = np.concatenate([v[1] for v in chunks])
            summaries.append(dict(partition=key[0], dataset=key[1], category=key[2], arm=arm, role=role, layer=layer,
                                  covered_sampled_pixels=len(scores), sampled_gt_pixels=int(labels.sum()),
                                  mean_gt=float(scores[labels].mean()) if labels.any() else None,
                                  mean_bg=float(scores[~labels].mean()) if (~labels).any() else None,
                                  pixel_auc_raw_DIAGNOSTIC=float(roc_auc_score(labels, scores)) if labels.any() and (~labels).any() else None))
        print(f"source spatial matching {key[1]}/{key[2]}", flush=True)
    write_csv(root/"matching/metrics.csv", summaries)
    indexed = {(r["dataset"], r["category"], r["role"], r["layer"], r["arm"]): r for r in summaries}
    comparisons = []
    for r in summaries:
        if r["arm"] != "real":
            continue
        control = indexed[(r["dataset"], r["category"], r["role"], r["layer"], "shuffled")]
        auc, other_auc = r["pixel_auc_raw_DIAGNOSTIC"], control["pixel_auc_raw_DIAGNOSTIC"]
        comparisons.append(dict(dataset=r["dataset"], category=r["category"], role=r["role"], layer=r["layer"],
                                real_minus_shuffle_auc_pp=100*(auc-other_auc) if auc is not None and other_auc is not None else None,
                                real_gt_minus_bg=r["mean_gt"]-r["mean_bg"] if r["mean_gt"] is not None and r["mean_bg"] is not None else None,
                                shuffle_gt_minus_bg=control["mean_gt"]-control["mean_bg"] if control["mean_gt"] is not None and control["mean_bg"] is not None else None))
    write_csv(root/"matching/comparisons.csv", comparisons)
    write_csv(root/"matching/shuffle_audit.csv", audit)
    write_csv(root/"matching/donors.csv", donors, ["partition", "dataset", "category", "image_id", "tile", "role", "donor_id", "donor_tile"])
    atomic_json(root/"matching/complete.json", dict(fingerprint=digest(s), source_only=True, groups=len(groups),
               valid_slots=sum(r["valid_slots"] for r in audit), changed_slots=sum(r["changed_slots"] for r in audit),
               note="Raw cosine AUC is a source diagnostic, not a deployable anomaly score or threshold."))


def save_matching_panel(path, image_path, gt, real, shuffled, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB").resize((gt.shape[1], gt.shape[0])))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Source RGB")
    axes[0, 1].imshow(gt, vmin=0, vmax=1, cmap="gray")
    axes[0, 1].set_title("Source GT (diagnostic only)")
    images = [real[:layers].mean(0), real[layers:2*layers].mean(0),
              real[2*layers:3*layers].mean(0), shuffled[2*layers:3*layers].mean(0)]
    for ax, values, title in zip([axes[0, 2], *axes[1]], images, ("Observation cosine", "Expectation cosine", "Real difference", "Shuffled difference")):
        im = ax.imshow(values, vmin=-1 if "cosine" in title else -2, vmax=1 if "cosine" in title else 2, cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def validate(model, source, cfg, s, device, mode, workers):
    metrics = {str(alpha): [] for alpha in [0.]+s["alphas"]}
    for key, rows in repair.groups_for(source, {"val"}).items():
        base, delta, masks = [], [], []
        with torch.inference_mode():
            for batch in batches_for(source, cfg, rows, s, mode, workers, verify=False):
                masks.extend(batch["mask"][:, 0].numpy().astype(bool))
                batch = {k: v.to(device) for k, v in batch.items() if k != "mask"}
                base.extend(batch["base"][:, 0].cpu().numpy())
                delta.extend(forward(model, batch, mode)[:, 0].cpu().numpy())
        base, delta, masks = np.asarray(base), np.asarray(delta), np.asarray(masks)
        seed = int(digest([cfg["seed"], list(key)])[:8], 16)
        for alpha in [0.]+s["alphas"]:
            pred = repair.adjust_numpy(base, delta, alpha)
            metrics[str(alpha)].append(source_metrics(masks, pred, base, seed, s["val_pixels"], s["fpr"]))
    return {k: dict(mean=aggregate_source(v), categories=v) for k, v in metrics.items()}


def train(a):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DistributedSampler
    root, source, cfg, s = context(a)
    require_stage(root, "quality", s)
    require_stage(root, "matching", s)
    world, rank, local = (int(os.environ.get(k, "1" if k == "WORLD_SIZE" else "0")) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if world != s["num_shards"]:
        raise ValueError("DDP WORLD_SIZE must match num_shards")
    device = repair.device_for(local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2))
    try:
        torch.manual_seed(cfg["seed"])
        train_rows, val_rows = old.records_for(source, "train"), old.records_for(source, "val")
        verify_reviews(source, cfg, train_rows+val_rows)
        dataset = P2Dataset(source, cfg, train_rows, s, a.mode, components=a.loss == "p2b")
        old.FeatureDataset(source, cfg, val_rows)
        if len(dataset) < world:
            raise ValueError("Too few source train samples for DDP")
        out = root/"heads"/a.loss/a.mode
        fingerprint = digest([s, a.loss, a.mode])
        if rank == 0 and a.mode == "shuffled":
            write_csv(out/"shuffle_audit.csv", dataset.audit)
            write_csv(out/"donors.csv", dataset.donors, ["partition", "dataset", "category", "image_id", "tile", "role", "donor_id", "donor_tile"])
            slots = sum(r["valid_slots"] for r in dataset.audit)
            changed = sum(r["changed_slots"] for r in dataset.audit)
            print(f"Training shuffle changed {changed}/{slots} valid slots; unchanged/missing donors remain explicit.", flush=True)
            if slots and changed == 0:
                raise ValueError("Shuffled training control has no effective changes; inspect semantic quality")
        shape = load_json(source/"sealed.json")["shape"]
        raw = MatchingHead(shape[0], shape[1], s["hidden"]).to(device)
        optimizer = torch.optim.AdamW(raw.parameters(), lr=s["lr"], weight_decay=1e-4)
        start, history, best_score = 0, [], None
        if (out/"last.pt").exists():
            last = torch.load(out/"last.pt", map_location=device, weights_only=False)
            selected = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
            if last["fingerprint"] != fingerprint or selected["fingerprint"] != fingerprint:
                raise ValueError("Checkpoint settings mismatch")
            raw.load_state_dict(last["state_dict"])
            optimizer.load_state_dict(last["optimizer"])
            start, history = last["epoch"]+1, last["history"]
            best_score = max(last["best_score"], selected["score"])
        model = DistributedDataParallel(raw, device_ids=[local] if device.type == "cuda" else None) if world > 1 else raw
        sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=cfg["seed"]) if world > 1 else None
        gen = torch.Generator()
        batches = DataLoader(dataset, batch_size=s["batch_size"], sampler=sampler, shuffle=sampler is None,
                             num_workers=a.workers, generator=gen)
        if rank == 0 and start == 0:
            raw.eval()
            baseline = validate(raw, source, cfg, s, device, a.mode, a.workers)["0.0"]
            best_score = selection_score(baseline["mean"], baseline["mean"], baseline["categories"], baseline["categories"], s["auc_tolerance"], s["fpr_slack"])
            if (out/"best.pt").exists():
                selected = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
                if selected["fingerprint"] != fingerprint:
                    raise ValueError("Selected checkpoint mismatch")
                best_score = max(best_score, selected["score"])
            else:
                old.save_torch(out/"best.pt", dict(state_dict=raw.state_dict(), alpha=0., epoch=-1, shape=shape,
                                                  fingerprint=fingerprint, selection="BASE_FALLBACK", score=best_score,
                                                  source_validation=baseline))
        if world > 1:
            dist.barrier()
        terms_order = ("bce", "dice", "background", "magnitude", "component", "ranking")
        for epoch in range(start, s["epochs"]):
            if sampler:
                sampler.set_epoch(epoch)
            gen.manual_seed(cfg["seed"]+epoch)
            model.train()
            sums = torch.zeros(8, dtype=torch.float64, device=device)
            for batch in batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                delta = forward(model, batch, a.mode)
                p1 = dict(bg_weight=s["bg_weight"], residual_weight=s["residual_weight"], hard_fraction=s["hard_fraction"])
                if a.loss == "p2a":
                    loss, terms = constrained_loss(batch["base"], delta, batch["mask"], **p1)
                else:
                    options = {k: s[k] for k in ("component_weight", "ranking_weight", "small_fraction", "small_weight", "near_radius", "rank_pixels", "rank_margin")}
                    loss, terms = component_loss(batch["base"], delta, batch["mask"], batch["components"], **options, **p1)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite P2 loss")
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                sums[0] += loss.detach().sum()
                sums[1] += len(loss)
                for i, k in enumerate(terms_order, 2):
                    if k in terms:
                        sums[i] += terms[k].detach().sum()
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
                    if score is not None and score > best_score+1e-8:
                        best_score = score
                        old.save_torch(out/"best.pt", dict(state_dict=raw.state_dict(), alpha=alpha, epoch=epoch, shape=shape,
                                                          fingerprint=fingerprint, selection="SOURCE_VALIDATED", score=score,
                                                          source_validation=candidate))
                history.append(dict(epoch=epoch+1, train_loss=float(sums[0]/sums[1]), loss_terms={k: float(sums[i]/sums[1]) for i, k in enumerate(terms_order, 2)},
                                    validation=metrics, decisions=decisions, best_score=best_score))
                selected = torch.load(out/"best.pt", map_location="cpu", weights_only=False)
                old.save_torch(out/"last.pt", dict(state_dict=raw.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                                                  alpha=selected["alpha"], fingerprint=fingerprint, history=history, best_score=best_score))
                atomic_json(out/"history.json", history)
                print(f"{a.loss}/{a.mode} epoch={epoch+1}/{s['epochs']} loss={history[-1]['train_loss']:.5f} alpha={selected['alpha']} {selected['selection']}", flush=True)
            if world > 1:
                dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def checkpoints(root, s):
    return {f"{loss}/{mode}/{ckpt}": file_digest(root/"heads"/loss/mode/f"{ckpt}.pt")
            for loss in LOSSES for mode in MODES for ckpt in ("best", "last")}


def evaluate(a):
    from tools.vlm_next_diagnostics import map_metrics
    root, source, cfg, s = context(a)
    if not 0 <= a.shard_id < s["num_shards"]:
        raise ValueError("Bad shard_id")
    device = repair.device_for()
    hashes = checkpoints(root, s)
    shape = load_json(source/"sealed.json")["shape"]
    groups = list(repair.groups_for(source, {"val", "eval"}).items())
    for key, rows in groups[a.shard_id::s["num_shards"]]:
        result_path = root/"results"/("_".join(key)+".json")
        if result_path.exists():
            previous = load_json(result_path)
            if previous["fingerprint"] == digest(s) and previous["checkpoints"] == hashes:
                print(f"Resume: {key} evaluation already complete", flush=True)
                continue
        results, comparisons = [], []
        # One model and one category of maps at a time bounds CPU/GPU memory.
        for loss in LOSSES:
            for mode in MODES:
                for ckpt in ("best", "last"):
                    saved = torch.load(root/"heads"/loss/mode/f"{ckpt}.pt", map_location=device, weights_only=False)
                    if saved["fingerprint"] != digest([s, loss, mode]):
                        raise ValueError("Checkpoint mismatch")
                    model = MatchingHead(shape[0], shape[1], s["hidden"]).to(device).eval()
                    model.load_state_dict(saved["state_dict"])
                    alpha = saved["alpha"] if ckpt == "best" else torch.load(root/"heads"/loss/mode/"best.pt", map_location="cpu", weights_only=False)["alpha"]
                    # Last is NEVER selected using target results: retain the
                    # source-selected best alpha. Its alpha=1 map is diagnostic.
                    base, masks, delta, off_delta, shuffle_delta = [], [], [], [], []
                    dataset = P2Dataset(source, cfg, rows, s, mode)
                    if mode == "real":
                        control = P2Dataset(source, cfg, rows, s, "shuffled", verify=False)
                    offset = 0
                    with torch.inference_mode():
                        for batch in DataLoader(dataset, batch_size=s["batch_size"], num_workers=a.workers):
                            masks.extend(batch["mask"][:, 0].numpy().astype(bool))
                            batch = {k: v.to(device) for k, v in batch.items() if k != "mask"}
                            base.extend(batch["base"][:, 0].cpu().numpy())
                            delta.extend(forward(model, batch, mode)[:, 0].cpu().numpy())
                            if mode == "real":
                                donor = torch.stack([control.bank[r["id"]] for r in rows[offset:offset+len(batch["base"])] ]).to(device)
                                off_delta.extend(forward(model, batch, "visual")[:, 0].cpu().numpy())
                                shuffle_delta.extend(forward(model, dict(batch, embeddings=donor), "real")[:, 0].cpu().numpy())
                            offset += len(batch["base"])
                    base, masks, delta = np.asarray(base), np.asarray(masks), np.asarray(delta)
                    seed = int(digest([cfg["seed"], list(key)])[:8], 16)
                    predictions = {"selected_alpha": repair.adjust_numpy(base, delta, alpha)}
                    if ckpt == "last":
                        predictions["alpha1_DIAGNOSTIC"] = repair.adjust_numpy(base, delta, 1.)
                    if mode == "real":
                        raw_shuf, raw_off = np.asarray(shuffle_delta), np.asarray(off_delta)
                        prob_shuf, prob_off = repair.adjust_numpy(base, raw_shuf, alpha), repair.adjust_numpy(base, raw_off, alpha)
                        predictions.update(inference_shuffled=prob_shuf, inference_branch_off=prob_off)
                        comparisons.append(dict(partition=key[0], dataset=key[1], category=key[2], loss=loss, checkpoint=ckpt, alpha=alpha,
                                                **difference(delta, raw_shuf, "raw_real_shuffle", 1e-6),
                                                **difference(delta, raw_off, "raw_real_off", 1e-6),
                                                **difference(predictions["selected_alpha"], prob_shuf, "prob_real_shuffle", 1e-6)))
                    if loss == "p2a" and mode == "visual" and ckpt == "best":
                        predictions["base"] = base
                    for intervention, pred in predictions.items():
                        operating = source_metrics(masks, pred, base, seed, s["val_pixels"], s["fpr"])
                        small, hits, recovered, lost = repair.small_counts(masks, pred, base)
                        results.append(dict(partition=key[0], dataset=key[1], category=key[2], loss=loss, training_mode=mode,
                                            checkpoint=ckpt, intervention=intervention, alpha=0. if intervention == "base" else 1. if intervention == "alpha1_DIAGNOSTIC" else alpha,
                                            selection="REFERENCE" if intervention == "base" else "LAST_DIAGNOSTIC" if ckpt == "last" else saved["selection"], epoch=saved["epoch"]+1,
                                            samples=len(rows), **map_metrics(masks, pred),
                                            region_recall_at_fpr_DIAGNOSTIC=operating["region_recall_at_fpr"],
                                            small_hit_at_fpr_DIAGNOSTIC=operating["small_hit_at_fpr"],
                                            background_FPR_at_05=float(np.mean(pred[~masks] >= .5)),
                                            small_components=small, small_hits_at_05=hits, recovered_small_at_05=recovered, lost_small_at_05=lost,
                                            mean_bg_increase=float(np.maximum(pred-base, 0)[~masks].mean()),
                                            changed_fraction=float(np.mean(np.abs(pred-base) > 1e-6))))
                    del model
        atomic_json(result_path, dict(fingerprint=digest(s), checkpoints=hashes, rows=results, interventions=comparisons))
        print(f"evaluated {key}; best source-selected, last/alpha1 and target-FPR are diagnostics only", flush=True)


def report(a):
    root, source, _, s = context(a)
    hashes = checkpoints(root, s)
    rows, interventions = [], []
    for key in repair.groups_for(source, {"val", "eval"}):
        value = load_json(root/"results"/("_".join(key)+".json"))
        if value["fingerprint"] != digest(s) or value["checkpoints"] != hashes:
            raise ValueError("Stale evaluation; rerun evaluate")
        rows.extend(value["rows"])
        interventions.extend(value["interventions"])
    keys = ("partition", "dataset", "loss", "training_mode", "checkpoint", "intervention")
    aggregates = defaultdict(list)
    for r in rows:
        aggregates[tuple(r[k] for k in keys)].append(r)
    sums = {"samples", "small_components", "small_hits_at_05", "recovered_small_at_05", "lost_small_at_05"}
    for key, values in sorted(aggregates.items()):
        mean = dict(values[0], category="MEAN")
        for k in ("alpha", "epoch", "selection"):
            unique = {v[k] for v in values}
            mean[k] = next(iter(unique)) if len(unique) == 1 else None if k != "selection" else "PER_CATEGORY_METADATA"
        for k in values[0]:
            if k in keys or k in ("category", "alpha", "epoch", "selection"):
                continue
            numbers = [v[k] for v in values if v[k] is not None]
            mean[k] = sum(numbers) if k in sums else float(np.mean(numbers)) if numbers else None
        rows.append(mean)
    write_csv(root/"results/metrics.csv", rows)
    write_csv(root/"results/interventions.csv", interventions)
    atomic_json(root/"results/summary.json", rows)
    comparisons = []
    index = {(r["partition"], r["dataset"], r["category"], r["loss"], r["training_mode"]): r
             for r in rows if r["checkpoint"] == "best" and r["intervention"] == "selected_alpha"}
    base_index = {(r["partition"], r["dataset"], r["category"]): r for r in rows if r["intervention"] == "base"}
    metric_names = ("PRO_exact", "P_AUROC", "background_FPR_at_05", "small_hit_at_fpr_DIAGNOSTIC")
    for key, r in sorted(index.items()):
        part, name, cat, loss, mode = key
        controls = [("head_vs_base", base_index[(part, name, cat)])]
        if mode == "real":
            controls.extend(("real_vs_"+control, index[(part, name, cat, loss, control)]) for control in ("visual", "shuffled"))
        if loss == "p2b":
            controls.append(("p2b_vs_p2a", index[(part, name, cat, "p2a", mode)]))
        for contrast, reference in controls:
            deltas = {"delta_"+k+"_pp": 100*(r[k]-reference[k]) if r[k] is not None and reference[k] is not None else None for k in metric_names}
            comparisons.append(dict(partition=part, dataset=name, category=cat, loss=loss, training_mode=mode, contrast=contrast, **deltas))
    write_csv(root/"results/comparisons.csv", comparisons)
    table = ["# P2 结果速览", "", "best 的 epoch/alpha 仅由源验证选择；alpha=0 是 Base 回退。下表为类别宏平均，数值以百分数显示。", "",
             "| partition/dataset | loss | training mode | alpha | epoch | PRO % | pixel AUROC % | small hit at matched FPR % (diagnostic) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["category"] == "MEAN" and r["checkpoint"] == "best" and r["intervention"] in ("selected_alpha", "base"):
            small = "NA" if r["small_hit_at_fpr_DIAGNOSTIC"] is None else f"{r['small_hit_at_fpr_DIAGNOSTIC']*100:.4f}"
            table.append(f"| {r['partition']}/{r['dataset']} | {r['loss']} | {r['training_mode'] if r['intervention'] != 'base' else 'BASE'} | {r['alpha']} | {r['epoch']} | {r['PRO_exact']*100:.4f} | {r['P_AUROC']*100:.4f} | {small} |")
    table.extend(["", "comparisons.csv：真实语义与独立训练的 visual/shuffled、各头与 Base、P2b 与 P2a 的配对差异（百分点）。",
                  "interventions.csv：同一真实语义头对文本干预的敏感性；它与独立训练的对照不同。",
                  "last、alpha1 和匹配目标 FPR 的行是诊断结果；不能用目标标签重新选择检查点、alpha 或部署阈值。"])
    (root/"results/README.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    print(f"Results: {root/'results/metrics.csv'}; compare p2a real vs visual/shuffled, then p2b vs p2a", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "quality", "matching", "train", "evaluate", "report"), required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--source_work_dir", default="")
    p.add_argument("--mode", choices=MODES, default="real")
    p.add_argument("--loss", choices=LOSSES, default="p2a")
    p.add_argument("--alphas", default="0.1,0.25,0.5,1")
    for k, v in dict(epochs=15, batch_size=4, hidden=96, num_shards=2, shard_id=0, workers=2, val_pixels=8192,
                     review_per_category=6, min_area=2, near_radius=8, rank_pixels=64, match_panels=1).items():
        p.add_argument("--"+k, type=int, default=v)
    for k, v in dict(lr=1e-4, fpr=.01, hard_fraction=.01, bg_weight=1., residual_weight=.01, auc_tolerance=.002,
                     fpr_slack=.002, component_weight=.1, ranking_weight=.1, small_fraction=.001,
                     small_weight=2., rank_margin=1.).items():
        p.add_argument("--"+k, type=float, default=v)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"prepare": prepare, "quality": quality, "matching": matching, "train": train,
     "evaluate": evaluate, "report": report}[args.stage](args)
