#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from Datasets import DATASET_CLASSES, DATASET_REGISTRY
from tools.mara_agent import MARAAgent, MARAConfig
from tools.utils_up import get_anomaly_map
from tools.visualization import visualization
from train_mara import layer_maps_from_debug
from train_up import (
    create_adapter_model,
    create_clip_model,
    create_prompt_learner,
    create_visual_backbone_for_name,
    parse_visual_layers,
    set_seed,
)
from test2 import (
    compute_best_f1,
    compute_i_auroc,
    compute_image_level_scores,
    compute_p_auroc,
    compute_pro,
    normalize_maps,
)


def sorted_mara_checkpoints(weight_path: str) -> List[str]:
    path = Path(weight_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {weight_path}")

    candidates = list(path.glob("mara_epoch_*.pth"))
    if not candidates:
        candidates = list(path.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No MARA checkpoint found in: {weight_path}")

    def key_fn(p: Path):
        nums = "".join(ch if ch.isdigit() else " " for ch in p.stem).split()
        return int(nums[-1]) if nums else 10**9

    return [str(p) for p in sorted(candidates, key=key_fn)]


def prepare_data(dataset_name: str, category: str, args):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=args.image_size,
        imagesize=args.image_size,
    )
    kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print(f"Loaded [{dataset_name}] ({category}) test set, size: {len(dataset)}")
    return loader


def config_from_payload(payload: Dict, args) -> MARAConfig:
    cfg_dict = dict(payload.get("mara_config", {}))
    if not cfg_dict:
        cfg_dict = {
            "num_layers": len(args.visual_layers),
            "map_size": args.mara_map_size,
            "num_regions": args.mara_regions,
            "roi_size": args.mara_roi_size,
            "hidden_dim": args.mara_hidden_dim,
            "max_steps": args.mara_steps,
            "group_size": 1,
        }
    valid_keys = MARAConfig.__dataclass_fields__.keys()
    return MARAConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})


def apply_payload_runtime_config(payload: Dict, args) -> None:
    if "visual_backbone" in payload:
        args.visual_backbone = payload["visual_backbone"]
    if "visual_layers" in payload:
        args.visual_layers = tuple(int(x) for x in payload["visual_layers"])


def build_models_from_checkpoint(args, device: torch.device, payload: Dict):
    apply_payload_runtime_config(payload, args)
    clip_model = create_clip_model(args, device)
    prompt_learner = create_prompt_learner(clip_model, device)
    dino_model, dino_adapters = create_visual_backbone_for_name(args, device, args.visual_backbone)
    model = create_adapter_model(clip_model, device, args.visual_backbone)
    mara_agent = MARAAgent(config_from_payload(payload, args)).to(device)

    clip_model.eval()
    prompt_learner.eval()
    model.eval()
    mara_agent.eval()
    if dino_model is not None:
        dino_model.eval()
    if dino_adapters is not None:
        dino_adapters.eval()

    return clip_model, prompt_learner, dino_model, dino_adapters, model, mara_agent


def load_mara_checkpoint(
    ckpt_path: str,
    payload: Dict,
    model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    dino_adapters: Optional[torch.nn.Module],
    mara_agent: MARAAgent,
    device: torch.device,
) -> None:
    model.cls_token_adapter.load_state_dict(payload["cls_token_adapter"], strict=False)
    model.patch_token_adapter.load_state_dict(payload["patch_token_adapter"], strict=False)
    model.prompt_adapter.load_state_dict(payload["prompt_adapter"], strict=False)
    if "prompt_learner" in payload:
        prompt_learner.load_state_dict(payload["prompt_learner"], strict=False)
    if dino_adapters is not None and "dino_adapters" in payload:
        dino_adapters.load_state_dict(payload["dino_adapters"], strict=False)
    mara_agent.load_state_dict(payload["mara_agent"], strict=True)

    model.to(device).eval()
    prompt_learner.to(device).eval()
    mara_agent.to(device).eval()
    if dino_adapters is not None:
        dino_adapters.to(device).eval()


def evaluate_category(
    args,
    category: str,
    clip_model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    dino_model: Optional[torch.nn.Module],
    model: torch.nn.Module,
    mara_agent: MARAAgent,
    device: torch.device,
    result_dir: str,
):
    pred_maps = []
    gt_masks = []
    img_paths = []

    loader = prepare_data(args.dataset, category, args)
    with torch.no_grad():
        for batch_idx, image_info in enumerate(tqdm(loader)):
            _, mask, base_map, base_logits, debug = get_anomaly_map(
                clip_model=clip_model,
                image_info=image_info,
                device=device,
                model=model,
                Dino_model=dino_model,
                prompt_learner=prompt_learner,
                idx=batch_idx,
                visual_backbone=args.visual_backbone,
                visual_layers=args.visual_layers,
                text_source=args.text_source,
                return_debug=True,
            )
            layer_maps = layer_maps_from_debug(
                debug=debug,
                device=device,
                fallback_prob=base_map,
                num_layers=len(args.visual_layers),
            )
            output = mara_agent.infer(
                base_prob=base_map,
                base_logits=base_logits,
                layer_maps=layer_maps,
                sample=args.sample_policy,
            )

            pred_maps.append(output["final_prob"][:, 1].detach().cpu().numpy())
            mask_np = mask[:, 0].detach().cpu().numpy() if mask.dim() == 4 else mask.detach().cpu().numpy()
            gt_masks.append((mask_np > 0.5).astype(np.uint8))
            img_paths.extend(list(image_info["image_path"]))

    pred_maps = np.concatenate(pred_maps, axis=0).astype(np.float32)
    gt_masks = np.concatenate(gt_masks, axis=0).astype(np.uint8)
    pred_eval = normalize_maps(pred_maps, mode=args.norm_mode)

    if args.save_vis:
        vis_dir = os.path.join(result_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        visualization(img_paths, normalize_maps(pred_maps, mode="per_class"), gt_masks, category, vis_dir)

    image_gt, image_pred = compute_image_level_scores(gt_masks, pred_eval)
    return {
        "category": category,
        "F1": compute_best_f1(gt_masks, pred_eval),
        "I_AUROC": compute_i_auroc(image_gt, image_pred),
        "P_AUROC": compute_p_auroc(gt_masks, pred_eval),
        "PRO": compute_pro(gt_masks, pred_eval, num_th=args.pro_num_th, max_fpr=args.pro_max_fpr),
    }


def write_results(result_dir: str, epoch_name: str, rows: List[Dict]) -> Dict:
    os.makedirs(result_dir, exist_ok=True)
    means = {
        "epoch": epoch_name,
        "mean_F1": float(np.mean([r["F1"] for r in rows])),
        "mean_I_AUROC": float(np.mean([r["I_AUROC"] for r in rows])),
        "mean_P_AUROC": float(np.mean([r["P_AUROC"] for r in rows])),
        "mean_PRO": float(np.mean([r["PRO"] for r in rows])),
    }

    metric_file = os.path.join(result_dir, "metric_mara.txt")
    with open(metric_file, "a", encoding="utf-8") as f:
        f.write(f"----------MARA epoch {epoch_name}----------\n")
        f.write(f"{'Classname':<18s}{'F1':>10s}{'I-AUROC':>12s}{'P-AUROC':>12s}{'PRO':>10s}\n")
        for row in rows:
            f.write(
                f"{row['category']:<18s}"
                f"{row['F1']:>10.5f}"
                f"{row['I_AUROC']:>12.5f}"
                f"{row['P_AUROC']:>12.5f}"
                f"{row['PRO']:>10.5f}\n"
            )
        f.write(
            f"{'Mean':<18s}"
            f"{means['mean_F1']:>10.5f}"
            f"{means['mean_I_AUROC']:>12.5f}"
            f"{means['mean_P_AUROC']:>12.5f}"
            f"{means['mean_PRO']:>10.5f}\n\n"
        )

    csv_file = os.path.join(result_dir, f"{epoch_name}_mara_metrics.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "F1", "I_AUROC", "P_AUROC", "PRO"])
        writer.writeheader()
        writer.writerows(rows)
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result_MARA_Test")
    parser.add_argument("--weight_path", type=str, required=True, help="MARA checkpoint file or checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--category", type=str, default="ALL")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--visual_backbone", type=str, default="dino", choices=["dino", "clip"])
    parser.add_argument("--visual_layers", type=parse_visual_layers, default=(5, 11, 17, 23))
    parser.add_argument("--text_source", type=str, default="prompt_learner", choices=["prompt_learner", "ensemble"])

    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--dino_repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--dino_bottleneck", type=int, default=256)

    parser.add_argument("--mara_steps", type=int, default=3)
    parser.add_argument("--mara_map_size", type=int, default=128)
    parser.add_argument("--mara_regions", type=int, default=16)
    parser.add_argument("--mara_roi_size", type=int, default=32)
    parser.add_argument("--mara_hidden_dim", type=int, default=64)

    parser.add_argument("--norm_mode", type=str, default="none", choices=["none", "per_image", "per_class"])
    parser.add_argument("--pro_num_th", type=int, default=1000)
    parser.add_argument("--pro_max_fpr", type=float, default=0.3)
    parser.add_argument("--eval_latest_only", action="store_true")
    parser.add_argument("--sample_policy", action="store_true", help="sample actions instead of greedy policy during evaluation")
    parser.add_argument("--save_vis", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.dataset = args.dataset.lower()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_result_dir = os.path.join(args.result_path, args.dataset)
    os.makedirs(dataset_result_dir, exist_ok=True)

    ckpts = sorted_mara_checkpoints(args.weight_path)
    if args.eval_latest_only:
        ckpts = [ckpts[-1]]

    first_payload = torch.load(ckpts[0], map_location=device)
    clip_model, prompt_learner, dino_model, dino_adapters, model, mara_agent = build_models_from_checkpoint(
        args=args,
        device=device,
        payload=first_payload,
    )

    summaries = []
    categories = sorted(DATASET_CLASSES[args.dataset]) if args.category == "ALL" else [args.category]
    for ckpt_path in ckpts:
        epoch_name = Path(ckpt_path).stem
        payload = torch.load(ckpt_path, map_location=device)
        load_mara_checkpoint(
            ckpt_path=ckpt_path,
            payload=payload,
            model=model,
            prompt_learner=prompt_learner,
            dino_adapters=dino_adapters,
            mara_agent=mara_agent,
            device=device,
        )
        print(f"===== Evaluating MARA checkpoint: {ckpt_path} =====")
        rows = [
            evaluate_category(
                args=args,
                category=category,
                clip_model=clip_model,
                prompt_learner=prompt_learner,
                dino_model=dino_model,
                model=model,
                mara_agent=mara_agent,
                device=device,
                result_dir=os.path.join(dataset_result_dir, epoch_name),
            )
            for category in categories
        ]
        summary = write_results(dataset_result_dir, epoch_name, rows)
        summary["ckpt_path"] = ckpt_path
        summaries.append(summary)
        print(
            f"MARA {epoch_name}: F1={summary['mean_F1']:.5f}, "
            f"I-AUROC={summary['mean_I_AUROC']:.5f}, "
            f"P-AUROC={summary['mean_P_AUROC']:.5f}, PRO={summary['mean_PRO']:.5f}"
        )

    ranking_file = os.path.join(dataset_result_dir, "mara_epoch_ranking.csv")
    with open(ranking_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "mean_F1", "mean_I_AUROC", "mean_P_AUROC", "mean_PRO", "ckpt_path"],
        )
        writer.writeheader()
        writer.writerows(summaries)

    best = max(summaries, key=lambda x: x["mean_PRO"])
    print("=" * 60)
    print(
        f"Best MARA checkpoint by mean_PRO: {best['epoch']} | "
        f"F1={best['mean_F1']:.5f}, I-AUROC={best['mean_I_AUROC']:.5f}, "
        f"P-AUROC={best['mean_P_AUROC']:.5f}, PRO={best['mean_PRO']:.5f}"
    )
    print(f"Best checkpoint file: {best['ckpt_path']}")


if __name__ == "__main__":
    main()
