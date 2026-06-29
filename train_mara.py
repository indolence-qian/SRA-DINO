import argparse
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.loss import BinaryDiceLoss, FocalLoss
from tools.mara_agent import MARAAgent, MARAConfig
from tools.bottleneckAdapter import install_bottleneck_adapters_into_dino
from tools.utils_up import get_anomaly_map
from train_up import (
    build_warmup_cosine_scheduler,
    create_adapter_model,
    create_clip_model,
    create_prompt_learner,
    prepare_data,
    parse_visual_layers,
    set_seed,
)


HFA_CHOICES = ("none", "l5", "l11", "l17", "l23", "hfa1", "hfa2", "hfa3", "hfa4")


def freeze_module(module: Optional[torch.nn.Module]) -> None:
    if module is None:
        return
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def resolve_hfa_layers(hfa_setting: str) -> Tuple[int, ...]:
    preset = {
        "none": (),
        "l5": (5,),
        "l11": (11,),
        "l17": (17,),
        "l23": (23,),
        "hfa1": (23,),
        "hfa2": (17, 23),
        "hfa3": (11, 17, 23),
        "hfa4": (5, 11, 17, 23),
    }
    key = str(hfa_setting).lower()
    if key not in preset:
        raise ValueError(f"Unsupported hfa_setting={hfa_setting}. Available: {list(preset.keys())}")
    return preset[key]


def normalize_layers_obj(layers_obj) -> Tuple[int, ...]:
    if layers_obj is None:
        return ()
    if isinstance(layers_obj, str):
        return tuple(int(x.strip()) for x in layers_obj.split(",") if x.strip())
    if torch.is_tensor(layers_obj):
        return tuple(int(x) for x in layers_obj.detach().cpu().tolist())
    return tuple(int(x) for x in layers_obj)


def load_checkpoint_payload(ckpt_path: str) -> Dict:
    if not ckpt_path:
        return {}
    return torch.load(ckpt_path, map_location="cpu")


def apply_base_runtime_config(args, payload: Dict) -> None:
    if payload.get("visual_backbone"):
        args.visual_backbone = payload["visual_backbone"]
    if payload.get("visual_layers"):
        args.visual_layers = normalize_layers_obj(payload["visual_layers"])

    if payload.get("hfa_bottleneck") is not None:
        args.dino_bottleneck = int(payload["hfa_bottleneck"])
    elif getattr(args, "hfa_bottleneck", None) is not None:
        args.dino_bottleneck = int(args.hfa_bottleneck)

    has_hfa_metadata = "hfa_layers" in payload or "hfa_setting" in payload
    hfa_layers = normalize_layers_obj(payload.get("hfa_layers"))
    hfa_setting = payload.get("hfa_setting", None)
    if not hfa_layers and hfa_setting:
        hfa_layers = resolve_hfa_layers(hfa_setting)
    if not hfa_layers and not has_hfa_metadata and payload.get("dino_adapters") is not None and payload.get("visual_layers"):
        # train_up checkpoints stored dino_adapters but not hfa metadata.
        hfa_layers = normalize_layers_obj(payload["visual_layers"])
        hfa_setting = "custom"
    if not hfa_layers and not has_hfa_metadata and getattr(args, "hfa_layers", ""):
        hfa_layers = normalize_layers_obj(args.hfa_layers)
        hfa_setting = "custom"
    if not hfa_layers and not has_hfa_metadata:
        hfa_setting = getattr(args, "hfa_setting", "hfa4")
        hfa_layers = resolve_hfa_layers(hfa_setting)

    args.hfa_setting_runtime = hfa_setting or getattr(args, "hfa_setting", "custom")
    args.hfa_layers_runtime = tuple(hfa_layers)
    args.hfa_bottleneck_runtime = int(args.dino_bottleneck)


def create_visual_backbone_for_mara(args, device: torch.device):
    dino_model = None
    dino_adapters = None

    if args.visual_backbone == "dino":
        dino_model = torch.hub.load(
            args.dino_repo_dir,
            args.dino_model_name,
            source="local",
            weights=args.dino_weights,
        )
        dino_model.eval()
        for param in dino_model.parameters():
            param.requires_grad_(False)

        hfa_layers = tuple(getattr(args, "hfa_layers_runtime", tuple(args.visual_layers)))
        if hfa_layers:
            dino_adapters, _ = install_bottleneck_adapters_into_dino(
                dino_model,
                layers=hfa_layers,
                bottleneck=int(args.dino_bottleneck),
            )
            dino_adapters.train()
        dino_model.to(device)

    return dino_model, dino_adapters


def load_base_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    dino_adapters: Optional[torch.nn.Module],
    device: torch.device,
    payload: Optional[Dict] = None,
) -> None:
    if not ckpt_path:
        print("[WARN] --base_ckpt is empty. MARA will train on an uncalibrated base detector.")
        return

    payload = payload or torch.load(ckpt_path, map_location="cpu")
    model.cls_token_adapter.load_state_dict(payload["cls_token_adapter"], strict=False)
    model.patch_token_adapter.load_state_dict(payload["patch_token_adapter"], strict=False)
    model.prompt_adapter.load_state_dict(payload["prompt_adapter"], strict=False)
    if "prompt_learner" in payload:
        prompt_learner.load_state_dict(payload["prompt_learner"], strict=False)
    if dino_adapters is not None:
        if "dino_adapters" in payload:
            dino_adapters.load_state_dict(payload["dino_adapters"], strict=True)
        else:
            print("[WARN] Current run uses DINO adapters, but the base checkpoint has no dino_adapters.")
    print(f"Loaded base checkpoint: {ckpt_path}")


def layer_maps_from_debug(debug, device: torch.device, fallback_prob: torch.Tensor, num_layers: int) -> torch.Tensor:
    if debug is None or "cross_prob_layers" not in debug or not debug["cross_prob_layers"]:
        return fallback_prob[:, 1:2].detach().repeat(1, num_layers, 1, 1)

    layer_maps = []
    for layer_map in debug["cross_prob_layers"][:num_layers]:
        if not torch.is_tensor(layer_map):
            layer_map = torch.as_tensor(layer_map)
        layer_maps.append(layer_map.to(device=device, dtype=fallback_prob.dtype))
    while len(layer_maps) < num_layers:
        layer_maps.append(layer_maps[-1])
    return torch.stack(layer_maps, dim=1).detach()


def build_mara_config(args) -> MARAConfig:
    return MARAConfig(
        num_layers=len(args.visual_layers),
        map_size=args.mara_map_size,
        num_regions=args.mara_regions,
        roi_size=args.mara_roi_size,
        hidden_dim=args.mara_hidden_dim,
        max_steps=args.mara_steps,
        group_size=args.grpo_group_size,
        gamma=args.grpo_gamma,
        grpo_clip_eps=args.grpo_clip_eps,
        entropy_coef=args.entropy_coef,
        step_cost=args.mara_step_cost,
        refine_cost=args.mara_refine_cost,
        reward_cls_weight=args.reward_cls_weight,
        reward_loc_weight=args.reward_loc_weight,
        reward_conf_weight=args.reward_conf_weight,
        reward_fp_weight=args.reward_fp_weight,
        delta_scale=args.mara_delta_scale,
        force_first_refine=args.force_first_refine,
    )


def save_mara_checkpoint(
    epoch: int,
    save_dir: str,
    mara_agent: MARAAgent,
    model: torch.nn.Module,
    prompt_learner: torch.nn.Module,
    dino_adapters: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    args,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "epoch": epoch,
        "base_ckpt": args.base_ckpt,
        "visual_backbone": args.visual_backbone,
        "visual_layers": list(args.visual_layers),
        "hfa_setting": getattr(args, "hfa_setting_runtime", args.hfa_setting),
        "hfa_layers": list(getattr(args, "hfa_layers_runtime", ())),
        "hfa_bottleneck": int(getattr(args, "hfa_bottleneck_runtime", args.dino_bottleneck)),
        "mara_agent": mara_agent.state_dict(),
        "mara_config": build_mara_config(args).__dict__,
        "cls_token_adapter": model.cls_token_adapter.state_dict(),
        "patch_token_adapter": model.patch_token_adapter.state_dict(),
        "prompt_adapter": model.prompt_adapter.state_dict(),
        "prompt_learner": prompt_learner.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if dino_adapters is not None:
        payload["dino_adapters"] = dino_adapters.state_dict()
    ckpt_path = os.path.join(save_dir, f"mara_epoch_{epoch}.pth")
    torch.save(payload, ckpt_path)
    return ckpt_path


def train_one_epoch(
    epoch: int,
    mara_agent: MARAAgent,
    optimizer: torch.optim.Optimizer,
    scheduler,
    clip_model: torch.nn.Module,
    model: torch.nn.Module,
    dino_model: Optional[torch.nn.Module],
    prompt_learner: torch.nn.Module,
    train_loader,
    device: torch.device,
    args,
):
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    mara_agent.train()

    meters = {
        "loss": [],
        "sup": [],
        "seg": [],
        "global": [],
        "consistency": [],
        "grpo": [],
        "entropy": [],
        "reward": [],
        "quality": [],
    }

    for idx, image_info in enumerate(train_loader):
        with torch.no_grad():
            anomaly_awareness, mask, base_map, base_logits, debug = get_anomaly_map(
                clip_model=clip_model,
                image_info=image_info,
                device=device,
                model=model,
                Dino_model=dino_model,
                prompt_learner=prompt_learner,
                idx=idx,
                visual_backbone=args.visual_backbone,
                visual_layers=args.visual_layers,
                text_source=args.text_source,
                return_debug=True,
            )

        labels = image_info["is_anomaly"].to(device).long()
        layer_maps = layer_maps_from_debug(
            debug=debug,
            device=device,
            fallback_prob=base_map,
            num_layers=len(args.visual_layers),
        )

        rollout = mara_agent.rollout(
            base_prob=base_map,
            base_logits=base_logits,
            layer_maps=layer_maps,
            mask=mask,
            labels=labels,
            group_size=args.grpo_group_size,
            sample=True,
        )

        group_size = args.grpo_group_size
        mask_g = mask.repeat_interleave(group_size, dim=0)
        labels_g = labels.repeat_interleave(group_size, dim=0)
        mask_bhw = mask_g[:, 0] if mask_g.dim() == 4 else mask_g

        final_map = rollout["final_prob"]
        final_logits = rollout["final_logits"]
        seg_focal = loss_focal(final_map, mask_g)
        seg_dice = loss_dice(final_map[:, 1], mask_bhw)
        seg_loss = seg_focal + seg_dice
        global_loss = F.cross_entropy(final_logits, labels_g)
        base_map_g = base_map.detach().repeat_interleave(group_size, dim=0)
        normal_weight = (1.0 - labels_g.float())
        drift_per_sample = (final_map[:, 1] - base_map_g[:, 1]).abs().mean(dim=(-2, -1))
        if normal_weight.sum() > 0:
            consistency_loss = (drift_per_sample * normal_weight).sum() / normal_weight.sum().clamp_min(1.0)
        else:
            consistency_loss = drift_per_sample.mean() * 0.0
        sup_loss = (
            args.w_seg * seg_loss
            + args.w_global * global_loss
            + args.w_base_consistency * consistency_loss
        )

        grpo_loss = rollout["policy_loss"]
        entropy = rollout["entropy"]
        total_loss = sup_loss + args.w_grpo * grpo_loss - args.entropy_coef * entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mara_agent.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        meters["loss"].append(float(total_loss.item()))
        meters["sup"].append(float(sup_loss.item()))
        meters["seg"].append(float(seg_loss.item()))
        meters["global"].append(float(global_loss.item()))
        meters["consistency"].append(float(consistency_loss.item()))
        meters["grpo"].append(float(grpo_loss.item()))
        meters["entropy"].append(float(entropy.item()))
        meters["reward"].append(float(rollout["mean_reward"].item()))
        meters["quality"].append(float(rollout["mean_quality"].item()))

        print(
            f"Epoch {epoch + 1}/{args.epoch} | Batch {idx + 1}/{len(train_loader)} "
            f"| loss {meters['loss'][-1]:.4f} | sup {meters['sup'][-1]:.4f} "
            f"| seg {meters['seg'][-1]:.4f} | global {meters['global'][-1]:.4f} "
            f"| cons {meters['consistency'][-1]:.4f} "
            f"| grpo {meters['grpo'][-1]:.4f} | entropy {meters['entropy'][-1]:.4f} "
            f"| reward {meters['reward'][-1]:.4f} | quality {meters['quality'][-1]:.4f}",
            end="\r",
            flush=True,
        )

    return {key: float(np.mean(values)) if values else 0.0 for key, values in meters.items()}


def run_train(args) -> None:
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    os.makedirs(args.result_path, exist_ok=True)
    base_payload = load_checkpoint_payload(args.base_ckpt)
    apply_base_runtime_config(args, base_payload)

    print(
        "Runtime config: "
        f"visual_backbone={args.visual_backbone}, "
        f"visual_layers={args.visual_layers}, "
        f"hfa_setting={args.hfa_setting_runtime}, "
        f"hfa_layers={args.hfa_layers_runtime}, "
        f"hfa_bottleneck={args.hfa_bottleneck_runtime}"
    )

    clip_model = create_clip_model(args, device)
    prompt_learner = create_prompt_learner(clip_model, device)
    dino_model, dino_adapters = create_visual_backbone_for_mara(args, device)
    model = create_adapter_model(clip_model, device, args.visual_backbone)

    load_base_checkpoint(args.base_ckpt, model, prompt_learner, dino_adapters, device, payload=base_payload)
    freeze_module(clip_model)
    freeze_module(prompt_learner)
    freeze_module(model)
    freeze_module(dino_adapters)
    freeze_module(dino_model)

    train_loader = prepare_data(
        dataset_name=args.dataset,
        category=args.category,
        batch_size=args.batch_size,
        split_name=args.train_split,
        image_size=args.image_size,
        shuffle=True,
    )

    mara_cfg = build_mara_config(args)
    mara_agent = MARAAgent(mara_cfg).to(device)
    optimizer = torch.optim.AdamW(
        mara_agent.parameters(),
        lr=args.mara_lr,
        weight_decay=args.mara_weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(1, args.epoch * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=max(1, int(args.warmup_ratio * total_steps)),
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    print("MARA config:", mara_cfg)
    for epoch in range(args.epoch):
        start = time.time()
        meters = train_one_epoch(
            epoch=epoch,
            mara_agent=mara_agent,
            optimizer=optimizer,
            scheduler=scheduler,
            clip_model=clip_model,
            model=model,
            dino_model=dino_model,
            prompt_learner=prompt_learner,
            train_loader=train_loader,
            device=device,
            args=args,
        )
        print()
        ckpt_path = save_mara_checkpoint(
            epoch=epoch,
            save_dir=os.path.join(args.result_path, "ckpt"),
            mara_agent=mara_agent,
            model=model,
            prompt_learner=prompt_learner,
            dino_adapters=dino_adapters,
            optimizer=optimizer,
            args=args,
        )

        with open(os.path.join(args.result_path, "loss_mara.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"epoch_{epoch}: "
                f"loss={meters['loss']:.6f}\t"
                f"sup={meters['sup']:.6f}\t"
                f"seg={meters['seg']:.6f}\t"
                f"global={meters['global']:.6f}\t"
                f"consistency={meters['consistency']:.6f}\t"
                f"grpo={meters['grpo']:.6f}\t"
                f"entropy={meters['entropy']:.6f}\t"
                f"reward={meters['reward']:.6f}\t"
                f"quality={meters['quality']:.6f}\n"
            )

        print(
            f"epoch_{epoch}: loss={meters['loss']:.6f}, sup={meters['sup']:.6f}, "
            f"consistency={meters['consistency']:.6f}, grpo={meters['grpo']:.6f}, reward={meters['reward']:.6f}, "
            f"quality={meters['quality']:.6f}, time={time.time() - start:.2f}s, ckpt={ckpt_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result_MARA")
    parser.add_argument("--base_ckpt", type=str, default="", help="checkpoint from train.py/train_up.py used as base detector")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--category", type=str, default="ALL")
    parser.add_argument("--train_split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--visual_backbone", type=str, default="dino", choices=["dino", "clip"])
    parser.add_argument("--visual_layers", type=parse_visual_layers, default=(5, 11, 17, 23))
    parser.add_argument("--text_source", type=str, default="prompt_learner", choices=["prompt_learner", "ensemble"])
    parser.add_argument("--hfa_setting", type=str, default="hfa4", choices=HFA_CHOICES)
    parser.add_argument("--hfa_layers", type=str, default="", help="optional comma-separated HFA adapter layers")

    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--dino_repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--dino_bottleneck", type=int, default=256)
    parser.add_argument("--hfa_bottleneck", type=int, default=None, help="alias for DINO adapter bottleneck fallback")

    parser.add_argument("--mara_steps", type=int, default=3)
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--mara_map_size", type=int, default=128)
    parser.add_argument("--mara_regions", type=int, default=16)
    parser.add_argument("--mara_roi_size", type=int, default=32)
    parser.add_argument("--mara_hidden_dim", type=int, default=64)
    parser.add_argument("--mara_delta_scale", type=float, default=0.25)
    parser.add_argument("--force_first_refine", action="store_true")
    parser.add_argument("--mara_lr", type=float, default=1e-4)
    parser.add_argument("--mara_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mara_step_cost", type=float, default=0.01)
    parser.add_argument("--mara_refine_cost", type=float, default=0.02)

    parser.add_argument("--grpo_gamma", type=float, default=0.95)
    parser.add_argument("--grpo_clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--reward_cls_weight", type=float, default=0.5)
    parser.add_argument("--reward_loc_weight", type=float, default=1.0)
    parser.add_argument("--reward_conf_weight", type=float, default=0.1)
    parser.add_argument("--reward_fp_weight", type=float, default=0.2)

    parser.add_argument("--w_seg", type=float, default=0.7)
    parser.add_argument("--w_global", type=float, default=0.3)
    parser.add_argument("--w_base_consistency", type=float, default=0.05)
    parser.add_argument("--w_grpo", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
