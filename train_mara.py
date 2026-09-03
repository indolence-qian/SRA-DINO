import argparse
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tools.loss import BinaryDiceLoss, FocalLoss
from tools.mara_agent import MARAAgent, MARAConfig
from tools.mara_evidence import build_mara_evidence
from tools.bottleneckAdapter import install_bottleneck_adapters_into_dino
from tools.dino_single_tower import (
    DinoSingleTowerDetector,
    config_from_checkpoint as single_tower_config_from_checkpoint,
    create_dino_single_tower,
    forward_dino_single_batch,
)
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


def setup_distributed(args) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = world_size > 1
    args.world_size = world_size
    args.rank = 0
    args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed MARA training requires CUDA/NCCL.")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        device = torch.device("cuda", args.local_rank)
    elif torch.cuda.is_available():
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    args.is_main_process = args.rank == 0
    return device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_mara_agent(module: torch.nn.Module) -> MARAAgent:
    return module.module if isinstance(module, DDP) else module


def build_distributed_loader(loader: DataLoader, args) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    if not args.distributed:
        return loader, None

    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    distributed_loader = DataLoader(
        loader.dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )
    return distributed_loader, sampler


def reduce_epoch_metrics(meters: Dict[str, float], device: torch.device, args) -> Dict[str, float]:
    if not args.distributed:
        return meters
    keys = list(meters.keys())
    values = torch.tensor([meters[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(args.world_size)
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


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
    args.base_arch = str(payload.get("base_arch", "clip_dino"))
    args.evidence_feature_channels = 0
    if args.base_arch == "dino_single":
        single_config = single_tower_config_from_checkpoint(payload)
        args.evidence_feature_channels = int(single_config.evidence_channels)
    if payload.get("visual_backbone"):
        args.visual_backbone = payload["visual_backbone"]
    if payload.get("dino_repo_dir"):
        args.dino_repo_dir = payload["dino_repo_dir"]
    if payload.get("dino_model_name"):
        args.dino_model_name = payload["dino_model_name"]
    if payload.get("dino_weights"):
        args.dino_weights = payload["dino_weights"]
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


def create_single_tower_from_checkpoint(
    args,
    device: torch.device,
    payload: Dict,
) -> DinoSingleTowerDetector:
    config = single_tower_config_from_checkpoint(payload)
    detector = create_dino_single_tower(
        config=config,
        repo_dir=args.dino_repo_dir,
        model_name=args.dino_model_name,
        weights=args.dino_weights,
        device=device,
    )
    detector.load_checkpoint_fields(payload, strict=True)
    return detector


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
    feature_channels = int(getattr(args, "evidence_feature_channels", 0))
    evidence_channels = (
        0
        if args.disable_evidence_bank
        else (4 + feature_channels) * len(args.visual_layers) + 2
    )
    global_evidence_dim = 0 if args.disable_evidence_bank else len(args.visual_layers)
    return MARAConfig(
        num_layers=len(args.visual_layers),
        evidence_channels=evidence_channels,
        global_evidence_dim=global_evidence_dim,
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
        include_base_trajectory=not args.disable_base_trajectory,
        base_anchor_margin=args.base_anchor_margin,
        negative_advantage_scale=args.negative_advantage_scale,
        advantage_clip=args.advantage_clip,
        gate_max=args.mara_gate_max,
        gate_init_bias=args.mara_gate_init_bias,
        use_gain_gate=not args.disable_gain_gate,
        gain_accept_threshold=args.gain_accept_threshold,
        gain_gate_temperature=args.gain_gate_temperature,
        gain_loss_clip=args.gain_loss_clip,
        gain_cls_weight=args.gain_cls_weight,
        gain_safety_margin=args.gain_safety_margin,
        gain_accept_probability=args.gain_accept_probability,
        hard_gain_gate=not args.disable_hard_gain_gate,
        gain_consistency_temperature=args.gain_consistency_temperature,
        use_gain_lower_bound=not args.disable_gain_lower_bound,
        gain_lower_quantile=args.gain_lower_quantile,
        gain_lower_weight=args.gain_lower_weight,
        quality_degradation_tolerance=args.quality_degradation_tolerance,
    )


def save_mara_checkpoint(
    epoch: int,
    save_dir: str,
    mara_agent: MARAAgent,
    model: Optional[torch.nn.Module],
    prompt_learner: Optional[torch.nn.Module],
    dino_adapters: Optional[torch.nn.Module],
    single_tower: Optional[DinoSingleTowerDetector],
    optimizer: torch.optim.Optimizer,
    args,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    mara_agent = unwrap_mara_agent(mara_agent)
    payload = {
        "epoch": epoch,
        "base_ckpt": args.base_ckpt,
        "base_arch": getattr(args, "base_arch", "clip_dino"),
        "visual_backbone": args.visual_backbone,
        "visual_layers": list(args.visual_layers),
        "dino_repo_dir": args.dino_repo_dir,
        "dino_model_name": args.dino_model_name,
        "dino_weights": args.dino_weights,
        "hfa_setting": getattr(args, "hfa_setting_runtime", args.hfa_setting),
        "hfa_layers": list(getattr(args, "hfa_layers_runtime", ())),
        "hfa_bottleneck": int(getattr(args, "hfa_bottleneck_runtime", args.dino_bottleneck)),
        "mara_agent": mara_agent.state_dict(),
        "mara_config": build_mara_config(args).__dict__,
        "optimizer": optimizer.state_dict(),
    }
    if single_tower is not None:
        payload.update(single_tower.checkpoint_fields())
    else:
        if model is None or prompt_learner is None:
            raise ValueError("Legacy CLIP/DINO checkpoint requires model and prompt_learner.")
        payload.update(
            {
                "cls_token_adapter": model.cls_token_adapter.state_dict(),
                "patch_token_adapter": model.patch_token_adapter.state_dict(),
                "prompt_adapter": model.prompt_adapter.state_dict(),
                "prompt_learner": prompt_learner.state_dict(),
            }
        )
        if dino_adapters is not None:
            payload["dino_adapters"] = dino_adapters.state_dict()
    ckpt_path = os.path.join(save_dir, f"mara_epoch_{epoch}.pth")
    torch.save(payload, ckpt_path)
    return ckpt_path


def train_one_epoch(
    epoch: int,
    mara_agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    clip_model: Optional[torch.nn.Module],
    model: Optional[torch.nn.Module],
    dino_model: Optional[torch.nn.Module],
    prompt_learner: Optional[torch.nn.Module],
    single_tower: Optional[DinoSingleTowerDetector],
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
        "kl": [],
        "clip_frac": [],
        "entropy": [],
        "gate": [],
        "gain_value": [],
        "gain_lower": [],
        "gain_consistency": [],
        "op_aux": [],
        "pred_gain": [],
        "pred_gain_lower": [],
        "accept": [],
        "gain_positive": [],
        "unsafe_accept": [],
        "degrade": [],
        "reward": [],
        "quality": [],
        "gain": [],
        "anchor_gain": [],
    }

    for idx, image_info in enumerate(train_loader):
        with torch.no_grad():
            if single_tower is not None:
                mask, base_map, base_logits, stage1_evidence = forward_dino_single_batch(
                    single_tower,
                    image_info,
                    device,
                )
            else:
                _, mask, base_map, base_logits, stage1_evidence = get_anomaly_map(
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
                    return_evidence=True,
                )

        labels = image_info["is_anomaly"].to(device).long()
        mara_evidence = build_mara_evidence(
            evidence=stage1_evidence,
            fallback_prob=base_map,
            num_layers=len(args.visual_layers),
            map_size=args.mara_map_size,
            feature_channels_per_layer=int(getattr(args, "evidence_feature_channels", 0)),
        )
        del stage1_evidence
        layer_maps = mara_evidence["layer_maps"]
        evidence_maps = None if args.disable_evidence_bank else mara_evidence["extra_maps"]
        global_evidence = None if args.disable_evidence_bank else mara_evidence["global_evidence"]

        group_size = args.grpo_group_size
        mask_g = mask.repeat_interleave(group_size, dim=0)
        labels_g = labels.repeat_interleave(group_size, dim=0)
        mask_bhw = mask_g[:, 0] if mask_g.dim() == 4 else mask_g
        base_map_g = base_map.detach().repeat_interleave(group_size, dim=0)
        normal_weight = (1.0 - labels_g.float())

        apply_gain_gate = epoch >= args.gain_warmup_epochs
        with torch.no_grad():
            behavior = unwrap_mara_agent(mara_agent).rollout(
                base_prob=base_map,
                base_logits=base_logits,
                layer_maps=layer_maps,
                mask=mask,
                labels=labels,
                evidence_maps=evidence_maps,
                global_evidence=global_evidence,
                group_size=group_size,
                sample=True,
                apply_gain_gate=apply_gain_gate,
                force_refine=not apply_gain_gate and not args.disable_force_refine_warmup,
            )
        trajectory = behavior["trajectory"]
        batch_meters = {key: [] for key in meters}

        for _ in range(args.grpo_update_epochs):
            rollout = mara_agent(
                base_prob=base_map,
                base_logits=base_logits,
                layer_maps=layer_maps,
                mask=mask,
                labels=labels,
                evidence_maps=evidence_maps,
                global_evidence=global_evidence,
                group_size=group_size,
                sample=False,
                trajectory=trajectory,
                apply_gain_gate=apply_gain_gate,
            )

            final_map = rollout["final_prob"]
            final_logits = rollout["final_logits"]
            seg_focal = loss_focal(final_map, mask_g)
            seg_dice = loss_dice(final_map[:, 1], mask_bhw)
            seg_loss = seg_focal + seg_dice
            global_loss = F.cross_entropy(final_logits, labels_g)
            drift_per_sample = (final_map[:, 1] - base_map_g[:, 1]).abs().mean(dim=(-2, -1))
            if normal_weight.sum() > 0:
                consistency_loss = (
                    (drift_per_sample * normal_weight).sum()
                    / normal_weight.sum().clamp_min(1.0)
                )
            else:
                consistency_loss = drift_per_sample.mean() * 0.0
            sup_loss = (
                args.w_seg * seg_loss
                + args.w_global * global_loss
                + args.w_base_consistency * consistency_loss
            )

            grpo_loss = rollout["policy_loss"]
            approx_kl = rollout["approx_kl"]
            entropy = rollout["entropy"]
            gate_loss = rollout["gate_l1"]
            gain_value_loss = rollout["gain_loss"]
            gain_lower_loss = rollout["gain_lower_loss"]
            gain_consistency_loss = rollout["gain_consistency_loss"]
            op_aux_loss = rollout["op_aux_loss"]
            rl_scale = 1.0 if apply_gain_gate else 0.0
            total_loss = (
                sup_loss
                + rl_scale * args.w_grpo * (grpo_loss + args.grpo_kl_coef * approx_kl)
                + args.w_gate_sparse * gate_loss
                + args.w_gain_value * gain_value_loss
                + args.w_gain_consistency * gain_consistency_loss
                + args.w_op_aux * op_aux_loss
                - rl_scale * args.entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mara_agent.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            values = {
                "loss": total_loss,
                "sup": sup_loss,
                "seg": seg_loss,
                "global": global_loss,
                "consistency": consistency_loss,
                "grpo": grpo_loss,
                "kl": approx_kl,
                "clip_frac": rollout["clip_fraction"],
                "entropy": entropy,
                "gate": gate_loss,
                "gain_value": gain_value_loss,
                "gain_lower": gain_lower_loss,
                "gain_consistency": gain_consistency_loss,
                "op_aux": op_aux_loss,
                "pred_gain": rollout["mean_predicted_gain"],
                "pred_gain_lower": rollout["mean_predicted_gain_lower"],
                "accept": rollout["mean_accept_score"],
                "gain_positive": rollout["mean_counterfactual_positive"],
                "unsafe_accept": rollout["negative_accept_rate"],
                "degrade": rollout["base_degradation_rate"],
                "reward": rollout["mean_reward"],
                "quality": rollout["mean_quality"],
                "gain": rollout["mean_quality_gain"],
                "anchor_gain": rollout["mean_anchored_quality_gain"],
            }
            for key, value in values.items():
                batch_meters[key].append(float(value.detach().item()))

        for key in meters:
            meters[key].append(float(np.mean(batch_meters[key])))

        if args.is_main_process:
            print(
                f"Epoch {epoch + 1}/{args.epoch} | Batch {idx + 1}/{len(train_loader)} "
                f"| loss {meters['loss'][-1]:.4f} | sup {meters['sup'][-1]:.4f} "
                f"| seg {meters['seg'][-1]:.4f} | global {meters['global'][-1]:.4f} "
                f"| cons {meters['consistency'][-1]:.4f} "
                f"| grpo {meters['grpo'][-1]:.4f} | kl {meters['kl'][-1]:.5f} "
                f"| clip {meters['clip_frac'][-1]:.3f} | gate {meters['gate'][-1]:.4f} "
                f"| gain_v {meters['gain_value'][-1]:.4f} | gain_c {meters['gain_consistency'][-1]:.4f} "
                f"| gain_q {meters['gain_lower'][-1]:.4f} | op_aux {meters['op_aux'][-1]:.4f} "
                f"| pred_g {meters['pred_gain'][-1]:.4f}/{meters['pred_gain_lower'][-1]:.4f} "
                f"| accept {meters['accept'][-1]:.4f} | cf_pos {meters['gain_positive'][-1]:.4f} "
                f"| unsafe {meters['unsafe_accept'][-1]:.4f} | degrade {meters['degrade'][-1]:.4f} "
                f"| entropy {meters['entropy'][-1]:.4f} | reward {meters['reward'][-1]:.4f} "
                f"| gain {meters['gain'][-1]:.4f}/{meters['anchor_gain'][-1]:.4f} "
                f"| quality {meters['quality'][-1]:.4f}",
                end="\r",
                flush=True,
            )

    epoch_metrics = {key: float(np.mean(values)) if values else 0.0 for key, values in meters.items()}
    return reduce_epoch_metrics(epoch_metrics, device, args)


def run_train(args) -> None:
    device = setup_distributed(args)
    set_seed(args.seed + args.rank)
    if args.is_main_process and args.train_split.lower() == "test":
        print(
            "[WARN] MARA is training on split=test with ground-truth masks. "
            "Use this only for the existing supervised/transductive protocol; "
            "a standard unsupervised benchmark needs a separate training/validation protocol."
        )
    if args.is_main_process:
        os.makedirs(args.result_path, exist_ok=True)
    if args.distributed:
        dist.barrier()
    base_payload = load_checkpoint_payload(args.base_ckpt)
    apply_base_runtime_config(args, base_payload)

    if args.is_main_process:
        print(
            "Runtime config: "
            f"base_arch={args.base_arch}, "
            f"visual_backbone={args.visual_backbone}, "
            f"visual_layers={args.visual_layers}, "
            f"hfa_setting={args.hfa_setting_runtime}, "
            f"hfa_layers={args.hfa_layers_runtime}, "
            f"hfa_bottleneck={args.hfa_bottleneck_runtime}, "
            f"evidence_bank={not args.disable_evidence_bank}, "
            f"feature_channels_per_layer={args.evidence_feature_channels}, "
            f"world_size={args.world_size}, per_gpu_batch={args.batch_size}, "
            f"global_batch={args.batch_size * args.world_size}"
        )

    single_tower = None
    clip_model = None
    prompt_learner = None
    dino_model = None
    dino_adapters = None
    model = None
    if args.base_arch == "dino_single":
        single_tower = create_single_tower_from_checkpoint(args, device, base_payload)
        freeze_module(single_tower)
        if args.is_main_process:
            print(f"Loaded language-free DINO single-tower base: {args.base_ckpt}")
    else:
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
        shuffle=not args.distributed,
    )
    train_loader, train_sampler = build_distributed_loader(train_loader, args)

    mara_cfg = build_mara_config(args)
    mara_agent = MARAAgent(mara_cfg).to(device)
    if args.distributed:
        mara_agent = DDP(
            mara_agent,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    optimizer = torch.optim.AdamW(
        mara_agent.parameters(),
        lr=args.mara_lr,
        weight_decay=args.mara_weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(1, args.epoch * len(train_loader) * args.grpo_update_epochs)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=max(1, int(args.warmup_ratio * total_steps)),
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    if args.is_main_process:
        print("MARA config:", mara_cfg)
    for epoch in range(args.epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
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
            single_tower=single_tower,
            train_loader=train_loader,
            device=device,
            args=args,
        )
        if args.is_main_process:
            print()
            ckpt_path = save_mara_checkpoint(
                epoch=epoch,
                save_dir=os.path.join(args.result_path, "ckpt"),
                mara_agent=mara_agent,
                model=model,
                prompt_learner=prompt_learner,
                dino_adapters=dino_adapters,
                single_tower=single_tower,
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
                    f"kl={meters['kl']:.6f}\t"
                    f"clip_frac={meters['clip_frac']:.6f}\t"
                    f"gate={meters['gate']:.6f}\t"
                    f"gain_value={meters['gain_value']:.6f}\t"
                    f"gain_lower={meters['gain_lower']:.6f}\t"
                    f"gain_consistency={meters['gain_consistency']:.6f}\t"
                    f"op_aux={meters['op_aux']:.6f}\t"
                    f"pred_gain={meters['pred_gain']:.6f}\t"
                    f"pred_gain_lower={meters['pred_gain_lower']:.6f}\t"
                    f"accept={meters['accept']:.6f}\t"
                    f"gain_positive={meters['gain_positive']:.6f}\t"
                    f"unsafe_accept={meters['unsafe_accept']:.6f}\t"
                    f"degrade={meters['degrade']:.6f}\t"
                    f"entropy={meters['entropy']:.6f}\t"
                    f"reward={meters['reward']:.6f}\t"
                    f"gain={meters['gain']:.6f}\t"
                    f"anchor_gain={meters['anchor_gain']:.6f}\t"
                    f"quality={meters['quality']:.6f}\n"
                )

            print(
                f"epoch_{epoch}: loss={meters['loss']:.6f}, sup={meters['sup']:.6f}, "
                f"consistency={meters['consistency']:.6f}, grpo={meters['grpo']:.6f}, "
                f"kl={meters['kl']:.6f}, clip={meters['clip_frac']:.6f}, gate={meters['gate']:.6f}, "
                f"gain_value={meters['gain_value']:.6f}, gain_lower={meters['gain_lower']:.6f}, "
                f"gain_consistency={meters['gain_consistency']:.6f}, op_aux={meters['op_aux']:.6f}, "
                f"pred_gain={meters['pred_gain']:.6f}/{meters['pred_gain_lower']:.6f}, "
                f"accept={meters['accept']:.6f}, cf_pos={meters['gain_positive']:.6f}, "
                f"unsafe={meters['unsafe_accept']:.6f}, degrade={meters['degrade']:.6f}, reward={meters['reward']:.6f}, "
                f"gain={meters['gain']:.6f}/{meters['anchor_gain']:.6f}, quality={meters['quality']:.6f}, "
                f"time={time.time() - start:.2f}s, ckpt={ckpt_path}"
            )
        if args.distributed:
            dist.barrier()


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
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)

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
    parser.add_argument("--disable_evidence_bank", action="store_true")
    parser.add_argument("--mara_delta_scale", type=float, default=0.50)
    parser.add_argument("--mara_gate_max", type=float, default=0.35)
    parser.add_argument("--mara_gate_init_bias", type=float, default=-2.0)
    parser.add_argument("--disable_gain_gate", action="store_true")
    parser.add_argument("--gain_accept_threshold", type=float, default=0.0)
    parser.add_argument("--gain_gate_temperature", type=float, default=1.0)
    parser.add_argument("--gain_loss_clip", type=float, default=1.0)
    parser.add_argument("--gain_cls_weight", type=float, default=0.5)
    parser.add_argument("--gain_safety_margin", type=float, default=0.0)
    parser.add_argument("--gain_accept_probability", type=float, default=0.50)
    parser.add_argument("--gain_consistency_temperature", type=float, default=0.05)
    parser.add_argument("--gain_lower_quantile", type=float, default=0.10)
    parser.add_argument("--gain_lower_weight", type=float, default=1.0)
    parser.add_argument("--quality_degradation_tolerance", type=float, default=1e-4)
    parser.add_argument("--disable_gain_lower_bound", action="store_true")
    parser.add_argument("--gain_warmup_epochs", type=int, default=5)
    parser.add_argument("--disable_force_refine_warmup", action="store_true")
    parser.add_argument("--disable_hard_gain_gate", action="store_true")
    parser.add_argument("--force_first_refine", action="store_true")
    parser.add_argument("--disable_base_trajectory", action="store_true")
    parser.add_argument("--mara_lr", type=float, default=1e-4)
    parser.add_argument("--mara_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mara_step_cost", type=float, default=0.001)
    parser.add_argument("--mara_refine_cost", type=float, default=0.002)

    parser.add_argument("--grpo_gamma", type=float, default=0.95)
    parser.add_argument("--grpo_clip_eps", type=float, default=0.2)
    parser.add_argument("--grpo_update_epochs", type=int, default=3)
    parser.add_argument("--grpo_kl_coef", type=float, default=0.01)
    parser.add_argument("--base_anchor_margin", type=float, default=0.0)
    parser.add_argument("--negative_advantage_scale", type=float, default=1.0)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--reward_cls_weight", type=float, default=0.5)
    parser.add_argument("--reward_loc_weight", type=float, default=1.0)
    parser.add_argument("--reward_conf_weight", type=float, default=0.0)
    parser.add_argument("--reward_fp_weight", type=float, default=0.2)

    parser.add_argument("--w_seg", type=float, default=0.7)
    parser.add_argument("--w_global", type=float, default=0.3)
    parser.add_argument("--w_base_consistency", type=float, default=0.05)
    parser.add_argument("--w_grpo", type=float, default=0.5)
    parser.add_argument("--w_gate_sparse", type=float, default=0.001)
    parser.add_argument("--w_gain_value", type=float, default=0.05)
    parser.add_argument("--w_gain_consistency", type=float, default=0.05)
    parser.add_argument("--w_op_aux", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_train(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
