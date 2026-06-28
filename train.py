import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm
import os
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted as model_adapter
import Datasets.visa as visa
import Datasets.mvtec as mvtec
import time
import argparse
from tools.loss import FocalLoss, BinaryDiceLoss
from tools.utils import get_anomaly_map, compute_reward_and_weight, dice_loss_per_sample
from torch.optim.lr_scheduler import LambdaLR
import math
from Datasets import DATASET_REGISTRY, DATASET_CLASSES
from tools.bottleneckAdapter import install_bottleneck_adapters_into_dino

from tools.promptLearner import AnomalyCLIP_PromptLearner

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def grad_norm_stats(named_params):
    total, cnt, max_abs = 0.0, 0, 0.0
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.norm(2).item() ** 2
        max_abs = max(max_abs, g.abs().max().item())
        cnt += 1
    return (math.sqrt(total) if cnt else 0.0), max_abs, cnt

set_seed(0)

visa_ALL = {"candle", "capsules", "cashew", "chewinggum",
            "fryum", "macaroni1", "macaroni2", "pcb1",
            "pcb2", "pcb3", "pcb4", "pipe_fryum"}

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)

def _downsample_mask_to_tokens(mask_bchw: torch.Tensor, S: int) -> torch.Tensor:
    if mask_bchw.dim() == 3:
        mask_bchw = mask_bchw.unsqueeze(1)
    m_small = F.interpolate(mask_bchw.float(), size=(S, S), mode="nearest")
    return (m_small > 0.5).squeeze(1)

def prepare_data(dataset_name, category, args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"❌ Unsupported dataset: {dataset_name}. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]

    test_dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=512,
        imagesize=512,
    )

    print(f"✅ Loaded [{dataset_name}] ({category}) train set, size: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    return test_loader

# 没有调用奖励函数的train_epoch
# def train_epoch(optimizer, loss_focal, loss_dice, epoch, anomaly_awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list, clip_model, start_time, train_data, prompt_learner):
#     for idx, image_info in enumerate(train_data):
#         anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score = get_anomaly_map(clip_model, image_info, device, model, Dino_model, prompt_learner,idx)
#         anomaly_awareness_loss = loss_focal(anomaly_map, mask) + loss_dice(anomaly_map[:, 1, :, :], mask)
#         seg_loss = loss_focal(anomaly_map_cross_modal, mask) + loss_dice(anomaly_map_cross_modal[:, 1, :, :], mask)
#         global_anomaly_loss = F.cross_entropy(global_anomaly_score.squeeze(1), image_info["is_anomaly"].to(device).long())
#         loss = 0.25 * anomaly_awareness_loss + 0.5 * seg_loss + 0.25 * global_anomaly_loss
#         print(
#                 f"Epoch {epoch+1}/{10} | Batch {idx+1}/{len(train_data)} "
#                 f"| loss: {loss.item():.4f} | anomaly_awareness_loss: {anomaly_awareness_loss.item():.4f} | seg_loss: {seg_loss.item():.4f} | global_anomaly_loss: {global_anomaly_loss.item():.4f} | Time: {time.time()-start_time:.2f}s",
#                 end="\r",
#                 flush=True
#         )
#         anomaly_awareness_loss_list.append(anomaly_awareness_loss.item())
#         seg_loss_list.append(seg_loss.item())
#         global_anomaly_loss_list.append(global_anomaly_loss.item())
#         loss_list.append(loss.item())
#         loss.requires_grad_(True)
#         optimizer.zero_grad()
#         loss.backward()
        
#         if idx % 50 == 0:
#             g_adapter = grad_norm_stats([(n,p) for n,p in model.named_parameters()])
#             g_prompt  = grad_norm_stats([(n,p) for n,p in prompt_learner.named_parameters()])
#             print(f"\n[grad] adapter norm={g_adapter[0]:.3e} max={g_adapter[1]:.3e} n={g_adapter[2]} | "
#                   f"prompt norm={g_prompt[0]:.3e} max={g_prompt[1]:.3e} n={g_prompt[2]}")
#         # no_grad = []
#         # has_grad = []
#         # for n, p in prompt_learner.named_parameters():
#         #     if p.requires_grad:
#         #         if p.grad is None:
#         #             no_grad.append(n)
#         #         else:
#         #             has_grad.append(n)

#         # print("HAS GRAD examples:", has_grad[:10])
#         # print("NO GRAD:", no_grad)
#         optimizer.step()
#         scheduler.step()
#     return 

def train_epoch(
    optimizer, loss_focal, loss_dice, epoch,
    anomaly_awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list,
    clip_model, start_time, train_data, prompt_learner,
    reward_state
):
    for idx, image_info in enumerate(train_data):
        anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score = get_anomaly_map(
            clip_model, image_info, device, model, Dino_model, prompt_learner, idx
        )

        # --- awareness（不动） ---
        anomaly_awareness_loss = loss_focal(anomaly_map, mask) + loss_dice(anomaly_map[:, 1, :, :], mask)

        # --- seg: focal（标量）不动 + dice（每样本）加权 ---
        if mask.dim() == 4:
            mask_bhw = mask[:, 0]
        else:
            mask_bhw = mask

        seg_focal = loss_focal(anomaly_map_cross_modal, mask_bhw)

        # cross modal prob (B,H,W)
        cm = anomaly_map_cross_modal
        if cm.min() < 0 or cm.max() > 1:
            cm_prob = torch.softmax(cm, dim=1)[:, 1]
        else:
            cm_prob = cm[:, 1].clamp(0, 1)

        seg_dice_per = dice_loss_per_sample(cm_prob, mask_bhw)  # (B,)

        # --- global ---
        gt_label = image_info["is_anomaly"].to(device).long()
        logits = global_anomaly_score.squeeze(1)  # (B,2)
        ce_per_sample = F.cross_entropy(logits, gt_label, reduction="none")  # (B,)

        # --- reward weights ---
        w_cls, w_seg, r_mean, baseline, dice_per, reward_state = compute_reward_and_weight(
            global_logits=logits,
            anomaly_map_cross_modal=anomaly_map_cross_modal,
            mask=mask,
            gt_label=gt_label,
            reward_state=reward_state,
            w_loc=1.0, w_conf=0.2,
            alpha_adv_cls=0.05,
            seg_k=3.0,                    # ★你可以改成 5.0 更强
        )
        
        # 冻结全局seg
        w_seg = torch.ones_like(w_seg)

        global_anomaly_loss = (ce_per_sample * w_cls).mean()

        # ★关键：对分割 dice loss 做样本级加权（真正增强像素分割作用）
        seg_dice = (seg_dice_per * w_seg).mean()
        seg_loss = seg_focal + seg_dice

        loss = 0.25 * anomaly_awareness_loss + 0.5 * seg_loss + 0.25 * global_anomaly_loss

        print(
            f"Epoch {epoch+1}/{10} | Batch {idx+1}/{len(train_data)} "
            f"| loss: {loss.item():.4f} | aw: {anomaly_awareness_loss.item():.4f} "
            f"| seg(focal+dice_w): {seg_focal.item():.4f}+{seg_dice.item():.4f}={seg_loss.item():.4f} "
            f"| global(w): {global_anomaly_loss.item():.4f} "
            f"| reward_mean: {r_mean:.3f} | baseline: {baseline:.3f} "
            f"| dice mean: {dice_per.mean().item():.3f} "
            f"| w_seg mean/min/max: {w_seg.mean().item():.3f}/{w_seg.min().item():.3f}/{w_seg.max().item():.3f} "
            f"| Time: {time.time()-start_time:.2f}s",
            end="\r",
            flush=True
        )

        anomaly_awareness_loss_list.append(anomaly_awareness_loss.item())
        seg_loss_list.append(seg_loss.item())
        global_anomaly_loss_list.append(global_anomaly_loss.item())
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        if idx % 50 == 0:
            g_adapter = grad_norm_stats([(n, p) for n, p in model.named_parameters()])
            g_prompt  = grad_norm_stats([(n, p) for n, p in prompt_learner.named_parameters()])
            print(f"\n[grad] adapter norm={g_adapter[0]:.3e} max={g_adapter[1]:.3e} n={g_adapter[2]} | "
                  f"prompt norm={g_prompt[0]:.3e} max={g_prompt[1]:.3e} n={g_prompt[2]}")

        optimizer.step()
        scheduler.step()

    return reward_state

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

def resolve_hfa_layers(hfa_setting: str):
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
    key = hfa_setting.lower()
    if key not in preset:
        raise ValueError(
            f"Unsupported hfa_setting={hfa_setting}. "
            f"Available: {list(preset.keys())}"
        )
    return preset[key]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result", help="path to result")
    parser.add_argument("--device", type=str, default="cuda:6", help="device")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--dataset", type=str, default="visa", help="dataset")
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--lr", type=float, default=0.00001, help="lr")
    parser.add_argument("--hfa_setting", type=str, default="hfa4",help="HFA setting: none/l5/l11/l17/l23/hfa1/hfa2/hfa3/hfa4")
    parser.add_argument("--hfa_bottleneck", type=int, default=256,help="bottleneck dim for HFA")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.result_path}", exist_ok=True)
    
    # loading dinov3
    repo_dir = './dinov3'
    Dinov3_model_path = './dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    Dino_model = torch.hub.load(repo_dir, 'dinov3_vitl16', source = 'local', weights = Dinov3_model_path)
    Dino_model.eval()
    for p in Dino_model.parameters():
        p.requires_grad_(False)
     
    # dino adapters
    # dino_adapters, dino_handles = install_bottleneck_adapters_into_dino(Dino_model, layers=(5,11,17,23), bottleneck=256)
    # dino_adapters.train()
    # Dino_model.to(device)
    hfa_layers = resolve_hfa_layers(args.hfa_setting)
    print(f"==> HFA setting: {args.hfa_setting}, layers={hfa_layers}, bottleneck={args.hfa_bottleneck}")

    if len(hfa_layers) > 0:
        dino_adapters, dino_handles = install_bottleneck_adapters_into_dino(
            Dino_model,
            layers=hfa_layers,
            bottleneck=args.hfa_bottleneck
        )
        dino_adapters.train()
    else:
        dino_adapters, dino_handles = None, []

    Dino_model.to(device)
    
    # loading clip
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device=device, pretrained='openai', require_pretrained=True)
    
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)   # 冻结权重，但仍允许建立计算图

    # loading prompt learner
    design_details = {
        "Prompt_length": 20,
        "learnabel_text_embedding_length": 4,
        "learnabel_text_embedding_depth": 1, # 关闭 compound prompts
    }
    prompt_learner = AnomalyCLIP_PromptLearner(clip_model.to("cpu"), design_details=design_details, classname="object")
    prompt_learner.to(device)
    prompt_learner.train()
    clip_model.to(device)

    # AD-DINOv3
    model = model_adapter(c_in=1024, device=device)
    model.to(device)
    model.train()

    params_to_update = []
    # 设置 adapter 需要更新的参数
    adapter_update_params = ['patch_token_adapter', 'cls_token_adapter', 'prompt_adapter']
    for name, param in model.named_parameters():
        print(f"Model parameter: {name}")
        for update_name in adapter_update_params:
            if update_name in name:
                params_to_update.append(param)

    # 设置 prompt learner 需要更新的参数
    prompt_update_params = ["ctx_pos", "ctx_neg"]   # 你想训练的字段名
    for name, param in prompt_learner.named_parameters():
        print(f"PromptLearner parameter: {name}")
        if any(k in name for k in prompt_update_params):
            params_to_update.append(param)

    train_data = prepare_data(args.dataset, 'ALL', args, **kwargs)

    # optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    prompt_params  = [p for p in prompt_learner.parameters() if p.requires_grad]
    dino_adapter_params = [p for p in dino_adapters.parameters() if p.requires_grad] if dino_adapters is not None else []

    optim_groups = [
        {"params": adapter_params, "lr": 1e-5, "weight_decay": 1e-2},
        {"params": prompt_params,  "lr": 5e-5, "weight_decay": 1e-3},
    ]

    if len(dino_adapter_params) > 0:
        optim_groups.insert(1, {"params": dino_adapter_params, "lr": 5e-6, "weight_decay": 1e-2})

    optimizer = torch.optim.AdamW(
        optim_groups,
        betas=(0.9, 0.999)
    )
    # optimizer = torch.optim.AdamW([
    #     {"params": adapter_params,      "lr": 1e-5, "weight_decay": 1e-2},
    #     {"params": dino_adapter_params, "lr": 5e-6, "weight_decay": 1e-2},
    #     {"params": prompt_params,       "lr": 5e-5, "weight_decay": 1e-3},
    # ], betas=(0.9, 0.999))


    total_steps = args.epoch * len(train_data)
    warmup_steps = int(0.05 * total_steps)
    prompt_stop_epoch = 10

    # scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.05
    )

    # If you want to speed up the convergence, please use the following line of code.
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch/10 + 1))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    reward_state = None

    
    for epoch in range(args.epoch):
        # epoch从0开始：0~9 共10个epoch训练；从epoch==10开始冻结
        if epoch == prompt_stop_epoch:
            prompt_learner.eval()
            for p in prompt_learner.parameters():
                p.requires_grad_(False)
        
        start_time = time.time()
        awareness_loss_list, seg_loss_list, loss_list, global_anomaly_loss_list = [], [], [], []

        # train_epoch(optimizer, loss_focal, loss_dice, epoch, awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list, clip_model, start_time, train_data, prompt_learner)
        
        reward_state = train_epoch(optimizer, loss_focal, loss_dice, epoch, awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list, clip_model, start_time, train_data, prompt_learner, reward_state)
        print()
        # scheduler.step()

        os.makedirs(f"{args.result_path}/ckpt", exist_ok=True)
        ckpt = {
            'cls_token_adapter': model.cls_token_adapter.state_dict(),
            'patch_token_adapter': model.patch_token_adapter.state_dict(),
            'prompt_adapter': model.prompt_adapter.state_dict(),
            'prompt_learner': prompt_learner.state_dict(),
            'hfa_setting': args.hfa_setting,
            'hfa_layers': hfa_layers,
            'hfa_bottleneck': args.hfa_bottleneck,
        }

        if dino_adapters is not None:
            ckpt['dino_adapters'] = dino_adapters.state_dict()

        torch.save(ckpt, f"{args.result_path}/ckpt/{epoch}.pth")
        
        with open(f"{args.result_path}/loss.txt", "a") as f:
            f.write(
                f"epoch_{epoch}: "
                f"awareness_loss={np.mean(awareness_loss_list):.6f}\t"
                f"seg_loss={np.mean(seg_loss_list):.6f}\t"
                f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}\t"
                f"total_loss={np.mean(loss_list):.6f}\n"
            )
        print(
            f"epoch_{epoch}: "
            f"awareness_loss={np.mean(awareness_loss_list):.6f}, "
            f"seg_loss={np.mean(seg_loss_list):.6f}, "
            f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}, "
            f"total_loss={np.mean(loss_list):.6f}"
        )