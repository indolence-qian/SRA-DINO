import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted as model_adapter
from tools.utils import get_anomaly_map
from tools.visualization import visualization
from Datasets import DATASET_REGISTRY, DATASET_CLASSES
from tools.bottleneckAdapter import install_bottleneck_adapters_into_dino
from tools.promptLearner import AnomalyCLIP_PromptLearner


use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


def prepare_data(dataset_name, category, args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]

    test_dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=512,
        imagesize=512,
    )

    print(f"Loaded [{dataset_name}] ({category}) test set, size: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    return test_loader


def normalize_minmax(arr):
    arr = arr.astype(np.float32)
    amin = arr.min()
    amax = arr.max()
    return (arr - amin) / (amax - amin + 1e-8)


def normalize_maps(amaps, mode="none"):
    """
    amaps: [N,H,W]
    mode:
        - none: 不归一化（推荐先用这个）
        - per_image: 每张图单独归一化
        - per_class: 整个类别一起归一化
    """
    amaps = amaps.astype(np.float32)

    if mode == "none":
        return amaps

    if mode == "per_class":
        return normalize_minmax(amaps)

    if mode == "per_image":
        out = np.empty_like(amaps, dtype=np.float32)
        for i in range(len(amaps)):
            out[i] = normalize_minmax(amaps[i])
        return out

    raise ValueError(f"Unsupported norm_mode: {mode}")


def compute_best_f1(gt_mask, pred_mask):
    gt_flat = gt_mask.flatten().astype(np.uint8)
    pred_flat = pred_mask.flatten().astype(np.float32)

    if len(np.unique(gt_flat)) < 2:
        return 0.0

    precisions, recalls, _ = precision_recall_curve(gt_flat, pred_flat)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    f1_scores = f1_scores[np.isfinite(f1_scores)]

    if len(f1_scores) == 0:
        return 0.0
    return float(np.max(f1_scores))


def compute_image_level_scores(masks, amaps):
    image_gt = (masks.reshape(masks.shape[0], -1).max(axis=1) > 0).astype(np.uint8)
    image_pred = amaps.reshape(amaps.shape[0], -1).max(axis=1).astype(np.float32)

    image_pred = normalize_minmax(image_pred)
    return image_gt, image_pred


def compute_i_auroc(image_gt, image_pred):
    if len(np.unique(image_gt)) < 2:
        return 0.0
    return float(roc_auc_score(image_gt, image_pred))


def compute_p_auroc(gt_mask, pred_mask):
    gt_flat = gt_mask.flatten().astype(np.uint8)
    pred_flat = pred_mask.flatten().astype(np.float32)

    if len(np.unique(gt_flat)) < 2:
        return 0.0
    return float(roc_auc_score(gt_flat, pred_flat))


def compute_pro(masks, amaps, num_th=1000, max_fpr=0.3, debug=False):
    masks = masks.astype(np.uint8)
    amaps = amaps.astype(np.float32)

    if masks.ndim != 3 or amaps.ndim != 3:
        raise ValueError("compute_pro expects masks and amaps with shape [N,H,W]")

    if masks.shape != amaps.shape:
        raise ValueError(f"Shape mismatch: masks {masks.shape}, amaps {amaps.shape}")

    masks = (masks > 0).astype(np.uint8)

    min_th = float(amaps.min())
    max_th = float(amaps.max())

    if not np.isfinite(min_th) or not np.isfinite(max_th):
        return 0.0

    if max_th - min_th < 1e-12:
        return 0.0

    thresholds = np.linspace(min_th, max_th, num_th, dtype=np.float32)
    inverse_masks = 1 - masks
    denom = inverse_masks.sum().astype(np.float64)

    if denom <= 0:
        return 0.0

    pros = []
    fprs = []

    for th in thresholds:
        binary_amaps = (amaps >= th).astype(np.uint8)

        pro_list = []
        for i in range(len(binary_amaps)):
            gt = masks[i]
            pred = binary_amaps[i]

            num_labels, labels = cv2.connectedComponents(gt)
            for region_id in range(1, num_labels):
                region = (labels == region_id)
                region_area = region.sum()
                if region_area == 0:
                    continue
                overlap = pred[region].sum() / (region_area + 1e-8)
                pro_list.append(float(overlap))

        fp_pixels = np.logical_and(binary_amaps == 1, inverse_masks == 1).sum().astype(np.float64)
        fpr = fp_pixels / (denom + 1e-12)

        if fpr <= max_fpr:
            fprs.append(float(fpr))
            pros.append(float(np.mean(pro_list)) if len(pro_list) > 0 else 0.0)

    if len(fprs) < 2:
        if debug:
            print("[PRO DEBUG] valid points < 2, return 0.0")
        return 0.0

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)

    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    pros = pros[sort_idx]

    uniq_fprs, uniq_indices = np.unique(fprs, return_index=True)
    fprs = uniq_fprs
    pros = pros[uniq_indices]

    if len(fprs) < 2:
        if debug:
            print("[PRO DEBUG] unique valid points < 2, return 0.0")
        return 0.0

    if fprs[0] > 0.0:
        fprs = np.insert(fprs, 0, 0.0)
        pros = np.insert(pros, 0, pros[0])

    if fprs[-1] < max_fpr:
        fprs = np.append(fprs, max_fpr)
        pros = np.append(pros, pros[-1])

    fprs = np.clip(fprs, 0.0, max_fpr)
    pros = np.clip(pros, 0.0, 1.0)

    pro_auc = auc(fprs / max_fpr, pros)

    if debug:
        print(
            f"[PRO DEBUG] "
            f"points={len(fprs)}, "
            f"fpr_min={fprs.min():.6f}, fpr_max={fprs.max():.6f}, "
            f"pro_min={pros.min():.6f}, pro_max={pros.max():.6f}, "
            f"pro_auc={pro_auc:.6f}"
        )

    return float(pro_auc)


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


def normalize_layers_obj(layers_obj):
    if layers_obj is None:
        return ()

    if isinstance(layers_obj, tuple):
        return tuple(int(x) for x in layers_obj)

    if isinstance(layers_obj, list):
        return tuple(int(x) for x in layers_obj)

    if torch.is_tensor(layers_obj):
        return tuple(int(x) for x in layers_obj.detach().cpu().tolist())

    return tuple(int(x) for x in list(layers_obj))


def find_checkpoint_path(weight_dir, epoch_idx):
    candidates = [
        os.path.join(weight_dir, f"{epoch_idx}.pth"),
        os.path.join(weight_dir, f"epoch_{epoch_idx}.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_first_checkpoint(weight_dir, max_epoch=1000):
    for i in range(max_epoch):
        p = find_checkpoint_path(weight_dir, i)
        if p is not None:
            return i, p
    return None, None


def build_models(device, hfa_layers, hfa_bottleneck):
    # loading dinov3
    repo_dir = './dinov3'
    Dinov3_model_path = './dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    Dino_model = torch.hub.load(
        repo_dir,
        'dinov3_vitl16',
        source='local',
        weights=Dinov3_model_path
    )
    Dino_model.eval()
    for p in Dino_model.parameters():
        p.requires_grad_(False)

    if len(hfa_layers) > 0:
        dino_adapters, dino_handles = install_bottleneck_adapters_into_dino(
            Dino_model,
            layers=hfa_layers,
            bottleneck=hfa_bottleneck
        )
        dino_adapters.eval()
    else:
        dino_adapters, dino_handles = None, []

    Dino_model.to(device)
    Dino_model.eval()

    # loading clip
    clip_model = create_model(
        model_name='ViT-L-14-336',
        img_size=512,
        device=device,
        pretrained='openai',
        require_pretrained=True
    )
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # loading prompt learner
    design_details = {
        "Prompt_length": 20,
        "learnabel_text_embedding_length": 4,
        "learnabel_text_embedding_depth": 1,
    }
    prompt_learner = AnomalyCLIP_PromptLearner(
        clip_model.to("cpu"),
        design_details=design_details,
        classname="object"
    )
    prompt_learner.to(device)
    prompt_learner.eval()
    clip_model.to(device)

    # loading AD-DINOv3
    model = model_adapter(c_in=1024, device=device)
    model.to(device)
    model.eval()

    return Dino_model, dino_adapters, clip_model, prompt_learner, model


def load_ckpt_into_models(state, model, prompt_learner, dino_adapters, strict_prompt=True):
    model.patch_token_adapter.load_state_dict(state['patch_token_adapter'])
    model.cls_token_adapter.load_state_dict(state['cls_token_adapter'])
    model.prompt_adapter.load_state_dict(state['prompt_adapter'])

    if 'prompt_learner' in state:
        prompt_learner.load_state_dict(state['prompt_learner'], strict=strict_prompt)

    if dino_adapters is not None:
        if 'dino_adapters' not in state:
            raise KeyError("Current checkpoint does not contain 'dino_adapters', but current test config expects HFA.")
        dino_adapters.load_state_dict(state['dino_adapters'])
    else:
        if 'dino_adapters' in state:
            print("[Warning] Checkpoint contains 'dino_adapters' but current HFA config is none. "
                  "This usually indicates a train/test config mismatch.")


@torch.no_grad()
def test_one_epoch(
    clip_model,
    prompt_learner,
    result_path,
    epoch,
    args,
    device,
    model,
    Dino_model
):
    F1_all = []
    I_AUROC_all = []
    P_AUROC_all = []
    PRO_all = []

    print(f"--------------------------------------Testing epoch {epoch}--------------------------------------")

    for category in sorted(DATASET_CLASSES[args.dataset]):
        os.makedirs(os.path.join(result_path, category), exist_ok=True)

        pixel_pred = []
        pixel_gt = []
        img_list = []

        test_data = prepare_data(args.dataset, category, args, **kwargs)

        for batch_idx, image_info in enumerate(tqdm(test_data)):
            _, mask, anomaly_map_cross_modal, _ = get_anomaly_map(
                clip_model,
                image_info,
                device,
                model,
                Dino_model,
                prompt_learner,
                batch_idx
            )

            mask_np = mask.squeeze(1).cpu().detach().numpy().astype(np.uint8)
            amap_np = anomaly_map_cross_modal[:, 1, :, :].cpu().detach().numpy().astype(np.float32)

            pixel_gt.extend(mask_np)
            pixel_pred.extend(amap_np)
            img_list.extend(image_info["image_path"])

        gt_mask_list = np.array(pixel_gt).astype(np.uint8)
        pred_mask_list = np.array(pixel_pred).astype(np.float32)

        pred_mask_eval = normalize_maps(pred_mask_list, mode=args.norm_mode)

        if args.save_vis:
            vis_mask_list = normalize_maps(pred_mask_list, mode="per_class")
            visualization(img_list, vis_mask_list, gt_mask_list, category, result_path)

        f1 = compute_best_f1(gt_mask_list, pred_mask_eval)
        F1_all.append(f1)

        image_gt, image_pred = compute_image_level_scores(gt_mask_list, pred_mask_eval)
        i_auroc = compute_i_auroc(image_gt, image_pred)
        I_AUROC_all.append(i_auroc)

        p_auroc = compute_p_auroc(gt_mask_list, pred_mask_eval)
        P_AUROC_all.append(p_auroc)

        pro = compute_pro(
            gt_mask_list,
            pred_mask_eval,
            num_th=args.pro_num_th,
            max_fpr=args.pro_max_fpr,
            debug=args.pro_debug
        )
        PRO_all.append(pro)

        if len(np.unique(image_gt)) < 2:
            print(f"[Warning] {category}: image_gt only has one class, I-AUROC is set to 0.0")

        if len(np.unique(gt_mask_list.flatten())) < 2:
            print(f"[Warning] {category}: pixel_gt only has one class, P-AUROC is set to 0.0")

        print(
            f"{category}: "
            f"F1={f1:.5f}\t"
            f"I-AUROC={i_auroc:.5f}\t"
            f"P-AUROC={p_auroc:.5f}\t"
            f"PRO={pro:.5f}"
        )

    mean_f1 = np.mean(F1_all) if len(F1_all) > 0 else 0.0
    mean_i_auroc = np.mean(I_AUROC_all) if len(I_AUROC_all) > 0 else 0.0
    mean_p_auroc = np.mean(P_AUROC_all) if len(P_AUROC_all) > 0 else 0.0
    mean_pro = np.mean(PRO_all) if len(PRO_all) > 0 else 0.0

    print("--------------------------------------------------")
    print(f"mean_F1:      {mean_f1:.5f}")
    print(f"mean_I-AUROC: {mean_i_auroc:.5f}")
    print(f"mean_P-AUROC: {mean_p_auroc:.5f}")
    print(f"mean_PRO:     {mean_pro:.5f}")

    metric_file = os.path.join(result_path, "metric.txt")
    with open(metric_file, "a") as f:
        f.write(
            f"----------Dataset: {args.dataset} | epoch {epoch} | "
            f"hfa_setting={args.hfa_setting} | hfa_layers={args.hfa_layers_runtime}----------\n"
        )
        f.write(f"{'Classname':<14s}{'F1':>10s}{'I-AUROC':>12s}{'P-AUROC':>12s}{'PRO':>10s}\n")
        for i, cname in enumerate(sorted(DATASET_CLASSES[args.dataset])):
            f.write(
                f"{cname:<14s}"
                f"{F1_all[i]:>10.5f}"
                f"{I_AUROC_all[i]:>12.5f}"
                f"{P_AUROC_all[i]:>12.5f}"
                f"{PRO_all[i]:>10.5f}\n"
            )
        f.write(
            f"{'mean':<14s}"
            f"{mean_f1:>10.5f}"
            f"{mean_i_auroc:>12.5f}"
            f"{mean_p_auroc:>12.5f}"
            f"{mean_pro:>10.5f}\n\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result", help="path to result")
    parser.add_argument("--weight_path", type=str, default="./checkpoint/ckpt", help="path to weight folder")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--dataset", type=str, default="mvtec", help="dataset")

    # HFA ablation settings
    parser.add_argument(
        "--hfa_setting",
        type=str,
        default="hfa4",
        choices=["none", "l5", "l11", "l17", "l23", "hfa1", "hfa2", "hfa3", "hfa4"],
        help="fallback HFA setting if checkpoint does not contain hfa metadata"
    )
    parser.add_argument("--hfa_bottleneck", type=int, default=256, help="fallback HFA bottleneck dim")
    parser.add_argument(
        "--auto_hfa_from_ckpt",
        action="store_true",
        help="automatically read hfa_layers / hfa_setting / hfa_bottleneck from the first checkpoint"
    )

    # evaluation normalization
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="none",
        choices=["none", "per_image", "per_class"],
        help="normalization mode for anomaly maps used in evaluation"
    )

    # PRO params
    parser.add_argument("--pro_num_th", type=int, default=1000, help="number of thresholds for PRO")
    parser.add_argument("--pro_max_fpr", type=float, default=0.3, help="max FPR for PRO")
    parser.add_argument("--pro_debug", action="store_true", help="print PRO debug info")

    # misc
    parser.add_argument("--max_epoch", type=int, default=100, help="maximum epochs to scan")
    parser.add_argument("--save_vis", action="store_true", help="save visualization results")
    parser.add_argument("--strict_prompt", action="store_true", help="strict load for prompt learner")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.result_path}/{args.dataset}", exist_ok=True)

    first_epoch, first_ckpt = find_first_checkpoint(args.weight_path, max_epoch=args.max_epoch)
    if first_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in: {args.weight_path}. "
            f"Supported patterns: '{{epoch}}.pth' or 'epoch_{{epoch}}.pth'"
        )

    print(f"[Info] first checkpoint found: epoch={first_epoch}, path={first_ckpt}")
    first_state = torch.load(first_ckpt, map_location=device)

    if args.auto_hfa_from_ckpt and ('hfa_layers' in first_state or 'hfa_setting' in first_state):
        ckpt_hfa_layers = normalize_layers_obj(first_state.get('hfa_layers', None))
        ckpt_hfa_setting = first_state.get('hfa_setting', None)
        ckpt_hfa_bottleneck = int(first_state.get('hfa_bottleneck', args.hfa_bottleneck))

        if len(ckpt_hfa_layers) == 0 and ckpt_hfa_setting is not None:
            ckpt_hfa_layers = resolve_hfa_layers(ckpt_hfa_setting)

        if ckpt_hfa_setting is not None:
            args.hfa_setting = ckpt_hfa_setting
        args.hfa_bottleneck = ckpt_hfa_bottleneck
        hfa_layers = ckpt_hfa_layers

        print(f"[Info] HFA config loaded from checkpoint: setting={args.hfa_setting}, "
              f"layers={hfa_layers}, bottleneck={args.hfa_bottleneck}")
    else:
        hfa_layers = resolve_hfa_layers(args.hfa_setting)
        print(f"[Info] HFA config from args: setting={args.hfa_setting}, "
              f"layers={hfa_layers}, bottleneck={args.hfa_bottleneck}")

    args.hfa_layers_runtime = hfa_layers

    Dino_model, dino_adapters, clip_model, prompt_learner, model = build_models(
        device=device,
        hfa_layers=hfa_layers,
        hfa_bottleneck=args.hfa_bottleneck
    )

    for epoch in range(args.max_epoch):
        ckpt = find_checkpoint_path(args.weight_path, epoch)

        if ckpt is None:
            print(f"[Info] checkpoint not found for epoch {epoch}. Skip.")
            continue

        print(f"[Info] loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)

        # optional consistency check
        if args.auto_hfa_from_ckpt and 'hfa_layers' in state:
            cur_layers = normalize_layers_obj(state['hfa_layers'])
            if tuple(cur_layers) != tuple(hfa_layers):
                raise ValueError(
                    f"HFA layer mismatch across checkpoints. "
                    f"Current runtime layers={hfa_layers}, but checkpoint {ckpt} uses {cur_layers}."
                )

        load_ckpt_into_models(
            state=state,
            model=model,
            prompt_learner=prompt_learner,
            dino_adapters=dino_adapters,
            strict_prompt=args.strict_prompt
        )

        model.eval()
        prompt_learner.eval()
        clip_model.eval()
        Dino_model.eval()
        if dino_adapters is not None:
            dino_adapters.eval()

        test_one_epoch(
            clip_model=clip_model,
            prompt_learner=prompt_learner,
            result_path=f"{args.result_path}/{args.dataset}",
            epoch=epoch,
            args=args,
            device=device,
            model=model,
            Dino_model=Dino_model
        )