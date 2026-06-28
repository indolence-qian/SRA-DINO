import torch
import torch.nn.functional as F
from pathlib import Path
import math
from utils import encode_text_with_prompt_ensemble, get_text_features_with_prompt_learner

def get_feature_dinov3(image_path, batch_img, device, Dino_model, layers=(5, 11, 17, 23)):
    """
    PEFT-compatible:
      - 不用 no_grad / inference_mode
      - 不 detach / 不 cpu
      - 一次 forward 整个 batch
    返回:
      cls_token:    List[Tensor] len=4, each (B,1,1024)
      patch_tokens: List[Tensor] len=4, each (B,N,1024) 其中 N = tokens-5
    """
    anchor = getattr(Dino_model, "norm", None) or getattr(Dino_model, "fc_norm", None)
    if anchor is None:
        raise RuntimeError("There is no norm/fc_norm module, please print(Dino_model) to confirm the name")

    tokens_dict = {}
    handles = []

    def _extract_x(out):
        return out[0] if isinstance(out, (tuple, list)) else out

    for i in layers:
        def _mk_hook(idx):
            def _hook(module, inp, out):
                x = _extract_x(out)      # (B, 1+N, 1024) 这里已经包含“注入后的输出”
                tokens_dict[idx] = anchor(x)
            return _hook
        handles.append(Dino_model.blocks[i].register_forward_hook(_mk_hook(i)))

    batch_img = batch_img.to(device, non_blocking=True)
    _ = Dino_model(batch_img)  # ✅ 不能 inference_mode/no_grad

    for h in handles:
        h.remove()

    cls_token_list = []
    patch_tokens_list = []

    for i in layers:
        if i not in tokens_dict:
            raise RuntimeError(f"Hook did not capture layer {i}. Check block indices and model structure.")

        toks = tokens_dict[i]           # (B, 1+N, 1024)
        patch = toks[:, 5:, :]          # (B, N, 1024)
        patch = (patch - patch.mean(dim=1, keepdim=True)) / (patch.std(dim=1, keepdim=True) + 1e-6)
        cls = toks[:, 0, :].unsqueeze(1)  # (B,1,1024)

        patch_tokens_list.append(patch)
        cls_token_list.append(cls)

    return cls_token_list, patch_tokens_list


def get_anomaly_map(clip_model, image_info, device, model, Dino_model, prompt_learner, idx):
    image = image_info["image"].to(device, non_blocking=True)
    image_path = image_info["image_path"]
    mask = image_info["mask"].to(device, non_blocking=True)
    mask = (mask > 0.5).float()
    y = image_info["is_anomaly"]  # 如果后续计算用到，可考虑 .to(device)

    B = image.shape[0]
    out_h, out_w = image.shape[-2:]

    # --------------------
    # textual branch
    # --------------------
    # 更稳：别硬编码 768（如果你确定永远是 768，也可以留着）
    # 这里沿用你的写法
    text_feature = torch.zeros(B, 768, 2, device=device)

    for i in range(B):
        # 更稳的取类别名方式（避免 split('/') 在 Windows 路径下出错）
        cls_name = Path(image_path[i]).parts[-4]
        # 注意：你这里传 y，若 encode 期望每样本标签，可能应该传 y[i] 而不是 y
        # text_feature[i] = encode_text_with_prompt_ensemble(clip_model, cls_name, device, '', y)
        # 提示词版本
        text_feature[i] = get_text_features_with_prompt_learner(clip_model=clip_model, prompt_learner=prompt_learner, device=device, adapter=None)  # shape: (768, 2)


    # prompt_adapter 向量化
    # f0 = model.prompt_adapter[0](text_feature[:, :, 0])  # (B,768)
    # f1 = model.prompt_adapter[1](text_feature[:, :, 1])  # (B,768)
    # adjusted_text_feature = torch.stack([f0, f1], dim=-1)    # (B,768,2)
    f0_768, _ = model.prompt_adapter[0](text_feature[:, :, 0])  # f0_768: (B,768)
    f1_768, _ = model.prompt_adapter[1](text_feature[:, :, 1])  # f1_768: (B,768)
    adjusted_text_feature = torch.stack([f0_768, f1_768], dim=-1)  # (B,768,2) 

    # 可选：如果你希望点积更像 cosine，相似度更稳定，可以 normalize 一下文本
    # adjusted_text_feature = F.normalize(adjusted_text_feature, dim=1)

    if idx % 50 == 0:
        t_norm = adjusted_text_feature[:, :, 0]
        t_abn  = adjusted_text_feature[:, :, 1]
        cos_sim = F.cosine_similarity(t_norm, t_abn, dim=1)
        print(
            "cos_sim mean/min/max:",
            cos_sim.mean().item(),
            cos_sim.min().item(),
            cos_sim.max().item()
        )

    # --------------------
    # visual branch
    # --------------------
    cls_token, patch_tokens = get_feature_dinov3(image_path, image, device, Dino_model)

    num_scales = len(cls_token)  # 不要硬编码 4，更稳
    sum_cross_modal = None
    sum_awareness   = None
    sum_global_score = None

    for i in range(num_scales):
        # cls_features = model.cls_token_adapter[i](cls_token[i])
        # patch_features = model.patch_token_adapter[i](patch_tokens[i])
        cls_features, _ = model.cls_token_adapter[i](cls_token[i])
        patch_features, _ = model.patch_token_adapter[i](patch_tokens[i])

        cls_features = F.normalize(cls_features, dim=-1)        # (B,768) 或 (B,1,768)
        patch_features = F.normalize(patch_features, dim=-1)    # (B,N,768)

        # 确保 cls_features 是 (B,768)
        if cls_features.dim() == 3 and cls_features.shape[1] == 1:
            cls_features = cls_features[:, 0, :]

        # 推断 patch 网格尺寸
        N = patch_features.shape[1]
        side = int(math.sqrt(N))
        # 如果 N 不是完全平方数，这里建议 assert 或用更通用的 reshape 方式
        # assert side * side == N

        # ---- cross-modal contrastive learning ----
        # (B,N,768) x (B,768,2) -> (B,N,2)
        cross_logits = 100.0 * torch.bmm(patch_features, adjusted_text_feature)
        cross_logits = cross_logits.permute(0, 2, 1).reshape(B, 2, side, side)
        cross_logits = F.interpolate(
            cross_logits, size=(out_h, out_w),
            mode='bilinear', align_corners=True
        )
        cross_prob = torch.softmax(cross_logits, dim=1)

        if sum_cross_modal is None:
            sum_cross_modal = cross_prob
        else:
            sum_cross_modal = sum_cross_modal + cross_prob

        # ---- anomaly-aware calibration ----
        # (B,N,768) x (B,768,1) -> (B,N,1)
        cls_vec = cls_features.unsqueeze(-1)  # (B,768,1)
        patch_cls_logits = 10.0 * torch.bmm(patch_features, cls_vec)
        patch_cls_logits = patch_cls_logits.permute(0, 2, 1).reshape(B, 1, side, side)
        patch_cls_logits = F.interpolate(
            patch_cls_logits, size=(out_h, out_w),
            mode='bilinear', align_corners=True
        )
        patch_cls_prob = torch.sigmoid(patch_cls_logits)
        aw = torch.cat([1.0 - patch_cls_prob, patch_cls_prob], dim=1)  # (B,2,H,W)

        if sum_awareness is None:
            sum_awareness = aw
        else:
            sum_awareness = sum_awareness + aw

        # ---- global anomaly score (修正形状) ----
        # (B,1,768) x (B,768,2) -> (B,1,2) -> (B,2)
        global_score = 100.0 * torch.bmm(cls_features.unsqueeze(1), adjusted_text_feature).squeeze(1)

        if sum_global_score is None:
            sum_global_score = global_score
        else:
            sum_global_score = sum_global_score + global_score

    anomaly_map_cross_modal = sum_cross_modal / num_scales
    anomaly_awareness = sum_awareness / num_scales
    global_anomaly_score = sum_global_score / num_scales

    return anomaly_awareness, mask, anomaly_map_cross_modal, global_anomaly_score


# 注重分类成功奖励
# def compute_reward_and_weight(
#     global_logits,              # (B,2) logits
#     anomaly_map_cross_modal,     # (B,2,H,W) prob/logits均可（下面会sigmoid/softmax兼容）
#     mask,                       # (B,H,W) or (B,1,H,W) 0/1
#     gt_label,                   # (B,) long 0/1
#     reward_state=None,          # 用于EMA baseline，跨batch保持
#     w_loc=1.0,                  # 定位奖励权重
#     w_conf=0.2,                 # 置信度奖励权重
#     alpha_adv=0.10,             # advantage影响样本权重强度
#     adv_clip=3.0,               # advantage裁剪
#     weight_clip=(0.2, 3.0),     # 最终权重裁剪
#     ema_beta=0.95,              # baseline EMA
#     eps=1e-6
# ):
#     """
#     返回：
#     - weight: (B,) 用于加权CE
#     - reward_mean: float（可选监控）
#     - baseline: float（可选监控）
#     """
#     if reward_state is None:
#         reward_state = {"baseline": 0.0}

#     with torch.no_grad():
#         # --- 分类奖励：连续，避免阈值+/-1 ---
#         p = torch.softmax(global_logits, dim=-1)  # (B,2)
#         p_correct = p.gather(1, gt_label.unsqueeze(1)).squeeze(1)  # (B,)
#         r_cls = 2.0 * p_correct - 1.0  # [-1,1]

#         # --- 定位奖励：soft dice（仅异常样本） ---
#         if mask.dim() == 4:
#             mask_bhw = mask[:, 0]
#         else:
#             mask_bhw = mask
#         mask_bhw = mask_bhw.float()

#         # anomaly_map_cross_modal 可能是logits(softmax前)或概率(softmax后)
#         # 你训练里用 focal+dice，通常它是概率图（softmax后），但这里做个健壮处理：
#         if anomaly_map_cross_modal.shape[1] == 2:
#             # 若值域可能不在[0,1]，做softmax得到概率
#             if anomaly_map_cross_modal.min() < 0 or anomaly_map_cross_modal.max() > 1:
#                 prob_map = torch.softmax(anomaly_map_cross_modal, dim=1)[:, 1]
#             else:
#                 prob_map = anomaly_map_cross_modal[:, 1].clamp(0, 1)
#         else:
#             # 如果是 (B,1,H,W) 一类
#             prob_map = anomaly_map_cross_modal.squeeze(1).sigmoid()

#         B = prob_map.shape[0]
#         p_flat = prob_map.reshape(B, -1)
#         g_flat = mask_bhw.reshape(B, -1)

#         inter = (p_flat * g_flat).sum(dim=1)
#         denom = p_flat.sum(dim=1) + g_flat.sum(dim=1)
#         dice = (2.0 * inter + eps) / (denom + eps)  # (B,) in [0,1]
#         r_loc = dice * gt_label.float()

#         # --- 置信度奖励：只在预测正确时给，避免“错得更自信” ---
#         pred = p.argmax(dim=-1)
#         correct = (pred == gt_label).float()
#         entropy = -(p * (p + 1e-8).log()).sum(dim=-1)        # (B,)
#         ent_norm = entropy / math.log(2.0)                   # 二分类最大熵log(2)
#         r_conf = (1.0 - ent_norm) * correct                  # (B,) in [0,1]

#         reward = r_cls + w_loc * r_loc + w_conf * r_conf     # (B,)

#         # --- baseline + advantage ---
#         batch_mean = reward.mean().item()
#         baseline = reward_state["baseline"]
#         baseline = ema_beta * baseline + (1.0 - ema_beta) * batch_mean
#         reward_state["baseline"] = baseline

#         adv = reward - baseline
#         # 标准化降低方差（可选但推荐）
#         adv = adv / (reward.std().clamp_min(1e-6))
#         adv = adv.clamp(-adv_clip, adv_clip)

#         # adv高 => weight低(让loss更小) => 更强化这些样本的正确方向
#         weight = (1.0 - alpha_adv * adv).clamp(weight_clip[0], weight_clip[1])  # (B,)

#     return weight, float(reward.mean().item()), float(reward_state["baseline"]), reward_state

# 注重分割成功奖励
def compute_reward_and_weight(
    global_logits,              # (B,2) logits
    anomaly_map_cross_modal,     # (B,2,H,W) prob/logits
    mask,                       # (B,H,W) or (B,1,H,W)
    gt_label,                   # (B,) long 0/1
    reward_state=None,

    # reward 组成（主要用于 monitoring 或 w_cls）
    w_loc=1.0,
    w_conf=0.2,

    # cls 权重（建议弱一些）
    alpha_adv_cls=0.05,
    cls_weight_clip=(0.5, 2.0),
    ema_beta=0.95,

    # seg 权重（重点：单调映射，强）
    seg_k=3.0,                  # ★关键：越大越强调分割差的样本（建议先 3，再试 5）
    seg_weight_clip=(0.5, 5.0),

    eps=1e-6
):
    """
    返回：
    - w_cls: (B,) 给 global CE 用（弱）
    - w_seg: (B,) 给分割 dice 用（强，且不会总贴近1）
    - reward_mean: float
    - baseline: float
    - dice: (B,) 方便你监控（可选）
    - reward_state
    """
    if reward_state is None:
        reward_state = {"baseline": 0.0}

    with torch.no_grad():
        # ---- 1) global 分类 reward（连续） ----
        p = torch.softmax(global_logits, dim=-1)  # (B,2)
        p_correct = p.gather(1, gt_label.unsqueeze(1)).squeeze(1)  # (B,)
        r_cls = 2.0 * p_correct - 1.0  # [-1,1]

        # ---- 2) 取 cross-modal anomaly prob map ----
        if mask.dim() == 4:
            mask_bhw = mask[:, 0]
        else:
            mask_bhw = mask
        mask_bhw = mask_bhw.float()

        if anomaly_map_cross_modal.shape[1] == 2:
            if anomaly_map_cross_modal.min() < 0 or anomaly_map_cross_modal.max() > 1:
                prob_map = torch.softmax(anomaly_map_cross_modal, dim=1)[:, 1]
            else:
                prob_map = anomaly_map_cross_modal[:, 1].clamp(0, 1)
        else:
            prob_map = anomaly_map_cross_modal.squeeze(1).sigmoid()

        # ---- 3) soft dice（每样本） ----
        B = prob_map.shape[0]
        p_flat = prob_map.reshape(B, -1)
        g_flat = mask_bhw.reshape(B, -1)

        inter = (p_flat * g_flat).sum(dim=1)
        denom = p_flat.sum(dim=1) + g_flat.sum(dim=1)
        dice = (2.0 * inter + eps) / (denom + eps)  # (B,) in [0,1]

        # 定位奖励只对异常样本给
        r_loc = dice * gt_label.float()

        # ---- 4) 置信度 reward（只在预测正确时） ----
        pred = p.argmax(dim=-1)
        correct = (pred == gt_label).float()
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
        ent_norm = entropy / math.log(2.0)
        r_conf = (1.0 - ent_norm) * correct  # [0,1]

        reward = r_cls + w_loc * r_loc + w_conf * r_conf  # (B,)

        # ---- 5) w_cls（弱）用 baseline 做一点稳定加权 ----
        batch_mean = reward.mean().item()
        baseline = reward_state["baseline"]
        baseline = ema_beta * baseline + (1.0 - ema_beta) * batch_mean
        reward_state["baseline"] = baseline

        adv_cls = reward - baseline
        # 不标准化也行；如果你想稳定一点可以除 std，这里保持简单
        w_cls = (1.0 - alpha_adv_cls * adv_cls).clamp(cls_weight_clip[0], cls_weight_clip[1])

        # ---- 6) w_seg（强）：单调映射，强调“分割差”的异常样本 ----
        # seg_bad ∈ [0,1]，分割越差越大；正常样本置0（不强调）
        seg_bad = (1.0 - dice).clamp(0, 1) * gt_label.float()  # (B,)

        # ★核心：不会围绕1抖动，差的样本直接给大权重
        w_seg = 1.0 + seg_k * seg_bad
        w_seg = w_seg.clamp(seg_weight_clip[0], seg_weight_clip[1])

    return w_cls, w_seg, float(reward.mean().item()), float(baseline), dice, reward_state

def dice_loss_per_sample(prob_map, mask, eps=1e-6):
    """
    prob_map: (B,H,W) in [0,1]
    mask:     (B,H,W) 0/1
    return:   (B,) dice_loss = 1 - dice
    """
    B = prob_map.shape[0]
    p = prob_map.reshape(B, -1)
    g = mask.reshape(B, -1).float()
    inter = (p * g).sum(dim=1)
    denom = p.sum(dim=1) + g.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice
