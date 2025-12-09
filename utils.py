import numpy as np
import torch
import torch.nn.functional as F
import kornia as K
from PIL import Image
from CLIP.tokenizer import tokenize as tokenize


def encode_text_with_prompt_ensemble(model, obj, device, cnn, adapter):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
                
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(text = prompted_sentence, cnn = cnn, device = device, adapter = adapter)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features

def get_text_features_with_prompt_learner(clip_model, prompt_learner, device, adapter=None):
    """
    返回形状大概为 [2, C] 的文本特征：
      [0] -> normal prompt embedding
      [1] -> anomaly prompt embedding
    具体 shape 要看你后面怎么 reshape，这里先给个思路
    """
    # prompts: (num_prompts, L, D)
    # tokenized_prompts: (num_prompts, context_len)
    prompts, tokenized_prompts = prompt_learner()

    # 这里要对应你 CLIP 的 encode_text 实现。
    # 如果 AD-DINOv3 的 clip_model.encode_text 还只能接受 token ids，
    # 那你就需要在 CLIP 里加一个新的接口，专门处理已经嵌入好的 prompts。
    # 伪代码结构大概这样：

    # ---- 以下是伪代码，和你原来的 encode_text 对应着改 ----
    x = prompts               # 已经是 token embedding 了
    x = x + clip_model.positional_embedding
    x = clip_model.transformer(x)
    x = x[0] 
    x = clip_model.ln_final(x)
    # 取每个序列里的 [EOS] 位置（从 tokenized_prompts 里找 eot 的 index）
    eos_indices = tokenized_prompts.argmax(dim=-1)
    text_embeddings = x[torch.arange(x.shape[0]), eos_indices] @ clip_model.text_projection

    # 现在 text_embeddings 的条数 = normal prompt 数 + anomaly prompt 数
    # 你的 PromptLearner 里 normal_num = 1, anomaly_num = 1，
    # 可以把前一半当 normal，后一半当 anomaly，再做平均 + 归一化
    num_total = text_embeddings.shape[0]
    num_half = num_total // 2

    normal_embeds = text_embeddings[:num_half]
    anomaly_embeds = text_embeddings[num_half:]

    normal_feat = normal_embeds / normal_embeds.norm(dim=-1, keepdim=True)
    normal_feat = normal_feat.mean(dim=0)
    normal_feat = normal_feat / normal_feat.norm()

    anomaly_feat = anomaly_embeds / anomaly_embeds.norm(dim=-1, keepdim=True)
    anomaly_feat = anomaly_feat.mean(dim=0)
    anomaly_feat = anomaly_feat / anomaly_feat.norm()

    text_features = torch.stack([normal_feat, anomaly_feat], dim=1)  # [C, 2]
    return text_features.to(device)  

def cos_sim(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x

def vflip_img(x):
    x = K.geometry.transform.vflip(x)
    return x

def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x


def augment(fewshot_img, fewshot_mask=None):

    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        augment_fewshot_mask = fewshot_mask

        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

            rotate_mask = rot_img(fewshot_mask, angle)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, rotate_mask], dim=0)
        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

            trans_mask = translation_img(fewshot_mask, a, b)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, trans_mask], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
        flipped_mask = hflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        flipped_mask = vflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

    else:
        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        B, _, H, W = augment_fewshot_img.shape
        augment_fewshot_mask = torch.zeros([B, 1, H, W])
    
    return augment_fewshot_img, augment_fewshot_mask

