import torch
import numpy as np
from copy import deepcopy
from CLIP.tokenizer import _tokenizer, tokenize
import torch.nn as nn

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details, classname: str = "object"):
        super().__init__()

        # ★★ 这里我加了一个 classname 参数，这样就可以传 mvtec 的类别名进来
        classnames = [classname]

        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]

        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames

        # normal / anomaly 的状态模板
        self.state_normal_list = ["{}"]
        self.state_anomaly_list = ["damaged {}"]

        normal_num = len(self.state_normal_list)
        anomaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anomaly_num = anomaly_num

        # ---- 初始化 learnable context ----
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx

        ctx_vectors_pos = torch.empty(
            self.n_cls, normal_num, n_ctx_pos, ctx_dim, dtype=dtype
        )
        ctx_vectors_neg = torch.empty(
            self.n_cls, anomaly_num, n_ctx_neg, ctx_dim, dtype=dtype
        )
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # (n_cls, normal_num, n_ctx_pos, dim)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # (n_cls, anomaly_num, n_ctx_neg, dim)

        # ---- 计算 prefix/suffix 的静态 embedding（用当前 clip_model 的 token_embedding）----
        classnames = [name.replace("_", " ") for name in classnames]

        prompts_pos = [
            " ".join(["X"] * n_ctx_pos) + " " + template.format(name) + "."
            for template in self.state_normal_list
            for name in classnames
        ]
        prompts_neg = [
            " ".join(["X"] * n_ctx_neg) + " " + template.format(name) + "."
            for template in self.state_anomaly_list
            for name in classnames
        ]

        tokenized_prompts_pos = torch.cat([tokenize(p) for p in prompts_pos])
        tokenized_prompts_neg = torch.cat([tokenize(p) for p in prompts_neg])

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)

            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anomaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        # prefix: CLS token；suffix: 类名 + EOS 等
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos :, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg :, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anomaly_num, self.n_cls, d).permute(1, 0, 2)

        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)

        # 复合可学习文本（如果你暂时不想用，也可以先保留不动）
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

    def forward(self):
        # 组装 final prompt embedding + token ids
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        # (n_cls, normal_num, L, D)
        prompts_pos = torch.cat([prefix_pos, ctx_pos, suffix_pos], dim=2)
        prompts_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=2)

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)  # (normal_num * n_cls, L, D)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)

        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)

        return prompts, tokenized_prompts