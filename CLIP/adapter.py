import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torchvision.models as models

class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class CLIP_Inplanted(nn.Module):
    def __init__(self, c_in, device):
        super().__init__()
        self.device = device
        self.cls_token_adapter = nn.ModuleList([ClipAdapter(c_in = 1024) for _ in range(4)])
        self.prompt_adapter = nn.ModuleList([ClipAdapter(c_in = 768) for _ in range(2)])
        self.patch_token_adapter = nn.ModuleList([ClipAdapter(c_in = 1024) for _ in range(4)])

    def forward(self,):
        return
# class BottleneckProjector(nn.Module):
#     """
#     x: (..., C_in)
#     -> down: (..., r)
#     -> act
#     -> up: (..., C_out)
#     residual: only when C_in == C_out
#     """
#     def __init__(self, c_in: int, c_out: int, bottleneck: int = 256,
#                  residual: bool = False, init_scale: float = 1e-3):
#         super().__init__()
#         self.residual = residual and (c_in == c_out)

#         self.down = nn.Linear(c_in, bottleneck, bias=False)
#         self.act  = nn.LeakyReLU(inplace=False)
#         self.up   = nn.Linear(bottleneck, c_out, bias=False)

#         # 小尺度起步更稳（尤其你冻结了 backbone）
#         self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         z = self.act(self.down(x))
#         y = self.up(z)

#         if self.residual:
#             return x + self.scale * y
#         else:
#             return self.scale * y

# class CLIP_Inplanted(nn.Module):
#     def __init__(self, c_in, device):
#         super().__init__()
#         self.device = device

#         # DINO: 1024 -> r -> 768（你最终要跟 text 的 768 对齐）
#         self.cls_token_adapter = nn.ModuleList([
#             BottleneckProjector(c_in=1024, c_out=768, bottleneck=256, residual=False)
#             for _ in range(4)
#         ])
#         self.patch_token_adapter = nn.ModuleList([
#             BottleneckProjector(c_in=1024, c_out=768, bottleneck=256, residual=False)
#             for _ in range(4)
#         ])

#         # text: 768 -> r -> 768（可 residual）
#         self.prompt_adapter = nn.ModuleList([
#             BottleneckProjector(c_in=768, c_out=768, bottleneck=128, residual=True)
#             for _ in range(2)
#         ])

#     def forward(self):
#         return