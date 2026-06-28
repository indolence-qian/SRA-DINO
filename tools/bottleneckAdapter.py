import torch
import torch.nn as nn

class BottleneckAdapter1024(nn.Module):
    """
    x: (..., 1024)
    -> down: (..., r)
    -> act
    -> up: (..., 1024)
    -> residual: x + scale * up(act(down(x)))
    """
    def __init__(self, dim=1024, bottleneck=256, init_scale=1e-3):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act  = nn.LeakyReLU(inplace=False)
        self.up   = nn.Linear(bottleneck, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(self.act(self.down(x)))
    
def install_bottleneck_adapters_into_dino(Dino_model, layers=(5,11,17,23), dim=1024, bottleneck=256, init_scale=1e-3):
    adapters = nn.ModuleDict({
        str(i): BottleneckAdapter1024(dim=dim, bottleneck=bottleneck, init_scale=init_scale)
        for i in layers
    })
    # 推荐：注册成子模块，方便 .to(device)
    Dino_model._peft_adapters = adapters

    handles = []

    for i in layers:
        adapter = adapters[str(i)]
        def _hook(module, inp, out, adapter=adapter):
            # out 必须是 Tensor；如果不是，先取出真正的 Tensor
            if isinstance(out, (tuple, list)):
                out = out[0]  # 保险：但通常不应该发生
            return adapter(out)  # ✅ 永远返回 Tensor
        handles.append(Dino_model.blocks[i].register_forward_hook(_hook))

    return adapters, handles