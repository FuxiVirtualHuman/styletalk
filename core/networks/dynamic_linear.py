import math

import torch
from torch import nn
from torch.nn import functional as F

from core.networks.dynamic_conv import DynamicConv


class DynamicLinear(nn.Module):
    def __init__(self, in_planes, out_planes, cond_planes, bias=True, K=4, temperature=30, ratio=4, init_weight=True):
        super().__init__()

        self.dynamic_conv = DynamicConv(
            in_planes,
            out_planes,
            cond_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            K=K,
            ratio=ratio,
            temperature=temperature,
            init_weight=init_weight,
        )

    def forward(self, x, cond):
        """

        Args:
            x (_type_): (L, B, C_in)
            cond (_type_): (B, C_style)

        Returns:
            _type_: (L, B, C_out)
        """
        x = x.permute(1, 2, 0).unsqueeze(-1)
        out = self.dynamic_conv(x, cond)
        # (B, C_out, L, 1)
        out = out.squeeze().permute(2, 0, 1)
        return out
