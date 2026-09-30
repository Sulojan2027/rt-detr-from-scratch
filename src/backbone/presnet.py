from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from common import ConvNormLayer, get_activation

_VARIANTS = ("b", "d")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int,
        shortcut: bool,          # True → identity, False → projection
        act: str | None = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(
                f"Variant must be one of {_VARIANTS}, got '{variant}'"
            )
        if stride not in (1, 2):
            raise ValueError(
                f"Stride must be 1 or 2, got {stride}"
            )
        if shortcut and (stride != 1 or ch_in != ch_out):
            raise ValueError(
                f"Identity shortcut needs stride=1 and ch_in == ch_out, "
                f"got stride={stride}, {ch_in}->{ch_out}"
            )
        
        
        self.shortcut = shortcut
        if not shortcut:
            if stride == 2 and variant == "d":
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1))
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        
        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=3,
            stride=stride,
            act=act
        )
        
        self.branch2b = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=3,
            stride=1,
            act=None
        )
        self.act = get_activation(act)
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)
    