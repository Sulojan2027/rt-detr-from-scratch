from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from .common import ConvNormLayer, get_activation

_VARIANTS = ("b", "d")

def _make_shortcut(ch_in: int, ch_out: int, stride: int, variant: str) -> nn.Module:
    """Projection shortcut. ch_out here = the block's TRUE output channels."""
    if variant == "d" and stride == 2:
        return nn.Sequential(
            OrderedDict(
                    [
                        ("pool", nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)),
                        ("conv", ConvNormLayer(ch_in, ch_out, 1, 1))
                    ]
                )
        )
    return ConvNormLayer(ch_in, ch_out, 1, stride)

def _validate_args(ch_in: int, ch_out: int, stride: int, shortcut: bool, variant: str) -> None:
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
        # Validate arguments
        _validate_args(ch_in, ch_out, stride, shortcut, variant)
        
        self.shortcut = shortcut
        if not shortcut:
            self.short = _make_shortcut(ch_in, ch_out, stride, variant)
        
        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=3,
            stride=stride,
            act=act
        )
        
        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
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

class BottleNeck(nn.Module):
    expansion = 4  # block output channels = ch_out * expansion

    def __init__(
        self,
        ch_in: int,
        ch_out: int,             # bottleneck WIDTH, not the block output
        stride: int,
        shortcut: bool,          # True → identity, False → projection
        act: str | None = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        
        width = ch_out
        out = ch_out * self.expansion
        _validate_args(ch_in, ch_out*self.expansion, stride, shortcut, variant)
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride=1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride, act=act)
        self.branch2c = ConvNormLayer(width, out, 1, 1, act=None)
        self.shortcut = shortcut
        if not self.shortcut:
            self.short = _make_shortcut(ch_in, ch_out=out, stride=stride, variant=variant)
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        out_ = self.branch2c(self.branch2b(self.branch2a(x)))
        short = x if self.shortcut else self.short(x)
        return self.act(out_ + short)

class Blocks(nn.Module):
    """One ResNet stage: `count` blocks; only the first may downsample / project."""

    def __init__(
        self,
        block: type[nn.Module],  # BasicBlock or BottleNeck (the class itself)
        ch_in: int,
        ch_out: int,             # width passed to each block
        count: int,
        stride: int,             # stride of the FIRST block (1 or 2)
        act: str | None = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        if count < 1:
            raise ValueError(
                f"Counts should be >= 1, but got {count}"
            )
        blocks_list = [
            block(ch_in, ch_out, stride=stride, shortcut=False, act=act, variant=variant),
        ]
        ch_in = ch_out * block.expansion
        blocks_list.extend(
            block(ch_in, ch_out, stride=1, shortcut=True, act=act, variant=variant) for _ in range(count - 1)
        )
        self.blocks = nn.Sequential(*blocks_list)
        self.out_channels = ch_out * block.expansion

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

# depth → number of blocks per stage
_DEPTH_CFG: dict[int, list[int]] = {
    # TODO: 18, 34, 50, 101
}


class PResNet(nn.Module):
    """ResNet-vd backbone. Returns a list of feature maps for `return_idx` stages."""

    def __init__(
        self,
        depth: int = 50,
        variant: str = "d",
        return_idx: tuple[int, ...] = (1, 2, 3),
        act: str = "relu",
        freeze_at: int = -1,
        freeze_norm: bool = True,
    ) -> None:
        super().__init__()
        # TODO: validate depth, return_idx
        # TODO: pick block type + block counts
        # TODO: stem self.conv1 (named Sequential: conv1_1, conv1_2, conv1_3)
        # TODO: self.res_layers (ModuleList of 4 Blocks), chaining ch_in
        # TODO: self.return_idx, self.out_channels, self.out_strides
        # TODO: freezing (freeze_at, then freeze_norm)

    def forward(self, x: Tensor) -> list[Tensor]:
        # TODO: stem → maxpool → stages, collect return_idx outputs
        ...