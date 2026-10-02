from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from .common import ConvNormLayer, get_activation, freeze_batch_norm2d

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

# depth → number of blocks in each of the 4 stages
_DEPTH_CFG: dict[int, list[int]] = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
}
_STEM_CH = 64                         # stem output channels (ResNet standard)
_STAGE_WIDTHS = [64, 128, 256, 512]   # width passed to the blocks of each stage
_STAGE_STRIDES = [4, 8, 16, 32]       # total stride after each stage (stem + pool = 4)
_BOTTLENECK_MIN_DEPTH = 50            # depth >= 50 uses BottleNeck, else BasicBlock


class PResNet(nn.Module):
    """ResNet-vd backbone ("PResNet" in the official RT-DETR repo).

    Returns the feature maps of the stages in `return_idx`.
    Default (1, 2, 3) → [C3, C4, C5] at strides [8, 16, 32].
    """

    def __init__(
        self,
        depth: int = 50,
        variant: str = "d",
        return_idx: tuple[int, ...] = (1, 2, 3),
        act: str = "relu",
        freeze_at: int = -1,      # -1: none, 0: stem, k: stem + first k stages
        freeze_norm: bool = True,
    ) -> None:
        super().__init__()
        num_stages = len(_STAGE_WIDTHS)
        if depth not in _DEPTH_CFG:
            raise ValueError(f"depth must be one of {sorted(_DEPTH_CFG)}, got {depth}")
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got '{variant}'")
        return_idx = tuple(return_idx)
        if (
            not return_idx
            or any(i < 0 or i >= num_stages for i in return_idx)
            or list(return_idx) != sorted(set(return_idx))
        ):
            raise ValueError(
                f"return_idx must be non-empty, strictly increasing, in [0, {num_stages - 1}], "
                f"got {return_idx}"
            )

        block = BottleNeck if depth >= _BOTTLENECK_MIN_DEPTH else BasicBlock

        # ---- Stem ----
        # (name, ch_in, ch_out, kernel, stride)
        if variant == "d":
            # ResNet-D: three 3x3 convs instead of one 7x7 (same view, more non-linearity)
            stem_def = [
                ("conv1_1", 3, _STEM_CH // 2, 3, 2),
                ("conv1_2", _STEM_CH // 2, _STEM_CH // 2, 3, 1),
                ("conv1_3", _STEM_CH // 2, _STEM_CH, 3, 1),
            ]
        else:
            stem_def = [("conv1_1", 3, _STEM_CH, 7, 2)]
        self.conv1 = nn.Sequential(
            OrderedDict(
                (name, ConvNormLayer(c_in, c_out, k, stride=s, act=act))
                for name, c_in, c_out, k, s in stem_def
            )
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # stride 2 → 4

        # ---- Stages ----
        self.res_layers = nn.ModuleList()
        all_channels = []
        ch_in = _STEM_CH
        for i, (width, count) in enumerate(zip(_STAGE_WIDTHS, _DEPTH_CFG[depth], strict=True)):
            stage = Blocks(
                block,
                ch_in,
                width,
                count=count,
                stride=1 if i == 0 else 2,  # stage 0: the max pool already downsampled
                act=act,
                variant=variant,
            )
            self.res_layers.append(stage)
            ch_in = stage.out_channels
            all_channels.append(stage.out_channels)

        # ---- Interface contract (Person B relies on these) ----
        self.return_idx = return_idx
        self.out_channels = [all_channels[i] for i in return_idx]
        self.out_strides = [_STAGE_STRIDES[i] for i in return_idx]

        # ---- Freezing (after everything is built) ----
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])
        if freeze_norm:
            freeze_batch_norm2d(self)  # root is not a BN → children replaced in place

    @staticmethod
    def _freeze_parameters(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.pool(self.conv1(x))
        outs = []
        for i, stage in enumerate(self.res_layers):
            x = stage(x)
            if i in self.return_idx:
                outs.append(x)
        return outs
