import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.backbone.common import ConvNormLayer, get_activation


class RepVggBlock(nn.Module):
    """act(3x3 conv-bn(x) + 1x1 conv-bn(x)); channels unchanged (HF: RTDetrRepVggBlock)."""

    def __init__(self, ch_in: int, ch_out: int, act: str = "silu") -> None:
        super().__init__()
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, padding=1)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, padding=0)
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        br1 = self.conv1(x)
        br2 = self.conv2(x)
        return self.act(br1 + br2)


class CSPRepLayer(nn.Module):
    """CSP fusion block: conv3(bottlenecks(conv1(x)) + conv2(x)) (HF: RTDetrCSPRepLayer, paper Fig. 5 "Fusion")."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden = int(ch_out * expansion)
        self.conv1 = ConvNormLayer(ch_in, hidden, 1, act=act)
        self.conv2 = ConvNormLayer(ch_in, hidden, 1, act=act)
        self.bottlenecks = nn.Sequential(
            *[RepVggBlock(hidden, hidden, act=act) for _ in range(num_blocks)]
        )
        if hidden != ch_out:
            self.conv3 = ConvNormLayer(hidden, ch_out, 1, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        br1 = self.bottlenecks(self.conv1(x))
        br2 = self.conv2(x)
        return self.conv3(br1 + br2)


class CCFM(nn.Module):
    """CNN-based cross-scale fusion: top-down FPN + bottom-up PAN (paper: CCFF).

    Input:  [S3, S4, F5], all `hidden_dim` channels, strides 8 / 16 / 32.
    Output: [P3, N4, N5], same shapes as the input.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_levels: int = 3,
        expansion: float = 1.0,
        num_blocks: int = 3,
        act: str = "silu",
    ) -> None:
        super().__init__()
        num_stages = num_levels - 1
        self.lateral_convs = nn.ModuleList(
            [ConvNormLayer(hidden_dim, hidden_dim, 1, act=act) for _ in range(num_stages)]
        )
        self.fpn_blocks = nn.ModuleList(
            [CSPRepLayer(2 * hidden_dim, hidden_dim, num_blocks, expansion, act) for _ in range(num_stages)]
        )
        self.downsample_convs = nn.ModuleList(
            [ConvNormLayer(hidden_dim, hidden_dim, 3, stride=2, act=act) for _ in range(num_stages)]
        )
        self.pan_blocks = nn.ModuleList(
            [CSPRepLayer(2 * hidden_dim, hidden_dim, num_blocks, expansion, act) for _ in range(num_stages)]
        )

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        # Feature inputs (unrolled for the 3 RT-DETR levels)
        s3, s4, f5 = feats

        # FPN
        # 1st block
        y5 = self.lateral_convs[0](f5)
        up2_y5 = F.interpolate(y5, scale_factor=2.0, mode="nearest")
        x4 = self.fpn_blocks[0](torch.cat([up2_y5, s4], dim=1))
        # 2nd block
        y4 = self.lateral_convs[1](x4)
        up2_y4 = F.interpolate(y4, scale_factor=2.0, mode="nearest")
        p3 = self.fpn_blocks[1](torch.cat([up2_y4, s3], dim=1))

        # PAN (re-uses the lateral outputs y4, y5)
        n4 = self.pan_blocks[0](torch.cat([self.downsample_convs[0](p3), y4], dim=1))
        n5 = self.pan_blocks[1](torch.cat([self.downsample_convs[1](n4), y5], dim=1))

        return [p3, n4, n5]
