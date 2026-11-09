import torch.nn as nn
from torch import Tensor

from src.encoder.aifi import AIFILayer
from src.encoder.ccfm import CCFM


class HybridEncoder(CCFM):
    """Input proj → AIFI on S5 → CCFF (FPN + PAN). [C3, C4, C5] → [P3, N4, N5] (HF: RTDetrHybridEncoder
    + RTDetrModel.encoder_input_proj)."""

    def __init__(
        self,
        in_channels: tuple[int, ...] = (512, 1024, 2048),
        hidden_dim: int = 256,
        feat_strides: tuple[int, ...] = (8, 16, 32),
        encode_proj_layers: tuple[int, ...] = (2,),
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_encoder_layers: int = 1,
        enc_act: str = "gelu",
        expansion: float = 1.0,
        act: str = "silu",
        temperature: float = 10000.0,
    ) -> None:
        super().__init__(hidden_dim, len(in_channels), expansion, act=act)
        self.encoder_input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=hidden_dim, kernel_size=1, bias=False
            ), nn.BatchNorm2d(hidden_dim)) for in_c in in_channels
        ])
        self.aifi = nn.ModuleList([
            AIFILayer(hidden_dim, num_heads, ffn_dim, num_encoder_layers, enc_act, temperature) for _ in range(len(encode_proj_layers))
        ])
        self.encode_proj_layers = encode_proj_layers
        self.in_channels = in_channels
        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = list(feat_strides)

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        if len(feats) != len(self.in_channels):
            raise ValueError(f"Number of features should match input channel count! {len(feats)}!={len(self.in_channels)}")
        enc_feats = [
            self.encoder_input_proj[idx](feats[idx]) for idx in range(len(feats))
        ]
        for i, idx in enumerate(self.encode_proj_layers):
            enc_feats[idx] = self.aifi[i](enc_feats[idx])
        return super().forward(enc_feats)
