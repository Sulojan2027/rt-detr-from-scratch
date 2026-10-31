import torch
import torch.nn as nn
from torch import Tensor


def build_2d_sinusoidal_position_embedding(
    height: int,
    width: int,
    embed_dim: int = 256,
    temperature: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Returns [height*width, embed_dim], layout [sin_h | cos_h | sin_w | cos_w], row-major tokens."""

    if embed_dim % 4 != 0:
        raise ValueError(f"Embed dim is not valid, got {embed_dim}")
    pos_dim = embed_dim // 4
    omega = 1.0 / temperature ** (torch.arange(pos_dim, dtype=torch.float64, device=device) / pos_dim)
    grid_h, grid_w = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij"
    )
    emb_h, emb_w = grid_h.flatten().outer(omega), grid_w.flatten().outer(omega)
    pos_embed = torch.concat((torch.sin(emb_h), torch.cos(emb_h), torch.sin(emb_w), torch.cos(emb_w)), dim=1)
    return pos_embed.to(dtype)


class SinePositionEmbedding(nn.Module):
    """No weights. HF: RTDetrSinePositionEmbedding (AIFI's `position_embedding`)."""

    def __init__(self, embed_dim: int = 256, temperature: float = 10000.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, height: int, width: int, device=None, dtype=torch.float32) -> Tensor:
        return build_2d_sinusoidal_position_embedding(
            height, width, self.embed_dim, self.temperature,
            device, dtype
        ).unsqueeze(0)