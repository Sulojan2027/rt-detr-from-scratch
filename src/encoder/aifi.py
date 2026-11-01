import torch
import torch.nn as nn
from torch import Tensor

from src.backbone.common import get_activation
from src.encoder.position_encoding import SinePositionEmbedding


class MLP(nn.Module):
    """fc1 → act → fc2 (HF: RTDetrMLP)."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 1024, act: str = "gelu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
        )
        self.fc2 = nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size
        )
        self.activation_fn = get_activation(activation=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class SelfAttention(nn.Module):
    """Multi-head self-attention; pos added to Q and K only (HF: RTDetrSelfAttention)."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size({hidden_size}) is not divisble by number of heads({num_heads}).")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: Tensor, pos: Tensor | None = None) -> Tensor:
        # x: [B, N, D], pos: [1, N, D] or None → [B, N, D]
        qk = x if pos is None else x + pos
        q = self.q_proj(qk)
        k = self.k_proj(qk)
        v = self.v_proj(x)
        
        B, N, _ = x.shape
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Score
        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores * self.scaling
        attn_weights = scaled_scores.softmax(dim=-1)

        # Attention
        attn_scores = attn_weights @ v
        attn_scores = attn_scores.transpose(1, 2).reshape(B, N, self.hidden_size)

        # Projection
        out = self.o_proj(attn_scores)
        return out


class EncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer (HF: RTDetrEncoderLayer)."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        act: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size=ffn_dim, act=act)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: Tensor, pos: Tensor | None = None) -> Tensor:
        x = self.self_attn_layer_norm(x + self.self_attn(x, pos))
        out = self.final_layer_norm(x + self.mlp(x))
        return out


class AIFILayer(nn.Module):
    """[B, C, H, W] → tokens → encoder layer(s) + sin-cos pos → [B, C, H, W] (HF: RTDetrAIFILayer)."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 1,
        act: str = "gelu",
        temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        self.position_embedding = SinePositionEmbedding(
            embed_dim=hidden_size,
            temperature=temperature,
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(hidden_size, num_heads, ffn_dim, act))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        pos = self.position_embedding(H, W, device=x.device, dtype=x.dtype)
        x = x.flatten(2).transpose(1, 2)            # [B, C, H, W] → [B, H*W, C]
        for layer in self.layers:
            x = layer(x, pos)
        return x.transpose(1, 2).reshape(B, C, H, W)  # back to a map