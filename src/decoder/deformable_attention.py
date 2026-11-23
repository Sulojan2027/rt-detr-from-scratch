"""Multi-scale deformable cross-attention (decoder queries → encoder features).

Paper: RT-DETR §4.3 (decoder from Deformable DETR, arXiv:2010.04159 §4.1).
HF reference: `modeling_rt_detr.py` — `MultiScaleDeformableAttention` (pure-PyTorch core) and
`RTDetrMultiscaleDeformableAttention` (the module with weights).

Idea in one line: instead of attending to all 8400 encoder tokens, every query looks at only
n_heads × n_levels × n_points = 8 × 3 × 4 = 96 points, placed around its reference box by learned offsets.

HF names (must match for strict loading; prefix `model.decoder.layers.N.encoder_attn.`):
    sampling_offsets   Linear(256, n_heads*n_levels*n_points*2 = 192)
    attention_weights  Linear(256, n_heads*n_levels*n_points   = 96)
    value_proj         Linear(256, 256)
    output_proj        Linear(256, 256)
Self-check: params = 205,600.

Shapes (B batch, Q queries = 300 (+ denoising), S = sum(H_l*W_l) = 8400 at 640², L levels = 3, h heads = 8,
d = 256 / 8 = 32, P points = 4):
    hidden_states          [B, Q, 256]        decoder queries
    position_embeddings    [B, Q, 256]        from query_pos_head(reference boxes)
    encoder_hidden_states  [B, S, 256]        flattened P3|N4|N5 after decoder_input_proj
    reference_points       [B, Q, 1, 4]       sigmoid boxes (cx, cy, w, h) in [0, 1]; the 1 broadcasts over levels
    spatial_shapes         [(80, 80), (40, 40), (20, 20)]  list of (H, W) per level
    → output               [B, Q, 256]
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def multi_scale_deformable_attn(
    value: Tensor,
    spatial_shapes: list[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Bilinear-sample the value maps at the sampling locations and take the weighted sum.

    value               [B, S, h, d]
    sampling_locations  [B, Q, h, L, P, 2]   (x, y) in [0, 1]
    attention_weights   [B, Q, h, L, P]      softmaxed over L*P
    → [B, Q, h*d]
    """
    B, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    # The tokens of all levels sit in one long sequence; cut it back into one chunk per level.
    value_per_level = value.split([H * W for H, W in spatial_shapes], dim=1)

    # grid_sample expects coordinates in [-1, 1], ours are in [0, 1].
    sampling_grids = 2 * sampling_locations - 1

    sampled_per_level = []
    for level, (H, W) in enumerate(spatial_shapes):
        # [B, H*W, h, d] → [B, h, d, H, W] → [B*h, d, H, W]  (heads are folded into the batch,
        # so every head can sample its own points in one grid_sample call)
        level_value = value_per_level[level].flatten(2).transpose(1, 2).reshape(B * num_heads, head_dim, H, W)

        # [B, Q, h, P, 2] → [B, h, Q, P, 2] → [B*h, Q, P, 2]
        level_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # Bilinear read at every sampling point → [B*h, d, Q, P]
        sampled = F.grid_sample(
            level_value,
            level_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_per_level.append(sampled)

    # [B, Q, h, L, P] → [B, h, Q, L, P] → [B*h, 1, Q, L*P], so it lines up with the samples below.
    attention_weights = attention_weights.transpose(1, 2).reshape(B * num_heads, 1, num_queries, num_levels * num_points)

    # [B*h, d, Q, L, P] → [B*h, d, Q, L*P], weight every sample, then sum over all L*P points.
    samples = torch.stack(sampled_per_level, dim=-2).flatten(-2)
    out = (samples * attention_weights).sum(dim=-1)

    # [B*h, d, Q] → [B, h*d, Q] → [B, Q, h*d]
    return out.view(B, num_heads * head_dim, num_queries).transpose(1, 2).contiguous()


class MultiscaleDeformableAttention(nn.Module):
    """Weights + offset/weight prediction around `multi_scale_deformable_attn` (HF: RTDetrMultiscaleDeformableAttention)."""

    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) is not divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # Every head, level and point gets an (x, y) offset and one attention weight.
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
        position_embeddings: Tensor | None = None,
    ) -> Tensor:
        # hidden_states [B, Q, D], encoder_hidden_states [B, S, D], reference_points [B, Q, 1, 4] → [B, Q, D]
        # Offsets and weights are predicted from the query (with its position added), values from the memory.
        query = hidden_states if position_embeddings is None else hidden_states + position_embeddings

        B, num_queries, _ = query.shape
        S = encoder_hidden_states.shape[1]
        if sum(H * W for H, W in spatial_shapes) != S:
            raise ValueError("spatial_shapes do not add up to the number of encoder tokens.")

        head_dim = self.d_model // self.n_heads
        value = self.value_proj(encoder_hidden_states).view(B, S, self.n_heads, head_dim)

        offsets = self.sampling_offsets(query).view(B, num_queries, self.n_heads, self.n_levels, self.n_points, 2)

        # One softmax over all levels and points together (not one per level).
        weights = self.attention_weights(query).view(B, num_queries, self.n_heads, self.n_levels * self.n_points)
        weights = weights.softmax(dim=-1).view(B, num_queries, self.n_heads, self.n_levels, self.n_points)

        # Reference box is (cx, cy, w, h). The centre is where we start; the offsets are scaled by the
        # box size, so a bigger object spreads its sampling points over a bigger area.
        ref_center = reference_points[:, :, None, :, None, :2]  # [B, Q, 1, L or 1, 1, 2]
        ref_size = reference_points[:, :, None, :, None, 2:]
        sampling_locations = ref_center + offsets / self.n_points * ref_size * 0.5

        out = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, weights)
        return self.output_proj(out)
