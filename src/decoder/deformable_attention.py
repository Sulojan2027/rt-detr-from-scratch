"""Multi-scale deformable cross-attention (decoder queries → encoder features).  [PERSON B — skeleton]

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

import torch.nn as nn
from torch import Tensor


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
    # Hints (follow HF's MultiScaleDeformableAttention.forward line by line — reshapes are the whole difficulty):
    # 1. value.split([H*W for H, W in spatial_shapes], dim=1) → one [B, H*W, h, d] tensor per level.
    # 2. grid_sample wants coords in [-1, 1]: sampling_grids = 2 * sampling_locations - 1.
    # 3. Per level l:
    #       v = value_l.flatten(2).transpose(1, 2).reshape(B*h, d, H, W)          ← heads folded into batch
    #       g = sampling_grids[:, :, :, l].transpose(1, 2).flatten(0, 1)          → [B*h, Q, P, 2]
    #       s = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)  → [B*h, d, Q, P]
    # 4. Weights: attention_weights.transpose(1, 2).reshape(B*h, 1, Q, L*P)
    # 5. out = (torch.stack(samples, dim=-2).flatten(-2) * weights).sum(-1).view(B, h*d, Q).transpose(1, 2)
    # Gotcha: align_corners=False + the "2x-1" mapping must match HF exactly, or every box shifts by half a pixel.
    raise NotImplementedError


class MultiscaleDeformableAttention(nn.Module):
    """Weights + offset/weight prediction around `multi_scale_deformable_attn` (HF: RTDetrMultiscaleDeformableAttention)."""

    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        # Hints:
        # 1. Validate d_model % n_heads == 0 (same as your SelfAttention).
        # 2. Store d_model, n_levels, n_heads, n_points; create the 4 Linear layers with the HF names above.
        # 3. No custom init needed — weights come from the checkpoint. (For training from scratch, HF/official
        #    init sampling_offsets bias to a ring of directions; see RTDetrPreTrainedModel._init_weights.)
        raise NotImplementedError

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
        position_embeddings: Tensor | None = None,
    ) -> Tensor:
        # Hints:
        # 1. query = hidden_states + position_embeddings (if given). Offsets + weights come from the QUERY,
        #    values come from the ENCODER memory.
        # 2. value = value_proj(encoder_hidden_states).view(B, S, h, d)
        # 3. offsets = sampling_offsets(query).view(B, Q, h, L, P, 2)
        # 4. weights = attention_weights(query).view(B, Q, h, L*P).softmax(-1).view(B, Q, h, L, P)
        #    ← softmax over ALL levels*points together, not per level.
        # 5. 4-coord reference boxes (RT-DETR always uses 4):
        #       loc = ref[:, :, None, :, None, :2] + offsets / P * ref[:, :, None, :, None, 2:] * 0.5
        #    i.e. offsets are scaled by half the box size → points land inside / around the box.
        # 6. out = multi_scale_deformable_attn(value, spatial_shapes, loc, weights); return output_proj(out)
        raise NotImplementedError
