"""Transformer decoder with iterative box refinement.  [PERSON B — skeleton]

Paper: RT-DETR §4.3 / Fig. 2 (decoder + auxiliary prediction heads).
HF reference: `modeling_rt_detr.py` — `RTDetrMLPPredictionHead`, `RTDetrDecoderLayer`, `RTDetrDecoder`, and
`RTDetrForObjectDetection.__init__` (which attaches `class_embed` / `bbox_embed` to the decoder).

HF names (prefix `model.decoder.`):
    layers.N.self_attn.{q,k,v,o}_proj        ← same names as Person A's `src.encoder.aifi.SelfAttention`
    layers.N.self_attn_layer_norm
    layers.N.encoder_attn.*                  ← MultiscaleDeformableAttention
    layers.N.encoder_attn_layer_norm
    layers.N.mlp.fc1 / fc2                   ← same names as `src.encoder.aifi.MLP` (act = "relu" here, not gelu)
    layers.N.final_layer_norm
    query_pos_head.layers.{0,1}              MLPPredictionHead(4, 512, 256, num_layers=2)
    class_embed.N                            Linear(256, 80)                    one per decoder layer
    bbox_embed.N.layers.{0,1,2}              MLPPredictionHead(256, 256, 4, 3)   one per decoder layer
Number of layers: R18 = 3, R50 = 6 (hf.config.decoder_layers).

Self-check params: MLPPredictionHead(256,256,4,3) = 132,612 | query_pos_head = 133,888 | one DecoderLayer = 995,872
                   Decoder total: R18 3,581,020 | R50 7,028,152
"""

import torch.nn as nn
from torch import Tensor


class MLPPredictionHead(nn.Module):
    """Linear → ReLU → ... → Linear (no ReLU after the last). HF: RTDetrMLPPredictionHead; attribute `layers`."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        # Hint: dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim];
        #       self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class DecoderLayer(nn.Module):
    """Post-norm: self-attn (+pos on Q, K) → cross deformable attn → FFN, each with residual + LayerNorm."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        n_levels: int = 3,
        n_points: int = 4,
        act: str = "relu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # Hints:
        # 1. Reuse Person A's blocks — the HF names are identical:
        #       from src.encoder.aifi import MLP, SelfAttention
        #    SelfAttention needs one addition for denoising training: an optional additive `attn_mask` [Q, Q]
        #    (0 = may attend, -inf = blocked) added to the scaled scores before softmax. Coordinate with Person A.
        # 2. self.encoder_attn = MultiscaleDeformableAttention(d_model, n_levels, num_heads, n_points)
        # 3. Three LayerNorms with the HF names in the module docstring.
        raise NotImplementedError

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
        encoder_hidden_states: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # Hints (same post-norm pattern as your encoder layer, one block longer):
        # 1. x = self_attn_layer_norm(x + self_attn(x, pos, attn_mask))
        # 2. x = encoder_attn_layer_norm(x + encoder_attn(x, encoder_hidden_states, reference_points, spatial_shapes, pos))
        # 3. x = final_layer_norm(x + mlp(x))
        raise NotImplementedError


class Decoder(nn.Module):
    """Stack of DecoderLayers; after each layer predict class logits and refine the reference boxes."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_classes: int = 80,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        n_levels: int = 3,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        # Hints: attributes `layers`, `query_pos_head`, `class_embed`, `bbox_embed` (ModuleLists, one head per layer).
        raise NotImplementedError

    def forward(
        self,
        target: Tensor,
        ref_points_unact: Tensor,
        memory: Tensor,
        spatial_shapes: list[tuple[int, int]],
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """target [B, Q, 256], ref_points_unact [B, Q, 4] (logit space), memory [B, S, 256]
        → (logits [B, num_layers, Q, C], boxes [B, num_layers, Q, 4] sigmoid cxcywh)."""
        # Hints (HF RTDetrDecoder.forward):
        # 1. ref = ref_points_unact.sigmoid()
        # 2. for i, layer in enumerate(self.layers):
        #       pos = self.query_pos_head(ref)                         ← position embedding FROM the current box
        #       x = layer(x, pos, memory, ref.unsqueeze(2), spatial_shapes, attn_mask)
        #       new_ref = (self.bbox_embed[i](x) + inverse_sigmoid(ref)).sigmoid()   ← refine in logit space
        #       boxes.append(new_ref); logits.append(self.class_embed[i](x))
        #       ref = new_ref.detach()                                 ← next layer starts from a detached box
        # 3. torch.stack(..., dim=1). The LAST layer gives the final prediction; earlier ones feed the aux losses.
        # Gotcha: append `new_ref` (with grad), but feed `ref.detach()` forward. Swapping these breaks training, not parity.
        raise NotImplementedError
