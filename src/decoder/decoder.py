"""Transformer decoder with iterative box refinement.

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

from itertools import pairwise

import torch
from torch import Tensor, nn

from src.decoder.deformable_attention import MultiscaleDeformableAttention
from src.encoder.aifi import MLP, SelfAttention
from src.utils.box_ops import inverse_sigmoid


class MLPPredictionHead(nn.Module):
    """Linear → ReLU → ... → Linear (no ReLU after the last). HF: RTDetrMLPPredictionHead; attribute `layers`."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        # e.g. (256, 256, 4, 3) → Linear(256, 256), Linear(256, 256), Linear(256, 4)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(n_in, n_out) for n_in, n_out in pairwise(dims))

    def forward(self, x: Tensor) -> Tensor:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = torch.relu(x)
        return x


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
        # 1) queries talk to each other
        self.self_attn = SelfAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # 2) queries look at the image features
        self.encoder_attn = MultiscaleDeformableAttention(d_model, n_levels, num_heads, n_points)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # 3) feed-forward
        self.mlp = MLP(d_model, intermediate_size=ffn_dim, act=act)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
        encoder_hidden_states: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = hidden_states

        x = self.self_attn_layer_norm(x + self.self_attn(x, position_embeddings, attn_mask))

        cross = self.encoder_attn(x, encoder_hidden_states, reference_points, spatial_shapes, position_embeddings)
        x = self.encoder_attn_layer_norm(x + cross)

        x = self.final_layer_norm(x + self.mlp(x))
        return x


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
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, num_heads, ffn_dim, n_levels, n_points) for _ in range(num_layers)
        )
        # Turns the current reference box (4 numbers) into a position embedding for the queries.
        self.query_pos_head = MLPPredictionHead(4, 2 * d_model, d_model, num_layers=2)
        # Every layer has its own class head and box-refinement head (so every layer can be supervised).
        self.class_embed = nn.ModuleList(nn.Linear(d_model, num_classes) for _ in range(num_layers))
        self.bbox_embed = nn.ModuleList(MLPPredictionHead(d_model, d_model, 4, num_layers=3) for _ in range(num_layers))

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
        x = target
        ref = ref_points_unact.sigmoid()

        all_logits, all_boxes = [], []
        for i, layer in enumerate(self.layers):
            pos = self.query_pos_head(ref)
            x = layer(x, pos, memory, ref.unsqueeze(2), spatial_shapes, attn_mask)

            # Each layer nudges the box a little: add its prediction to the old box in logit space.
            new_ref = (self.bbox_embed[i](x) + inverse_sigmoid(ref)).sigmoid()

            all_boxes.append(new_ref)  # keep the gradient for the loss ...
            all_logits.append(self.class_embed[i](x))
            ref = new_ref.detach()  # ... but the next layer starts from a detached box

        return torch.stack(all_logits, dim=1), torch.stack(all_boxes, dim=1)
