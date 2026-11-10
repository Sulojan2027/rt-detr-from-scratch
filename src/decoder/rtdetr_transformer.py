"""Query selection + decoder: encoder features [P3, N4, N5] → 300 detections.  [PERSON B — skeleton]

Paper: RT-DETR §4.2 (uncertainty-minimal / IoU-aware query selection), §4.3 (decoder).
HF reference: `RTDetrModel.__init__` / `RTDetrModel.forward` from "sources = []" onwards, and `generate_anchors`.
Official name: `RTDETRTransformer` (rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py).

Where it plugs in:   image → PResNet → HybridEncoder → [P3, N4, N5] → RTDetrTransformer → logits, boxes
Input  feats: [B, 256, H/8, W/8], [B, 256, H/16, W/16], [B, 256, H/32, W/32]   (Person A's HybridEncoder output)

HF names — these sit directly under `model.` (siblings of `model.encoder`), so load with
`src.utils.hf_weights.decoder_state_dict` (strip `model.` only):
    decoder_input_proj.{0,1,2}.{0: Conv2d(256,256,1,bias=False), 1: BatchNorm2d(256)}
    denoising_class_embed      Embedding(81, 256, padding_idx=80)
    enc_output.{0: Linear(256,256), 1: LayerNorm(256)}
    enc_score_head             Linear(256, 80)
    enc_bbox_head.layers.{0,1,2}   MLPPredictionHead(256, 256, 4, 3)
    decoder.*                  see decoder.py
Self-check: keys R18 137 | R50 239;  params R18 4,019,376 | R50 7,466,508
            (decoder_input_proj 198,144, enc_output 66,304, enc_score_head 20,560, enc_bbox_head 132,612,
             denoising_class_embed 20,736, decoder 3,581,020 / 7,028,152)

Parity self-check (the goal, like Person A's Tasks 18 / 20), in eval mode:
    out = hf(pixel_values=x)                               # HF end to end
    ours = RTDetrTransformer(num_decoder_layers=3).eval(); ours.load_state_dict(decoder_state_dict(hf.state_dict()))
    res = ours(list(out.encoder_last_hidden_state))
    compare res["enc_outputs_class"] vs out.enc_outputs_class   [1, 8400, 80]   (query selection)
            res["logits"].sigmoid() vs out.logits.sigmoid()     [1, 300, 80]    max diff < 1e-4
            res["pred_boxes"]       vs out.pred_boxes           [1, 300, 4]     max diff < 1e-4
"""

import torch.nn as nn
from torch import Tensor


def generate_anchors(
    spatial_shapes: list[tuple[int, int]],
    grid_size: float = 0.05,
    device=None,
    dtype=None,
) -> tuple[Tensor, Tensor]:
    """One anchor box per encoder token → (anchors_unact [1, S, 4] logit space, valid_mask [1, S, 1])."""
    # Hints (HF RTDetrModel._cached_generate_anchors):
    # 1. Per level l with (H, W): meshgrid(arange(H), arange(W), indexing="ij") → centre x = (j + 0.5) / W,
    #    y = (i + 0.5) / H; box size w = h = grid_size * 2**l  (0.05 at stride 8, 0.1 at 16, 0.2 at 32).
    # 2. Concatenate levels → [1, S, 4] as (cx, cy, w, h), S = 6400 + 1600 + 400 = 8400 at 640².
    # 3. valid_mask = all 4 coords in (0.01, 0.99). Anchors at the border are invalid.
    # 4. anchors = log(a / (1 - a)); invalid → torch.finfo(dtype).max (so they never win top-k after adding offsets).
    raise NotImplementedError


class RTDetrTransformer(nn.Module):
    """[P3, N4, N5] → dict(logits, pred_boxes, + training extras)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 80,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_levels: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        n_points: int = 4,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
    ) -> None:
        super().__init__()
        # Hints:
        # 1. Build every attribute in the module docstring with EXACTLY those names (strict load decides).
        # 2. decoder_input_proj: same pattern as Person A's encoder_input_proj (Conv2d 1×1 no bias + BatchNorm2d).
        # 3. self.decoder = Decoder(hidden_dim, num_decoder_layers, num_classes, num_heads, ffn_dim, num_levels, n_points)
        # 4. Don't hard-code R18's 3 layers — R50 has 6 (hf.config.decoder_layers).
        raise NotImplementedError

    def forward(self, feats: list[Tensor], targets: list[dict] | None = None) -> dict:
        # Hints (HF RTDetrModel.forward after the encoder):
        # 1. sources = [proj(f) for proj, f in zip(self.decoder_input_proj, feats)]
        #    spatial_shapes = [(H, W) per level]; memory = cat([s.flatten(2).transpose(1, 2)], dim=1)   → [B, S, 256]
        # 2. anchors, valid_mask = generate_anchors(spatial_shapes, device=..., dtype=...)
        # 3. Query selection:
        #       output_memory = self.enc_output(valid_mask * memory)
        #       enc_outputs_class = self.enc_score_head(output_memory)                  [B, S, 80]
        #       enc_coord_unact   = self.enc_bbox_head(output_memory) + anchors         [B, S, 4]
        #       topk_ind = enc_outputs_class.max(-1).values.topk(num_queries, dim=1).indices      [B, 300]
        #    gather (dim=1) with topk_ind: ref_unact, enc_topk_logits, target = output_memory[topk].detach()
        #    "IoU-aware" part of the paper is in TRAINING: the VFL loss teaches enc_score_head to score ≈ IoU,
        #    so top-k by class score also picks well-localised boxes. Inference is plain top-k.
        # 4. Training + targets: dn_class, dn_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(...)
        #    → prepend: target = cat([dn_class, target], 1), ref_unact = cat([dn_bbox_unact, ref_unact], 1).
        # 5. logits, boxes = self.decoder(target, ref_unact.detach(), memory, spatial_shapes, attn_mask)
        #    Gotcha: the decoder attends to `memory` (UNmasked), the enc_* heads use valid_mask * memory. HF does this.
        # 6. Training: split off the denoising part with dn_meta["dn_num_split"] before building outputs.
        # 7. Return {"logits": logits[:, -1], "pred_boxes": boxes[:, -1],
        #            "aux_outputs": [{"logits", "pred_boxes"} for each earlier layer] + [enc_topk_logits / sigmoid(ref)],
        #            "dn_outputs": [...], "dn_meta": dn_meta, "enc_outputs_class": enc_outputs_class}
        raise NotImplementedError
