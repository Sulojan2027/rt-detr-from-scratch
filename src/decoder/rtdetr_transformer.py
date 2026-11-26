"""Query selection + decoder: encoder features [P3, N4, N5] → 300 detections.

RT-DETR §4.2 (IoU-aware query selection) and §4.3 (decoder). Mirrors HF `RTDetrModel.forward` from the decoder
input projection onwards (and `generate_anchors`); the official repo calls this `RTDETRTransformer`.
Input feats: [B, 256, H/8, W/8], [B, 256, H/16, W/16], [B, 256, H/32, W/32] from the HybridEncoder.

HF parameter names sit directly under `model.` (siblings of `model.encoder`); load with
`src.utils.hf_weights.decoder_state_dict`:
    decoder_input_proj.{0,1,2}.{0: Conv2d(256,256,1,bias=False), 1: BatchNorm2d(256)}
    denoising_class_embed      Embedding(81, 256, padding_idx=80)
    enc_output.{0: Linear(256,256), 1: LayerNorm(256)}
    enc_score_head             Linear(256, 80)
    enc_bbox_head.layers.{0,1,2}   MLPPredictionHead(256, 256, 4, 3)
    decoder.*                  see decoder.py
"""

import torch
from torch import Tensor, nn

from src.decoder.decoder import Decoder, MLPPredictionHead
from src.decoder.denoising import get_contrastive_denoising_training_group


def generate_anchors(
    spatial_shapes: list[tuple[int, int]],
    grid_size: float = 0.05,
    device=None,
    dtype=None,
) -> tuple[Tensor, Tensor]:
    """One anchor box per encoder token → (anchors_unact [1, S, 4] logit space, valid_mask [1, S, 1])."""
    dtype = dtype or torch.float32

    anchors = []
    for level, (height, width) in enumerate(spatial_shapes):
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        # cell centres, normalised to [0, 1]
        cx = (xs + 0.5) / width
        cy = (ys + 0.5) / height
        # anchor size doubles per level: 0.05, 0.1, 0.2
        size = torch.full_like(cx, grid_size * 2.0**level)
        level_anchors = torch.stack([cx, cy, size, size], dim=-1)  # [H, W, 4]
        anchors.append(level_anchors.reshape(1, height * width, 4))
    anchors = torch.cat(anchors, dim=1)  # [1, S, 4] as (cx, cy, w, h)

    # Anchors near the image border are marked invalid.
    eps = 1e-2
    valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(dim=-1, keepdim=True)  # [1, S, 1]

    # To logit space; invalid anchors are set to the largest float (as in HF).
    anchors = torch.log(anchors / (1 - anchors))
    anchors = torch.where(valid_mask, anchors, torch.full_like(anchors, torch.finfo(dtype).max))
    return anchors, valid_mask


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
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.decoder_input_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for _ in range(num_levels)
        )

        # Query selection heads: class score and box for every encoder token.
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLPPredictionHead(hidden_dim, hidden_dim, 4, num_layers=3)

        # Label embedding for denoising queries; the last index is padding.
        self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

        self.decoder = Decoder(hidden_dim, num_decoder_layers, num_classes, num_heads, ffn_dim, num_levels, n_points)

    def forward(self, feats: list[Tensor], targets: list[dict] | None = None) -> dict:
        # Project and flatten the three levels into one token sequence [B, S, 256].
        sources = [proj(f) for proj, f in zip(self.decoder_input_proj, feats)]
        spatial_shapes = [tuple(s.shape[-2:]) for s in sources]
        memory = torch.cat([s.flatten(2).transpose(1, 2) for s in sources], dim=1)

        anchors, valid_mask = generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)

        # Query selection: keep the top `num_queries` tokens by max class score.
        # Tokens with invalid anchors are zeroed before the heads (the decoder still sees the full memory).
        output_memory = self.enc_output(valid_mask.to(memory.dtype) * memory)
        enc_outputs_class = self.enc_score_head(output_memory)  # [B, S, C]
        enc_outputs_coord = self.enc_bbox_head(output_memory) + anchors  # [B, S, 4] logit space

        topk_ind = enc_outputs_class.max(dim=-1).values.topk(self.num_queries, dim=1).indices  # [B, Q]

        def pick(x: Tensor) -> Tensor:
            return x.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        ref_unact = pick(enc_outputs_coord)  # initial reference boxes
        enc_topk_logits = pick(enc_outputs_class)
        target = pick(output_memory).detach()  # initial query content

        # Training: prepend the denoising queries.
        attn_mask, dn_meta = None, None
        if self.training and targets is not None:
            dn_class, dn_boxes_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                self.num_denoising,
                self.label_noise_ratio,
                self.box_noise_scale,
            )
            if dn_meta is not None:  # None when the batch has no GT boxes
                target = torch.cat([dn_class, target], dim=1)
                ref_unact = torch.cat([dn_boxes_unact, ref_unact], dim=1)

        # Reference boxes are detached; the encoder heads get gradients from their own (aux) loss.
        logits, boxes = self.decoder(target, ref_unact.detach(), memory, spatial_shapes, attn_mask)

        # Split the denoising queries off again.
        dn_outputs = None
        if dn_meta is not None:
            num_dn = dn_meta["dn_num_split"][0]
            dn_logits, logits = logits[:, :, :num_dn], logits[:, :, num_dn:]
            dn_boxes, boxes = boxes[:, :, :num_dn], boxes[:, :, num_dn:]
            dn_outputs = [{"logits": dn_logits[:, i], "pred_boxes": dn_boxes[:, i]} for i in range(dn_logits.shape[1])]
            ref_unact = ref_unact[:, num_dn:]

        # Auxiliary outputs: earlier decoder layers, then the encoder top-k predictions.
        aux_outputs = [{"logits": logits[:, i], "pred_boxes": boxes[:, i]} for i in range(logits.shape[1] - 1)]
        aux_outputs.append({"logits": enc_topk_logits, "pred_boxes": ref_unact.sigmoid()})

        out = {
            "logits": logits[:, -1],
            "pred_boxes": boxes[:, -1],
            "aux_outputs": aux_outputs,
            "enc_outputs_class": enc_outputs_class,
        }
        if dn_meta is not None:
            out["dn_outputs"] = dn_outputs
            out["dn_meta"] = dn_meta
        return out
