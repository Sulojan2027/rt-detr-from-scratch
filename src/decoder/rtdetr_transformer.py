"""Query selection + decoder: encoder features [P3, N4, N5] → 300 detections.

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
        # centre of every cell, as a fraction of the image
        cx = (xs + 0.5) / width
        cy = (ys + 0.5) / height
        # the box gets twice as big on every coarser level: 0.05, 0.1, 0.2
        size = torch.full_like(cx, grid_size * 2.0**level)
        level_anchors = torch.stack([cx, cy, size, size], dim=-1)  # [H, W, 4]
        anchors.append(level_anchors.reshape(1, height * width, 4))
    anchors = torch.cat(anchors, dim=1)  # [1, S, 4] as (cx, cy, w, h)

    # Anchors touching the image border are not useful, so they are marked invalid.
    eps = 1e-2
    valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(dim=-1, keepdim=True)  # [1, S, 1]

    # To logit space. Invalid anchors get the largest float, so they can never win the top-k later.
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

        # Project each encoder output (already 256 channels) with 1×1 conv + BatchNorm.
        self.decoder_input_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for _ in range(num_levels)
        )

        # Query selection heads: score and box for every encoder token.
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLPPredictionHead(hidden_dim, hidden_dim, 4, num_layers=3)

        # Embedding for the (noisy) ground-truth labels used by denoising training; the last index is padding.
        self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

        self.decoder = Decoder(hidden_dim, num_decoder_layers, num_classes, num_heads, ffn_dim, num_levels, n_points)

    def forward(self, feats: list[Tensor], targets: list[dict] | None = None) -> dict:
        # 1. Project the three feature maps and flatten them into one token sequence [B, S, 256].
        sources = [proj(f) for proj, f in zip(self.decoder_input_proj, feats)]
        spatial_shapes = [tuple(s.shape[-2:]) for s in sources]
        memory = torch.cat([s.flatten(2).transpose(1, 2) for s in sources], dim=1)

        # 2. One anchor box for every token.
        anchors, valid_mask = generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)

        # 3. Query selection: score every token, keep the best `num_queries`.
        #    Tokens whose anchor is invalid are zeroed before the heads (the decoder still sees the full memory).
        output_memory = self.enc_output(valid_mask.to(memory.dtype) * memory)
        enc_outputs_class = self.enc_score_head(output_memory)  # [B, S, C]
        enc_outputs_coord = self.enc_bbox_head(output_memory) + anchors  # [B, S, 4] logit space

        topk_ind = enc_outputs_class.max(dim=-1).values.topk(self.num_queries, dim=1).indices  # [B, Q]

        def pick(x: Tensor) -> Tensor:
            return x.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        ref_unact = pick(enc_outputs_coord)  # starting boxes
        enc_topk_logits = pick(enc_outputs_class)
        target = pick(output_memory).detach()  # starting query content

        # 4. Training with ground truth: add noisy copies of the GT as extra queries in front of the normal ones.
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
            if dn_meta is not None:  # (None when the batch has no GT boxes at all)
                target = torch.cat([dn_class, target], dim=1)
                ref_unact = torch.cat([dn_boxes_unact, ref_unact], dim=1)

        # 5. Decoder. Reference boxes are detached: gradients reach them through the encoder heads' own loss.
        logits, boxes = self.decoder(target, ref_unact.detach(), memory, spatial_shapes, attn_mask)

        # 6. Split the denoising queries off again, so the normal outputs look the same as at inference.
        dn_outputs = None
        if dn_meta is not None:
            num_dn = dn_meta["dn_num_split"][0]
            dn_logits, logits = logits[:, :, :num_dn], logits[:, :, num_dn:]
            dn_boxes, boxes = boxes[:, :, :num_dn], boxes[:, :, num_dn:]
            dn_outputs = [{"logits": dn_logits[:, i], "pred_boxes": dn_boxes[:, i]} for i in range(dn_logits.shape[1])]
            ref_unact = ref_unact[:, num_dn:]  # the encoder top-k boxes, without the denoising ones

        # Earlier decoder layers and the encoder top-k predictions feed the auxiliary losses.
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
