"""Contrastive denoising (CDN) query group — training only.  [PERSON B — skeleton]

Paper: RT-DETR §4.3 / Appendix (from DINO, arXiv:2203.03605 §3.3). HF: `get_contrastive_denoising_training_group`.

Idea: besides the 300 normal queries, feed the decoder noisy copies of the ground-truth boxes. "Positive" copies
(small noise) must reconstruct the GT box; "negative" copies (bigger noise) must predict "no object". They get a
fixed GT assignment (no Hungarian matching), which stabilises and speeds up training. Inference never uses this.

Config (hf.config): num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0.
HF name: `model.denoising_class_embed` = nn.Embedding(81, 256, padding_idx=80)  (80 classes + 1 padding).

Self-check (shapes are deterministic even though values are random):
    targets with 3 and 1 GT boxes, num_denoising=100, num_queries=300:
        max_gt = 3, num_groups = 100 // 3 = 33, num_dn = 3 * 2 * 33 = 198
        dn_class [2, 198, 256], dn_bbox_unact [2, 198, 4], attn_mask [498, 498]
        dn_meta = {"dn_positive_idx": (99 idx, 33 idx), "dn_num_group": 33, "dn_num_split": [198, 300]}
    Value parity with HF: call torch.manual_seed(k) before both and use the same random ops in the same order.
"""

import torch.nn as nn
from torch import Tensor


def get_contrastive_denoising_training_group(
    targets: list[dict[str, Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: nn.Embedding,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, dict | None]:
    """→ (dn_class [B, N_dn, 256], dn_bbox_unact [B, N_dn, 4], attn_mask [N_dn+Q, N_dn+Q], dn_meta)."""
    # Hints (follow HF; return (None,)*4 if num_denoising <= 0 or no image has a GT box):
    # 1. max_gt = max GTs per image; num_groups = max(num_denoising // max_gt, 1).
    # 2. Pad every image's labels to max_gt with `num_classes` (the padding index) and boxes with 0; keep pad_gt_mask.
    # 3. Tile ×(2 * num_groups): each group = [positives (max_gt) | negatives (max_gt)].
    # 4. Label noise: with prob label_noise_ratio * 0.5 replace a (real, not padded) label by a random class.
    # 5. Box noise (in xyxy): diff = (w/2, h/2, w/2, h/2) * box_noise_scale;
    #       positives: |noise| in [0, 1) * diff      negatives: |noise| in [1, 2) * diff
    #    random sign, clip to [0, 1], back to cxcywh, then inverse_sigmoid (decoder works in logit space).
    # 6. dn_class = class_embed(labels).
    # 7. attn_mask [N_dn + Q, N_dn + Q], 0 = allowed, -inf = blocked:
    #       normal queries must NOT see any denoising query (they would peek at the GT);
    #       each denoising group must NOT see the other groups.
    # 8. dn_positive_idx: indices of positive, non-padded queries per image (used by the loss as the fixed matching).
    raise NotImplementedError
