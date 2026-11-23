"""Contrastive denoising (CDN) query group — training only.

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

import torch
from torch import Tensor, nn

from src.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, inverse_sigmoid


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
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["class_labels"]) for t in targets]
    max_gt = max(num_gts)
    if max_gt == 0:  # nothing to copy in the whole batch
        return None, None, None, None

    device = targets[0]["class_labels"].device
    batch_size = len(targets)
    # As many groups as fit into the denoising budget; a group holds one copy of every GT (padded to max_gt).
    num_groups = max(num_denoising // max_gt, 1)

    # 1. Pad every image to max_gt boxes. Padded slots get the extra "padding" class and are masked out later.
    labels = torch.full([batch_size, max_gt], num_classes, dtype=torch.int32, device=device)
    boxes = torch.zeros([batch_size, max_gt, 4], device=device)
    is_real = torch.zeros([batch_size, max_gt], dtype=torch.bool, device=device)
    for i, (target, n) in enumerate(zip(targets, num_gts)):
        if n > 0:
            labels[i, :n] = target["class_labels"]
            boxes[i, :n] = target["boxes"]
            is_real[i, :n] = True

    # 2. Every group is [positives | negatives], so repeat everything 2 * num_groups times.
    labels = labels.tile([1, 2 * num_groups])
    boxes = boxes.tile([1, 2 * num_groups, 1])
    is_real = is_real.tile([1, 2 * num_groups])

    is_negative = torch.zeros([batch_size, max_gt * 2, 1], device=device)
    is_negative[:, max_gt:] = 1
    is_negative = is_negative.tile([1, num_groups, 1])  # [B, N_dn, 1]

    # 3. Remember which queries are real positives: the loss matches them to their GT without the matcher.
    positive = (1 - is_negative).squeeze(-1) * is_real
    positive_idx = torch.nonzero(positive)[:, 1]
    positive_idx = torch.split(positive_idx, [n * num_groups for n in num_gts])

    num_dn = max_gt * 2 * num_groups

    # 4. Label noise: some real labels are swapped for a random class.
    if label_noise_ratio > 0:
        flip = torch.rand_like(labels, dtype=torch.float) < (label_noise_ratio * 0.5)
        random_labels = torch.randint_like(flip, 0, num_classes, dtype=labels.dtype)
        labels = torch.where(flip & is_real, random_labels, labels)

    # 5. Box noise, done on the corners. The shift is a random fraction of half the box size, so a small box
    #    is moved a little and a big box a lot. Positives move by 0..1 units, negatives by 1..2 units
    #    (negatives are further away, so the model must learn to say "no object" for them).
    if box_noise_scale > 0:
        corners = box_cxcywh_to_xyxy(boxes)
        unit = torch.tile(boxes[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        sign = torch.randint_like(boxes, 0, 2) * 2.0 - 1.0
        amount = torch.rand_like(boxes)
        amount = (amount + 1.0) * is_negative + amount * (1 - is_negative)
        corners = (corners + amount * sign * unit).clamp(min=0.0, max=1.0)
        # The decoder works in logit space, so the noisy boxes are converted with inverse_sigmoid.
        boxes = inverse_sigmoid(box_xyxy_to_cxcywh(corners))

    dn_class = class_embed(labels)

    # 6. Attention mask over [denoising queries | normal queries]; 0 = may attend, -inf = blocked.
    size = num_dn + num_queries
    attn_mask = torch.zeros([size, size], dtype=torch.float, device=device)
    # Normal queries must not see the denoising queries, or they could read the answer.
    attn_mask[num_dn:, :num_dn] = -torch.inf
    # Each group only sees itself, not the other groups' copies of the same GT.
    for g in range(num_groups):
        start, end = max_gt * 2 * g, max_gt * 2 * (g + 1)
        attn_mask[start:end, :start] = -torch.inf
        attn_mask[start:end, end:num_dn] = -torch.inf

    dn_meta = {
        "dn_positive_idx": positive_idx,
        "dn_num_group": num_groups,
        "dn_num_split": [num_dn, num_queries],
    }
    return dn_class, boxes, attn_mask, dn_meta
