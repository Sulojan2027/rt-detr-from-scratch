"""Contrastive denoising (CDN) query group, used in training only.

RT-DETR §4.3 / Appendix, from DINO (arXiv:2203.03605 §3.3); mirrors HF `get_contrastive_denoising_training_group`.
Noisy copies of the GT boxes are added as extra decoder queries: positives (small noise) must reconstruct the GT
box, negatives (larger noise) must predict "no object". Their GT assignment is fixed (no Hungarian matching).
Defaults match the HF config: num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0.
Random ops are called in the same order as HF, so results match HF under the same torch seed.
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
    if max_gt == 0:
        return None, None, None, None

    device = targets[0]["class_labels"].device
    batch_size = len(targets)
    # Groups that fit in the budget; each group holds a positive and a negative copy of every GT (padded to max_gt).
    num_groups = max(num_denoising // max_gt, 1)

    # Pad every image to max_gt boxes; padded slots get the padding class (index num_classes).
    labels = torch.full([batch_size, max_gt], num_classes, dtype=torch.int32, device=device)
    boxes = torch.zeros([batch_size, max_gt, 4], device=device)
    is_real = torch.zeros([batch_size, max_gt], dtype=torch.bool, device=device)
    for i, (target, n) in enumerate(zip(targets, num_gts)):
        if n > 0:
            labels[i, :n] = target["class_labels"]
            boxes[i, :n] = target["boxes"]
            is_real[i, :n] = True

    # Each group is laid out as [positives | negatives].
    labels = labels.tile([1, 2 * num_groups])
    boxes = boxes.tile([1, 2 * num_groups, 1])
    is_real = is_real.tile([1, 2 * num_groups])

    is_negative = torch.zeros([batch_size, max_gt * 2, 1], device=device)
    is_negative[:, max_gt:] = 1
    is_negative = is_negative.tile([1, num_groups, 1])  # [B, N_dn, 1]

    # Indices of the real positives: the loss assigns them to their GT directly, without the matcher.
    positive = (1 - is_negative).squeeze(-1) * is_real
    positive_idx = torch.nonzero(positive)[:, 1]
    positive_idx = torch.split(positive_idx, [n * num_groups for n in num_gts])

    num_dn = max_gt * 2 * num_groups

    # Label noise: flip a fraction of the real labels to a random class.
    if label_noise_ratio > 0:
        flip = torch.rand_like(labels, dtype=torch.float) < (label_noise_ratio * 0.5)
        random_labels = torch.randint_like(flip, 0, num_classes, dtype=labels.dtype)
        labels = torch.where(flip & is_real, random_labels, labels)

    # Box noise on the xyxy corners, in units of half the box size: positives move by [0, 1) units,
    # negatives by [1, 2) units.
    if box_noise_scale > 0:
        corners = box_cxcywh_to_xyxy(boxes)
        unit = torch.tile(boxes[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        sign = torch.randint_like(boxes, 0, 2) * 2.0 - 1.0
        amount = torch.rand_like(boxes)
        amount = (amount + 1.0) * is_negative + amount * (1 - is_negative)
        corners = (corners + amount * sign * unit).clamp(min=0.0, max=1.0)
        # Back to cxcywh, in logit space for the decoder.
        boxes = inverse_sigmoid(box_xyxy_to_cxcywh(corners))

    dn_class = class_embed(labels)

    # Attention mask over [denoising queries | normal queries]; 0 = attend, -inf = blocked.
    size = num_dn + num_queries
    attn_mask = torch.zeros([size, size], dtype=torch.float, device=device)
    # Normal queries must not see the denoising queries (they would leak the GT).
    attn_mask[num_dn:, :num_dn] = -torch.inf
    # Each denoising group sees only itself.
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
