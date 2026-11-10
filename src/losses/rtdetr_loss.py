"""RT-DETR set-prediction loss: VFL + L1 + GIoU, on final, auxiliary and denoising outputs.  [PERSON B — skeleton]

Paper: RT-DETR §4.2 (VFL makes the class score IoU-aware → "uncertainty-minimal query selection").
HF: `RTDetrLoss` + `RTDetrForObjectDetectionLoss` in `transformers/loss/loss_rt_detr.py`. Training only.

Weights (hf.config): weight_loss_vfl = 1, weight_loss_bbox = 5, weight_loss_giou = 2;
VFL focal_loss_alpha = 0.75, focal_loss_gamma = 2.0.  (Matcher uses its OWN alpha 0.25 — different number.)

Which outputs get losses (HF with auxiliary_loss=True):
    final layer                         → matched with the Hungarian matcher
    each earlier decoder layer (aux)    → re-matched per layer          keys "..._aux_{i}"
    encoder top-k proposals (aux)       → re-matched                     (appended as the last aux output)
    denoising groups (dn)               → FIXED matching from dn_meta    keys "..._dn_{i}", num_boxes × dn_num_group

Self-check: build HF's criterion `RTDetrLoss(hf.config)` and feed it the SAME outputs dict + targets → identical
loss dict (same keys, values to ~1e-6). Easiest with fake random outputs, no model needed.
"""

import torch.nn as nn
from torch import Tensor


class RTDetrLoss(nn.Module):
    def __init__(
        self,
        matcher: nn.Module,
        num_classes: int = 80,
        weight_dict: dict[str, float] | None = None,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        # Hint: default weight_dict = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        raise NotImplementedError

    @staticmethod
    def _src_idx(indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Matched (batch_idx, query_idx) for advanced indexing: outputs["pred_boxes"][idx] → [K_total, 4]."""
        # Hint: batch_idx = cat([full_like(src, i) for i, (src, _) in enumerate(indices)]); query_idx = cat([src ...])
        raise NotImplementedError

    def loss_vfl(self, outputs: dict, targets: list[dict], indices: list, num_boxes: float) -> dict[str, Tensor]:
        """Varifocal loss: BCE where the target for the matched class is the IoU (not 1)."""
        # Hints (HF loss_labels_vfl):
        # 1. idx = _src_idx(indices); src_boxes = pred_boxes[idx]; tgt_boxes = matched GT boxes (same order)
        # 2. ious = diag(box_iou(xyxy(src_boxes.detach()), xyxy(tgt_boxes))[0])          ← detach: IoU is a target
        # 3. target_classes [B, Q] filled with num_classes; target_classes[idx] = matched GT labels
        #    one_hot(target_classes, num_classes + 1)[..., :-1]  → [B, Q, C] (drop the "no object" column)
        # 4. target_score = one_hot * iou (per matched query)
        # 5. weight = alpha * sigmoid(logits).detach()**gamma * (1 - one_hot) + target_score
        # 6. loss = BCE_with_logits(logits, target_score, weight=weight, reduction="none").mean(1).sum() * Q / num_boxes
        raise NotImplementedError

    def loss_boxes(self, outputs: dict, targets: list[dict], indices: list, num_boxes: float) -> dict[str, Tensor]:
        """L1 on cxcywh + (1 - GIoU) on xyxy, both summed over matched pairs / num_boxes."""
        raise NotImplementedError

    @staticmethod
    def get_cdn_matched_indices(dn_meta: dict, targets: list[dict]) -> list[tuple[Tensor, Tensor]]:
        """Denoising queries have a fixed assignment: dn_positive_idx[i] ↔ arange(num_gt).tile(dn_num_group)."""
        raise NotImplementedError

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, Tensor]:
        # Hints (HF RTDetrLoss.forward):
        # 1. num_boxes = max(total GTs in batch, 1)   (float; with multi-GPU you'd all-reduce it)
        # 2. indices = matcher(final outputs); losses = loss_vfl + loss_boxes, each × weight_dict[key]
        # 3. for i, aux in enumerate(outputs["aux_outputs"]): re-match, same losses, keys + f"_aux_{i}"
        # 4. if "dn_outputs" in outputs: indices = get_cdn_matched_indices(...); num_boxes *= dn_num_group;
        #    keys + f"_dn_{i}"
        # 5. Total training loss = sum(losses.values()).
        raise NotImplementedError
