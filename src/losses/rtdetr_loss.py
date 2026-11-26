"""RT-DETR set-prediction loss: VFL + L1 + GIoU on final, auxiliary and denoising outputs.

RT-DETR §4.2 (VFL makes the class score IoU-aware). Mirrors HF `RTDetrLoss` (`transformers/loss/loss_rt_detr.py`).
Weights: vfl 1, bbox 5, giou 2; VFL alpha = 0.75, gamma = 2.0 (the matcher uses its own alpha = 0.25).

Which outputs get losses:
    final layer                         → Hungarian matching
    each earlier decoder layer (aux)    → re-matched per layer          keys "..._aux_{i}"
    encoder top-k proposals (aux)       → re-matched                     (appended as the last aux output)
    denoising groups (dn)               → fixed matching from dn_meta    keys "..._dn_{i}", num_boxes × dn_num_group
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


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
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict or {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _src_idx(indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Matched (batch_idx, query_idx) for advanced indexing: outputs["pred_boxes"][idx] → [K_total, 4]."""
        batch_idx = torch.cat([torch.full_like(query_idx, i) for i, (query_idx, _) in enumerate(indices)])
        query_idx = torch.cat([query_idx for query_idx, _ in indices])
        return batch_idx, query_idx

    @staticmethod
    def _matched_targets(targets: list[dict], indices: list[tuple[Tensor, Tensor]], key: str) -> Tensor:
        """The ground-truth `key` values of all matched pairs, in the same order as `_src_idx`."""
        return torch.cat([t[key][gt_idx] for t, (_, gt_idx) in zip(targets, indices)])

    def loss_vfl(self, outputs: dict, targets: list[dict], indices: list, num_boxes: float) -> dict[str, Tensor]:
        """Varifocal loss: BCE where the target for the matched class is the IoU (not 1)."""
        logits = outputs["logits"]  # [B, Q, C]
        idx = self._src_idx(indices)

        # IoU of each matched pair; used as a target, so computed without gradient.
        pred_boxes = outputs["pred_boxes"][idx]
        gt_boxes = self._matched_targets(targets, indices, "boxes")
        ious = box_iou(box_cxcywh_to_xyxy(pred_boxes.detach()), box_cxcywh_to_xyxy(gt_boxes))[0].diag()

        # One-hot class targets [B, Q, C]; unmatched queries get the extra "no object" class, which is then
        # dropped, leaving all zeros.
        target_classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device)
        target_classes[idx] = self._matched_targets(targets, indices, "class_labels")
        one_hot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        # Target for the matched class is the IoU instead of 1.
        iou_per_query = torch.zeros(logits.shape[:2], dtype=logits.dtype, device=logits.device)
        iou_per_query[idx] = ious.to(logits.dtype)
        target_score = iou_per_query.unsqueeze(-1) * one_hot

        # Negatives weighted focal-style by the predicted score; positives by their IoU.
        pred_score = logits.detach().sigmoid()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - one_hot) + target_score

        loss = F.binary_cross_entropy_with_logits(logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs: dict, targets: list[dict], indices: list, num_boxes: float) -> dict[str, Tensor]:
        """L1 on cxcywh + (1 - GIoU) on xyxy, both summed over matched pairs / num_boxes."""
        pred_boxes = outputs["pred_boxes"][self._src_idx(indices)]
        gt_boxes = self._matched_targets(targets, indices, "boxes")

        loss_bbox = F.l1_loss(pred_boxes, gt_boxes, reduction="none").sum() / num_boxes

        giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes)).diag()
        loss_giou = (1 - giou).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    @staticmethod
    def get_cdn_matched_indices(dn_meta: dict, targets: list[dict]) -> list[tuple[Tensor, Tensor]]:
        """Denoising queries have a fixed assignment: dn_positive_idx[i] ↔ arange(num_gt).tile(dn_num_group)."""
        device = targets[0]["class_labels"].device
        matches = []
        for positive_idx, target in zip(dn_meta["dn_positive_idx"], targets):
            num_gt = len(target["class_labels"])
            if num_gt == 0:
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                matches.append((empty, empty))
                continue
            # Every denoising group holds a copy of every GT, in the same order.
            gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(dn_meta["dn_num_group"])
            if len(positive_idx) != len(gt_idx):
                raise ValueError("dn_positive_idx and the number of ground-truth boxes do not agree.")
            matches.append((positive_idx, gt_idx))
        return matches

    def _weighted_losses(self, outputs: dict, targets: list[dict], indices: list, num_boxes: float, suffix: str = ""):
        """VFL + box losses for one set of predictions, each multiplied by its weight."""
        losses = {**self.loss_vfl(outputs, targets, indices, num_boxes), **self.loss_boxes(outputs, targets, indices, num_boxes)}
        return {f"{name}{suffix}": value * self.weight_dict[name] for name, value in losses.items()}

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, Tensor]:
        """outputs: {"logits", "pred_boxes", optional "aux_outputs", "dn_outputs", "dn_meta"}; targets as in the matcher.
        → dict of weighted losses; the training loss is `sum(losses.values())`."""
        # Number of GT boxes in the batch (at least 1), used to normalise every loss.
        num_boxes = max(sum(len(t["class_labels"]) for t in targets), 1)

        indices = self.matcher(outputs, targets)
        losses = self._weighted_losses(outputs, targets, indices, num_boxes)

        # Earlier decoder layers and the encoder proposals are re-matched on their own predictions.
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            losses.update(self._weighted_losses(aux, targets, self.matcher(aux, targets), num_boxes, f"_aux_{i}"))

        # Denoising queries use the fixed assignment, not the matcher.
        if "dn_outputs" in outputs:
            dn_meta = outputs["dn_meta"]
            dn_indices = self.get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = num_boxes * dn_meta["dn_num_group"]
            for i, dn in enumerate(outputs["dn_outputs"]):
                losses.update(self._weighted_losses(dn, targets, dn_indices, dn_num_boxes, f"_dn_{i}"))

        return losses
