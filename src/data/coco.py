"""COCO 2017 detection dataset (images + targets in the HF format, see `transforms.py`).

COCO category ids are 1..90 with gaps; the checkpoints predict contiguous labels 0..79, so
label = index of the category id in `ds.cat_ids` (sorted), and `ds.cat_ids[label]` maps back for evaluation.
"""

from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data import Dataset


class CocoDetection(Dataset):
    """Returns (image [3, H, W], target dict); boxes are normalised cxcywh, crowd annotations skipped."""

    def __init__(self, img_folder: str | Path, ann_file: str | Path, transforms=None) -> None:
        self.coco = COCO(str(ann_file))
        self.ids = sorted(self.coco.imgs.keys())
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_to_label = {c: i for i, c in enumerate(self.cat_ids)}
        self.img_folder = Path(img_folder)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        image_id = self.ids[idx]
        info = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.img_folder / info["file_name"]).convert("RGB")
        W, H = image.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, iscrowd=False))
        boxes, labels = [], []
        for ann in anns:                                  # read-only: shared with COCOeval's ground truth
            x, y, w, h = ann["bbox"]
            x1, y1 = max(x, 0.0), max(y, 0.0)             # clip to the image
            x2, y2 = min(x + w, W), min(y + h, H)
            if x2 <= x1 or y2 <= y1:                      # degenerate box → skip
                continue
            boxes.append([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
            labels.append(self.cat_to_label[ann["category_id"]])

        target = {
            "image_id": image_id,
            "class_labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "orig_size": torch.tensor([H, W]),
            "size": torch.tensor([H, W]),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

def collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, list[dict]]:
    """Stack images (all 640×640 after Resize) into [B, 3, H, W]; keep targets as a list (each has a different N)."""
    images, targets = zip(*batch, strict=True)
    return torch.stack(images, dim=0), list(targets)
