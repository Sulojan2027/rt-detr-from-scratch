"""COCO 2017 detection dataset.  [SHARED — skeleton]

Needs `pycocotools` (in requirements.txt; install it in the env: `pip install pycocotools`).

Layout (README "Data"):
    data/coco/annotations/instances_val2017.json
    data/coco/val2017/*.jpg

Category ids — the classic COCO trap:
    COCO json uses 80 category ids in 1..90 **with gaps** (no 12, 26, 29, ...).
    The HF / official checkpoints predict **contiguous** labels 0..79 (hf.config.id2label[0] == "person").
    So: label = index of cat_id in sorted(coco.getCatIds()); for eval map back with the same list.

Self-checks once written (val2017):
    len(ds) == 5000
    ds.cat_ids[:3] == [1, 2, 3] and len(ds.cat_ids) == 80 and ds.cat_ids[-1] == 90
    img, t = ds[0] → img [3, 640, 640] in [0, 1]; t["boxes"] in [0, 1]; t["orig_size"] = original (h, w)
    Boxes round-trip: cxcywh * (w, h, w, h) → xywh equals the json "bbox" values.
"""

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset


class CocoDetection(Dataset):
    """Returns (image [3, H, W], target dict) in the format described in `transforms.py`."""

    def __init__(self, img_folder: str | Path, ann_file: str | Path, transforms=None) -> None:
        # Hints:
        # 1. from pycocotools.coco import COCO; self.coco = COCO(str(ann_file))
        # 2. self.ids = sorted(self.coco.imgs.keys())
        # 3. self.cat_ids = sorted(self.coco.getCatIds()); self.cat_to_label = {c: i for i, c in enumerate(self.cat_ids)}
        # (Alternative: subclass torchvision.datasets.CocoDetection, which does the loading for you.)
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        # Hints:
        # 1. image_id = self.ids[idx]; info = self.coco.loadImgs(image_id)[0]
        #    image = PIL.Image.open(img_folder / info["file_name"]).convert("RGB")   ← some COCO images are grayscale
        # 2. anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)); drop ann["iscrowd"] == 1
        # 3. COCO bbox is [x, y, w, h] in PIXELS → normalised cxcywh: cx = (x + w/2) / W, cy = (y + h/2) / H, w / W, h / H
        #    Drop boxes with w <= 0 or h <= 0; clip to the image first.
        # 4. class_labels = [self.cat_to_label[a["category_id"]] for a in anns]
        # 5. Empty images exist: boxes must still be shape [0, 4], labels [0] (torch.zeros((0, 4))).
        # 6. target = {"image_id", "class_labels", "boxes", "orig_size": (H, W), "size": (H, W)}; then apply self.transforms.
        raise NotImplementedError


def collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, list[dict]]:
    """Stack images (all 640×640 after Resize) into [B, 3, H, W]; keep targets as a list (each has a different N)."""
    raise NotImplementedError
