"""Eval-time transforms matching the HF RT-DETR image processor exactly (no training augmentations).

Recipe: uint8 → resize to 640×640 (bilinear, antialias, no aspect-ratio keep) → float in [0, 1], NO mean/std.
Resizing the uint8 tensor before converting to float (as HF does with torchvision v2) is what gives an exact
match; PIL / v1 resize differ by ±1/255 on some pixels.

Each transform maps (image, target) → (image, target). Target (HF convention):
    {"image_id": int, "class_labels": [N] int64 (0..79), "boxes": [N, 4] normalised cxcywh,
     "orig_size": [2] (h, w), "size": [2] (h, w)}
Boxes are normalised, so Resize only updates target["size"].
"""

import torch
import torchvision.transforms.v2.functional as F2
from torch import Tensor
from torchvision.transforms import InterpolationMode

class Compose:
    """Apply transforms in order: `image, target = t(image, target)` for each t."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image, target: dict):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

class PILToTensor:
    """PIL RGB image → uint8 tensor [3, H, W] (no scaling yet)."""

    def __call__(self, image, target: dict) -> tuple[Tensor, dict]:
        return F2.pil_to_tensor(image), target

class Resize:
    """uint8 [3, H, W] → uint8 [3, h, w]. Boxes unchanged (normalised); update target["size"]."""

    def __init__(self, size: int | tuple[int, int] = 640) -> None:
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, image: Tensor, target: dict) -> tuple[Tensor, dict]:
        resized_image = F2.resize(image, [self.h, self.w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        return resized_image, {**target, "size": torch.tensor([self.h, self.w])}

class ToDtype:
    """uint8 [0, 255] → float32 [0, 1]. No mean/std."""

    def __call__(self, image: Tensor, target: dict) -> tuple[Tensor, dict]:
        return F2.to_dtype(image, torch.float32, scale=True), target

def build_transforms(size: int = 640) -> Compose:
    """PILToTensor → Resize(size) → ToDtype."""
    return Compose([PILToTensor(), Resize(size), ToDtype()])
