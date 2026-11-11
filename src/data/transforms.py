"""Eval-time image + target transforms (no training augmentations — we use pretrained weights).  [SHARED]

Every transform takes and returns `(image, target)` so boxes stay in sync with the image.

What the PekingU checkpoints expect (`AutoImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")`):
    resize to 640×640 (NO aspect-ratio keep, NO padding), bilinear + antialias
    rescale ×1/255
    NO mean/std normalisation  ← feeding ImageNet-normalised images silently drops mAP

EXACT recipe (verified: diff 0.0 vs the HF processor on 4 image sizes). HF uses **torchvision v2** on **uint8**:
    import torchvision.transforms.v2.functional as F2
    x = F2.pil_to_tensor(img)                                          # uint8 [3, H, W]
    x = F2.resize(x, [640, 640], interpolation=InterpolationMode.BILINEAR, antialias=True)   # still uint8
    x = F2.to_dtype(x, torch.float32, scale=True)                      # float [0, 1]
Order matters: resize BEFORE converting to float. PIL `Image.resize` or v1 `TF.resize` give ±1/255 on ~0.14 % of pixels.

Target format inside the pipeline (HF convention):
    target = {"image_id": int, "class_labels": LongTensor[N] (0..79),
              "boxes": FloatTensor[N, 4] normalised (cx, cy, w, h),
              "orig_size": LongTensor[2] (h, w), "size": LongTensor[2] (h, w)}
Key insight: boxes are **normalised**, so `Resize` does NOT touch them — only update target["size"].

Self-check once written:
    x, t = build_transforms()(img, target)
    x.shape == (3, 640, 640), x.dtype == float32, 0 <= x <= 1
    (x - proc(images=img, return_tensors="pt").pixel_values[0]).abs().max() == 0.0
    t["boxes"] unchanged, t["size"] == [640, 640], t["orig_size"] unchanged
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
