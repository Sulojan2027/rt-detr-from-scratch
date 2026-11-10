"""Image + target transforms.  [SHARED — skeleton]

Every transform takes and returns `(image, target)` so boxes stay in sync with the image.

What the PekingU checkpoints expect (from `AutoImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")`):
    do_resize=True,  size 640×640 (NO aspect-ratio keep, NO padding), resample=BILINEAR (PIL id 2)
    do_rescale=True, factor 1/255
    do_normalize=False  ← no ImageNet mean/std! Feeding normalised images silently drops mAP.

Target format inside the pipeline (HF convention):
    target = {"image_id": int, "class_labels": LongTensor[N] (0..79),
              "boxes": FloatTensor[N, 4] normalised (cx, cy, w, h),
              "orig_size": LongTensor[2] (h, w), "size": LongTensor[2] (h, w)}

Key insight: boxes are **normalised**, so `Resize` does NOT touch them — only flips / crops do.

Self-check once written: for one val image, `ToTensor()(Resize(640)(img, t))[0]` vs
`proc(images=img, return_tensors="pt").pixel_values[0]` → max abs diff 0.0 (both use PIL bilinear) or < 1/255.
"""

from torch import Tensor


class Compose:
    """Apply transforms in order: `image, target = t(image, target)` for each t."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image, target: dict):
        raise NotImplementedError


class Resize:
    """PIL image → PIL image of (height, width). Boxes unchanged (normalised); update target["size"]."""

    def __init__(self, size: int | tuple[int, int] = 640) -> None:
        # Hint: store (h, w); an int means square.
        raise NotImplementedError

    def __call__(self, image, target: dict):
        # Hints:
        # 1. PIL wants (w, h): image.resize((w, h), Image.BILINEAR)  ← same op as HF, so parity is exact.
        # 2. target["size"] = torch.tensor([h, w]). Don't touch target["boxes"].
        raise NotImplementedError


class RandomHorizontalFlip:
    """Training only. With prob p: flip the image and set box cx → 1 - cx (w, h, cy unchanged)."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, target: dict):
        # Hint: image.transpose(Image.FLIP_LEFT_RIGHT); boxes[:, 0] = 1 - boxes[:, 0] (clone first, don't edit in place).
        raise NotImplementedError


class ToTensor:
    """PIL uint8 [H, W, 3] → float [3, H, W] in [0, 1] (rescale only, no mean/std)."""

    def __call__(self, image, target: dict) -> tuple[Tensor, dict]:
        # Hint: torchvision.transforms.functional.pil_to_tensor(image).float() / 255.0  (or F.to_tensor).
        raise NotImplementedError


def build_transforms(train: bool, size: int = 640) -> Compose:
    """train: flip → resize → to-tensor. eval: resize → to-tensor."""
    # Hint: RT-DETR training also uses photometric distort, zoom-out, IoU crop. Skip them unless you train.
    raise NotImplementedError
