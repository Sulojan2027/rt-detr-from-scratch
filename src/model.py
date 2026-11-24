"""The whole RT-DETR, built only from our modules, and its deploy wrapper for ONNX / TensorRT.  [SHARED — Step 2]

    image [B, 3, 640, 640]
      → PResNet (Person A)          → [C3, C4, C5]
      → HybridEncoder (Person A)    → [P3, N4, N5]   all 256 channels
      → RTDetrTransformer (Person B)→ {"logits" [B, 300, 80], "pred_boxes" [B, 300, 4] cxcywh 0..1, + training extras}

Two classes:
    RTDETR        — the model. `RTDETR.from_hf(18)` = our code + the authors' pretrained weights (HF only downloads them).
    RTDETRDeploy  — what goes into ONNX / TensorRT: model + `topk_detections`, so the engine returns final
                    detections (no post-processing left in Python, like the paper's end-to-end latency).

Self-check (tests/test_model.py):
    RTDETR.from_hf(18) vs HF `RTDetrForObjectDetection` on one image: sigmoid(logits), pred_boxes max |diff| < 1e-4
    params: R18 20.17 M | R50 42.89 M
    RTDETRDeploy(model)(x, sizes) → scores [1, 300], labels [1, 300] int32, boxes [1, 300, 4] — equal to
    `topk_detections` called on the model's own output.
"""

import torch
from torch import Tensor, nn

from src.backbone.presnet import PResNet
from src.decoder.rtdetr_transformer import RTDetrTransformer
from src.encoder.hybrid_encoder import HybridEncoder
from src.postprocess import topk_detections
from src.utils.hf_weights import HF_CONFIGS, backbone_state_dict, decoder_state_dict, encoder_state_dict


class RTDETR(nn.Module):
    """Backbone → hybrid encoder → query selection + decoder."""

    def __init__(self, depth: int = 18) -> None:
        super().__init__()
        cfg = HF_CONFIGS[depth]
        self.backbone = PResNet(**cfg["backbone"])
        self.encoder = HybridEncoder(**cfg["encoder"])
        self.decoder = RTDetrTransformer(**cfg["decoder"])

    def forward(self, images: Tensor, targets: list[dict] | None = None) -> dict:
        feats = self.encoder(self.backbone(images))
        return self.decoder(feats, targets)

    @classmethod
    def from_hf(cls, depth: int = 18) -> "RTDETR":
        """Load HuggingFace weights to our model."""
        from transformers import RTDetrForObjectDetection
        hf_sd = RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").state_dict()
        model = cls(depth)
        model.backbone.load_state_dict(backbone_state_dict(hf_sd), strict=True)
        model.encoder.load_state_dict(encoder_state_dict(hf_sd), strict=True)
        model.decoder.load_state_dict(decoder_state_dict(hf_sd), strict=True)
        return model.eval()                                # BatchNorm running stats, no denoising queries


class RTDETRDeploy(nn.Module):
    """ONNX / TensorRT wrapper: (images, orig_sizes) → final detections.

    images      [B, 3, H, W]  float, 0..1 (our transforms: no mean/std)
    orig_sizes  [B, 2]        (H, W) of the ORIGINAL images, so boxes come out in original pixels
    →  scores [B, K] float, labels [B, K] int32 (0..79), boxes [B, K, 4] xyxy pixels       K = num_top (300)
    """

    def __init__(self, model: RTDETR, num_top: int = 300) -> None:
        super().__init__()
        self.num_top = num_top
        self.model = model

    def forward(self, images: Tensor, orig_sizes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.model(images)
        scores, labels, boxes = topk_detections(out["logits"], out["pred_boxes"], orig_sizes, self.num_top)
        return scores, labels.to(torch.int32), boxes


__all__ = ["RTDETR", "RTDETRDeploy"]

