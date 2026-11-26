"""Full RT-DETR model built from this repo's modules, plus a deploy wrapper for ONNX / TensorRT.

    image [B, 3, 640, 640] → PResNet → [C3, C4, C5] → HybridEncoder → [P3, N4, N5] (256 ch)
      → RTDetrTransformer → {"logits" [B, 300, 80], "pred_boxes" [B, 300, 4] normalised cxcywh, ...}
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
        """Build the model and load the pretrained PekingU/rtdetr_r{depth}vd weights from Hugging Face."""
        from transformers import RTDetrForObjectDetection
        hf_sd = RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").state_dict()
        model = cls(depth)
        model.backbone.load_state_dict(backbone_state_dict(hf_sd), strict=True)
        model.encoder.load_state_dict(encoder_state_dict(hf_sd), strict=True)
        model.decoder.load_state_dict(decoder_state_dict(hf_sd), strict=True)
        return model.eval()                                # BatchNorm running stats, no denoising queries


class RTDETRDeploy(nn.Module):
    """Model + top-k post-processing in one graph, so an exported engine returns final detections.

    images      [B, 3, H, W]  float in [0, 1] (no mean/std normalisation)
    orig_sizes  [B, 2]        (H, W) of the original images; boxes are returned in original pixels
    →  scores [B, K] float, labels [B, K] int32 (0..79), boxes [B, K, 4] xyxy pixels; K = num_top
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

