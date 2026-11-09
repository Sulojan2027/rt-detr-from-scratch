"""Map Hugging Face `RTDetrForObjectDetection` state-dict keys onto our standalone modules."""

from torch import Tensor

BACKBONE_PREFIX = "model.backbone.model."
ENCODER_PREFIXES = ("model.encoder_input_proj.", "model.encoder.")
HF_CONFIGS: dict[int, dict[str, dict]] = {
    18: {"backbone": dict(depth=18), "encoder": dict(in_channels=(128, 256, 512), expansion=0.5)},
    50: {"backbone": dict(depth=50), "encoder": dict(in_channels=(512, 1024, 2048), expansion=1.0)},
}


def strip_prefix(sd: dict[str, Tensor], prefix: str, keep: str = "") -> dict[str, Tensor]:
    """Keys starting with `prefix` → `keep + rest`; drop the others."""
    return {keep + k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def backbone_state_dict(hf_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """HF `model.backbone.model.*` → `PResNet` keys."""
    return strip_prefix(hf_sd, BACKBONE_PREFIX)


def encoder_state_dict(hf_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """HF `model.encoder_input_proj.*` + `model.encoder.*` → `HybridEncoder` keys."""
    proj = strip_prefix(hf_sd, ENCODER_PREFIXES[0], keep="encoder_input_proj.")
    enc_body = strip_prefix(hf_sd, ENCODER_PREFIXES[1])
    return proj | enc_body


# Person B: HF keeps query selection + decoder directly under `model.` (siblings of `model.encoder`).
DECODER_PARTS = ("decoder_input_proj", "denoising_class_embed", "enc_output", "enc_score_head", "enc_bbox_head", "decoder")


def decoder_state_dict(hf_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """HF `model.<part>.*` for each of DECODER_PARTS → `RTDetrTransformer` keys (strip `model.` only).  [skeleton]"""
    # Hints:
    # 1. One strip_prefix call per part: prefix f"model.{part}.", keep f"{part}." — then merge the dicts.
    # 2. Same dot trick as the encoder: "model.decoder_input_proj.0..." does NOT start with "model.decoder.".
    # Self-check: 137 keys (R18) / 239 (R50); strict load into RTDetrTransformer(num_decoder_layers=3 / 6).
    raise NotImplementedError
