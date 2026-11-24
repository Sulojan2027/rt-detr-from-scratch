"""src/model.py — RTDETR (our whole model) and RTDETRDeploy (what goes into ONNX / TensorRT)."""

import warnings

import pytest
import torch

from src.model import RTDETR, RTDETRDeploy
from src.postprocess import topk_detections
from src.utils.hf_weights import HF_CONFIGS


@pytest.mark.parametrize("depth, params_m", [(18, 20.17), (50, 42.89)])
def test_param_count(depth, params_m):
    model = RTDETR(depth)
    assert round(sum(p.numel() for p in model.parameters()) / 1e6, 2) == params_m
    assert len(model.decoder.decoder.layers) == HF_CONFIGS[depth]["decoder"]["num_decoder_layers"]


def test_forward_shapes_random_weights():
    model = RTDETR(18).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 320, 320))
    assert out["logits"].shape == (2, 300, 80) and out["pred_boxes"].shape == (2, 300, 4)


def test_deploy_matches_topk_on_model_output():
    torch.manual_seed(0)
    model = RTDETR(18).eval()
    deploy = RTDETRDeploy(model, num_top=100).eval()
    x = torch.rand(1, 3, 320, 320)
    sizes = torch.tensor([[480, 640]])
    with torch.no_grad():
        scores, labels, boxes = deploy(x, sizes)
        out = model(x)
        ref = topk_detections(out["logits"], out["pred_boxes"], sizes, 100)
    assert scores.shape == (1, 100) and labels.shape == (1, 100) and boxes.shape == (1, 100, 4)
    assert labels.dtype == torch.int32 and int(labels.max()) < 80
    torch.testing.assert_close(scores, ref[0])
    torch.testing.assert_close(labels, ref[1].to(torch.int32))
    torch.testing.assert_close(boxes, ref[2])


def test_deploy_traces_for_onnx():
    """torch.onnx.export (dynamo=False) traces with torch.jit.trace — tracing must work and replay the same result."""
    model = RTDETR(18).eval()
    deploy = RTDETRDeploy(model).eval()
    x, sizes = torch.rand(1, 3, 320, 320), torch.tensor([[320, 320]])
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)   # shape checks become constants: fine, shapes are static
        traced = torch.jit.trace(deploy, (x, sizes), check_trace=False)
        for got, want in zip(traced(x, sizes), deploy(x, sizes), strict=True):
            torch.testing.assert_close(got, want)


@pytest.mark.slow
@pytest.mark.parametrize("depth", [18, 50])
def test_from_hf_matches_hf(depth):
    transformers = pytest.importorskip("transformers")
    hf = transformers.RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").eval()
    model = RTDETR.from_hf(depth)
    assert not model.training
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        ref = hf(pixel_values=x)
        out = model(x)
    assert (out["logits"].sigmoid() - ref.logits.sigmoid()).abs().max() < 1e-4
    assert (out["pred_boxes"] - ref.pred_boxes).abs().max() < 1e-4
