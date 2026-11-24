"""scripts/export_trt.py — YOLO wrapper (runs anywhere with ultralytics + the .pt) and the EfficientNMS plugin
wiring (needs tensorrt + onnx + CUDA → runs on the 3090 server only)."""

import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
ns = runpy.run_path(str(ROOT / "scripts" / "export_trt.py"), run_name="export_trt")   # functions, no main()


@pytest.fixture(scope="module")
def yolo_net():
    pytest.importorskip("ultralytics")
    weights = ROOT / "weights" / "yolov8s.pt"
    if not weights.exists():
        pytest.skip("weights/yolov8s.pt not downloaded")
    return ns["load_yolo_net"]("yolov8s", str(weights.parent))


def test_yolo_wrapper_shapes_and_box_format(yolo_net):
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        raw = yolo_net(x)                                  # export mode → one tensor [1, 84, 8400]
        boxes, scores = ns["YOLOBoxesScores"](yolo_net)(x)
    assert raw.shape == (1, 84, 8400)
    assert boxes.shape == (1, 8400, 4) and scores.shape == (1, 8400, 80)
    cx, cy, w, h = raw[0, :4]
    torch.testing.assert_close(boxes[0], torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1))
    torch.testing.assert_close(scores[0], raw[0, 4:].T)
    assert (scores >= 0).all() and (scores <= 1).all()    # already sigmoid → score_activation=0 in the plugin


def test_engine_names():
    """benchmark.py / run_all.sh rely on these file names."""
    assert ns["RTDETR_INPUTS"] == ["images", "orig_sizes"]
    assert ns["YOLO_OUTPUTS"] == ["num_dets", "det_boxes", "det_scores", "det_classes"]


class _PassThrough(nn.Module):
    """(boxes, scores) in → (boxes, scores) out, through real ops so the graph isn't empty."""

    def forward(self, boxes, scores):
        return torch.relu(boxes), scores.clamp(0, 1)


@pytest.mark.slow
def test_efficient_nms_plugin_engine(tmp_path):
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnx")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from src.utils.trt import TRTModule, build_engine

    boxes = torch.tensor([[[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]]], dtype=torch.float32)
    scores = torch.tensor([[[0.9, 0.0], [0.8, 0.0], [0.0, 0.7]]])
    onnx_path = tmp_path / "nms.onnx"
    with torch.no_grad():
        torch.onnx.export(_PassThrough(), (boxes, scores), str(onnx_path), input_names=["in_boxes", "in_scores"],
                          output_names=ns["YOLO_RAW_OUTPUTS"], opset_version=17, dynamo=False)
    ns["add_efficient_nms"](onnx_path, conf=0.25, iou=0.5, max_det=10)
    engine = TRTModule(build_engine(onnx_path, tmp_path / "nms.engine", fp16=False))
    assert list(engine.outputs) == ns["YOLO_OUTPUTS"]

    out = engine(boxes.cuda(), scores.cuda())
    torch.cuda.synchronize()
    # box 1 overlaps box 0 (IoU 0.81, same class) → suppressed; box 2 is another class → kept
    assert int(out["num_dets"][0, 0]) == 2
    torch.testing.assert_close(out["det_scores"][0, :2].cpu(), torch.tensor([0.9, 0.7]))
    assert out["det_classes"][0, :2].tolist() == [0, 1]
    torch.testing.assert_close(out["det_boxes"][0, :2].cpu(), boxes[0, [0, 2]])   # corners in = corners out
