"""Export RT-DETR, or YOLOv8 with the EfficientNMS_TRT plugin, to ONNX and a TensorRT engine; `--check-ap` gives the
engine's COCO AP. Engines are GPU + TensorRT-version specific: build them on the machine that runs them."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `import src` works
from src.data.coco import CocoDetection  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.eval.coco_eval import evaluate_coco, to_coco_detections  # noqa: E402
from src.model import RTDETR, RTDETRDeploy  # noqa: E402
from src.utils.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from src.utils.trt import TRTModule, build_engine  # noqa: E402

RTDETR_INPUTS = ["images", "orig_sizes"]
RTDETR_OUTPUTS = ["scores", "labels", "boxes"]
YOLO_RAW_OUTPUTS = ["boxes", "scores"]                      # before the NMS node
YOLO_OUTPUTS = ["num_dets", "det_boxes", "det_scores", "det_classes"]   # after it (EfficientNMS_TRT names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="rtdetr_r18", help="rtdetr_r18 | rtdetr_r50 | yolov8s | yolov8m | yolov8l | yolov8x")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="--no-fp16 for an FP32 engine")
    p.add_argument("--opset", type=int, default=17, help=">= 16 needed for grid_sample (deformable attention)")
    p.add_argument("--workspace-gb", type=float, default=4.0)
    p.add_argument("--out-dir", default="engines")
    p.add_argument("--check-ap", action="store_true", help="COCO AP of the built engine")
    p.add_argument("--coco", default="data/coco")
    p.add_argument("--limit", type=int, default=None, help="AP on the first N val images only")
    p.add_argument("--conf", type=float, default=0.001, help="YOLO: NMS score threshold baked into the engine")
    p.add_argument("--iou", type=float, default=0.7, help="YOLO: NMS IoU threshold baked into the engine")
    p.add_argument("--max-det", type=int, default=300, help="YOLO: max detections per image")
    p.add_argument("--weights-dir", default="weights", help="YOLO .pt files (downloaded on first use)")
    return p.parse_args()


def export_onnx(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> Path:
    """Trace `module` once with `inputs` and save a static-shape ONNX graph. Returns onnx_path."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module, inputs, str(onnx_path), input_names=input_names, output_names=output_names,
            opset_version=opset, dynamo=False,
        )
    try:
        import onnxslim

        onnxslim.slim(str(onnx_path), str(onnx_path))     # folds constants (anchor grid), drops no-op nodes
    except ImportError:
        print("onnxslim not installed → unslimmed graph (still valid)")
    return onnx_path


def export_rtdetr(depth: int, size: int, onnx_path: Path, opset: int) -> Path:
    """RTDETR.from_hf(depth) wrapped in RTDETRDeploy → ONNX (on CPU, FP32 — TensorRT does the FP16 part)."""
    deploy = RTDETRDeploy(RTDETR.from_hf(depth)).eval()
    # Dummy inputs: real shapes + dtypes, values don't matter (tracing records ops, not values)
    images = torch.rand(1, 3, size, size)                  # float32, 0..1
    orig_sizes = torch.tensor([[size, size]])              # int64, (H, W)
    return export_onnx(deploy, (images, orig_sizes), onnx_path, RTDETR_INPUTS, RTDETR_OUTPUTS, opset)


def rtdetr_engine_ap(engine_path: Path, coco_root: str, size: int, limit: int | None) -> list[float]:
    """COCO val2017 AP of an RT-DETR engine — the same metric code as scripts/eval.py."""
    dataset = CocoDetection(
        Path(coco_root, "val2017"), Path(coco_root, "annotations/instances_val2017.json"), build_transforms(size)
    )
    data = Subset(dataset, range(limit)) if limit else dataset   # `dataset` keeps .coco / .cat_ids for the metric

    engine = TRTModule(engine_path)

    detections, image_ids = [], []
    with torch.inference_mode():
        for image, target in tqdm(data, desc=f"AP {engine_path.name}"):
            out = engine(image[None].cuda(), target["orig_size"][None].cuda())   # batch 1, (H, W) of the original
            # Output buffers are reused by the next call → to_coco_detections copies them to Python numbers now.
            results = [{"scores": out["scores"][0], "labels": out["labels"][0], "boxes": out["boxes"][0]}]
            detections += to_coco_detections(results, [target["image_id"]], dataset.cat_ids)
            image_ids.append(target["image_id"])

    return evaluate_coco(dataset.coco, detections, image_ids=image_ids)


# ------------------------------------------------------------------------------------------------- YOLO ----


class YOLOBoxesScores(nn.Module):
    """ultralytics net in export mode → (boxes [B, A, 4] xyxy pixels of the 640 input, scores [B, A, 80]).

    The raw head output is [B, 4 + 80, A] with (cx, cy, w, h) boxes and sigmoid class scores (A = 8400 anchors).
    EfficientNMS wants boxes [B, A, 4] and scores [B, A, C] → transpose + split. xyxy (box_coding=0) makes the
    output boxes unambiguous: corner format in = corner format out.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.net(images).transpose(1, 2)            # [B, A, 4 + C]
        return box_cxcywh_to_xyxy(pred[..., :4]), pred[..., 4:]


def load_yolo_net(model: str, weights_dir: str) -> nn.Module:
    """YOLOv8 DetectionModel, fused, FP32, eval, with the Detect head in export mode (returns one tensor)."""
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    weights = Path(weights_dir, f"{model}.pt")
    weights.parent.mkdir(parents=True, exist_ok=True)
    net = YOLO(str(weights)).model.float().fuse().eval()   # Conv+BN fused, as ultralytics deploys it
    for m in net.modules():
        if isinstance(m, Detect):
            m.export, m.format, m.dynamic = True, "onnx", False   # same switches ultralytics' own exporter sets
    return net


def add_efficient_nms(onnx_path: Path, conf: float, iou: float, max_det: int) -> Path:
    """Append TensorRT's EfficientNMS_TRT plugin node after the (boxes, scores) outputs, in place."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    batch = graph.output[0].type.tensor_type.shape.dim[0].dim_value   # static: 1

    graph.node.append(
        helper.make_node(
            "EfficientNMS_TRT",                            # plugin name in TensorRT's registry
            inputs=YOLO_RAW_OUTPUTS,
            outputs=YOLO_OUTPUTS,
            name="efficient_nms",
            plugin_version="1",
            background_class=-1,                           # no background class (all 80 are real)
            max_output_boxes=max_det,
            score_threshold=conf,
            iou_threshold=iou,
            score_activation=0,                            # scores are already sigmoid-ed by the head
            class_agnostic=0,                              # per-class NMS, like ultralytics
            box_coding=0,                                  # 0 = corners (x1, y1, x2, y2)
        )
    )
    del graph.output[:]                                    # the graph now ends at the NMS outputs
    graph.output.extend([
        helper.make_tensor_value_info("num_dets", TensorProto.INT32, [batch, 1]),
        helper.make_tensor_value_info("det_boxes", TensorProto.FLOAT, [batch, max_det, 4]),
        helper.make_tensor_value_info("det_scores", TensorProto.FLOAT, [batch, max_det]),
        helper.make_tensor_value_info("det_classes", TensorProto.INT32, [batch, max_det]),
    ])
    onnx.save(model, str(onnx_path))                       # no onnx.checker: it doesn't know plugin ops
    return onnx_path


def export_yolo(model: str, weights_dir: str, size: int, onnx_path: Path, opset: int,
                conf: float, iou: float, max_det: int) -> Path:
    """YOLOv8 → ONNX (boxes, scores) → + EfficientNMS_TRT node."""
    wrapper = YOLOBoxesScores(load_yolo_net(model, weights_dir)).eval()
    export_onnx(wrapper, (torch.rand(1, 3, size, size),), onnx_path, ["images"], YOLO_RAW_OUTPUTS, opset)
    return add_efficient_nms(onnx_path, conf, iou, max_det)   # after onnxslim: slim doesn't know the plugin op


def letterbox_image(path: str, size: int) -> torch.Tensor:
    """Image file → [1, 3, size, size] RGB 0..1, letterboxed exactly like ultralytics predict / val."""
    import cv2
    from ultralytics.data.augment import LetterBox

    img = LetterBox(new_shape=(size, size), auto=False)(image=cv2.imread(path))     # BGR, gray padding
    return torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float().div(255)[None]


def yolo_engine_ap(engine_path: Path, coco_root: str, size: int, limit: int | None) -> list[float]:
    """COCO val2017 AP of a YOLO + EfficientNMS engine — same metric code as scripts/eval_yolo.py."""
    from ultralytics.utils.ops import scale_boxes

    dataset = CocoDetection(Path(coco_root, "val2017"), Path(coco_root, "annotations/instances_val2017.json"))
    ids = dataset.ids[:limit] if limit else dataset.ids
    engine = TRTModule(engine_path)

    detections = []
    with torch.inference_mode():
        for image_id in tqdm(ids, desc=f"AP {engine_path.name}"):
            info = dataset.coco.imgs[image_id]
            out = engine(letterbox_image(str(Path(coco_root, "val2017", info["file_name"])), size).cuda())
            n = int(out["num_dets"][0, 0])                 # only the first n rows are real detections
            # letterboxed 640 pixels → original pixels (undo scale + padding); clone: the buffer is reused
            boxes = scale_boxes((size, size), out["det_boxes"][0, :n].float().clone(), (info["height"], info["width"]))
            results = [{"scores": out["det_scores"][0, :n], "labels": out["det_classes"][0, :n], "boxes": boxes}]
            detections += to_coco_detections(results, [image_id], dataset.cat_ids)   # YOLO order == COCO sorted ids

    return evaluate_coco(dataset.coco, detections, image_ids=list(ids))


# ------------------------------------------------------------------------------------------------- main ----


def main() -> None:
    args = parse_args()
    precision = "fp16" if args.fp16 else "fp32"
    is_yolo = args.model.startswith("yolo")
    if is_yolo:
        name = f"{args.model}_conf{args.conf}_iou{args.iou}_{precision}"   # NMS settings are part of the engine
    elif args.model.startswith("rtdetr_r"):
        name = f"{args.model}_{precision}"
    else:
        raise ValueError(f"unknown --model {args.model!r}")
    onnx_path = Path(args.out_dir, name + ".onnx")
    engine_path = onnx_path.with_suffix(".engine")

    print(f"[1/3] ONNX   → {onnx_path}")
    if is_yolo:
        export_yolo(args.model, args.weights_dir, args.size, onnx_path, args.opset, args.conf, args.iou, args.max_det)
    else:
        export_rtdetr(int(args.model.split("_r")[1]), args.size, onnx_path, args.opset)

    print(f"[2/3] engine → {engine_path}  (TensorRT autotunes every layer: takes minutes)")
    t0 = time.perf_counter()
    build_engine(onnx_path, engine_path, fp16=args.fp16, workspace_gb=args.workspace_gb)
    build_s = time.perf_counter() - t0

    import tensorrt                                        # lazy, like src/utils/trt.py

    report = {
        "model": args.model,
        "precision": precision,
        "input": [1, 3, args.size, args.size],
        "opset": args.opset,
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "tensorrt": tensorrt.__version__,
        "gpu": torch.cuda.get_device_name(),
        "build_s": round(build_s, 1),
    }
    if is_yolo:
        report |= {"nms": "EfficientNMS_TRT", "conf": args.conf, "iou": args.iou, "max_det": args.max_det}
    if args.check_ap:
        print("[3/3] COCO AP of the engine")
        ap_fn = yolo_engine_ap if is_yolo else rtdetr_engine_ap
        stats = ap_fn(engine_path, args.coco, args.size, args.limit)
        report |= {"AP": round(stats[0], 4), "AP50": round(stats[1], 4), "ap_images": args.limit or 5000}

    report_path = engine_path.with_suffix(".json")         # scripts/benchmark.py --engine merges this report
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved → {report_path}")


if __name__ == "__main__":
    main()
