"""Speed + size benchmark, RT-DETR vs YOLO: params, GFLOPs, latency (batch 1), end-to-end incl. NMS.  [SHARED]

Usage:
    python scripts/benchmark.py --model rtdetr_r18                             # our whole model, FP16 (default)
    python scripts/benchmark.py --model rtdetr_r50
    python scripts/benchmark.py --model rtdetr_r18 --mode hf                   # HF's model, for reference
    python scripts/benchmark.py --model yolov8l                                # model + NMS, + NMS threshold sweep
    python scripts/benchmark.py --model yolov8s --no-fp16                      # FP32
    python scripts/benchmark.py --engine engines/rtdetr_r18_ours_fp16.engine   # TensorRT engine (scripts/export_trt.py)
    python scripts/benchmark.py --engine engines/yolov8l_conf0.001_iou0.7_fp16.engine   # YOLO + EfficientNMS plugin

What is measured (batch 1, 640×640, like the paper), on ONE real COCO image (NMS cost depends on content):
    params / gflops   size of the network (torch FlopCounterMode, conv/matmul; 1 multiply-add = 2 FLOPs)
    model_ms          median latency of the network forward
    post_ms           RT-DETR: sigmoid + top-k (no NMS)      YOLO: NMS at --conf / --iou
    e2e_ms / fps      model + post — the number the paper's FPS column is about
    nms_sweep (YOLO)  boxes entering NMS + NMS ms for conf 0.001 / 0.01 / 0.05 / 0.25, averaged over
                      --sweep-images real val2017 images (paper §3, Fig. 2 / Table 1 — NMS cost depends on content)

Fairness notes:
    - same timer, same image, same FP16 autocast, batch 1, cudnn.benchmark for both families.
    - YOLO is Conv+BN-fused (as ultralytics deploys it); our RT-DETR is NOT fused → conservative for RT-DETR.
    - Paper: T4 + TensorRT FP16 + EfficientNMS plugin. Here: PyTorch on your GPU. Compare models on the SAME GPU.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `import src` works
from src.backbone.presnet import PResNet  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.encoder.hybrid_encoder import HybridEncoder  # noqa: E402
from src.model import RTDETR  # noqa: E402
from src.postprocess import post_process  # noqa: E402
from src.utils.hf_weights import HF_CONFIGS, backbone_state_dict, encoder_state_dict  # noqa: E402

DECODER_SIDE = ("decoder_input_proj", "denoising_class_embed", "enc_output", "enc_score_head", "enc_bbox_head", "decoder")
NMS_SWEEP = (0.001, 0.01, 0.05, 0.25)
DEFAULT_IMAGE = "data/coco/val2017/000000039769.jpg"


def count_params(model: torch.nn.Module) -> float:
    """Millions of parameters."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gflops(fn, x) -> float:
    """GFLOPs of one fn(x) call (conv / matmul ops counted by torch; 1 MAC = 2 FLOPs)."""
    with FlopCounterMode(display=False) as flop_counter:
        fn(x)
    return flop_counter.get_total_flops() / 1e9


def time_fn(fn, x, warmup: int = 50, iters: int = 200) -> dict:
    """Median / p90 latency in ms of fn(x), GPU-synchronised."""
    cuda = x.is_cuda
    for _ in range(warmup):                       # cuDNN autotune, memory allocation, lazy init
        fn(x)
    times = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()              # CUDA calls are async: wait before starting the clock …
        t = time.perf_counter()
        fn(x)
        if cuda:
            torch.cuda.synchronize()              # … and until the GPU has really finished
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"median_ms": round(statistics.median(times), 3), "p90_ms": round(times[int(0.9 * len(times)) - 1], 3)}


def load_image(path: str | None, size: int, letterbox: bool, device: str) -> torch.Tensor:
    """One real image as [1, 3, size, size] in [0, 1]: RT-DETR = plain resize (HF processor), YOLO = letterbox."""
    if not path or not Path(path).exists():
        print(f"image {path!r} not found → random input (NMS timings will be unrealistically low)")
        return torch.rand(1, 3, size, size, device=device)
    if letterbox:
        import cv2
        from ultralytics.data.augment import LetterBox

        img = LetterBox(new_shape=(size, size), auto=False)(image=cv2.imread(path))     # BGR, gray padding
        x = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255     # → RGB CHW
    else:
        x, _ = build_transforms(size)(Image.open(path).convert("RGB"), {})
    return x[None].to(device)


def build_rtdetr(model: str, mode: str, device: str):
    """→ (model_fn(x) -> (logits, pred_boxes), post_fn(out, size), params_M). Same modes as scripts/eval.py."""
    depth = int(model.split("_r")[1])
    if mode == "full":
        net = RTDETR.from_hf(depth).to(device)                           # our whole model

        def model_fn(x):
            out = net(x)
            return out["logits"], out["pred_boxes"]

        params = count_params(net)
    else:
        from transformers import RTDetrForObjectDetection

        hf = RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").eval().to(device)
        if mode == "hf":
            def model_fn(x):
                out = hf(pixel_values=x)
                return out.logits, out.pred_boxes

            params = count_params(hf)
        else:
            sd = hf.state_dict()
            backbone = PResNet(**HF_CONFIGS[depth]["backbone"])
            backbone.load_state_dict(backbone_state_dict(sd), strict=True)
            encoder = HybridEncoder(**HF_CONFIGS[depth]["encoder"])
            encoder.load_state_dict(encoder_state_dict(sd), strict=True)
            backbone.eval().to(device)
            encoder.eval().to(device)

            def model_fn(x):
                feats = encoder(backbone(x))
                out = hf(pixel_values=None, inputs_embeds=feats[0], encoder_outputs=(feats,))   # HF backbone/encoder skipped
                return out.logits, out.pred_boxes

            # what actually runs: our backbone + encoder + HF's query selection / decoder
            params = count_params(backbone) + count_params(encoder) + sum(count_params(getattr(hf.model, n)) for n in DECODER_SIDE)

    def post_fn(out, size):
        logits, pred_boxes = out
        sizes = torch.tensor([[size, size]], device=logits.device)
        return post_process(logits, pred_boxes, sizes, threshold=0.5)

    return model_fn, post_fn, params


def build_yolo(model: str, device: str, weights_dir: str, conf: float, iou: float, max_det: int):
    """→ (model_fn, post_fn(out), params_M, nms(pred, conf)). Raw fused network + ultralytics NMS."""
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression

    weights = Path(weights_dir, f"{model}.pt")
    weights.parent.mkdir(parents=True, exist_ok=True)
    net = YOLO(str(weights)).model.fuse().eval().to(device)          # Conv+BN fused, like ultralytics deployment

    def model_fn(x):
        return net(x)[0]                                              # [1, 84, 8400]: 4 box + 80 class scores

    def nms(pred, conf_thres):
        return non_max_suppression(pred, conf_thres, iou, max_det=max_det)   # same call as ultralytics predict

    def post_fn(pred, size):
        return nms(pred, conf)

    return model_fn, post_fn, count_params(net), nms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="rtdetr_r18", help="rtdetr_r18 | rtdetr_r50 | yolov8s | yolov8m | yolov8l | yolov8x")
    p.add_argument("--mode", default="full", choices=["full", "ours", "hf"],
                   help="RT-DETR only: full = our whole model, ours = our backbone+encoder + HF decoder, hf = HF")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="--no-fp16 for FP32")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--image", default=DEFAULT_IMAGE, help="real COCO image used as input")
    p.add_argument("--conf", type=float, default=0.001, help="YOLO NMS confidence threshold for e2e (paper best AP)")
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--sweep-images", type=int, default=50, help="YOLO NMS sweep: average over the first N val images")
    p.add_argument("--coco", default="data/coco")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--engine", default=None, help="time a TensorRT engine from scripts/export_trt.py instead")
    p.add_argument("--out", default=None, help="JSON path; default results/speed_<model>[_<mode>].json")
    return p.parse_args()


def bench_engine(args) -> dict:
    """End-to-end latency of a TensorRT engine (RT-DETR: incl. top-k; YOLO: incl. EfficientNMS plugin)."""
    from src.utils.trt import TRTModule

    engine_path = Path(args.engine)
    engine = TRTModule(engine_path)
    is_rtdetr = "orig_sizes" in engine.input_names
    x = load_image(args.image, args.size, letterbox=not is_rtdetr, device="cuda")
    inputs = (x, torch.tensor([[args.size, args.size]], device="cuda")) if is_rtdetr else (x,)
    t = time_fn(lambda _: engine(*inputs), x, args.warmup, args.iters)
    record = {"engine": str(engine_path), "backend": "tensorrt",
              "device": torch.cuda.get_device_name(), "input": [1, 3, args.size, args.size], "image": args.image,
              "post": "sigmoid + top-k inside engine (no NMS)" if is_rtdetr else "EfficientNMS_TRT plugin inside engine",
              "e2e_ms": t["median_ms"], "e2e_p90_ms": t["p90_ms"], "fps": round(1000 / t["median_ms"], 1)}
    build_report = engine_path.with_suffix(".json")                     # written by export_trt.py
    if build_report.exists():
        record = {**json.loads(build_report.read_text()), **record}
    return record


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    if args.engine:
        record = bench_engine(args)
        print(json.dumps(record, indent=2))
        out_path = Path(args.out or Path("results", f"speed_trt_{Path(args.engine).stem}.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        print(f"saved → {out_path}")
        return
    is_yolo = args.model.startswith("yolo")

    if is_yolo:
        model_fn, post_fn, params, nms = build_yolo(args.model, device, args.weights_dir, args.conf, args.iou, args.max_det)
    elif args.model.startswith("rtdetr_r"):
        model_fn, post_fn, params = build_rtdetr(args.model, args.mode, device)
    else:
        raise ValueError(f"unknown --model {args.model!r}")

    x = load_image(args.image, args.size, letterbox=is_yolo, device=device)
    fp16 = args.fp16 and device == "cuda"
    amp = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16)

    with torch.inference_mode():
        gflops = count_gflops(model_fn, x)                            # FP32, counts don't depend on dtype
        with amp:
            out = model_fn(x)                                         # fixed output for timing post alone

            def e2e_fn(inp):
                return post_fn(model_fn(inp), args.size)

            t_model = time_fn(model_fn, x, args.warmup, args.iters)
            t_post = time_fn(lambda _: post_fn(out, args.size), x, args.warmup, args.iters)
            t_e2e = time_fn(e2e_fn, x, args.warmup, args.iters)

            sweep = []
            if is_yolo:                                               # paper §3: NMS cost vs confidence threshold
                files = sorted(Path(args.coco, "val2017").glob("*.jpg"))[: args.sweep_images] or [None]
                preds = [model_fn(load_image(str(f) if f else None, args.size, True, device)) for f in files]
                for c in NMS_SWEEP:
                    boxes_in, boxes_out, ms = [], [], []
                    for pred in preds:
                        boxes_in.append(int((pred[:, 4:].amax(1) > c).sum()))     # best class score per anchor
                        boxes_out.append(len(nms(pred, c)[0]))
                        ms.append(time_fn(lambda _, p=pred, c=c: nms(p, c), x, warmup=5, iters=20)["median_ms"])
                    sweep.append({"conf": c, "images": len(preds), "boxes_in_mean": round(statistics.mean(boxes_in), 1),
                                  "boxes_in_max": max(boxes_in), "boxes_out_mean": round(statistics.mean(boxes_out), 1),
                                  "nms_ms_mean": round(statistics.mean(ms), 3), "nms_ms_max": round(max(ms), 3)})

    record = {
        "model": args.model, **({} if is_yolo else {"mode": args.mode}),
        "precision": "fp16" if fp16 else "fp32",
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "input": [1, 3, args.size, args.size], "image": args.image,
        "params_M": round(params, 2), "gflops": round(gflops, 1),
        "model_ms": t_model["median_ms"],
        "post": f"NMS conf={args.conf} iou={args.iou}" if is_yolo else "sigmoid + top-k (no NMS)",
        "post_ms": t_post["median_ms"],
        "e2e_ms": t_e2e["median_ms"], "e2e_p90_ms": t_e2e["p90_ms"], "fps": round(1000 / t_e2e["median_ms"], 1),
        **({"nms_sweep": sweep} if is_yolo else {}),
        "torch": torch.__version__,
    }
    print(json.dumps(record, indent=2))

    default_name = f"speed_{args.model}" + ("" if is_yolo else f"_{args.mode}") + ".json"
    out_path = Path(args.out or Path("results", default_name))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
