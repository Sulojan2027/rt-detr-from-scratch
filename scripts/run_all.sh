#!/usr/bin/env bash
# Every RT-DETR vs YOLOv8 benchmark on the server (stages: check accuracy engines speed nms visualize).
#   LIMIT=50 bash scripts/run_all.sh    # dry run first (minutes)      bash scripts/run_all.sh    # full run (hours)
#   Options via env vars: STAGES, RTDETR, YOLO, COCO, PY, OUT, ENGINES, REBUILD=1 — see the defaults below.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
COCO=${COCO:-data/coco}
LIMIT=${LIMIT:-}
RTDETR=${RTDETR:-"rtdetr_r18 rtdetr_r50"}
YOLO=${YOLO:-"yolov8s yolov8m yolov8l yolov8x"}
NMS_MODEL=${NMS_MODEL:-yolov8l}
NMS_CONFS=${NMS_CONFS:-"0.001 0.01 0.05"}
STAGES=${STAGES:-"check accuracy engines speed nms visualize"}
REBUILD=${REBUILD:-0}
OUT=${OUT:-results${LIMIT:+/limit$LIMIT}}
ENGINES=${ENGINES:-engines${LIMIT:+/limit$LIMIT}}
LIMIT_ARGS=()
[[ -n $LIMIT ]] && LIMIT_ARGS=(--limit "$LIMIT")
mkdir -p "$OUT/logs" "$ENGINES"

has() { [[ " $STAGES " == *" $1 "* ]]; }
run() {   # run <log name> <command...>: echo, run, tee to log; stops the whole script on failure (set -e + pipefail)
    local name=$1
    shift
    echo -e "\n\033[1m▶ $name\033[0m  $*"
    "$@" 2>&1 | tee "$OUT/logs/$name.log"
}
engine_name() {   # same naming as scripts/export_trt.py
    if [[ $1 == yolo* ]]; then echo "$1_conf$2_iou0.7_fp16"; else echo "$1_fp16"; fi
}
build_engine() {   # build_engine <model> [conf] — export + build + engine AP, skipped if already built
    local model=$1 conf=${2:-0.001} name
    name=$(engine_name "$model" "$conf")
    if [[ $REBUILD != 1 && -f $ENGINES/$name.engine && -f $ENGINES/$name.json ]]; then
        echo "✓ $name already built (REBUILD=1 to redo)"
        return
    fi
    run "export_$name" "$PY" scripts/export_trt.py --model "$model" --conf "$conf" --out-dir "$ENGINES" \
        --coco "$COCO" --check-ap "${LIMIT_ARGS[@]}"
}

# ---------------------------------------------------------------------------------------------------------------
if has check; then
    {
        date
        nvidia-smi
        "$PY" - <<'EOF'
import importlib
for name in ("torch", "torchvision", "transformers", "ultralytics", "onnx", "onnxslim", "tensorrt", "pycocotools"):
    try:
        print(f"{name:13s}", getattr(importlib.import_module(name), "__version__", "?"))
    except ImportError:
        print(f"{name:13s} MISSING")
import torch
print("cuda", torch.version.cuda, "|", torch.cuda.get_device_name() if torch.cuda.is_available() else "NO GPU")
EOF
    } 2>&1 | tee "$OUT/logs/env.log"
    n=$(find "$COCO/val2017" -name '*.jpg' 2>/dev/null | wc -l)
    [[ $n == 5000 && -f $COCO/annotations/instances_val2017.json ]] || { echo "COCO val2017 incomplete ($n/5000 images) in $COCO"; exit 1; }
    run tests_fast "$PY" -m pytest tests -q -m "not slow"
    run tests_trt "$PY" -m pytest tests/test_trt.py tests/test_export_trt.py tests/test_model.py -q -m slow
fi

if has accuracy; then
    for m in $RTDETR; do
        run "acc_$m" "$PY" scripts/eval.py --depth "${m#rtdetr_r}" --mode full --coco "$COCO" \
            --out "$OUT/acc_$m.json" "${LIMIT_ARGS[@]}"
    done
    for m in $YOLO; do
        run "acc_$m" "$PY" scripts/eval_yolo.py --model "$m" --coco "$COCO" --out "$OUT/acc_$m.json" "${LIMIT_ARGS[@]}"
    done
fi

if has engines; then
    for m in $RTDETR $YOLO; do build_engine "$m"; done
fi

if has speed; then
    for m in $RTDETR $YOLO; do
        name=$(engine_name "$m" 0.001)
        run "speed_trt_$name" "$PY" scripts/benchmark.py --engine "$ENGINES/$name.engine" --out "$OUT/speed_trt_$name.json"
        run "speed_torch_$m" "$PY" scripts/benchmark.py --model "$m" --coco "$COCO" --out "$OUT/speed_torch_$m.json"
    done
fi

if has nms; then
    for c in $NMS_CONFS; do
        build_engine "$NMS_MODEL" "$c"
        name=$(engine_name "$NMS_MODEL" "$c")
        run "nms_$name" "$PY" scripts/benchmark.py --engine "$ENGINES/$name.engine" --out "$OUT/nms_$name.json"
    done
fi

if has visualize; then
    run visualize "$PY" scripts/visualize.py --rtdetr rtdetr_r50 --yolo yolov8l --num 12 --gt --coco "$COCO" --out "$OUT/vis"
fi

echo -e "\n\033[1mDone.\033[0m Results in $OUT/:"
ls -1 "$OUT"
