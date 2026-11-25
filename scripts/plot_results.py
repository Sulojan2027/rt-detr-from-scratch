"""Plot the benchmark results in results/*.json (accuracy, speed vs AP, NMS sweep, AP by object size).

Usage: python scripts/plot_results.py    # → results/plots/*.png (300 dpi) and *.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"   # results/*.json from scripts/run_all.sh
OUT = RES / "plots"

BLUE, ORANGE, INK, SOFT, GRID, BG = "#2F7FC8", "#A04A12", "#23384D", "#4A6177", "#C3D9E8", "#F5FBFE"

# Paper numbers (Zhao et al., CVPR 2024): Table 2 AP, Table 1 NMS sweep for YOLOv8-L
PAPER_AP = {"rtdetr_r18": 46.5, "rtdetr_r50": 53.1, "yolov8s": 44.9, "yolov8m": 50.2, "yolov8l": 52.9, "yolov8x": 53.9}
PAPER_NMS = {0.001: 52.9, 0.01: 52.4, 0.05: 51.2}
NAMES = {"rtdetr_r18": "RT-DETR-R18", "rtdetr_r50": "RT-DETR-R50",
         "yolov8s": "YOLOv8-S", "yolov8m": "YOLOv8-M", "yolov8l": "YOLOv8-L", "yolov8x": "YOLOv8-X"}

plt.rcParams.update({
    "font.size": 13, "axes.edgecolor": SOFT, "axes.labelcolor": SOFT, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.facecolor": BG, "figure.facecolor": "white",
    "text.color": INK, "legend.frameon": False,
})


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def trt(model: str) -> dict:
    """TensorRT FP16 result (speed + AP on 5,000 images); YOLO at conf 0.001, the paper's setting."""
    suffix = "_fp16" if model.startswith("rtdetr") else "_conf0.001_iou0.7_fp16"
    return load(f"speed_trt_{model}{suffix}.json")


def color(model: str) -> str:
    return BLUE if model.startswith("rtdetr") else ORANGE


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"saved → {OUT / name}.png / .pdf")


# ---------------------------------------------------------------- 1. accuracy: paper vs ours ----
def plot_accuracy() -> None:
    models = list(PAPER_AP)
    ours = [trt(m)["AP"] * 100 for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(models))
    ax.scatter(x, [PAPER_AP[m] for m in models], s=320, facecolors="none", edgecolors=SOFT, linewidths=2, label="Paper", zorder=3)
    ax.scatter(x, ours, s=90, c=[color(m) for m in models], label="Ours (TensorRT FP16)", zorder=4)
    for i, (m, v) in enumerate(zip(models, ours)):
        ax.annotate(f"{v:.1f}", (i, v), xytext=(16, -4), textcoords="offset points", fontsize=11, color=INK)
    ax.set_xticks(list(x), [NAMES[m] for m in models], rotation=20)
    ax.set_ylabel("COCO AP (val2017)")
    ax.set_ylim(43, 56)
    ax.set_title("Accuracy: every model within 0.2 AP of the paper", loc="left", color=INK, fontweight="bold")
    ax.legend(loc="lower right")
    save(fig, "accuracy")


# ---------------------------------------------------------------- 2. speed: AP vs latency ----
def plot_speed() -> None:
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for family, models in (("RT-DETR", ["rtdetr_r18", "rtdetr_r50"]), ("YOLOv8", ["yolov8s", "yolov8m", "yolov8l", "yolov8x"])):
        r = [trt(m) for m in models]
        lat, ap = [d["e2e_ms"] for d in r], [d["AP"] * 100 for d in r]
        c = color(models[0])
        ax.plot(lat, ap, "-o", color=c, lw=2.5, ms=9, mec="white", mew=2, label=family, zorder=3)
        for m, x, y in zip(models, lat, ap):
            ax.annotate(NAMES[m].split("-")[-1], (x, y), xytext=(9, -14 if m == "yolov8s" else 6),
                        textcoords="offset points", fontweight="bold", color=INK)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(43.5, 55)
    ax.set_xlabel("Latency per image, ms\n(TensorRT FP16, RTX 3090, batch 1, NMS included)")
    ax.set_ylabel("COCO AP")
    ax.set_title("Speed vs accuracy: up and left is better", loc="left", color=INK, fontweight="bold")
    ax.legend(loc="lower right")
    save(fig, "speed_vs_ap")


# ---------------------------------------------------------------- 3. NMS threshold sweep ----
def plot_nms() -> None:
    rows = []
    for conf in PAPER_NMS:
        d = load(f"nms_yolov8l_conf{conf}_iou0.7_fp16.json")
        rows.append((f"YOLOv8-L\nconf {conf}", d["e2e_ms"], d["AP"] * 100, ORANGE))
    r50 = trt("rtdetr_r50")
    rows.append(("RT-DETR-R50\n(no NMS)", r50["e2e_ms"], r50["AP"] * 100, BLUE))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels, ms, ap, cols = zip(*rows)
    bars = ax.bar(labels, ms, color=cols, width=0.6, zorder=3)
    for b, t, a in zip(bars, ms, ap):
        ax.text(b.get_x() + b.get_width() / 2, t + 0.12, f"{t:.2f} ms\n{a:.1f} AP", ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_ylim(0, max(ms) * 1.3)
    ax.set_ylabel("End-to-end latency, ms")
    ax.grid(axis="x", visible=False)
    ax.set_title("Cost of NMS: lower threshold → higher AP, slower", loc="left", color=INK, fontweight="bold")
    save(fig, "nms_sweep")


# ---------------------------------------------------------------- 4. AP by object size ----
def plot_ap_by_size() -> None:
    sizes = [("APs", "Small"), ("APm", "Medium"), ("APl", "Large")]
    models = ["rtdetr_r50", "yolov8l"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.36
    for k, m in enumerate(models):
        d = load(f"acc_{m}.json")
        vals = [d[key] * 100 for key, _ in sizes]
        xs = [i + (k - 0.5) * w for i in range(len(sizes))]
        bars = ax.bar(xs, vals, width=w * 0.94, color=color(m), label=NAMES[m], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=11, color=INK)
    ax.set_xticks(range(len(sizes)), [n for _, n in sizes])
    ax.set_ylim(0, 80)
    ax.set_ylabel("COCO AP (PyTorch)")
    ax.grid(axis="x", visible=False)
    ax.set_title("AP by object size: small objects are hard for both", loc="left", color=INK, fontweight="bold")
    ax.legend(loc="upper left")
    save(fig, "ap_by_size")


if __name__ == "__main__":
    plot_accuracy()
    plot_speed()
    plot_nms()
    plot_ap_by_size()
