# RT-DETR From Scratch

A from-scratch PyTorch reimplementation of the core ideas from:

> **DETRs Beat YOLOs on Real-time Object Detection**
> Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wu, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen
> [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)

This project is built for learning purposes, using the [official PaddlePaddle implementation](https://github.com/lyuwenyu/RT-DETR) as a reference for correctness — not copied line-for-line. Where our implementation choices diverge from the official repo, it's noted in [`docs/implementation_notes.md`](docs/implementation_notes.md).

## Team

| Name | Focus area |
|---|---|
| TBD (Person A) | Backbone + Hybrid Encoder (AIFI + CCFM) |
| TBD (Person B) | Decoder + Query Selection + Matcher + Losses |

## Why RT-DETR

RT-DETR removes the need for NMS post-processing (unlike YOLO-family detectors) while matching or beating YOLO's real-time speed/accuracy tradeoff, by:
1. An **efficient hybrid encoder** — decouples intra-scale interaction (transformer) from cross-scale fusion (CNN-based), instead of naive multi-scale deformable attention.
2. **Uncertainty-minimal query selection** — selects encoder features for decoder queries using both classification and localization scores (IoU-aware), not just objectness.
3. Support for **flexible speed tuning** by adjusting decoder layers without retraining.

## Architecture Status

| Component | Status | Notes |
|---|---|---|
| Backbone (ResNet/HGNetv2) | ⬜ Not started | |
| AIFI (intra-scale transformer) | ⬜ Not started | |
| CCFM (cross-scale fusion) | ⬜ Not started | |
| Query selection (IoU-aware) | ⬜ Not started | |
| Decoder + auxiliary heads | ⬜ Not started | |
| Hungarian matcher | ⬜ Not started | |
| Loss (VFL + L1 + GIoU) | ⬜ Not started | |
| Training loop | ⬜ Not started | |
| COCO eval (mAP) | ⬜ Not started | |
| Speed benchmark (FPS/latency) | ⬜ Not started | |

Update this table as PRs land. Use ⬜ Not started / 🟨 In progress / ✅ Done.

## Results (target: reproduce paper Table 1, scaled to our compute budget)

| Model | mAP (paper) | mAP (ours) | FPS (paper, T4) | FPS (ours) |
|---|---|---|---|---|
| RT-DETR-R18 | 46.5 | — | 217 | — |
| RT-DETR-R50 | 53.1 | — | 108 | — |

## Setup

```bash
git clone https://github.com/<org>/rt-detr-from-scratch.git
cd rt-detr-from-scratch
conda env create -f environment.yml
conda activate rtdetr
# or: pip install -r requirements.txt
```

## Data

We use COCO 2017. Expected layout:

```
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

Download helper:
```bash
bash scripts/download_coco.sh
```

## Training

```bash
python scripts/train.py --config configs/rtdetr_r18.yaml
```

## Evaluation

```bash
python scripts/eval.py --config configs/rtdetr_r18.yaml --checkpoint checkpoints/best.pth
```

## Repo Structure

```
src/
├── backbone/     # ResNet / HGNetv2 feature extractors
├── encoder/      # Hybrid encoder: AIFI + CCFM
├── decoder/      # Transformer decoder + query selection
├── matcher/      # Hungarian bipartite matcher
├── losses/       # VFL / L1 / GIoU losses
├── data/         # COCO dataset + augmentations
└── utils/        # box ops, misc helpers
```

## Development workflow

- `main`: stable, always mergeable
- `dev`: integration branch, PRs land here first
- `feat/<component>`: feature branches (e.g. `feat/hybrid-encoder`)
- PRs require: shape checks with dummy input, a note on which paper section it implements, and (where feasible) a unit test

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## References

- Paper: [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- Official code: https://github.com/lyuwenyu/RT-DETR
- Deformable DETR (prior work, useful background): https://arxiv.org/abs/2010.04159

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Chosen for compatibility with the official RT-DETR repo's license.
