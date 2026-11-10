# RT-DETR From Scratch

A from-scratch PyTorch reimplementation of the core ideas from:

> **DETRs Beat YOLOs on Real-time Object Detection**
> Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wu, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen
> [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)

This project is built for learning purposes. References are the [official implementation](https://github.com/lyuwenyu/RT-DETR) and the [Hugging Face `transformers` implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/rt_detr). Module names follow Hugging Face, so the pretrained `PekingU/rtdetr_r18vd` / `rtdetr_r50vd` weights load into our modules with `load_state_dict(strict=True)` after only stripping a key prefix, and our outputs are checked numerically against HF's. Divergences are noted in [`docs/implementation_notes.md`](docs/implementation_notes.md).

## Team

| Name | Focus area |
|---|---|
| Person A | Backbone + Hybrid Encoder (AIFI + CCFM) |
| Person B | Decoder + Query Selection + Matcher + Losses |
| Shared | Box ops, post-processing, data pipeline, COCO eval |

## Why RT-DETR

RT-DETR removes the need for NMS post-processing (unlike YOLO-family detectors) while matching or beating YOLO's real-time speed/accuracy tradeoff, by:
1. An **efficient hybrid encoder** — decouples intra-scale interaction (transformer) from cross-scale fusion (CNN-based), instead of naive multi-scale deformable attention.
2. **Uncertainty-minimal query selection** — selects encoder features for decoder queries using both classification and localization scores (IoU-aware), not just objectness.
3. Support for **flexible speed tuning** by adjusting decoder layers without retraining.

## Architecture Status

| Component | Status | Notes |
|---|---|---|
| Backbone (ResNet-vd R18 / R50) | ✅ Done | HF weights, C3–C5 bit-exact vs HF |
| AIFI (intra-scale transformer) | ✅ Done | HF parity ~2e-6 |
| CCFM / CCFF (cross-scale fusion) | ✅ Done | HF parity exact |
| Hybrid encoder (proj + AIFI + CCFM) | ✅ Done | image → P3/N4/N5 vs HF ≤ 4.3e-6 |
| Query selection (IoU-aware) | 🟨 Skeleton | `src/decoder/rtdetr_transformer.py` (Person B) |
| Decoder + auxiliary heads | 🟨 Skeleton | `src/decoder/{deformable_attention,decoder,denoising}.py` (Person B) |
| Hungarian matcher | 🟨 Skeleton | `src/matcher/hungarian.py` (Person B) |
| Loss (VFL + L1 + GIoU) | 🟨 Skeleton | `src/losses/rtdetr_loss.py` (Person B) |
| Post-processing | 🟨 Skeleton | `src/postprocess.py` (shared) |
| COCO data pipeline | 🟨 Skeleton | `src/data/` (shared) |
| COCO eval (mAP) | 🟨 Skeleton | `src/eval/coco_eval.py`, `scripts/eval.py` (shared) |
| Training loop | ⬜ Not planned | goal is pretrained-weight parity, not training |
| Speed benchmark (FPS/latency) | ⬜ Optional | |

Use ⬜ Not started / 🟨 Skeleton or in progress / ✅ Done. Skeleton files contain signatures, shapes, hints and self-check numbers; they raise `NotImplementedError` until filled in.

## Correctness: parity with Hugging Face

Same input through our module and HF's, both in `eval()` with the PekingU weights. Max abs difference:

| Check | R18 | R50 |
|---|---|---|
| Backbone C3 / C4 / C5 | 0.0 | 0.0 |
| Hybrid encoder P3 / N4 / N5 (same C3–C5) | ≤ 4e-6 | ≤ 1.1e-5 |
| image → our backbone → our encoder vs HF full forward | ≤ 3e-6 | ≤ 4.3e-6 |
| our P3–P5 into HF's decoder: scores / boxes | 2.6e-6 / 6.4e-6 | 5.8e-7 / 7.0e-6 |

Tolerance is 1e-4. See [`notebooks/03_hf_parity.ipynb`](notebooks/03_hf_parity.ipynb) and `pytest -m slow`.

| Model | Params backbone + encoder (ours) | mAP (paper) | mAP (ours) |
|---|---|---|---|
| RT-DETR-R18 | 16.16 M | 46.5 | — (needs `scripts/eval.py`) |
| RT-DETR-R50 | 35.42 M | 53.1 | — |

### Using the pretrained weights

```python
from transformers import RTDetrForObjectDetection
from src.backbone import PResNet
from src.encoder import HybridEncoder
from src.utils.hf_weights import HF_CONFIGS, backbone_state_dict, encoder_state_dict

sd = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd").state_dict()
backbone = PResNet(**HF_CONFIGS[18]["backbone"]).eval()
backbone.load_state_dict(backbone_state_dict(sd), strict=True)
encoder = HybridEncoder(**HF_CONFIGS[18]["encoder"]).eval()
encoder.load_state_dict(encoder_state_dict(sd), strict=True)

p3, n4, n5 = encoder(backbone(images))   # images: [B, 3, 640, 640] in [0, 1], no mean/std
```

## Setup

```bash
git clone https://github.com/<org>/rt-detr-from-scratch.git
cd rt-detr-from-scratch
conda env create -f environment.yml
conda activate rtdetr
# or: pip install -r requirements.txt
```

## Data

We use COCO 2017 (val2017 is enough for evaluation). Expected layout:

```
data/coco/
├── annotations/
│   └── instances_val2017.json
└── val2017/
```

Download `val2017.zip` and `annotations_trainval2017.zip` from https://cocodataset.org/#download and unzip them into `data/coco/`. Don't commit data or weights.

## Evaluation

```bash
python scripts/eval.py --depth 18 --coco data/coco --mode ours   # skeleton for now, see the file's hints
```

`--mode hf` evaluates HF's model; `--mode ours` runs our backbone + encoder with HF's decoder (`encoder_outputs=`); `--mode full` will use Person B's decoder.

## Repo Structure

```
src/
├── backbone/     # PResNet (ResNet-vd) — done
├── encoder/      # position embedding, AIFI, CCFM, HybridEncoder — done
├── decoder/      # query selection, deformable attention, decoder, denoising — skeleton (Person B)
├── matcher/      # Hungarian matcher — skeleton (Person B)
├── losses/       # VFL + L1 + GIoU — skeleton (Person B)
├── data/         # COCO dataset + transforms — skeleton (shared)
├── eval/         # COCO mAP — skeleton (shared)
├── utils/        # hf_weights (HF key mapping) — done; box_ops — skeleton (shared)
└── postprocess.py  # logits/boxes → detections — skeleton (shared)
scripts/eval.py   # COCO evaluation CLI — skeleton
notebooks/        # 03_hf_parity.ipynb: HF parity proof
docs/             # implementation notes, HF key dumps
tests/            # pytest (fast) + -m slow (HF parity, downloads weights)
```

## Development workflow

- `main`: stable, always mergeable
- `encoder`: backbone + hybrid encoder work (Person A); merges into `main`
- No CI yet (added at the end). Run `pytest tests/ -m "not slow"` and `ruff check src/ tests/` locally; `-m slow` runs the HF parity tests (downloads weights)
- PRs require: shape checks with dummy input, a note on which paper section it implements, and (where feasible) a unit test or HF parity check

## References

- Paper: [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- Official code: https://github.com/lyuwenyu/RT-DETR
- Deformable DETR (prior work, useful background): https://arxiv.org/abs/2010.04159

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Chosen for compatibility with the official RT-DETR repo's license.
