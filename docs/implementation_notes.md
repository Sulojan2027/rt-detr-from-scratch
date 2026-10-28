# Implementation Notes

Log of deliberate divergences from the paper and/or the official repo, and any tricky implementation details worth remembering. Add an entry per component as you build it.

## Template

### `<component name>`

- **Paper reference:** Section X.X / Figure X
- **Official repo reference:** `path/to/file.py` in lyuwenyu/RT-DETR (or N/A)
- **What we did:**
- **Why it differs (if it does):**
- **Gotchas / things that took a while to debug:**

---

## Backbone

### `PResNet` (ResNet-vd) — `src/backbone/presnet.py`, `src/backbone/common.py`

- **Paper reference:** Section 4 / Table 1 (R18 / R50 backbones)
- **Official repo reference:** `rtdetr_pytorch/src/nn/backbone/presnet.py`; HF `transformers/models/rt_detr/modeling_rt_detr_resnet.py`
- **What we did:** a simplified ResNet-vd whose module names mirror Hugging Face, so `PekingU/rtdetr_r18vd` / `rtdetr_r50vd` backbone weights load with `load_state_dict(strict=True)` after only stripping the prefix `model.backbone.model.`. No key remapping.
- **Parity:** same random `[1, 3, 640, 640]` input → C3/C4/C5 match `hf.model.backbone.model(x).feature_maps` **bit-exactly** (max abs diff 0.0) for R18 and R50 (`notebooks/03_hf_parity.ipynb`, `tests/test_backbone.py::test_hf_parity`).
- **Reference version:** keys dumped with `transformers 5.15.0.dev0` (source checked at v5.10.2). Dumps: `docs/hf_keys_r18.txt`, `docs/hf_keys_r50.txt`. Names can change between versions — re-dump if a load fails.

| Ours | HF | Official |
|---|---|---|
| `ConvLayer.convolution / normalization / activation` | `RTDetrResNetConvLayer` (same) | `ConvNormLayer.conv / norm / act` |
| `ShortCut.convolution / normalization` | `RTDetrResNetShortCut` | `short` (`ConvNormLayer`) |
| vd shortcut `shortcut.0` (AvgPool) + `shortcut.1` (ShortCut) | same (unnamed Sequential) | `short.pool` + `short.conv` |
| block `layer.{0,1,2}`, `shortcut`, `activation` | same | `branch2a/2b/2c`, `short`, `act` |
| `Blocks.layers` | `RTDetrResNetStage.layers` | `Blocks.blocks` |
| `embedder.embedder.{0,1,2}`, `embedder.pooler` | `RTDetrResNetEmbeddings` | `conv1.conv1_{1,2,3}`, `pool` |
| `encoder.stages` | `RTDetrResNetEncoder.stages` | `res_layers` |

- **Why it differs:**
  1. HF names, not official names → HF checkpoints load with one line and no remapping.
  2. `Blocks` takes an explicit `stride` instead of the official `stage_num`.
  3. `embedder` / `encoder` are plain `nn.Sequential` / `nn.ModuleDict` containers instead of separate HF classes — same keys, less code.
  4. `ConvNormLayer` (`conv` / `norm` / `act`) is kept for the hybrid encoder, because HF's encoder uses those names; the backbone uses `ConvLayer`.
  5. `get_activation` never uses `inplace=True`; arguments are validated early (the references don't).
- **Gotchas:**
  - The first block of **every** stage projects, including R18/R34 stage 0 (64→64, stride 1) — matches official and HF (`should_apply_shortcut=True` for basic layers).
  - HF backbone norms are frozen BN (no `num_batches_tracked` in the checkpoint), so use `freeze_norm=True` for loading/inference. `FrozenBatchNorm2d` also drops `num_batches_tracked` on load.
  - `freeze_batch_norm2d` copies values, so freezing works before or after loading.
  - Parameter counts with `freeze_norm=False`: R18 11.20 M, R34 21.31 M, R50 23.53 M, R101 42.52 M.

## Hybrid Encoder (AIFI + CCFM)

*(fill in once implemented)*

## Query Selection

*(fill in once implemented)*

## Decoder

*(fill in once implemented)*

## Matcher

*(fill in once implemented)*

## Losses

*(fill in once implemented)*

## Training / data pipeline

*(fill in once implemented)*
