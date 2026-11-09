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

`src/encoder/position_encoding.py`, `aifi.py`, `ccfm.py`, `hybrid_encoder.py`

- **Paper reference:** §4.1, Fig. 3 (efficient hybrid encoder), Fig. 5 (CCFF fusion block), Table 1 / §5 (R18 / R50 configs)
- **Official repo reference:** `rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py`; HF `modeling_rt_detr.py` (`RTDetrHybridEncoder`, `RTDetrAIFILayer`, `RTDetrCSPRepLayer`, `RTDetrRepVggBlock`, `build_2d_sinusoidal_position_embedding`)
- **What we did:** a simplified hybrid encoder with HF module names. `[C3, C4, C5]` (R18: 128/256/512 ch, R50: 512/1024/2048) → 1×1 projection to 256 → AIFI on S5 only → CCFF (top-down FPN + bottom-up PAN) → `[P3, N4, N5]`, all 256 ch, strides 8/16/32.

```
S3, S4, S5 = encoder_input_proj[0..2](C3, C4, C5)        # 1×1 conv + BN → 256 ch
F5 = aifi[0](S5)                                          # attention on the 20×20 map only
Y5 = lateral_convs[0](F5);  X4 = fpn_blocks[0](cat[up2(Y5), S4])     # top-down (FPN)
Y4 = lateral_convs[1](X4);  P3 = fpn_blocks[1](cat[up2(Y4), S3])
N4 = pan_blocks[0](cat[downsample_convs[0](P3), Y4])                 # bottom-up (PAN), re-uses Y4, Y5
N5 = pan_blocks[1](cat[downsample_convs[1](N4), Y5])
return [P3, N4, N5]
```

| Ours | HF | Official | Notes |
|---|---|---|---|
| `build_2d_sinusoidal_position_embedding`, `SinePositionEmbedding` | same | `build_2d_sincos_position_embedding` | no weights |
| `AIFILayer.position_embedding`, `.layers` | `RTDetrAIFILayer` | `encoder.{i}` (`TransformerEncoder`) | |
| `EncoderLayer.self_attn / self_attn_layer_norm / mlp / final_layer_norm` | `RTDetrEncoderLayer` | `self_attn / norm1 / linear1, linear2 / norm2` | |
| `SelfAttention.q_proj / k_proj / v_proj / o_proj` | `RTDetrSelfAttention` | `nn.MultiheadAttention` (`in_proj_weight`) | that's why we can't use `nn.MultiheadAttention` |
| `MLP.fc1 / fc2 / activation_fn` | `RTDetrMLP` | `linear1 / linear2` | |
| `RepVggBlock.conv1 (3×3) / conv2 (1×1) / act` | `RTDetrRepVggBlock` (`activation`) | same + deploy `conv` | paper Fig. 5 "RepBlock" |
| `CSPRepLayer.conv1 / conv2 / bottlenecks / conv3` | `RTDetrCSPRepLayer` | same | paper Fig. 5 "Fusion" |
| `CCFM.lateral_convs / fpn_blocks / downsample_convs / pan_blocks` | `RTDetrHybridEncoder` (same) | same | paper v1 "CCFM", later "CCFF" |
| `HybridEncoder.encoder_input_proj` | `RTDetrModel.encoder_input_proj` (**outside** the encoder) | `HybridEncoder.input_proj` | two HF prefixes |
| `HybridEncoder.aifi` | `RTDetrHybridEncoder.aifi` | `encoder` | one AIFI per `encode_proj_layers` entry (`[2]` = S5) |

- **Weight loading:** `src/utils/hf_weights.py`: `encoder_state_dict(hf_sd)` merges `model.encoder_input_proj.*` → `encoder_input_proj.*` (strip `model.` only) and `model.encoder.*` → strip the prefix. `HF_CONFIGS[depth]["encoder"]` holds the kwargs (`in_channels`, `expansion`). Keys: R18 274, R50 250.
- **Parity:**

| Check | R18 | R50 | Where |
|---|---|---|---|
| Position embedding vs HF | 0.0 | 0.0 | `test.ipynb` 13–15 |
| AIFI vs HF | ~2e-6 | ~2e-6 | `test.ipynb` 16–19 |
| RepVgg / CSPRepLayer / CCFM vs HF | 0.0 | 0.0 | `test.ipynb` 20–28 |
| HybridEncoder, same C3–C5 (640², 320×416) | ≤ 4e-6 | ≤ 1.1e-5 | `test.ipynb` 29–31, `03_hf_parity.ipynb` §4 |
| image → PResNet → HybridEncoder vs HF full forward | ≤ 3e-6 | ≤ 4.3e-6 | `03_hf_parity.ipynb` §5, `tests/test_encoder.py` |
| our P3–P5 → HF decoder: scores / boxes | 2.6e-6 / 6.4e-6 | 5.8e-7 / 7.0e-6 | `03_hf_parity.ipynb` §6 |

  The small non-zero AIFI diffs are float32 rounding in attention (different matmul order), not a logic difference. Everything without attention is bit-exact.
- **Parameters:** R18 4,965,120 (proj 230,912 + AIFI 789,760 + CCFM 3,944,448); R50 11,950,848 (proj 919,040 + AIFI 789,760 + CCFM 10,242,048). Backbone + encoder: R18 16,155,232, R50 35,424,864.
- **Why it differs:**
  1. `encoder_input_proj` lives **inside** our `HybridEncoder` (HF has it in `RTDetrModel`), so the encoder is one self-contained module: C3–C5 in, P3–P5 out. Cost: two prefixes when loading.
  2. `HybridEncoder` **inherits** `CCFM` and calls `super().forward()` after projection + AIFI. `CCFM(num_levels=len(in_channels))` builds `num_levels - 1` stages, like HF's `num_fpn_stages = len(in_channels) - 1`.
  3. `CCFM.forward` is **unrolled for 3 levels** (readable, matches the paper figure) instead of HF's loops. Other level counts raise in `HybridEncoder.forward`.
  4. Activation attribute is `act` (HF: `activation`). Activations have no weights, so keys are unaffected.
  5. No dropout modules (`dropout=0.0` in both configs), no attention / hidden-state outputs, no deploy-time RepVgg fusion, no position-embedding cache (HF uses `lru_cache`, official precomputes for `eval_size`; `eval_size` is `None` in the PekingU configs anyway).
- **Gotchas:**
  - **Position-embedding order.** HF: `meshgrid(arange(H), arange(W), indexing="ij")`, concat `[sin_h, cos_h, sin_w, cos_w]`, math in float64. The official repo uses `[sin_w, cos_w, sin_h, cos_h]` with a swapped grid. Both agree **only on square maps**, so always test a non-square input (320×416).
  - **Pos goes to Q and K only**, never V. `v_proj(x + pos)` still runs and gives plausible numbers, but parity fails.
  - **Head split needs a transpose.** `view(B, N, h, d).transpose(1, 2)` on the way in, `transpose(1, 2).reshape(B, N, D)` on the way out. Reshape without the transpose silently mixes heads.
  - **Post-norm** (`LayerNorm(x + sublayer(x))`), not pre-norm.
  - **`conv3` is `nn.Identity` when `hidden == ch_out`** (R50, expansion 1.0). R18 (0.5) has a real `conv3`. Key sets differ per depth.
  - **Encoder BNs are regular `BatchNorm2d`** (the checkpoint has `num_batches_tracked`), unlike the frozen backbone BNs. Don't freeze them, or strict loading fails.
  - **Inputs must be multiples of 32.** Otherwise `up2(Y5)` and S4 differ by one pixel and `torch.cat` fails.
  - Recurring bug while writing it: an `act` argument accepted but not passed down (3 times). `test_ccfm_act_passed_down` guards it.

## Query Selection

Skeleton: `src/decoder/rtdetr_transformer.py` (`generate_anchors`, `RTDetrTransformer`), Person B. HF keys under `model.` are listed in `docs/hf_keys_r{18,50}.txt` § decoder; loader stub `src/utils/hf_weights.py::decoder_state_dict`.

*(fill in once implemented)*

## Decoder

Skeletons: `src/decoder/deformable_attention.py`, `src/decoder/decoder.py`, `src/decoder/denoising.py`, Person B. Can reuse `src.encoder.aifi.SelfAttention` / `MLP` (same HF names); `SelfAttention` needs an optional `attn_mask` for denoising.

*(fill in once implemented)*

## Matcher

Skeleton: `src/matcher/hungarian.py`, Person B.

*(fill in once implemented)*

## Losses

Skeleton: `src/losses/rtdetr_loss.py`, Person B.

*(fill in once implemented)*

## Training / data pipeline / eval

Skeletons (shared): `src/utils/box_ops.py`, `src/postprocess.py`, `src/data/{coco,transforms}.py`, `src/eval/coco_eval.py`, `scripts/eval.py`.
Preprocessing must match the PekingU image processor: resize to 640×640 (no aspect keep, bilinear), divide by 255, **no mean/std normalisation**. Labels are contiguous 0..79; COCO category ids (1..90 with gaps) are mapped via `sorted(coco.getCatIds())`.

*(fill in once implemented)*
