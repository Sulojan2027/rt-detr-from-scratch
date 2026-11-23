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

`src/decoder/rtdetr_transformer.py` (`generate_anchors`, `RTDetrTransformer`); weight loader `src/utils/hf_weights.py::decoder_state_dict`.

- **Paper reference:** §4.2 (uncertainty-minimal query selection), Fig. 2
- **Official repo reference:** `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py` (`RTDETRTransformer`); HF `RTDetrModel.__init__` / `RTDetrModel.forward` after the encoder, and `RTDetrModel._cached_generate_anchors`
- **What we did:** `[P3, N4, N5]` → `decoder_input_proj` (1×1 conv + BN) → flatten to one `[B, S, 256]` token sequence (S = 8400 at 640²) → one anchor box per token → `enc_output` + `enc_score_head` + `enc_bbox_head` score and box every token → top-300 tokens by their best class score become the decoder's starting boxes (`ref_unact`) and content queries (`target`) → decoder.

```
anchors, valid = generate_anchors(spatial_shapes)           # (cx, cy, w, h) = (cell centre, 0.05 · 2^level), logit space
out_mem  = enc_output(valid * memory)                       # invalid anchors are zeroed BEFORE the heads
cls      = enc_score_head(out_mem)                          # [B, S, 80]
coord    = enc_bbox_head(out_mem) + anchors                 # [B, S, 4]  logit space
top      = cls.max(-1).values.topk(300).indices             # [B, 300]
ref, target = coord[top], out_mem[top].detach()
logits, boxes = decoder(target, ref.detach(), memory, spatial_shapes)    # memory is the UNmasked sequence
```

- **Parity** (random image, our decoder fed HF's own P3–P5, PekingU weights, max abs diff; `tests/test_rtdetr_transformer.py`, `pytest -m slow`):

| Check | R18 | R50 |
|---|---|---|
| query selection `enc_outputs_class` | 0.0 | 0.0 |
| final scores (sigmoid), 640² / 320×416 | 6.9e-7 / 1.9e-6 | 7.0e-7 / 5.1e-7 |
| final boxes, 640² / 320×416 | 8.6e-6 / 2.7e-6 | 1.0e-6 / 2.6e-6 |
| image → our backbone + encoder + decoder, vs HF full forward, on a real COCO image: scores / boxes (px) | 2.4e-7 / 6.1e-5 | 1.2e-7 / 4.8e-5 |

- **Parameters** (query selection + decoder): R18 4,019,376 (3 layers), R50 7,466,508 (6 layers). Whole model: R18 20.17 M, R50 42.89 M.
- **Weight loading:** `decoder_state_dict(hf_sd)` strips `model.` from `decoder_input_proj`, `denoising_class_embed`, `enc_output`, `enc_score_head`, `enc_bbox_head` and `decoder`. Keys: R18 137, R50 239. With the backbone and encoder loaders, all 526 (R18) / 764 (R50) checkpoint keys are loaded exactly once (checked by tensor identity; none missed, none twice).
- **Why it differs from HF:**
  1. `forward` returns a plain dict (`logits`, `pred_boxes`, `aux_outputs`, `enc_outputs_class`, and `dn_outputs` / `dn_meta` while training with targets) instead of `RTDetrModelOutput`. The last entry of `aux_outputs` is the encoder top-k proposal, as in HF's loss input.
  2. No `learn_initial_query`, `anchor_image_size` anchor cache or extra stride-2 input projections: the PekingU configs don't use them. Anchors are recomputed on every forward (cheap).
  3. Dropout and output-recording hooks (hidden states, attentions) are omitted.
- **Gotchas:**
  - **Two different memories.** The score / box heads see `valid_mask * memory`; the decoder cross-attends to the **unmasked** memory. Using the masked one for both runs fine and quietly breaks parity.
  - **Invalid anchors are `finfo.max`, not `inf`** (same as HF, so adding the box head output stays finite). The top-k ranks tokens by class score only, so an invalid token is not excluded by this; its box is simply saturated. Matching HF here is what keeps parity exact.
  - **`target` is detached and `ref` is detached** before the decoder. The "IoU-aware" part of the paper happens in training (VFL teaches `enc_score_head` to score ≈ IoU), so inference is a plain top-k.
  - **`denoising_class_embed` has 81 rows** (80 classes + padding index 80), so the strict load fails if it has 80.
  - In tests, HF's `RTDetrConfig()` defaults to `num_labels=2` and 6 decoder layers; build it with `num_labels=80, decoder_layers=3` for R18-shaped checks.

## Decoder

`src/decoder/deformable_attention.py`, `decoder.py`, `denoising.py`.

- **Paper reference:** §4.3 / Fig. 2 (decoder with auxiliary heads); deformable attention from Deformable DETR (arXiv:2010.04159 §4.1); denoising from DINO (arXiv:2203.03605 §3.3)
- **Official repo reference:** `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py` (`MSDeformableAttention`, `TransformerDecoderLayer`, `TransformerDecoder`), `denoising.py`; HF `RTDetrMultiscaleDeformableAttention`, `MultiScaleDeformableAttention`, `RTDetrDecoderLayer`, `RTDetrDecoder`, `RTDetrMLPPredictionHead`, `get_contrastive_denoising_training_group`
- **What we did:**

| Ours | HF | Notes |
|---|---|---|
| `multi_scale_deformable_attn` | `MultiScaleDeformableAttention` (PyTorch path) | `grid_sample` per level, heads folded into the batch |
| `MultiscaleDeformableAttention` (`sampling_offsets`, `attention_weights`, `value_proj`, `output_proj`) | `RTDetrMultiscaleDeformableAttention` | 205,600 params |
| `DecoderLayer` (`self_attn`, `encoder_attn`, `mlp`, 3 layer norms) | `RTDetrDecoderLayer` | post-norm, relu FFN; 995,872 params |
| `Decoder` (`layers`, `query_pos_head`, `class_embed`, `bbox_embed`) | `RTDetrDecoder` + heads attached by `RTDetrForObjectDetection` | R18 3 layers (3,581,020 params), R50 6 (7,028,152) |
| `MLPPredictionHead` | `RTDetrMLPPredictionHead` | `layers` ModuleList, ReLU between, none after the last |

  Per layer: `pos = query_pos_head(ref)` → self-attention (pos on Q, K only) → deformable cross-attention over the memory (pos on the offset / weight prediction) → FFN → `new_ref = sigmoid(bbox_embed[i](x) + inverse_sigmoid(ref))` → `logits_i = class_embed[i](x)`. The last layer is the prediction; earlier ones feed the auxiliary losses.
- **Reuse:** `SelfAttention` and `MLP` come from `src/encoder/aifi.py` (same HF names). **`SelfAttention` gained an optional additive `attn_mask`** (default `None`, so the encoder is unchanged) for denoising.
- **Parity:** `tests/test_deformable_attention.py` and `tests/test_decoder.py` load random HF weights into our modules with `strict=True` (no key remapping) and compare: core function ≤ 1e-6, attention module ≤ 1e-5, decoder (3 and 6 layers) ≤ 1e-4; end-to-end numbers are in the Query Selection table.
- **Why it differs:** pure-PyTorch `grid_sample` only (no CUDA kernel, no `level_start_index` / `im2col_step`), no dropout, no recorded intermediate hidden states, `Decoder.forward` returns `(logits, boxes)` stacked over layers `[B, L, Q, ·]`.
- **Gotchas:**
  - **`grid_sample` convention.** `align_corners=False` and the `2·loc − 1` mapping must match exactly, otherwise every box shifts by half a pixel. A hand-checked test (sampling a pixel centre returns that pixel) guards it.
  - **One softmax over levels × points together,** not one per level.
  - **Offsets are scaled by half the reference box size** (`offsets / n_points · wh · 0.5`), so points land inside / around the box.
  - **Keep the gradient on the output box, feed a detached one forward.** `all_boxes.append(new_ref)` but `ref = new_ref.detach()`. Swapping them does not break parity, only training.
  - **`reference_points` shape.** The deformable attention takes `[B, Q, 1, 4]` (a level axis of size 1 that broadcasts); the decoder passes `ref.unsqueeze(2)`.
- **Denoising (`denoising.py`, training only):**
  - A group is `[positives | negatives]`, each a noisy copy of every GT padded to `max_gt`; `num_groups = max(100 // max_gt, 1)`. 3 GTs → 33 groups → 198 denoising queries.
  - Label noise swaps a real label for a random class with probability `label_noise_ratio · 0.5` (padding is never touched). Box noise shifts the corners by a random fraction of half the box size: positives 0–1×, negatives 1–2×, clipped to [0, 1], then `inverse_sigmoid`.
  - The attention mask is `[N_dn + Q]²`, 0 = allowed, −inf = blocked: normal queries cannot see any denoising query (they would read the answer), and each group cannot see the other groups.
  - `dn_positive_idx` (per image, `n_gt · num_groups` entries) is the fixed assignment the loss uses instead of the matcher.
  - **Same seed → bit-identical to HF.** We call the same random ops in the same order (`rand_like`, `randint_like`, `randint_like`, `rand_like`) and keep the labels `int32` like HF. `tests/test_denoising.py` checks equality with HF under a shared seed, and `tests/test_rtdetr_transformer.py` checks the whole training forward (normal and denoising outputs) against HF.
  - `RTDetrTransformer.forward` builds the group only when `self.training and targets is not None`, then splits the denoising queries off again, so `logits` / `pred_boxes` look the same as at inference and the denoising part is returned as `dn_outputs` (one entry per decoder layer, including the last).

## Matcher

`src/matcher/hungarian.py`

- **Paper reference:** §4 (set prediction as in DETR, arXiv:2005.12872 §3.1)
- **Official repo reference:** `rtdetr_pytorch/src/zoo/rtdetr/matcher.py`; HF `RTDetrHungarianMatcher` in `transformers/loss/loss_rt_detr.py`
- **What we did:** one cost matrix for the whole batch (rows = all `B·Q` queries, columns = every GT of every image), `cost = 2·class + 5·L1 + 2·(−GIoU)`, then `scipy.optimize.linear_sum_assignment` on each image's own block of columns. Returns `(query_idx, gt_idx)` per image, `K = min(Q, N)`; an image without GT gives empty tensors.
  - Class cost is focal-style on sigmoid probabilities: `α(1−p)^γ·−log p − (1−α)p^γ·−log(1−p)` with the **matcher's own α = 0.25, γ = 2** (the VFL loss uses α = 0.75).
  - L1 is on `(cx, cy, w, h)` (`torch.cdist(p=1)`); GIoU on the corner boxes.
- **Parity:** identical index lists to HF's matcher on random batches with 300 queries (`tests/test_matcher.py`).
- **Why it differs:** only the focal branch is implemented (the HF softmax branch is not used by RT-DETR).
- **Gotchas:**
  - **Costs that are `nan` / `inf` are replaced by `finfo.max`** so the solver cannot crash; such a pair is only chosen when nothing better exists. This covers e.g. nan class scores. A nan *box* is rejected earlier: `generalized_box_iou` raises on any box that fails `x2 ≥ x1`, and nan fails that test (HF behaves the same).
  - The split by `sizes` matters: an image may only be matched to its own GTs, and the returned `gt_idx` is local to the image.
  - Matching runs under `@torch.no_grad()` on CPU (`scipy`), so it does not slow down the GPU graph but does synchronise.

## Losses

`src/losses/rtdetr_loss.py`

- **Paper reference:** §4.2 (VFL makes the class score IoU-aware, which is what the query selection relies on)
- **Official repo reference:** `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py`; HF `RTDetrLoss` (`loss_labels_vfl`, `loss_boxes`, `get_cdn_matched_indices`)
- **What we did:** `RTDetrLoss(matcher)` returns a dict of **already weighted** losses; the training loss is `sum(losses.values())`.
  - `loss_vfl` (weight 1): BCE where the target of the matched class is the IoU of the matched pair (detached) instead of 1, with weight `α·σ(logit)^γ` on negatives (α = 0.75, γ = 2) and the IoU on positives; `mean(over classes).sum() · Q / num_boxes`.
  - `loss_bbox` (weight 5): L1 on `(cx, cy, w, h)`, summed over matched pairs / `num_boxes`. `loss_giou` (weight 2): `1 − GIoU`, same normalisation.
  - Final layer: matched once. Each entry of `outputs["aux_outputs"]` (earlier decoder layers, then the encoder top-k proposals) is **re-matched** on its own predictions → keys `…_aux_{i}`.
  - Denoising (`outputs["dn_outputs"]`, one per layer): fixed assignment from `get_cdn_matched_indices`, `num_boxes · dn_num_group` → keys `…_dn_{i}`.
  - `num_boxes = max(total GTs in the batch, 1)`.
- **Parity:** same keys and values within 1e-5 of HF's `RTDetrLoss` for random outputs with 3 auxiliary layers, an image with no objects, and 2 denoising layers (`tests/test_loss.py`).
- **Why it differs:** input keys are `aux_outputs` / `dn_outputs` / `dn_meta` (HF: `auxiliary_outputs` / `dn_auxiliary_outputs` / `denoising_meta_values`; the tests convert). Only VFL + boxes are implemented, not HF's other variants (CE, BCE, focal, masks, cardinality). Splitting the denoising queries off the main output is done in `RTDetrTransformer.forward`, not in the loss (HF does it in `RTDetrForObjectDetectionLoss`). No multi-GPU all-reduce of `num_boxes`.
- **Gotchas:**
  - The IoU inside VFL is a **target**: compute it from `pred_boxes.detach()`, or the class loss pushes the boxes.
  - The **no-object** class exists only as an extra one-hot column that is dropped again, so unmatched queries are all-zero targets (not a separate class).
  - The matcher's α (0.25) and the VFL's α (0.75) are different numbers on purpose.
  - Images without GT must still work: `num_boxes` is clamped to 1, the box losses are 0 and VFL only teaches "no object".
  - There is no training loop in this repo; the matcher, loss and denoising are checked against HF and run end to end (`test_training_with_denoising_output_structure_and_backward`: transformer → matcher → loss → backward).

## Training / data pipeline / eval

Skeletons (shared): `src/utils/box_ops.py`, `src/postprocess.py`, `src/data/{coco,transforms}.py`, `src/eval/coco_eval.py`, `scripts/eval.py`.
Preprocessing must match the PekingU image processor: torchvision **v2** `resize` of the **uint8** image to 640×640 (bilinear, antialias, no aspect keep), then `to_dtype(float32, scale=True)`, **no mean/std normalisation**. This is bit-exact with HF; PIL or v1 resize is off by 1/255 on ~0.14 % of pixels. Labels are contiguous 0..79; COCO category ids (1..90 with gaps) are mapped via `sorted(coco.getCatIds())`.

### `box_iou` / `generalized_box_iou` — `src/utils/box_ops.py`

- **What we did:** pairwise `[N, 4] × [M, 4]` xyxy → IoU (+ union) and GIoU in `[-1, 1]`, used by the matcher and the loss. Matches torchvision within 1e-6 and HF's versions exactly (`tests/test_box_ops.py`).
- **Gotchas:** the overlap size is clamped to 0 (no overlap must not give a negative area). `generalized_box_iou` raises if any box has `x2 < x1` or `y2 < y1`, which catches `(cx, cy, w, h)` passed by mistake; a nan box also fails that test, so it raises too.

### `scripts/eval.py --mode full`

- **What we did:** our backbone + encoder + `RTDetrTransformer`, no HF module runs (HF only downloads the PekingU weights and is dropped after the state dicts are loaded). All three modes now return `(logits, pred_boxes)` from their `forward`, so they share one evaluation loop.
- **Checked:** on a 2-image mini COCO folder (the COCO "cats" image and its mirror, with HF's confident detections as stand-in annotations) `hf`, `ours` and `full` give identical metrics; on the cats image the post-processed detections of our full model match HF's (same labels, score diff ≤ 2.4e-7, box diff ≤ 6.1e-5 px for R18 and R50). **Real val2017 mAP has not been run yet** (needs the dataset; expect the same number as `--mode hf`, paper: R18 46.5, R50 53.1).
- **Gotcha:** the final `print` used `→`, which crashes on Windows consoles (cp1252) after the metrics are computed; it is plain ASCII now.
