"""COCO val2017 mAP for RT-DETR.  [SHARED — skeleton]

Usage (once filled in):
    python scripts/eval.py --depth 18 --coco data/coco --mode hf      # HF model end to end (reference AP)
    python scripts/eval.py --depth 18 --coco data/coco --mode ours    # our PResNet + HybridEncoder → HF decoder
    python scripts/eval.py --depth 18 --coco data/coco --mode ours --limit 200   # quick subset

Modes:
    hf    — `RTDetrForObjectDetection` alone. Gives the reference AP (R18 ≈ 0.465, R50 ≈ 0.531).
    ours  — our backbone + encoder make [P3, N4, N5]; HF runs only query selection + decoder:
                feats = encoder(backbone(pixel_values))
                out = hf(pixel_values=pixel_values, encoder_outputs=(feats,))
            Same AP as `hf` = our half of the paper is correct on real data (Task 20 stretch).
    full  — later, when Person B's `RTDetrTransformer` exists: no HF at all.
"""

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, default=18, choices=[18, 50])
    p.add_argument("--coco", default="data/coco", help="folder with annotations/ and val2017/")
    p.add_argument("--mode", default="hf", choices=["hf", "ours", "full"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="only the first N images (quick check)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Hints:
    # 1. Dataset: CocoDetection(coco/val2017, coco/annotations/instances_val2017.json, build_transforms(train=False))
    #    --limit → torch.utils.data.Subset(ds, range(limit)). DataLoader(..., collate_fn=collate_fn, num_workers=4).
    # 2. Model: hf = RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{args.depth}vd").eval().to(device)
    #    mode "ours": build PResNet + HybridEncoder from HF_CONFIGS[depth], strict-load with backbone_state_dict /
    #    encoder_state_dict(hf.state_dict()) — exactly like notebooks/03_hf_parity.ipynb §5–6.
    # 3. Loop under torch.inference_mode():
    #       out = ...                                           (see module docstring per mode)
    #       sizes = torch.stack([t["orig_size"] for t in targets])
    #       results = post_process(out.logits, out.pred_boxes, sizes, threshold=0.0)   ← threshold 0 for mAP
    #       detections += to_coco_detections(results, [t["image_id"] for t in targets], ds.cat_ids)
    # 4. stats = evaluate_coco(ds.coco, detections, image_ids=<ids you ran>); print(f"AP {stats[0]:.3f} AP50 {stats[1]:.3f}")
    # 5. Sanity before a 5000-image run: --limit 50 for both modes should give the same AP.
    raise NotImplementedError(f"scripts/eval.py is a skeleton (args: {args})")


if __name__ == "__main__":
    main()
