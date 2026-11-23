"""COCO val2017 mAP for RT-DETR.  [SHARED]

Usage:
    python scripts/eval.py --depth 18 --mode hf                       # HF model end to end (reference AP)
    python scripts/eval.py --depth 18 --mode ours                     # our PResNet + HybridEncoder → HF decoder
    python scripts/eval.py --depth 18 --mode ours --limit 200         # quick subset
    python scripts/eval.py --depth 50 --mode ours --coco /data/coco --out results/r50_ours.json   # on the 3090

Modes:
    hf    — `RTDetrForObjectDetection` alone. Gives the reference AP (paper: R18 46.5, R50 53.1).
    ours  — OUR backbone + encoder make [P3, N4, N5]; HF runs only query selection + decoder.
            HF's own backbone and encoder are never called (pixel_values=None, verified with forward hooks).
            Same AP as `hf` = our half of the paper is correct on real data.
    full  — OUR backbone + encoder + query selection + decoder (`RTDetrTransformer`). HF is only used to
            download the pretrained weights; none of its modules run. Same AP as `hf` = the whole model is correct.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import RTDetrForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `import src` works
from src.backbone.presnet import PResNet  # noqa: E402
from src.data.coco import CocoDetection, collate_fn  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.decoder.rtdetr_transformer import RTDetrTransformer  # noqa: E402
from src.encoder.hybrid_encoder import HybridEncoder  # noqa: E402
from src.eval.coco_eval import evaluate_coco, to_coco_detections  # noqa: E402
from src.postprocess import post_process  # noqa: E402
from src.utils.hf_weights import HF_CONFIGS, backbone_state_dict, decoder_state_dict, encoder_state_dict  # noqa: E402

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, default=18, choices=[18, 50])
    p.add_argument("--coco", default="data/coco", help="folder with annotations/ and val2017/")
    p.add_argument("--mode", default="hf", choices=["hf", "ours", "full"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="only the first N images (quick check)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="save the metrics as JSON here (e.g. results/r18_ours.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    dataset = CocoDetection(
        img_folder=Path(args.coco, "val2017"),
        ann_file=Path(args.coco, "annotations/instances_val2017.json"),
        transforms=build_transforms(),
    )
    data = Subset(dataset, range(args.limit)) if args.limit is not None else dataset   # dataset keeps .coco/.cat_ids
    dataloader = DataLoader(data, args.batch_size, collate_fn=collate_fn, num_workers=args.workers)

    hf_model = RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{args.depth}vd").eval().to(device)

    # Every mode gets a `forward(images) -> (logits [B, 300, 80], pred_boxes [B, 300, 4])`.
    if args.mode == "hf":
        def forward(x):
            out = hf_model(pixel_values=x)
            return out.logits, out.pred_boxes
    else:
        sd = hf_model.state_dict()
        backbone = PResNet(**HF_CONFIGS[args.depth]["backbone"])                 # our backbone
        backbone.load_state_dict(backbone_state_dict(sd), strict=True)
        backbone.eval().to(device)
        encoder = HybridEncoder(**HF_CONFIGS[args.depth]["encoder"])             # our encoder
        encoder.load_state_dict(encoder_state_dict(sd), strict=True)
        encoder.eval().to(device)

        if args.mode == "ours":
            def forward(x):
                feats = encoder(backbone(x))
                out = hf_model(pixel_values=None, inputs_embeds=feats[0], encoder_outputs=(feats,))
                return out.logits, out.pred_boxes
        else:
            transformer = RTDetrTransformer(num_decoder_layers=hf_model.config.decoder_layers)   # our decoder
            transformer.load_state_dict(decoder_state_dict(sd), strict=True)
            transformer.eval().to(device)
            del hf_model, sd                                                     # weights are copied, HF no longer needed

            def forward(x):
                out = transformer(encoder(backbone(x)))
                return out["logits"], out["pred_boxes"]

    detections, ran_ids = [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc=f"R{args.depth} {args.mode}"):
            logits, pred_boxes = forward(images.to(device))
            sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = post_process(logits, pred_boxes, sizes, threshold=0.0)   # threshold 0 for mAP
            ids = [t["image_id"] for t in targets]
            detections += to_coco_detections(results, ids, dataset.cat_ids)
            ran_ids += ids
    elapsed = time.perf_counter() - t0

    stats = evaluate_coco(dataset.coco, detections, image_ids=ran_ids)
    metrics = dict(zip(STAT_NAMES, (round(float(s), 4) for s in stats), strict=True))
    print(f"\nRT-DETR-R{args.depth} [{args.mode}] on {len(ran_ids)} val2017 images "
          f"({len(ran_ids) / elapsed:.1f} img/s incl. data loading, {device}):")
    print(f"AP: {stats[0]:.3f}, AP50: {stats[1]:.3f}, AP75: {stats[2]:.3f}")
    print(f"APs: {stats[3]:.3f}, APm: {stats[4]:.3f}, APl: {stats[5]:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": f"rtdetr_r{args.depth}vd",
            "mode": args.mode,
            "images": len(ran_ids),
            "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "throughput_img_s": round(len(ran_ids) / elapsed, 2),
            **metrics,
        }
        out_path.write_text(json.dumps(record, indent=2))
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
