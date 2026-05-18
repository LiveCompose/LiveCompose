import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image

from common import (
    ACTIONS,
    bbox_cxcywh_to_xyxy,
    box_iou_xyxy,
    box_state,
    canonical_box_xyxy,
    clamp_xywh,
    find_adacrop_root,
    load_records,
    load_teacher,
    render_full_image,
    render_crop,
    step_box,
    xywh_to_xyxy,
)


def parse_args():
    root = find_adacrop_root()
    parser = argparse.ArgumentParser(description="Evaluate teacher BBox head and BBox+RL rollout IoU.")
    parser.add_argument("--teacher-ckpt", type=Path, default=root.parent / "ppo_best_val_final_score.pth")
    parser.add_argument("--eval-json", type=Path, default=root / "data" / "splits" / "val_mixed.json")
    parser.add_argument("--output-json", type=Path, default=root / "distillation" / "teacher_eval.json")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-delta", type=float, default=0.05)
    parser.add_argument(
        "--teacher-action-selection",
        choices=["sample_logits", "argmax"],
        default="sample_logits",
        help="How the teacher chooses the next rollout action. sample_logits uses Categorical(logits=logits).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-cudnn", action="store_true", help="Work around occasional Windows/cuDNN stream errors.")
    parser.add_argument("--skip-errors", action="store_true", help="Skip images that fail during evaluation instead of aborting.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def best_gt_iou(box_xywh, gt_boxes, width, height, img_path):
    if not gt_boxes:
        return None
    pred = xywh_to_xyxy(box_xywh)
    return max(box_iou_xyxy(pred, canonical_box_xyxy(gt, width, height, img_path=img_path)) for gt in gt_boxes)


def best_gt_iou_xyxy(box_xyxy, gt_boxes, width, height, img_path):
    if not gt_boxes:
        return None
    return max(box_iou_xyxy(box_xyxy, canonical_box_xyxy(gt, width, height, img_path=img_path)) for gt in gt_boxes)


@torch.no_grad()
def predict_teacher_bbox(teacher, img: Image.Image, args, device):
    width, height = img.size
    img_t = render_full_image(img, args.img_size).unsqueeze(0).to(device)
    pred = teacher.backbone_forward(img_t).squeeze(0).detach().cpu().clamp(0.0, 1.0).tolist()
    xyxy = bbox_cxcywh_to_xyxy(pred, width, height)
    x1, y1, x2, y2 = xyxy
    init_box = clamp_xywh([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], width, height, delta=args.action_delta)
    return xyxy, init_box


@torch.no_grad()
def predict_teacher_action(teacher, img: Image.Image, box_xywh, args, device):
    width, height = img.size
    obs = render_crop(img, box_xywh, args.img_size).unsqueeze(0).to(device)
    state = box_state(box_xywh, width, height).unsqueeze(0).to(device)
    feats = teacher.backbone(obs)
    logits = teacher.actor(torch.cat([feats, state], dim=1))
    if args.teacher_action_selection == "sample_logits":
        return int(torch.distributions.Categorical(logits=logits.squeeze(0)).sample().item())
    probs = torch.softmax(logits, dim=1)
    return int(probs.argmax(dim=1).item())


@torch.no_grad()
def rollout_teacher(teacher, img: Image.Image, init_box, args, device):
    width, height = img.size
    box = list(init_box)
    actions = []
    for _ in range(args.max_steps):
        action = predict_teacher_action(teacher, img, box, args, device)
        actions.append(ACTIONS[action])
        if ACTIONS[action] == "stop":
            break
        box = step_box(box, action, width, height, delta=args.action_delta)
    return box, actions


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / max(1, len(values)) if values else None


def rate_at(rows, key, threshold):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return sum(1.0 for v in vals if v >= threshold) / max(1, len(vals)) if vals else None


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[runtime] cuDNN disabled; CUDA is still used, but convolution may be slower.")

    device = torch.device(args.device)
    root = find_adacrop_root()

    records = load_records(args.eval_json, root, require_images=True)
    records = [r for r in records if r.get("boxes")]
    if args.max_images > 0:
        records = records[: args.max_images]
    if not records:
        raise RuntimeError("No evaluation records with both images and bbox labels were resolved.")

    teacher = load_teacher(args.teacher_ckpt, device)
    rows = []

    for idx, rec in enumerate(records, start=1):
        try:
            img = Image.open(rec["img"]).convert("RGB")
        except Exception as exc:
            print(f"[skip] failed to open {rec['img']}: {exc}")
            continue

        try:
            bbox_xyxy, bbox_box = predict_teacher_bbox(teacher, img, args, device)
            width, height = img.size
            bbox_iou = best_gt_iou_xyxy(bbox_xyxy, rec["boxes"], width, height, rec["img"])

            rl_box, actions = rollout_teacher(teacher, img, bbox_box, args, device)
            rl_iou = best_gt_iou(rl_box, rec["boxes"], width, height, rec["img"])
        except RuntimeError as exc:
            if not args.skip_errors:
                raise
            print(f"[skip] runtime error on {rec['img']}: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        rows.append(
            {
                "img": rec["img"],
                "bbox_iou": bbox_iou,
                "rl_iou": rl_iou,
                "iou_gain": None if bbox_iou is None or rl_iou is None else rl_iou - bbox_iou,
                "steps": len(actions),
                "stopped": bool(actions and actions[-1] == "stop"),
                "bbox_box_xywh": bbox_box,
                "rl_box_xywh": rl_box,
                "actions": actions,
            }
        )

        if idx % 25 == 0:
            print(
                f"[eval] {idx}/{len(records)} "
                f"bbox_iou={mean([r['bbox_iou'] for r in rows]):.4f} "
                f"rl_iou={mean([r['rl_iou'] for r in rows]):.4f}"
            )

    metrics = {
        "teacher_action_selection": args.teacher_action_selection,
        "num_eval": len(rows),
        "avg_bbox_iou": mean([r["bbox_iou"] for r in rows]),
        "bbox_iou_at_0_3": rate_at(rows, "bbox_iou", 0.3),
        "bbox_iou_at_0_5": rate_at(rows, "bbox_iou", 0.5),
        "bbox_iou_at_0_7": rate_at(rows, "bbox_iou", 0.7),
        "avg_rl_iou": mean([r["rl_iou"] for r in rows]),
        "rl_iou_at_0_3": rate_at(rows, "rl_iou", 0.3),
        "rl_iou_at_0_5": rate_at(rows, "rl_iou", 0.5),
        "rl_iou_at_0_7": rate_at(rows, "rl_iou", 0.7),
        "avg_iou_gain": mean([r["iou_gain"] for r in rows]),
        "avg_steps": mean([r["steps"] for r in rows]),
        "stop_rate": mean([1.0 if r["stopped"] else 0.0 for r in rows]),
        "rl_better_rate": mean([1.0 if r["iou_gain"] is not None and r["iou_gain"] > 0 else 0.0 for r in rows]),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "rows": rows}, f, indent=2, ensure_ascii=False)
        print(f"[save] {args.output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["img", "bbox_iou", "rl_iou", "iou_gain", "steps", "stopped"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[save] {args.output_csv}")


if __name__ == "__main__":
    main()
