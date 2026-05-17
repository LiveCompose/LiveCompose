import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from common import (
    ACTIONS,
    BBoxEvalDataset,
    PolicyStateDataset,
    bbox_cxcywh_to_xyxy,
    box_iou_xyxy,
    box_state,
    canonical_box_xyxy,
    find_adacrop_root,
    load_records,
    load_student,
    load_teacher,
    random_box,
    render_crop,
    step_box,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def parse_args():
    root = find_adacrop_root()
    parser = argparse.ArgumentParser(description="Evaluate distilled MobileNet policy against the ResNet50 teacher.")
    parser.add_argument("--teacher-ckpt", type=Path, default=root.parent / "ppo_best_val_final_score.pth")
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, default=root / "data" / "splits" / "val_mixed.json")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--single-step-samples", type=int, default=512)
    parser.add_argument("--rollout-images", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--action-delta", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def single_step_metrics(teacher, student, records, args, device):
    sample_count = min(args.single_step_samples, max(1, len(records)))
    ds = PolicyStateDataset(
        records[:sample_count],
        img_size=args.img_size,
        samples_per_image=max(1, args.single_step_samples // sample_count),
        random_box_prob=0.65,
        jitter=0.12,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    total = 0
    kl_sum = 0.0
    ce_sum = 0.0
    agree_sum = 0.0
    stop_agree_sum = 0.0
    stop_cases = 0

    for imgs, states in loader:
        imgs = imgs.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        teacher_probs, _ = teacher(imgs, states)
        student_probs, student_logits = student(imgs, states)
        teacher_action = teacher_probs.argmax(dim=1)
        student_action = student_probs.argmax(dim=1)

        kl = F.kl_div(student_probs.clamp_min(1e-8).log(), teacher_probs, reduction="batchmean")
        ce = F.cross_entropy(student_logits, teacher_action)
        agree = (teacher_action == student_action).float()
        stop_mask = teacher_action == ACTIONS.index("stop")

        bs = imgs.size(0)
        total += bs
        kl_sum += kl.item() * bs
        ce_sum += ce.item() * bs
        agree_sum += agree.sum().item()
        if stop_mask.any():
            stop_cases += int(stop_mask.sum().item())
            stop_agree_sum += agree[stop_mask].sum().item()

    return {
        "single_step_kl": kl_sum / max(1, total),
        "single_step_ce": ce_sum / max(1, total),
        "single_step_top1_agreement": agree_sum / max(1, total),
        "single_step_stop_agreement": stop_agree_sum / max(1, stop_cases) if stop_cases else None,
        "single_step_samples": total,
    }


@torch.no_grad()
def bbox_metrics(teacher, student, records, args, device):
    ds = BBoxEvalDataset(records, img_size=args.img_size)
    if len(ds) == 0:
        return {
            "bbox_samples": 0,
            "bbox_student_gt_iou": None,
            "bbox_teacher_gt_iou": None,
            "bbox_student_teacher_iou": None,
            "bbox_student_gt_l1": None,
            "bbox_student_teacher_l1": None,
        }
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    total = 0
    student_gt_iou = 0.0
    teacher_gt_iou = 0.0
    student_teacher_iou = 0.0
    student_gt_l1 = 0.0
    student_teacher_l1 = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher_boxes = teacher.backbone_forward(imgs).clamp(0.0, 1.0)
        student_boxes = student.backbone_forward(imgs).clamp(0.0, 1.0)

        bs = imgs.size(0)
        total += bs
        student_gt_l1 += torch.nn.functional.l1_loss(student_boxes, targets).item() * bs
        student_teacher_l1 += torch.nn.functional.l1_loss(student_boxes, teacher_boxes).item() * bs

        for s_box, t_box, gt_boxes in zip(student_boxes.cpu(), teacher_boxes.cpu(), targets.cpu()):
            s_xyxy = bbox_cxcywh_to_xyxy(s_box.tolist(), 1, 1)
            t_xyxy = bbox_cxcywh_to_xyxy(t_box.tolist(), 1, 1)
            gt_xyxys = [bbox_cxcywh_to_xyxy(gt.tolist(), 1, 1) for gt in gt_boxes]
            student_gt_iou += max(box_iou_xyxy(s_xyxy, gt_xyxy) for gt_xyxy in gt_xyxys)
            teacher_gt_iou += max(box_iou_xyxy(t_xyxy, gt_xyxy) for gt_xyxy in gt_xyxys)
            student_teacher_iou += box_iou_xyxy(s_xyxy, t_xyxy)

    return {
        "bbox_samples": total,
        "bbox_student_gt_iou": student_gt_iou / max(1, total),
        "bbox_teacher_gt_iou": teacher_gt_iou / max(1, total),
        "bbox_student_teacher_iou": student_teacher_iou / max(1, total),
        "bbox_student_gt_l1": student_gt_l1 / max(1, total),
        "bbox_student_teacher_l1": student_teacher_l1 / max(1, total),
    }


@torch.no_grad()
def predict_action(model, img: Image.Image, box_xywh, args, device) -> int:
    width, height = img.size
    obs = render_crop(img, box_xywh, args.img_size).unsqueeze(0).to(device)
    state = box_state(box_xywh, width, height).unsqueeze(0).to(device)
    out = model(obs, state)
    probs = out[0]
    return int(probs.argmax(dim=1).item())


@torch.no_grad()
def rollout(model, img: Image.Image, init_box, args, device):
    width, height = img.size
    box = list(init_box)
    actions = []
    for _ in range(args.max_steps):
        action = predict_action(model, img, box, args, device)
        actions.append(action)
        if ACTIONS[action] == "stop":
            break
        box = step_box(box, action, width, height, delta=args.action_delta)
    return box, actions


def best_gt_iou(box_xywh, gt_boxes, width, height, img_path):
    if not gt_boxes:
        return None
    pred = xywh_to_xyxy(box_xywh)
    return max(box_iou_xyxy(pred, canonical_box_xyxy(gt, width, height, img_path=img_path)) for gt in gt_boxes)


@torch.no_grad()
def rollout_metrics(teacher, student, records, args, device):
    rows = []
    for rec in records[: args.rollout_images]:
        try:
            img = Image.open(rec["img"]).convert("RGB")
        except Exception:
            continue
        width, height = img.size
        gt_boxes = rec.get("boxes") or []
        if gt_boxes and random.random() < 0.5:
            init_box = xyxy_to_xywh(random.choice(gt_boxes))
        else:
            init_box = random_box(width, height)

        teacher_box, teacher_actions = rollout(teacher, img, init_box, args, device)
        student_box, student_actions = rollout(student, img, init_box, args, device)
        prefix = min(len(teacher_actions), len(student_actions))
        action_prefix_agree = (
            sum(1 for i in range(prefix) if teacher_actions[i] == student_actions[i]) / max(1, prefix)
        )

        rows.append(
            {
                "img": rec["img"],
                "teacher_steps": len(teacher_actions),
                "student_steps": len(student_actions),
                "step_diff": abs(len(teacher_actions) - len(student_actions)),
                "trajectory_action_agreement": action_prefix_agree,
                "final_box_iou_teacher_student": box_iou_xyxy(xywh_to_xyxy(teacher_box), xywh_to_xyxy(student_box)),
                "teacher_gt_iou": best_gt_iou(teacher_box, gt_boxes, width, height, rec["img"]),
                "student_gt_iou": best_gt_iou(student_box, gt_boxes, width, height, rec["img"]),
            }
        )
    if not rows:
        return {}, []

    def mean_key(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / max(1, len(vals)) if vals else None

    return {
        "rollout_images": len(rows),
        "rollout_final_box_iou_teacher_student": mean_key("final_box_iou_teacher_student"),
        "rollout_action_prefix_agreement": mean_key("trajectory_action_agreement"),
        "rollout_avg_step_diff": mean_key("step_diff"),
        "rollout_teacher_gt_iou": mean_key("teacher_gt_iou"),
        "rollout_student_gt_iou": mean_key("student_gt_iou"),
        "rollout_teacher_avg_steps": mean_key("teacher_steps"),
        "rollout_student_avg_steps": mean_key("student_steps"),
    }, rows


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    root = find_adacrop_root()

    records = load_records(args.eval_json, root, require_images=True)
    if args.max_images > 0:
        records = records[: args.max_images]
    if not records:
        raise RuntimeError("No evaluation images were resolved. Check --eval-json and path handling.")

    print(f"[data] eval images: {len(records)}")
    print(f"[data] first eval image: {records[0]['img']}")
    teacher = load_teacher(args.teacher_ckpt, device)
    student = load_student(args.student_ckpt, device)

    bbox = bbox_metrics(teacher, student, records, args, device)
    single = single_step_metrics(teacher, student, records, args, device)
    roll, roll_rows = rollout_metrics(teacher, student, records, args, device)
    metrics = {**bbox, **single, **roll}

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "rollouts": roll_rows}, f, indent=2, ensure_ascii=False)
        print(f"[save] {args.output_json}")


if __name__ == "__main__":
    main()
