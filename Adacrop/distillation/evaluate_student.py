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
    render_crop,
    render_full_image,
    step_box,
    xywh_to_xyxy,
    clamp_xywh,
)


def parse_args():
    root = find_adacrop_root()
    parser = argparse.ArgumentParser(description="Evaluate distilled MobileNet policy against the ResNet50 teacher.")
    parser.add_argument("--teacher-ckpt", type=Path, default=root.parent / "ppo_best_val_final_score.pth")
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, default=root / "data" / "splits" / "val_mixed.json")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--single-step-samples", type=int, default=512)
    parser.add_argument("--rollout-images", type=int, default=0, help="Number of images for rollout metrics; <=0 uses all eval images.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--action-delta", type=float, default=0.05)
    parser.add_argument(
        "--student-action-selection",
        choices=["sample_logits", "argmax"],
        default="sample_logits",
        help="How the student chooses the next rollout action. sample_logits uses Categorical(logits=logits).",
    )
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
    student_gt_ious = []
    teacher_gt_ious = []

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher_boxes = teacher.backbone_forward(imgs).clamp(0.0, 1.0)
        student_boxes = student.backbone_forward(imgs).clamp(0.0, 1.0)

        bs = imgs.size(0)
        total += bs
        if targets.ndim == 3:
            # targets: [B, K, 4]. Use the closest acceptable GT box for L1.
            per_gt_l1 = torch.abs(student_boxes.unsqueeze(1) - targets).mean(dim=2)
            student_gt_l1 += per_gt_l1.min(dim=1).values.sum().item()
        else:
            student_gt_l1 += torch.nn.functional.l1_loss(student_boxes, targets, reduction="sum").item() / 4.0
        student_teacher_l1 += torch.nn.functional.l1_loss(student_boxes, teacher_boxes).item() * bs

        for s_box, t_box, gt_boxes in zip(student_boxes.cpu(), teacher_boxes.cpu(), targets.cpu()):
            s_xyxy = bbox_cxcywh_to_xyxy(s_box.tolist(), 1, 1)
            t_xyxy = bbox_cxcywh_to_xyxy(t_box.tolist(), 1, 1)
            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes.unsqueeze(0)
            gt_xyxys = [bbox_cxcywh_to_xyxy(gt.tolist(), 1, 1) for gt in gt_boxes]
            s_iou = max(box_iou_xyxy(s_xyxy, gt_xyxy) for gt_xyxy in gt_xyxys)
            t_iou = max(box_iou_xyxy(t_xyxy, gt_xyxy) for gt_xyxy in gt_xyxys)
            student_gt_iou += s_iou
            teacher_gt_iou += t_iou
            student_gt_ious.append(s_iou)
            teacher_gt_ious.append(t_iou)
            student_teacher_iou += box_iou_xyxy(s_xyxy, t_xyxy)

    return {
        "bbox_samples": total,
        "avg_bbox_iou": student_gt_iou / max(1, total),
        "bbox_iou_at_0_3": threshold_rate(student_gt_ious, 0.3),
        "bbox_iou_at_0_5": threshold_rate(student_gt_ious, 0.5),
        "bbox_iou_at_0_7": threshold_rate(student_gt_ious, 0.7),
        "bbox_student_gt_iou": student_gt_iou / max(1, total),
        "bbox_teacher_gt_iou": teacher_gt_iou / max(1, total),
        "teacher_bbox_iou_at_0_3": threshold_rate(teacher_gt_ious, 0.3),
        "teacher_bbox_iou_at_0_5": threshold_rate(teacher_gt_ious, 0.5),
        "teacher_bbox_iou_at_0_7": threshold_rate(teacher_gt_ious, 0.7),
        "bbox_student_teacher_iou": student_teacher_iou / max(1, total),
        "bbox_student_gt_l1": student_gt_l1 / max(1, total),
        "bbox_student_teacher_l1": student_teacher_l1 / max(1, total),
    }


@torch.no_grad()
def predict_action(model, img: Image.Image, box_xywh, args, device, selection: str = "argmax") -> int:
    width, height = img.size
    obs = render_crop(img, box_xywh, args.img_size).unsqueeze(0).to(device)
    state = box_state(box_xywh, width, height).unsqueeze(0).to(device)
    probs, second = model(obs, state)
    if selection == "sample_logits":
        if second.shape != probs.shape:
            raise RuntimeError("sample_logits requires the model forward() to return action logits as its second output.")
        return int(torch.distributions.Categorical(logits=second.squeeze(0)).sample().item())
    return int(probs.argmax(dim=1).item())


@torch.no_grad()
def rollout(model, img: Image.Image, init_box, args, device, selection: str = "argmax"):
    width, height = img.size
    box = list(init_box)
    actions = []
    for _ in range(args.max_steps):
        action = predict_action(model, img, box, args, device, selection=selection)
        actions.append(action)
        if ACTIONS[action] == "stop":
            break
        box = step_box(box, action, width, height, delta=args.action_delta)
    return box, actions


@torch.no_grad()
def predict_bbox(model, img: Image.Image, args, device):
    width, height = img.size
    img_t = render_full_image(img, args.img_size).unsqueeze(0).to(device)
    pred = model.backbone_forward(img_t).squeeze(0).detach().cpu().clamp(0.0, 1.0).tolist()
    raw_xyxy = bbox_cxcywh_to_xyxy(pred, width, height)
    x1, y1, x2, y2 = raw_xyxy
    init_box = clamp_xywh(
        [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)],
        width,
        height,
        delta=args.action_delta,
    )
    return raw_xyxy, init_box


def best_gt_iou(box_xywh, gt_boxes, width, height, img_path):
    if not gt_boxes:
        return None
    pred = xywh_to_xyxy(box_xywh)
    return max(box_iou_xyxy(pred, canonical_box_xyxy(gt, width, height, img_path=img_path)) for gt in gt_boxes)


def best_gt_iou_xyxy(box_xyxy, gt_boxes, width, height, img_path):
    if not gt_boxes:
        return None
    return max(box_iou_xyxy(box_xyxy, canonical_box_xyxy(gt, width, height, img_path=img_path)) for gt in gt_boxes)


def mean_key(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / max(1, len(vals)) if vals else None


def threshold_rate(values, threshold):
    vals = [v for v in values if v is not None]
    return sum(1.0 for v in vals if v >= threshold) / max(1, len(vals)) if vals else None


def rate_at(rows, key, threshold):
    return threshold_rate([r.get(key) for r in rows], threshold)


@torch.no_grad()
def rollout_metrics(teacher, student, records, args, device):
    rows = []
    rollout_records = records if args.rollout_images <= 0 else records[: args.rollout_images]
    for rec in rollout_records:
        try:
            img = Image.open(rec["img"]).convert("RGB")
        except Exception:
            continue
        width, height = img.size
        gt_boxes = rec.get("boxes") or []

        teacher_bbox_xyxy, teacher_init_box = predict_bbox(teacher, img, args, device)
        student_bbox_xyxy, student_init_box = predict_bbox(student, img, args, device)
        teacher_box, teacher_actions = rollout(teacher, img, teacher_init_box, args, device, selection="argmax")
        student_box, student_actions = rollout(
            student,
            img,
            student_init_box,
            args,
            device,
            selection=args.student_action_selection,
        )
        prefix = min(len(teacher_actions), len(student_actions))
        action_prefix_agree = (
            sum(1 for i in range(prefix) if teacher_actions[i] == student_actions[i]) / max(1, prefix)
        )
        student_bbox_iou = best_gt_iou_xyxy(student_bbox_xyxy, gt_boxes, width, height, rec["img"])
        teacher_bbox_iou = best_gt_iou_xyxy(teacher_bbox_xyxy, gt_boxes, width, height, rec["img"])
        student_rl_iou = best_gt_iou(student_box, gt_boxes, width, height, rec["img"])
        teacher_rl_iou = best_gt_iou(teacher_box, gt_boxes, width, height, rec["img"])

        rows.append(
            {
                "img": rec["img"],
                "bbox_iou": student_bbox_iou,
                "rl_iou": student_rl_iou,
                "iou_gain": None if student_bbox_iou is None or student_rl_iou is None else student_rl_iou - student_bbox_iou,
                "teacher_bbox_iou": teacher_bbox_iou,
                "teacher_rl_iou": teacher_rl_iou,
                "teacher_steps": len(teacher_actions),
                "student_steps": len(student_actions),
                "teacher_stopped": bool(teacher_actions and ACTIONS[teacher_actions[-1]] == "stop"),
                "student_stopped": bool(student_actions and ACTIONS[student_actions[-1]] == "stop"),
                "step_diff": abs(len(teacher_actions) - len(student_actions)),
                "trajectory_action_agreement": action_prefix_agree,
                "final_box_iou_teacher_student": box_iou_xyxy(xywh_to_xyxy(teacher_box), xywh_to_xyxy(student_box)),
                "teacher_gt_iou": teacher_rl_iou,
                "student_gt_iou": student_rl_iou,
            }
        )
    if not rows:
        return {}, []

    return {
        "teacher_action_selection": "argmax",
        "student_action_selection": args.student_action_selection,
        "rollout_images": len(rows),
        "avg_rl_iou": mean_key(rows, "rl_iou"),
        "rl_iou_at_0_3": rate_at(rows, "rl_iou", 0.3),
        "rl_iou_at_0_5": rate_at(rows, "rl_iou", 0.5),
        "rl_iou_at_0_7": rate_at(rows, "rl_iou", 0.7),
        "avg_iou_gain": mean_key(rows, "iou_gain"),
        "avg_steps": mean_key(rows, "student_steps"),
        "stop_rate": sum(1.0 for r in rows if r.get("student_stopped")) / max(1, len(rows)),
        "rl_better_rate": sum(1.0 for r in rows if r.get("iou_gain") is not None and r["iou_gain"] > 0) / max(1, len(rows)),
        "rollout_final_box_iou_teacher_student": mean_key(rows, "final_box_iou_teacher_student"),
        "rollout_action_prefix_agreement": mean_key(rows, "trajectory_action_agreement"),
        "rollout_avg_step_diff": mean_key(rows, "step_diff"),
        "rollout_teacher_gt_iou": mean_key(rows, "teacher_gt_iou"),
        "rollout_student_gt_iou": mean_key(rows, "student_gt_iou"),
        "rollout_teacher_avg_steps": mean_key(rows, "teacher_steps"),
        "rollout_student_avg_steps": mean_key(rows, "student_steps"),
    }, rows


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
