import argparse
import csv
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    ACTIONS,
    BBoxDataset,
    BBoxEvalDataset,
    MobileNetPolicy,
    PolicyStateDataset,
    bbox_cxcywh_to_xyxy,
    box_iou_xyxy,
    find_adacrop_root,
    load_records,
    load_teacher,
    soften_probs,
)


def parse_args():
    root = find_adacrop_root()
    parser = argparse.ArgumentParser(description="Two-stage distillation: BBox head + PPO actor policy.")
    parser.add_argument("--teacher-ckpt", type=Path, default=root.parent / "ppo_best_val_final_score.pth")
    parser.add_argument("--train-jsonl", type=Path, default=root / "data" / "outpainted_dataset" / "training_pairs.jsonl")
    parser.add_argument("--val-json", type=Path, default=root / "data" / "splits" / "val_mixed.json")
    parser.add_argument("--output-dir", type=Path, default=root / "distillation" / "runs")
    parser.add_argument("--arch", choices=["mobilenet_v3_small", "mobilenet_v3_large"], default="mobilenet_v3_small")
    parser.add_argument("--resume-student", type=Path, default=None, help="Load an existing student checkpoint before training.")
    parser.add_argument("--skip-bbox-stage", action="store_true", help="Skip Stage 1 and go directly to Stage 2 policy distillation.")

    parser.add_argument("--bbox-epochs", type=int, default=5, help="Stage 1 epochs for bbox head distillation/supervision.")
    parser.add_argument("--epochs", type=int, default=10, help="Stage 2 epochs for actor policy distillation.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bbox-batch-size", type=int, default=0, help="Stage 2 bbox regularization batch size; 0 uses --batch-size.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bbox-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pinned memory. Off by default to reduce Windows CUDA OOM risk.")
    parser.add_argument("--samples-per-image", type=int, default=1)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=512)
    parser.add_argument("--img-size", type=int, default=224)

    parser.add_argument("--random-box-prob", type=float, default=0.65)
    parser.add_argument("--jitter", type=float, default=0.12)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--bbox-gt-weight", type=float, default=1.0)
    parser.add_argument("--bbox-teacher-weight", type=float, default=0.25)
    parser.add_argument("--stage2-bbox-weight", type=float, default=0.10)

    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8, help="Stage 2 early-stop patience in epochs; <=0 disables.")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory=False, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=drop_last,
    )


def iou_from_cxcywh_batch(preds, targets):
    preds = preds.detach().cpu().clamp(0.0, 1.0)
    targets = targets.detach().cpu().clamp(0.0, 1.0)
    ious = []
    for pred, target in zip(preds, targets):
        ious.append(box_iou_xyxy(bbox_cxcywh_to_xyxy(pred.tolist(), 1, 1), bbox_cxcywh_to_xyxy(target.tolist(), 1, 1)))
    return sum(ious) / max(1, len(ious))


def best_iou_against_targets(pred_box, target_boxes):
    pred_xyxy = bbox_cxcywh_to_xyxy(pred_box.tolist(), 1, 1)
    return max(box_iou_xyxy(pred_xyxy, bbox_cxcywh_to_xyxy(t.tolist(), 1, 1)) for t in target_boxes)


@torch.no_grad()
def validate_bbox(student, teacher, loader, device, bbox_gt_weight, bbox_teacher_weight):
    student.eval()
    teacher.eval()
    total = 0
    total_loss = 0.0
    gt_loss_sum = 0.0
    teacher_loss_sum = 0.0
    gt_iou_sum = 0.0
    teacher_iou_sum = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = student.backbone_forward(imgs)
        teacher_preds = teacher.backbone_forward(imgs).clamp(0.0, 1.0)

        if targets.ndim == 3:
            # Evaluation records can have multiple acceptable GT boxes. Use the
            # closest GT for loss, and best IoU for reporting.
            per_box_l1 = torch.abs(preds.unsqueeze(1) - targets).mean(dim=2)
            best_idx = per_box_l1.argmin(dim=1)
            chosen_targets = targets[torch.arange(targets.size(0), device=targets.device), best_idx]
        else:
            chosen_targets = targets

        gt_loss = F.smooth_l1_loss(preds, chosen_targets)
        teacher_loss = F.smooth_l1_loss(preds, teacher_preds)
        loss = bbox_gt_weight * gt_loss + bbox_teacher_weight * teacher_loss

        bs = imgs.size(0)
        total += bs
        total_loss += loss.item() * bs
        gt_loss_sum += gt_loss.item() * bs
        teacher_loss_sum += teacher_loss.item() * bs
        if targets.ndim == 3:
            preds_cpu = preds.detach().cpu().clamp(0.0, 1.0)
            teacher_cpu = teacher_preds.detach().cpu().clamp(0.0, 1.0)
            targets_cpu = targets.detach().cpu().clamp(0.0, 1.0)
            gt_iou_sum += sum(best_iou_against_targets(p, ts) for p, ts in zip(preds_cpu, targets_cpu))
            teacher_iou_sum += sum(best_iou_against_targets(p, ts) for p, ts in zip(teacher_cpu, targets_cpu))
        else:
            gt_iou_sum += iou_from_cxcywh_batch(preds, chosen_targets) * bs
            teacher_iou_sum += iou_from_cxcywh_batch(teacher_preds, chosen_targets) * bs

    return {
        "bbox_loss": total_loss / max(1, total),
        "bbox_gt_loss": gt_loss_sum / max(1, total),
        "bbox_teacher_loss": teacher_loss_sum / max(1, total),
        "bbox_gt_iou": gt_iou_sum / max(1, total),
        "bbox_teacher_iou": teacher_iou_sum / max(1, total),
        "bbox_samples": total,
    }


@torch.no_grad()
def validate_policy(student, teacher, loader, device, temperature):
    student.eval()
    teacher.eval()
    total = 0
    total_kl = 0.0
    total_ce = 0.0
    total_agree = 0.0

    for imgs, states in loader:
        imgs = imgs.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        teacher_probs, _ = teacher(imgs, states)
        student_probs, student_logits = student(imgs, states)
        target_probs = soften_probs(teacher_probs, temperature)
        kl = F.kl_div(F.log_softmax(student_logits / temperature, dim=1), target_probs, reduction="batchmean")
        kl = kl * (temperature * temperature)
        ce = F.cross_entropy(student_logits, teacher_probs.argmax(dim=1))
        agree = (student_probs.argmax(dim=1) == teacher_probs.argmax(dim=1)).float().mean()

        bs = imgs.size(0)
        total += bs
        total_kl += kl.item() * bs
        total_ce += ce.item() * bs
        total_agree += agree.item() * bs

    return {
        "policy_kl": total_kl / max(1, total),
        "policy_ce": total_ce / max(1, total),
        "policy_top1_agreement": total_agree / max(1, total),
        "policy_samples": total,
    }


def save_ckpt(path, student, optimizer, args, epoch, stage, metrics):
    torch.save(
        {
            "arch": args.arch,
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def load_student_checkpoint(student, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[resume] missing keys: {missing[:8]}")
    if unexpected:
        print(f"[resume] unexpected keys: {unexpected[:8]}")
    print(
        f"[resume] loaded student checkpoint: {ckpt_path} "
        f"(stage={ckpt.get('stage', 'unknown')}, epoch={ckpt.get('epoch', 'unknown')})"
    )
    return student.to(device)


def train_bbox_stage(args, student, teacher, train_loader, val_loader, device, run_dir, writer, csv_file):
    print(f"[stage1] bbox distillation/supervision for {args.bbox_epochs} epoch(s)")
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.bbox_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_iou = -1.0

    for epoch in range(1, args.bbox_epochs + 1):
        student.train()
        total = 0
        loss_sum = 0.0
        gt_loss_sum = 0.0
        teacher_loss_sum = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_targets = teacher.backbone_forward(imgs).clamp(0.0, 1.0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = student.backbone_forward(imgs)
                gt_loss = F.smooth_l1_loss(preds, targets)
                teacher_loss = F.smooth_l1_loss(preds, teacher_targets)
                loss = args.bbox_gt_weight * gt_loss + args.bbox_teacher_weight * teacher_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            total += bs
            loss_sum += loss.item() * bs
            gt_loss_sum += gt_loss.item() * bs
            teacher_loss_sum += teacher_loss.item() * bs

        val_bbox = validate_bbox(student, teacher, val_loader, device, args.bbox_gt_weight, args.bbox_teacher_weight)
        row = {
            "stage": "bbox",
            "epoch": epoch,
            "train_loss": loss_sum / max(1, total),
            "train_bbox_gt_loss": gt_loss_sum / max(1, total),
            "train_bbox_teacher_loss": teacher_loss_sum / max(1, total),
            "val_bbox_loss": val_bbox["bbox_loss"],
            "val_bbox_gt_loss": val_bbox["bbox_gt_loss"],
            "val_bbox_teacher_loss": val_bbox["bbox_teacher_loss"],
            "val_bbox_gt_iou": val_bbox["bbox_gt_iou"],
            "val_bbox_teacher_iou": val_bbox["bbox_teacher_iou"],
            "val_bbox_samples": val_bbox["bbox_samples"],
        }
        writer.writerow(row)
        csv_file.flush()

        save_ckpt(run_dir / "student_bbox_stage1_last.pth", student, optimizer, args, epoch, "bbox", row)
        if val_bbox["bbox_gt_iou"] > best_iou + args.min_delta:
            best_iou = val_bbox["bbox_gt_iou"]
            save_ckpt(run_dir / "student_bbox_stage1_best.pth", student, optimizer, args, epoch, "bbox", row)
            print(f"[stage1][save] best bbox: {run_dir / 'student_bbox_stage1_best.pth'}")

        print(
            f"[stage1][epoch {epoch}] loss={row['train_loss']:.4f} "
            f"val_bbox_iou={row['val_bbox_gt_iou']:.3f} "
            f"val_teacher_iou={row['val_bbox_teacher_iou']:.3f}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()


def train_policy_stage(args, student, teacher, policy_loader, bbox_loader, val_policy_loader, val_bbox_loader, device, run_dir, writer, csv_file):
    print(f"[stage2] actor policy distillation for {args.epochs} epoch(s)")
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    bbox_iter = cycle(bbox_loader) if args.stage2_bbox_weight > 0 and len(bbox_loader) > 0 else None

    best_agreement = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        total = 0
        loss_sum = 0.0
        kl_sum = 0.0
        ce_sum = 0.0
        bbox_sum = 0.0
        agree_sum = 0.0

        for step, (imgs, states) in enumerate(policy_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_probs, _ = teacher(imgs, states)
                target_probs = soften_probs(teacher_probs, args.temperature)
                hard_targets = teacher_probs.argmax(dim=1)

            bbox_loss = torch.zeros((), device=device)
            bbox_bs = imgs.size(0)
            if bbox_iter is not None:
                bbox_imgs, bbox_targets = next(bbox_iter)
                bbox_imgs = bbox_imgs.to(device, non_blocking=True)
                bbox_targets = bbox_targets.to(device, non_blocking=True)
                bbox_bs = bbox_imgs.size(0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                student_probs, student_logits = student(imgs, states)
                kl = F.kl_div(F.log_softmax(student_logits / args.temperature, dim=1), target_probs, reduction="batchmean")
                kl = kl * (args.temperature * args.temperature)
                ce = F.cross_entropy(student_logits, hard_targets)
                policy_loss = kl + args.ce_weight * ce

                if bbox_iter is not None:
                    bbox_preds = student.backbone_forward(bbox_imgs)
                    bbox_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
                loss = policy_loss + args.stage2_bbox_weight * bbox_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            total += bs
            loss_sum += loss.item() * bs
            kl_sum += kl.item() * bs
            ce_sum += ce.item() * bs
            bbox_sum += bbox_loss.item() * bbox_bs
            agree_sum += (student_probs.argmax(dim=1) == hard_targets).float().mean().item() * bs

            if step % 50 == 0:
                print(
                    f"[stage2][epoch {epoch}] step {step}/{len(policy_loader)} "
                    f"loss={loss_sum / total:.4f} kl={kl_sum / total:.4f} "
                    f"agree={agree_sum / total:.3f}"
                )

        val_policy = validate_policy(student, teacher, val_policy_loader, device, args.temperature)
        val_bbox = validate_bbox(student, teacher, val_bbox_loader, device, args.bbox_gt_weight, args.bbox_teacher_weight)
        row = {
            "stage": "policy",
            "epoch": epoch,
            "train_loss": loss_sum / max(1, total),
            "train_policy_kl": kl_sum / max(1, total),
            "train_policy_ce": ce_sum / max(1, total),
            "train_policy_top1_agreement": agree_sum / max(1, total),
            "train_stage2_bbox_loss": bbox_sum / max(1, total),
            "val_policy_kl": val_policy["policy_kl"],
            "val_policy_ce": val_policy["policy_ce"],
            "val_policy_top1_agreement": val_policy["policy_top1_agreement"],
            "val_policy_samples": val_policy["policy_samples"],
            "val_bbox_loss": val_bbox["bbox_loss"],
            "val_bbox_gt_iou": val_bbox["bbox_gt_iou"],
            "val_bbox_teacher_iou": val_bbox["bbox_teacher_iou"],
        }

        improved = row["val_policy_top1_agreement"] > best_agreement + args.min_delta
        if improved:
            best_agreement = row["val_policy_top1_agreement"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        should_stop = args.patience > 0 and epochs_without_improvement >= args.patience

        row["best_val_policy_top1_agreement"] = best_agreement
        row["epochs_without_improvement"] = epochs_without_improvement
        row["early_stop"] = bool(should_stop)

        save_ckpt(run_dir / "student_last.pth", student, optimizer, args, epoch, "policy", row)
        if improved:
            save_ckpt(run_dir / "student_best.pth", student, optimizer, args, epoch, "policy", row)
            print(f"[stage2][save] best policy: {run_dir / 'student_best.pth'}")
        if args.save_every > 0 and epoch % args.save_every == 0:
            path = run_dir / f"student_epoch_{epoch:03d}.pth"
            save_ckpt(path, student, optimizer, args, epoch, "policy", row)
            print(f"[stage2][save] periodic checkpoint: {path}")

        writer.writerow(row)
        csv_file.flush()

        print(
            f"[stage2][epoch {epoch}] loss={row['train_loss']:.4f} "
            f"val_agree={row['val_policy_top1_agreement']:.3f} "
            f"val_bbox_iou={row['val_bbox_gt_iou']:.3f} "
            f"best={best_agreement:.3f} stale={epochs_without_improvement}/{args.patience if args.patience > 0 else 'off'}"
        )

        if should_stop:
            print(f"[early-stop] no policy agreement improvement for {args.patience} epoch(s).")
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    root = find_adacrop_root()

    run_dir = args.output_dir / f"{args.arch}_twostage_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(args.train_jsonl, root, require_images=True)
    val_records = load_records(args.val_json, root, require_images=True) if args.val_json.exists() else []
    if args.max_train_images > 0:
        train_records = train_records[: args.max_train_images]
    if args.max_val_images > 0:
        val_records = val_records[: args.max_val_images]
    if not train_records:
        raise RuntimeError("No training images were resolved. Check --train-jsonl and path handling.")

    print(f"[data] train images: {len(train_records)}")
    print(f"[data] val images: {len(val_records)}")
    print(f"[data] first train image: {train_records[0]['img']}")

    bbox_train_ds = BBoxDataset(train_records, img_size=args.img_size, samples_per_image=args.samples_per_image)
    bbox_val_ds = BBoxEvalDataset(val_records or train_records[: min(256, len(train_records))], img_size=args.img_size)
    policy_train_ds = PolicyStateDataset(
        train_records,
        img_size=args.img_size,
        samples_per_image=args.samples_per_image,
        random_box_prob=args.random_box_prob,
        jitter=args.jitter,
    )
    policy_val_ds = PolicyStateDataset(
        val_records or train_records[: min(256, len(train_records))],
        img_size=args.img_size,
        samples_per_image=1,
        random_box_prob=args.random_box_prob,
        jitter=args.jitter,
    )
    if len(bbox_train_ds) == 0:
        raise RuntimeError("No bbox labels found for Stage 1. Check box/orig_bbox fields.")

    bbox_batch_size = args.bbox_batch_size if args.bbox_batch_size > 0 else args.batch_size
    bbox_train_loader = make_loader(
        bbox_train_ds,
        bbox_batch_size,
        True,
        args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    bbox_val_loader = make_loader(
        bbox_val_ds,
        bbox_batch_size,
        False,
        max(0, min(args.num_workers, 4)),
        pin_memory=args.pin_memory,
    )
    policy_train_loader = make_loader(
        policy_train_ds,
        args.batch_size,
        True,
        args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    policy_val_loader = make_loader(
        policy_val_ds,
        args.batch_size,
        False,
        max(0, min(args.num_workers, 4)),
        pin_memory=args.pin_memory,
    )

    teacher = load_teacher(args.teacher_ckpt, device)
    student = MobileNetPolicy(arch=args.arch, n_actions=len(ACTIONS)).to(device)
    if args.resume_student is not None:
        student = load_student_checkpoint(student, args.resume_student, device)

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "stage",
        "epoch",
        "train_loss",
        "train_bbox_gt_loss",
        "train_bbox_teacher_loss",
        "train_policy_kl",
        "train_policy_ce",
        "train_policy_top1_agreement",
        "train_stage2_bbox_loss",
        "val_bbox_loss",
        "val_bbox_gt_loss",
        "val_bbox_teacher_loss",
        "val_bbox_gt_iou",
        "val_bbox_teacher_iou",
        "val_bbox_samples",
        "val_policy_kl",
        "val_policy_ce",
        "val_policy_top1_agreement",
        "val_policy_samples",
        "best_val_policy_top1_agreement",
        "epochs_without_improvement",
        "early_stop",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if args.skip_bbox_stage:
            print("[stage1] skipped by --skip-bbox-stage")
        elif args.bbox_epochs > 0:
            train_bbox_stage(args, student, teacher, bbox_train_loader, bbox_val_loader, device, run_dir, writer, f)
        if args.epochs > 0:
            train_policy_stage(
                args,
                student,
                teacher,
                policy_train_loader,
                bbox_train_loader,
                policy_val_loader,
                bbox_val_loader,
                device,
                run_dir,
                writer,
                f,
            )

    print(f"[done] run dir: {run_dir}")


if __name__ == "__main__":
    main()
