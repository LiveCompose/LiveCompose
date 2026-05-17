import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (
    ACTIONS,
    bbox_cxcywh_to_xyxy,
    box_state,
    clamp_xywh,
    find_adacrop_root,
    load_records,
    load_teacher,
    render_crop,
    render_full_image,
    step_box,
)


def parse_args():
    root = find_adacrop_root()
    parser = argparse.ArgumentParser(description="Analyze teacher PPO actor action distribution on a dataset.")
    parser.add_argument("--teacher-ckpt", type=Path, default=root.parent / "ppo_best_val_final_score.pth")
    parser.add_argument("--eval-json", type=Path, default=root / "data" / "splits" / "val_mixed.json")
    parser.add_argument("--output-json", type=Path, default=root / "distillation" / "teacher_action_distribution.json")
    parser.add_argument("--output-csv", type=Path, default=root / "distillation" / "teacher_action_rollouts.csv")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-delta", type=float, default=0.05)
    parser.add_argument("--init", choices=["bbox", "center", "random"], default="bbox")
    parser.add_argument("--selection", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--min-steps-no-stop", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def teacher_bbox_init(teacher, img: Image.Image, args, device):
    width, height = img.size
    img_t = render_full_image(img, args.img_size).unsqueeze(0).to(device)
    pred = teacher.backbone_forward(img_t).squeeze(0).detach().cpu().clamp(0.0, 1.0).tolist()
    x1, y1, x2, y2 = bbox_cxcywh_to_xyxy(pred, width, height)
    return clamp_xywh([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], width, height, delta=args.action_delta)


def center_init(img: Image.Image):
    width, height = img.size
    scale = 0.6
    w, h = width * scale, height * scale
    return [(width - w) * 0.5, (height - h) * 0.5, w, h]


def random_init(img: Image.Image):
    width, height = img.size
    orig_ratio = width / max(1, height)
    scale = np.random.uniform(0.3, 0.8)
    if orig_ratio >= 1:
        w = max(10.0, width * scale)
        h = max(10.0, w / orig_ratio)
    else:
        h = max(10.0, height * scale)
        w = max(10.0, h * orig_ratio)
    x = np.random.uniform(0.0, max(1.0, width - w))
    y = np.random.uniform(0.0, max(1.0, height - h))
    return clamp_xywh([x, y, w, h], width, height)


@torch.no_grad()
def predict_action_probs(teacher, img: Image.Image, box_xywh, args, device):
    width, height = img.size
    obs = render_crop(img, box_xywh, args.img_size).unsqueeze(0).to(device)
    state = box_state(box_xywh, width, height).unsqueeze(0).to(device)
    probs, _ = teacher(obs, state)
    return probs.squeeze(0).detach().cpu()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[runtime] cuDNN disabled.")

    device = torch.device(args.device)
    root = find_adacrop_root()
    records = load_records(args.eval_json, root, require_images=True)
    if args.max_images > 0:
        records = records[: args.max_images]
    if not records:
        raise RuntimeError("No evaluation images resolved.")

    teacher = load_teacher(args.teacher_ckpt, device)
    total_actions = Counter()
    first_actions = Counter()
    final_actions = Counter()
    per_step = defaultdict(Counter)
    rollout_rows = []
    zoom_out_only = 0
    no_stop = 0

    for idx, rec in enumerate(records, start=1):
        try:
            img = Image.open(rec["img"]).convert("RGB")
        except Exception as exc:
            print(f"[skip] failed to open {rec['img']}: {exc}")
            continue

        if args.init == "bbox":
            box = teacher_bbox_init(teacher, img, args, device)
        elif args.init == "center":
            box = center_init(img)
        else:
            box = random_init(img)

        actions = []
        max_probs = []
        zoom_out_probs = []
        stop_probs = []

        for step in range(args.max_steps):
            probs = predict_action_probs(teacher, img, box, args, device)
            effective_probs = probs.clone()
            if step < args.min_steps_no_stop:
                effective_probs[ACTIONS.index("stop")] = 0.0
                effective_probs = effective_probs / effective_probs.sum().clamp_min(1e-8)

            if args.selection == "sample":
                action_idx = int(torch.distributions.Categorical(probs=effective_probs).sample().item())
            else:
                action_idx = int(effective_probs.argmax().item())
            action = ACTIONS[action_idx]
            actions.append(action)
            max_probs.append(float(effective_probs[action_idx].item()))
            zoom_out_probs.append(float(probs[ACTIONS.index("zoom_out")].item()))
            stop_probs.append(float(probs[ACTIONS.index("stop")].item()))

            total_actions[action] += 1
            per_step[step][action] += 1
            if step == 0:
                first_actions[action] += 1

            if action == "stop":
                break
            box = step_box(box, action_idx, img.width, img.height, delta=args.action_delta)

        if actions:
            final_actions[actions[-1]] += 1
        if actions and all(a == "zoom_out" for a in actions):
            zoom_out_only += 1
        if "stop" not in actions:
            no_stop += 1

        rollout_rows.append(
            {
                "img": rec["img"],
                "steps": len(actions),
                "first_action": actions[0] if actions else "",
                "final_action": actions[-1] if actions else "",
                "stop": "stop" in actions,
                "zoom_out_only": bool(actions and all(a == "zoom_out" for a in actions)),
                "zoom_out_fraction": actions.count("zoom_out") / max(1, len(actions)),
                "actions": " ".join(actions),
                "avg_max_prob": sum(max_probs) / max(1, len(max_probs)),
                "avg_zoom_out_prob": sum(zoom_out_probs) / max(1, len(zoom_out_probs)),
                "avg_stop_prob": sum(stop_probs) / max(1, len(stop_probs)),
            }
        )

        if idx % 25 == 0:
            print(f"[analyze] {idx}/{len(records)} actions={dict(total_actions)}")

    total_n = sum(total_actions.values())
    image_n = len(rollout_rows)
    metrics = {
        "num_images": image_n,
        "total_actions": total_n,
        "action_counts": dict(total_actions),
        "action_rates": {k: v / max(1, total_n) for k, v in total_actions.items()},
        "first_action_counts": dict(first_actions),
        "first_action_rates": {k: v / max(1, image_n) for k, v in first_actions.items()},
        "final_action_counts": dict(final_actions),
        "final_action_rates": {k: v / max(1, image_n) for k, v in final_actions.items()},
        "zoom_out_only_images": zoom_out_only,
        "zoom_out_only_rate": zoom_out_only / max(1, image_n),
        "no_stop_images": no_stop,
        "no_stop_rate": no_stop / max(1, image_n),
        "avg_steps": sum(r["steps"] for r in rollout_rows) / max(1, image_n),
        "init": args.init,
        "selection": args.selection,
        "min_steps_no_stop": args.min_steps_no_stop,
        "per_step_counts": {str(k): dict(v) for k, v in per_step.items()},
    }

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_step_counts"}, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "rollouts": rollout_rows}, f, indent=2, ensure_ascii=False)
        print(f"[save] {args.output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rollout_rows[0].keys()) if rollout_rows else ["img"])
            writer.writeheader()
            writer.writerows(rollout_rows)
        print(f"[save] {args.output_csv}")


if __name__ == "__main__":
    main()
