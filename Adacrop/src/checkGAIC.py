import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image

try:
    from Adacrop.src.scorers import load_aesthetic_model, full_box_xywh
except ModuleNotFoundError:
    from src.scorers import load_aesthetic_model, full_box_xywh


def _get(rec, keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def _resolve_img_path(json_path: str, img_value: str) -> str:
    p = Path(img_value)
    if p.is_absolute() and p.exists():
        return str(p)

    jp = Path(json_path).resolve()
    cands = [
        jp.parent / img_value,
        jp.parents[1] / img_value if len(jp.parents) >= 2 else None,
        jp.parents[2] / img_value if len(jp.parents) >= 3 else None,
    ]
    for c in cands:
        if c is not None and c.exists():
            return str(c)
    return img_value


def normalize_box(box):
    if box is None:
        return None

    if isinstance(box, dict):
        if all(k in box for k in ["x", "y", "w", "h"]):
            return [box["x"], box["y"], box["w"], box["h"]]
        if all(k in box for k in ["x1", "y1", "x2", "y2"]):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            return [x1, y1, x2 - x1, y2 - y1]
        return None

    if isinstance(box, (list, tuple)):
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
            return [x1, y1, x2 - x1, y2 - y1]

        if len(box) > 0 and isinstance(box[0], (list, tuple)):
            first = random.choice(box)
            if len(first) == 4 and all(isinstance(v, (int, float)) for v in first):
                x1, y1, x2, y2 = first
                return [x1, y1, x2 - x1, y2 - y1]

        if len(box) == 1 and isinstance(box[0], (list, tuple, dict)):
            return normalize_box(box[0])

    return None


def box_iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def random_box_like(img: Image.Image, min_scale=0.3, max_scale=1.0, ref_box=None, max_iou=0.3, max_trials=50):
    W, H = img.size
    img_area = W * H

    for _ in range(max_trials):
        scale = random.uniform(min_scale, max_scale)
        target_area = img_area * scale * random.uniform(0.5, 1.0)
        aspect = random.uniform(0.5, 2.0)

        rw = int((target_area * aspect) ** 0.5)
        rh = int((target_area / aspect) ** 0.5)

        rw = max(8, min(W, rw))
        rh = max(8, min(H, rh))

        max_x = max(0, W - rw)
        max_y = max(0, H - rh)
        rx = random.randint(0, max_x) if max_x > 0 else 0
        ry = random.randint(0, max_y) if max_y > 0 else 0

        cand = [rx, ry, rw, rh]
        if ref_box is None or box_iou_xywh(cand, ref_box) <= max_iou:
            return cand

    return [0, 0, W, H]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, default="./data/splits/train.json", help="pair/val json path")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    recs = json.load(open(args.json, "r", encoding="utf-8"))
    scorer = load_aesthetic_model()

    total = 0
    good_gt_full = 0
    good_gt_rand = 0
    sample_logs = []

    for rec in recs[:args.max_samples]:
        img_rel = _get(rec, ["img", "image", "img_path", "image_path"])
        good_box = _get(rec, ["good_box", "bbox", "box", "crop_box", "best_box"])
        good_box = normalize_box(good_box)

        if img_rel is None or good_box is None:
            continue

        img_path = _resolve_img_path(args.json, img_rel)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        full_box = full_box_xywh(img)
        rand_box = random_box_like(img, ref_box=good_box, max_iou=0.6)

        try:
            s_good = scorer.score_box(img, good_box)
            s_full = scorer.score_box(img, full_box)
            s_rand = scorer.score_box(img, rand_box)
        except NotImplementedError as e:
            print(f"scorer 尚未实现 score_box: {e}")
            return

        total += 1
        good_gt_full += int(s_good > s_full)
        good_gt_rand += int(s_good > s_rand)

        if len(sample_logs) < 10:
            sample_logs.append({
                "img": img_path,
                "good_box": good_box,
                "good": round(float(s_good), 4),
                "full": round(float(s_full), 4),
                "rand": round(float(s_rand), 4),
                "good>full": bool(s_good > s_full),
                "good>rand": bool(s_good > s_rand),
            })

    if total == 0:
        print("没有可评估样本。")
        print(f"skipped_no_box={skipped_no_box}")
        print(f"skipped_bad_path={skipped_bad_path}")
        print(f"skipped_bad_open={skipped_bad_open}")
    

    print(f"total={total}")
    print(f"good > full : {good_gt_full / total:.4f} ({good_gt_full}/{total})")
    print(f"good > rand : {good_gt_rand / total:.4f} ({good_gt_rand}/{total})")
    print("samples:")
    for x in sample_logs:
        print(x)


if __name__ == "__main__":
    main()