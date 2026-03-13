# 为pairwise rank训练生成训练对
import os
import json
import random
from pathlib import Path
from PIL import Image 
from typing import Optional

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _fix_box_xywh(box):
    # box: [x,y,w,h]
    x, y, w, h = box
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x = int(round(x))
    y = int(round(y))
    return [x, y, w, h]


def _clip_box_to_image(box, W, H):
    x, y, w, h = _fix_box_xywh(box)
    x = _clamp(x, 0, W - 1)
    y = _clamp(y, 0, H - 1)
    w = _clamp(w, 1, W - x)
    h = _clamp(h, 1, H - y)
    return [x, y, w, h]


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)


def _iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)

def _resolve_img_path(p: str, img_root: Optional[str]) -> str:
    # 绝对路径直接返回
    if os.path.isabs(p):
        return p

    if img_root:
        pn = p.replace("\\", "/")
        # 你的 json 里经常带 "Adacrop/..."，这里去掉前缀，避免拼出 Adacrop/Adacrop/...
        if pn.startswith("Adacrop/"):
            pn = pn[len("Adacrop/"):]
        return os.path.join(img_root, pn)

    return p

def _jitter_bad_box(good_box, W, H, rng: random.Random,
                    shift_frac=0.15,
                    shrink_min=0.55,
                    shrink_max=0.85):
    """
    以 good_box 为中心做平移+缩放，生成一个候选 bad_box（像素坐标）。
    """
    gx, gy, gw, gh = _clip_box_to_image(good_box, W, H)
    cx = gx + gw / 2.0
    cy = gy + gh / 2.0

    s = rng.uniform(shrink_min, shrink_max)
    nw = max(1.0, gw * s)
    nh = max(1.0, gh * s)

    dx = (rng.uniform(-shift_frac, shift_frac)) * gw
    dy = (rng.uniform(-shift_frac, shift_frac)) * gh

    ncx = cx + dx
    ncy = cy + dy

    x = int(round(ncx - nw / 2.0))
    y = int(round(ncy - nh / 2.0))
    bad = [x, y, int(round(nw)), int(round(nh))]
    bad = _clip_box_to_image(bad, W, H)
    return bad

def _rand_bad_box(W, H, rng: random.Random,
                  min_side_frac=0.25,
                  max_side_frac=0.95):
    """
    在整张图上随机生成一个 box [x,y,w,h]。
    side frac 是相对于短边/长边的粗略比例，避免太极端。
    """
    # 采样尺寸
    w = int(round(rng.uniform(min_side_frac, max_side_frac) * W))
    h = int(round(rng.uniform(min_side_frac, max_side_frac) * H))
    w = _clamp(w, 1, W)
    h = _clamp(h, 1, H)

    # 采样位置
    x = int(rng.randint(0, max(0, W - w)))
    y = int(rng.randint(0, max(0, H - h)))
    return [x, y, w, h]


def make_pairs_from_split_boxes(
    split_json: str,
    out_json: str,
    mode: str = "good_vs_full",    # "good_vs_full" | "good_vs_other" | "all_good_vs_full | "all_good_vs_jitter" | "all_good_vs_random"
    seed: int = 42,
    max_pairs_per_image: int = 1,
    jitter_per_good: int = 2,
    jitter_tries: int = 30,
    iou_max: float = 0.75,
    area_ratio_min: float = 0.45,
    area_ratio_max: float = 1.45,
    random_per_good: int = 3,
    random_tries: int = 60,
    iou_max_random: float = 0.30,
    random_area_ratio_min: float = 0.30,
    random_area_ratio_max: float = 1.50,
):
    """
    从 crop 标注/候选框 json 生成 rank training pairs。
    输出每条：
      {"img": "...jpg", "good_box":[x,y,w,h], "bad_box":[x,y,w,h] or null, "bad_type":"full|box"}
    box 格式：与 train.json 一致 [x, y, w, h]（看起来是左上+宽高）
    """
    random.seed(seed)
    rng = random.Random(seed)  # used by jitter

    with open(split_json, "r", encoding="utf-8") as f:
        recs = json.load(f)

    # split_json: .../Adacrop/data/splits/train.json -> parents[2] = .../Adacrop
    sp = Path(split_json).resolve()
    img_root = str(sp.parents[2]) if len(sp.parents) >= 3 else None

    pairs = []

    for r in recs:
        img = r.get("img")
        boxes = r.get("box") or []
        if not img or not isinstance(boxes, list) or len(boxes) == 0:
            continue

        if mode in ("all_good_vs_full", "all_good_vs_jitter", "all_good_vs_random"):
            # 需要图像尺寸信息时才打开图
            W = H = None
            if mode in ("all_good_vs_jitter", "all_good_vs_random"):
                try:
                    img_path = _resolve_img_path(img, img_root)
                    with Image.open(img_path) as im:
                        W, H = im.size
                except Exception:
                    continue

            for good in boxes:
                good = _fix_box_xywh(good)
                gx, gy, gw, gh = _clip_box_to_image(good, W, H) if W and H else good
                good_c = [gx, gy, gw, gh]
                good_area = float(gw * gh) if (W and H) else float(good_c[2] * good_c[3])

                if mode == "all_good_vs_full":
                    pairs.append({"img": img, "good_box": good, "bad_box": None, "bad_type": "full"})
                    continue

                if mode == "all_good_vs_jitter":
                    made = 0
                    for _ in range(jitter_per_good):
                        bad = None
                        for _t in range(jitter_tries):
                            cand = _jitter_bad_box(good_c, W, H, rng=rng)
                            iou = _iou_xywh(good_c, cand)
                            area = float(cand[2] * cand[3])
                            area_ratio = area / max(1.0, good_area)
                            if iou <= iou_max and area_ratio_min <= area_ratio <= area_ratio_max:
                                bad = cand
                                break
                        if bad is None:
                            continue
                        pairs.append({"img": img, "good_box": good_c, "bad_box": bad, "bad_type": "box"})
                        made += 1
                    if made == 0:
                        pairs.append({"img": img, "good_box": good_c, "bad_box": None, "bad_type": "full"})
                    continue

                if mode == "all_good_vs_random":
                    made = 0
                    for _ in range(random_per_good):
                        bad = None
                        for _t in range(random_tries):
                            cand = _rand_bad_box(W, H, rng=rng)
                            iou = _iou_xywh(good_c, cand)
                            area = float(cand[2] * cand[3])
                            area_ratio = area / max(1.0, good_area)
                            if iou <= iou_max_random and random_area_ratio_min <= area_ratio <= random_area_ratio_max:
                                bad = cand
                                break
                        if bad is None:
                            continue
                        pairs.append({"img": img, "good_box": good_c, "bad_box": bad, "bad_type": "box"})
                        made += 1
                    if made == 0:
                        # 兜底：保底还是给一个 full
                        pairs.append({"img": img, "good_box": good_c, "bad_box": None, "bad_type": "full"})
                    continue
                
            continue

        for _ in range(max_pairs_per_image):
            good = random.choice(boxes)

            if mode == "good_vs_full":
                pairs.append({"img": img, "good_box": good, "bad_box": None, "bad_type": "full"})
            elif mode == "good_vs_other":
                if len(boxes) < 2:
                    continue
                bad = random.choice([b for b in boxes if b != good])
                pairs.append({"img": img, "good_box": good, "bad_box": bad, "bad_type": "box"})
            else:
                raise ValueError(f"Unknown mode: {mode}")

    os.makedirs(str(Path(out_json).parent), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"* pairs saved: {out_json}")
    print(f"* pairs: {len(pairs)}  mode={mode}  max_pairs_per_image={max_pairs_per_image}")


if __name__ == "__main__":
    make_pairs_from_split_boxes(
        split_json="./data/splits/train.json",
        out_json="./data/splits/rank_pairs_train.json",
        mode="all_good_vs_random",
        max_pairs_per_image=2,
        iou_max=0.85,
    )