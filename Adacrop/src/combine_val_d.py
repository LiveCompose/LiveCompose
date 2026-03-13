import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"

VAL_JSON = SPLITS_DIR / "val.json"
VAL_GAIC_JSON = SPLITS_DIR / "val_gaic.json"
OUT_JSON = SPLITS_DIR / "val_mixed.json"


def _is_num(x):
    return isinstance(x, (int, float))


def _normalize_box_list(box):
    if box is None:
        return []

    # 单框: [x1, y1, x2, y2]
    if isinstance(box, list) and len(box) == 4 and all(_is_num(x) for x in box):
        return [[int(x) for x in box]]

    norm_boxes = []
    if isinstance(box, list):
        for b in box:
            # 正常情况: [[x1, y1, x2, y2], ...]
            if isinstance(b, list) and len(b) == 4 and all(_is_num(x) for x in b):
                norm_boxes.append([int(x) for x in b])
            # 异常多包一层: [[[x1, y1, x2, y2]], ...]
            elif (
                isinstance(b, list)
                and len(b) == 1
                and isinstance(b[0], list)
                and len(b[0]) == 4
                and all(_is_num(x) for x in b[0])
            ):
                norm_boxes.append([int(x) for x in b[0]])

    return norm_boxes[:3]


def normalize_rec(rec):
    if not isinstance(rec, dict):
        return None

    img = rec.get("img") or rec.get("file")
    if not img:
        return None
    img = str(img).replace("\\", "/")

    box = rec.get("box", rec.get("boxes", []))
    box = _normalize_box_list(box)

    out = {
        "img": img,
        "box": box,
    }

    if "source" in rec:
        out["source"] = rec["source"]
    if "score" in rec:
        out["score"] = rec["score"]

    return out


def load_json(path: Path):
    if not path.exists():
        print(f"warning: file not found: {path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for rec in data:
        nr = normalize_rec(rec)
        if nr:
            out.append(nr)
    return out


def main(seed=42):
    val_recs = load_json(VAL_JSON)
    gaic_recs = load_json(VAL_GAIC_JSON)

    val_map = {r["img"]: r for r in val_recs}
    gaic_map = {r["img"]: r for r in gaic_recs}

    overlap = len(set(val_map.keys()) & set(gaic_map.keys()))

    merged = dict(val_map)
    merged.update(gaic_map)  # priority: gaic > val

    merged_list = list(merged.values())
    random.Random(seed).shuffle(merged_list)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    print(f"val.json: {len(val_recs)}")
    print(f"val_gaic.json: {len(gaic_recs)}")
    print(f"overlap: {overlap}")
    print(f"saved: {OUT_JSON}")
    print(f"total: {len(merged_list)}")


if __name__ == "__main__":
    main()