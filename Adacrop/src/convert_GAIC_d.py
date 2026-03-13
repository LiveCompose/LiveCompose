import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAIC_ROOT = ROOT / "data" / "GAIC_dataset"
ANN_ROOT = GAIC_ROOT / "annotations"
IMG_ROOT = GAIC_ROOT / "images"
OUT_DIR = ROOT / "data" / "splits"


def parse_topk_boxes(txt_path: Path, k: int = 3):
    scored_boxes = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            x1, y1, x2, y2 = map(int, parts[:4])
            score = float(parts[4])
            scored_boxes.append(([x1, y1, x2, y2], score))

    if not scored_boxes:
        return [], None

    scored_boxes.sort(key=lambda x: x[1], reverse=True)
    topk = scored_boxes[:k]
    boxes = [box for box, _ in topk]
    best_score = topk[0][1]
    return boxes, best_score


def to_adacrop_relpath(path: Path) -> str:
    parts = path.parts
    if "Adacrop" in parts:
        idx = parts.index("Adacrop")
        return "/".join(parts[idx:])
    return str(path).replace("\\", "/")


def build_split(split: str, topk: int = 3):
    ann_dir = ANN_ROOT / split
    img_dir = IMG_ROOT / split

    records = []
    missing_imgs = 0
    bad_txt = 0

    for txt_path in sorted(ann_dir.glob("*.txt")):
        stem = txt_path.stem
        img_path = img_dir / f"{stem}.jpg"

        if not img_path.exists():
            img_path = img_dir / f"{stem}.png"

        if not img_path.exists():
            missing_imgs += 1
            continue

        best_box, best_score = parse_topk_boxes(txt_path)
        if best_box is None:
            bad_txt += 1
            continue

        records.append({
            "img": to_adacrop_relpath(img_path),
            "box": best_box,
            "source": "gaic",
            "score": best_score,
        })

    return records, missing_imgs, bad_txt


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_records, train_missing, train_bad = build_split("train", topk=3)
    val_records, val_missing, val_bad = build_split("val", topk=3)
    test_records, test_missing, test_bad = build_split("test", topk=3)

    train_out = OUT_DIR / "train_gaic.json"
    val_out = OUT_DIR / "val_gaic.json"
    test_out = OUT_DIR / "test_gaic.json"

    with train_out.open("w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with val_out.open("w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    with test_out.open("w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    print(f"train_gaic.json: {len(train_records)} records")
    print(f"  missing_imgs={train_missing}, bad_txt={train_bad}")
    print(f"val_gaic.json: {len(val_records)} records")
    print(f"  missing_imgs={val_missing}, bad_txt={val_bad}")
    print(f"test_gaic.json: {len(test_records)} records")
    print(f"  missing_imgs={test_missing}, bad_txt={test_bad}")
    print(f"saved: {train_out}")
    print(f"saved: {val_out}")
    print(f"saved: {test_out}")


if __name__ == "__main__":
    main()