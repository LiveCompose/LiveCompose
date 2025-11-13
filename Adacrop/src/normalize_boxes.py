import os
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
TP_JSONL = DATA_DIR / "training_pairs.jsonl"
TP_JSON = DATA_DIR / "training_pairs.json"

FILES_JSON = [
    SPLITS_DIR / "train.json",
    SPLITS_DIR / "val.json",
]

def backup(p: Path):
    try:
        shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
    except Exception:
        pass

def to_int_list(lst):
    try:
        return [int(x) for x in lst]
    except Exception:
        return None

def normalize_box_field(value: Any):
    """
    Normalize box field to a list of [x1,y1,x2,y2] lists.
    Returns normalized list (possibly empty) and flag whether changed/invalid.
    """
    # None -> empty
    if value is None:
        return [], False
    # already list
    if isinstance(value, list):
        # single box as flat list of 4 numbers
        if len(value) == 4 and all(not isinstance(x, list) for x in value):
            ints = to_int_list(value)
            if ints is None:
                return [], True
            return [ints], True
        # list of lists -> validate each inner
        new_boxes = []
        changed = False
        for item in value:
            if isinstance(item, list) and len(item) == 4:
                ints = to_int_list(item)
                if ints is None:
                    continue
                new_boxes.append(ints)
            else:
                # skip invalid inner; mark changed
                changed = True
        return new_boxes, changed
    # if it's dict or other, skip
    return [], True

def normalize_records_list(recs):
    total = len(recs)
    modified = 0
    zero_box = 0
    invalid = 0
    kept = []
    for rec in recs:
        if not isinstance(rec, dict):
            invalid += 1
            continue
        img = rec.get("img") or rec.get("file")
        if not img:
            invalid += 1
            continue
        # prefer existing fields in order
        candidate = None
        for k in ("box", "boxes", "orig_bbox"):
            if k in rec:
                candidate = rec.get(k)
                break
        norm_boxes, changed_flag = normalize_box_field(candidate)
        # assign normalized
        rec["box"] = norm_boxes
        # remove legacy keys
        for k in ("boxes", "orig_bbox", "file"):
            rec.pop(k, None)
        if changed_flag:
            modified += 1
        if not norm_boxes:
            zero_box += 1
        kept.append(rec)
    return kept, {"total": total, "kept": len(kept), "modified": modified, "zero_box": zero_box, "invalid": invalid}

def process_json(path: Path):
    if not path.exists():
        print("skip missing:", path)
        return
    backup(path)
    with open(path, "r", encoding="utf-8") as f:
        recs = json.load(f)
    recs_new, stats = normalize_records_list(recs)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs_new, f, ensure_ascii=False, indent=2)
    print(f"Processed {path.name}: total={stats['total']} kept={stats['kept']} modified={stats['modified']} zero_box={stats['zero_box']} invalid={stats['invalid']} (backup: {path.name}.bak)")

def process_jsonl(path: Path):
    if not path.exists():
        print("skip missing:", path)
        return
    backup(path)
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            objs.append(obj)
    objs_new, stats = normalize_records_list(objs)
    # overwrite jsonl
    with open(path, "w", encoding="utf-8") as f:
        for o in objs_new:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    # also export json array
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(objs_new, f, ensure_ascii=False, indent=2)
    print(f"Processed {path.name}: total={stats['total']} kept={stats['kept']} modified={stats['modified']} zero_box={stats['zero_box']} invalid={stats['invalid']} (backup: {path.name}.bak). Exported {path.with_suffix('.json').name}")

def main():
    for p in FILES_JSON:
        process_json(p)
    process_jsonl(TP_JSONL)
    # also normalize training_pairs.json if exists
    if TP_JSON.exists():
        process_json(TP_JSON)

if __name__ == "__main__":
    main()