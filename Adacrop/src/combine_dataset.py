# 制作混合数据集
import os
import sys
import json
import random
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
TRAIN_JSON = os.path.join(SPLITS_DIR, "train.json")
TRAINING_PAIRS = os.path.join(DATA_DIR, "training_pairs.jsonl")
GAIC_TRAIN = os.path.join(SPLITS_DIR, "train_gaic.json")
OUT_JSON = os.path.join(SPLITS_DIR, "train_mixed2.json")

# try import fix_img_path from modify_dataset if available
try:
    from modify_dataset import fix_img_path
except Exception:
    def fix_img_path(p):
        if not p:
            return p
        return p.replace("\\", "/")

def normalize_rec(rec):
    if not isinstance(rec, dict):
        return None
    img = rec.get("img") or rec.get("file")
    if not img:
        return None
    img = fix_img_path(img)
    box = rec.get("box") or rec.get("boxes") or rec.get("orig_bbox")
    if isinstance(box, list) and len(box) == 4:
        try:
            box = [int(x) for x in box]
        except Exception:
            pass
    else:
        box = [] if box in (None, []) else box
    return {"img": img, "box": box}

def load_train(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = []
    for rec in data:
        nr = normalize_rec(rec)
        if nr:
            recs.append(nr)
    return recs

def load_pairs(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            nr = normalize_rec(obj)
            if nr:
                recs.append(nr)
    return recs

def main(seed=None):
    if not os.path.exists(TRAIN_JSON):
        print("train.json not found:", TRAIN_JSON); return
    if not os.path.exists(TRAINING_PAIRS):
        print("training_pairs.jsonl not found:", TRAINING_PAIRS); return
    if not os.path.exists(GAIC_TRAIN):
        print("train_gaic.json not found:", GAIC_TRAIN); return

    train_recs = load_train(TRAIN_JSON)
    pairs_recs = load_pairs(TRAINING_PAIRS)
    gaic_recs = load_train(GAIC_TRAIN)

    train_map = {r["img"]: r for r in train_recs}
    pairs_map = {r["img"]: r for r in pairs_recs}
    gaic_map = {r["img"]: r for r in gaic_recs}

    overlap_train_pairs = len(set(train_map.keys()) & set(pairs_map.keys()))
    overlap_train_gaic = len(set(train_map.keys()) & set(gaic_map.keys()))
    overlap_pairs_gaic = len(set(pairs_map.keys()) & set(gaic_map.keys()))

    # merge: pairs override train
    merged_map = dict(train_map)
    merged_map.update(pairs_map)
    merged_map.update(gaic_map)

    merged_list = list(merged_map.values())

    # shuffle to mix order (not simple append). deterministic if seed provided.
    if seed is not None:
        random.Random(seed).shuffle(merged_list)
    else:
        random.shuffle(merged_list)

    # save merged file
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    print("train.json entries (with img):", len(train_recs))
    print("training_pairs.jsonl entries (with img):", len(pairs_recs))
    print("train_gaic.json entries (with img):", len(gaic_recs))
    print("overlap train/pairs:", overlap_train_pairs)
    print("overlap train/gaic:", overlap_train_gaic)
    print("overlap pairs/gaic:", overlap_pairs_gaic)
    print("Saved merged file:", OUT_JSON)
    print("Total merged entries:", len(merged_list))
    print("Sample 5 entries:")
    for s in merged_list[:5]:
        print(" ", s["img"], "box:", s["box"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed (use None for non-deterministic)")
    args = parser.parse_args()
    main(seed=args.seed)