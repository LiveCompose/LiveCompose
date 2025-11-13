import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
CUHK_DIR = os.path.join(DATA_DIR, "cuhk_images")
OUTPAINTED_DIR = os.path.join(DATA_DIR, "outpainted")

ADD_ADACROP_PREFIX = True  # 统一加前缀 Adacrop/

def _add_prefix(p: str) -> str:
    # 输入是已标准化的相对路径
    if not ADD_ADACROP_PREFIX:
        return p
    p = p.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("Adacrop/"):
        return p
    if p.startswith("data/"):
        return "Adacrop/" + p
    return p

def fix_img_path(img_path):
    if not img_path:
        return img_path
    img_path = img_path.replace("\\", "/")

    # 统一去掉前导 ./，并把 ./Adacrop/ 变成 Adacrop/
    if img_path.startswith("./"):
        img_path = img_path[2:]
    if img_path.startswith("./Adacrop/"):
        img_path = img_path[2:]

    # 已是 Adacrop/data/... 的情况
    if img_path.startswith("Adacrop/data/"):
        return img_path

    # 已是 data/... 的情况 → 加 Adacrop/ 前缀
    if img_path.startswith("data/"):
        return _add_prefix(img_path)

    # 其它情况：按文件名在已知目录里匹配
    fname = os.path.basename(img_path)
    cand1 = os.path.join(CUHK_DIR, fname)
    cand2 = os.path.join(OUTPAINTED_DIR, fname)
    if os.path.exists(cand1):
        return _add_prefix(os.path.join("data", "cuhk_images", fname).replace("\\", "/"))
    if os.path.exists(cand2):
        return _add_prefix(os.path.join("data", "outpainted", fname).replace("\\", "/"))
    cand_rel = os.path.join(DATA_DIR, fname)
    if os.path.exists(cand_rel):
        return _add_prefix(os.path.join("data", fname).replace("\\", "/"))

    # 找不到则仅规范化分隔符并尽量加前缀
    if "/" in img_path and img_path.split("/", 1)[0] == "Adacrop":
        return img_path  # 已有 Adacrop
    if img_path.startswith("cuhk_images/"):
        return _add_prefix("data/" + img_path)
    if img_path.startswith("outpainted/"):
        return _add_prefix("data/" + img_path)
    return img_path  # 保守返回

def fix_json(file_path, overwrite=True):
    with open(file_path, "r", encoding="utf-8") as f:
        recs = json.load(f)
    changed = False
    n_changed = 0
    for rec in recs:
        # 修正 img/file 字段
        if "file" in rec and rec.get("file"):
            newp = fix_img_path(rec["file"])
            if rec.get("img") != newp:
                rec["img"] = newp
                changed = True
                n_changed += 1
            rec.pop("file", None)
        elif "img" in rec and rec.get("img"):
            newp = fix_img_path(rec["img"])
            if rec["img"] != newp:
                rec["img"] = newp
                changed = True
                n_changed += 1

        # boxes -> box
        if "boxes" in rec:
            if rec.get("box") != rec["boxes"]:
                rec["box"] = rec["boxes"]
                changed = True
                n_changed += 1
            rec.pop("boxes", None)

        # orig_bbox -> box
        if "orig_bbox" in rec:
            if rec.get("box") != rec["orig_bbox"]:
                rec["box"] = rec["orig_bbox"]
                changed = True
                n_changed += 1
            rec.pop("orig_bbox", None)

    out = file_path if overwrite else file_path.replace(".json", "_fixed.json")
    if changed:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"Fixed and saved: {out}  (modified items: {n_changed})")
    else:
        print(f"No change: {file_path}")

def fix_jsonl(file_path, key="file", overwrite=True):
    out_lines = []
    changed = False
    n_changed = 0
    objs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            original = dict(obj)

            # file -> img
            if key in obj and isinstance(obj[key], str):
                newp = fix_img_path(obj[key])
                if obj.get("img") != newp:
                    obj["img"] = newp
                if key != "img":
                    obj.pop(key, None)
            elif "img" in obj and isinstance(obj["img"], str):
                obj["img"] = fix_img_path(obj["img"])

            # orig_bbox / boxes -> box
            if "orig_bbox" in obj:
                if obj.get("box") != obj["orig_bbox"]:
                    obj["box"] = obj["orig_bbox"]
                obj.pop("orig_bbox", None)
            if "boxes" in obj:
                if obj.get("box") != obj["boxes"]:
                    obj["box"] = obj["boxes"]
                obj.pop("boxes", None)

            if obj != original:
                changed = True
                n_changed += 1
            out_lines.append(json.dumps(obj, ensure_ascii=False))
            objs.append(obj)

    out = file_path if overwrite else file_path.replace(".jsonl", "_fixed.jsonl")
    if changed:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"Fixed and saved: {out}  (modified lines: {n_changed})")
    else:
        print(f"No change: {file_path}")

    # 额外导出为 json 数组文件
    json_out = file_path.replace(".jsonl", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(objs, f, ensure_ascii=False, indent=2)
    print(f"Also exported JSON array: {json_out}  (total items: {len(objs)})")

if __name__ == "__main__":
    fix_json(os.path.join(DATA_DIR, "splits", "train.json"))
    fix_json(os.path.join(DATA_DIR, "splits", "val.json"))
    tp = os.path.join(DATA_DIR, "training_pairs.jsonl")
    if os.path.exists(tp):
        fix_jsonl(tp, key="file")
    else:
        print("training_pairs.jsonl not found in data/")