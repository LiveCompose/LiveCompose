"""
annotate_app.py  ·  GUI 人工数据筛选工具 
-------------------------------------------------------------
• 支持任意数量的扩图版本 (version_01, version_02, …)
• 单击任意版本 → 选中并保存到 FINAL_SET
• 🗑️  抛弃  /  ⏭️  跳过
• 断点续跑：selected.jsonl / discarded.jsonl 已记录的 key 自动跳过
• 单元格尺寸根据屏幕 & 行列动态调整，保证整屏可见
"""

import os
import re
import json
import math
import shutil
import yaml
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ----------------------------------------------------------------------
# 读取扩图配置，获取 OUTPUT_ROOT
# ----------------------------------------------------------------------
CFG_PATH = Path(os.getenv("EXPANDER_CFG", "configs/expander.yaml")).expanduser()
if not CFG_PATH.exists():
    messagebox.showerror("错误", f"找不到配置文件：{CFG_PATH}")
    raise SystemExit

with CFG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
OUTPUT_ROOT = Path(cfg["output_root"]).expanduser().resolve()

# ----------------------------------------------------------------------
# 获取所有 version_* 目录 (自动排序)
# ----------------------------------------------------------------------
VERS_DIRS = sorted(
    [p for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p.name.startswith("version_")],
    key=lambda x: x.name
)
N_VER = len(VERS_DIRS)
if N_VER == 0:
    messagebox.showinfo("提示", "未找到任何 version_* 目录，请先运行扩图脚本")
    raise SystemExit

# 其它重要路径
CROPS_DIR = OUTPUT_ROOT / "crops"
COORDS    = OUTPUT_ROOT / "coords.jsonl"
FINAL_DIR = OUTPUT_ROOT / "FINAL_SET"
FINAL_DIR.mkdir(exist_ok=True)

SELECTED_JSON = OUTPUT_ROOT / "selected.jsonl"
DISCARD_JSON  = OUTPUT_ROOT / "discarded.jsonl"

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
# 文件名举例：   nature_00023_v03.png
NAME_PAT = re.compile(r"(?P<comp>.+)_(?P<id>\d+)_v\d{2}\.png$")

def key_in_file(key, file_path: Path) -> bool:
    if not file_path.exists():
        return False
    with file_path.open(encoding="utf-8") as f:
        return any(key in line for line in f)

def write_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------
# 收集实例：保证每个 key 在所有版本目录都有对应图片
# ----------------------------------------------------------------------
def collect_instances():
    bucket = {}
    for ver_dir in VERS_DIRS:
        for p in ver_dir.glob("*.png"):
            m = NAME_PAT.match(p.name)
            if not m:
                continue
            key = f"{m['comp']}_{m['id']}"
            bucket.setdefault(key, {})[ver_dir.name] = p
    # 仅保留“版本齐全”的 key
    full = {k: v for k, v in bucket.items() if len(v) == N_VER}
    # 按数字 id 排序
    keys = sorted(full.keys(), key=lambda x: int(x.split("_")[-1]))
    # 过滤掉已处理过的
    keys = [
        k for k in keys
        if not key_in_file(k, SELECTED_JSON) and not key_in_file(k, DISCARD_JSON)
    ]
    return [(k, full[k]) for k in keys]

INSTANCES = collect_instances()
if not INSTANCES:
    messagebox.showinfo("完成", "🎉 无需标注，目录已空")
    raise SystemExit

# ----------------------------------------------------------------------
# GUI 构建
# ----------------------------------------------------------------------
root = tk.Tk()
root.title("Outpainting Annotator")
root.configure(bg="#ebebeb")

# 动态计算网格行列 (N_VER 扩图 + 1 张原图扩展)
TOTAL_TILES = N_VER + 1
COLS = math.ceil(math.sqrt(TOTAL_TILES))
ROWS = math.ceil(TOTAL_TILES / COLS)

MARGIN = 20
avail_w = root.winfo_screenwidth()  - 100
avail_h = root.winfo_screenheight() - 160
CELL_MAX = min(
    (avail_w - MARGIN * (COLS - 1)) // COLS,
    (avail_h - MARGIN * (ROWS - 1)) // ROWS,
    600
)

# 统一 ttk 样式
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TButton", font=("Arial", 12), padding=6)
style.map("TButton",
          background=[("active", "#d8d8d8")],
          relief=[("pressed", "sunken"), ("!pressed", "raised")])

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

# 为每个 tile 创建 Label
img_labels = []
for r in range(ROWS):
    for c in range(COLS):
        lbl = ttk.Label(frame, borderwidth=2, relief="groove")
        lbl.grid(row=r, column=c, padx=MARGIN//2, pady=MARGIN//2)
        img_labels.append(lbl)

thumbs = []          # PhotoImage 引用避免被垃圾回收
cur_key = ""
cur_paths = {}       # {version_xx: Path}

def load_image(path: Path):
    img = Image.open(path)
    w, h = img.size
    if max(w, h) > CELL_MAX:
        scale = CELL_MAX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

def next_instance():
    """加载下一组实例"""
    global cur_key, cur_paths, thumbs
    if not INSTANCES:
        messagebox.showinfo("完成", "全部实例已处理！")
        root.quit()
        return
    cur_key, cur_paths = INSTANCES.pop(0)
    thumbs = []

    # 先清空所有 label
    for lbl in img_labels:
        lbl.configure(image="", relief="groove")

    # 按 version_* 目录顺序放置扩图
    for idx, ver_dir in enumerate(VERS_DIRS):
        p = cur_paths[ver_dir.name]
        thumbs.append(load_image(p))
        img_labels[idx].configure(image=thumbs[-1], relief="flat")

    # 最后一格放原图扩展 (crop)
    crop_name = f"crop_{cur_paths[VERS_DIRS[0].name].name}"
    crop_path = CROPS_DIR / crop_name
    if crop_path.exists():
        thumbs.append(load_image(crop_path))
        img_labels[TOTAL_TILES - 1].configure(image=thumbs[-1], relief="flat")

    root.title(f"Annotating {cur_key}    |    剩余 {len(INSTANCES)}")

def save_choice(idx):
    """选择第 idx 个版本 (0-based)"""
    if idx >= N_VER:
        return
    sel_dir  = VERS_DIRS[idx]
    sel_path = cur_paths[sel_dir.name]

    shutil.copy(sel_path, FINAL_DIR / sel_path.name)

    # 从 coords.jsonl 中找对应记录
    if COORDS.exists():
        with COORDS.open(encoding="utf-8") as f:
            for line in f:
                if sel_path.name in line:
                    write_jsonl(SELECTED_JSON, json.loads(line))
                    break

    next_instance()

def discard_instance():
    """抛弃当前 key"""
    write_jsonl(DISCARD_JSON, {"discard": cur_key})
    next_instance()

def skip_instance():
    """跳过当前 key，不写任何记录"""
    next_instance()

# 绑定点击事件 (只给扩图版本绑定)
for i in range(N_VER):
    img_labels[i].bind("<Button-1>", lambda e, idx=i: save_choice(idx))

# ----------------------------------------------------------------------
# 控制按钮
# ----------------------------------------------------------------------
btn_bar = ttk.Frame(root, padding=(10, 0, 10, 10))
btn_bar.grid(row=1, column=0)

ttk.Button(btn_bar, text="🗑️  抛弃实例 (Del)", command=discard_instance)\
    .grid(row=0, column=0, padx=10)
ttk.Button(btn_bar, text="⏭️  跳过 (Space)", command=skip_instance)\
    .grid(row=0, column=1, padx=10)

root.bind("<Delete>", lambda e: discard_instance())
root.bind("<space>",  lambda e: skip_instance())

# ----------------------------------------------------------------------
# 启动
# ----------------------------------------------------------------------
next_instance()
root.mainloop()
