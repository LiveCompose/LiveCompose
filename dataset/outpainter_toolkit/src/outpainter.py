"""
统一版数据集扩图脚本
- 支持固定 N 张 or 随机抽取版本
- 自动网格拼图 (生成图 N 张 + 原图扩展版 1 张)
- 所有可变项来自 configs/config.yaml
"""

import os
import math
import json
import random
from pathlib import Path

import yaml
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration

# ----------------------------------------------------------------------
# 读取配置
# ----------------------------------------------------------------------
CFG_PATH = Path(os.getenv("EXPANDER_CFG", "configs/config.yaml")).expanduser()
if not CFG_PATH.exists():
    raise FileNotFoundError(
        f"找不到配置文件: {CFG_PATH}\n"
    )

with CFG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def _get(key, default=None, cast=lambda x: x):
    if key not in cfg and default is None:
        raise KeyError(f"配置缺失: {key}")
    return cast(cfg.get(key, default))

# 路径
INPUT_ROOT  = Path(_get("input_root")).expanduser().resolve()
OUTPUT_ROOT = Path(_get("output_root")).expanduser().resolve()

# 模型
MODEL_PATH  = _get("model_path")
BLIP_PATH   = _get("blip_path")

# 生成参数
NEG_PROMPT  = _get("negative_prompt")
MIN_RATIO   = _get("min_ratio", 0.10, float)
MAX_RATIO   = _get("max_ratio", 0.475, float)
MAX_SIDE    = _get("max_side", 600, int)
BORDER_CHECK = _get("border_check", False, bool)
GRID_MARGIN  = _get("grid_margin", 20, int)

# 版本控制
if "version_params" in cfg:
    VERSIONS = [tuple(v) for v in cfg["version_params"]]        # 固定张数
else:
    pool = [tuple(v) for v in cfg.get("version_pool", [])]
    pick = int(cfg.get("random_pick", 1))
    if len(pool) < pick:
        raise ValueError("version_pool 数量不足以随机抽取 random_pick 个版本")
    VERSIONS = random.sample(pool, pick)
N_VERSIONS = len(VERSIONS)

# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def pad_to_multiple_of_8(img: Image.Image, mask: Image.Image):
    w, h = img.size
    pad_r = (-w) % 8
    pad_b = (-h) % 8
    if pad_r or pad_b:
        img  = ImageOps.expand(img , border=(0, 0, pad_r, pad_b), fill="white")
        mask = ImageOps.expand(mask, border=(0, 0, pad_r, pad_b), fill=255)
    return img, mask

def detect_border(pil_img, edge_ratio=0.04):
    gray  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ratio = edges.sum() / 255 / (edges.shape[0] * edges.shape[1])
    return ratio > edge_ratio          # True → 有边框

def blip_caption(pil_img, processor, blip_model):
    inputs = processor(pil_img, return_tensors="pt").to("cuda")
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

# ----------------------------------------------------------------------
# 模型加载
# ----------------------------------------------------------------------
print("🔄 加载 BLIP...")
processor = BlipProcessor.from_pretrained(BLIP_PATH, local_files_only=False)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_PATH, local_files_only=False).to("cuda")

print("🔄 加载 Stable Diffusion Inpaint...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, local_files_only=False
).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# ----------------------------------------------------------------------
# 目录准备
# ----------------------------------------------------------------------
GRID_DIR   = OUTPUT_ROOT / "grids"
CROPS_DIR  = OUTPUT_ROOT / "crops"
(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
GRID_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)
for i in range(1, N_VERSIONS + 1):
    (OUTPUT_ROOT / f"version_{i:02d}").mkdir(exist_ok=True)
JSONL_PATH = OUTPUT_ROOT / "coords.jsonl"
jsonl_f = JSONL_PATH.open("a", encoding="utf-8", buffering=1)

# ----------------------------------------------------------------------
# 收集输入图片
# ----------------------------------------------------------------------
images = []
for comp_dir in INPUT_ROOT.iterdir():
    if not comp_dir.is_dir():
        continue
    for f in comp_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                idx = int(f.stem)          # 以数字前缀排序
            except ValueError:
                idx = 0
            images.append((idx, comp_dir.name, f.name))
images.sort(key=lambda x: x[0])

print(f"📂 共发现 {len(images)} 张图片，准备处理…")

# ----------------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------------
for idx, comp_dir, fname in images:
    key = f"{comp_dir}_{Path(fname).stem}"
    # 如果所有版本都已存在则跳过
    if all((OUTPUT_ROOT / f"version_{i:02d}" / f"{key}_v{i:02d}.png").exists()
           for i in range(1, N_VERSIONS + 1)):
        print(f"⏩ 已全部生成，跳过 {key}")
        continue

    # 读取 & 缩放原图
    src_path = INPUT_ROOT / comp_dir / fname
    orig = Image.open(src_path).convert("RGB")
    w, h = orig.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        w, h = int(w * s), int(h * s)
        orig = orig.resize((w, h), Image.LANCZOS)

    # BLIP 自动描述
    caption_text = blip_caption(orig, processor, blip_model)
    print(f"{key} | BLIP → {caption_text}")

    # 随机白边尺寸
    l = random.randint(int(w*MIN_RATIO), int(w*MAX_RATIO))
    r = random.randint(int(w*MIN_RATIO), int(w*MAX_RATIO))
    t = random.randint(int(h*MIN_RATIO), int(h*MAX_RATIO))
    b = random.randint(int(h*MIN_RATIO), int(h*MAX_RATIO))

    # 生成 canvas & mask
    canvas = ImageOps.expand(orig, border=(l,t,r,b), fill="white")
    mask   = Image.new("L", canvas.size, 255)
    mask.paste(0, (l, t, l+w, t+h))
    canvas, mask = pad_to_multiple_of_8(canvas, mask)
    cw, ch = canvas.size

    generated_imgs = []
    meta_records   = []

    # === 生成多版本 ===
    for vidx, (g_scale, steps) in enumerate(VERSIONS, 1):
        out_dir  = OUTPUT_ROOT / f"version_{vidx:02d}"
        out_file = out_dir / f"{key}_v{vidx:02d}.png"
        if out_file.exists():
            print(f"   ⏭ 版本 v{vidx:02d} 已存在")
            gen_img = Image.open(out_file)
        else:
            seed = random.randint(1, 1_000_000)
            prompt = f"Expand the scene around: {caption_text}, natural, seamless, photorealistic"
            print(f"   ➜ v{vidx:02d} | gs={g_scale} steps={steps} seed={seed}")

            gen_img = pipe(
                prompt=prompt,
                negative_prompt=NEG_PROMPT,
                image=canvas,
                mask_image=mask,
                guidance_scale=g_scale,
                num_inference_steps=steps,
                width=cw, height=ch,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images[0]

            if BORDER_CHECK and detect_border(gen_img):
                print("      ⚠️ 检测到边框，跳过保存")
                continue

            gen_img.save(out_file)

        generated_imgs.append(gen_img)
        meta_records.append({
            "file": str(out_file),
            "orig_bbox": [l, t, l+w, t+h],
            "caption": caption_text,
            "composition": comp_dir,
            "guidance_scale": g_scale,
            "steps": steps
        })

    if not generated_imgs:
        print(f"   ⚠️ {key} 没有有效生成，跳过后续操作")
        continue

    # === 保存 crop & 写 JSONL ===
    for rec, img in zip(meta_records, generated_imgs):
        crop = Image.new("RGB", (w, h))
        crop.paste(img.crop((l, t, l+w, t+h)))
        crop_name = f"crop_{Path(rec['file']).name}"
        crop.save(CROPS_DIR / crop_name)
        jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # === 拼接网格 ===
    total_tiles = len(generated_imgs) + 1      # +1 = 原图扩展版
    cols = math.ceil(math.sqrt(total_tiles))
    rows = math.ceil(total_tiles / cols)

    grid_w = cw * cols + GRID_MARGIN * (cols - 1)
    grid_h = ch * rows + GRID_MARGIN * (rows - 1)
    grid = Image.new("RGB", (grid_w, grid_h), (255,255,255))

    # 依次填充生成图
    for n, img in enumerate(generated_imgs):
        r = n // cols
        c = n % cols
        x = c * (cw + GRID_MARGIN)
        y = r * (ch + GRID_MARGIN)
        grid.paste(img, (x, y))

    # 最后一格放原图扩展
    orig_exp = ImageOps.expand(orig, border=(l,t,r,b), fill="white")
    r = (total_tiles - 1) // cols
    c = (total_tiles - 1) % cols
    x = c * (cw + GRID_MARGIN)
    y = r * (ch + GRID_MARGIN)
    grid.paste(orig_exp, (x, y))

    grid.save(GRID_DIR / f"grid_{key}.png")
    print(f"   ✓ 完成 {len(generated_imgs)} 张，网格已保存")

jsonl_f.close()
print("✅ 全部处理结束")
