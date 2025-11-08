# outpainter_toolkit

用于 **批量图像扩图 + 快速人工筛选** 的轻量级工具集  
（Stable Diffusion v2 Inpaint + BLIP 自动描述）。

- **`src/expander.py`** 自动为每张原图生成 *N* 张扩图  
- **`src/annotate_app.py`** 图片人工筛选工具，Tk/Ttk GUI，点选最佳版本或丢弃  
- **`configs/expander.yaml`** 配置参数

**模型下载**
 • Stable Diffusion v2 Inpaint：https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
 • BLIP Image Captioning (base)：https://huggingface.co/Salesforce/blip-image-captioning-base

---

## 1 · 快速开始

```bash
# 1) 克隆该工具集创建 Python 环境（建议 ≥3.9，CUDA 环境速度更佳）
python -m pip install -r requirements.txt

# 2) 编辑专属配置
编辑configs/expander.yaml，参数说明见后
# ↳  修改 input_root / output_root / 模型路径 等字段

# 3) 运行扩图脚本
python src/expander.py
# 也可以：EXPANDER_CFG=configs/my_other.yaml  python src/expander.py

# 4) 打开标注 GUI
python src/annotate_app.py
```

------

## 2 · 目录结构

```
dataset-expander/
├── configs/
│   └── config.yaml           # 私有配置
├── src/
│   ├── expander.py             # 自动扩图
│   └── annotate_app.py         # GUI 清洗
├── outputs/…                   # 运行时自动生成
├── requirements.txt
└── README.md
```

运行 `expander.py` 后会得到类似结构：

```
output_root/
├── version_01/
├── version_02/
├── …               # version_xx 存放各版本扩图
├── crops/          # 裁下的原图区域，用于验证
├── grids/          # 拼接预览
├── coords.jsonl    # 每张扩图的元数据
└── FINAL_SET/      # GUI 选出的最终图片
```

------

## 3 · 配置文件 (`expander.yaml`) 说明

| 字段                           | 类型                | 说明                                                    |
| ------------------------------ | ------------------- | ------------------------------------------------------- |
| `input_root`                   | 路径                | 原始图片根目录（可含子文件夹）                          |
| `output_root`                  | 路径                | 所有输出存放位置                                        |
| `model_path`                   | HF 模型名或本地路径 | Stable Diffusion v2 Inpaint 权重                        |
| `blip_path`                    | HF 模型名或本地路径 | BLIP 图像描述模型                                       |
| `version_params`               | `[[gs, steps], …]`  | **固定** 版本列表，长度 = 每张原图要生成的张数          |
| 或                             |                     |                                                         |
| `version_pool` + `random_pick` | 列表 + 整数         | **随机**：每张原图从 pool 中随机抽 `random_pick` 组参数 |
| 其他字段                       | —                   | 负向提示词、边框检测、缩放上限等                        |

> 想变成“一张扩图”只需让 `version_params` 只有一条即可，
>  想改成 “五张扩图” 就写五条 —— **无需改任何代码**。

------

## 4 · 断点续跑

- **outpainter.py** 检测到目标文件已存在会自动跳过，随时可中断／继续。
- **annotate_app.py** 使用 `selected.jsonl` / `discarded.jsonl` 记录进度，已处理的 key 不会再次出现。

------

## 5 · 依赖与硬件

| 组件    | 版本 / 说明                                |
| ------- | ------------------------------------------ |
| Python  | ≥ 3.9                                      |
| PyTorch | **请按自己 CUDA 版本单独安装**（示例见下） |
| 其他库  | 见 `requirements.txt`                      |
| GPU     | 建议 8 GB VRAM+                            |

安装 PyTorch 示例（CUDA 11.8）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

