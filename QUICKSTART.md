# 🚀 快速开始 (Quick Start)

本文档将引导你从零开始完成环境搭建、数据准备、模型训练和推理测试的全流程。

---

## 1. 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Linux (x86_64) | Ubuntu 18.04+ |
| **Python** | 3.8 | 3.8（GAIC CUDA 扩展编译依赖此版本） |
| **CUDA** | 11.x | 11.1+ |
| **GPU 显存** | 16GB | NVIDIA A800 / V100（80GB 更佳） |
| **内存** | 32GB RAM | 64GB+ RAM |
| **磁盘** | 20GB 空闲 | 100GB+（含数据集） |

---

## 2. 安装依赖

```bash
# 克隆项目
git clone <repo-url> LiveCompose
cd LiveCompose

# 创建 conda 环境（推荐）
conda create -n livecompose python=3.8 -y
conda activate livecompose

# 安装 PyTorch（请根据你的 CUDA 版本调整）
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装核心依赖
pip install pyyaml pillow numpy scipy matplotlib onnx

# （可选）若使用 ONNX 导入的 NIMA 模型
pip install onnx2torch
```

---

## 3. 编译 GAIC CUDA 扩展

训练使用的 GAIC 评分器依赖自定义的 `RoIAlign` 和 `RoDAlign` CUDA 算子，需要编译：

```bash
cd Adacrop/GAIC/untils

# 修改 roi_align/make.sh 和 rod_align/make.sh 中的参数：
#   CUDA_HOME=/usr/local/cuda
#   -arch=sm_80  (A100/A800)
#   -arch=sm_70  (V100)
#   -arch=sm_86  (RTX 3090/4090)

# 编译
bash make_all.sh

# 验证编译产物
ls roi_align/roi_align_api*.so rod_align/rod_align_api*.so
```

> ⚠️ **注意**: 仓库中预编译的 `.so` 文件基于 **Python 3.8 + x86_64 Linux**。如果你的 Python 版本不同，**必须重新编译**。

---

## 4. 准备评分模型权重

训练时强化学习的奖励信号来自美学评分模型。项目默认使用 **GAIC** 评分器。

### GAIC 评分器（默认，已包含）

预训练权重已包含在 `Adacrop/GAIC/pretrained_models/` 目录下：

```
Adacrop/GAIC/pretrained_models/
├── GAIC-mobilenetv2-reddim16.pth   ← 当前配置使用
├── GAIC-shufflenetv2-reddim32.pth
└── GAIC-vgg16-reddim32.pth
```

### NIMA 评分器（可选）

如需切换为 NIMA 评分器，请下载权重文件到 `NIMA/weights/` 目录，并修改 `config.yaml` 中的 `nima.scorer_type` 和 `nima.weights_path`。

---

## 5. 准备训练数据

### 数据格式

训练数据为 JSON 格式，每条记录包含图片路径和标注框：

```json
[
  {
    "img": "Adacrop/data/outpainted/2828_single.png",
    "box": [[42, 68, 362, 248]]
  },
  {
    "img": "Adacrop/data/GAIC_dataset/images/train/346062.jpg",
    "box": [[28, 43, 655, 896]]
  }
]
```

其中 `box` 为 `[x1, y1, x2, y2]` 格式的像素坐标列表，每张图片可以有多个标注框。

### 数据集

我们开源了训练用的扩图数据集 **LiveCompose-outpainted-17k**，可从 HuggingFace 下载：

📥 **[LiveCompose/LiveCompose-outpainted-17k](https://huggingface.co/datasets/LiveCompose/LiveCompose-outpainted-17k)**

```bash
# 使用 huggingface-cli 下载（推荐）
pip install huggingface_hub
huggingface-cli download LiveCompose/LiveCompose-outpainted-17k --repo-type dataset --local-dir ./Adacrop/data/outpainted

# 或使用 git clone（需要 git-lfs）
git lfs install
git clone https://huggingface.co/datasets/LiveCompose/LiveCompose-outpainted-17k ./Adacrop/data/outpainted
```

该数据集包含约 17,000 张通过 Stable Diffusion Outpainting 生成的扩图，每张图片附带专业标注的裁剪框。

此外，你也可以使用以下方式构建自定义数据集：

- **公开数据集**：支持 [GAICD](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping) 数据集
- **CUHK 数据集**：使用 `Adacrop/src/transform_dataset.py` 将 txt 格式标注转为 JSON
- **自行生成**：使用 `PreProcess/dataset/outpainter_toolkit/` 中的工具，通过 Stable Diffusion Outpainting 生成扩图数据

### 配置数据路径

将你的训练集和验证集分别放置，并更新 `Adacrop/config.yaml` 中的路径：

```yaml
data:
  train_json: "./Adacrop/data/splits/train_mixed2.json"
  val_json:   "./Adacrop/data/splits/val_mixed.json"
  num_envs: 16
```

---

## 6. 配置说明

核心配置文件为 `Adacrop/config.yaml`，以下为关键参数说明：

### 环境参数 (`env`)

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `img_size` | 224 | 输入图像尺寸 |
| `max_steps` | 200 | 每个 episode 最大步数 |
| `action_delta` | 0.05 | 动作增量（平移/缩放幅度） |

### 训练参数 (`train`)

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `training_gpus` | 0 | 训练 GPU 编号 |
| `algorithm` | PPO | 训练算法 |
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `clip_param` | 0.2 | PPO 裁剪参数 |
| `n_steps` | 128 | 每次 rollout 步数 |
| `batch_size` | 512 | 批大小 |
| `minibatch_size` | 256 | Mini-batch 大小 |
| `ppo_epochs` | 6 | 每次更新的 epoch 数 |
| `max_steps` | 1000000 | 总训练步数 |

### 监督预训练参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `supervised_pretrain` | false | 是否执行监督预训练 |
| `pretrain_epochs` | 20 | 预训练轮数 |
| `pretrain_lr` | 1e-4 | 预训练学习率 |
| `init_ckpt` | — | 预训练权重路径（设置后跳过预训练） |

### 评分器参数 (`nima`)

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `scorer_type` | gaic | 评分器类型：`gaic` / `rank` / NIMA ONNX |
| `gaic_repo_dir` | — | GAIC 仓库路径 |
| `gaic_ckpt` | — | GAIC 权重路径 |
| `gaic_backbone` | mobilenetv2 | GAIC backbone |
| `real_score_interval` | 1 | 每 N 步执行一次真实评分 |

---

## 7. 开始训练

训练分为两个阶段：**监督预训练**（可选）和 **PPO 强化学习训练**，通过 `config.yaml` 控制。

### 方式一：加载已有预训练权重 → PPO 训练（推荐）

如果你已有监督预训练的权重文件：

```yaml
# config.yaml
train:
  supervised_pretrain: false
  init_ckpt: "./Adacrop/logs/xxx/pretrain_best_iou.pth"  # 指向你的预训练权重
```

```bash
cd /ai/lzy/LiveCompose
bash Adacrop/start_training.sh
```

### 方式二：从零开始（监督预训练 → PPO）

```yaml
# config.yaml
train:
  supervised_pretrain: true
  init_ckpt: null   # 清空此字段
```

然后运行训练脚本。预训练完成后会自动进入 PPO 训练。

### 方式三：完全从头（不预训练）

```yaml
# config.yaml
train:
  supervised_pretrain: false
  init_ckpt: null
  apply_init_action_bias: true  # 初始化 Actor 偏置，鼓励平移动作
```

> ⚠️ 不预训练直接 RL 收敛较慢，不推荐。

### 训练输出

训练产物保存在 `Adacrop/logs/run_<timestamp>_<scorer>_<gpu>_<envs>/` 目录下：

```
logs/
└── run_20260513_234706_gaic_norm_gpu0_env128/
    ├── config.yaml                        # 本次运行的配置快照
    ├── meta.json                          # 运行元信息
    ├── train_log.jsonl                    # 训练指标日志
    ├── pretrain_best_iou.pth              # 最佳预训练模型（IoU）
    ├── pretrain_last.pth                  # 最新预训练模型
    ├── ppo_latest.pth                     # 最新 PPO 模型
    ├── ppo_best_train_reward.pth          # 最佳训练奖励模型
    ├── ppo_best_val_final_score.pth       # 最佳验证分数模型
    └── final_model.pth                    # 训练结束后的最终模型
```

---

## 8. 推理测试

使用训练好的模型对单张图片进行裁剪推理：

```bash
# 1. 修改 Adacrop/use.py 中的路径
#    IMAGE_PATH = r"./Adacrop/data/your_image.jpg"
#    CKPT_PATH  = r"./Adacrop/logs/run_xxx/ppo_best_val_final_score.pth"

# 2. 运行推理
cd /ai/lzy/LiveCompose
python Adacrop/use.py
```

**输出：**
- `Adacrop/output_img/<name>_crop.jpg` — 裁剪结果
- `Adacrop/output_img/<name>_traj.jpg` — 裁剪轨迹可视化（蓝色=起始，黄色=中间，红色=最终）

---

## 9. 模型导出（ONNX → CoreML）

将 PyTorch 模型导出为 ONNX 格式，用于后续 CoreML 转换和 iOS 端部署：

```bash
cd /ai/lzy/LiveCompose/Adacrop

# 修改 src/export.py 中的路径后运行
python src/export.py
```

**ONNX 模型规格：**

| 名称 | 形状 | 说明 |
|------|------|------|
| **输入** `img` | `[B, 3, 224, 224]` | 图像张量 |
| **输入** `state` | `[B, 4]` | 归一化状态 `(cx, cy, w, h)` |
| **输出** `action_probs` | `[B, 7]` | 动作概率分布 |
| **输出** `value` | `[B, 1]` | 状态价值估计 |

7 个动作分别为：`left, right, up, down, zoom_in, zoom_out, stop`

---

## 常见问题 (FAQ)

<details>
<summary><b>Q: CUDA 扩展编译失败怎么办？</b></summary>

1. 确保 `CUDA_HOME` 环境变量正确指向 CUDA 安装目录
2. 确保 `make.sh` 中的 `-arch` 参数与你的 GPU 架构匹配
3. 确保 `make.sh` 文件格式为 Unix（可在 Vim 中执行 `:set ff=unix`）
4. Python 版本必须为 3.8，否则需要重新编译

</details>

<details>
<summary><b>Q: 训练时显存不足 (OOM) 怎么办？</b></summary>

在 `config.yaml` 中调小以下参数：

```yaml
train:
  n_steps: 64          # 减小 rollout 步数（原 128）
  batch_size: 256      # 减小批大小（原 512）
  minibatch_size: 128  # 减小 mini-batch（原 256）
data:
  num_envs: 8          # 减少并行环境数（原 16）
nima:
  batch_size: 16       # 减小评分批大小（原 32）
```

</details>

<details>
<summary><b>Q: 如何使用自己的数据集训练？</b></summary>

1. 准备图片和 `[x1, y1, x2, y2]` 格式的标注框
2. 按 JSON 格式组织数据（参考 [第 5 步](#5-准备训练数据)）
3. 更新 `config.yaml` 中的 `data.train_json` 和 `data.val_json` 路径
4. 可使用 `Adacrop/src/transform_dataset.py` 辅助处理 CUHK 格式的 txt 标注

</details>

<details>
<summary><b>Q: 如何切换评分器？</b></summary>

修改 `config.yaml` 中的 `nima` 配置段：

```yaml
# 使用 GAIC（默认）
nima:
  scorer_type: gaic
  gaic_repo_dir: /path/to/Adacrop/GAIC
  gaic_ckpt: /path/to/Adacrop/GAIC/pretrained_models/GAIC-mobilenetv2-reddim16.pth
  gaic_backbone: mobilenetv2

# 或使用 PairwiseRank 评分器
nima:
  scorer_type: rank
  rank_ckpt: /path/to/rank_model.pth
  rank_backbone: resnet50
```

</details>

<details>
<summary><b>Q: 训练日志在哪里？如何监控训练进度？</b></summary>

训练日志以 JSONL 格式保存在运行目录下的 `train_log.jsonl` 中，每行一条记录，包含：

- `step`: 全局步数
- `rollout`: rollout 编号
- `mean_reward`: 平均奖励
- `best_reward`: 历史最佳奖励
- `val_avg_final_score`: 验证集平均最终分数
- `gpu_memory_gb`: GPU 显存占用

可以使用以下命令快速查看：

```bash
# 查看最近 10 条日志
tail -10 Adacrop/logs/run_*/train_log.jsonl | python -m json.tool

# 提取奖励曲线数据
cat Adacrop/logs/run_*/train_log.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if 'mean_reward' in d:
        print(d['rollout'], d['mean_reward'])
"
```

</details>