# Quick Start

This guide walks you through the full pipeline: environment setup → data preparation → training → distillation → export (ONNX / CoreML) → inference.

---

## 1. Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| **OS** | Linux (x86_64) | Ubuntu 18.04+ |
| **Python** | 3.8 | 3.8 (required for GAIC CUDA extensions) |
| **CUDA** | 11.x | 11.1+ |
| **GPU VRAM** | 16 GB | NVIDIA A800 / V100 (80 GB preferred) |
| **RAM** | 32 GB | 64 GB+ |
| **Disk** | 20 GB free | 100 GB+ (including dataset) |

---

## 2. Install Dependencies

```bash
# Clone the repo
git clone <repo-url> LiveCompose
cd LiveCompose

# Create conda environment (recommended)
conda create -n livecompose python=3.8 -y
conda activate livecompose

# Install PyTorch (adjust for your CUDA version)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Core dependencies
pip install pyyaml pillow numpy scipy matplotlib onnx

# For NIMA ONNX scorer (optional)
pip install onnx2torch

# For CoreML export (macOS recommended, Linux also supported)
pip install coremltools
```

---

## 3. Compile GAIC CUDA Extensions

The GAIC scorer uses custom `RoIAlign` and `RoDAlign` CUDA operators that must be compiled:

```bash
cd Adacrop/GAIC/untils

# Edit roi_align/make.sh and rod_align/make.sh:
#   CUDA_HOME=/usr/local/cuda
#   -arch=sm_80  # A100 / A800
#   -arch=sm_70  # V100
#   -arch=sm_86  # RTX 3090 / 4090

# Build
bash make_all.sh

# Verify
ls roi_align/roi_align_api*.so rod_align/rod_align_api*.so
```

> ⚠️ The pre-compiled `.so` files target **Python 3.8 + x86_64 Linux**. If your Python version differs, you **must recompile**.

---

## 4. Prepare Scoring Model Weights

The RL reward signal comes from an aesthetic scoring model. The project uses **GAIC** by default.

### GAIC Scorer (default, bundled)

Pre-trained weights are included in `Adacrop/GAIC/pretrained_models/`:

```
Adacrop/GAIC/pretrained_models/
├── GAIC-mobilenetv2-reddim16.pth   ← used by default config
├── GAIC-shufflenetv2-reddim32.pth
└── GAIC-vgg16-reddim32.pth
```

### NIMA Scorer (optional)

To switch to NIMA, download weights to `NIMA/weights/` and update `config.yaml` → `nima.scorer_type` and `nima.weights_path`.

---

## 5. Prepare Training Data

### Data Format

Training data is a JSON file where each record contains an image path and bounding box annotations:

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

`box` is a list of `[x1, y1, x2, y2]` pixel-coordinate annotations. Each image may have multiple valid crop boxes.

### Dataset

We open-sourced our outpainting training dataset **LiveCompose-outpainted-17k** on HuggingFace:

**[LiveCompose/LiveCompose-outpainted-17k](https://huggingface.co/datasets/LiveCompose/LiveCompose-outpainted-17k)**

```bash
# Download via huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download LiveCompose/LiveCompose-outpainted-17k \
    --repo-type dataset --local-dir ./Adacrop/data/outpainted

# Or via git clone (requires git-lfs)
git lfs install
git clone https://huggingface.co/datasets/LiveCompose/LiveCompose-outpainted-17k \
    ./Adacrop/data/outpainted
```

This dataset contains ~17,000 outpainted images generated via Stable Diffusion, each with professionally annotated crop bounding boxes.

You can also use alternative data sources:

- **GAICD dataset**: [Grid-Anchor-based-Image-Cropping](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping)
- **CUHK dataset**: Convert txt annotations with `Adacrop/src/transform_dataset.py`
- **Custom outpainting**: Use `PreProcess/dataset/outpainter_toolkit/` to generate data via Stable Diffusion Outpainting

### Configure Data Paths

Update `Adacrop/config.yaml` to point to your data:

```yaml
data:
  train_json: "./Adacrop/data/splits/train_mixed2.json"
  val_json:   "./Adacrop/data/splits/val_mixed.json"
  num_envs: 128
```

---

## 6. Configuration Reference

All training hyperparameters live in `Adacrop/config.yaml`. Key sections:

### Environment (`env`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 224 | Input image size for the model |
| `max_steps` | 200 | Max steps per RL episode |
| `action_delta` | 0.05 | Action increment (pan/zoom magnitude) |
| `init_with_pred_prob` | 0.4 | Probability of initializing crop box from BBox Head prediction |
| `init_box_jitter` | 0.05 | Random jitter on initial box for exploration |

### Training (`train`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_gpus` | 1 | GPU index for training |
| `algorithm` | PPO | Training algorithm |
| `lr` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `clip_param` | 0.2 | PPO clip range |
| `n_steps` | 256 | Steps per rollout |
| `batch_size` | 2048 | Batch size |
| `minibatch_size` | 2048 | Mini-batch size for PPO update |
| `ppo_epochs` | 4 | PPO update epochs per rollout |
| `entropy_coef` | 0.07 | Entropy bonus coefficient |
| `max_steps` | 1000000 | Total training steps |
| `val_interval` | 20 | Validate every N rollouts |
| `save_interval` | 20 | Save checkpoint every N rollouts |

### Supervised Pretraining

| Parameter | Default | Description |
|-----------|---------|-------------|
| `supervised_pretrain` | true | Run supervised pretraining before PPO |
| `pretrain_epochs` | 20 | Pretraining epochs |
| `pretrain_lr` | 5e-5 | Pretraining learning rate |
| `pretrain_batch_size` | 512 | Pretraining batch size |
| `init_ckpt` | "" | Path to checkpoint; if set, skip pretraining and load directly |

### Scorer (`nima`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scorer_type` | gaic | Scorer type: `gaic` / `rank` / NIMA ONNX path |
| `gaic_repo_dir` | — | Path to GAIC repository |
| `gaic_ckpt` | — | Path to GAIC weights |
| `gaic_backbone` | mobilenetv2 | GAIC backbone architecture |
| `batch_size` | 128 | Scorer batch size |
| `real_score_interval` | 1 | Score every N env steps (geometric estimate otherwise) |

### Export (`export`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `onnx_opset` | 13 | ONNX opset version |
| `coreml_precision` | float16 | CoreML compute precision |

---

## 7. Training

Training has two stages controlled by `config.yaml`: **supervised pretraining** (optional) and **PPO reinforcement learning**.

### Option A: Load Pretrained Weights → PPO (Recommended)

If you already have a supervised pretrained checkpoint:

```yaml
# config.yaml
train:
  supervised_pretrain: false
  init_ckpt: "./Adacrop/logs/<run>/pretrain_best_iou.pth"
```

```bash
cd /path/to/LiveCompose
bash Adacrop/start_training.sh
```

Or directly:

```bash
python -u Adacrop/src/trainer.py
```

### Option B: From Scratch (Pretrain → PPO)

```yaml
# config.yaml
train:
  supervised_pretrain: true
  init_ckpt: ""
```

The trainer will first run supervised pretraining (BBox Head regression), then automatically transition to PPO.

### Option C: Pure RL (No Pretraining)

```yaml
# config.yaml
train:
  supervised_pretrain: false
  init_ckpt: ""
  apply_init_action_bias: true  # Encourage pan actions early on
```

> ⚠️ Convergence is significantly slower without pretraining. Not recommended.

### Training Outputs

Artifacts are saved to `Adacrop/logs/run_<timestamp>_<scorer>_<gpu>_<envs>/`:

```
logs/
└── run_20260513_234706_gaic_norm_gpu0_env128/
    ├── config.yaml                        # Config snapshot for this run
    ├── meta.json                          # Run metadata
    ├── train_log.jsonl                    # Training metrics log
    ├── pretrain_best_iou.pth              # Best pretrain model (IoU)
    ├── pretrain_last.pth                  # Latest pretrain model
    ├── ppo_latest.pth                     # Latest PPO model
    ├── ppo_best_train_reward.pth          # Best training reward model
    ├── ppo_best_val_final_score.pth       # Best validation score model
    └── final_model.pth                    # Final model after training
```

---

## 8. Inference

Use a trained model to crop a single image:

```bash
# Edit paths in Adacrop/use.py:
#   IMAGE_PATH = r"./Adacrop/data/your_image.jpg"
#   CKPT_PATH  = r"./Adacrop/logs/run_xxx/ppo_best_val_final_score.pth"

cd /path/to/LiveCompose
python Adacrop/use.py
```

**Outputs:**
- `Adacrop/output_img/<name>_crop.jpg` — Cropped result
- `Adacrop/output_img/<name>_traj.jpg` — Trajectory visualization (blue=start, yellow=intermediate, red=final)

---

## 9. Knowledge Distillation

To deploy on iOS, the ResNet50 teacher model is distilled into a lightweight MobileNetV3-Small student via two-stage distillation.

### Stage 1: BBox Head Distillation

The student's BBox regression head learns from both ground-truth annotations and the teacher's predictions.

### Stage 2: Actor Policy Distillation

The student's Actor learns to mimic the teacher's action probability distribution using KL divergence + cross-entropy loss, with bbox regularization.

### Run Distillation

```bash
cd Adacrop/distillation

# Full two-stage distillation
python train_mobilenet_distill.py \
    --teacher-ckpt ../logs/run_xxx/ppo_best_val_final_score.pth \
    --train-jsonl ../data/outpainted_dataset/training_pairs.jsonl \
    --val-json ../data/splits/val_mixed.json \
    --arch mobilenet_v3_small \
    --bbox-epochs 5 \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-4 \
    --temperature 2.0

# Skip Stage 1 (if student already has a trained BBox head)
python train_mobilenet_distill.py \
    --teacher-ckpt ../logs/run_xxx/ppo_best_val_final_score.pth \
    --skip-bbox-stage \
    --resume-student runs/xxx/student_bbox_stage1_best.pth

# Resume from a checkpoint
python train_mobilenet_distill.py \
    --teacher-ckpt ../logs/run_xxx/ppo_best_val_final_score.pth \
    --resume-student runs/xxx/student_last.pth
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher-ckpt` | `../ppo_best_val_final_score.pth` | Teacher model checkpoint |
| `--arch` | `mobilenet_v3_small` | Student architecture (`mobilenet_v3_small` / `mobilenet_v3_large`) |
| `--bbox-epochs` | 5 | Stage 1 epochs |
| `--epochs` | 10 | Stage 2 epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Stage 2 learning rate |
| `--temperature` | 2.0 | Softmax temperature for KL divergence |
| `--ce-weight` | 0.25 | Cross-entropy loss weight |
| `--patience` | 8 | Early-stop patience (epochs) |

### Distillation Outputs

```
distillation/runs/mobilenet_v3_small_twostage_<timestamp>/
├── metrics.csv                      # Full training metrics
├── student_bbox_stage1_best.pth     # Best Stage 1 model
├── student_bbox_stage1_last.pth     # Latest Stage 1 model
├── student_best.pth                 # Best Stage 2 model (policy agreement)
└── student_last.pth                 # Latest Stage 2 model
```

### Evaluate Student

```bash
# Compare student vs teacher performance
python evaluate_student.py --student-ckpt runs/xxx/student_best.pth
python evaluate_teacher.py --teacher-ckpt ../logs/run_xxx/ppo_best_val_final_score.pth
```

---

## 10. Model Export

### 10a. ONNX Export

Export the teacher model directly to ONNX format:

```bash
cd Adacrop

# Edit src/export.py paths, then run:
python src/export.py
```

Or programmatically:

```python
from src.export import export_onnx
export_onnx("path/to/ppo_best.pth", "adacrop.onnx")
```

**ONNX model spec:**

| Name | Shape | Description |
|------|-------|-------------|
| **Input** `img` | `[B, 3, 224, 224]` | Image tensor |
| **Input** `state` | `[B, 4]` | Normalized state `(cx, cy, w, h)` |
| **Output** `action_probs` | `[B, 7]` | Action probability distribution |
| **Output** `value` | `[B, 1]` | State value estimate |

7 actions: `left, right, up, down, zoom_in, zoom_out, stop`

### 10b. CoreML Export (Teacher)

Export the ResNet50 teacher model to CoreML format. This produces two `.mlpackage` bundles:

- **BBox model**: Takes a full image → predicts initial crop box coordinates
- **Actor model**: Takes a cropped image + state → predicts action probabilities

```bash
cd Adacrop/coreml_export

# Export teacher (ResNet50)
python export_teacher_coreml.py \
    --teacher-ckpt ../logs/run_xxx/ppo_best_val_final_score.pth \
    --out-dir ./teacher \
    --img-size 224 \
    --ios-target iOS16 \
    --precision float16
```

**Outputs:**

```
coreml_export/teacher/
├── AdacropTeacherBBox.mlpackage     # BBox prediction model
└── AdacropTeacherActor.mlpackage    # Actor policy model
```

**Teacher BBox model:**

| Name | Shape | Description |
|------|-------|-------------|
| **Input** `full_img` | `[1, 3, 224, 224]` | Full resized image |
| **Output** `bbox` | `[1, 4]` | Predicted crop box `(cx, cy, w, h)` normalized |

**Teacher Actor model:**

| Name | Shape | Description |
|------|-------|-------------|
| **Input** `crop_img` | `[1, 3, 224, 224]` | Cropped image region |
| **Input** `state` | `[1, 4]` | Current box state `(cx, cy, w, h)` |
| **Output** `action_probs` | `[1, 7]` | Action probabilities |

### 10c. CoreML Export (Student / MobileNet)

Export the distilled MobileNetV3 student model — **this is the recommended path for iOS deployment**:

```bash
cd Adacrop/coreml_export

# Export student (MobileNetV3-Small)
python export_student_coreml.py \
    --student-ckpt ../distillation/runs/xxx/student_best.pth \
    --out-dir ./student \
    --img-size 224 \
    --ios-target iOS16 \
    --precision float16
```

**Outputs:**

```
coreml_export/student/
├── AdacropStudentBBox.mlpackage     # Lightweight BBox model
└── AdacropStudentActor.mlpackage    # Lightweight Actor model
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher-ckpt` / `--student-ckpt` | — | Path to model checkpoint |
| `--out-dir` | `./teacher` or `./student` | Output directory |
| `--img-size` | 224 | Input image size |
| `--ios-target` | `iOS16` | Minimum iOS deployment target (`iOS15`/`iOS16`/`iOS17`/`iOS18`) |
| `--precision` | `float16` | Compute precision (`float16` / `float32`) |

### iOS Integration

The exported `.mlpackage` files can be directly added to your Xcode project. The inference pipeline on iOS is:

1. **BBox model**: Feed the camera frame → get initial crop box prediction
2. **Actor model**: Feed the cropped region + current state → get action probabilities
3. **Action loop**: Apply the highest-probability action, update the crop box, repeat until `stop`
4. **BoxCenterManager**: Tracks the target point using gyroscope data for smooth physical-level guidance

---

## FAQ

<details>
<summary><b>Q: CUDA extension compilation fails?</b></summary>

1. Ensure `CUDA_HOME` points to your CUDA installation directory
2. Ensure `-arch` in `make.sh` matches your GPU architecture
3. Ensure `make.sh` has Unix line endings (`:set ff=unix` in Vim)
4. Python must be 3.8; other versions require recompilation

</details>

<details>
<summary><b>Q: Out of GPU memory (OOM) during training?</b></summary>

Reduce these parameters in `config.yaml`:

```yaml
train:
  n_steps: 64          # Reduce rollout steps (was 256)
  batch_size: 512      # Reduce batch size (was 2048)
  minibatch_size: 256  # Reduce mini-batch (was 2048)
data:
  num_envs: 16         # Reduce parallel environments (was 128)
nima:
  batch_size: 32       # Reduce scorer batch size (was 128)
```

</details>

<details>
<summary><b>Q: How to train with my own dataset?</b></summary>

1. Prepare images with `[x1, y1, x2, y2]` bounding box annotations
2. Organize as JSON format (see [Section 5](#5-prepare-training-data))
3. Update `data.train_json` and `data.val_json` in `config.yaml`
4. Use `Adacrop/src/transform_dataset.py` to convert CUHK-format txt annotations

</details>

<details>
<summary><b>Q: How to switch the scoring model?</b></summary>

Modify the `nima` section in `config.yaml`:

```yaml
# GAIC scorer (default)
nima:
  scorer_type: gaic
  gaic_repo_dir: /path/to/Adacrop/GAIC
  gaic_ckpt: /path/to/Adacrop/GAIC/pretrained_models/GAIC-mobilenetv2-reddim16.pth
  gaic_backbone: mobilenetv2

# Or PairwiseRank scorer
nima:
  scorer_type: rank
  rank_ckpt: /path/to/rank_model.pth
  rank_backbone: resnet50
```

</details>

<details>
<summary><b>Q: How to monitor training progress?</b></summary>

Training logs are saved as JSONL in `<run_dir>/train_log.jsonl`. Key fields:

- `rollout`: Rollout number
- `mean_reward`: Average reward
- `best_reward`: Best reward so far
- `val_avg_final_score`: Validation average final score
- `gpu_memory_gb`: GPU memory usage

Quick commands:

```bash
# View last 10 log entries
tail -10 Adacrop/logs/run_*/train_log.jsonl | python -m json.tool

# Extract reward curve
cat Adacrop/logs/run_*/train_log.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if 'mean_reward' in d:
        print(d['rollout'], d['mean_reward'])
"
```

</details>

<details>
<summary><b>Q: CoreML export fails?</b></summary>

1. Ensure `coremltools` is installed: `pip install coremltools`
2. CoreML export is best run on **macOS** (some conversions work on Linux too)
3. If tracing fails, check that the checkpoint matches the expected model architecture
4. For student export, ensure the checkpoint was saved by `train_mobilenet_distill.py` (contains `arch` metadata)

</details>