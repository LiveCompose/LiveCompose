# LiveCompose

[简体中文](README_zh.md) | English

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LiveCompose-yellow)](https://huggingface.co/LiveCompose)
[![GitHub](https://img.shields.io/badge/GitHub-LiveCompose-black?logo=github)](https://github.com/LiveCompose)
[![App Store](https://img.shields.io/badge/App_Store-LiveCapture-blue)](https://apps.apple.com/cn/app/%E6%9E%84%E5%A6%99/id6754213088)
![Code Size](https://img.shields.io/badge/Code_Size-10k%2B_Lines-green)
![Model](https://img.shields.io/badge/Framework-PyTorch_%7C_CoreML-red)
![Platform](https://img.shields.io/badge/Platform-iOS_%7C_Python-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**LiveCompose** is a reinforcement learning-based AI-powered on-device composition assistance system. By analyzing the camera viewfinder in real time, combined with gyroscope tracking and aesthetic score-driven guidance, it actively directs users to move their phone for the optimal composition — ensuring every shot captures the best possible moment.

> Now available on the App Store: [LiveCapture (构妙)](https://apps.apple.com/cn/app/%E6%9E%84%E5%A6%99/id6754213088)

## Overview

Modern smartphones boast powerful cameras, yet composition remains the biggest pain point for everyday users. Traditional camera apps only provide static grid overlays — they can't tell you *how to move* your phone. LiveCompose bridges this gap by using AI models to analyze the scene in real time, compute optimal cropping regions, and fuse with device gyroscope data for physically smooth tracking. It guides users to align with target composition points and automatically captures the photo, completing a full "detect → track → guide → capture" pipeline.

Key innovations:
- **Real-time Dynamic Guidance**: AI models analyze the viewfinder and provide visual movement cues — not static reference lines.
- **Sensor-fused Tracking**: Gyroscope data enables physically smooth composition point tracking with magnetic snap behavior.
- **Aesthetic Score-Driven**: Reinforcement learning powered by NIMA/GAIC aesthetic scoring models ensures professional-quality compositions.
- **Efficient On-device Inference**: Knowledge distillation compresses a ResNet50 teacher model into MobileNet, deployed via CoreML on iOS.

## System Architecture

```
Data Acquisition Layer (AVFoundation + CoreMotion)
    ↓
Intelligent Decision & Tracking Layer (AdacropModel + BoxCenterManager)
    ↓
Core Coordination Layer (CaptureViewModel / State Machine)
    ↓
View & Interaction Layer (SwiftUI)
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Data Acquisition | CameraManager, MotionStabilityMonitor | Capture 60 FPS video frames and gyroscope data |
| Intelligent Decision | AdacropModel (CoreML), BoxCenterManager | AI composition analysis and physical tracking |
| Core Coordination | CaptureViewModel | State machine management (waiting → detecting → tracking → capturing) |
| View & Interaction | ContentView, UserGuidanceView | SwiftUI interface and haptic feedback |

## Model & Training

### Model Architecture

- **Backbone**: ResNet50 (ImageNet pretrained), extracting 2048-dim semantic features
- **Actor Branch**: MLP (2048+4 → 1024 → 512 → N), outputs action probability distribution
- **Critic Branch**: MLP (2048+4 → 1024 → 512 → 1), estimates state value
- **BBox Head**: MLP (2048 → 512 → 4), supervised pretraining regression head

Action space consists of 7 discrete actions: `left, right, up, down, zoom_in, zoom_out, stop`

### Training Pipeline

1. **Supervised Pretraining**: The BBox Head regresses human-annotated / teacher-model-generated crop boxes, enabling the Backbone to learn composition-relevant features.
2. **Reinforcement Learning (PPO)**: The Actor-Critic interacts with CropEnv, jointly optimized with rewards driven by NIMA/GAIC aesthetic score changes.
3. **Knowledge Distillation**: Two-stage distillation to MobileNetV3-Small — Stage 1 distills the BBox Head; Stage 2 distills the Actor policy.
4. **On-device Deployment**: The distilled student model is converted to CoreML format via `coremltools` and deployed on iOS.

### Dataset

We open-sourced the training outpainting dataset **LiveCompose-outpainted-17k**:

**[LiveCompose/LiveCompose-outpainted-17k](https://huggingface.co/datasets/LiveCompose/LiveCompose-outpainted-17k)** — ~17,000 outpainted images generated via Stable Diffusion Outpainting, with professionally annotated crop boxes.

The dataset was constructed using an innovative **dual-model workflow**:
1. **Semantic Understanding**: BLIP model generates image captions.
2. **Content Generation**: Stable Diffusion v2 Inpainting performs outpainting to increase data diversity.
3. **Quality Control**: Canny edge detection combined with manual screening ensures data quality.

## Project Structure

```
LiveCompose/
├── Adacrop/                    # Core cropping model & training
│   ├── src/                    # Training scripts (PPO, environment, model definitions)
│   ├── config.yaml             # Training hyperparameters
│   ├── distillation/           # Knowledge distillation (Teacher→MobileNet)
│   ├── coreml_export/          # CoreML conversion scripts
│   └── GAIC/                   # GAIC aesthetic scoring model
├── NIMA/                       # NIMA aesthetic scoring model
│   ├── train_*.py              # Multi-backbone training scripts
│   ├── evaluate_*.py           # Model evaluation scripts
│   └── weights/                # Pretrained weights
├── PreProcess/                 # Data preprocessing
│   ├── NIMA_Inception_Res/     # Inception-ResNet NIMA implementation
│   ├── dataset/                # Dataset indices
│   └── Article/                # References & paper notes
├── Technic Profile.md          # Detailed technical documentation
└── README.md
```

## Quick Start

👉 **See [QUICKSTART.md](QUICKSTART.md) for the complete guide**, covering environment setup, data preparation, training, knowledge distillation, inference testing, and ONNX / CoreML model export.

<details>
<summary><b>Brief Steps</b></summary>

```bash
# 1. Supervised pretraining + PPO reinforcement learning
cd Adacrop
python src/trainer.py

# 2. Knowledge distillation (ResNet50 → MobileNetV3-Small)
cd distillation
python train_mobilenet_distill.py --teacher-ckpt ../ppo_best_val_final_score.pth

# 3. Export CoreML (for iOS deployment)
cd coreml_export
python export_student_coreml.py --student-ckpt ../distillation/runs/student_best.pth
```

</details>

## Related Projects

| Platform | URL | Description |
|----------|-----|-------------|
| GitHub Organization | [github.com/LiveCompose](https://github.com/LiveCompose) | All open-source code |
| Hugging Face | [huggingface.co/LiveCompose](https://huggingface.co/LiveCompose) | Model weights & datasets |
| App Store | [LiveCapture (构妙)](https://apps.apple.com/cn/app/%E6%9E%84%E5%A6%99/id6754213088) | iOS app |

## Acknowledgements

We gratefully acknowledge the following open-source repositories and their contributors:

- [GAIC-Pytorch](https://github.com/bo-zhang-cs/GAIC-Pytorch) by **bo-zhang-cs**, based on the works:
  - Hui Zeng, Lida Li, Zisheng Cao, Lei Zhang. *Reliable and Efficient Image Cropping: A Grid Anchor based Approach*. CVPR 2019.
  - Hui Zeng, Lida Li, Zisheng Cao, Lei Zhang. *Grid Anchor based Image Cropping: A New Benchmark and An Efficient Model*. IEEE TPAMI, 2020.

- [Grid-Anchor-based-Image-Cropping-Pytorch](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch) by **Hui Zeng**, official PyTorch implementation of the above works.

- [Neural Image Assessment](https://github.com/titu1994/neural-image-assessment) by **Somshubra Majumdar (titu1994)**, **Eren Sezener**, **Simon Brugman**, **Panayiotis Panayiotou**, based on:
  - Hossein Talebi, Peyman Milanfar. *NIMA: Neural Image Assessment*. IEEE Transactions on Image Processing, 2018.

- [ProCrop: Learning Aesthetic Image Cropping from Professional Compositions](https://arxiv.org/abs/2505.22490) — 2025.
  Our dataset construction approach (generating cropping training pairs via diffusion-model outpainting) was inspired by this work.
