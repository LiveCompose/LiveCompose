# LiveCompose

简体中文 | [English](README.md)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LiveCompose-yellow)](https://huggingface.co/LiveCompose)
[![GitHub](https://img.shields.io/badge/GitHub-LiveCompose-black?logo=github)](https://github.com/LiveCompose)
[![App Store](https://img.shields.io/badge/App_Store-构妙_LiveCapture-blue)](https://apps.apple.com/cn/app/%E6%9E%84%E5%A6%99/id6754213088)
![Code Size](https://img.shields.io/badge/Code_Size-10k%2B_Lines-green)
![Model](https://img.shields.io/badge/Framework-PyTorch_%7C_CoreML-red)
![Platform](https://img.shields.io/badge/Platform-iOS_%7C_Python-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

构妙 LiveCompose 是一款基于强化学习的 AI 端侧智能构图辅助系统。通过实时分析取景画面，结合陀螺仪追踪与美学评分驱动，主动引导用户移动手机以获得最佳构图，让每一次快门都定格最美的瞬间。

> 已上线 App Store：[构妙 LiveCapture](https://apps.apple.com/cn/app/%E6%9E%84%E5%A6%99/id6754213088)

## 项目概述

现代智能手机拍照功能日益强大，但构图依然是普通用户的最大痛点。传统相机仅提供静态九宫格辅助线，无法主动告知用户"如何移动手机"。LiveCompose 通过 AI 模型实时分析画面内容，计算最优构图区域，并结合设备陀螺仪实现物理级流畅追踪，引导用户对齐目标点后自动拍摄，实现"检测-追踪-引导-拍摄"全流程闭环。

核心创新点：
- **实时动态引导**：AI 模型分析画面，提供可视化移动指引，而非静态参考线
- **传感器融合追踪**：结合陀螺仪数据，实现物理级流畅的构图点追踪与磁性吸附
- **美学评分驱动**：基于 NIMA/GAIC 美学评分模型的强化学习，确保构图专业性
- **端侧高效推理**：通过知识蒸馏将 ResNet50 教师模型压缩为 MobileNet，经 CoreML 部署至 iOS

## 系统架构

```
数据采集层 (AVFoundation + CoreMotion)
    ↓
智能决策与追踪层 (AdacropModel + BoxCenterManager)
    ↓
核心协调层 (CaptureViewModel / 状态机)
    ↓
视图与交互层 (SwiftUI)
```

| 层级 | 模块 | 职责 |
|------|------|------|
| 数据采集 | CameraManager, MotionStabilityMonitor | 获取 60FPS 视频帧与陀螺仪数据 |
| 智能决策 | AdacropModel (CoreML), BoxCenterManager | AI 构图分析与物理追踪 |
| 核心协调 | CaptureViewModel | 状态机管理（等待→检测→追踪→拍照） |
| 视图交互 | ContentView, UserGuidanceView | SwiftUI 界面与触觉反馈 |

## 模型与训练

### 模型架构

- **Backbone**: ResNet50 (ImageNet 预训练)，提取 2048 维语义特征
- **Actor 分支**: MLP (2048+4 → 1024 → 512 → N)，输出动作概率分布
- **Critic 分支**: MLP (2048+4 → 1024 → 512 → 1)，估计状态价值
- **BBox Head**: MLP (2048 → 512 → 4)，监督预训练专用回归头

动作空间共 7 个离散动作：`left, right, up, down, zoom_in, zoom_out, stop`

### 训练流程

1. **监督预训练**：使用 BBox Head 回归人工标注/教师模型生成的裁剪框，让 Backbone 学习构图相关特征
2. **强化学习 (PPO)**：在 CropEnv 中与环境交互，Actor-Critic 联合优化，奖励由 NIMA/GAIC 美学分数变化驱动
3. **知识蒸馏**：两阶段蒸馏至 MobileNetV3-Small —— Stage 1 蒸馏 BBox Head，Stage 2 蒸馏 Actor 策略
4. **端侧部署**：将蒸馏后的学生模型通过 `coremltools` 转换为 CoreML 格式，部署至 iOS

### 数据集

我们开源了训练用的扩图数据集 **LiveCompose-outpainted-17k**：

**[LiveCompose/LiveCompose-outpainted-17k](https://huggingface.co/datasets/LiveCompose/LiveCompose-outpainted-17k)** — 约 17,000 张通过 Stable Diffusion Outpainting 生成的扩图，附带专业标注裁剪框。

数据集构建采用了创新的**双模型工作流**：
1.  **语义理解**: 使用 BLIP 模型生成图像描述。
2.  **内容生成**: 使用 Stable Diffusion v2 Inpaint 进行 Outpainting（扩图），增加数据多样性。
3.  **质量控制**: 结合 Canny 边缘检测与人工筛选，确保数据质量。

## 项目结构

```
LiveCompose/
├── Adacrop/                    # 核心裁剪模型与训练
│   ├── src/                    # 训练脚本 (PPO, 环境, 模型定义)
│   ├── config.yaml             # 训练超参数配置
│   ├── distillation/           # 知识蒸馏 (Teacher→MobileNet)
│   ├── coreml_export/          # CoreML 转换脚本
│   └── GAIC/                   # GAIC 美学评分模型
├── NIMA/                       # NIMA 美学评分模型
│   ├── train_*.py              # 多种 Backbone 训练脚本
│   ├── evaluate_*.py           # 模型评估脚本
│   └── weights/                # 预训练权重
├── PreProcess/                 # 数据预处理
│   ├── NIMA_Inception_Res/     # Inception-ResNet NIMA 实现
│   ├── dataset/                # 数据集索引
│   └── Article/                # 参考文献与论文笔记
├── Technic Profile.md          # 详细技术文档
└── README.md
```

## 快速开始

👉 **完整指南请参考 [QUICKSTART.md](QUICKSTART.md)**（English），涵盖环境搭建、数据准备、训练、知识蒸馏、推理测试、ONNX / CoreML 模型导出的全流程。

<details>
<summary><b>简要步骤</b></summary>

```bash
# 1. 监督预训练 + PPO 强化学习
cd Adacrop
python src/trainer.py

# 2. 知识蒸馏 (ResNet50 → MobileNetV3-Small)
cd distillation
python train_mobilenet_distill.py --teacher-ckpt ../ppo_best_val_final_score.pth

# 3. 导出 CoreML (for iOS deployment)
cd coreml_export
python export_student_coreml.py --student-ckpt ../distillation/runs/student_best.pth
```

</details>

## 关联项目

| 平台 | 地址 | 说明 |
|------|------|------|
| GitHub 组织 | [github.com/LiveCompose](https://github.com/LiveCompose) | 全部开源代码 |
| Hugging Face | [huggingface.co/LiveCompose](https://huggingface.co/LiveCompose) | 模型权重与数据集 |
| App Store | [构妙 LiveCapture](https://apps.apple.com/cn/app/%E6%9E%84%E5%A6%99/id6754213088) | iOS 应用 |

## 致谢 (Acknowledgements)

We gratefully acknowledge the following open-source repositories and their contributors:

- [GAIC-Pytorch](https://github.com/bo-zhang-cs/GAIC-Pytorch) by **bo-zhang-cs**, based on the works:
  - Hui Zeng, Lida Li, Zisheng Cao, Lei Zhang. *Reliable and Efficient Image Cropping: A Grid Anchor based Approach*. CVPR 2019.
  - Hui Zeng, Lida Li, Zisheng Cao, Lei Zhang. *Grid Anchor based Image Cropping: A New Benchmark and An Efficient Model*. IEEE TPAMI, 2020.

- [Grid-Anchor-based-Image-Cropping-Pytorch](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch) by **Hui Zeng**, official PyTorch implementation of the above works.

- [Neural Image Assessment](https://github.com/titu1994/neural-image-assessment) by **Somshubra Majumdar (titu1994)**, **Eren Sezener**, **Simon Brugman**, **Panayiotis Panayiotou**, based on:
  - Hossein Talebi, Peyman Milanfar. *NIMA: Neural Image Assessment*. IEEE Transactions on Image Processing, 2018.

- [ProCrop: Learning Aesthetic Image Cropping from Professional Compositions](https://arxiv.org/abs/2505.22490) — 2025.
  Our dataset construction approach (generating cropping training pairs via diffusion-model outpainting) was inspired by this work.
