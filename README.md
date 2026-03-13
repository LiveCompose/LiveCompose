# LiveCompose

![Code Size](https://img.shields.io/badge/Code_Size-10k%2B_Lines-green)
![Data Size](https://img.shields.io/badge/Data_Size-65GB-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Platform](https://img.shields.io/badge/Platform-iOS_%7C_Python-blueviolet)

## 📖 项目介绍 (Introduction)

**LiveCompose** 是一款旨在帮助普通用户拍摄出专业级构图照片的智能辅助系统。

虽然现代智能手机的拍照功能日益强大，但构图依然是许多用户的痛点。LiveCompose 通过集成先进的 AI 技术，实时分析取景画面，并提供动态的构图引导。不同于传统的静态九宫格辅助线，LiveCompose 能够主动“告诉”用户如何移动手机以获得最佳构图，甚至在对齐完美构图时自动完成拍摄。

我们的核心目标是成为您的“智能构图助手”，让每一次快门都能定格最美的瞬间。

> 更多详细技术细节请参考项目内的 [技术文档](Technic%20Profile.md)。

## ✨ 核心特性 (Key Features)

- **实时动态引导**: 基于 AI 模型实时分析画面，提供可视化的移动指引。
- **智能追踪系统**: 结合陀螺仪数据，实现物理级流畅的构图点追踪与吸附。
- **闭环拍摄体验**: “检测-追踪-引导-拍摄”全流程自动化，降低操作门槛。
- **美学评分驱动**: 基于 NIMA 美学评分模型的强化学习算法，确保构图的专业性。

## 🏗️ 系统架构 (System Architecture)

项目采用分层架构设计，确保高效处理与模块解耦：

1.  **数据采集层**: 基于 AVFoundation 和 CoreMotion 获取实时视频流与运动数据。
2.  **智能决策与追踪层**: 核心 `AdacropModel` (RL模型) 分析画面，`BoxCenterManager` 负责物理追踪。
3.  **核心协调层**: `CaptureViewModel` 管理状态机（等待、检测、追踪、拍照）。
4.  **视图与交互层**: 使用 SwiftUI 构建的用户界面，提供直观的视觉与触觉反馈。

## 🧠 模型与算法 (Model & Algorithms)

### 模型架构
- **Backbone**: ResNet50 (ImageNet Pretrained) 用于提取高层语义特征。
- **Actor-Critic**: 强化学习模型，Actor 决策裁剪动作，Critic 评估状态价值。
- **BBox Head**: 辅助监督预训练，加速收敛。

### 数据集构建
我们采用了创新的**双模型工作流**来构建高质量训练数据：
1.  **语义理解**: 使用 BLIP 模型生成图像描述。
2.  **内容生成**: 使用 Stable Diffusion v2 Inpaint 进行 Outpainting（扩图），增加数据多样性。
3.  **质量控制**: 结合 Canny 边缘检测与人工筛选，确保数据质量。

## 📂 项目结构 (Project Structure)

```
LiveCompose/
├── Adacrop/                # 核心裁剪模型与训练代码
│   ├── data/               # 训练数据与日志
│   ├── logs/               # 训练过程中的模型权重与日志
│   ├── z_inference.py      # 推理脚本
│   └── ...
├── NIMA/                   # 美学评分模型 (NIMA) 相关代码
├── src/                    # 辅助工具代码 (Dataset, Env, Utils等)
├── livecompose_outpainted_datasets/ # 扩图生成的数据集
├── PreProcess/             # 数据预处理脚本
├── Technic Profile.md      # 技术文档与答辩方案
└── README.md               # 项目说明文件
```

## 🚀 快速开始 (Getting Started)

### 环境依赖
- Python 3.x
- PyTorch
- Torchvision
- 其他依赖请参考代码中的 import

## 致谢 (Acknowledgements)

We gratefully acknowledge the following open-source repositories and their contributors:

- [GAIC-Pytorch](https://github.com/bo-zhang-cs/GAIC-Pytorch) by **bo-zhang-cs**, based on the works:
  - Hui Zeng, Lida Li, Zisheng Cao, Lei Zhang. *Reliable and Efficient Image Cropping: A Grid Anchor based Approach*. CVPR 2019.
  - Hui Zeng, Lida Li, Zisheng Cao, Lei Zhang. *Grid Anchor based Image Cropping: A New Benchmark and An Efficient Model*. IEEE TPAMI, 2020.

- [Grid-Anchor-based-Image-Cropping-Pytorch](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch) by **Hui Zeng**, official PyTorch implementation of the above works.

- [Neural Image Assessment](https://github.com/titu1994/neural-image-assessment) by **Somshubra Majumdar (titu1994)**, **Eren Sezener**, **Simon Brugman**, **Panayiotis Panayiotou**, based on:
  - Hossein Talebi, Peyman Milanfar. *NIMA: Neural Image Assessment*. IEEE Transactions on Image Processing, 2018.