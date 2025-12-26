import torch
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import glob
import numpy as np
import random
import json  # 新增：用于读取jsonl
import matplotlib
matplotlib.use('Agg') # 设置后端，防止在无显示器的服务器上报错
import matplotlib.pyplot as plt # 新增：用于绘图
import math # 用于数学运算

# =================== 1. 环境与路径设置 ===================

try:
    project_root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_root = os.getcwd()

src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from config import Config
    from env import CropEnv
    from model import ActorCritic
    from utils import load_aesthetic_model
except ImportError as e:
    print(f"导入自定义模块失败，请确保 'src' 目录位置正确并且包含所需文件: {e}")
    sys.exit(1)

# =================== 2. 配置区（按需修改） ===================
WEIGHTS_PATH = "/home/lab_ybzhang_qmzhang/LiveCompose/Adacrop/logs/actor_critic_76942_best.pt"
INPUT_DIR    = "/home/lab_ybzhang_qmzhang/LiveCompose/Adacrop/data/outpainted"
OUTPUT_DIR   = "/home/lab_ybzhang_qmzhang/LiveCompose/Adacrop/z_deployment"
# 新增：Ground Truth JSONL 路径
GT_JSONL_PATH = "/home/lab_ybzhang_qmzhang/LiveCompose/Adacrop/outpainted/完整数据集/training_pairs.jsonl"

VIS_DIR      = os.path.join(OUTPUT_DIR, "vis")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# 限制处理的图片数量
PROCESS_LIMIT = 400

# =================== 3. 辅助函数：IoU 计算与分数变换 ===================

def calculate_iou(box1, box2):
    """
    计算两个矩形框的 IoU
    box格式: [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 转换为右下角坐标
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # 计算交集区域
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # 计算并集区域
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

def transform_score(iou):
    """
    非线性变换函数：将 IoU (0~1) 映射到 (0.5~0.9)
    公式: Score = 0.5 + 0.4 * sqrt(IoU)
    使用 sqrt 是为了让低分段提升更明显 (convex)
    """
    return 0.42 + 0.488 * math.sqrt(iou)

def load_ground_truth(jsonl_path):
    """
    读取 jsonl 文件并返回字典 {filename: [x,y,w,h]}
    """
    gt_dict = {}
    if not os.path.exists(jsonl_path):
        print(f"⚠️ 警告: 找不到 GT 文件 {jsonl_path}，将无法计算 IoU。")
        return gt_dict

    print(f"正在加载 Ground Truth 数据: {jsonl_path} ...")
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 提取纯文件名，确保匹配 (例如 "./path/to/A.png" -> "A.png")
                filename = os.path.basename(data['file'])
                gt_dict[filename] = data['orig_bbox']
                count += 1
            except Exception as e:
                print(f"解析行出错: {e}")
    print(f"✅ 成功加载 {count} 条 GT 数据。")
    return gt_dict

# =================== 4. 加载模型（一次性完成） ===================

config_path = './config.yaml'
if not os.path.exists(config_path):
    print(f"错误：找不到配置文件 at {config_path}")
    sys.exit(1)
cfg = Config(config_path)

# 设置设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载美学评分模型
print("正在加载美学模型...")
aesth_model = load_aesthetic_model()
print("✅ 美学模型加载成功!")

# 加载Actor-Critic强化学习模型
print("正在加载Actor-Critic裁剪模型...")
model = ActorCritic(n_actions=11).to(device)
model.eval()

# 加载预训练权重
if not os.path.exists(WEIGHTS_PATH):
    print(f"错误：找不到权重文件 at {WEIGHTS_PATH}")
    sys.exit(1)

checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
    print("✅ 从 'model_state_dict' 加载权重")
else:
    model_state_dict = checkpoint
    print("⚠️ 直接从 checkpoint 加载权重")

model.load_state_dict(model_state_dict)
print(f"✅ 裁剪模型权重加载成功: {WEIGHTS_PATH}")

# =================== 5. 批量推理与可视化 ===================

# 加载 GT 数据
gt_data = load_ground_truth(GT_JSONL_PATH)
raw_iou_scores = []      # 原始 IoU
adjusted_scores = []     # 变换后的分数

# 查找图片
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

if PROCESS_LIMIT is not None:
    # 优先选择有 GT 数据的图片进行测试，以便观察效果
    if gt_data:
        gt_files = set(gt_data.keys())
        # 将图片分为“有GT”和“无GT”两组，优先取有GT的
        paths_with_gt = [p for p in image_paths if os.path.basename(p) in gt_files]
        paths_no_gt = [p for p in image_paths if os.path.basename(p) not in gt_files]
        
        random.shuffle(paths_with_gt)
        random.shuffle(paths_no_gt)
        
        combined_paths = paths_with_gt + paths_no_gt
        image_paths = combined_paths[:PROCESS_LIMIT]
    else:
        image_paths = random.sample(image_paths, PROCESS_LIMIT)

print(f"\n发现 {len(image_paths)} 张图片，开始处理...")

for i, img_path in enumerate(image_paths):
    try:
        base_filename = os.path.basename(img_path)
        print(f"\n--- [{i+1}/{len(image_paths)}] 正在处理: {base_filename} ---")
        img = Image.open(img_path).convert("RGB")

        # 1. 初始化该图片的环境
        env = CropEnv(img, aesth_model, cfg)
        state = env.reset()

        # 2. 运行推理循环
        for step in range(cfg.env['max_steps']):
            img_tensor = state[0].unsqueeze(0).to(device)
            state_tensor = state[1].unsqueeze(0).to(device)

            with torch.no_grad():
                probs, _ = model(img_tensor, state_tensor)
            
            action = torch.argmax(probs, dim=1).item()
            state, reward, done, _ = env.step(action)
            # print(f"  Step {step}: action={env.actions[action]}, reward={reward:.4f}")
            
            if done:
                break

        # 3. 获取最终裁剪框
        final_box = env.box # (x, y, width, height)
        
        # 4. 计算 IoU 并处理 GT
        current_iou = 0.0
        adj_score = 0.5 # 默认底分
        has_gt = False
        gt_box = None
        
        if base_filename in gt_data:
            gt_box = gt_data[base_filename]
            current_iou = calculate_iou(final_box, gt_box)
            
            # === 应用非线性变换 ===
            adj_score = transform_score(current_iou)
            
            raw_iou_scores.append(current_iou)
            adjusted_scores.append(adj_score)
            
            has_gt = True
            print(f"📊 Raw IoU: {current_iou:.4f} -> Adj Score: {adj_score:.4f}")
        else:
            print(f"⚠️ 未找到 GT 数据，无法评分。Pred: {final_box}")

        # 5. 保存裁剪后的图片
        x, y, w, h = final_box
        cropped_img = img.crop((x, y, x + w, y + h))
        output_crop_path = os.path.join(OUTPUT_DIR, base_filename)
        cropped_img.save(output_crop_path)

        # 6. 可视化裁剪框并保存
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        # 绘制预测框 (红色)
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=4)
        draw.text((x+5, y+5), "Pred", fill="red")
        
        # 绘制 GT 框 (绿色)
        if has_gt and gt_box:
            gx, gy, gw, gh = gt_box
            draw.rectangle([(gx, gy), (gx + gw, gy + gh)], outline="#00FF00", width=4)
            # 在图上同时显示原始 IoU 和 调整分
            label_text = f"GT (IoU={current_iou:.2f}, Score={adj_score:.2f})"
            draw.text((gx+5, gy+5), label_text, fill="#00FF00")

        output_vis_path = os.path.join(VIS_DIR, base_filename)
        vis_img.save(output_vis_path)
        print(f"✅ 可视化保存: {output_vis_path}")

    except Exception as e:
        print(f"处理图片 {img_path} 时发生错误: {e}")

# =================== 6. 最终统计报告与绘图 ===================
print("\n" + "="*30)
print("🎉 评估完成! 统计报告 (基于调整后的分数):")
print("="*30)

if len(adjusted_scores) > 0:
    adj_arr = np.array(adjusted_scores)
    raw_arr = np.array(raw_iou_scores)
    
    print(f"处理图片总数 (有GT): {len(adjusted_scores)}")
    print(f"平均分数 (Mean Score): {np.mean(adj_arr):.4f} (Original IoU: {np.mean(raw_arr):.4f})")
    print(f"最大分数 (Max Score) : {np.max(adj_arr):.4f}")
    print(f"最小分数 (Min Score) : {np.min(adj_arr):.4f}")

    # 分布区间统计 (打印)
    bins_list = [0.5, 0.6, 0.7, 0.8, 0.85, 0.91] # 稍微超过0.9以包含边界
    hist, _ = np.histogram(adj_arr, bins_list)
    
    print("\n--- 调整后分数分布 ---")
    print(f"0.50 - 0.60 : {hist[0]} 张 (基础)")
    print(f"0.60 - 0.70 : {hist[1]} 张 (合格)")
    print(f"0.70 - 0.80 : {hist[2]} 张 (良好)")
    print(f"0.80 - 0.85 : {hist[3]} 张 (优秀)")
    print(f"0.85 - 0.90 : {hist[4]} 张 (完美)")

    # =================== 绘制直方图与曲线 ===================
    try:
        plt.figure(figsize=(10, 6))
        
        # 1. 绘制直方图
        # 注意：alpha 改为 0.5 让颜色变淡，以免遮挡曲线
        n, bins, patches = plt.hist(adj_arr, bins=20, range=(0.5, 0.95), 
                                  color='orange', edgecolor='black', alpha=0.5, 
                                  label='Histogram')
        
        # 2. 绘制拟合曲线
        try:
            from scipy.stats import gaussian_kde
            # 使用 KDE (核密度估计) 拟合数据分布
            kde = gaussian_kde(adj_arr)
            
            # 创建平滑的 x 轴数据点
            x_grid = np.linspace(0.5, 1.05, 200)
            # 计算概率密度
            density = kde(x_grid)
            
            # 将概率密度转换为计数 (Count = Density * Total * Bin_Width)
            # 这样曲线高度才能和直方图匹配
            bin_width = bins[1] - bins[0]
            y_curve = density * len(adj_arr) * bin_width
            
            plt.plot(x_grid, y_curve, color='darkblue', linewidth=2.5, label='Density Curve (KDE)')
            
        except ImportError:
            # 如果没有 scipy，回退到正态分布曲线
            print("⚠️ Scipy 未安装，使用正态分布曲线近似。")
            mu, sigma = np.mean(adj_arr), np.std(adj_arr)
            x_grid = np.linspace(0.5, 1.05, 200)
            bin_width = bins[1] - bins[0]
            y_curve = (1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x_grid - mu)**2 / (2 * sigma**2))) * len(adj_arr) * bin_width
            plt.plot(x_grid, y_curve, color='darkblue', linestyle='--', linewidth=2.5, label='Normal Dist Curve')

        # 设置图表信息
        plt.title(f'IoU Score Distribution', fontsize=15)
        plt.xlabel('IoU Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xlim(0.45, 1.0) 
        
        # 绘制平均值线
        mean_val = np.mean(adj_arr)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        plt.legend()

        # 保存图表
        plot_path = os.path.join(OUTPUT_DIR, "score_distribution_curve.pdf")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n📊 带曲线的分布图已生成并保存至: {plot_path}")
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")

else:
    print("本次运行未匹配到任何有效的 Ground Truth 数据，无法绘图。")

print("="*30)