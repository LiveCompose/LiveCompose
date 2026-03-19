import os
import json
from PIL import Image
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from env import CropEnv
from model import ActorCritic
from utils import PPOTrainer
from scorers import load_aesthetic_model
from transform_dataset import SmallCropDataset

def make_envs_from_json(cfg, init_model=None, init_device=None):
    recs = json.load(open(cfg.data['train_json'], 'r'))
    random.shuffle(recs)
    aesth = load_aesthetic_model()
    envs = []
    for rec in recs[: cfg.data['num_envs']]:
        img = Image.open(rec['img']).convert('RGB')
        envs.append(CropEnv(img, aesth, cfg, init_model=init_model, init_device=init_device))
    return envs

def make_pretrain_loaders(cfg):
    train_recs = json.load(open(cfg.data['train_json'], 'r'))
    val_recs   = json.load(open(cfg.data['val_json'],   'r'))

    sz = cfg.env["img_size"]
    train_ds = SmallCropDataset(train_recs, img_size=sz)
    val_ds   = SmallCropDataset(val_recs,   img_size=sz)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train['pretrain_batch_size'],
        shuffle=True,  num_workers=cfg.train.get('num_workers', 4)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train['pretrain_batch_size'],
        shuffle=False, num_workers=cfg.train.get('num_workers', 2)
    )
    return train_loader, val_loader

def _xywh_to_xyxy(box_xywh):
    cx, cy, w, h = [float(v) for v in box_xywh]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return [x1, y1, x2, y2]

def _box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 1e-8:
        return 0.0
    return inter / union

def evaluate_pretrain(model, val_loader, device):
    model.eval()
    total_mse = 0.0
    total_count = 0
    ious = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model.backbone_forward(imgs)
            loss = F.mse_loss(preds, targets, reduction="sum")
            total_mse += loss.item()
            total_count += imgs.size(0)

            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            for p, t in zip(preds_np, targets_np):
                p = [max(0.0, min(1.0, float(v))) for v in p]
                t = [max(0.0, min(1.0, float(v))) for v in t]
                iou = _box_iou_xyxy(_xywh_to_xyxy(p), _xywh_to_xyxy(t))
                ious.append(iou)

    val_mse = total_mse / max(1, total_count)
    val_mean_iou = float(sum(ious) / max(1, len(ious)))
    val_iou_03 = float(sum(i >= 0.3 for i in ious) / max(1, len(ious)))
    val_iou_05 = float(sum(i >= 0.5 for i in ious) / max(1, len(ious)))

    return {
        "val_mse": val_mse,
        "val_mean_iou": val_mean_iou,
        "val_iou_03": val_iou_03,
        "val_iou_05": val_iou_05,
    }

def supervised_pretrain(cfg, n_actions, device, run_dir):
    train_loader, val_loader = make_pretrain_loaders(cfg)

    net = ActorCritic(n_actions=n_actions).to(device)
    lr = float(cfg.train['pretrain_lr'])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_iou = -1.0
    best_path = os.path.join(run_dir, "pretrain_best_iou.pth")
    last_path = os.path.join(run_dir, "pretrain_last.pth")

    for epoch in range(cfg.train['pretrain_epochs']):
        net.train()
        total_loss = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = net.backbone_forward(imgs)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_pretrain(net, val_loader, device)

        print(
            f"[Pretrain] Epoch {epoch+1}/{cfg.train['pretrain_epochs']} "
            f"train_loss={train_loss:.4f} "
            f"val_mse={val_metrics['val_mse']:.4f} "
            f"val_iou={val_metrics['val_mean_iou']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                **val_metrics,
            },
            last_path
        )

        if val_metrics["val_mean_iou"] > best_iou:
            best_iou = val_metrics["val_mean_iou"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    **val_metrics,
                },
                best_path
            )
            print(f"* 保存最佳预训练模型: {best_path}")

    rl_model = ActorCritic(n_actions=n_actions).to(device)
    rl_model.backbone.load_state_dict(net.backbone.state_dict())
    rl_model.bbox_head.load_state_dict(net.bbox_head.state_dict())
    return rl_model

def infer_n_actions(env):
    # 支持 Gym API
    try:
        import gym
        if hasattr(env, "action_space") and isinstance(env.action_space, gym.spaces.Space):
            if hasattr(env.action_space, "n"):
                return env.action_space.n
            if hasattr(env.action_space, "shape"):
                return env.action_space.shape[0]
    except ImportError:
        pass

    # 支持自定义 CropEnv.actions 列表
    if hasattr(env, "actions"):
        return len(env.actions)

    raise RuntimeError(f"无法推断动作数：env 类型为 {type(env)}")

def _get_training_device(cfg):
    g = cfg.train.get("training_gpus", 0)
    if isinstance(g, (list, tuple)):
        g = g[0] if len(g) > 0 else 0
    try:
        idx = int(g)
        return torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _prepare_run_dir(cfg):
    save_root = cfg.train.get("save_dir", "./Adacrop/logs")
    os.makedirs(save_root, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    scorer_type = str(cfg.nima.get("scorer_type", "unknown"))
    train_gpu = str(cfg.train.get("training_gpus", "cpu"))
    num_envs = str(cfg.data.get("num_envs", "na"))
    norm_flag = "norm" if cfg.nima.get("normalize_to_nima_scale", False) else "raw"

    run_name = f"run_{ts}_{scorer_type}_{norm_flag}_gpu{train_gpu}_env{num_envs}"
    run_dir = os.path.join(save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _load_model_ckpt_if_any(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    兼容两种保存格式：
    1) torch.save({"model_state_dict": model.state_dict(), ...})
    2) torch.save(model.state_dict())
    """
    ckpt_path = str(ckpt_path)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"init_ckpt not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"* Loaded init_ckpt: {ckpt_path}")
    if missing:
        print(f"  - missing keys (show up to 10): {missing[:10]}")
    if unexpected:
        print(f"  - unexpected keys (show up to 10): {unexpected[:10]}")

    return model

def main():
    # 加载配置
    cfg = Config()
    device = _get_training_device(cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
    run_dir = _prepare_run_dir(cfg)
    
    # 先用一个临时环境推断动作数
    aesth = load_aesthetic_model()
    sample_recs = json.load(open(cfg.data['train_json'], 'r'))
    sample_img = Image.open(sample_recs[0]['img']).convert('RGB')
    tmp_env = CropEnv(sample_img, aesth, cfg, inference=True)
    n_actions = infer_n_actions(tmp_env)

    # 监督预训练 or 直接初始化
    init_ckpt = cfg.train.get("init_ckpt", None)
    if init_ckpt:
        # 直接加载，跳过预训练
        model = ActorCritic(n_actions=n_actions).to(device)
        _load_model_ckpt_if_any(model, init_ckpt, device)
        print("* Skip supervised pretrain because train.init_ckpt is set.")
    else:
        # 走 supervised pretrain
        if cfg.train.get('supervised_pretrain', False):
            model = supervised_pretrain(cfg, n_actions, device, run_dir)
        else:
            # 从头开始
            model = ActorCritic(n_actions=n_actions).to(device)

    if cfg.train.get("apply_init_action_bias", True) and (not init_ckpt) and (not cfg.train.get('supervised_pretrain', False)):
        model.init_action_bias(tmp_env.actions)

    # 用预训练模型作为 init_model 创建正式 envs
    init_model = model if cfg.train.get('supervised_pretrain', False) else None
    envs = make_envs_from_json(cfg, init_model=init_model, init_device=device)

    print(">>> 开始 PPO 训练，envs 数量：", len(envs), " n_actions：", n_actions)
    trainer = PPOTrainer(model, envs, cfg, log_dir=run_dir)
    trainer.train()

    final_path = os.path.join(run_dir, "final_model.pth")
    torch.save({"model_state_dict": model.state_dict()}, final_path)
    print(f"Model saved to {final_path}")

if __name__ == "__main__":
    main()