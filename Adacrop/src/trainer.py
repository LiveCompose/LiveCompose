import os
import json
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from env import CropEnv
from model import ActorCritic
from utils import PPOTrainer, load_aesthetic_model
from transform_dataset import SmallCropDataset

def make_envs_from_json(cfg):
    recs = json.load(open(cfg.data['train_json'], 'r'))
    aesth = load_aesthetic_model()
    envs = []
    for rec in recs[: cfg.data['num_envs']]:
        img = Image.open(rec['img']).convert('RGB')
        envs.append(CropEnv(img, aesth, cfg))
    return envs

def make_pretrain_loaders(cfg):
    train_recs = json.load(open(cfg.data['train_json'], 'r'))
    val_recs   = json.load(open(cfg.data['val_json'],   'r'))

    sz = cfg['env']['img_size']
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

def supervised_pretrain(cfg, n_actions, device):
    train_loader, _ = make_pretrain_loaders(cfg)

    # 用推断到的 n_actions 初始化网络
    net = ActorCritic(n_actions=n_actions).to(device)
    lr = float(cfg.train['pretrain_lr'])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(cfg.train['pretrain_epochs']):
        net.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = net.backbone_forward(imgs)   # [B,4]
            loss  = F.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Pretrain] Epoch {epoch+1}/{cfg.train['pretrain_epochs']}  "
              f"Loss: {total_loss/len(train_loader):.4f}")

    # 把预训练的 backbone & bbox_head 权重迁移到完整模型
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
        
def main():
    # 1. 加载配置
    #cfg = Config("config.yaml")
    cfg = Config()
    device = _get_training_device(cfg)
    torch.cuda.set_device(device)
    
    # 2. 构造环境并推断动作数
    envs = make_envs_from_json(cfg)
    n_actions = infer_n_actions(envs[0])

    # 3. 监督预训练 or 直接初始化
    if cfg.train.get('supervised_pretrain', False):
        model = supervised_pretrain(cfg, n_actions, device)
    else:
        model = ActorCritic(n_actions=n_actions).to(device)

    # 4. 强化训练（PPO）
    print(">>> 开始 PPO 训练，envs 数量：", len(envs), " n_actions：", n_actions)
    trainer = PPOTrainer(model, envs, cfg)
    trainer.train()

    # 5. 保存模型
    save_dir = cfg.train.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"Model saved to {save_dir}/final_model.pth")

if __name__ == "__main__":
    main()