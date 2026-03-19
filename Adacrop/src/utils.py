import os
import time
import json
import numpy as np
import math
import socket
import pickle
import io
import gc
from collections import deque, namedtuple
from PIL import Image  
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models
import torchvision.transforms as T

try:
    from Adacrop.src.config import Config
except ModuleNotFoundError:
    try:
        from src.config import Config
    except ModuleNotFoundError:
        from config import Config
try:
    from scorers import load_aesthetic_model
    from env import CropEnv
except ModuleNotFoundError:
    from Adacrop.src.scorers import load_aesthetic_model
    from Adacrop.src.env import CropEnv

# 轨迹存储 & GAE 计算
Rollout = namedtuple(
    'Rollout',
    ['states', 'actions', 'logps', 'values', 'rewards', 'dones']
)

def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards: [T, N]  
    values:  [T+1, N]  
    dones:   [T, N] (done flag, 1 if episode ended)
    返回：advantages [T, N], returns [T, N]
    """
    T, N = rewards.shape
    advs = np.zeros((T, N), dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        advs[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    returns = advs + values[:-1]
    return advs, returns


class TrainerMixin:
    def __init__(self, model, cfg, log_dir="./Adacrop/logs"):
        if os.path.basename(log_dir).startswith("run_"):
            self.run_dir = log_dir
        else:
            run_name = time.strftime("run_%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        training_gpu = cfg.train.get("training_gpus", 0)
        print(f" 单A800优化模式: GPU {training_gpu}")
        
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{training_gpu}")
        else:
            device = torch.device("cpu")
        model.to(device)
        self.device = device
        self.training_gpu = training_gpu
        
        self.model = model
        self.cfg = cfg
        self.log_dir = self.run_dir
        self.start_time = time.time()

        self._save_run_config()
        if self.device.type == "cuda":
            self._warmup_gpu()

    def log_metrics(self, step, **kwargs):
        log_path = os.path.join(self.run_dir, "train_log.jsonl")

        if self.device.type == "cuda":
            gpu_memory = torch.cuda.memory_allocated(self.training_gpu) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(self.training_gpu) / 1024**3
        else:
            gpu_memory = 0.0
            gpu_reserved = 0.0

        record = {
            "step": step,
            "gpu_memory_gb": round(gpu_memory, 2),
            "gpu_reserved_gb": round(gpu_reserved, 2),
            "training_time": round(time.time() - self.start_time, 2),
            **kwargs
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
    def _warmup_gpu(self):
        """预热GPU，优化内存分配模式"""
        print("🔥 预热GPU内存分配...")
        
        # 创建一些大张量来预分配内存池
        try:
            with torch.cuda.device(self.device):
                # 预热不同大小的张量分配
                warmup_tensors = []
                for size in [64, 128, 256, 512]:
                    tensor = torch.randn(size, 3, 224, 224, device=self.device)
                    warmup_tensors.append(tensor)
                
                # 测试模型前向传播
                test_img = torch.randn(32, 3, 224, 224, device=self.device)
                test_state = torch.randn(32, 4, device=self.device)
                
                with torch.no_grad():
                    probs, values = self.model(test_img, test_state)
                
                print(f"✔ GPU预热完成: {probs.shape}, {values.shape}")
                
                # 清理预热张量
                del warmup_tensors, test_img, test_state, probs, values
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"! GPU预热失败: {e}")
    
    def _save_run_config(self):
        import yaml

        cfg_path = os.path.join(self.run_dir, "config.yaml")
        meta_path = os.path.join(self.run_dir, "meta.json")

        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.cfg._cfg, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"! 保存 config.yaml 失败: {e}")

        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_gpu": self.training_gpu,
            "device": str(self.device),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def save_model(self, filename="checkpoint.pth", tag=None, extra=None):
        path = os.path.join(self.run_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": getattr(self, "optimizer", None) and self.optimizer.state_dict(),
            "timestamp": time.time(),
            "training_time": time.time() - self.start_time,
            "tag": tag,
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        print(f"✔ 模型已保存: {path}")

class A2CTrainer(TrainerMixin):
    def __init__(self, model, envs, cfg):
        super().__init__(model, cfg)
        self.envs = envs
        self.n_envs = len(envs)
        self.gamma = cfg.train.get("gamma", 0.99)
        self.ent_coef = cfg.train.get("entropy_coef", 0.01)
        self.vf_coef  = cfg.train.get("value_loss_coef", 0.5)
        self.max_grad_norm = cfg.train.get("max_grad_norm", 0.5)
        raw_lr = cfg.train["lr"]
        try:
            lr = float(raw_lr)
        except:
            raise ValueError(f"config.train.lr 必须是数值类型 or 能转 float 的字符串，当前是 {raw_lr!r}")
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self):
        step = 0
        states = [env.reset() for env in self.envs]

        while step < self.cfg.train["max_steps"]:
            # 1. 收集 rollout
            imgs_list, st_list = [], []
            acts_list, logps_list, vals_list = [], [], []
            rewards_list, dones_list = [], []

            for _ in range(self.cfg.train["n_steps"]):
                imgs = torch.stack([s[0] for s in states]).to(self.device)
                st   = torch.stack([s[1] for s in states]).to(self.device)

                with torch.no_grad():
                    probs, values = self.model(imgs, st)
                dist    = torch.distributions.Categorical(probs)
                actions = dist.sample()
                logps   = dist.log_prob(actions)

                next_states, rewards, dones = [], [], []
                for i, env in enumerate(self.envs):
                    ns, r, done, _ = env.step(actions[i].item())
                    next_states.append(env.reset() if done else ns)
                    rewards.append(r); dones.append(done)

                # 存储
                imgs_list.append(imgs);       st_list.append(st)
                acts_list.append(actions);    logps_list.append(logps)
                vals_list.append(values.squeeze())
                rewards_list.append(torch.tensor(rewards))
                dones_list.append(torch.tensor(dones, dtype=torch.float32))

                states = next_states
                step  += 1

            # 2. GAE 与 Returns
            # 把最后一步 value 加到 vals_list 尾
            with torch.no_grad():
                _, next_val = self.model(imgs_list[-1].to(self.device), st_list[-1].to(self.device))
            vals_all = vals_list + [next_val.squeeze().cpu()]
            rewards = torch.stack(rewards_list).numpy()  # [T, N]
            dones   = torch.stack(dones_list).numpy()    # [T, N]

            advs, returns = compute_gae(
                rewards, 
                torch.stack(vals_all).numpy(), 
                dones, 
                self.gamma, 
                self.cfg.train.get("lam", 0.95)
            )

            # 3. 扁平化到 GPU Tensor
            imgs_batch = torch.cat(imgs_list).to(self.device)           # (T*N, C, H, W)
            st_batch   = torch.cat(st_list).to(self.device)             # (T*N, dim)
            acts_batch = torch.cat(acts_list).to(self.device)           # (T*N,)
            old_logps  = torch.cat(logps_list).to(self.device)          # (T*N,)
            advs_t     = torch.tensor(advs.flatten(), dtype=torch.float32).to(self.device)
            ret_t      = torch.tensor(returns.flatten(), dtype=torch.float32).to(self.device)

            # 4. 单步更新
            probs, vals = self.model(imgs_batch, st_batch)
            dist    = torch.distributions.Categorical(probs)
            logps   = dist.log_prob(acts_batch)
            entropy = dist.entropy().mean()

            policy_loss = -(advs_t * logps).mean()
            value_loss  = F.mse_loss(vals.squeeze(), ret_t)
            loss        = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # 日志与模型保存
            if step % self.cfg.train.get("log_interval", 1000) == 0:
                self.log_metrics(
                    step,
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                    entropy=entropy.item()
                )
                print(f"[A2C] step {step} | pl {policy_loss:.3f} | vl {value_loss:.3f} | ent {entropy:.3f}")

            if step % self.cfg.train.get("save_interval", 10000) == 0:
                self.save_model()
'''
class PPOTrainer(A2CTrainer): # DDP版本
    def __init__(self, model, envs, cfg):
        super().__init__(model, envs, cfg)
        
        # 初始化分布式训练
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # 使用DDP而不是DataParallel
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        model = model.to(device)
        self.model = DDP(model, device_ids=[local_rank])
        
        self.clip_param = cfg.train.get("clip_param", 0.2)
        self.ppo_epochs = cfg.train.get("ppo_epochs", 4)
        self.batch_size = cfg.train.get("batch_size", 768)  # 使用完整batch_size
        self.scaler = GradScaler()
        
        # 更大的minibatch以提高GPU利用率
        self.minibatch_size = min(256, self.batch_size)  # 增加到256
        
        lr = float(cfg.train["lr"])
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.best_reward = -float("inf")

    def train(self):
        print(">>> 进入高效PPO训练")
        device = next(self.model.parameters()).device
        step = 0
        
        # 预分配张量以减少内存分配开销
        states = [env.reset() for env in self.envs]
        states = [(s[0].to(device, non_blocking=True), s[1].to(device, non_blocking=True)) for s in states]

        while step < self.cfg.train["max_steps"]:
            # 收集数据阶段 - 减少频繁的GPU-CPU传输
            rollout_data = self._collect_rollout(states, device)
            states = rollout_data['next_states']
            step += rollout_data['steps']
            
            # 批量处理优势和回报
            advs, returns = self._compute_advantages(rollout_data)
            mean_reward = np.mean(returns)
            
            # 高效的多轮更新
            self._ppo_update(rollout_data, advs, returns, device)
            
            # 减少日志频率
            if step % (self.cfg.train.get("log_interval", 1000) * 4) == 0:
                self._log_progress(step, mean_reward)
            
            if step % self.cfg.train.get("save_interval", 500) == 0:
                self.save_model()
                
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save_model(tag="best")

    def _collect_rollout(self, states, device):
        """高效的rollout收集，减少内存分配"""
        n_steps = self.cfg.train["n_steps"]
        n_envs = len(self.envs)
        
        # 预分配张量
        imgs_batch = torch.zeros((n_steps, n_envs, 3, 224, 224), device=device)
        states_batch = torch.zeros((n_steps, n_envs, 4), device=device)
        actions_batch = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
        logps_batch = torch.zeros((n_steps, n_envs), device=device)
        values_batch = torch.zeros((n_steps, n_envs), device=device)
        rewards_batch = torch.zeros((n_steps, n_envs), device=device)
        dones_batch = torch.zeros((n_steps, n_envs), device=device)
        
        for t in range(n_steps):
            imgs = torch.stack([s[0] for s in states])
            st = torch.stack([s[1] for s in states])
            
            with torch.no_grad():
                probs, values = self.model(imgs, st)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                logps = dist.log_prob(actions)
            
            # 批量环境步进
            next_states, rewards, dones = self._batch_env_step(actions.cpu().numpy())
            
            # 存储到预分配的张量
            imgs_batch[t] = imgs
            states_batch[t] = st
            actions_batch[t] = actions
            logps_batch[t] = logps
            values_batch[t] = values.squeeze()
            rewards_batch[t] = torch.tensor(rewards, device=device)
            dones_batch[t] = torch.tensor(dones, device=device)
            
            states = next_states
        
        return {
            'imgs': imgs_batch,
            'states': states_batch,
            'actions': actions_batch,
            'logps': logps_batch,
            'values': values_batch,
            'rewards': rewards_batch,
            'dones': dones_batch,
            'next_states': states,
            'steps': n_steps
        }
    
    def _ppo_update(self, rollout_data, advs, returns, device):
        """高效的PPO更新，减少数据传输"""
        n_steps, n_envs = rollout_data['rewards'].shape
        dataset_size = n_steps * n_envs
        
        # 展平数据，保持在GPU上
        imgs_flat = rollout_data['imgs'].view(-1, 3, 224, 224)
        states_flat = rollout_data['states'].view(-1, 4)
        actions_flat = rollout_data['actions'].view(-1)
        logps_flat = rollout_data['logps'].view(-1)
        advs_flat = torch.tensor(advs.flatten(), device=device)
        returns_flat = torch.tensor(returns.flatten(), device=device)
        
        # 使用更大的minibatch
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size, device=device)
            
            for start in range(0, dataset_size, self.minibatch_size):
                end = min(start + self.minibatch_size, dataset_size)
                mb_indices = indices[start:end]
                
                # 直接索引，无需额外内存分配
                mb_imgs = imgs_flat[mb_indices]
                mb_states = states_flat[mb_indices]
                mb_actions = actions_flat[mb_indices]
                mb_old_logps = logps_flat[mb_indices]
                mb_advs = advs_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]
                
                # 前向传播和损失计算
                with autocast():
                    probs, values = self.model(mb_imgs, mb_states)
                    dist = torch.distributions.Categorical(probs)
                    new_logps = dist.log_prob(mb_actions)
                    
                    ratio = torch.exp(new_logps - mb_old_logps)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advs
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values.squeeze(), mb_returns)
                    entropy = dist.entropy().mean()
                    
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # 梯度更新
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

'''                
class PPOTrainer(A2CTrainer):
    def __init__(self, model, envs, cfg, log_dir="./Adacrop/logs"):
        TrainerMixin.__init__(self, model, cfg, log_dir=log_dir)
        self.envs = envs
        self.n_envs = len(envs)
        self.gamma = cfg.train.get("gamma", 0.99)
        self.vf_coef = cfg.train.get("value_loss_coef", 0.5)
        self.max_grad_norm = cfg.train.get("max_grad_norm", 0.5)

        self.clip_param = cfg.train.get("clip_param", 0.2)
        self.ppo_epochs = cfg.train.get("ppo_epochs", 4)
        self.batch_size = cfg.train.get("batch_size", 2048)
        self.minibatch_size = cfg.train.get("minibatch_size", 256)  
        
        raw_lr = cfg.train["lr"]
        lr = float(raw_lr)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.log_interval = cfg.train.get("log_interval", 10)
        self.save_interval = cfg.train.get("save_interval", 50)
        self.best_reward = -float("inf")
        self.best_val_final_score = -float("inf")

        with open(self.cfg.data['train_json'], 'r', encoding='utf-8') as f:
           recs = json.load(f)
        self._img_paths = [r['img'] for r in recs if 'img' in r]
        self._img_paths = list(dict.fromkeys(self._img_paths))
        np.random.shuffle(self._img_paths)
        self._img_idx = 0
        self._img_lock = threading.Lock()   

        self._render_workers = int(self.cfg.train.get("num_workers", 16))
        self._render_workers = max(1, min(self._render_workers, self.n_envs))
        self._render_pool = ThreadPoolExecutor(max_workers=self._render_workers)

        self._val_scorer = None

        self.val_json = self.cfg.data.get("val_json", None)
        self.val_eval_episodes = int(self.cfg.train.get("val_eval_episodes", 64))
        self.val_interval = int(self.cfg.train.get("val_interval", 20))

        # 熵退火参数
        self.ent_coef_init  = float(cfg.train.get("entropy_coef", 0.10))
        self.ent_coef_final = float(cfg.train.get("entropy_coef_final", 0.01))
        self.ent_coef = self.ent_coef_init
        self.max_steps_total = int(cfg.train.get("max_steps", cfg.train.get("total_timesteps", 1)))

        # 早期屏蔽 stop（只在采样用）
        self.stop_idx = self.envs[0].actions.index("stop")
        self.stop_mask_steps = int(cfg.train.get("stop_mask_steps", 8))              # 每个 episode 前K步屏蔽
        self.stop_mask_warmup = int(cfg.train.get("stop_mask_warmup_steps", 200000)) # 全局前多少步启用屏蔽

        self.memory_threshold = 75.0  # 70GB阈值
        self.prefetch_factor = 8      # 预取倍数
        self.pin_memory = True        # 启用固定内存

        print(f" PPO: batch_size={self.batch_size}, minibatch_size={self.minibatch_size}")
        print(f" 环境数: {self.n_envs}")

    def _next_image(self) -> Image.Image:
        """线程安全地取下一张训练图"""
        with self._img_lock:
            if self._img_idx >= len(self._img_paths):
                np.random.shuffle(self._img_paths)
                self._img_idx = 0
            path = self._img_paths[self._img_idx]
            self._img_idx += 1
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"! 加载图像失败 {path}: {e}，改为占位图")
            return Image.new('RGB', (224, 224), color=(127,127,127))

    def estimate_memory_usage(self):
        """估算内存使用量"""
        # 模型参数 (~2GB)
        model_memory = 3.0
        
        # rollout数据 (n_steps * n_envs * data_size)
        n_steps = self.cfg.train.get("n_steps", 512)
        rollout_memory = (n_steps * self.n_envs * 3 * 224 * 224 * 4) / 1024**3  # 图像数据
        rollout_memory += (n_steps * self.n_envs * 10 * 4) / 1024**3            # 其他数据
        
        # minibatch内存 (最大并发)
        minibatch_memory = (self.minibatch_size * 3 * 224 * 224 * 4) / 1024**3
        
        # NIMA内存 (~3GB)
        nima_memory = 4.0
        
        # 缓冲区 (~10GB)
        buffer_memory = 15.0
        
        total = model_memory + rollout_memory + minibatch_memory + nima_memory + buffer_memory
        return total

    def _monitor_memory(self, stage=""):
        """监控内存使用"""
        gpu_id = self.device.index
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        
        if allocated > self.memory_threshold:
            print(f"! {stage} 显存使用过高: {allocated:.2f}GB")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        return allocated, reserved

    def evaluate_policy(self, max_episodes=None):
        if not self.val_json or not os.path.exists(self.val_json):
            return None

        with open(self.val_json, "r", encoding="utf-8") as f:
            recs = json.load(f)

        if not recs:
            return None

        recs = recs[: max_episodes or self.val_eval_episodes]
        if self._val_scorer is None:
            self._val_scorer = load_aesthetic_model()
        scorer = self._val_scorer

        final_scores = []
        score_gains = []
        steps_list = []

        self.model.eval()
        with torch.no_grad():
            for rec in recs:
                try:
                    img = Image.open(rec["img"]).convert("RGB")
                    env = CropEnv(img, scorer, self.cfg, inference=False)

                    state = env.reset()
                    init_score = env.prev_score
                    done = False

                    while not done:
                        img_t = state[0].unsqueeze(0).to(self.device)
                        st_t = state[1].unsqueeze(0).to(self.device)

                        probs, _ = self.model(img_t, st_t)
                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample().item()

                        state, _, done, _ = env.step(int(action))

                    final_scores.append(float(env.prev_score))
                    score_gains.append(float(env.prev_score - init_score))
                    steps_list.append(int(env.step_count))
                except Exception as e:
                    print(f"! validate failed for {rec.get('img', 'unknown')}: {e}")

        self.model.train()

        if not final_scores:
            return None

        return {
            "avg_final_score": float(np.mean(final_scores)),
            "avg_score_gain": float(np.mean(score_gains)),
            "avg_steps": float(np.mean(steps_list)),
            "num_eval": len(final_scores),
        }

    def train(self):
        print(" 进入A800优化PPO训练")
        device = self.device
        global_step = 0 
        rollout_count = 0
        
        #  初始化状态，使用固定内存
        states = [env.reset() for env in self.envs] 
        #  批量移动到GPU，减少传输开销
        with torch.cuda.device(device):
            states = [(s[0].to(device, non_blocking=True), s[1].to(device, non_blocking=True)) for s in states]

        while global_step < self.cfg.train["max_steps"]:
            rollout_count += 1
            print(f"→ Rollout {rollout_count}, step={global_step}")
            # 每个rollout开始时强制所有env换图
            for i, env in enumerate(self.envs):
                next_img = self._next_image()
                states[i] = env.load_image(next_img)
            states = [(s[0].to(device, non_blocking=True), s[1].to(device, non_blocking=True)) for s in states]
            
            allocated, reserved = self._monitor_memory(f"Rollout {rollout_count} 开始")
            
            try:
                # 收集rollout数据
                t0 = time.perf_counter()
                rollout_data = self._collect_rollout_a800(states, device, global_step)
                rollout_sec = time.perf_counter() - t0

                states = rollout_data['next_states']
                global_step += rollout_data['steps']
                
                # 计算GAE
                t1 = time.perf_counter()
                advs, returns = self._compute_gae_a800(rollout_data)
                gae_sec = time.perf_counter() - t1
                mean_reward = float(np.mean(returns))

                # 熵退火（线性）
                frac = min(1.0, global_step / max(1, self.max_steps_total))
                self.ent_coef = (1 - frac) * self.ent_coef_init + frac * self.ent_coef_final

                t2 = time.perf_counter()
                self._ppo_update_a800(rollout_data, advs, returns, device)
                update_sec = time.perf_counter() - t2
      
                del rollout_data, advs, returns
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"! Rollout {rollout_count} 失败: {e}")
                # 紧急内存清理
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            
            print(f"[Perf] rollout={rollout_sec:.1f}s gae={gae_sec:.1f}s update={update_sec:.1f}s")
            
            #  日志和保存
            if rollout_count % self.log_interval == 0:
                self.log_metrics(
                    step=global_step,
                    rollout=rollout_count,
                    mean_reward=mean_reward,
                    best_reward=self.best_reward,
                    num_envs=self.n_envs,
                    rollout_sec=round(rollout_sec, 3),
                    gae_sec=round(gae_sec, 3),
                    update_sec=round(update_sec, 3),
                )
                print(f"[A800-PPO] rollout {rollout_count} | step {global_step} | reward {mean_reward:.3f} | mem {allocated:.1f}GB")

            if rollout_count % self.save_interval == 0:
                self.save_model(
                    filename="ppo_latest.pth",
                    tag="latest",
                    extra={
                        "global_step": global_step,
                        "rollout": rollout_count,
                        "mean_reward": mean_reward,
                        "best_reward": self.best_reward,
                    }
                )

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save_model(
                    filename="ppo_best_train_reward.pth",
                    tag="best_train_reward",
                    extra={
                        "global_step": global_step,
                        "rollout": rollout_count,
                        "mean_reward": mean_reward,
                        "best_reward": self.best_reward,
                    }
                )
                print(f"** 最佳训练reward模型: {mean_reward:.3f}")
            
            if rollout_count % self.val_interval == 0:
                val_metrics = self.evaluate_policy()
                if val_metrics is not None:
                    self.log_metrics(
                        step=global_step,
                        rollout=rollout_count,
                        val_avg_final_score=val_metrics["avg_final_score"],
                        val_avg_score_gain=val_metrics["avg_score_gain"],
                        val_avg_steps=val_metrics["avg_steps"],
                        val_num_eval=val_metrics["num_eval"],
                    )
                    print(
                        f"[VAL] rollout {rollout_count} | "
                        f"final_score={val_metrics['avg_final_score']:.3f} | "
                        f"gain={val_metrics['avg_score_gain']:.3f}"
                    )

                    if val_metrics["avg_final_score"] > self.best_val_final_score:
                        self.best_val_final_score = val_metrics["avg_final_score"]
                        self.save_model(
                            filename="ppo_best_val_final_score.pth",
                            tag="best_val_final_score",
                            extra={
                                "global_step": global_step,
                                "rollout": rollout_count,
                                **val_metrics,
                            }
                        )
                        print(f"** 最佳验证final_score模型: {self.best_val_final_score:.3f}")

    def _collect_rollout_a800(self, states, device, global_step: int):
        n_steps = self.cfg.train.get("n_steps", 512)
        rollout_t0 = time.perf_counter() #

        pin = bool(self.cfg.train.get("pin_rollout", True))
        cpu_dev = torch.device("cpu")

        batch_imgs = torch.empty(n_steps, self.n_envs, 3, 224, 224, device=cpu_dev, dtype=torch.float32, pin_memory=pin)
        batch_states = torch.empty(n_steps, self.n_envs, 4, device=cpu_dev, dtype=torch.float32, pin_memory=pin)
        batch_actions = torch.empty(n_steps, self.n_envs, device=cpu_dev, dtype=torch.long, pin_memory=pin)
        batch_logps = torch.empty(n_steps, self.n_envs, device=cpu_dev, dtype=torch.float32, pin_memory=pin)
        batch_values = torch.empty(n_steps, self.n_envs, device=cpu_dev, dtype=torch.float32, pin_memory=pin)
        batch_rewards = torch.empty(n_steps, self.n_envs, device=cpu_dev, dtype=torch.float32, pin_memory=pin)
        batch_dones = torch.empty(n_steps, self.n_envs, device=cpu_dev, dtype=torch.float32, pin_memory=pin)

        for step in range(n_steps):
            try:
                imgs = torch.stack([s[0] for s in states]).to(device, non_blocking=True)
                st = torch.stack([s[1] for s in states]).to(device, non_blocking=True)
                with torch.no_grad():
                    probs, values = self.model(imgs, st)
                    # 早期对部分env屏蔽 stop
                    if global_step < self.stop_mask_warmup:
                        mask = torch.tensor(
                            [1 if env.step_count < self.stop_mask_steps else 0 for env in self.envs],
                            device=device, dtype=torch.bool
                        )
                        probs[mask, self.stop_idx] = 0.0
                    
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum(dim=1, keepdim=True)
                    dist = torch.distributions.Categorical(probs=probs)
                    actions = dist.sample()
                    logps = dist.log_prob(actions)

                if step in (0, 1, 2, 5, 10) or (step % 50 == 0):
                    a = actions.detach().cpu()
                    counts = torch.bincount(a, minlength=len(self.envs[0].actions)).float()
                    frac = (counts / counts.sum()).tolist()
                    act_names = self.envs[0].actions
                    top = sorted([(act_names[i], frac[i]) for i in range(len(act_names))], key=lambda x: -x[1])[:3]
                    print("[ActionFracTop3]", ", ".join([f"{n}={v:.3f}" for n, v in top]))

                # 批量环境步进
                t_env0 = time.perf_counter() #
                next_states, rewards, dones = self._batch_env_step_a800(actions.cpu().numpy())
                t_env = time.perf_counter() - t_env0 #
                if step in (0, 1, 2, 5, 10) or (step % 50 == 0): #
                    print(f"[RolloutDebug] env_step_dt={t_env:.3f}s") #

                batch_imgs[step].copy_(imgs.detach().to("cpu", non_blocking=True))
                batch_states[step].copy_(st.detach().to("cpu", non_blocking=True))
                batch_actions[step].copy_(actions.detach().to("cpu", non_blocking=True))
                batch_logps[step].copy_(logps.detach().to("cpu", non_blocking=True))
                batch_values[step].copy_(values.squeeze().detach().to("cpu", non_blocking=True))
                batch_rewards[step].copy_(torch.as_tensor(rewards, dtype=torch.float32))
                batch_dones[step].copy_(torch.as_tensor(dones, dtype=torch.float32))

                states = next_states
                
                # 清理临时张量
                del imgs, st, probs, values, dist, actions, logps
                
            except Exception as e:
                print(f"步骤 {step} 收集失败: {e}")
                break

        return {
            'imgs': batch_imgs,
            'states': batch_states,
            'actions': batch_actions,
            'logps': batch_logps,
            'values': batch_values,
            'rewards': batch_rewards,
            'dones': batch_dones,
            'next_states': states,
            'steps': n_steps * self.n_envs
        }

    # def _batch_env_step_a800(self, actions):
    #     """批量环境步进，优化数据传输"""
    #     from concurrent.futures import ThreadPoolExecutor, as_completed

    #     n_envs = len(self.envs)
    #     next_states = [None] * n_envs
    #     rewards = [0.0] * n_envs
    #     dones = [False] * n_envs
        
    #     def worker(i, act):
    #         """调用 env.step 并返回规范化结果 (i, img_tensor, state_tensor, reward, done)"""
    #         try:
    #             res = self.envs[i].step(act)
    #             # env.step 返回 ((img_tensor, state_tensor), reward, done, info)
    #             (img_cpu, st_cpu), r, done, _ = res
    #             if done:
    #                 next_img = self._next_image()
    #                 img_cpu, st_cpu = self.envs[i].load_image(next_img)

    #             # 如果 env 返回 PIL.Image，转为 tensor；如果返回 tensor，直接使用
    #             if not isinstance(img_cpu, torch.Tensor):
    #                 # 只在极少数情况发生，尽量把转换并行到线程中
    #                 img_tensor = T.ToTensor()(img_cpu)
    #             else:
    #                 img_tensor = img_cpu

    #             if not isinstance(st_cpu, torch.Tensor):
    #                 st_tensor = torch.as_tensor(st_cpu, dtype=torch.float32)
    #             else:
    #                 st_tensor = st_cpu

    #             # 直接移动到训练设备（非阻塞）
    #             img_gpu = img_tensor.to(self.device, non_blocking=True)
    #             st_gpu = st_tensor.to(self.device, non_blocking=True)

    #             return (i, (img_gpu, st_gpu), float(r), bool(done))

    #         except Exception as e:
    #             # 出错时返回占位（保持训练可以继续）
    #             print(f"环境 {i} 并行 step 失败: {e}")
    #             default_img = torch.zeros(3, 224, 224, device=self.device)
    #             default_state = torch.zeros(4, device=self.device)
    #             return (i, (default_img, default_state), 0.0, False)

    #     max_workers = min(32, n_envs)
    #     with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #         futures = [ex.submit(worker, i, actions[i]) for i in range(n_envs)]
    #         for fut in as_completed(futures):
    #             i, state_pair, r, done = fut.result()
    #             next_states[i] = state_pair
    #             rewards[i] = r
    #             dones[i] = done

    #     return next_states, rewards, dones

    # def _batch_env_step_a800(self, actions):
    #     # 等待
    #     next_states = [None] * len(self.envs)
    #     rewards = [0.0] * len(self.envs)
    #     dones = [False] * len(self.envs)

    #     for i, act in enumerate(actions):
    #         try:
    #             (img_cpu, st_cpu), r, done, _ = self.envs[i].step(act)
    #             if done:
    #                 next_img = self._next_image()
    #                 img_cpu, st_cpu = self.envs[i].load_image(next_img)

    #             img_tensor = img_cpu if isinstance(img_cpu, torch.Tensor) else T.ToTensor()(img_cpu)
    #             st_tensor = st_cpu if isinstance(st_cpu, torch.Tensor) else torch.as_tensor(st_cpu, dtype=torch.float32)

    #             next_states[i] = (
    #                 img_tensor.to(self.device, non_blocking=True),
    #                 st_tensor.to(self.device, non_blocking=True),
    #             )
    #             rewards[i] = float(r)
    #             dones[i] = bool(done)

    #         except Exception as e:
    #             print(f"环境 {i} step 失败: {e}")
    #             next_states[i] = (
    #                 torch.zeros(3, 224, 224, device=self.device),
    #                 torch.zeros(4, device=self.device),
    #             )
    #             rewards[i] = 0.0
    #             dones[i] = False

    #     return next_states, rewards, dones

    def _batch_env_step_a800(self, actions_np): # 两阶段版
        # 第一阶段：执行 step，但不阻塞评分
        step_outs = [None] * self.n_envs

        for i, act in enumerate(actions_np):
            step_outs[i] = self.envs[i].step(int(act), defer_score=True)

        # 第二阶段：统一等待评分结果并补全 reward
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)

        for i, (ns, r_stub, done, info) in enumerate(step_outs):
            dones[i] = bool(done)

            # ✅ stop：env 已经返回真实 reward，直接用，不走 finalize
            if isinstance(info, dict) and info.get("is_stop", False):
                rewards[i] = float(r_stub)
            else:
                fut = info.get("score_future", None) if isinstance(info, dict) else None
                if fut is not None:
                    try:
                        score = self.envs[i].resolve_score(fut, timeout=1.5)
                    except Exception:
                        score = 5.0
                    rewards[i] = float(self.envs[i].finalize_reward_with_score(score, info))
                else:
                    old_score = 5.0
                    if isinstance(info, dict):
                        old_score = float(info.get("old_score", 5.0))
                    rewards[i] = float(self.envs[i].finalize_reward_with_score(old_score, info))

            if dones[i]:
                next_img = self._next_image()
                self.envs[i].load_image(next_img)

        need_image = (not bool(self.cfg.env.get("observation_mode", False)))  # False=>需要真实图像
        next_states = [None] * self.n_envs

        if not need_image:
            # state-only：直接用 env.step 返回的占位图像+state
            for i, (ns, _, _, _) in enumerate(step_outs):
                img_cpu, st_cpu = ns
                next_states[i] = (
                    img_cpu.to(self.device, non_blocking=True),
                    st_cpu.to(self.device, non_blocking=True),
                )
            return next_states, rewards, dones

        # image+state：并行 crop/resize/toTensor（CPU），再搬到 GPU
        def render_worker(i: int):
            try:
                img_tensor = self.envs[i].render_obs_image()
            except Exception:
                img_tensor = torch.zeros(3, 224, 224)
            try:
                st_tensor = self.envs[i].state_only()
            except Exception:
                st_tensor = torch.zeros(4)
            return i, img_tensor, st_tensor

        max_workers = int(self.cfg.train.get("num_workers", 16))
        max_workers = max(1, min(max_workers, self.n_envs))

        futures = [self._render_pool.submit(render_worker, i) for i in range(self.n_envs)]
        for fut in as_completed(futures):
            i, img_cpu, st_cpu = fut.result()
            next_states[i] = (
                img_cpu.to(self.device, non_blocking=True),
                st_cpu.to(self.device, non_blocking=True),
            )

        return next_states, rewards, dones


    def _ppo_update_a800(self, rollout_data, advs, returns, device):
        """A800优化的PPO更新"""
        n_steps, n_envs = rollout_data['rewards'].shape
        dataset_size = n_steps * n_envs
        
        print(f"   PPO更新: dataset_size={dataset_size}, epochs={self.ppo_epochs}")
        
        imgs_flat = rollout_data['imgs'].view(-1, 3, 224, 224)
        states_flat = rollout_data['states'].view(-1, 4)
        actions_flat = rollout_data['actions'].view(-1)
        logps_flat = rollout_data['logps'].view(-1)

        advs_cpu = torch.as_tensor(advs.flatten(), dtype=torch.float32)      # CPU
        advs_cpu = (advs_cpu - advs_cpu.mean()) / (advs_cpu.std(unbiased=False) + 1e-8)
        returns_cpu = torch.as_tensor(returns.flatten(), dtype=torch.float32) # CPU

        for epoch in range(self.ppo_epochs):
            #  GPU上生成随机索引
            indices = torch.randperm(dataset_size, device=device)
            
            total_batches = (dataset_size + self.minibatch_size - 1) // self.minibatch_size
            
            for batch_idx, start in enumerate(range(0, dataset_size, self.minibatch_size)):
                end = min(start + self.minibatch_size, dataset_size)
                mb_indices = indices[start:end].detach().cpu()

                try:
                    mb_imgs = imgs_flat[mb_indices].to(device, non_blocking=True)
                    mb_states = states_flat[mb_indices].to(device, non_blocking=True)
                    mb_actions = actions_flat[mb_indices].to(device, non_blocking=True)
                    mb_old_logps = logps_flat[mb_indices].to(device, non_blocking=True)
                    mb_advs = advs_cpu[mb_indices].to(device, non_blocking=True)
                    mb_returns = returns_cpu[mb_indices].to(device, non_blocking=True)

                    # 前向传播
                    probs, vals = self.model(mb_imgs, mb_states)
                    dist = torch.distributions.Categorical(probs)
                    new_logps = dist.log_prob(mb_actions)
                    ratio = torch.exp(new_logps - mb_old_logps)

                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(vals.squeeze(), mb_returns)
                    entropy = dist.entropy().mean()
                    
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    #  梯度更新
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # if batch_idx % 10 == 0:
                    #     torch.cuda.empty_cache()

                except Exception as e:
                    print(f"❌ Epoch {epoch}, batch {batch_idx} 失败: {e}")
                    torch.cuda.empty_cache()
                    continue

    def _compute_gae_a800(self, rollout_data):
        """GAE：rollout buffer 在 CPU pinned，GAE 在 CPU 计算"""
        rewards_cpu = rollout_data["rewards"]  # CPU tensor [T,N]
        dones_cpu = rollout_data["dones"]      # CPU tensor [T,N]
        values_cpu = rollout_data["values"]    # CPU tensor [T,N]

        # 最后一步 value：GPU forward 后搬回 CPU
        with torch.no_grad():
            last_states = rollout_data["next_states"]
            last_imgs = torch.stack([s[0] for s in last_states]).to(self.device, non_blocking=True)
            last_st = torch.stack([s[1] for s in last_states]).to(self.device, non_blocking=True)
            _, last_values = self.model(last_imgs, last_st)
            last_values_cpu = last_values.squeeze().detach().to("cpu", non_blocking=True)  # [N]

        # 拼成 [T+1, N]（CPU）
        all_values_cpu = torch.cat([values_cpu, last_values_cpu.unsqueeze(0)], dim=0)

        rewards_np = rewards_cpu.numpy()
        dones_np = dones_cpu.numpy()
        values_np = all_values_cpu.numpy()

        # 清理
        del last_imgs, last_st, last_values, last_values_cpu, all_values_cpu

        advs, returns = compute_gae(rewards_np, values_np, dones_np, self.gamma, 0.95)
        return advs, returns


class A800SharedNIMAScorer:
    def __init__(self, cfg=None):
        print("  初始化A800共享NIMA...")

        if cfg is not None and hasattr(cfg, 'nima') and 'training_gpus' in cfg.nima:
            gpu_list = cfg.nima['training_gpus']
            if isinstance(gpu_list, list):
                self.nima_gpu = gpu_list[0]  
            else:
                self.nima_gpu = gpu_list     
        else:
            self.nima_gpu = 0 
        
        self.device = torch.device(f"cuda:{self.nima_gpu}") 
        self.weights_path = cfg.nima.get('weights_path', './NIMA/weights/nima_inception_pytorch.pth')
        self.batch_size = cfg.nima.get('batch_size', 32)

        if cfg is not None and hasattr(cfg, 'nima'):
            self.weights_path = cfg.nima.get('weights_path', './NIMA/weights/nima_inception_pytorch.pth')
            self.batch_size = cfg.nima.get('batch_size', 32)
        else:
            print("* 未提供配置，使用默认设置")
            self.weights_path = './NIMA/weights/nima_inception_pytorch.pth'
            self.batch_size = 32
        
        print(f" 权重路径: {self.weights_path}")
        self.model, self.input_size = self._load_model()
        self.model.eval()
        
        #  预处理器
        self.preprocessor = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        #  批处理队列
        self.batch_queue = []
        self.batch_futures = []
        
        print(f" A800共享NIMA就绪")

    def _load_model(self):
        """加载NIMA模型"""
        if 'inception' in self.weights_path:
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=False, aux_logits=False)
            model.fc = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(model.fc.in_features, 10)
            )
            input_size = 299
        else:
            model = models.resnet50(pretrained=True)
            model.fc = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(2048, 10)
            )
            input_size = 224
        
        model = model.to(self.device)
        
        if os.path.exists(self.weights_path):
            ckpt = torch.load(self.weights_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            try:
                model.load_state_dict(ckpt, strict=True)
                print(" 成功加载NIMA权重(strict=True)")
            except Exception as e:
                print(f"！ 严格加载失败，尝试非严格: {e}")
                model.load_state_dict(ckpt, strict=False)
        else:
            print(f"！ 未找到权重: {self.weights_path}，使用随机初始化")
        return model, input_size

    def __call__(self, img, source_gpu=0):
        """同步评分接口"""
        try:
            #  预处理
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    tensor = img.unsqueeze(0)
                else:
                    tensor = img
                if tensor.shape[-1] != self.input_size or tensor.shape[-2] != self.input_size:
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False
                    )
                tensor = tensor.to(self.device, non_blocking=True)
            else:
                tensor = self.preprocessor(img).unsqueeze(0).to(self.device, non_blocking=True)
            
            #  批处理推理
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                labels = torch.arange(1, 11, dtype=torch.float32, device=self.device)
                score = (probs * labels).sum()
                
            return float(score.clamp(1.0, 10.0).cpu())
            
        except Exception as e:
            print(f"！ NIMA评分失败: {e}")
            return 5.0
    
    def get_stats(self):
        queue_sizes = {gpu_id: self.local_queues[gpu_id].qsize() for gpu_id in self.training_gpus}
        total_queue_size = sum(queue_sizes.values())
        return {
            'architecture': 'PCIe-PyTorch-optimized',
            'model_format': '.pth',
            'training_gpus': str(self.device),
            'queue_sizes': queue_sizes,
            'total_queue_size': total_queue_size,
            'memory_allocated': torch.cuda.memory_allocated(self.nima_gpu) / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved(self.nima_gpu) / 1024**3,
            'cross_gpu_transfers': 0
        }

class NIMAScorer:
    """基于 Inception-ResNet (.h5) 的TF/Keras版NIMA打分器"""
    def __init__(self, weights_path: str, target_size=(299, 299), use_gpu=True, device_id=0):
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dropout, Dense
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

        self.tf = tf
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.device = f"/GPU:{device_id}" if (use_gpu and tf.config.list_physical_devices('GPU')) else "/CPU:0"

        # 显存按需
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except Exception: pass

        with tf.device(self.device):
            base = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling="avg", weights=None)
            x = Dropout(0.75)(base.output)
            x = Dense(10, activation="softmax")(x)
            self.model = Model(base.input, x)
            self.model.load_weights(weights_path)
        print(f"NIMA(TF) loaded: {weights_path} @ {self.device}")

    def __call__(self, img: Image.Image) -> float:
        img = img.convert("RGB").resize(self.target_size)
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        arr = self.preprocess_input(arr)
        with self.tf.device(self.device):
            preds = self.model.predict(arr, batch_size=1, verbose=0)[0]
        return mean_score(preds)

'''    
    score = __call__
 
def load_aesthetic_model(): # 转为有.pth的版本
    print(" 检测GPU架构和权重格式...")
    
    try:
        cfg = Config()
        
        # 检查权重文件格式
        weights_path = cfg.nima.get('weights_path', './NIMA/weights/nima_resnet50.pth')
        
        if weights_path.endswith('.pth'):
            print(" 检测到PyTorch权重格式，使用PCIe本地优化版本")
            return A800SharedNIMAScorer(cfg)
        elif weights_path.endswith('.h5'):
            print("* 检测到TensorFlow权重格式")
            print(" 建议转换为.pth格式以获得更好性能")
            # 可以在这里调用转换函数或使用TF版本
            return A800SharedNIMAScorer(cfg)  # 回退到.pth版本
        else:
            print(" 未知权重格式，使用默认PyTorch版本")
            return A800SharedNIMAScorer(cfg)
            
    except Exception as e:
        print(f"！ 配置加载失败: {e}")
        print("* 使用默认PCIe PyTorch版本")
        return A800SharedNIMAScorer(cfg=None)
'''
def log_gpu(step, note=""):
    print(f"\n>>> Step {step}  【{note}】 显存状态:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i)/1024**3
        resv = torch.cuda.memory_reserved(i)/1024**3
        print(f"  GPU{i}: allocated={alloc:.2f}GB, reserved={resv:.2f}GB")
    gc.collect()
    torch.cuda.empty_cache()

'''
class RemoteNIMAScorer: # 分离版 NIMA
    def __init__(self, host='localhost', port=9999, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._test_connection()
    
    def _test_connection(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.host, self.port))
            sock.close()
            print(" NIMA服务器连接成功")
        except Exception as e:
            print(f"！ NIMA服务器连接失败: {e}")
    
    def __call__(self, img: Image.Image) -> float:
        try:
            # 编码图像
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=95)
            img_data = img_bytes.getvalue()
            
            # 发送请求
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            # 发送数据
            sock.send(len(img_data).to_bytes(4, 'big'))
            sock.send(img_data)
            
            # 接收结果
            response_length = int.from_bytes(sock.recv(4), 'big')
            response_data = b''
            while len(response_data) < response_length:
                chunk = sock.recv(min(4096, response_length - len(response_data)))
                if not chunk:
                    break
                response_data += chunk
            
            score = pickle.loads(response_data)
            sock.close()
            return float(score)
            
        except Exception as e:
            print(f"NIMA调用失败: {e}")
            return np.random.uniform(4.0, 6.0)  # 返回随机分数
    
    score = __call__
'''

'''
class NIMAScorer: # 未分离版本的 NIMA
    """
    使用 InceptionResNetV2 + NIMA 权重计算美学分数。
    调用：score = scorer(pil_image)
    """

    def __init__(self,
                 weights_path: str = "weights/inception_resnet_weights.h5",
                 target_size: tuple = (224, 224), use_gpu: bool = True,
                 device_id: int = 0):

        self.device_id = device_id

        if use_gpu:
            # 使用GPU 3，避免与主训练模型冲突
            with tf.device(f'/GPU:{device_id}'):
                base = InceptionResNetV2(input_shape=(None, None, 3),
                                       include_top=False,
                                       pooling="avg",
                                       weights=None)
                x = Dropout(0.75)(base.output)
                x = Dense(10, activation="softmax")(x)
                self.model = Model(base.input, x)
                self.model.load_weights(weights_path)
        else:
            with tf.device('/CPU:0'):
                base = InceptionResNetV2(input_shape=(None, None, 3),
                                            include_top=False,
                                            pooling="avg",
                                            weights=None)
                x = Dropout(0.75)(base.output)
                x = Dense(10, activation="softmax")(x)
                model = Model(base.input, x)
                model.load_weights(weights_path)
                self.model = model
                self.target_size = target_size 
        #print(">> [DEBUG] model summary start")
        #self.model.summary()           
        #print(">> [DEBUG] total params:", self.model.count_params())
        #print(">> [DEBUG] weights count:", len(self.model.get_weights()))
        #print(">> [DEBUG] model summary end\n")
        self.target_size = target_size
        self.use_gpu = use_gpu

    def __call__(self, img: Image.Image) -> float:

        # 1) Resize & to numpy
        img = img.resize(self.target_size)
        arr = np.array(img, dtype=np.float32)
        #print(">> [DEBUG] raw img min/max:", arr.min(), arr.max())
        # 2) Batch + preprocess
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        #print(">> [DEBUG] after preprocess min/max:", arr.min(), arr.max())
        # 3) 前向
        if self.use_gpu:
            with tf.device(f'/GPU:{self.device_id}'):
                preds = self.model.predict(arr, batch_size=1, verbose=0)[0]
        else:
            with tf.device('/CPU:0'):
                preds = self.model.predict(arr, batch_size=1, verbose=0)[0]
        #print(">> [DEBUG] preds distribution:", np.round(preds, 3))
        #print(">> [DEBUG] preds sum:", preds.sum())
        # 4) 计算期望分
        sc = mean_score(preds)
            #print(">> [DEBUG] mean_score:", sc)
        return float(sc)

    score = __call__

def load_aesthetic_model():
    #NIMA单独占一张卡
    import os
    old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    try:
        scorer = NIMAScorer(
            weights_path="./NIMA/weights/inception_resnet_weights.h5",
            target_size=(224, 224),
            use_gpu=True,
            device_id=0  
        )
        print("NIMA模型已加载到GPU 7")
        return scorer
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
'''