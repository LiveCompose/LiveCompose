import os
import time
import json
import numpy as np
import socket
import pickle
import io
import gc
from collections import deque, namedtuple
from PIL import Image  
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
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
from src.config import Config


'''
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # 只用4GB
            )
    except RuntimeError as e:
        print(e)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2, preprocess_input
)
import concurrent.futures
import threading
'''

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
        
        # ✅ 强制使用单GPU，最大化利用A800
        training_gpu = cfg.train.get("training_gpus", 0)
        print(f"🚀 单A800优化模式: GPU {training_gpu}")
        print(f"📊 可用显存: ~80GB")
        
        device = torch.device(f"cuda:{training_gpu}")
        model.to(device)
        self.device = device
        self.training_gpu = training_gpu
        
        # ✅ 不使用DataParallel，避免内存碎片
        self.model = model
        self.cfg = cfg
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.start_time = time.time()

        # ✅ 预热GPU，优化内存分配
        self._warmup_gpu()
        
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
                
                print(f"✅ GPU预热完成: {probs.shape}, {values.shape}")
                
                # 清理预热张量
                del warmup_tensors, test_img, test_state, probs, values
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ GPU预热失败: {e}")

    def save_model(self, suffix="pt", tag=None):
        fn = f"actor_critic_{int(time.time()-self.start_time)}.{suffix}"
        if tag:
            name, ext = fn.rsplit(".", 1)
            fn = f"{name}_{tag}.{ext}"
        path = os.path.join(self.log_dir, fn)
        
        # ✅ 保存时包含更多信息
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': getattr(self, 'optimizer', None) and self.optimizer.state_dict(),
            'config': self.cfg.__dict__ if hasattr(self.cfg, '__dict__') else str(self.cfg),
            'timestamp': time.time(),
            'training_time': time.time() - self.start_time
        }
        
        torch.save(checkpoint, path)
        print(f"📁 模型已保存: {path}")

    def log_metrics(self, step, **kwargs):
        """增强的日志记录"""
        log_path = os.path.join(self.log_dir, "log.txt")
        
        # ✅ 添加GPU内存使用信息
        gpu_memory = torch.cuda.memory_allocated(self.training_gpu) / 1024**3
        gpu_reserved = torch.cuda.memory_reserved(self.training_gpu) / 1024**3
        
        record = {
            "step": step, 
            "gpu_memory_gb": round(gpu_memory, 2),
            "gpu_reserved_gb": round(gpu_reserved, 2),
            "training_time": round(time.time() - self.start_time, 2),
            **kwargs
        }
        
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

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
                imgs = torch.stack([s[0] for s in states]).to(device)
                st   = torch.stack([s[1] for s in states]).to(device)

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
                _, next_val = self.model(imgs_list[-1].to(device), st_list[-1].to(device))
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
            imgs_batch = torch.cat(imgs_list).to(device)           # (T*N, C, H, W)
            st_batch   = torch.cat(st_list).to(device)             # (T*N, dim)
            acts_batch = torch.cat(acts_list).to(device)           # (T*N,)
            old_logps  = torch.cat(logps_list).to(device)          # (T*N,)
            advs_t     = torch.tensor(advs.flatten(), dtype=torch.float32).to(device)
            ret_t      = torch.tensor(returns.flatten(), dtype=torch.float32).to(device)

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
    def __init__(self, model, envs, cfg):
        super().__init__(model, envs, cfg)
        self.clip_param = cfg.train.get("clip_param", 0.2)
        self.ppo_epochs = cfg.train.get("ppo_epochs", 4)
        self.batch_size = cfg.train.get("batch_size", 2048)      # ✅ 大batch充分利用A800
        self.minibatch_size = cfg.train.get("minibatch_size", 256)  # ✅ 合理minibatch
        
        # ✅ 移除GradScaler，简化内存管理
        raw_lr = cfg.train["lr"]
        lr = float(raw_lr)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.n_envs = len(envs)
        self.log_interval = cfg.train.get("log_interval", 10)
        self.save_interval = cfg.train.get("save_interval", 50)
        self.best_reward = -float("inf")
        
        # ✅ A800专用内存管理参数
        self.memory_threshold = 75.0  # 70GB阈值
        self.prefetch_factor = 8      # 预取倍数
        self.pin_memory = True        # 启用固定内存

        print(f"🚀 A800优化PPO: batch_size={self.batch_size}, minibatch_size={self.minibatch_size}")
        print(f"📊 环境数: {self.n_envs}, 预计峰值显存: ~{self.estimate_memory_usage():.1f}GB")

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
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        if allocated > self.memory_threshold:
            print(f"⚠️ {stage} 显存使用过高: {allocated:.2f}GB")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        return allocated, reserved

    def train(self):
        print("🚀 进入A800优化PPO训练")
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
            
            allocated, reserved = self._monitor_memory(f"Rollout {rollout_count} 开始")
            
            try:
                # 高效收集rollout数据
                rollout_data = self._collect_rollout_a800(states, device)
                states = rollout_data['next_states']
                global_step += rollout_data['steps']
                
                self._monitor_memory("收集完成")
                
                # 计算GAE
                advs, returns = self._compute_gae_a800(rollout_data)
                mean_reward = float(np.mean(returns))
                
                self._monitor_memory("GAE完成")
                
                self._ppo_update_a800(rollout_data, advs, returns, device)
                
                self._monitor_memory("更新完成")

                del rollout_data, advs, returns
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Rollout {rollout_count} 失败: {e}")
                # 紧急内存清理
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            
            #  日志和保存
            if rollout_count % self.log_interval == 0:
                self.log_metrics(
                    step=global_step,
                    rollout=rollout_count,
                    mean_reward=mean_reward,
                    num_envs=self.n_envs
                )
                print(f"[A800-PPO] rollout {rollout_count} | step {global_step} | reward {mean_reward:.3f} | mem {allocated:.1f}GB")

            if rollout_count % self.save_interval == 0:
                checkpoint_name = f"rollout_{rollout_count}"
                self.save_model(tag=checkpoint_name)
                print(f"📁 模型已保存 (rollout {rollout_count})")

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save_model(tag="best")
                print(f"🏆 最佳模型 (reward {mean_reward:.3f})")

    def _collect_rollout_a800(self, states, device):
        """A800优化的rollout收集"""

        n_steps = self.cfg.train.get("n_steps", 512)
        
        # ✅ 预分配大张量，充分利用A800显存
        batch_imgs = torch.zeros(n_steps, self.n_envs, 3, 224, 224, device=device, dtype=torch.float32)
        batch_states = torch.zeros(n_steps, self.n_envs, 4, device=device, dtype=torch.float32)
        batch_actions = torch.zeros(n_steps, self.n_envs, device=device, dtype=torch.long)
        batch_logps = torch.zeros(n_steps, self.n_envs, device=device, dtype=torch.float32)
        batch_values = torch.zeros(n_steps, self.n_envs, device=device, dtype=torch.float32)
        batch_rewards = torch.zeros(n_steps, self.n_envs, device=device, dtype=torch.float32)
        batch_dones = torch.zeros(n_steps, self.n_envs, device=device, dtype=torch.float32)

        for step in range(n_steps):
            try:
                # ✅ 批量组装输入
                imgs = torch.stack([s[0] for s in states])
                st = torch.stack([s[1] for s in states])

                with torch.no_grad():
                    probs, values = self.model(imgs, st)
                    
                    # ✅ 数值稳定性
                    if torch.isnan(probs).any():
                        probs = torch.ones_like(probs) / probs.shape[1]
                    
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum(dim=1, keepdim=True)

                    dist = torch.distributions.Categorical(probs=probs)
                    actions = dist.sample()
                    logps = dist.log_prob(actions)

                # ✅ 批量环境步进
                next_states, rewards, dones = self._batch_env_step_a800(actions.cpu().numpy())

                # ✅ 直接存储到预分配张量
                batch_imgs[step] = imgs
                batch_states[step] = st
                batch_actions[step] = actions
                batch_logps[step] = logps
                batch_values[step] = values.squeeze()
                batch_rewards[step] = torch.tensor(rewards, device=device, dtype=torch.float32)
                batch_dones[step] = torch.tensor(dones, device=device, dtype=torch.float32)

                # ✅ 更新状态
                states = next_states
                
                # ✅ 清理临时张量
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

    def _batch_env_step_a800(self, actions):
        """批量环境步进，优化数据传输"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_envs = len(self.envs)
        next_states = [None] * n_envs
        rewards = [0.0] * n_envs
        dones = [False] * n_envs
        
        def worker(i, act):
            """调用 env.step 并返回规范化结果 (i, img_tensor, state_tensor, reward, done)"""
            try:
                res = self.envs[i].step(act)
                # env.step 返回 ((img_tensor, state_tensor), reward, done, info)
                (img_cpu, st_cpu), r, done, _ = res

                # 如果 env 返回 PIL.Image，转为 tensor；如果返回 tensor，直接使用
                if not isinstance(img_cpu, torch.Tensor):
                    # 只在极少数情况发生，尽量把转换并行到线程中
                    img_tensor = T.ToTensor()(img_cpu)
                else:
                    img_tensor = img_cpu

                if not isinstance(st_cpu, torch.Tensor):
                    st_tensor = torch.as_tensor(st_cpu, dtype=torch.float32)
                else:
                    st_tensor = st_cpu

                # 直接移动到训练设备（非阻塞）
                img_gpu = img_tensor.to(self.device, non_blocking=True)
                st_gpu = st_tensor.to(self.device, non_blocking=True)

                return (i, (img_gpu, st_gpu), float(r), bool(done))

            except Exception as e:
                # 出错时返回占位（保持训练可以继续）
                print(f"环境 {i} 并行 step 失败: {e}")
                default_img = torch.zeros(3, 224, 224, device=self.device)
                default_state = torch.zeros(4, device=self.device)
                return (i, (default_img, default_state), 0.0, False)

        max_workers = min(32, n_envs)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, i, actions[i]) for i in range(n_envs)]
            for fut in as_completed(futures):
                i, state_pair, r, done = fut.result()
                next_states[i] = state_pair
                rewards[i] = r
                dones[i] = done

        return next_states, rewards, dones

    def _ppo_update_a800(self, rollout_data, advs, returns, device):
        """A800优化的PPO更新"""
        n_steps, n_envs = rollout_data['rewards'].shape
        dataset_size = n_steps * n_envs
        
        print(f"  🔄 PPO更新: dataset_size={dataset_size}, epochs={self.ppo_epochs}")
        
        # ✅ 展平数据，保持在GPU上
        imgs_flat = rollout_data['imgs'].view(-1, 3, 224, 224)
        states_flat = rollout_data['states'].view(-1, 4)
        actions_flat = rollout_data['actions'].view(-1)
        logps_flat = rollout_data['logps'].view(-1)
        advs_flat = torch.tensor(advs.flatten(), device=device, dtype=torch.float32)
        returns_flat = torch.tensor(returns.flatten(), device=device, dtype=torch.float32)

        for epoch in range(self.ppo_epochs):
            # ✅ GPU上生成随机索引
            indices = torch.randperm(dataset_size, device=device)
            
            total_batches = (dataset_size + self.minibatch_size - 1) // self.minibatch_size
            
            for batch_idx, start in enumerate(range(0, dataset_size, self.minibatch_size)):
                end = min(start + self.minibatch_size, dataset_size)
                mb_indices = indices[start:end]

                try:
                    # ✅ 直接在GPU上索引，避免CPU-GPU传输
                    mb_imgs = imgs_flat[mb_indices]
                    mb_states = states_flat[mb_indices]
                    mb_actions = actions_flat[mb_indices]
                    mb_old_logps = logps_flat[mb_indices]
                    mb_advs = advs_flat[mb_indices]
                    mb_returns = returns_flat[mb_indices]

                    # ✅ 前向传播
                    probs, vals = self.model(mb_imgs, mb_states)
                    dist = torch.distributions.Categorical(probs)
                    new_logps = dist.log_prob(mb_actions)
                    
                    # ✅ PPO损失计算
                    ratio = torch.exp(new_logps - mb_old_logps)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(vals.squeeze(), mb_returns)
                    entropy = dist.entropy().mean()
                    
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    # ✅ 梯度更新
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # ✅ 定期清理，但不过于频繁
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"❌ Epoch {epoch}, batch {batch_idx} 失败: {e}")
                    torch.cuda.empty_cache()
                    continue

    def _compute_gae_a800(self, rollout_data):
        """A800优化的GAE计算"""
        # ✅ 在GPU上计算，然后移动到CPU
        rewards_gpu = rollout_data['rewards']
        dones_gpu = rollout_data['dones']
        values_gpu = rollout_data['values']
        
        # ✅ 添加最后一个value
        with torch.no_grad():
            last_states = rollout_data['next_states']
            last_imgs = torch.stack([s[0] for s in last_states]).to(self.device)
            last_st = torch.stack([s[1] for s in last_states]).to(self.device)
            _, last_values = self.model(last_imgs, last_st)
            last_values = last_values.squeeze()
        
        # ✅ 拼接values
        all_values = torch.cat([values_gpu, last_values.unsqueeze(0)], dim=0)
        
        # ✅ 移动到CPU进行GAE计算
        rewards_np = rewards_gpu.cpu().numpy()
        dones_np = dones_gpu.cpu().numpy()
        values_np = all_values.cpu().numpy()
        
        # ✅ 清理GPU张量
        del last_imgs, last_st, last_values, all_values
        torch.cuda.empty_cache()
        
        advs, returns = compute_gae(rewards_np, values_np, dones_np, self.gamma, 0.95)
        return advs, returns


def mean_score(preds):
    """
    计算 preds 的均值，支持多种格式。
    """
    labels = np.arange(1, preds.shape[-1] + 1, dtype=np.float32)
    score = float((preds * labels).sum())
    return score


class A800SharedNIMAScorer:
    
    def __init__(self, cfg=None):
        print("🚀 初始化A800共享NIMA...")

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
        self.model = self._load_model()
        self.model.eval()
        
        #  预处理器
        self.preprocessor = T.Compose([
            T.Resize((224, 224)),
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
            model = inception_v3(pretrained=False)
            if hasattr(model, 'AuxLogits'):
                model.AuxLogits = None
            model.fc = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(model.fc.in_features, 10)
            )
        else:
            model = models.resnet50(pretrained=True)
            model.fc = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(2048, 10)
            )
        
        model = model.to(self.device)
        
        if os.path.exists(self.weights_path):
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print(f" 成功加载NIMA权重")
        
        return model

    def __call__(self, img, source_gpu=0):
        """同步评分接口"""
        try:
            #  预处理
            if isinstance(img, torch.Tensor):
                if img.device != self.device:
                    tensor = img.to(self.device, non_blocking=True)
                else:
                    tensor = img
            else:
                tensor = self.preprocessor(img).to(self.device, non_blocking=True)
            
            #  批处理推理
            with torch.no_grad():
                logits = self.model(tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                labels = torch.arange(1, 11, dtype=torch.float32, device=self.device)
                score = (probs * labels).sum()
                
            return float(score.cpu())
            
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
    
    score = __call__

def load_aesthetic_model():
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