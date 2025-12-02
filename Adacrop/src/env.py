import numpy as np
import hashlib
from PIL import Image
import torch
import torchvision.transforms as T

class CropEnv:

    _global_score_cache = {}
    _pending_requests = {}
    _batch_scorer = None

    def __init__(self, img: Image.Image, aesthetic_model, cfg, inference=False):
        self.orig = img
        self.model = aesthetic_model
        self.img_size = cfg.env["img_size"]  # 图像大小
        self.max_steps = cfg.env.get("max_steps",200) # 最大步数
        self.delta = cfg.env["action_delta"] # 动作增量
        self.actions = ["left","right","up","down",
                        "zoom_in","zoom_out", "wider","narrower",
                        "taller", "shorter", "stop"]
        self.inference = inference

        if hasattr(cfg, 'nima') and 'training_gpus' in cfg.nima:
            nima_gpus = cfg.nima['training_gpus']
            if isinstance(nima_gpus, list):
                self.nima_gpu_list = nima_gpus
            else:
                self.nima_gpu_list = [nima_gpus]
        else:
            self.nima_gpu_list = [0]
        
        if not hasattr(CropEnv, '_gpu_counter'):
            CropEnv._gpu_counter = 0

        self.assigned_gpu = self.nima_gpu_list[CropEnv._gpu_counter % len(self.nima_gpu_list)]
        CropEnv._gpu_counter += 1    
        # 使用全局缓存

        self.score_cache = CropEnv._global_score_cache
        if CropEnv._batch_scorer is None:
            CropEnv._batch_scorer = aesthetic_model
        
        # 性能优化参数
        self.no_op_penalty = cfg.reward.get("no_op_penalty", 0.005) # 无操作惩罚
        self.history = set()
        self.repeat_count = 0
        self.max_repeat = cfg.reward.get("max_repeat", 10)        # 最大重复次数
        self.repeat_penalty = cfg.reward.get("repeat_penalty", 0.1) # 重复惩罚
        self.visit_penalty = cfg.reward.get("visit_penalty", 0.01) # 访问惩罚
        
        # 每N步进行真实NIMA评分，其余使用估计
        self.real_score_interval = cfg.nima.get("real_score_interval", 5) # 真实NIMA评分间隔
        self.max_cache_size = cfg.nima.get("cache_size", 50000) # 最大缓存大小
        self.cache_cleanup_interval = 100 # 缓存清理间隔
        self.precomputed_crops = {}
        
        self.reset()

    def _get_crop_hash(self, box):
        return hashlib.md5(str(tuple(np.round(box, 3))).encode()).hexdigest()[:16]

    def _safe_nima_score(self, img, source_gpu=None):
        try:
            if source_gpu is None:
                source_gpu = self.assigned_gpu
            score = self.model(img )
            
            if np.isnan(score) or np.isinf(score):
                print(f"! NIMA返回异常值: {score}，使用默认分数5.0")
                return 5.0
            
            score = max(1.0, min(10.0, float(score)))
            return score
            
        except Exception as e:
            print(f"! NIMA评分失败: {e}，使用默认分数5.0")
            return 5.0
            
    def load_image(self, img: Image.Image):
        """切换到新图像并重置环境"""
        self.orig = img.convert("RGB")
        self.precomputed_crops.clear()
        return self.reset()

    def reset(self):
        orig_ratio = self.orig.width / self.orig.height
        scale = np.random.uniform(0.3, 0.8) # 随机缩放比例
        if orig_ratio >= 1:
            # 宽图，w最大为原图宽*scale，h按比例缩放
            w0 = max(10, self.orig.width * scale)
            h0 = max(10, w0 / orig_ratio)
        else:
            # 高图，h最大为原图高*scale，w按比例缩放
            h0 = max(10, self.orig.height * scale)
            w0 = max(10, h0 * orig_ratio)

        # 随机放置初始框
        x0 = np.random.uniform(0, self.orig.width - w0)
        y0 = np.random.uniform(0, self.orig.height - h0)

        x0 = max(0, min(x0, self.orig.width - w0))
        y0 = max(0, min(y0, self.orig.height - h0))

        self.box = np.array([x0, y0, w0, h0], dtype=float)

        if not self._is_valid_box(self.box):
            print(f"! 重置时生成无效box: {self.box}，使用默认box")
            self.box = np.array([0, 0, self.orig.width * 0.6, self.orig.height * 0.6], dtype=float)
        
        start_crop = (
            self.orig
            .crop((x0, y0, x0 + w0, y0 + h0))
            .resize((self.img_size, self.img_size))
        )

        if self.inference:
            init_score = 5.0
        else:
            init_score = self._safe_nima_score(start_crop)
        
        self.prev_score = init_score
        self.best_score = init_score

        self.history = set()
        self.repeat_count = 0
        self.step_count = 0

        if len(self.score_cache) > self.max_cache_size:   # 清理过大的缓存
            keys_to_remove = list(self.score_cache.keys())[:-self.max_cache_size//2]
            for key in keys_to_remove:
                del self.score_cache[key]
    
        return self._state()
    
    def _is_valid_box(self, box):
        x, y, w, h = box
        
        # 检查NaN或inf
        if np.isnan(box).any() or np.isinf(box).any():
            return False
        
        # 检查尺寸
        if w <= 0 or h <= 0:
            return False
        
        # 检查边界
        if x < 0 or y < 0:
            return False
        
        if x + w > self.orig.width or y + h > self.orig.height:
            return False
        
        return True

    def _get_score(self, box):
        if self.inference:
            return self.prev_score
        x, y, w, h = box
        crop_key = self._get_crop_hash(box)
        
        # 优先使用缓存
        if crop_key in self.score_cache:
            return self.score_cache[crop_key]
        
        # 按间隔决定是否真实评分
        if self.step_count % self.real_score_interval == 0:
            try:
                cropped = self.orig.crop((x, y, x + w, y + h)).resize(
                    (self.img_size, self.img_size))
                score = self._safe_nima_score(cropped)
                self.score_cache[crop_key] = score
                return score
            except Exception as e:
                print(f"! 实时NIMA评分失败: {e}")
                return self.prev_score
        else:
            # 使用估计分数
            return self._estimate_score_safe(box)
    
    def step(self, action_idx: int):
        dx = self.delta * self.box[2]
        dy = self.delta * self.box[3]
        reward = 0.0

        if np.isnan(dx) or np.isnan(dy) or np.isinf(dx) or np.isinf(dy):
            print(f"! 无效的动作增量: dx={dx}, dy={dy}")
            dx = dy = 1.0

        act = self.actions[action_idx]
        if not hasattr(self, "last_action"):
            self.last_action = None
            self.same_action_run = 0
        if act == self.last_action:
            self.same_action_run += 1
        else:
            self.same_action_run = 0
        self.last_action = act
        old_box = self.box.copy()
        old_score = self.prev_score

        if act == "stop":
            final_score = self._get_score(self.box)
            improv = final_score - self.best_score
            
            # stop奖励：与最佳分数比较
            #reward = final_score - self.best_score
            reward = (final_score - self.prev_score) * 2.0

            if self.step_count < 5:
                reward -= 1 # 0.5
                
            if improv < 0.02:
                reward -= 0.3
            # 鼓励适时stop
            #if self.step_count >= 15:
            #    #reward += 0.1  # 适时stop的小奖励
            #    reward += 0.2
            
            done = True
            return self._state(), reward, done, {}

        # 执行动作
        x, y, w, h = old_box
        if act == "left":
            x = max(0, x - dx)
        elif act == "right":
            x = min(self.orig.width - w, x + dx)
        elif act == "up":
            y = max(0, y - dy)
        elif act == "down":
            y = min(self.orig.height - h, y + dy)
        elif act == "zoom_in":
            w *= (1 - self.delta); h *= (1 - self.delta)
        elif act == "zoom_out":
            w *= (1 + self.delta); h *= (1 + self.delta)
        elif act == "wider":
            w *= (1 + self.delta)
        elif act == "narrower":
            w *= (1 - self.delta)
        elif act == "taller":
            h *= (1 + self.delta)
        elif act == "shorter":
            h *= (1 - self.delta)

        hit_boundary = False
        bx0, by0 = x, y
        if x < 0: x = 0; hit_boundary = True
        if y < 0: y = 0; hit_boundary = True
        if x + w > self.orig.width: x = self.orig.width - w; hit_boundary = True
        if y + h > self.orig.height: y = self.orig.height - h; hit_boundary = True

        # 应用边界约束
        min_size = max(10, min(self.orig.width, self.orig.height) * 0.05)
        w = max(min_size, min(self.orig.width - x, max(w, self.delta * self.orig.width)))
        h = max(min_size,min(self.orig.height - y, max(h, self.delta * self.orig.height)))

        new_box = np.array([x, y, w, h], dtype=float)

        if not self._is_valid_box(new_box):
            print(f"! 动作{act}产生无效box: {new_box}，保持原box")
            new_box = old_box.copy()

        self.box = new_box
        changed = not np.allclose(old_box, new_box)

        if self.inference:
            # 不计算评分与奖励, 只维护状态
            self.step_count += 1
            done = (self.step_count >= self.max_steps) or (self.actions[action_idx] == "stop")
            return self._state(), 0.0, done, {}

        if not changed:
            self.repeat_count += 1
            base_reward = -self.no_op_penalty
            new_score = self.prev_score
        else:
            self.repeat_count = 0  # 重置重复计数
            new_score = self._get_score(new_box)
            score_diff = new_score - self.prev_score

            #base_reward = np.tanh(score_diff * 2.0)
            base_reward = score_diff

            move_bonus = 0.0
            if act in {"left","right","up","down"}:
                # 位移距离归一化
                dist = (abs(new_box[0]-old_box[0]) + abs(new_box[1]-old_box[1])) / (self.orig.width + self.orig.height)
                if dist > 0:
                    move_bonus += 0.05 + 0.25 * dist
                # 新区域奖励（哈希未出现）
                key = self._get_crop_hash(new_box)
                if key not in self.history:
                    move_bonus += 0.05
            base_reward += move_bonus

            # 震荡检测惩罚
            if not hasattr(self, "recent_actions"):
                self.recent_actions = []
            self.recent_actions.append(act)
            if len(self.recent_actions) > 12:
                self.recent_actions.pop(0)
            oscillation_pairs = {
                ("wider","narrower"),("narrower","wider"),
                ("taller","shorter"),("shorter","taller"),
                ("zoom_in","zoom_out"),("zoom_out","zoom_in")
            }
            osc = sum(1 for a,b in zip(self.recent_actions, self.recent_actions[1:]) if (a,b) in oscillation_pairs)
            if osc >= 5:
                base_reward -= 0.08
        
        if self.step_count % 10 == 0:
            print(f"Debug Step {self.step_count}: "
                f"score={new_score:.3f}, "
                f"score_diff={new_score-self.prev_score:.3f}, "
                f"repeat_count={self.repeat_count}")
        
        # 惩罚机制
        penalty = 0.0
        if hit_boundary:
            penalty += 0.02
        if self.same_action_run >= 6:
            penalty += 0.03
        if self.same_action_run >= 10:
            penalty += 0.06
        # 重复访问惩罚
        box_key = self._get_crop_hash(new_box)
        if box_key in self.history:
            penalty += self.visit_penalty  
        else:
            self.history.add(box_key)
        
        # 连续重复动作惩罚（大幅降低）
        if self.repeat_count >= self.max_repeat:
            penalty += self.repeat_penalty 
            self.repeat_count = 0
        
        reward = base_reward - penalty
        reward = np.clip(reward, -2.0, 3.0)  # 限制奖励范围
        
        self.prev_score = new_score
        self.step_count += 1
        if new_score > self.best_score:
            self.best_score = new_score
        done = (self.step_count >= self.max_steps)
    
        
        return self._state(), reward, done, {}
    
    def _estimate_score_safe(self, box):
        try:
            x, y, w, h = box
            if w <= 0 or h <= 0:
                return self.prev_score
            
            area_ratio = (w * h) / (self.orig.width * self.orig.height)

            if h == 0:
                aspect_ratio = 1.0
            else:
                aspect_ratio = w / h
            
            if np.isnan(area_ratio) or np.isinf(area_ratio):
                area_ratio = 0.4
            if np.isnan(aspect_ratio) or np.isinf(aspect_ratio):
                aspect_ratio = 1.618
            
            ideal_ratio = 1.618
            area_score = 1.0 - abs(area_ratio - 0.4)
            
            if ideal_ratio == 0:
                ratio_score = 0.0
            else:
                ratio_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
            
            if np.isnan(area_score) or np.isinf(area_score):
                area_score = 0.0
            if np.isnan(ratio_score) or np.isinf(ratio_score):
                ratio_score = 0.0
        
            score_change = 0.2 * area_score + 0.3 * ratio_score - 0.1
            
            if np.isnan(score_change) or np.isinf(score_change):
                score_change = 0.0
            
            estimated_score = max(1.0, min(10.0, self.prev_score + score_change))
            
            if np.isnan(estimated_score) or np.isinf(estimated_score):
                return self.prev_score
            
            return estimated_score
            
        except Exception as e:
            print(f"! 几何美学估计失败: {e}")
            return self.prev_score
    
    def _cleanup_cache(self):
        if len(self.score_cache) > self.max_cache_size:
            keys_to_keep = list(self.score_cache.keys())[-self.max_cache_size//2:]
            new_cache = {k: self.score_cache[k] for k in keys_to_keep}
            self.score_cache.clear()
            self.score_cache.update(new_cache)
            
        if len(self.precomputed_crops) > 50:
            self.precomputed_crops.clear()

    def _state(self):
        x, y, w, h = self.box
        iw, ih = self.orig.size
        try:
            state = np.array([x/iw, y/ih, w/iw, h/ih], dtype=np.float32)
            
            if np.isnan(state).any() or np.isinf(state).any():
                print(f"！ 状态包含NaN/inf: {state}")
                state = np.array([0.2, 0.2, 0.6, 0.6], dtype=np.float32)  # 默认状态
            
            state_tensor = torch.from_numpy(state).float()
        except Exception as e:
            print(f"！ 状态计算失败: {e}")
            state = np.array([0.2, 0.2, 0.6, 0.6], dtype=np.float32)
            state_tensor = torch.from_numpy(state).float()

        crop_key = self._get_crop_hash(self.box)
        if crop_key in self.precomputed_crops:
            img_tensor = self.precomputed_crops[crop_key]
        else:
            try:
                cropped = self.orig.crop((x, y, x + w, y + h)).resize(
                    (self.img_size, self.img_size)
                )
                img_tensor = T.ToTensor()(cropped)
                
                if len(self.precomputed_crops) < 100:
                    self.precomputed_crops[crop_key] = img_tensor
            except Exception as e:
                print(f"！ 图像tensor创建失败: {e}")
                img_tensor = torch.zeros(3, self.img_size, self.img_size)

        return img_tensor, state_tensor