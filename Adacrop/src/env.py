import numpy as np
import hashlib
from PIL import Image
import torch
import torchvision.transforms as T

class CropEnv:

    _global_score_cache = {}
    _pending_requests = {}
    _batch_scorer = None

    def __init__(self, img: Image.Image, aesthetic_model, cfg, inference=False, init_model=None, init_device=None):
        self.orig = img
        self.model = aesthetic_model
        self.init_model = init_model
        self.init_device = init_device
        self.init_with_pred_prob = float(cfg.env.get("init_with_pred_prob", 0.7))
        self.orig_init_jitter = float(cfg.env.get("init_box_jitter", 0.05))
        self.img_size = cfg.env["img_size"]  # 图像大小
        self.max_steps = cfg.env.get("max_steps", 200) # 最大步数
        self.delta = cfg.env["action_delta"] # 动作增量
        # self.actions = ["left","right","up","down",
        #                 "zoom_in","zoom_out", "wider","narrower",
        #                 "taller", "shorter", "stop"]
        self.actions = [
            "left", "right", "up", "down",
            "zoom_in", "zoom_out",
            "stop"
        ]
        self.inference = inference
  
        # 使用全局缓存

        self.score_cache = CropEnv._global_score_cache
        if CropEnv._batch_scorer is None:
            CropEnv._batch_scorer = aesthetic_model
        
        # 性能优化参数
        self.no_op_penalty = cfg.reward.get("no_op_penalty", 0.002) # 无操作惩罚
        self.history = set()
        self.repeat_count = 0
        self.max_repeat = cfg.reward.get("max_repeat", 10)        # 最大重复次数
        self.repeat_penalty = cfg.reward.get("repeat_penalty", 0.08) # 重复惩罚
        self.visit_penalty = cfg.reward.get("visit_penalty", 0.02) # 访问惩罚

        self.boundary_penalty = float(cfg.reward.get("boundary_penalty", 0.05))

        # stop shaping
        self.stop_early_step = int(cfg.reward.get("stop_early_step", 8))
        self.stop_early_penalty = float(cfg.reward.get("stop_early_penalty", 0.8))
        self.stop_close_best_eps1 = float(cfg.reward.get("stop_close_best_eps1", 0.01))
        self.stop_close_best_eps2 = float(cfg.reward.get("stop_close_best_eps2", 0.03))
        self.stop_close_best_eps3 = float(cfg.reward.get("stop_close_best_eps3", 0.08))
        self.stop_reward_eps1 = float(cfg.reward.get("stop_reward_eps1", 0.35))
        self.stop_reward_eps2 = float(cfg.reward.get("stop_reward_eps2", 0.15))
        self.stop_penalty_eps3 = float(cfg.reward.get("stop_penalty_eps3", 0.10))
        self.stop_penalty_far = float(cfg.reward.get("stop_penalty_far", 0.30))
        self.stop_prev_clip = float(cfg.reward.get("stop_prev_clip", 0.2))

        # oscillation
        self.osc_window = int(cfg.reward.get("osc_window", 12))
        self.osc_t1 = int(cfg.reward.get("osc_t1", 2))
        self.osc_t2 = int(cfg.reward.get("osc_t2", 4))
        self.osc_t3 = int(cfg.reward.get("osc_t3", 6))
        self.osc_p1 = float(cfg.reward.get("osc_p1", 0.06))
        self.osc_p2 = float(cfg.reward.get("osc_p2", 0.18))
        self.osc_p3 = float(cfg.reward.get("osc_p3", 0.35))

        # backtrack (回到上一步 box)
        self.backtrack_penalty = float(cfg.reward.get("backtrack_penalty", 0.20))

        # same-action run penalty
        self.same_action_t1 = int(cfg.reward.get("same_action_t1", 4))
        self.same_action_t2 = int(cfg.reward.get("same_action_t2", 6))
        self.same_action_p1 = float(cfg.reward.get("same_action_p1", 0.03))
        self.same_action_p2 = float(cfg.reward.get("same_action_p2", 0.08))

        # move bonus
        self.move_base = float(cfg.reward.get("move_base", 0.05))
        self.move_dist_scale = float(cfg.reward.get("move_dist_scale", 0.25))
        self.move_new_region_bonus = float(cfg.reward.get("move_new_region_bonus", 0.05))

        # zoom bonus
        self.zoom_bonus = float(cfg.reward.get("zoom_bonus", 0.06))
        self.zoom_allow_drop = float(cfg.reward.get("zoom_allow_drop", 0.01))
        
        # 每N步进行真实评分，其余使用估计
        self.real_score_interval = cfg.nima.get("real_score_interval", 5) # 真实评分间隔
        self.max_cache_size = cfg.nima.get("cache_size", 50000) # 最大缓存大小
        self.cache_cleanup_interval = 100 # 缓存清理间隔
        self.precomputed_crops = {}

        self.box = None
        self.prev_score = 5.0
        self.best_score = 5.0
        self.step_count = 0
        self.history = set()
        self.repeat_count = 0
        self.last_action = None
        self.same_action_run = 0
        self.recent_actions = []

        self._prev_box = None

        self.debug_log = bool(cfg.env.get("debug_log", False))
        self.debug_log_interval = int(cfg.env.get("debug_log_interval", 50))
        
        self.reset()

    def _random_init_box(self):
        orig_ratio = self.orig.width / self.orig.height
        scale = np.random.uniform(0.3, 0.8)
        if orig_ratio >= 1:
            w0 = max(10, self.orig.width * scale)
            h0 = max(10, w0 / orig_ratio)
        else:
            h0 = max(10, self.orig.height * scale)
            w0 = max(10, h0 * orig_ratio)

        x0 = np.random.uniform(0, self.orig.width - w0)
        y0 = np.random.uniform(0, self.orig.height - h0)

        x0 = max(0, min(x0, self.orig.width - w0))
        y0 = max(0, min(y0, self.orig.height - h0))
        return np.array([x0, y0, w0, h0], dtype=float)

    def _predict_init_box(self):
        if self.init_model is None:
            return None

        try:
            self.init_model.eval()
            img_resized = self.orig.resize((self.img_size, self.img_size))
            img_t = T.ToTensor()(img_resized).unsqueeze(0)
            if self.init_device is not None:
                img_t = img_t.to(self.init_device)

            with torch.no_grad():
                pred = self.init_model.backbone_forward(img_t).detach().cpu().numpy()[0]

            cx, cy, w, h = [float(v) for v in pred]

            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            w = np.clip(w, 0.05, 1.0)
            h = np.clip(h, 0.05, 1.0)

            x = (cx - 0.5 * w) * self.orig.width
            y = (cy - 0.5 * h) * self.orig.height
            bw = w * self.orig.width
            bh = h * self.orig.height

            jitter = self.orig_init_jitter
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter) * self.orig.width
                y += np.random.uniform(-jitter, jitter) * self.orig.height
                bw *= np.random.uniform(1.0 - jitter, 1.0 + jitter)
                bh *= np.random.uniform(1.0 - jitter, 1.0 + jitter)

            x = max(0.0, min(x, self.orig.width - bw))
            y = max(0.0, min(y, self.orig.height - bh))

            min_size = max(10, min(self.orig.width, self.orig.height) * 0.05)
            bw = max(min_size, min(bw, self.orig.width - x))
            bh = max(min_size, min(bh, self.orig.height - y))

            box = np.array([x, y, bw, bh], dtype=float)
            if self._is_valid_box(box):
                return box
            return None
        except Exception as e:
            print(f"! 预测初始框失败: {e}")
            return None

    def _get_crop_hash(self, box):
        return hashlib.md5(str(tuple(np.round(box, 3))).encode()).hexdigest()[:16]

    def _safe_score_box(self, box):
        try:
            x, y, w, h = [float(v) for v in box]

            if np.isnan([x, y, w, h]).any() or np.isinf([x, y, w, h]).any():
                print(f"! 非法box(NaN/Inf): {box}")
                return 5.0

            x = max(0.0, min(x, self.orig.width - 1.0))
            y = max(0.0, min(y, self.orig.height - 1.0))
            w = max(1.0, min(w, self.orig.width - x))
            h = max(1.0, min(h, self.orig.height - y))

            safe_box = [int(round(x)), int(round(y)), int(round(w)), int(round(h))]

            if hasattr(self.model, "score_box"):
                score = self.model.score_box(self.orig, safe_box)
            else:
                sx, sy, sw, sh = safe_box
                crop = self.orig.crop((sx, sy, sx + sw, sy + sh)).resize(
                    (self.img_size, self.img_size)
                )
                score = self.model(crop)
            
            if isinstance(score, (tuple, list)) and len(score) > 0:
                score = score[0]
            elif isinstance(score, dict) and len(score) > 0:
                score = score.get("score", next(iter(score.values())))

            score = float(score)

            if np.isnan(score) or np.isinf(score):
                print(f"! scorer返回异常值: {score}，使用默认分数5.0")
                return 5.0

            return score

        except Exception as e:
            print(f"! scorer评分失败: {e}，使用默认分数5.0")
            return 5.0

            
    def load_image(self, img: Image.Image):
        """切换到新图像并重置环境"""
        self.orig = img.convert("RGB")
        self.precomputed_crops.clear()
        return self.reset()

    def reset(self):
        use_pred = (
            (not self.inference)
            and self.init_model is not None
            and np.random.rand() < self.init_with_pred_prob
        )

        if use_pred:
            pred_box = self._predict_init_box()
            self.box = pred_box if pred_box is not None else self._random_init_box()
        else:
            self.box = self._random_init_box()

        if not self._is_valid_box(self.box):
            print(f"! 重置时生成无效box: {self.box}，使用默认box")
            self.box = np.array([0, 0, self.orig.width * 0.6, self.orig.height * 0.6], dtype=float)

        if self.inference:
            init_score = 5.0
        else:
            init_score = self._get_score(self.box)

        self.prev_score = init_score
        self.best_score = init_score

        self.history = set()
        self.repeat_count = 0
        self.step_count = 0
        self.last_action = None
        self.same_action_run = 0
        self.recent_actions = []

        self._prev_box = None

        if len(self.score_cache) > self.max_cache_size:
            keys_to_remove = list(self.score_cache.keys())[:-self.max_cache_size // 2]
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

        crop_key = self._get_crop_hash(box)

        if crop_key in self.score_cache:
            return self.score_cache[crop_key]

        if self.step_count % self.real_score_interval == 0:
            try:
                score = self._safe_score_box(box)
                self.score_cache[crop_key] = score
                return score
            except Exception as e:
                print(f"! 实时评分失败: {e}")
                return self.prev_score
        else:
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
            #improv = final_score - self.best_score
            gap_to_best = self.best_score - final_score  # >0 表示比历史最好差
            
            # stop奖励：与最佳分数比较
            #reward = final_score - self.best_score
            reward = 0.0
            if gap_to_best <= self.stop_close_best_eps1:
                reward += self.stop_reward_eps1
            elif gap_to_best <= self.stop_close_best_eps2:
                reward += self.stop_reward_eps2
            elif gap_to_best <= self.stop_close_best_eps3:
                reward -= self.stop_penalty_eps3
            else:
                reward -= self.stop_penalty_far

            if self.step_count < self.stop_early_step:
                reward -= self.stop_early_penalty

            reward += float(np.clip(final_score - self.prev_score, -self.stop_prev_clip, self.stop_prev_clip))

            done = True
            return self._state(), float(np.clip(reward, -2.0, 3.0)), done, {}

        # 执行动作
        x, y, w, h = old_box
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        if act == "left":
            x = max(0, x - dx)
        elif act == "right":
            x = min(self.orig.width - w, x + dx)
        elif act == "up":
            y = max(0, y - dy)
        elif act == "down":
            y = min(self.orig.height - h, y + dy)
        elif act == "zoom_in":
            w *= (1 - self.delta)
            h *= (1 - self.delta)
            x = cx - 0.5 * w
            y = cy - 0.5 * h
        elif act == "zoom_out":
            w *= (1 + self.delta)
            h *= (1 + self.delta)
            x = cx - 0.5 * w
            y = cy - 0.5 * h
        # elif act == "wider":
        #     w *= (1 + self.delta)
        # elif act == "narrower":
        #     w *= (1 - self.delta)
        # elif act == "taller":
        #     h *= (1 + self.delta)
        # elif act == "shorter":
        #     h *= (1 - self.delta)

        hit_boundary = False
        bx0, by0 = x, y
        min_size = max(10, min(self.orig.width, self.orig.height) * 0.05)

        w = max(min_size, min(w, float(self.orig.width)))
        h = max(min_size, min(h, float(self.orig.height)))

        x = min(max(0.0, x), self.orig.width - w)
        y = min(max(0.0, y), self.orig.height - h)

        hit_boundary = (
            x <= 0.0 or y <= 0.0 or
            x + w >= self.orig.width or
            y + h >= self.orig.height
        )
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
        
        # backtrack 惩罚
        backtrack_pen = 0.0
        if self._prev_box is not None:
            if np.allclose(new_box, self._prev_box, atol=1e-3):
                backtrack_pen = self.backtrack_penalty
        self._prev_box = old_box.copy()

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
            if act in {"left", "right", "up", "down"}:
                dist = (abs(new_box[0] - old_box[0]) + abs(new_box[1] - old_box[1])) / (self.orig.width + self.orig.height)
                if dist > 0:
                    move_bonus += self.move_base + self.move_dist_scale * dist
                key = self._get_crop_hash(new_box)
                if key not in self.history:
                    move_bonus += self.move_new_region_bonus

            if act in {"zoom_in", "zoom_out"}:
                if score_diff >= -self.zoom_allow_drop:
                    move_bonus += self.zoom_bonus
            
            base_reward += move_bonus

            # 震荡检测惩罚
            if not hasattr(self, "recent_actions"):
                self.recent_actions = []
            self.recent_actions.append(act)
            if len(self.recent_actions) > self.osc_window:
                self.recent_actions.pop(0)
            # oscillation_pairs = {
            #     ("wider","narrower"),("narrower","wider"),
            #     ("taller","shorter"),("shorter","taller"),
            #     ("zoom_in","zoom_out"),("zoom_out","zoom_in")
            # }
            oscillation_pairs = {
                ("left", "right"), ("right", "left"),
                ("up", "down"), ("down", "up"),
                ("zoom_in", "zoom_out"), ("zoom_out", "zoom_in"),
            }
            # osc = sum(1 for a,b in zip(self.recent_actions, self.recent_actions[1:]) if (a,b) in oscillation_pairs)
            # if osc >= 5:
            #     base_reward -= 0.08

            osc = sum(
                1
                for a, b in zip(self.recent_actions, self.recent_actions[1:])
                if (a, b) in oscillation_pairs
            )
            if osc >= 2:
                base_reward -= 0.06
            if osc >= 4:
                base_reward -= 0.18
            if osc >= 6:
                base_reward -= 0.35
        
        if self.debug_log and self.step_count % self.debug_log_interval == 0:
            print(
                f"[EnvDebug] step={self.step_count} act={act} "
                f"score={new_score:.3f} diff={new_score-old_score:.3f} "
                f"repeat={self.repeat_count}"
            )
        
        # 惩罚机制
        penalty = 0.0
        if hit_boundary:
            penalty += self.boundary_penalty

        if self.same_action_run >= self.same_action_t1:
            penalty += self.same_action_p1
        if self.same_action_run >= self.same_action_t2:
            penalty += self.same_action_p2

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
        
        reward = base_reward - penalty - backtrack_pen
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

        state = np.array([
            (x + 0.5 * w) / iw,
            (y + 0.5 * h) / ih,
            w / iw,
            h / ih,
        ], dtype=np.float32)

        if np.isnan(state).any() or np.isinf(state).any():
            print(f"！ 状态包含NaN/inf: {state}")
            state = np.array([0.5, 0.5, 0.6, 0.6], dtype=np.float32)

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