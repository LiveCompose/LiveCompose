import os
import math
import json
import numpy as np
from PIL import Image
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

try:
    from Adacrop.src.config import Config
except ModuleNotFoundError:
    try:
        from src.config import Config
    except ModuleNotFoundError:
        from config import Config

def mean_score(preds: np.ndarray) -> float:
    labels = np.arange(1, preds.shape[-1] + 1, dtype=np.float32)
    return float((preds * labels).sum())

def build_rank_preprocess(img_size: int = 224):
    # PairwiseRankScorer 的预处理，和 ImageNet 分类模型一致
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def crop_xywh(img: Image.Image, box_xywh):
    x, y, w, h = box_xywh
    W, H = img.size
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))

def full_box_xywh(img: Image.Image):
    return [0, 0, img.width, img.height]

class BaseCropScorer:
    """
    统一接口：
    - score(crop_img) -> float
    - score_box(orig_img, box_xywh) -> float
    """
    def score(self, img: Image.Image) -> float:
        raise NotImplementedError

    def score_box(self, orig_img: Image.Image, box_xywh) -> float:
        crop = crop_xywh(orig_img, box_xywh)
        if crop is None:
            return float("-inf")
        return self.score(crop)

    def __call__(self, img: Image.Image) -> float:
        return self.score(img)

class OnnxTorchNIMAScorer:
    """使用 ONNX→PyTorch 转换后的 NIMA 模型进行打分（仅 PyTorch 依赖）"""
    def __init__(self, onnx_path: str, device: str = "cuda:0"):
        from onnx2torch import convert
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = convert(onnx_path).to(self.device).eval()
        self.input_size = 299

    @staticmethod
    def _preprocess_pil(img: Image.Image, size: int) -> torch.Tensor:
        arr = np.array(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        t = (t - 0.5) * 2.0                         # [-1,1]
        return t

    def __call__(self, img: Image.Image) -> float:
        with torch.no_grad():
            x = self._preprocess_pil(img, self.input_size).unsqueeze(0).to(self.device)
            out = self.model(x)
            if out.shape[-1] != 10:
                out = torch.softmax(out, dim=1)
            labels = torch.arange(1, 11, dtype=torch.float32, device=self.device)
            score = (out * labels).sum(dim=1)
            return float(score.clamp(1.0, 10.0).item())

    score = __call__


class PairwiseRankScorer(nn.Module):
    """
    单图标量分数 s(img)。训练时使用 pairwise loss。

    - 训练：forward(x: Tensor[B,3,H,W]) -> Tensor[B,1]
    - 推理：score(pil_img) -> float
    """
    def __init__(self, device="cuda:0", backbone_name="resnet50", freeze_backbone=True, img_size=224):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name in ("efficientnet_b0", "efn_b0"):
            backbone = models.efficientnet_b0(pretrained=True)
            feat_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        self.preprocess = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.to(self.device)
        self.freeze_backbone(freeze_backbone)
        self.eval()

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)  # [B,1]

    @torch.no_grad()
    def score(self, img: Image.Image) -> float:
        self.eval()
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device, non_blocking=True)
        s = self.forward(x)
        return float(s.item())

    @torch.no_grad()
    def prob_better(self, img_a: Image.Image, img_b: Image.Image) -> float:
        sa = self.score(img_a)
        sb = self.score(img_b)
        return float(1.0 / (1.0 + math.exp(-(sa - sb))))

class GAICScorer(BaseCropScorer):
    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:0",
        repo_dir: str = "",
        backbone: str = "vgg16",  # 可选：'mobilenetv2', 'vgg16'
        normalize_to_nima: bool = True,
    ):
        if not repo_dir:
            raise ValueError("GAICScorer requires repo_dir")
        try:
            from Adacrop.src.gaic_adapter import GAICAdapter
        except ModuleNotFoundError:
            try:
                from src.gaic_adapter import GAICAdapter
            except ModuleNotFoundError:
                from gaic_adapter import GAICAdapter

        self.normalize_to_nima = normalize_to_nima 
        self._lock = threading.Lock()     
        self.adapter = GAICAdapter(
            repo_dir=repo_dir,
            ckpt_path=model_path,
            device=device,
            backbone=backbone,
        )

    @staticmethod
    def _gaic_to_nima_scale(score: float) -> float:
        # GAIC: 1~5 -> unified: 0~10
        score = max(1.0, min(5.0, float(score)))
        return (score - 1.0) / 4.0 * 10.0

    def score(self, img: Image.Image) -> float:
        raise NotImplementedError("GAIC 更适合使用 score_box(orig_img, box_xywh)")

    def score_box(self, orig_img: Image.Image, box_xywh) -> float:
        with self._lock:
            raw_score = float(self.adapter.score_box(orig_img, box_xywh))
        if self.normalize_to_nima:
            return self._gaic_to_nima_scale(raw_score)
        return raw_score


def load_rank_scorer(path: str, device="cuda:0", backbone_name="resnet50", img_size=224) -> PairwiseRankScorer:
    m = PairwiseRankScorer(device=device, backbone_name=backbone_name, freeze_backbone=True, img_size=img_size)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    m.load_state_dict(sd, strict=True)
    m.to(m.device).eval()
    return m


class _PILScoreWrapper(BaseCropScorer):
    """
    统一 env 侧接口：scorer(img: PIL) -> float
    """
    def __init__(self, model):
        self.model = model

    def score(self, img: Image.Image) -> float:
        return self.model.score(img)

    def score_box(self, orig_img: Image.Image, box_xywh) -> float:
        if hasattr(self.model, "score_box"):
            return self.model.score_box(orig_img, box_xywh)
        return super().score_box(orig_img, box_xywh)


def load_aesthetic_model():
    """
    统一入口：返回一个可调用对象 scorer(img)->float
    """
    cfg = Config()
    scorer_type = str(cfg.nima.get("scorer_type", "")).lower().strip()

    if scorer_type in ("rank", "rankbased", "pairwise_rank"):
        device_id = int(cfg.nima.get("device_id", 0))
        device = f"cuda:{device_id}"
        backbone_name = str(cfg.nima.get("rank_backbone", "resnet50"))
        img_size = int(cfg.nima.get("rank_img_size", 224))
        ckpt_path = str(cfg.nima.get("rank_ckpt", "")).strip()

        if ckpt_path:
            print(f"* Using RankBased scorer (ckpt): {ckpt_path}")
            model = load_rank_scorer(ckpt_path, device=device, backbone_name=backbone_name, img_size=img_size)
        else:
            freeze = bool(cfg.nima.get("rank_freeze_backbone", True))
            print(f"* Using RankBased scorer (untrained): backbone={backbone_name}, freeze={freeze}, device={device}")
            model = PairwiseRankScorer(device=device, backbone_name=backbone_name, freeze_backbone=freeze, img_size=img_size)

        return _PILScoreWrapper(model)

    if scorer_type in ("gaic",):
        device_id = int(cfg.nima.get("device_id", 0))
        device = f"cuda:{device_id}"
        model_path = str(cfg.nima.get("gaic_ckpt", "")).strip()
        repo_dir = str(cfg.nima.get("gaic_repo_dir", "")).strip()
        backbone = str(cfg.nima.get("gaic_backbone", "vgg16")).strip()
        normalize_to_nima = bool(cfg.nima.get("normalize_to_nima_scale", True))
        print(f"* Using GAIC scorer: repo={repo_dir}, ckpt={model_path}, backbone={backbone}")
        return _PILScoreWrapper(
            GAICScorer(
                model_path=model_path,
                device=device,
                repo_dir=repo_dir,
                backbone=backbone,
                normalize_to_nima=normalize_to_nima,
            )
        )

    w = str(cfg.nima.get("weights_path", "../NIMA/weights/nima_inception_resnet.onnx"))
    if w.lower().endswith(".onnx"):
        device_id = int(cfg.nima.get("device_id", 0))
        return _PILScoreWrapper(OnnxTorchNIMAScorer(w, device=f"cuda:{device_id}"))

    raise RuntimeError(f"Unsupported scorer_type={scorer_type!r} and weights_path={w!r}")