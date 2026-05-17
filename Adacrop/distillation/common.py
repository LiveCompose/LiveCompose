import json
import math
import pathlib
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models


ACTIONS = ["left", "right", "up", "down", "zoom_in", "zoom_out", "stop"]


def find_adacrop_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strip_adacrop_prefix(path_text: str) -> str:
    path_text = path_text.replace("\\", "/")
    if path_text.startswith("./"):
        path_text = path_text[2:]
    if path_text.startswith("Adacrop/"):
        path_text = path_text[len("Adacrop/") :]
    return path_text


def resolve_image_path(raw_path: str, adacrop_root: Path, source_file: Optional[Path] = None) -> Path:
    """Resolve mixed project paths, including JSONL paths like ./outpainted/a.png."""
    raw = str(raw_path).replace("\\", "/")
    candidates: List[Path] = []

    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)

    if source_file is not None:
        candidates.append(source_file.parent / raw)
        if raw.startswith("./"):
            candidates.append(source_file.parent / raw[2:])

    stripped = _strip_adacrop_prefix(raw)
    candidates.append(adacrop_root / stripped)
    candidates.append(adacrop_root.parent / raw)

    # Old merged JSONs may contain Adacrop/data/outpainted/foo.png, while this
    # workspace stores those files under data/outpainted_dataset/outpainted.
    if stripped.startswith("data/outpainted/"):
        suffix = stripped[len("data/outpainted/") :]
        candidates.append(adacrop_root / "data" / "outpainted_dataset" / "outpainted" / suffix)

    # The outpainted JSONL stores paths as ./outpainted/foo.png relative to the
    # JSONL file: data/outpainted_dataset/training_pairs.jsonl.
    if stripped.startswith("outpainted/"):
        candidates.append(adacrop_root / "data" / "outpainted_dataset" / stripped)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return candidates[0].resolve()


def normalize_boxes(value) -> List[List[float]]:
    if value is None:
        return []
    if isinstance(value, dict):
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return [[float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])]]
        if all(k in value for k in ("x", "y", "w", "h")):
            x, y, w, h = float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"])
            return [[x, y, x + w, y + h]]
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            return [[float(v) for v in value]]
        boxes: List[List[float]] = []
        for item in value:
            boxes.extend(normalize_boxes(item))
        return boxes
    return []


def canonical_box_xyxy(box: Sequence[float], width: int, height: int, img_path: Optional[str] = None) -> List[float]:
    """Return a pixel-space [x1,y1,x2,y2] box.

    The outpainted JSONL is xyxy, while the CUHK split files in this workspace
    use yxyx-like coordinates. Use the image path when it is unambiguous, then
    fall back to bounds checks.
    """
    a, b, c, d = [float(v) for v in box]
    path_text = (img_path or "").replace("\\", "/").lower()

    if "cuhk_images" in path_text:
        x1, y1, x2, y2 = b, a, d, c
    elif "outpainted" in path_text or "gaic_dataset" in path_text:
        x1, y1, x2, y2 = a, b, c, d
    else:
        xyxy_valid = 0 <= a < c <= width and 0 <= b < d <= height
        yxyx_valid = 0 <= b < d <= width and 0 <= a < c <= height
        if yxyx_valid and not xyxy_valid:
            x1, y1, x2, y2 = b, a, d, c
        else:
            x1, y1, x2, y2 = a, b, c, d

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [x1, y1, x2, y2]


def load_records(path: Path, adacrop_root: Path, require_images: bool = True) -> List[Dict]:
    path = Path(path)
    rows: List[Dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)

    records: List[Dict] = []
    for row in rows:
        raw_img = row.get("img") or row.get("file")
        if not raw_img:
            continue
        img_path = resolve_image_path(raw_img, adacrop_root, source_file=path)
        if require_images and not img_path.exists():
            continue
        boxes = normalize_boxes(row.get("box") or row.get("boxes") or row.get("orig_bbox"))
        records.append({"img": str(img_path), "boxes": boxes, "raw": row})
    return records


def resnet50_no_weights():
    try:
        return models.resnet50(weights=None)
    except TypeError:
        return models.resnet50(pretrained=False)


def mobilenet_v3_no_weights(arch: str):
    if arch == "mobilenet_v3_large":
        try:
            return models.mobilenet_v3_large(weights=None)
        except TypeError:
            return models.mobilenet_v3_large(pretrained=False)
    if arch == "mobilenet_v3_small":
        try:
            return models.mobilenet_v3_small(weights=None)
        except TypeError:
            return models.mobilenet_v3_small(pretrained=False)
    raise ValueError(f"Unsupported student arch: {arch}")


class TeacherActorCritic(nn.Module):
    def __init__(self, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.backbone = resnet50_no_weights()
        self.backbone.fc = nn.Identity()
        feat_dim = 2048
        self.actor = nn.Sequential(
            nn.Linear(feat_dim + 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(feat_dim + 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )
        self.bbox_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, 4))

    def forward(self, img_tensor: torch.Tensor, state: torch.Tensor):
        feats = self.backbone(img_tensor)
        x = torch.cat([feats, state], dim=1)
        logits = self.actor(x)
        return F.softmax(logits, dim=1), self.critic(x)

    def backbone_forward(self, img_tensor: torch.Tensor):
        feats = self.backbone(img_tensor)
        return self.bbox_head(feats)


class MobileNetPolicy(nn.Module):
    def __init__(self, arch: str = "mobilenet_v3_small", n_actions: int = len(ACTIONS)):
        super().__init__()
        base = mobilenet_v3_no_weights(arch)
        self.arch = arch
        self.features = base.features
        self.avgpool = base.avgpool
        feat_dim = base.classifier[0].in_features
        self.actor = nn.Sequential(
            nn.Linear(feat_dim + 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

    def extract_feats(self, img_tensor: torch.Tensor):
        feats = self.features(img_tensor)
        feats = self.avgpool(feats)
        return torch.flatten(feats, 1)

    def forward(self, img_tensor: torch.Tensor, state: torch.Tensor):
        feats = self.extract_feats(img_tensor)
        logits = self.actor(torch.cat([feats, state], dim=1))
        return F.softmax(logits, dim=1), logits

    def backbone_forward(self, img_tensor: torch.Tensor):
        feats = self.extract_feats(img_tensor)
        return torch.sigmoid(self.bbox_head(feats))


def load_teacher(ckpt_path: Path, device: torch.device) -> TeacherActorCritic:
    ckpt = torch_load_portable(ckpt_path)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = TeacherActorCritic(n_actions=len(ACTIONS))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[teacher] unexpected keys: {unexpected[:8]}")
    missing_required = [k for k in missing if not k.startswith("critic.") and not k.startswith("bbox_head.")]
    if missing_required:
        raise RuntimeError(f"Teacher checkpoint missing required keys: {missing_required[:8]}")
    return model.to(device).eval()


def load_student(ckpt_path: Path, device: torch.device, arch: Optional[str] = None) -> MobileNetPolicy:
    ckpt = torch_load_portable(ckpt_path)
    ckpt_arch = ckpt.get("arch", arch or "mobilenet_v3_small")
    model = MobileNetPolicy(arch=ckpt_arch, n_actions=len(ACTIONS))
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def torch_load_portable(ckpt_path: Path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except NotImplementedError as exc:
        if "WindowsPath" not in str(exc):
            raise
        # Checkpoints saved on Windows may pickle pathlib.WindowsPath inside
        # metadata such as args. On POSIX, remap it before loading.
        pathlib.WindowsPath = pathlib.PosixPath
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 1e-8 else inter / union


def clamp_xywh(box: Sequence[float], width: int, height: int, delta: float = 0.05) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    min_size = max(10.0, min(width, height) * 0.05)
    w = max(min_size, min(w, float(width)))
    h = max(min_size, min(h, float(height)))
    x = min(max(0.0, x), float(width) - w)
    y = min(max(0.0, y), float(height) - h)
    w = max(min_size, min(float(width) - x, max(w, delta * width)))
    h = max(min_size, min(float(height) - y, max(h, delta * height)))
    return [x, y, w, h]


def random_box(width: int, height: int) -> List[float]:
    ratio = width / max(1, height)
    scale = random.uniform(0.3, 0.8)
    if ratio >= 1:
        w = max(10.0, width * scale)
        h = max(10.0, w / ratio)
    else:
        h = max(10.0, height * scale)
        w = max(10.0, h * ratio)
    x = random.uniform(0.0, max(1.0, width - w))
    y = random.uniform(0.0, max(1.0, height - h))
    return clamp_xywh([x, y, w, h], width, height)


def jitter_box(box_xywh: Sequence[float], width: int, height: int, jitter: float = 0.12) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    x += random.uniform(-jitter, jitter) * width
    y += random.uniform(-jitter, jitter) * height
    w *= random.uniform(1.0 - jitter, 1.0 + jitter)
    h *= random.uniform(1.0 - jitter, 1.0 + jitter)
    return clamp_xywh([x, y, w, h], width, height)


def box_state(box_xywh: Sequence[float], width: int, height: int) -> torch.Tensor:
    x, y, w, h = [float(v) for v in box_xywh]
    state = [
        (x + 0.5 * w) / max(1.0, width),
        (y + 0.5 * h) / max(1.0, height),
        w / max(1.0, width),
        h / max(1.0, height),
    ]
    if not all(math.isfinite(v) for v in state):
        state = [0.5, 0.5, 0.6, 0.6]
    return torch.tensor(state, dtype=torch.float32)


def render_crop(img: Image.Image, box_xywh: Sequence[float], img_size: int) -> torch.Tensor:
    x, y, w, h = [float(v) for v in box_xywh]
    crop = img.crop((x, y, x + w, y + h)).resize((img_size, img_size))
    return T.ToTensor()(crop)


def render_full_image(img: Image.Image, img_size: int) -> torch.Tensor:
    return T.ToTensor()(img.resize((img_size, img_size)))


def bbox_target_from_xyxy(box_xyxy: Sequence[float], width: int, height: int) -> torch.Tensor:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    target = [
        ((x1 + x2) * 0.5) / max(1.0, width),
        ((y1 + y2) * 0.5) / max(1.0, height),
        max(1.0, x2 - x1) / max(1.0, width),
        max(1.0, y2 - y1) / max(1.0, height),
    ]
    return torch.tensor([min(1.0, max(0.0, v)) for v in target], dtype=torch.float32)


def bbox_cxcywh_to_xyxy(box_cxcywh: Sequence[float], width: int, height: int) -> List[float]:
    cx, cy, w, h = [float(v) for v in box_cxcywh]
    bw = w * width
    bh = h * height
    x1 = cx * width - 0.5 * bw
    y1 = cy * height - 0.5 * bh
    x2 = x1 + bw
    y2 = y1 + bh
    return [
        min(max(0.0, x1), float(width)),
        min(max(0.0, y1), float(height)),
        min(max(0.0, x2), float(width)),
        min(max(0.0, y2), float(height)),
    ]


def step_box(box_xywh: Sequence[float], action_idx: int, width: int, height: int, delta: float = 0.05) -> List[float]:
    act = ACTIONS[int(action_idx)]
    x, y, w, h = [float(v) for v in box_xywh]
    dx, dy = delta * w, delta * h
    cx, cy = x + 0.5 * w, y + 0.5 * h
    if act == "left":
        x = max(0.0, x - dx)
    elif act == "right":
        x = min(width - w, x + dx)
    elif act == "up":
        y = max(0.0, y - dy)
    elif act == "down":
        y = min(height - h, y + dy)
    elif act == "zoom_in":
        w *= 1.0 - delta
        h *= 1.0 - delta
        x = cx - 0.5 * w
        y = cy - 0.5 * h
    elif act == "zoom_out":
        w *= 1.0 + delta
        h *= 1.0 + delta
        x = cx - 0.5 * w
        y = cy - 0.5 * h
    return clamp_xywh([x, y, w, h], width, height, delta=delta)


class PolicyStateDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict],
        img_size: int = 224,
        samples_per_image: int = 1,
        random_box_prob: float = 0.65,
        jitter: float = 0.12,
    ):
        self.records = list(records)
        self.img_size = int(img_size)
        self.samples_per_image = max(1, int(samples_per_image))
        self.random_box_prob = float(random_box_prob)
        self.jitter = float(jitter)

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_image

    def __getitem__(self, idx: int):
        rec = self.records[idx % len(self.records)]
        img = Image.open(rec["img"]).convert("RGB")
        width, height = img.size
        boxes = rec.get("boxes") or []

        if boxes and random.random() > self.random_box_prob:
            gt_box = canonical_box_xyxy(random.choice(boxes), width, height, img_path=rec["img"])
            box = jitter_box(xyxy_to_xywh(gt_box), width, height, jitter=self.jitter)
        else:
            box = random_box(width, height)

        return render_crop(img, box, self.img_size), box_state(box, width, height)


class BBoxDataset(Dataset):
    def __init__(self, records: Sequence[Dict], img_size: int = 224, samples_per_image: int = 1):
        self.records = [r for r in records if r.get("boxes")]
        self.img_size = int(img_size)
        self.samples_per_image = max(1, int(samples_per_image))

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_image

    def __getitem__(self, idx: int):
        rec = self.records[idx % len(self.records)]
        img = Image.open(rec["img"]).convert("RGB")
        width, height = img.size
        box = canonical_box_xyxy(random.choice(rec["boxes"]), width, height, img_path=rec["img"])
        return render_full_image(img, self.img_size), bbox_target_from_xyxy(box, width, height)


class BBoxEvalDataset(Dataset):
    def __init__(self, records: Sequence[Dict], img_size: int = 224):
        self.records = [r for r in records if r.get("boxes")]
        self.img_size = int(img_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["img"]).convert("RGB")
        width, height = img.size
        targets = torch.stack(
            [
                bbox_target_from_xyxy(canonical_box_xyxy(box, width, height, img_path=rec["img"]), width, height)
                for box in rec["boxes"]
            ]
        )
        return render_full_image(img, self.img_size), targets


def soften_probs(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 1.0:
        return probs
    softened = probs.clamp_min(1e-8).pow(1.0 / temperature)
    return softened / softened.sum(dim=1, keepdim=True)
