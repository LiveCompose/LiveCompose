import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class GAICAdapter:
    def __init__(
        self,
        repo_dir: str,
        ckpt_path: str,
        device: str = "cuda:0",
        backbone: str = "mobilenetv2",
        scale: str = "multi",
        alignsize: int = 9,
        reddim: int = None,
        loadweight: bool = False,
    ):
        self.repo_dir = str(Path(repo_dir).resolve())
        self.ckpt_path = str(Path(ckpt_path).resolve())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.scale = scale
        self.alignsize = alignsize
        self.loadweight = loadweight

        if reddim is None:
            if backbone in ("vgg16", "shufflenetv2"):
                reddim = 32
            elif backbone == "mobilenetv2":
                reddim = 16
            else:
                raise ValueError(f"Unsupported GAIC backbone: {backbone}")
        self.reddim = reddim

        if not os.path.isdir(self.repo_dir):
            raise FileNotFoundError(f"GAIC repo_dir not found: {self.repo_dir}")
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"GAIC ckpt not found: {self.ckpt_path}")

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        from networks.GAIC_model import build_crop_model

        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])

        self.model = build_crop_model(
            scale=self.scale,
            alignsize=self.alignsize,
            reddim=self.reddim,
            loadweight=self.loadweight,
            model=self.backbone,
        )

        state = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    def _resize_image_like_demo(self, img: Image.Image) -> Tuple[torch.Tensor, int, int]:
        """
        与 GAIC/evaluate/demo.py 一致：
        - 按短边缩放到 256
        - h/w round 到 32 的倍数
        - ToTensor + ImageNet Normalize
        """
        im_width, im_height = img.size
        scale = 256.0 / min(im_height, im_width)
        h = round(im_height * scale / 32.0) * 32
        w = round(im_width * scale / 32.0) * 32
        resized_image = img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        im_tensor = self.image_transformer(resized_image).unsqueeze(0).to(self.device)
        return im_tensor, w, h

    @staticmethod
    def _xywh_to_xyxy(box_xywh: List[float]):
        x, y, w, h = box_xywh
        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w)
        y2 = float(y + h)
        return [x1, y1, x2, y2]

    def _map_box_to_resized(self, box_xywh, orig_w: int, orig_h: int, resized_w: int, resized_h: int):
        x1, y1, x2, y2 = self._xywh_to_xyxy(box_xywh)

        sx = float(resized_w) / float(orig_w)
        sy = float(resized_h) / float(orig_h)

        rx1 = x1 * sx
        ry1 = y1 * sy
        rx2 = x2 * sx
        ry2 = y2 * sy

        rx1 = max(0.0, min(rx1, resized_w - 1))
        ry1 = max(0.0, min(ry1, resized_h - 1))
        rx2 = max(rx1 + 1.0, min(rx2, resized_w))
        ry2 = max(ry1 + 1.0, min(ry2, resized_h))

        return [rx1, ry1, rx2, ry2]

    def score_box(self, img: Image.Image, box_xywh) -> float:
        orig_w, orig_h = img.size
        im_tensor, resized_w, resized_h = self._resize_image_like_demo(img)

        roi_xyxy = self._map_box_to_resized(
            box_xywh=box_xywh,
            orig_w=orig_w,
            orig_h=orig_h,
            resized_w=resized_w,
            resized_h=resized_h,
        )

        rois_np = np.asarray([[0.0, roi_xyxy[0], roi_xyxy[1], roi_xyxy[2], roi_xyxy[3]]], dtype=np.float32)
        rois = torch.from_numpy(rois_np).to(self.device, non_blocking=True)

        with torch.no_grad():
            out = self.model(im_tensor, rois)

        out = out.reshape(-1)
        return float(out[0].item())