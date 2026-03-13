import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

def parse_txt_annotations(txt_path: str, img_root: str):
    """
     records ：
      [
        {
          "img": "/full/path/to/1116.jpg",
          "boxes": [ [x1,y1,x2,y2], [x1,y1,x2,y2], [x1,y1,x2,y2] ]
        },
        …
      ]
    """
    records = []
    lines = [l.strip() for l in open(txt_path, "r", encoding="utf-8") if l.strip()]
    i = 0
    while i < len(lines):
        # 1. 图片行，例如 "animal\1116.jpg" 或 "class/11425.jpg"
        img_line = lines[i]
        img_fname = img_line.split("\\")[-1].split("/")[-1]
        img_path = os.path.join(img_root, img_fname)
        # 框坐标
        boxes = []
        for k in (1, 2, 3):
            xs_min, xs_max, ys_min, ys_max = map(int, lines[i + k].split())
            # 保证 x1<x2, y1<y2
            x1, x2 = sorted([xs_min, xs_max])
            y1, y2 = sorted([ys_min, ys_max])
            boxes.append([x1, y1, x2, y2])
        records.append({"img": img_path, "boxes": boxes})
        i += 4
    return records

def split_and_save(records, val_ratio=0.2, seed=42, out_folder="data/splits"):
    random.seed(seed)
    random.shuffle(records)
    n_val = int(len(records) * val_ratio)
    train_recs = records[n_val:]
    val_recs   = records[:n_val]

    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_recs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_folder, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_recs,   f, indent=2, ensure_ascii=False)

    return train_recs, val_recs

def letterbox_with_info(img: Image.Image, size: int, fill=0):
    """
    将 img 等比缩放到短边=size，然后在另—边做镜像填充到正方形。
    返回：填充后的 PIL.Image, 缩放比例, 左侧/顶部 填充像素数
    """
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # 计算 pad
    pad_w = size - new_w
    pad_h = size - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    img_padded = F.pad(
        img_resized,
        padding=(left, top, right, bottom),
        fill=fill,
        padding_mode="reflect"
    )
    return img_padded, scale, left, top

class SmallCropDataset(Dataset):
    def __init__(self, records, img_size: int = 224, transform=None):
        self.records = records
        self.tf = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.extra_tf = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["img"]).convert("RGB")
        W, H = img.size
        # 随机选一个框
        boxes = rec.get("box", rec.get("boxes", []))
        x1, y1, x2, y2 = random.choice(boxes)
        # 转成 (cx, cy, w, h)，并归一化到 [0,1]
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        img_lb, scale, pad_x, pad_y = letterbox_with_info(img, self.img_size)

        cx_pix = cx * W
        cy_pix = cy * H
        w_pix  = w * W
        h_pix  = h * H

        cx_new = (cx_pix * scale + pad_x) / self.img_size
        cy_new = (cy_pix * scale + pad_y) / self.img_size
        w_new  = (w_pix  * scale)          / self.img_size
        h_new  = (h_pix  * scale)          / self.img_size
        target = torch.tensor([cx_new, cy_new, w_new, h_new],
                              dtype=torch.float32)

        if self.extra_tf:
            img_lb = self.extra_tf(img_lb)
        img_t = self.to_tensor(img_lb)

        return img_t, target


def make_loaders(txt_path: str,
                 img_root: str,
                 img_size=224,
                 batch_size=16,
                 val_split=0.2):
    """
    从 txt 构建 train/val DataLoader
    """
    records = parse_txt_annotations(txt_path, img_root)
    train_recs, val_recs = split_and_save(records,
                                          val_ratio=val_split,
                                          out_folder="data/splits")
    train_ds = SmallCropDataset(train_recs, img_size)
    val_ds   = SmallCropDataset(val_recs,   img_size)
    train_loader = DataLoader(train_ds, batch_size,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size,
                              shuffle=False, num_workers=2)
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试
    TL, VL = make_loaders(
        txt_path="/home/Zyan/codes/DH_LiveCompose/Adacrop/data/annotations.txt",
        img_root="/home/Zyan/codes/DH_LiveCompose/Adacrop/data/cuhk_images",
        img_size=224,
        batch_size=8
    )
    print("Train batches:", len(TL), " Val batches:", len(VL))
    for imgs, boxes in TL:
        print(imgs.shape, boxes.shape)
        break
