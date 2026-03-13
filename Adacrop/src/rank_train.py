import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scorers import PairwiseRankScorer, build_rank_preprocess


def logistic_rank_loss(s_good: torch.Tensor, s_bad: torch.Tensor) -> torch.Tensor:
    s_good = s_good.view(-1)
    s_bad = s_bad.view(-1)
    return F.softplus(-(s_good - s_bad)).mean()

def _crop_xywh(img: Image.Image, box_xywh):
    """
    box_xywh: [x, y, w, h] (左上角 + 宽高)
    """
    W, H = img.size
    x, y, w, h = box_xywh

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

class PairwiseRankDataset(Dataset):

    def __init__(self, pairs, img_size=224, img_root =None):
        self.pairs = pairs
        self.tf = build_rank_preprocess(img_size)

        self.img_root = img_root

        self.good_keys = ["good", "orig", "original", "img", "gt"]
        self.bad_keys  = ["bad", "expanded", "extend", "aug", "outpaint"]

    def __len__(self):
        return len(self.pairs)

    def _get(self, rec, keys):
        for k in keys:
            if k in rec:
                return rec[k]
        return None

    def _resolve_img_path(self, p: str) -> str:
        # 绝对路径直接返回
        if os.path.isabs(p):
            return p

        if self.img_root is not None:
            # normalize to forward slashes for matching
            pn = p.replace("\\", "/")
            if pn.startswith("Adacrop/"):
                pn = pn[len("Adacrop/"):]
            return os.path.join(self.img_root, pn)

        return p

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        
        if "img" in rec and "good_box" in rec:
            img_path = self._resolve_img_path(rec["img"])
            img = Image.open(img_path).convert("RGB")

            good_img = _crop_xywh(img, rec["good_box"])

            bad_type = rec.get("bad_type", "full")
            bad_box = rec.get("bad_box", None)

            if bad_type == "full" or bad_box is None:
                bad_img = img
            else:
                bad_img = _crop_xywh(img, bad_box)

            return self.tf(good_img), self.tf(bad_img)
        
        good_path = self._get(rec, self.good_keys)
        bad_path  = self._get(rec, self.bad_keys)
        if good_path is None or bad_path is None:
            raise KeyError(f"Missing good/bad keys; got {list(rec.keys())}")
        good = self.tf(Image.open(good_path).convert("RGB"))
        bad  = self.tf(Image.open(bad_path).convert("RGB"))
        return good, bad


def train_rank_scorer_from_json(
    json_path: str,
    out_path: str,
    device="cuda:0",
    backbone_name="resnet50",
    img_size=224,
    batch_size=32,
    num_workers=4,
    epochs_frozen=2,
    epochs_unfrozen=6,
    lr_head=1e-4,
    lr_backbone=1e-5,
    weight_decay=1e-4,
    grad_clip=1.0,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    pairs = json.load(open(json_path, "r"))

    jp = Path(json_path).resolve()
    # json_path = Adacrop/data/splits/xxx.json -> parents[2] = Adacrop/
    img_root = str(jp.parents[2]) if len(jp.parents) >= 3 else None

    ds = PairwiseRankDataset(pairs, img_size=img_size, img_root=img_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    scorer = PairwiseRankScorer(device=device, backbone_name=backbone_name, freeze_backbone=True, img_size=img_size)
    scorer.train()

    def run_phase(n_epochs: int, phase: str, opt: torch.optim.Optimizer):
        for ep in range(n_epochs):
            total = 0.0
            n = 0
            for good, bad in dl:
                good = good.to(scorer.device, non_blocking=True)
                bad  = bad.to(scorer.device, non_blocking=True)

                s_good = scorer(good)
                s_bad  = scorer(bad)
                loss = logistic_rank_loss(s_good, s_bad)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), grad_clip)
                opt.step()

                total += float(loss.item())
                n += 1
            print(f"[RankTrain:{phase}] epoch {ep+1}/{n_epochs} loss={total/max(1,n):.4f}")

    # phase 1: frozen backbone
    scorer.freeze_backbone(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, scorer.parameters()), lr=lr_head, weight_decay=weight_decay)
    run_phase(epochs_frozen, "frozen", opt)

    # phase 2: unfrozen backbone
    if epochs_unfrozen > 0:
        scorer.freeze_backbone(False)
        opt = torch.optim.AdamW(
            [
                {"params": scorer.head.parameters(), "lr": lr_head},
                {"params": scorer.backbone.parameters(), "lr": lr_backbone},
            ],
            weight_decay=weight_decay,
        )
        run_phase(epochs_unfrozen, "unfrozen", opt)

    scorer.eval()
    torch.save({"state_dict": scorer.state_dict()}, out_path)
    print(f"* saved rank scorer -> {out_path}")
    return scorer

def _build_argparser():
    import argparse

    p = argparse.ArgumentParser("Train rank-based (pairwise) scorer")
    p.add_argument("--pairs", type=str, default="data/splits/rank_pairs_train.json", help="pairwise json path")
    p.add_argument("--out", type=str, default="checkpoints/rank_scorer.pth", help="output checkpoint path")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "efn_b0"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs-frozen", type=int, default=2)
    p.add_argument("--epochs-unfrozen", type=int, default=2)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    from pathlib import Path

    args = _build_argparser().parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    train_rank_scorer_from_json(
        json_path=args.pairs,
        out_path=args.out,
        device=args.device,
        backbone_name=args.backbone,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
