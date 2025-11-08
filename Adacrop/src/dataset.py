import os, json
from PIL import Image
from torch.utils.data import Dataset

class CropDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.files = [f[:-5] for f in os.listdir(ann_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = Image.open(f"{self.img_dir}/{fn}.jpg").convert("RGB")
        data = json.load(open(f"{self.ann_dir}/{fn}.json"))
        box = data["box"]  # [x_ctr,y_ctr,w_rel,h_rel]
        if self.transforms:
            img = self.transforms(img)
        return img, box
