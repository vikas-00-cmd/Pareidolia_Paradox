import os, cv2, time, gc, math
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import timm
from tqdm import tqdm
from multiprocessing import Pool

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class CFG:
    seed = 42
    input_dir = Path(".")
    train_meta_path = "Train/train_metadata.csv"
    work_dir = Path("working")
    norm_train_dir = work_dir / "norm_train"
    ckpt_dir = work_dir / "checkpoints"
    
    img_size = 224
    crop_size = 180
    pad_pixels = 60
    
    backbone = "convnext_tiny"
    epochs = 25
    batch_size = 32
    num_workers = 0
    lr = 3e-4
    weight_decay = 1e-4
    num_folds = 5
    early_stop_patience = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_azimuth(row):
    # Physics Normalization Engine
    src_path, dst_dir, angle, pad, crop, out_sz = row
    dst_path = Path(dst_dir) / Path(src_path).name
    if dst_path.exists(): return True
    
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = img_pad.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(img_pad, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    c_x, c_y = center
    half = crop // 2
    cropped = rotated[c_y - half : c_y + half, c_x - half : c_x + half]
    final = cv2.resize(cropped, (out_sz, out_sz), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(str(dst_path), final)
    return True

class LunarDataset(Dataset):
    def __init__(self, meta_df, image_dir, cfg, transform=None):
        self.meta = meta_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.azimuths = (self.meta["sun_azimuth_angle"].values / 360.0).astype(np.float32)
        if "label" in self.meta.columns:
            self.labels = self.meta["label"].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.meta.loc[idx, "image_id"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            img = self.transform(image=img)["image"]
            
        az = torch.tensor([
            math.sin(self.azimuths[idx] * 2 * math.pi),
            math.cos(self.azimuths[idx] * 2 * math.pi)
        ], dtype=torch.float32)
        
        if self.labels is not None:
            return img, az, torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, az

class LunarClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(cfg.backbone, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.az_mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features + 32, 1))

    def forward(self, x, az):
        img_feats = self.backbone(x)
        az_feats = self.az_mlp(az)
        combined = torch.cat([img_feats, az_feats], dim=1)
        return self.head(combined)

# Standard training loops would follow here.
# Note: For full codebase execution, please copy the detailed PyTorch loops from the Kaggle Notebook.
if __name__ == '__main__':
    print('Run the full Kaggle pipeline functions here.')
