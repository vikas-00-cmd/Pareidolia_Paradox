import os, cv2, math
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import CFG, LunarDataset, LunarClassifier, normalize_azimuth
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_val_transforms(cfg):
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def predict_test(cfg, threshold=0.520):
    test_meta = pd.read_csv(cfg.input_dir / "Test/test_metadata.csv")
    eval_ds = LunarDataset(test_meta, cfg.work_dir / "norm_eval", cfg, transform=get_val_transforms(cfg))
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)
    
    device = cfg.device
    models = []
    for fold in range(cfg.num_folds):
        ckpt = cfg.ckpt_dir / f"fold{fold}_best.pt"
        if not ckpt.exists():
            print(f"Missing {ckpt}")
            continue
        model = LunarClassifier(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models.append(model)
        
    all_preds = []
    with torch.no_grad():
        for images, az in tqdm(eval_loader, desc="Inference"):
            images, az = images.to(device), az.to(device)
            fold_preds = []
            for model in models:
                logits = model(images, az)
                fold_preds.append(torch.sigmoid(logits).cpu().numpy())
            avg_preds = np.mean(fold_preds, axis=0)
            all_preds.extend(avg_preds)
            
    test_meta["label"] = (np.array(all_preds) > threshold).astype(int)
    sub_path = cfg.work_dir / f"submission_{cfg.backbone}.csv"
    test_meta[["image_id", "label"]].to_csv(sub_path, index=False)
    print(f"Saved: {sub_path}")

if __name__ == '__main__':
    print('Running inference...')
    predict_test(CFG)
