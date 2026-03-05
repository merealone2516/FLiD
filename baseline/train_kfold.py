#!/usr/bin/env python3
"""
Baseline — 5-Fold Stratified Cross-Validation

Trains MobileNetV2 from scratch (Gonzalez & Tapia) with 5-fold CV and
reports ISO/IEC 30107-3 metrics (EER, BPCER10/20/50) plus bootstrap CIs.

For Both_attack the baseline uses a two-stage cascade:
    score = min( Face_model(x), Text_model(x) )

Usage:
    python -m baseline.train_kfold
    python -m baseline.train_kfold --attack Face_attack
"""

import argparse
import json
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from PIL import Image

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import TRAIN_TEST_DATA, KFOLD_OUTPUT, get_device
from baseline.model import (MobileNetV2PAD, GaussianNoise, IMG_SIZE,
                            get_train_transforms, get_val_transforms)
from flid.metrics import compute_metrics

SEED = 42
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 0


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ═══════════════════════════════════════════════════════════════
# Image path loader
# ═══════════════════════════════════════════════════════════════

def load_image_paths(attack_type, data_root):
    """Load all image paths and labels from train+test splits combined."""
    paths, labels = [], []
    root = Path(data_root) / attack_type

    for label, cat in enumerate(['Real', 'Fake']):
        for split in ['train', 'test']:
            split_dir = root / cat / split
            if not split_dir.exists():
                continue
            for f in sorted(split_dir.iterdir()):
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}:
                    paths.append(str(f))
                    labels.append(label)

    return paths, np.array(labels)


# ═══════════════════════════════════════════════════════════════
# Training one fold
# ═══════════════════════════════════════════════════════════════

def train_one_fold(train_paths, train_labels, val_paths, val_labels, device):
    """Train MobileNetV2 from scratch for one fold, return val bona-fide scores."""
    train_ds = ImageDataset(train_paths, train_labels, get_train_transforms())
    val_ds   = ImageDataset(val_paths, val_labels, get_val_transforms())

    # Class weights + sampler
    lc = Counter(train_labels.tolist())
    n = len(train_labels)
    nc = len(lc)
    cw = {c: n / (nc * cnt) for c, cnt in lc.items()}
    wt = torch.tensor([cw.get(0, 1), cw.get(1, 1)], dtype=torch.float32).to(device)
    sw = [cw[l] for l in train_labels.tolist()]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model = MobileNetV2PAD(2, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=wt)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    best_loss, best_state, wait = float('inf'), None, 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labs)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        vloss = 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                vloss += criterion(model(imgs), labs).item() * imgs.size(0)
        vloss /= len(val_ds)
        scheduler.step()

        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    all_labels, all_scores = [], []
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)
            all_labels.extend(labs.tolist())
            all_scores.extend(probs[:, 0].cpu().tolist())

    return np.array(all_labels), np.array(all_scores)


# ═══════════════════════════════════════════════════════════════
# K-Fold runner for single attack
# ═══════════════════════════════════════════════════════════════

def run_kfold(attack_type, data_root, device, n_folds=5):
    paths, labels = load_image_paths(attack_type, data_root)
    print(f"  Total images: {len(paths)} ({(labels==0).sum()}R, {(labels==1).sum()}F)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), 1):
        trp = [paths[i] for i in train_idx]
        trl = labels[train_idx]
        vlp = [paths[i] for i in val_idx]
        vll = labels[val_idx]

        print(f"  Fold {fold}: train={len(trp)}, val={len(vlp)}")
        y_val, scores_val = train_one_fold(trp, trl, vlp, vll, device)
        m = compute_metrics(y_val, scores_val)
        fold_metrics.append(m)
        print(f"    AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  "
              f"BPCER10={m['bpcer10']:.1f}%  BPCER20={m['bpcer20']:.1f}%")

    return fold_metrics


# ═══════════════════════════════════════════════════════════════
# Cascade K-Fold for Both_attack
# ═══════════════════════════════════════════════════════════════

def run_cascade_kfold(data_root, device, n_folds=5):
    """
    For Each fold: train Face and Text models separately,
    evaluate on Both_attack split using cascade min-score.
    """
    # Load all paths for all three attack types
    face_paths, face_labels = load_image_paths('Face_attack', data_root)
    text_paths, text_labels = load_image_paths('Text_attack', data_root)
    both_paths, both_labels = load_image_paths('Both_attack', data_root)

    print(f"  Cascade: Face={len(face_paths)}, Text={len(text_paths)}, Both={len(both_paths)}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(both_paths, both_labels), 1):
        both_val_paths = [both_paths[i] for i in val_idx]
        both_val_labels = both_labels[val_idx]

        # Train face model on all face data except this fold's both-val overlap
        face_ds = ImageDataset(face_paths, face_labels, get_train_transforms())
        face_val_ds = ImageDataset(face_paths[:10], face_labels[:10], get_val_transforms())

        # Simple approach: train on all face data for this fold
        face_y, face_s = train_one_fold(
            face_paths, face_labels,
            both_val_paths, both_val_labels, device
        )

        text_y, text_s = train_one_fold(
            text_paths, text_labels,
            both_val_paths, both_val_labels, device
        )

        # Cascade: min(face_score, text_score)
        cascade_scores = np.minimum(face_s, text_s)
        m = compute_metrics(both_val_labels, cascade_scores)
        fold_metrics.append(m)
        print(f"  Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%")

    return fold_metrics


# ═══════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.std(arr)), \
           float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Baseline 5-fold CV')
    parser.add_argument('--attack', default='all',
                        choices=['Face_attack', 'Text_attack', 'Both_attack', 'all'])
    parser.add_argument('--data_root', default=str(TRAIN_TEST_DATA))
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = get_device(args.device)
    print(f"Device: {device}")

    KFOLD_OUTPUT.mkdir(parents=True, exist_ok=True)
    attacks = (['Face_attack', 'Text_attack', 'Both_attack']
               if args.attack == 'all' else [args.attack])

    all_results = {}

    for attack in attacks:
        print(f"\n{'='*60}")
        print(f"  Baseline {attack} — {args.n_folds}-fold CV")
        print(f"{'='*60}")

        if attack == 'Both_attack':
            fm = run_cascade_kfold(args.data_root, device, args.n_folds)
        else:
            fm = run_kfold(attack, args.data_root, device, args.n_folds)

        summary = {}
        for key in ['auc', 'eer', 'accuracy', 'f1', 'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
            vals = [m[key] for m in fm]
            mean, std, lo, hi = bootstrap_ci(vals)
            summary[key] = {'mean': round(mean, 4), 'std': round(std, 4),
                            'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}

        all_results[attack] = {'folds': fm, 'summary': summary}

        print(f"\n  Summary:")
        for k, v in summary.items():
            print(f"    {k:<10s}: {v['mean']:.4f} ± {v['std']:.4f}")

    out_path = KFOLD_OUTPUT / 'baseline_kfold_results.json'

    def ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: ser(v) for k, v in o.items()}
        if isinstance(o, list):
            return [ser(v) for v in o]
        return o

    with open(out_path, 'w') as f:
        json.dump(ser(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
