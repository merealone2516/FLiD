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
from sklearn.model_selection import StratifiedGroupKFold
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
    """Load all image paths, labels, and doc_ids from train+test splits combined."""
    paths, labels, doc_ids = [], [], []
    root = Path(data_root) / attack_type

    for label, cat in enumerate(['Real', 'Fake']):
        for split in ['train', 'test']:
            split_dir = root / cat / split
            if not split_dir.exists():
                continue
            for f in sorted(split_dir.iterdir()):
                if f.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}:
                    continue
                paths.append(str(f))
                labels.append(label)
                # Get doc_id from paired JSON, fall back to filename parsing
                json_path = f.with_suffix('.json')
                if json_path.exists():
                    try:
                        import json as _json
                        meta = _json.load(open(json_path))
                        doc_id = meta.get('person_info', {}).get(
                            'face_id', f.stem.split('-', 1)[-1])
                    except Exception:
                        doc_id = f.stem.split('-', 1)[-1]
                else:
                    doc_id = f.stem.split('-', 1)[-1]
                doc_ids.append(doc_id)

    return paths, np.array(labels), doc_ids


# ===============================================================
# Bootstrap CI
# ===============================================================

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.std(arr)), \
           float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
