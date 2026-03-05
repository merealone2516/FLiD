#!/usr/bin/env python3
"""
Evaluation — Fair YOLO vs Coordinate-Based Crop Comparison

Ensures an apples-to-apples comparison between YOLO-detected and
coordinate-based crops by using the exact same:
    • MobileNetV3-Small backbone
    • 224×224 resize + ImageNet normalisation
    • MLP architecture and hyperparameters
    • No augmentation / no patch-based embedding (raw single-crop)
    • 5-run averaged metrics for stability

Usage:
    python -m evaluation.fair_comparison
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import PAIR_DATA, PAIR_DATA_YOLO, ABLATION_OUTPUT, get_device
from flid.models import MobileNetV3Extractor, make_mlp, TRANSFORM
from flid.metrics import compute_metrics
from flid.data import (load_coord_face_images, load_coord_text_images,
                        load_yolo_face_images, load_yolo_text_images,
                        extract_embeddings_from_images)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def train_and_eval_mlp(X, y, in_dim, hidden_dims=[256, 128, 64, 32],
                       test_size=0.2, doc_names=None, doc_level=False,
                       epochs=100, patience=15, lr=1e-3, n_runs=5, device=None):
    """Train MLP with n_runs random seeds and return averaged metrics."""
    if device is None:
        device = get_device()

    all_metrics = []
    for run in range(n_runs):
        seed = SEED + run
        np.random.seed(seed)
        torch.manual_seed(seed)

        if doc_level and doc_names is not None:
            unique = list(dict.fromkeys(doc_names))
            dy = np.array([{d: l for d, l in zip(doc_names, y)}[d] for d in unique])
            tr_docs, te_docs = train_test_split(unique, test_size=test_size,
                                                random_state=seed, stratify=dy)
            te_set = set(te_docs)
            train_idx = [i for i, d in enumerate(doc_names) if d not in te_set]
            test_idx = [i for i, d in enumerate(doc_names) if d in te_set]
        else:
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=test_size,
                                                   random_state=seed, stratify=y)

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model = make_mlp(in_dim, hidden_dims).to(device)
        n_pos, n_neg = (y_tr == 1).sum(), (y_tr == 0).sum()
        pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        X_v = torch.tensor(X_te, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_te, dtype=torch.float32).to(device)

        best_loss, best_state, wait = float('inf'), None, 0
        for _ in range(epochs):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb).squeeze(-1), yb).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                vl = criterion(model(X_v).squeeze(-1), y_v).item()
            scheduler.step(vl)
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            bf = (1 - torch.sigmoid(model(X_v).squeeze(-1))).cpu().numpy()
        all_metrics.append(compute_metrics(y_te, bf))

    avg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        avg[key] = float(np.mean(vals))
        avg[f'{key}_std'] = float(np.std(vals))
    return avg


def main():
    ABLATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    EMB_DIM = 576

    print("=" * 70)
    print("  FAIR YOLO vs COORDINATE CROP COMPARISON")
    print("=" * 70)

    extractor = MobileNetV3Extractor().eval().to(device)

    # Load images
    print("\n  Loading image sets...")
    cf_paths, cf_y, cf_docs = load_coord_face_images()
    yf_paths, yf_y, yf_docs = load_yolo_face_images()
    ct_paths, ct_y, ct_docs = load_coord_text_images()
    yt_paths, yt_y, yt_docs = load_yolo_text_images()

    for label, paths, y in [("Coord Face", cf_paths, cf_y), ("YOLO Face", yf_paths, yf_y),
                             ("Coord Text", ct_paths, ct_y), ("YOLO Text", yt_paths, yt_y)]:
        print(f"    {label}: {len(paths)} ({(y==0).sum()}R, {(y==1).sum()}F)")

    # Extract embeddings
    print("\n  Extracting embeddings...")
    cf_X = extract_embeddings_from_images(cf_paths, extractor, device)
    yf_X = extract_embeddings_from_images(yf_paths, extractor, device)
    ct_X = extract_embeddings_from_images(ct_paths, extractor, device)
    yt_X = extract_embeddings_from_images(yt_paths, extractor, device)

    # Train MLPs
    results = {}
    print("\n  Training MLPs (5-run average)...")

    for label, X, y, docs, dl in [
        ('coord_face', cf_X, cf_y, cf_docs, True),
        ('yolo_face', yf_X, yf_y, yf_docs, True),
        ('coord_text', ct_X, ct_y, None, False),
        ('yolo_text', yt_X, yt_y, None, False),
    ]:
        m = train_and_eval_mlp(X, y, EMB_DIM, doc_names=docs, doc_level=dl, device=device)
        results[label] = m
        print(f"    {label:15s}: AUC={m['auc']:.4f}±{m['auc_std']:.4f}  "
              f"EER={m['eer']:.2f}±{m['eer_std']:.2f}%")

    # Save
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

    out_path = ABLATION_OUTPUT / 'fair_yolo_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(ser(results), f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
