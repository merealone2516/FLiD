#!/usr/bin/env python3
"""
Face Augmentation Ablation — 5-Fold Stratified CV

Compares:
  (a) No augmentation: 153 samples (100R + 53F)
  (b) With augmentation: 612 samples (400R + 212F), 4 versions per image

Also generates a visualization of one ID face crop with all 4 augmented versions.

Usage:
    python -m evaluation.augmentation_ablation
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

BASE = Path('/Users/akumar/Downloads/Turing')
PAIR = BASE / 'pair_data'
SEED = 42
K_FOLDS = 5


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(labels, bf_scores, n_thresh=5000):
    bf = bf_scores[labels == 0]
    pa = bf_scores[labels == 1]
    if len(bf) == 0 or len(pa) == 0:
        return {'auc': 0.5, 'eer': 50.0}
    tau = np.linspace(0, 1, n_thresh)
    apcer = np.array([np.mean(pa > t) for t in tau])
    bpcer = np.array([np.mean(bf <= t) for t in tau])
    diff = np.abs(bpcer - apcer)
    eidx = np.argmin(diff)
    eer = (bpcer[eidx] + apcer[eidx]) / 2 * 100
    tpr = 1 - bpcer
    fpr = apcer
    si = np.argsort(fpr)
    auc = abs(np.trapz(tpr[si], fpr[si]))
    auc = max(0.0, min(1.0, auc))
    return {'auc': auc, 'eer': eer}


# ═══════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════

def load_face_noaug():
    """Non-augmented face embeddings: 153 samples."""
    emb_dir = PAIR / 'Face_attack_crop' / 'Mobilenetv3_small' / 'Face_attack_crop_embeddings'
    X, y, names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        d = emb_dir / cat
        for f in sorted(d.glob('*.npy')):
            X.append(np.load(f).astype(np.float32).flatten())
            y.append(label)
            names.append(f.stem)
    return np.array(X, dtype=np.float32), np.array(y), names


def load_face_aug():
    """Augmented face embeddings: 612 samples (4× per image)."""
    emb_dir = PAIR / 'Face_attack_crop' / 'data_aug' / 'face_attack_crop_embeddings_aug'
    X, y, names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        d = emb_dir / cat
        for f in sorted(d.glob('*.npy')):
            X.append(np.load(f).astype(np.float32).flatten())
            y.append(label)
            # Strip _augN suffix for document name
            names.append(re.sub(r'_aug\d+$', '', f.stem))
    return np.array(X, dtype=np.float32), np.array(y), names


# ═══════════════════════════════════════════════════════════════
# MLP + Training
# ═══════════════════════════════════════════════════════════════

def make_mlp(in_dim=576, hidden_dims=[256, 128, 64, 32]):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device,
                   epochs=100, patience=15, lr=1e-3):
    model = model.to(device)
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    X_v = torch.tensor(X_vl, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_vl, dtype=torch.float32).to(device)

    best_loss = float('inf')
    best_state = None
    wait = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
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
        logits = model(X_v).squeeze(-1)
        p_fake = torch.sigmoid(logits)
        bf_scores = (1.0 - p_fake).cpu().numpy()
    return bf_scores


# ═══════════════════════════════════════════════════════════════
# K-Fold with document-level splitting
# ═══════════════════════════════════════════════════════════════

def run_kfold_doc_level(X, y, doc_names, device):
    """5-fold CV with document-level splitting (augmented versions stay together)."""
    unique_docs = list(dict.fromkeys(doc_names))
    doc_to_label = {}
    for name, label in zip(doc_names, y):
        doc_to_label[name] = label
    doc_labels = np.array([doc_to_label[d] for d in unique_docs])

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (doc_train_idx, doc_val_idx) in enumerate(skf.split(unique_docs, doc_labels), 1):
        np.random.seed(SEED + fold)
        torch.manual_seed(SEED + fold)

        val_docs = set(unique_docs[i] for i in doc_val_idx)
        train_idx = [i for i, n in enumerate(doc_names) if n not in val_docs]
        val_idx = [i for i, n in enumerate(doc_names) if n in val_docs]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]

        model = make_mlp()
        bf = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
        m = compute_metrics(y_vl, bf)
        fold_metrics.append(m)
        print(f"      Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  "
              f"(train={len(train_idx)}, val={len(val_idx)})")

    return fold_metrics


def run_kfold_sample_level(X, y, device):
    """5-fold CV with sample-level splitting (for non-augmented)."""
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        np.random.seed(SEED + fold)
        torch.manual_seed(SEED + fold)

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]

        model = make_mlp()
        bf = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
        m = compute_metrics(y_vl, bf)
        fold_metrics.append(m)
        print(f"      Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  "
              f"(train={len(train_idx)}, val={len(val_idx)})")

    return fold_metrics


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def generate_augmentation_figure(out_dir):
    """
    Show one face crop with its 4 augmented versions side by side.
    Uses deterministic seeds for reproducible augmentations.
    """
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image

    # Pick a representative face crop
    face_crops = sorted(
        (PAIR / 'Face_attack_crop' / 'Real').rglob('face_crop.png'))
    img_path = face_crops[5]  # Pick a good example
    img = Image.open(img_path).convert('RGB')

    # Define the 4 augmentation transforms (image-level, before normalization)
    aug_transforms = {
        'Original': transforms.Compose([
            transforms.Resize((224, 224)),
        ]),
        'Aug 1: Rotation\n+ Brightness': transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2),
            transforms.Resize((224, 224)),
        ]),
        'Aug 2: H-Flip\n+ Contrast': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(contrast=0.2),
            transforms.Resize((224, 224)),
        ]),
        'Aug 3: Rotation\n+ Saturation': transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(saturation=0.2),
            transforms.Resize((224, 224)),
        ]),
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, (title, tfm) in zip(axes, aug_transforms.items()):
        torch.manual_seed(42)  # Deterministic
        augmented = tfm(img)
        ax.imshow(augmented)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    fig.suptitle('Face Crop Data Augmentation (4× per document)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        out = out_dir / f'face_augmentation_examples.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = Path(__file__).resolve().parent.parent / 'results' / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── (a) No augmentation ──
    print(f"\n{'='*60}")
    print("  Face Attack — No Augmentation")
    print(f"{'='*60}")
    X_na, y_na, names_na = load_face_noaug()
    print(f"    Samples: {len(X_na)} ({(y_na==0).sum()}R + {(y_na==1).sum()}F)  dim={X_na.shape[1]}")

    fold_na = run_kfold_sample_level(X_na, y_na, device)
    summary_na = {}
    for key in ['auc', 'eer']:
        vals = [m[key] for m in fold_na]
        summary_na[key] = {
            'mean': round(float(np.mean(vals)), 4),
            'std': round(float(np.std(vals)), 4),
            'per_fold': [round(float(v), 4) for v in vals],
        }
    results['no_augmentation'] = {
        'samples': len(X_na),
        'metrics': summary_na,
    }
    print(f"    >> AUC={summary_na['auc']['mean']:.4f}±{summary_na['auc']['std']:.4f}  "
          f"EER={summary_na['eer']['mean']:.2f}±{summary_na['eer']['std']:.2f}%")

    # ── (b) With augmentation (document-level split) ──
    print(f"\n{'='*60}")
    print("  Face Attack — With Augmentation (4× per image)")
    print(f"{'='*60}")
    X_aug, y_aug, names_aug = load_face_aug()
    n_docs = len(set(names_aug))
    print(f"    Samples: {len(X_aug)} ({(y_aug==0).sum()}R + {(y_aug==1).sum()}F)  "
          f"dim={X_aug.shape[1]}  docs={n_docs}")

    fold_aug = run_kfold_doc_level(X_aug, y_aug, names_aug, device)
    summary_aug = {}
    for key in ['auc', 'eer']:
        vals = [m[key] for m in fold_aug]
        summary_aug[key] = {
            'mean': round(float(np.mean(vals)), 4),
            'std': round(float(np.std(vals)), 4),
            'per_fold': [round(float(v), 4) for v in vals],
        }
    results['with_augmentation'] = {
        'samples': len(X_aug),
        'docs': n_docs,
        'metrics': summary_aug,
    }
    print(f"    >> AUC={summary_aug['auc']['mean']:.4f}±{summary_aug['auc']['std']:.4f}  "
          f"EER={summary_aug['eer']['mean']:.2f}±{summary_aug['eer']['std']:.2f}%")

    # Delta
    delta_auc = summary_aug['auc']['mean'] - summary_na['auc']['mean']
    delta_eer = summary_aug['eer']['mean'] - summary_na['eer']['mean']
    print(f"\n    Δ AUC = {delta_auc:+.4f}   Δ EER = {delta_eer:+.2f}%")

    # Save results
    out_path = out_dir / 'augmentation_ablation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate visualization
    print("\nGenerating augmentation visualization...")
    generate_augmentation_figure(out_dir)


if __name__ == '__main__':
    main()
