#!/usr/bin/env python3
"""
Whole-Image vs ROI Ablation — 5-Fold Stratified Cross-Validation

Trains the same MLP on MobileNetV3-Small embeddings extracted from
**full document images** (no ROI cropping) for all three attack scenarios.

Whole-image embeddings:
  Face: pair_data/Face_attack/Mobilenetv3_small/face_attack_embeddings/
  Text: pair_data/Text_attack/Mobilenetv3_small/Text_attack_embeddings/
  Both: pair_data/Both_attack/Mobilenetv3_small/Both_attack_embeddings/

These are compared against FLiD's ROI-crop results to quantify the AUC
gain from field-level extraction.

Usage:
    python -m evaluation.roi_vs_wholeimage_ablation
"""

import json
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
# Metrics  (same as backbone ablation for consistency)
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
# Data loaders — whole-image embeddings (576-D)
# ═══════════════════════════════════════════════════════════════

def load_npy_dir(emb_dir):
    """Load Real/Fake .npy files from a directory."""
    X, y = [], []
    for label, cat in enumerate(['Real', 'Fake']):
        d = emb_dir / cat
        if not d.exists():
            print(f"  WARNING: {d} not found")
            continue
        for f in sorted(d.glob('*.npy')):
            arr = np.load(f).astype(np.float32).flatten()
            X.append(arr)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def load_wholeimage_face():
    d = PAIR / 'Face_attack' / 'Mobilenetv3_small' / 'face_attack_embeddings'
    return load_npy_dir(d)


def load_wholeimage_text():
    d = PAIR / 'Text_attack' / 'Mobilenetv3_small' / 'Text_attack_embeddings'
    return load_npy_dir(d)


def load_wholeimage_both():
    d = PAIR / 'Both_attack' / 'Mobilenetv3_small' / 'Both_attack_embeddings'
    return load_npy_dir(d)


# ═══════════════════════════════════════════════════════════════
# MLP + Training
# ═══════════════════════════════════════════════════════════════

def make_mlp(in_dim, hidden_dims):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device,
                   epochs=100, patience=15, lr=1e-3):
    """Train one fold, return bf_scores on validation set."""
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


def run_kfold(X, y, in_dim, hidden_dims, device):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        np.random.seed(SEED + fold)
        torch.manual_seed(SEED + fold)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]
        model = make_mlp(in_dim, hidden_dims)
        bf = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
        m = compute_metrics(y_vl, bf)
        fold_metrics.append(m)
        print(f"      Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%")
    return fold_metrics


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else torch.device('cpu'))
    print(f"Device: {device}")

    out_dir = Path(__file__).resolve().parent.parent / 'results' / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load FLiD ROI results for comparison
    roi_path = out_dir / 'backbone_kfold_results.json'
    if roi_path.exists():
        with open(roi_path) as f:
            roi_data = json.load(f)['MobileNetV3-Small']
    else:
        roi_data = None

    attacks = {
        'Face': {
            'loader': load_wholeimage_face,
            'hidden': [256, 128, 64, 32],
        },
        'Text': {
            'loader': load_wholeimage_text,
            'hidden': [256, 128, 64, 32],
        },
        'Both': {
            'loader': load_wholeimage_both,
            'hidden': [256, 128, 64, 32],  # Single 576-D, not concatenated
        },
    }

    results = {}

    for attack, cfg in attacks.items():
        print(f"\n{'='*60}")
        print(f"  Whole-Image: {attack} Attack")
        print(f"{'='*60}")

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        X, y = cfg['loader']()
        in_dim = X.shape[1]
        print(f"    Samples: {len(X)} ({(y==0).sum()}R + {(y==1).sum()}F)  dim={in_dim}")

        fold_metrics = run_kfold(X, y, in_dim, cfg['hidden'], device)

        summary = {}
        for key in ['auc', 'eer']:
            vals = [m[key] for m in fold_metrics]
            summary[key] = {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'per_fold': [round(float(v), 4) for v in vals],
            }

        results[f'{attack}_attack'] = summary

        auc = summary['auc']
        eer = summary['eer']
        print(f"    >> Whole-image AUC={auc['mean']:.4f}±{auc['std']:.4f}  "
              f"EER={eer['mean']:.2f}±{eer['std']:.2f}%")

        if roi_data:
            roi_auc = roi_data[f'{attack}_attack']['auc']['mean']
            delta = auc['mean'] - roi_auc
            print(f"    >> FLiD ROI AUC={roi_auc:.4f}  Δ={delta:+.3f}")

    # Save
    out_path = out_dir / 'wholeimage_kfold_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("  ROI vs WHOLE-IMAGE ABLATION (5-fold CV, MobileNetV3-Small)")
    print(f"{'='*80}")
    print(f"{'Attack':<12s} {'ROI AUC':>12s} {'Whole-Img AUC':>16s} {'Δ AUC':>10s}")
    print("-" * 55)
    for attack in ['Face', 'Text', 'Both']:
        wi = results[f'{attack}_attack']['auc']
        if roi_data:
            roi_auc = roi_data[f'{attack}_attack']['auc']['mean']
            delta = wi['mean'] - roi_auc
            print(f"{attack:<12s} {roi_auc:>12.3f} "
                  f"{wi['mean']:.3f}±{wi['std']:.3f}  {delta:>+10.3f}")
        else:
            print(f"{attack:<12s} {'N/A':>12s} "
                  f"{wi['mean']:.3f}±{wi['std']:.3f}")


if __name__ == '__main__':
    main()
