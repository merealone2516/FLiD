#!/usr/bin/env python3


import argparse
import json
import warnings
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import get_device, KFOLD_OUTPUT
from flid.models import FaceClassifier, TextClassifier, BothClassifier
from flid.metrics import compute_metrics
from flid.data import load_face_embeddings, load_text_embeddings, load_both_embeddings

SEED = 42


# ═══════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════

def train_mlp_fold(model, X_train, y_train, X_val, y_val, device,
                   epochs=100, patience=15, lr=1e-3):
    """Train one MLP fold with early stopping and return bona-fide scores on val."""
    model = model.to(device)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
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
# K-Fold CV runner
# ═══════════════════════════════════════════════════════════════

def run_kfold_cv(X, y, make_model_fn, in_dim, device, n_folds=5,
                 doc_names=None, doc_level=False):
    """
    Run stratified k-fold CV.  For face-attack the split is done at the
    document level to avoid data leakage from augmentation.

    Returns:
        List of per-fold metric dicts.
    """
    fold_metrics = []

    if doc_level and doc_names is not None:
        unique_docs = list(dict.fromkeys(doc_names))
        doc_label = {}
        for d, l in zip(doc_names, y):
            doc_label[d] = l
        doc_y = np.array([doc_label[d] for d in unique_docs])

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for fold, (train_doc_idx, val_doc_idx) in enumerate(skf.split(unique_docs, doc_y), 1):
            train_docs = {unique_docs[i] for i in train_doc_idx}
            val_docs = {unique_docs[i] for i in val_doc_idx}
            train_idx = [i for i, d in enumerate(doc_names) if d in train_docs]
            val_idx = [i for i, d in enumerate(doc_names) if d in val_docs]

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            model = make_model_fn()
            bf = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
            m = compute_metrics(y_vl, bf)
            fold_metrics.append(m)
            print(f"    Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  "
                  f"BPCER10={m['bpcer10']:.1f}%  BPCER20={m['bpcer20']:.1f}%")
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            model = make_model_fn()
            bf = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
            m = compute_metrics(y_vl, bf)
            fold_metrics.append(m)
            print(f"    Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  "
                  f"BPCER10={m['bpcer10']:.1f}%  BPCER20={m['bpcer20']:.1f}%")

    return fold_metrics


# ═══════════════════════════════════════════════════════════════
# Bootstrap CIs
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(fold_values, n_boot=1000, alpha=0.05):
    """95 % bootstrap confidence interval from per-fold values."""
    arr = np.array(fold_values)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(np.std(arr)), float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FLiD 5-fold CV')
    parser.add_argument('--attack', choices=['Face', 'Text', 'Both', 'all'],
                        default='all', help='Attack scenario')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device(args.device)
    print(f"Device: {device}")

    KFOLD_OUTPUT.mkdir(parents=True, exist_ok=True)
    attacks = ['Face', 'Text', 'Both'] if args.attack == 'all' else [args.attack]

    all_results = {}

    for attack in attacks:
        print(f"\n{'='*60}")
        print(f"  {attack} Attack — {args.n_folds}-fold CV")
        print(f"{'='*60}")

        if attack == 'Face':
            X, y, doc_names = load_face_embeddings()
            make_fn = FaceClassifier
            doc_level = True
            in_dim = 576
        elif attack == 'Text':
            X, y = load_text_embeddings()
            doc_names = None
            make_fn = TextClassifier
            doc_level = False
            in_dim = 576
        else:
            X, y = load_both_embeddings()
            doc_names = None
            make_fn = BothClassifier
            doc_level = False
            in_dim = 1152

        print(f"  Samples: {len(X)} ({(y==0).sum()} Real, {(y==1).sum()} Fake)")

        fold_metrics = run_kfold_cv(
            X, y, make_fn, in_dim, device,
            n_folds=args.n_folds, doc_names=doc_names, doc_level=doc_level,
        )

        # Aggregate with bootstrap CIs
        summary = {}
        for metric_key in ['auc', 'eer', 'accuracy', 'f1',
                           'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
            vals = [m[metric_key] for m in fold_metrics]
            mean, std, lo, hi = bootstrap_ci(vals, n_boot=args.n_bootstraps)
            summary[metric_key] = {
                'mean': round(mean, 4),
                'std': round(std, 4),
                'ci_lo': round(lo, 4),
                'ci_hi': round(hi, 4),
            }

        all_results[attack] = {
            'folds': fold_metrics,
            'summary': summary,
        }

        print(f"\n  Summary:")
        for k, v in summary.items():
            print(f"    {k:<10s}: {v['mean']:.4f} ± {v['std']:.4f}  "
                  f"[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    # Save
    out_path = KFOLD_OUTPUT / 'kfold_bootstrap_results.json'

    def serialize(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: serialize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [serialize(v) for v in o]
        return o

    with open(out_path, 'w') as f:
        json.dump(serialize(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
