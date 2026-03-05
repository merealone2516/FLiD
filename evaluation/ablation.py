#!/usr/bin/env python3
"""
Evaluation — Ablation Study

Five ablations:
    (i)   Whole-image vs field-localised ROI crops
    (ii)  YOLO-detected crops vs coordinate-based crops (fair comparison)
    (iii) Frozen backbone + MLP vs fine-tuned end-to-end
    (iv)  Fusion (Both = face + text) vs single-field
    (v)   Backbone choice: MobileNetV3-Small vs EfficientNet-B0 vs ResNet-18

Usage:
    python -m evaluation.ablation
"""

import json
import os
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

from configs.paths import (PAIR_DATA, PAIR_DATA_YOLO, ABLATION_OUTPUT,
                            BASELINE_RESULTS, get_device)
from flid.models import (MobileNetV3Extractor, EfficientNetExtractor,
                          ResNet18Extractor, make_mlp, TRANSFORM)
from flid.metrics import compute_metrics
from flid.data import (load_face_embeddings, load_text_embeddings,
                        load_both_embeddings, load_full_image_embeddings,
                        load_yolo_face_images, load_yolo_text_images,
                        extract_embeddings_from_images)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ═══════════════════════════════════════════════════════════════
# MLP trainer (single split, multiple random seeds)
# ═══════════════════════════════════════════════════════════════

def train_and_eval_mlp(X, y, in_dim, hidden_dims=[256, 128, 64, 32],
                       test_size=0.2, doc_names=None, doc_level=False,
                       epochs=100, patience=15, lr=1e-3, device=None):
    """Train MLP with a single 80/20 split and return metrics."""
    if device is None:
        device = get_device()

    if doc_level and doc_names is not None:
        unique_docs = list(dict.fromkeys(doc_names))
        doc_labels = {d: l for d, l in zip(doc_names, y)}
        doc_y = np.array([doc_labels[d] for d in unique_docs])
        train_docs, test_docs = train_test_split(
            unique_docs, test_size=test_size, random_state=SEED, stratify=doc_y)
        test_set = set(test_docs)
        train_idx = [i for i, n in enumerate(doc_names) if n not in test_set]
        test_idx = [i for i, n in enumerate(doc_names) if n in test_set]
    else:
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=SEED, stratify=y)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    model = make_mlp(in_dim, hidden_dims).to(device)
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
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
        p_fake = torch.sigmoid(model(X_v).squeeze(-1))
        bf_scores = (1.0 - p_fake).cpu().numpy()

    return compute_metrics(y_te, bf_scores)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ABLATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    all_results = {}

    print("=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)

    # ── Existing FLiD numbers ──
    print("\n  Loading FLiD embeddings for baseline comparison...")
    X_face, y_face, face_docs = load_face_embeddings()
    X_text, y_text = load_text_embeddings()
    X_both, y_both = load_both_embeddings()

    for name, X, y, docs, dl in [
        ('Face', X_face, y_face, face_docs, True),
        ('Text', X_text, y_text, None, False),
        ('Both', X_both, y_both, None, False),
    ]:
        m = train_and_eval_mlp(X, y, X.shape[1], doc_names=docs,
                               doc_level=dl, device=device)
        all_results[f'flid_{name}'] = m
        print(f"    FLiD {name}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%")

    # ── (i) Whole-image vs ROI ──
    print("\n  ABLATION (i): Whole-image vs ROI (Face)")
    try:
        X_full, y_full = load_full_image_embeddings()
        m = train_and_eval_mlp(X_full, y_full, 576, device=device)
        all_results['whole_image_face'] = m
        print(f"    Whole-image: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%")
    except FileNotFoundError:
        print("    Skipped — full-image embeddings not found")

    # ── (ii) Load fair YOLO comparison results if available ──
    print("\n  ABLATION (ii): YOLO vs Coord crops")
    fair_path = ABLATION_OUTPUT / 'fair_yolo_comparison.json'
    if fair_path.exists():
        with open(fair_path) as f:
            fair = json.load(f)
        for k in ['coord_face', 'yolo_face', 'coord_text', 'yolo_text']:
            all_results[k] = fair[k]
            print(f"    {k}: AUC={fair[k]['auc']:.4f}")
    else:
        print("    Run `evaluation.fair_comparison` first for full results.")

    # ── (iii) Frozen vs Fine-tuned ──
    print("\n  ABLATION (iii): Frozen vs Fine-tuned")
    for atk in ['Face_attack', 'Text_attack', 'Both_attack']:
        rp = Path(BASELINE_RESULTS).parent / 'results_pretrained' / atk / 'results.json'
        if rp.exists():
            with open(rp) as f:
                d = json.load(f)
            iso = d.get('iso_metrics', {})
            short = atk.replace('_attack', '')
            all_results[f'finetuned_{short}'] = {
                'auc': iso.get('auc', 0.5), 'eer': iso.get('eer', 50)}
            print(f"    Fine-tuned {short}: AUC={iso.get('auc', 0):.4f}  EER={iso.get('eer', 0):.2f}%")

    # ── (iv) Fusion vs single-field — already covered by flid_Face/Text/Both ──
    print("\n  ABLATION (iv): Single-field vs Fusion — see FLiD results above")

    # ── (v) Backbone choice ──
    print("\n  ABLATION (v): Backbone choice (Face crops)")
    try:
        face_paths, face_y, _ = load_yolo_face_images() if len(load_yolo_face_images()[0]) > 20 else ([], [], [])
    except:
        face_paths, face_y = [], np.array([])

    if len(face_paths) > 20:
        for get_ext, dim, bname in [
            (lambda: MobileNetV3Extractor().eval().to(device), 576, 'MobileNetV3-Small'),
            (lambda: EfficientNetExtractor().eval().to(device), 1280, 'EfficientNet-B0'),
            (lambda: ResNet18Extractor().eval().to(device), 512, 'ResNet-18'),
        ]:
            ext = get_ext()
            embs = extract_embeddings_from_images(face_paths, ext, device)
            m = train_and_eval_mlp(embs, face_y, dim, device=device)
            all_results[f'backbone_{bname}'] = m
            print(f"    {bname}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%")
            del ext
    else:
        print("    Skipped — not enough YOLO face crops found")

    # ── Save ──
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

    with open(ABLATION_OUTPUT / 'ablation_results.json', 'w') as f:
        json.dump(ser(all_results), f, indent=2)
    print(f"\n  Results saved to {ABLATION_OUTPUT / 'ablation_results.json'}")


if __name__ == '__main__':
    main()
