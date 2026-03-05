#!/usr/bin/env python3
"""
Backbone Ablation — 5-Fold Stratified Cross-Validation

Compares three frozen backbones (MobileNetV3-Small, EfficientNet-B0, ResNet-18)
across all three attack scenarios (Face, Text, Both) using identical MLP heads
and training setup.

For a fair comparison, all backbones use the **same** pre-extracted YOLO crops
(no augmentation), so differences are purely due to the backbone encoder.

Face-attack uses document-level splitting to avoid augmentation leakage.

Results are saved to results/ablation/backbone_kfold_results.json.

Usage:
    python -m evaluation.backbone_kfold_ablation
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flid.metrics import compute_metrics

SEED = 42
N_FOLDS = 5
BASE = Path('/Users/akumar/Downloads/Turing')


# ═══════════════════════════════════════════════════════════════
# Data loading helpers  (pre-extracted .npy embeddings)
# ═══════════════════════════════════════════════════════════════

def load_npy_dir(emb_dir):
    """Load all .npy files from emb_dir/{Real,Fake}/ → (X, y, names)."""
    X, y, names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = emb_dir / cat
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing directory: {cat_dir}")
        for f in sorted(cat_dir.glob('*.npy')):
            X.append(np.load(f).astype(np.float32).flatten())
            y.append(label)
            names.append(f.stem)
    return np.array(X), np.array(y), names


# ── MobileNetV3-Small ────────────────────────────────────────

def load_mnv3_face():
    """Augmented face embeddings (576-D) — same as FLiD main pipeline."""
    d = BASE / 'pair_data/Face_attack_crop/data_aug/face_attack_crop_embeddings_aug'
    X, y, names = load_npy_dir(d)
    doc_names = [re.sub(r'_aug\d+$', '', n) for n in names]
    return X, y, doc_names


def load_mnv3_text():
    """Patch-based text embeddings from JSON (576-D) — same as FLiD."""
    p = BASE / 'pair_data/Text_attack_crop/patch/text_attack_crop_embeddings_patch/embeddings.json'
    with open(p) as f:
        data = json.load(f)
    X = np.array([e['embedding'] for e in data], dtype=np.float32)
    y = np.array([int(e['label']) for e in data])
    return X, y, None


def load_mnv3_both():
    """Concatenated face+text embeddings (1152-D) — same as FLiD."""
    d = BASE / 'pair_data/Both_attack_crop/Mobilenetv3_small/Both_attack_crop_embeddings'
    X, y, _ = load_npy_dir(d)
    return X, y, None


# ── EfficientNet-B0 ──────────────────────────────────────────

def load_eff_face():
    """EfficientNet-B0 face embeddings (1280-D), non-augmented."""
    d = BASE / 'pair_data/Face_attack_crop/Efficientnetbo/Face_attack_crop_embeddings'
    return load_npy_dir(d)


def load_eff_text():
    """EfficientNet-B0 text embeddings (1280-D)."""
    d = BASE / 'pair_data/Text_attack_crop/Efficientnetbo/Text_attack_crop_embeddings'
    X, y, names = load_npy_dir(d)
    return X, y, None


def load_eff_both():
    """EfficientNet-B0 both embeddings (2560-D)."""
    d = BASE / 'pair_data/Both_attack_crop/Efficientnetbo/Both_attack_crop_embeddings'
    X, y, _ = load_npy_dir(d)
    return X, y, None


# ── ResNet-18 (will extract on the fly from YOLO crops) ──────

def extract_resnet18_embeddings_from_yolo_crops(attack_type):
    """
    Extract ResNet-18 embeddings from YOLO face/text crops on the fly.
    Returns (X, y, doc_names_or_None).
    """
    from torchvision import models, transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build frozen ResNet-18 extractor
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    extractor = nn.Sequential(*list(m.children())[:-1])
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    extractor = extractor.to(device)

    from PIL import Image

    if attack_type == 'Face':
        crop_dir = BASE / 'pair_data_yolo/face_attack_crop'
        crop_file = 'face_crop.png'
    elif attack_type == 'Text':
        crop_dir = BASE / 'pair_data_yolo/text_attack'
        crop_file = 'text_crop.png'
    else:
        raise ValueError("For Both, combine Face + Text embeddings")

    X, y, doc_names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = crop_dir / cat
        if not cat_dir.exists():
            print(f"  WARNING: {cat_dir} not found")
            continue
        for d in sorted(cat_dir.iterdir()):
            if not d.is_dir():
                continue
            img_path = d / crop_file
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = extractor(t).flatten().cpu().numpy()
                X.append(emb.astype(np.float32))
                y.append(label)
                doc_names.append(d.name)
            except Exception as e:
                print(f"  WARNING: Failed on {img_path}: {e}")

    return np.array(X), np.array(y), doc_names


def load_resnet_face():
    print("    Extracting ResNet-18 face embeddings...")
    return extract_resnet18_embeddings_from_yolo_crops('Face')


def load_resnet_text():
    print("    Extracting ResNet-18 text embeddings...")
    X, y, names = extract_resnet18_embeddings_from_yolo_crops('Text')
    return X, y, None


def load_resnet_both():
    """Extract ResNet-18 face+text embeddings from Both_attack_crop → 1024-D."""
    print("    Extracting ResNet-18 both embeddings from Both_attack_crop...")
    from torchvision import models, transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    extractor = nn.Sequential(*list(m.children())[:-1])
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    extractor = extractor.to(device)

    both_dir = BASE / 'pair_data/Both_attack_crop'
    X, y = [], []
    # Real: Real/{country}/{doc}/  Fake: Fake/digital_1/{country}/{doc}/
    # Use only digital_1 for Fake to match MNV3 (211F)
    search_roots = [
        (0, both_dir / 'Real'),       # label 0 = Real
        (1, both_dir / 'Fake' / 'digital_1'),  # label 1 = Fake (digital_1 only)
    ]
    for label, root in search_roots:
        if not root.exists():
            print(f"  WARNING: {root} not found")
            continue
        for country_dir in sorted(root.iterdir()):
            if not country_dir.is_dir():
                continue
            for doc_dir in sorted(country_dir.iterdir()):
                if not doc_dir.is_dir():
                    continue
                face_path = doc_dir / 'face_crop.png'
                text_path = doc_dir / 'text_crop.png'
                if not face_path.exists() or not text_path.exists():
                    continue
                try:
                    face_img = Image.open(face_path).convert('RGB')
                    text_img = Image.open(text_path).convert('RGB')
                    ft = transform(face_img).unsqueeze(0).to(device)
                    tt = transform(text_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        fe = extractor(ft).flatten().cpu().numpy()
                        te = extractor(tt).flatten().cpu().numpy()
                    X.append(np.concatenate([fe, te]).astype(np.float32))
                    y.append(label)
                except Exception as e:
                    print(f"  WARNING: {doc_dir.name}: {e}")
    return np.array(X, dtype=np.float32), np.array(y), None


# ═══════════════════════════════════════════════════════════════
# MLP builder + trainer
# ═══════════════════════════════════════════════════════════════

def make_mlp(in_dim, hidden_dims):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp_fold(model, X_train, y_train, X_val, y_val, device,
                   epochs=100, patience=15, lr=1e-3):
    model = model.to(device)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(y_train, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

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
        logits = model(X_v).squeeze(-1)
        p_fake = torch.sigmoid(logits)
        bf_scores = (1.0 - p_fake).cpu().numpy()
    return bf_scores


def run_kfold(X, y, in_dim, hidden_dims, device, doc_names=None, doc_level=False):
    """Run 5-fold stratified CV. Returns list of per-fold metric dicts."""
    fold_metrics = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    if doc_level and doc_names is not None:
        unique_docs = list(dict.fromkeys(doc_names))
        doc_label = {}
        for d, l in zip(doc_names, y):
            doc_label[d] = l
        doc_y = np.array([doc_label[d] for d in unique_docs])

        for fold, (tr_di, va_di) in enumerate(skf.split(unique_docs, doc_y), 1):
            train_docs = {unique_docs[i] for i in tr_di}
            val_docs = {unique_docs[i] for i in va_di}
            train_idx = [i for i, d in enumerate(doc_names) if d in train_docs]
            val_idx = [i for i, d in enumerate(doc_names) if d in val_docs]
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            model = make_mlp(in_dim, hidden_dims)
            bf = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
            m = compute_metrics(y_vl, bf)
            fold_metrics.append(m)
            print(f"      Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%")
    else:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            model = make_mlp(in_dim, hidden_dims)
            bf = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
            m = compute_metrics(y_vl, bf)
            fold_metrics.append(m)
            print(f"      Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%")

    return fold_metrics


# ═══════════════════════════════════════════════════════════════
# Backbone configurations
# ═══════════════════════════════════════════════════════════════

BACKBONES = {
    'MobileNetV3-Small': {
        'face_dim': 576,
        'text_dim': 576,
        'both_dim': 1152,
        'params_backbone': 2_542_856,
        'flops_M': 118.7,
        'face_hidden': [256, 128, 64, 32],
        'text_hidden': [256, 128, 64, 32],
        'both_hidden': [512, 256, 128, 64],
        'load_face': load_mnv3_face,
        'load_text': load_mnv3_text,
        'load_both': load_mnv3_both,
    },
    'EfficientNet-B0': {
        'face_dim': 1280,
        'text_dim': 1280,
        'both_dim': 2560,
        'params_backbone': 5_288_548,
        'flops_M': 390.0,
        'face_hidden': [256, 128, 64, 32],
        'text_hidden': [256, 128, 64, 32],
        'both_hidden': [512, 256, 128, 64],
        'load_face': load_eff_face,
        'load_text': load_eff_text,
        'load_both': load_eff_both,
    },
    'ResNet-18': {
        'face_dim': 512,
        'text_dim': 512,
        'both_dim': 1024,
        'params_backbone': 11_689_512,
        'flops_M': 1_820.0,
        'face_hidden': [256, 128, 64, 32],
        'text_hidden': [256, 128, 64, 32],
        'both_hidden': [512, 256, 128, 64],
        'load_face': load_resnet_face,
        'load_text': load_resnet_text,
        'load_both': load_resnet_both,
    },
}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    out_dir = Path(__file__).resolve().parent.parent / 'results' / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for bb_name, cfg in BACKBONES.items():
        print(f"\n{'='*60}")
        print(f"  Backbone: {bb_name}")
        print(f"  Params: {cfg['params_backbone']:,}  FLOPs: {cfg['flops_M']:.1f}M")
        print(f"{'='*60}")

        all_results[bb_name] = {
            'params_backbone': cfg['params_backbone'],
            'flops_M': cfg['flops_M'],
        }

        for attack in ['Face', 'Text', 'Both']:
            print(f"\n  --- {attack} Attack ---")
            # Reset seeds for reproducibility
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            loader = cfg[f'load_{attack.lower()}']
            result = loader()
            if len(result) == 3:
                X, y, doc_names = result
            else:
                X, y = result
                doc_names = None

            in_dim = cfg[f'{attack.lower()}_dim']
            hidden = cfg[f'{attack.lower()}_hidden']
            doc_level = (attack == 'Face' and doc_names is not None)

            print(f"    Samples: {len(X)} ({(y==0).sum()}R + {(y==1).sum()}F)  dim={X.shape[1]}")

            fold_metrics = run_kfold(X, y, in_dim, hidden, device,
                                     doc_names=doc_names, doc_level=doc_level)

            # Aggregate
            metrics_summary = {}
            for key in ['auc', 'eer', 'accuracy', 'f1', 'bpcer10', 'bpcer20', 'bpcer50']:
                vals = [m[key] for m in fold_metrics]
                metrics_summary[key] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'std': round(float(np.std(vals)), 4),
                    'per_fold': [round(float(v), 4) for v in vals],
                }

            all_results[bb_name][f'{attack}_attack'] = metrics_summary

            auc = metrics_summary['auc']
            eer = metrics_summary['eer']
            print(f"    >> AUC={auc['mean']:.4f}±{auc['std']:.4f}  "
                  f"EER={eer['mean']:.2f}±{eer['std']:.2f}%")

    # ── Save ──
    out_path = out_dir / 'backbone_kfold_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Print summary table ──
    print(f"\n{'='*80}")
    print("  BACKBONE ABLATION SUMMARY (5-fold CV)")
    print(f"{'='*80}")
    print(f"{'Backbone':<20s} {'Params':>10s} {'FLOPs':>10s} "
          f"{'Face AUC':>12s} {'Text AUC':>12s} {'Both AUC':>12s}")
    print("-" * 80)
    for bb_name, res in all_results.items():
        fa = res.get('Face_attack', {}).get('auc', {})
        ta = res.get('Text_attack', {}).get('auc', {})
        ba = res.get('Both_attack', {}).get('auc', {})
        print(f"{bb_name:<20s} {res['params_backbone']:>10,} {res['flops_M']:>8.1f}M "
              f"{fa.get('mean',0):.3f}±{fa.get('std',0):.3f}  "
              f"{ta.get('mean',0):.3f}±{ta.get('std',0):.3f}  "
              f"{ba.get('mean',0):.3f}±{ba.get('std',0):.3f}")


if __name__ == '__main__':
    main()
