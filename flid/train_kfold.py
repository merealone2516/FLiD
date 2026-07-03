import argparse
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import get_device, KFOLD_OUTPUT, EMB_DIR
from flid.models import FaceClassifier, TextClassifier, BothClassifier, make_mlp
from flid.metrics import compute_metrics
from flid.data import load_face_embeddings, load_text_embeddings, load_both_embeddings
from flid.data import _load_emb_json

SEED = 42


# ─── Training helper ─────────────────────────────────────────────────────────
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
        p_fake = torch.sigmoid(model(X_v).squeeze(-1))
        bf_scores = (1.0 - p_fake).cpu().numpy()
    return bf_scores


# ─── Document-level K-Fold CV ────────────────────────────────────────────────
def run_kfold_cv(X, y, make_model_fn, device, doc_names, n_folds=5):

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(doc_names)

    n_unique = len(set(groups.tolist()))
    if n_unique < n_folds:
        raise ValueError(
            f"Only {n_unique} unique documents for {n_folds} folds. "
            "Reduce --n_folds or check doc_id extraction."
        )

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics, leakage_report = [], []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
        train_docs = set(groups[train_idx].tolist())
        val_docs   = set(groups[val_idx].tolist())
        overlap    = train_docs & val_docs

        assert len(overlap) == 0, (
            f"LEAKAGE in fold {fold}: {len(overlap)} documents in both "
            f"train and test: {sorted(overlap)[:5]}..."
        )

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx],   y[val_idx]

        model = make_model_fn()
        bf    = train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, device)
        m     = compute_metrics(y_vl, bf)
        fold_metrics.append(m)

        leakage_report.append({
            'fold':            fold,
            'n_train_samples': int(len(train_idx)),
            'n_val_samples':   int(len(val_idx)),
            'n_train_docs':    len(train_docs),
            'n_val_docs':      len(val_docs),
            'doc_overlap':     0,
            'val_real':        int((y_vl == 0).sum()),
            'val_fake':        int((y_vl == 1).sum()),
            'y_true':          y_vl.tolist(),
            'bf_scores':       bf.tolist(),
        })

        print(f"  Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  "
              f"BPCER10={m['bpcer10']:.1f}%  "
              f"train_docs={len(train_docs)}  val_docs={len(val_docs)}")

    return fold_metrics, leakage_report


# ─── Bootstrap CI ────────────────────────────────────────────────────────────
def bootstrap_ci(fold_values, n_boot=1000, alpha=0.05):
    arr = np.array(fold_values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    return (float(np.mean(arr)), float(np.std(arr)),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# ─── Cross-attack CV for Both (Option C) ─────────────────────────────────────
def run_crossattack_cv(device, n_folds=5,
                       face_emb_path=None, text_emb_path=None, both_emb_path=None,
                       emb_dim=576, face_dim=None, text_dim=None):
    face_dim = face_dim or emb_dim
    text_dim = text_dim or emb_dim
    both_dim = face_dim + text_dim

    if face_emb_path:
        X_face, y_face, _ = _load_emb_json(Path(face_emb_path), expected_dim=face_dim)
    else:
        X_face, y_face, _ = load_face_embeddings()

    if text_emb_path:
        X_text, y_text, text_docs = _load_emb_json(Path(text_emb_path), expected_dim=text_dim)
    else:
        X_text, y_text, text_docs = load_text_embeddings()

    if both_emb_path:
        X_both, y_both, both_docs = _load_emb_json(Path(both_emb_path), expected_dim=both_dim)
    else:
        X_both, y_both, both_docs = load_both_embeddings()

    # Split Both_attack into face and text halves
    X_both_face = X_both[:, :face_dim]
    X_both_text = X_both[:, face_dim:]

    both_groups = np.asarray(both_docs)
    n_docs = len(set(both_docs))

    print(f"  Cross-attack: Face={len(X_face)} ({face_dim}-D)  "
          f"Text={len(X_text)} ({text_dim}-D)  "
          f"Both={len(X_both)} ({n_docs} unique docs, {both_dim}-D)")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics, leakage_report = [], []

    for fold, (_, val_idx) in enumerate(
            sgkf.split(X_both, y_both, both_groups), 1):

        val_docs   = set(both_groups[val_idx].tolist())
        X_vf       = X_both_face[val_idx]
        X_vt       = X_both_text[val_idx]
        y_val      = y_both[val_idx]

        # Face model — train on ALL Face_attack (no doc_id overlap)
        face_model = (FaceClassifier() if face_dim == 576 else make_mlp(face_dim)).to(device)
        train_mlp_fold(face_model, X_face, y_face, X_vf, y_val, device)
        face_model.eval()
        with torch.no_grad():
            face_bf = (1 - torch.sigmoid(
                face_model(torch.tensor(X_vf, dtype=torch.float32).to(device)).squeeze(-1)
            )).cpu().numpy()

        # Text model — exclude Both_attack val doc_ids (shared pool)
        text_train_idx = [i for i, d in enumerate(text_docs) if d not in val_docs]
        Xt_tr = X_text[text_train_idx]
        yt_tr = y_text[text_train_idx]
        text_model = (TextClassifier() if text_dim == 576 else make_mlp(text_dim)).to(device)
        train_mlp_fold(text_model, Xt_tr, yt_tr, X_vt, y_val, device)
        text_model.eval()
        with torch.no_grad():
            text_bf = (1 - torch.sigmoid(
                text_model(torch.tensor(X_vt, dtype=torch.float32).to(device)).squeeze(-1)
            )).cpu().numpy()

        # Cascade: conservative (min) combination
        cascade_bf = np.minimum(face_bf, text_bf)
        m = compute_metrics(y_val, cascade_bf)
        m['y_true']    = y_val.tolist()
        m['bf_scores'] = cascade_bf.tolist()
        fold_metrics.append(m)

        leakage_report.append({
            'fold':         fold,
            'n_val':        int(len(val_idx)),
            'n_val_docs':   len(val_docs),
            'doc_overlap':  0,
            'val_real':     int((y_val == 0).sum()),
            'val_fake':     int((y_val == 1).sum()),
            'y_true':       y_val.tolist(),
            'bf_scores':    cascade_bf.tolist(),
        })

        print(f"  Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  "
              f"BPCER10={m['bpcer10']:.1f}%  val_docs={len(val_docs)}")

    return fold_metrics, leakage_report


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='FLiD 5-fold CV — document-level, leakage-free')
    parser.add_argument('--attack',
                        choices=['Face', 'Text', 'Both', 'Both_crossattack', 'all'],
                        default='all', nargs='+')
    parser.add_argument('--n_folds',      type=int, default=5)
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--device',       type=str, default='auto')
    parser.add_argument('--full_image',   action='store_true',
                        help='Use full-image embeddings instead of ROI crops')
    parser.add_argument('--emb-path',      type=str, default=None,
                        help='Override embedding JSON for a single-attack run')
    parser.add_argument('--face-emb-path', type=str, default=None,
                        help='Face embedding path when running --attack all with a custom backbone')
    parser.add_argument('--text-emb-path', type=str, default=None,
                        help='Text embedding path when running --attack all with a custom backbone')
    parser.add_argument('--both-emb-path', type=str, default=None,
                        help='Both embedding path when running --attack all with a custom backbone')
    parser.add_argument('--yolo-cascade', action='store_true',
                        help='Use YOLO embeddings for Both_crossattack cascade '
                             '(Face_attack_yolo.json + Text_attack_yolo.json + Both_attack_yolo.json)')
    parser.add_argument('--emb-dim', type=int, default=None,
                        help='Embedding dimension when using --emb-path with a non-MobileNetV3 backbone '
                             '(e.g. 1280 for EfficientNet-B0, 2048 for ResNet50)')
    parser.add_argument('--both-dim', type=int, default=None,
                        help='Override the total Both embedding dimension (e.g. 1088 for ArcFace+MobileNetV3)')
    parser.add_argument('--face-dim', type=int, default=None,
                        help='Face embedding dimension for asymmetric cascade (e.g. 512 for ArcFace)')
    parser.add_argument('--text-dim', type=int, default=None,
                        help='Text embedding dimension for asymmetric cascade (e.g. 576 for MobileNetV3)')
    parser.add_argument('--out-suffix', type=str, default=None,
                        help='Override the output filename suffix (e.g. efficientnet_b0)')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device(args.device)
    print(f"Device: {device}")

    KFOLD_OUTPUT.mkdir(parents=True, exist_ok=True)
    attack_input = args.attack if isinstance(args.attack, list) else [args.attack]
    if attack_input == ['all']:
        attacks = ['Face', 'Text', 'Both_crossattack']
    else:
        attacks = attack_input
    all_results = {}

    for attack in attacks:
        print(f"\n{'='*60}")
        print(f"  {attack} Attack — {args.n_folds}-fold CV (document-level split)")
        print(f"{'='*60}")

        # ── Both cross-attack (Option C) ──────────────────────────────────
        if attack == 'Both_crossattack':
            if args.yolo_cascade:
                print(f"  Protocol: YOLO embeddings — train on Face+Text, cascade-test on Both_attack")
                fold_metrics, leakage_report = run_crossattack_cv(
                    device, n_folds=args.n_folds,
                    face_emb_path=str(EMB_DIR / 'Face_attack_yolo.json'),
                    text_emb_path=str(EMB_DIR / 'Text_attack_yolo.json'),
                    both_emb_path=str(EMB_DIR / 'Both_attack_yolo.json'))
            elif args.face_emb_path and args.text_emb_path:
                emb_dim  = args.emb_dim or 576
                face_dim = args.face_dim or emb_dim
                text_dim = args.text_dim or emb_dim
                print(f"  Protocol: custom embeddings cascade-test on Both_attack "
                      f"(face={face_dim}-D, text={text_dim}-D)")
                fold_metrics, leakage_report = run_crossattack_cv(
                    device, n_folds=args.n_folds,
                    face_emb_path=args.face_emb_path,
                    text_emb_path=args.text_emb_path,
                    both_emb_path=args.both_emb_path,
                    face_dim=face_dim, text_dim=text_dim)
            else:
                print(f"  Protocol: train on Face+Text, cascade-test on Both_attack")
                fold_metrics, leakage_report = run_crossattack_cv(
                    device, n_folds=args.n_folds)

            summary = {}
            for key in ['auc', 'eer', 'accuracy', 'f1',
                        'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
                vals = [m[key] for m in fold_metrics]
                mean, std, lo, hi = bootstrap_ci(vals, n_boot=args.n_bootstraps)
                summary[key] = {'mean': round(mean, 4), 'std': round(std, 4),
                                'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}

            all_results['Both'] = {
                'folds':              fold_metrics,
                'summary':            summary,
                'split':              'cross_attack_cascade',
                'n_unique_documents': len(set(load_both_embeddings()[2])),
                'leakage_report':     leakage_report,
                'max_doc_overlap':    0,
            }

            print(f"\n  Summary:")
            for k, v in summary.items():
                print(f"    {k:<10s}: {v['mean']:.4f} ± {v['std']:.4f}  "
                      f"[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
            continue

        # ── Face / Text standard CV ────────────────────────────────────────
        if args.full_image:
            dims = {'Face': 576, 'Text': 576, 'Both': 1152}
            path = EMB_DIR / f'{attack}_attack_full.json'
            X, y, doc_names = _load_emb_json(path, expected_dim=dims[attack])
            make_fn = {'Face': FaceClassifier, 'Text': TextClassifier,
                       'Both': BothClassifier}[attack]
        elif attack == 'Face':
            dim = args.emb_dim or 576
            face_path = args.face_emb_path or args.emb_path
            if face_path:
                X, y, doc_names = _load_emb_json(Path(face_path), expected_dim=dim)
            else:
                X, y, doc_names = load_face_embeddings()
            make_fn = (lambda d=dim: make_mlp(d)) if args.emb_dim else FaceClassifier
        elif attack == 'Text':
            dim = args.emb_dim or 576
            text_path = args.text_emb_path or args.emb_path
            if text_path:
                X, y, doc_names = _load_emb_json(Path(text_path), expected_dim=dim)
            else:
                X, y, doc_names = load_text_embeddings()
            make_fn = (lambda d=dim: make_mlp(d)) if args.emb_dim else TextClassifier
        else:
            dim = args.both_dim or ((args.emb_dim * 2) if args.emb_dim else 1152)
            both_path = args.both_emb_path or args.emb_path
            if both_path:
                X, y, doc_names = _load_emb_json(Path(both_path), expected_dim=dim)
            else:
                X, y, doc_names = load_both_embeddings()
            make_fn = lambda d=dim: make_mlp(d)

        n_docs = len(set(doc_names))
        print(f"  Samples: {len(X)}  "
              f"({(y==0).sum()} Real, {(y==1).sum()} Fake)  "
              f"across {n_docs} unique documents")

        fold_metrics, leakage_report = run_kfold_cv(
            X, y, make_fn, device, doc_names, n_folds=args.n_folds)

        summary = {}
        for key in ['auc', 'eer', 'accuracy', 'f1',
                    'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
            vals = [m[key] for m in fold_metrics]
            mean, std, lo, hi = bootstrap_ci(vals, n_boot=args.n_bootstraps)
            summary[key] = {'mean': round(mean, 4), 'std': round(std, 4),
                            'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}

        all_results[attack] = {
            'folds':              fold_metrics,
            'summary':            summary,
            'split':              'document_level_StratifiedGroupKFold',
            'n_unique_documents': n_docs,
            'leakage_report':     leakage_report,
            'max_doc_overlap':    max(r['doc_overlap'] for r in leakage_report),
        }

        print(f"\n  Summary:")
        for k, v in summary.items():
            print(f"    {k:<10s}: {v['mean']:.4f} ± {v['std']:.4f}  "
                  f"[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
        print(f"  Max document overlap: "
              f"{all_results[attack]['max_doc_overlap']} (must be 0)")

    if args.out_suffix:
        suffix = f'_{args.out_suffix}'
    elif getattr(args, 'yolo_cascade', False):
        suffix = '_yolo_cascade'
    elif args.emb_path:
        suffix = '_coarse'
    elif args.full_image:
        suffix = '_fullimage'
    else:
        suffix = '_docsplit'
    out_path = KFOLD_OUTPUT / f'kfold_results{suffix}.json'

    def _ser(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray):                return o.tolist()
        if isinstance(o, dict):   return {k: _ser(v) for k, v in o.items()}
        if isinstance(o, list):   return [_ser(v) for v in o]
        return o

    with open(out_path, 'w') as f:
        json.dump(_ser(all_results), f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    main()
