#!/usr/bin/env python3
"""
Leakage audit: compare train/test document overlap under the OLD sample-level
split (StratifiedKFold) vs the NEW document-level split (StratifiedGroupKFold).

Usage:
    python -m evaluation.leakage_audit
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flid.data import load_face_embeddings, load_text_embeddings, load_both_embeddings

SEED = 42
N_FOLDS = 5


def audit(name, X, y, doc_names):
    X      = np.asarray(X)
    y      = np.asarray(y)
    groups = np.asarray(doc_names)
    n_docs = len(set(groups.tolist()))

    print(f"\n=== {name} ===")
    print(f"samples={len(X)}  unique_documents={n_docs}")

    # OLD: sample-level split (what Text/Both used before the fix)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    old_overlaps = []
    for tr, va in skf.split(X, y):
        ov = set(groups[tr].tolist()) & set(groups[va].tolist())
        old_overlaps.append(len(ov))
    print(f"OLD sample-level split    doc overlap/fold: {old_overlaps}  "
          f"(mean {np.mean(old_overlaps):.1f})")

    # NEW: document-level split
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    new_overlaps = []
    for tr, va in sgkf.split(X, y, groups):
        ov = set(groups[tr].tolist()) & set(groups[va].tolist())
        new_overlaps.append(len(ov))
    print(f"NEW document-level split  doc overlap/fold: {new_overlaps}  "
          f"(mean {np.mean(new_overlaps):.1f})")

    assert all(o == 0 for o in new_overlaps), \
        "FAIL: new split still has document overlap!"
    print("PASS: zero document overlap with new split.")


def main():
    Xf, yf, df = load_face_embeddings()
    audit("Face", Xf, yf, df)

    try:
        Xt, yt, dt = load_text_embeddings()
        audit("Text", Xt, yt, dt)
    except Exception as e:
        print(f"\n=== Text ===\nSkipped: {e}")

    try:
        Xb, yb, db = load_both_embeddings()
        audit("Both", Xb, yb, db)
    except Exception as e:
        print(f"\n=== Both ===\nSkipped: {e}")


if __name__ == '__main__':
    main()
