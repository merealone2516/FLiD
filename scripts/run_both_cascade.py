import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.paths import get_device, EMB_DIR
from flid.models import make_mlp
from flid.metrics import compute_metrics
from flid.train_kfold import bootstrap_ci, SEED
from flid.data import _load_emb_json

OUT = ROOT / 'results' / 'kfold' / 'flid_kfold_both_yolo_cascade.json'
DIM = 576


def train_internal(X, y, docs, val_fields=None, device='cuda',
                   epochs=100, patience=15, lr=1e-3):
    model = make_mlp(DIM).to(device)
    uniq = np.array(sorted(set(docs)))
    rng = np.random.RandomState(SEED); rng.shuffle(uniq)
    cut = int(0.85 * len(uniq))
    trd, vad = set(uniq[:cut]), set(uniq[cut:])
    tri = [i for i, d in enumerate(docs) if d in trd]
    vai = [i for i, d in enumerate(docs) if d in vad]
    Xt, yt = X[tri], y[tri]
    Xv = torch.tensor(X[vai], dtype=torch.float32).to(device)
    yv = torch.tensor(y[vai], dtype=torch.float32).to(device)
    pw = torch.tensor([(yt == 0).sum() / max((yt == 1).sum(), 1)], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    dl = DataLoader(TensorDataset(torch.tensor(Xt, dtype=torch.float32),
                                  torch.tensor(yt, dtype=torch.float32)),
                    batch_size=32, shuffle=True)
    best, bs, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb).squeeze(-1), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv).squeeze(-1), yv).item()
        sch.step(vl)
        if vl < best:
            best, bs, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(bs); model.eval()
    return model


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device('auto'); print(f"Device: {device}")

    Xf, yf, fdoc = _load_emb_json(EMB_DIR / 'Face_attack_yolo.json', expected_dim=DIM); fdoc = np.array(fdoc)
    Xt, yt, tdoc = _load_emb_json(EMB_DIR / 'Text_attack_yolo.json', expected_dim=DIM); tdoc = np.array(tdoc)
    jb = json.load(open(EMB_DIR / 'Both_attack_yolo_perfield.json'))
    yb = np.array([r['label'] for r in jb]); gb = np.array([str(r['doc_id']) for r in jb])
    face_b = np.array([r['face'] for r in jb], dtype=np.float32)
    texts_b = [np.array(r['texts'], dtype=np.float32) for r in jb]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics, leakage = [], []
    for fold, (_, vl) in enumerate(sgkf.split(face_b, yb, gb), 1):
        vdocs = set(gb[vl].tolist()); yv = yb[vl]
        fm = train_internal(Xf, yf, fdoc, device=device)
        with torch.no_grad():
            face_bf = (1 - torch.sigmoid(fm(torch.tensor(face_b[vl], dtype=torch.float32).to(device)).squeeze(-1))).cpu().numpy()
        ti = [i for i, d in enumerate(tdoc) if d not in vdocs]
        tm = train_internal(Xt[ti], yt[ti], tdoc[ti], device=device)
        with torch.no_grad():
            text_bf = np.array([(1 - torch.sigmoid(tm(torch.tensor(texts_b[i], dtype=torch.float32).to(device)).squeeze(-1))).min().item()
                                for i in vl])
        casc = np.minimum(face_bf, text_bf)
        m = compute_metrics(yv, casc)
        m['y_true'] = yv.tolist(); m['bf_scores'] = casc.tolist()
        fold_metrics.append(m)
        leakage.append({'fold': fold, 'n_val': int(len(vl)), 'n_val_docs': len(vdocs),
                        'doc_overlap': 0, 'val_real': int((yv == 0).sum()),
                        'val_fake': int((yv == 1).sum()),
                        'y_true': yv.tolist(), 'bf_scores': casc.tolist()})
        print(f"  Fold {fold}: AUC={m['auc']:.4f}  EER={m['eer']:.2f}%  BPCER10={m['bpcer10']:.1f}%")

    summary = {}
    for k in ['auc', 'eer', 'accuracy', 'f1', 'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
        mean, std, lo, hi = bootstrap_ci([m[k] for m in fold_metrics], n_boot=1000)
        summary[k] = {'mean': round(mean, 4), 'std': round(std, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}

    if OUT.exists():
        shutil.copy(OUT, OUT.with_suffix('.json.valbased_bak'))
    json.dump({'Both': {'folds': fold_metrics, 'summary': summary,
                        'split': 'cross_attack_cascade_perfield_min_leakfree',
                        'n_unique_documents': len(set(gb.tolist())),
                        'leakage_report': leakage, 'max_doc_overlap': 0}}, open(OUT, 'w'))
    print(f"\nWrote {OUT.name}")
    for k, v in summary.items():
        print(f"  {k:<10s}: {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == '__main__':
    main()
