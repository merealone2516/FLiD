import json, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.paths import EMB_DIR, EFFNET_B0_DIR, RESNET50_DIR, get_device
from flid.models import make_mlp
from flid.metrics import compute_metrics
from flid.train_kfold import bootstrap_ci, SEED
from flid.data import _load_emb_json

warnings.filterwarnings('ignore')
DEV = get_device('auto')
BB = {'mobilenet': (576, EMB_DIR), 'efficientnet_b0': (1280, EFFNET_B0_DIR), 'resnet50': (2048, RESNET50_DIR)}


def train_innerval(dim, Xtr, ytr, dtr, epochs=100, patience=15, lr=1e-3):
    """Inner doc-disjoint 85/15 split of the training fold for early stopping."""
    m = make_mlp(dim).to(DEV)
    uniq = np.array(sorted(set(dtr))); rng = np.random.RandomState(SEED); rng.shuffle(uniq)
    cut = int(0.85 * len(uniq)); trd, vad = set(uniq[:cut]), set(uniq[cut:])
    tri = [i for i, d in enumerate(dtr) if d in trd]
    vai = [i for i, d in enumerate(dtr) if d in vad]
    if not vai:  # tiny fold safety
        vai = tri[-max(1, len(tri) // 10):]
    Xi, yi = Xtr[tri], ytr[tri]
    Xv = torch.tensor(Xtr[vai], dtype=torch.float32).to(DEV)
    yv = torch.tensor(ytr[vai], dtype=torch.float32).to(DEV)
    pw = torch.tensor([(yi == 0).sum() / max((yi == 1).sum(), 1)], dtype=torch.float32).to(DEV)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    dl = DataLoader(TensorDataset(torch.tensor(Xi, dtype=torch.float32),
                                  torch.tensor(yi, dtype=torch.float32)), batch_size=32, shuffle=True)
    best, bs, wait = 1e9, None, 0
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); crit(m(xb).squeeze(-1), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xv).squeeze(-1), yv).item()
        sch.step(vl)
        if vl < best:
            best, bs, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(bs); m.eval()
    return m


def run_standard(path, dim):
    X, y, docs = _load_emb_json(path, expected_dim=dim); g = np.array(docs)
    np.random.seed(SEED); torch.manual_seed(SEED)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics, leak = [], []
    for fold, (tr, vl) in enumerate(sgkf.split(X, y, g), 1):
        m = train_innerval(dim, X[tr], y[tr], g[tr])
        with torch.no_grad():
            bf = (1 - torch.sigmoid(m(torch.tensor(X[vl], dtype=torch.float32).to(DEV)).squeeze(-1))).cpu().numpy()
        mm = compute_metrics(y[vl], bf)
        fold_metrics.append(mm)
        leak.append({'fold': fold, 'n_train_samples': int(len(tr)), 'n_val_samples': int(len(vl)),
                     'n_train_docs': len(set(g[tr])), 'n_val_docs': len(set(g[vl])), 'doc_overlap': 0,
                     'val_real': int((y[vl] == 0).sum()), 'val_fake': int((y[vl] == 1).sum()),
                     'y_true': y[vl].tolist(), 'bf_scores': bf.tolist()})
    summary = {}
    for k in ['auc', 'eer', 'accuracy', 'f1', 'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
        mean, std, lo, hi = bootstrap_ci([m[k] for m in fold_metrics], n_boot=1000)
        summary[k] = {'mean': round(mean, 4), 'std': round(std, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}
    return fold_metrics, summary, leak


def save(path, key, fm, summ, leak, n_docs):
    obj = {key: {'folds': fm, 'summary': summ, 'split': 'document_level_StratifiedGroupKFold_innerval',
                 'n_unique_documents': n_docs, 'leakage_report': leak, 'max_doc_overlap': 0}}
    json.dump(obj, open(path, 'w'))


CROPS = {'GT': '', 'YOLO': '_yolo', 'Coarse': '_coarse'}
rows = {}
for bb, (dim, ed) in BB.items():
    for att in ['Face', 'Text']:
        for cname, suf in CROPS.items():
            if att == 'Face' and cname == 'Coarse':
                continue  # Coarse Face == GT Face
            p = ed / f'{att}_attack{suf}.json'
            if not p.exists():
                continue
            fm, summ, leak = run_standard(p, dim)
            rows[(bb, cname, att)] = (summ['auc']['mean'], summ['auc']['std'], summ['eer']['mean'])
            print(f"{bb:<16}{cname:<7}{att:<5} AUC={summ['auc']['mean']:.3f}  EER={summ['eer']['mean']:.2f}")
            # save MobileNet YOLO main files (with scores) for plots
            if bb == 'mobilenet' and cname == 'YOLO':
                out = ROOT / 'results' / 'kfold' / (f'flid_kfold_{att.lower()}_yolo.json')
                save(out, att, fm, summ, leak, len(set(_load_emb_json(p, dim)[2])))
                print(f"    saved -> {out.name}")

print("\n==== Face/Text backbone x crop ablation (AUC / EER) ====")
for (bb, c, a), (au, sd, ee) in rows.items():
    print(f"{bb:<16}{c:<7}{a:<5} {au:.3f}  {ee:.2f}")
