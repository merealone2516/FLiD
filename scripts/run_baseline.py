import json, shutil, sys, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
from configs.paths import TRAIN_TEST_DATA, get_device
from baseline.model import MobileNetV2PAD, get_train_transforms, get_val_transforms
from baseline.train_kfold import (ImageDataset, load_image_paths, bootstrap_ci,
                                  SEED, BATCH_SIZE, LR, EPOCHS, PATIENCE, NUM_WORKERS)
from flid.metrics import compute_metrics

OUT = ROOT / 'results' / 'kfold' / 'baseline_kfold_results.json'


def train_lf(tr_paths, tr_labels, tr_docs, score_paths, score_labels, device):
    """Inner doc-disjoint 85/15 split for early stopping; score on score_*."""
    tr_docs = np.asarray(tr_docs)
    uniq = np.array(sorted(set(tr_docs.tolist()))); rng = np.random.RandomState(SEED); rng.shuffle(uniq)
    cut = int(0.85 * len(uniq)); trd, vad = set(uniq[:cut]), set(uniq[cut:])
    ti = [i for i, d in enumerate(tr_docs) if d in trd]
    vi = [i for i, d in enumerate(tr_docs) if d in vad]
    if not vi:
        vi = ti[-max(1, len(ti) // 10):]
    itp = [tr_paths[i] for i in ti]; itl = tr_labels[ti]
    ivp = [tr_paths[i] for i in vi]; ivl = tr_labels[vi]

    train_ds = ImageDataset(itp, itl, get_train_transforms())
    inval_ds = ImageDataset(ivp, ivl, get_val_transforms())
    score_ds = ImageDataset(score_paths, score_labels, get_val_transforms())

    lc = Counter(itl.tolist()); n = len(itl); nc = len(lc)
    cw = {c: n / (nc * cnt) for c, cnt in lc.items()}
    wt = torch.tensor([cw.get(0, 1), cw.get(1, 1)], dtype=torch.float32).to(device)
    sw = [cw[l] for l in itl.tolist()]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    inval_loader = DataLoader(inval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    score_loader = DataLoader(score_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MobileNetV2PAD(2, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=wt)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    best, bs, wait = float('inf'), None, 0
    for _ in range(EPOCHS):
        model.train()
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad(); criterion(model(imgs), labs).backward(); optimizer.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for imgs, labs in inval_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                vl += criterion(model(imgs), labs).item() * imgs.size(0)
        vl /= len(inval_ds); scheduler.step()
        if vl < best:
            best, bs, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(bs); model.eval()
    yl, ss = [], []
    with torch.no_grad():
        for imgs, labs in score_loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1)
            yl.extend(labs.tolist()); ss.extend(probs[:, 0].cpu().tolist())
    return np.array(yl), np.array(ss)


def summarize(fm):
    s = {}
    for k in ['auc', 'eer', 'accuracy', 'f1', 'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
        mean, std, lo, hi = bootstrap_ci([m[k] for m in fm])
        s[k] = {'mean': round(mean, 4), 'std': round(std, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}
    return s


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device('auto'); dr = TRAIN_TEST_DATA
    results = {}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    for att in ['Face_attack', 'Text_attack']:
        paths, labels, docs = load_image_paths(att, dr); g = np.asarray(docs)
        print(f"\n[{att}] {len(paths)} imgs, {len(set(docs))} docs")
        fm = []
        for fold, (tr, vl) in enumerate(sgkf.split(paths, labels, g), 1):
            yv, sv = train_lf([paths[i] for i in tr], labels[tr], g[tr],
                              [paths[i] for i in vl], labels[vl], device)
            m = compute_metrics(yv, sv); m['y_true'] = yv.tolist(); m['bf_scores'] = sv.tolist()
            fm.append(m); print(f"  Fold {fold}: AUC={m['auc']:.4f} EER={m['eer']:.2f}")
        results[att] = {'folds': fm, 'summary': summarize(fm)}

    # Both cascade
    fp, fl, fd = load_image_paths('Face_attack', dr)
    tp, tl, td = load_image_paths('Text_attack', dr)
    bp, bl, bd = load_image_paths('Both_attack', dr); bg = np.asarray(bd)
    print(f"\n[Both cascade] Both={len(bp)} ({len(set(bd))} docs)")
    fm = []
    for fold, (_, vl) in enumerate(sgkf.split(bp, bl, bg), 1):
        bvp = [bp[i] for i in vl]; bvl = bl[vl]; vdoc = set(bg[vl].tolist())
        _, fs = train_lf(fp, fl, fd, bvp, bvl, device)
        ti = [i for i, d in enumerate(td) if d not in vdoc]
        _, ts = train_lf([tp[i] for i in ti], tl[ti], np.asarray(td)[ti], bvp, bvl, device)
        cs = np.minimum(fs, ts)
        m = compute_metrics(bvl, cs); m['y_true'] = bvl.tolist(); m['bf_scores'] = cs.tolist()
        fm.append(m); print(f"  Fold {fold}: AUC={m['auc']:.4f} EER={m['eer']:.2f}")
    results['Both_attack'] = {'folds': fm, 'summary': summarize(fm)}

    if OUT.exists():
        shutil.copy(OUT, OUT.with_suffix('.json.testfold_es_bak'))
    json.dump(results, open(OUT, 'w'))
    print("\n==== BASELINE (5-fold CV) ====")
    for k in results:
        s = results[k]['summary']
        print(f"  {k:<12} AUC={s['auc']['mean']:.3f}  EER={s['eer']['mean']:.2f}")


if __name__ == '__main__':
    main()
