import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from configs.paths import BASE, EMB_DIR, EFFNET_B0_DIR, RESNET50_DIR, get_device
from flid.models import (MobileNetV3Extractor, EfficientNetExtractor,
                         ResNet50Extractor, make_mlp)
from flid.metrics import compute_metrics
from flid.train_kfold import train_mlp_fold, bootstrap_ci, SEED
from flid.data import _load_emb_json
# GT crop helpers (region annotations, with Fake->Real fallback by face_id)
from backbone_ablation import (_collect, _fallback_regions, _text_crops,
                               _face_crop, _load_json, _face_id, _embed)
# YOLO crop helpers
from extract_yolo_embeddings import detect, get_face_crop, get_text_crops

BACKBONES = {
    'mobilenet':       (MobileNetV3Extractor, 576,  EMB_DIR),
    'efficientnet_b0': (EfficientNetExtractor, 1280, EFFNET_B0_DIR),
    'resnet50':        (ResNet50Extractor,     2048, RESNET50_DIR),
}
YOLO_W = BASE / 'yolo_finetuned' / 'field_detector' / 'weights' / 'best.pt'


# ── per-field Both extraction ────────────────────────────────────────────────
def extract_perfield(backbone, crop, device):
    cls, dim, embdir = BACKBONES[backbone]
    out = embdir / f'Both_{crop}_perfield.json'
    if out.exists():
        print(f"  [{backbone}/{crop}] per-field Both exists -> skip")
        return out
    extractor = cls().to(device).eval()
    res = []
    if crop == 'yolo':
        from ultralytics import YOLO
        yolo = YOLO(str(YOLO_W)); ydev = str(device).replace('cuda', '0')
        for img_path, label, split in _collect_yolo('Both_attack'):
            img = Image.open(img_path).convert('RGB')
            boxes, ids = detect(yolo, str(img_path), 0.3, ydev)
            face = get_face_crop(img, boxes, ids)
            tcrops = [c for c, _ in get_text_crops(img, boxes, ids)]
            res.append({'doc_id': _face_id(img_path.stem), 'label': label,
                        'face': _embed(face, extractor, device).tolist(),
                        'texts': [_embed(c, extractor, device).tolist() for c in tcrops]})
    else:  # gt
        entries = _collect('Both_attack')
        fallback = _fallback_regions('Both_attack')
        for img_path, jpath, label, split in entries:
            data = _load_json(jpath) if jpath else None
            if data and 'person_info' in data:
                doc_id = data['person_info'].get('face_id', _face_id(img_path.stem))
                regions = data.get('regions', [])
            else:
                doc_id = _face_id(img_path.stem)
                regions = fallback.get(doc_id, [])
            img = Image.open(img_path)
            face = _face_crop(img, regions)
            tcrops = _text_crops(img, regions)          # no label filter for Both
            res.append({'doc_id': doc_id, 'label': label,
                        'face': _embed(face, extractor, device).tolist(),
                        'texts': [_embed(c, extractor, device).tolist() for c in tcrops]})
    json.dump(res, open(out, 'w'))
    nt = [len(r['texts']) for r in res]
    print(f"  [{backbone}/{crop}] wrote {len(res)} docs, fields/doc mean={np.mean(nt):.1f} -> {out.name}")
    del extractor
    torch.cuda.empty_cache()
    return out


def _collect_yolo(attack_dir):
    """(img_path, label, split) like extract_yolo_embeddings.collect_files."""
    from configs.paths import TRAIN_TEST_DATA
    base = TRAIN_TEST_DATA / attack_dir
    out = []
    for ln, lab in [('Real', 0), ('Fake', 1)]:
        for sp in ['train', 'test']:
            d = base / ln / sp
            if d.exists():
                for ip in sorted(d.glob('*.jpg')):
                    out.append((ip, lab, sp))
    return out


# ── text head trained with per-field-min early stopping ──────────────────────
def train_text_perfield(dim, X_tr, y_tr, val_fields, y_val, device,
                        epochs=100, patience=15, lr=1e-3):
    model = make_mlp(dim).to(device)
    pw = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                      dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                  torch.tensor(y_tr, dtype=torch.float32)),
                    batch_size=32, shuffle=True)
    yvt = torch.tensor(y_val, dtype=torch.float32).to(device)
    best, bs, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb).squeeze(-1), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tb = [(1.0 - torch.sigmoid(model(torch.tensor(f, dtype=torch.float32).to(device)).squeeze(-1))).min().item()
                  for f in val_fields]
            pf = (1.0 - torch.tensor(tb, dtype=torch.float32).to(device)).clamp(1e-6, 1 - 1e-6)
            vl = nn.functional.binary_cross_entropy(pf, yvt).item()
        sch.step(vl)
        if vl < best:
            best, bs, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(bs); model.eval()
    return model


def run_cascade(backbone, crop, device):
    cls, dim, embdir = BACKBONES[backbone]
    facep = embdir / ('Face_attack_yolo.json' if crop == 'yolo' else 'Face_attack.json')
    textp = embdir / ('Text_attack_yolo.json' if crop == 'yolo' else 'Text_attack.json')
    perfp = embdir / f'Both_{crop}_perfield.json'

    Xf, yf, _    = _load_emb_json(facep, expected_dim=dim)
    Xt, yt, tdoc = _load_emb_json(textp, expected_dim=dim)
    tdoc = np.array(tdoc)
    jb = json.load(open(perfp))
    yb = np.array([r['label'] for r in jb]); gb = np.array([str(r['doc_id']) for r in jb])
    face_b = np.array([r['face'] for r in jb], dtype=np.float32)
    texts_b = [np.array(r['texts'], dtype=np.float32) for r in jb]

    np.random.seed(SEED); torch.manual_seed(SEED)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics, leakage = [], []
    for fold, (_, vl) in enumerate(sgkf.split(face_b, yb, gb), 1):
        vdocs = set(gb[vl].tolist()); yv = yb[vl]
        fm = train_mlp_fold(make_mlp(dim), Xf, yf, face_b[vl], yv, device)
        face_bf = fm if isinstance(fm, np.ndarray) else fm  # train_mlp_fold returns bf scores
        ti = [i for i, d in enumerate(tdoc) if d not in vdocs]
        tmodel = train_text_perfield(dim, Xt[ti], yt[ti], [texts_b[i] for i in vl], yv, device)
        with torch.no_grad():
            text_bf = np.array([(1.0 - torch.sigmoid(tmodel(torch.tensor(texts_b[i], dtype=torch.float32).to(device)).squeeze(-1))).min().item()
                                for i in vl])
        casc = np.minimum(face_bf, text_bf)
        m = compute_metrics(yv, casc)
        m['y_true'] = yv.tolist(); m['bf_scores'] = casc.tolist()
        fold_metrics.append(m)
        leakage.append({'fold': fold, 'n_val': int(len(vl)), 'n_val_docs': len(vdocs),
                        'doc_overlap': 0, 'val_real': int((yv == 0).sum()),
                        'val_fake': int((yv == 1).sum()),
                        'y_true': yv.tolist(), 'bf_scores': casc.tolist()})
    summary = {}
    for k in ['auc', 'eer', 'accuracy', 'f1', 'bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']:
        mean, std, lo, hi = bootstrap_ci([m[k] for m in fold_metrics], n_boot=1000)
        summary[k] = {'mean': round(mean, 4), 'std': round(std, 4),
                      'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}
    return summary, fold_metrics, leakage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='all')
    ap.add_argument('--crop', default='all')
    args = ap.parse_args()
    device = get_device('auto')
    bbs = list(BACKBONES) if args.backbone == 'all' else [args.backbone]
    crops = ['yolo', 'gt'] if args.crop == 'all' else [args.crop]

    rows = {}
    for bb in bbs:
        for cr in crops:
            print(f"\n=== {bb} / {cr} ===")
            extract_perfield(bb, cr, device)
            summary, folds, leak = run_cascade(bb, cr, device)
            rows[(bb, cr)] = summary
            outdir = ROOT / 'results' / ('resnet50' if bb == 'resnet50' else
                                         'efficientnet_b0' if bb == 'efficientnet_b0' else 'kfold')
            outdir.mkdir(parents=True, exist_ok=True)
            outp = outdir / f'both_perfield_{bb}_{cr}.json'
            json.dump({'Both': {'folds': folds, 'summary': summary,
                                'split': 'cross_attack_cascade_perfield_min',
                                'leakage_report': leak, 'max_doc_overlap': 0}}, open(outp, 'w'))
            a, e = summary['auc'], summary['eer']
            print(f"  AUC={a['mean']:.4f}±{a['std']:.4f}  EER={e['mean']:.2f}  -> {outp.name}")

    print("\n================ SUMMARY (per-field-min Both) ================")
    print(f"{'backbone':<16}{'crop':<6}{'AUC':>16}{'EER':>8}")
    for (bb, cr), s in rows.items():
        print(f"{bb:<16}{cr:<6}{s['auc']['mean']:>10.3f}±{s['auc']['std']:.3f}{s['eer']['mean']:>8.2f}")


if __name__ == '__main__':
    main()
