import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

RES = Path(__file__).resolve().parent.parent / 'results'

# (label, path, attack_key) — FLiD uses YOLO11 crops; cascade for Both.
FLID = [
    ('Face', RES / 'kfold' / 'flid_kfold_face_yolo.json',          'Face'),
    ('Text', RES / 'kfold' / 'flid_kfold_text_yolo.json',          'Text'),
    ('Both', RES / 'kfold' / 'flid_kfold_both_yolo_cascade.json',  'Both'),
]
BASE_FILE = RES / 'kfold' / 'baseline_kfold_results.json'
BASE_KEYS = {'Face': 'Face_attack', 'Text': 'Text_attack', 'Both': 'Both_attack'}

COLOURS = {'Face': '#d62728', 'Text': '#1f77b4', 'Both': '#2ca02c'}
TITLES  = {'Face': 'Face Attack', 'Text': 'Text Attack', 'Both': 'Face+Text Attack'}


def fold_scores(block):
    """Yield (y_true, bf_scores) per fold from folds or leakage_report."""
    folds = block.get('folds', [])
    if folds and 'bf_scores' in folds[0]:
        src = folds
    else:
        src = block.get('leakage_report', [])
    return [(np.asarray(f['y_true']), np.asarray(f['bf_scores'])) for f in src]


def summary_auc(block):
    a = block['summary']['auc']
    return a['mean'], a['std']


def mean_roc(per_fold):
    """Interpolate per-fold ROC onto a common FPR grid; return grid, mean, std."""
    grid = np.linspace(0, 1, 200)
    tprs = []
    for y, bf in per_fold:
        # attack = positive class (label 1); score = 1 - bona_fide
        fpr, tpr, _ = roc_curve(y, 1.0 - bf)
        tprs.append(np.interp(grid, fpr, tpr))
        tprs[-1][0] = 0.0
    tprs = np.vstack(tprs)
    return grid, tprs.mean(0), tprs.std(0)


def plot_roc():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    base = json.load(open(BASE_FILE))

    for label, path, key in FLID:
        block = json.load(open(path))[key]
        grid, mean, std = mean_roc(fold_scores(block))
        m, s = summary_auc(block)
        c = COLOURS[label]
        ax.plot(grid, mean, color=c, lw=2.2,
                label=f'FLiD – {TITLES[label]} (AUC={m:.3f}±{s:.3f})')
        ax.fill_between(grid, np.clip(mean-std, 0, 1), np.clip(mean+std, 0, 1),
                        color=c, alpha=0.15)

    for label in ['Face', 'Text', 'Both']:
        block = base[BASE_KEYS[label]]
        grid, mean, _ = mean_roc(fold_scores(block))
        m, s = summary_auc(block)
        ax.plot(grid, mean, color=COLOURS[label], lw=1.6, ls='--',
                label=f'Baseline – {TITLES[label]} (AUC={m:.3f}±{s:.3f})')

    ax.plot([0, 1], [0, 1], color='grey', lw=1, ls=':', label='Random')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate (APCER)')
    ax.set_ylabel('True Positive Rate (1 - BPCER)')
    ax.set_title('ROC Curves — Baseline vs. FLiD YOLO11 (5-Fold CV)')
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(RES / f'roc_yolo11.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Wrote roc_yolo11.png / .pdf')


def eer_threshold(y, bf):
    tau = np.linspace(0, 1, 5000)
    pa = bf[y == 1]; bo = bf[y == 0]
    apcer = np.array([np.mean(pa > t) for t in tau])
    bpcer = np.array([np.mean(bo <= t) for t in tau])
    return tau[np.argmin(np.abs(bpcer - apcer))]


def plot_scores():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    base = json.load(open(BASE_FILE))

    rows = [('FLiD', FLID),
            ('Baseline', [(l, BASE_FILE, BASE_KEYS[l]) for l, _, _ in FLID])]

    for r, (row_label, cfg) in enumerate(rows):
        for cidx, (label, path, key) in enumerate(cfg):
            ax = axes[r][cidx]
            block = (base[key] if row_label == 'Baseline'
                     else json.load(open(path))[key])
            ys, bfs = [], []
            for y, bf in fold_scores(block):
                ys.append(y); bfs.append(bf)
            y = np.concatenate(ys); bf = np.concatenate(bfs)
            bona = bf[y == 0]; atk = bf[y == 1]
            bins = np.linspace(0, 1, 31)
            ax.hist(bona, bins=bins, color='#2ca02c', alpha=0.6, label='Bona fide', density=True)
            ax.hist(atk,  bins=bins, color='#ff7f0e', alpha=0.6, label='Attack', density=True)
            thr = eer_threshold(y, bf)
            ax.axvline(thr, color='k', ls='--', lw=1.2, label=f'EER thr={thr:.2f}')
            m = block['summary']['auc']['mean']
            ax.set_title(f'{row_label} – {TITLES[label]} (AUC={m:.3f})', fontsize=10)
            ax.set_xlabel('Bona-fide score P(Real)')
            if cidx == 0:
                ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

    fig.suptitle('Score Distributions — FLiD YOLO11 vs. Baseline (pooled 5-fold)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ('png', 'pdf'):
        fig.savefig(RES / f'score_yolo11.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Wrote score_yolo11.png / .pdf')


if __name__ == '__main__':
    plot_roc()
    plot_scores()
    print('Done.')
