#!/usr/bin/env python3
"""
Recreate the three combined cascade figures with updated leakage-free results:
  - eer_cascade.png   : 2x3 EER analysis (FLiD top, Baseline bottom)
  - roc_cascade.png   : single ROC plot all 6 curves
  - score_cascade.png : 2x3 score distributions (FLiD top, Baseline bottom)

Usage:
    python scripts/generate_cascade_plots.py
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

warnings.filterwarnings('ignore')

REPO      = Path(__file__).resolve().parent.parent
OUT_DIR   = REPO / 'outputs' / 'kfold_results'
PLOT_DIR  = REPO / 'results' / 'plots'
ATTACKS   = ['Face', 'Text', 'Both']
ATK_LABEL = {'Face': 'Face Attack', 'Text': 'Text Attack', 'Both': 'Both Attacks'}
COLORS    = {'Face': '#E53935', 'Text': '#1E88E5', 'Both': '#43A047'}


# ─── helpers ─────────────────────────────────────────────────────────────────
def load():
    flid = json.load(open(OUT_DIR / 'kfold_results_docsplit.json'))
    base = json.load(open(OUT_DIR / 'baseline_kfold_results.json'))
    return flid, base


def pool(results, attack, score_key='leakage_report'):
    """Pool y_true / bf_scores across all folds."""
    key = attack if attack in results else f'{attack}_attack'
    ys, ss = [], []
    for src in ['leakage_report', 'folds']:
        for fold in results[key].get(src, []):
            if 'y_true' in fold:
                ys.extend(fold['y_true'])
                ss.extend(fold['bf_scores'])
        if ys:
            break
    return np.array(ys), np.array(ss)


def per_fold_eer(results, attack):
    """Return list of per-fold EER values."""
    key = attack if attack in results else f'{attack}_attack'
    eers = []
    for src in ['leakage_report', 'folds']:
        for fold in results[key].get(src, []):
            if 'y_true' in fold:
                y = np.array(fold['y_true'])
                s = np.array(fold['bf_scores'])
                real_s, fake_s = s[y == 0], s[y == 1]
                tau = np.linspace(0, 1, 5000)
                ap  = np.array([np.mean(fake_s > t) for t in tau])
                bp  = np.array([np.mean(real_s <= t) for t in tau])
                idx = np.argmin(np.abs(ap - bp))
                eers.append((ap[idx] + bp[idx]) / 2 * 100)
        if eers:
            break
    return eers


def fig_save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → results/plots/{name}")


# ─── 1. EER cascade (2×3) ────────────────────────────────────────────────────
def plot_eer_cascade(flid, base):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('EER Analysis — FLiD vs. Baseline', fontsize=14, fontweight='bold')

    rows = [('FLiD', flid), ('Baseline', base)]
    for row_i, (method, results) in enumerate(rows):
        for col_i, attack in enumerate(ATTACKS):
            ax = axes[row_i][col_i]
            y, s = pool(results, attack)
            if len(y) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            real_s, fake_s = s[y == 0], s[y == 1]
            tau  = np.linspace(0, 1, 5000)
            apcer = np.array([np.mean(fake_s > t) for t in tau]) * 100
            bpcer = np.array([np.mean(real_s <= t) for t in tau]) * 100
            idx   = np.argmin(np.abs(apcer - bpcer))
            eer   = (apcer[idx] + bpcer[idx]) / 2
            eer_t = tau[idx]

            # per-fold EER for subtitle
            fold_eers = per_fold_eer(results, attack)
            mu  = np.mean(fold_eers)
            std = np.std(fold_eers)

            ax.plot(tau, apcer, color='red',  linewidth=1.8, label='APCER')
            ax.plot(tau, bpcer, color='blue', linewidth=1.8, label='BPCER')
            ax.axvline(eer_t, color='green', linestyle='--', alpha=0.7)
            ax.axhline(eer,   color='green', linestyle='--', alpha=0.7)
            ax.plot(eer_t, eer, 'go', markersize=8,
                    label=f'EER={eer:.1f}%\nτ={eer_t:.3f}')

            ax.set_title(f'{method} — {ATK_LABEL[attack]}\n'
                         f'5-Fold CV EER: {mu:.1f}±{std:.1f}%', fontsize=9)
            ax.set_xlabel('Threshold (τ)', fontsize=8)
            ax.set_ylabel('Error Rate (%)', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 105)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_save(fig, 'eer_cascade.png')


# ─── 2. ROC cascade (single plot, all 6 curves) ──────────────────────────────
def plot_roc_cascade(flid, base):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title('ROC Curves — Baseline vs. FLiD (5-Fold CV)', fontsize=13)

    for attack in ATTACKS:
        color = COLORS[attack]
        label = ATK_LABEL[attack]

        # Baseline — dashed
        y_b, s_b = pool(base, attack)
        if len(y_b):
            fpr, tpr, _ = roc_curve(y_b, s_b, pos_label=0)
            auc_b = sk_auc(fpr, tpr)
            # per-fold std
            fold_aucs = [base[f'{attack}_attack']['folds'][i]['auc']
                         for i in range(len(base[f'{attack}_attack']['folds']))]
            std_b = np.std(fold_aucs)
            ax.plot(fpr, tpr, color=color, linestyle='--', linewidth=1.8,
                    label=f'Baseline – {label} (AUC={auc_b:.3f}±{std_b:.3f})',
                    alpha=0.75)

        # FLiD — solid
        y_f, s_f = pool(flid, attack)
        if len(y_f):
            fpr, tpr, _ = roc_curve(y_f, s_f, pos_label=0)
            auc_f = sk_auc(fpr, tpr)
            fold_aucs = [flid[attack]['folds'][i]['auc']
                         for i in range(len(flid[attack]['folds']))]
            std_f = np.std(fold_aucs)
            ax.plot(fpr, tpr, color=color, linestyle='-', linewidth=2.2,
                    label=f'FLiD – {label} (AUC={auc_f:.3f}±{std_f:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate (APCER)', fontsize=11)
    ax.set_ylabel('True Positive Rate (1 – BPCER)', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_save(fig, 'roc_cascade.png')


# ─── 3. Score distribution cascade (2×3) ─────────────────────────────────────
def plot_score_cascade(flid, base):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Score Distributions — FLiD vs. Baseline', fontsize=14, fontweight='bold')

    rows = [('FLiD', flid), ('Baseline', base)]
    for row_i, (method, results) in enumerate(rows):
        for col_i, attack in enumerate(ATTACKS):
            ax = axes[row_i][col_i]
            y, s = pool(results, attack)
            if len(y) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            real_s = s[y == 0]
            fake_s = s[y == 1]

            # EER threshold
            tau  = np.linspace(0, 1, 5000)
            ap   = np.array([np.mean(fake_s > t) for t in tau])
            bp   = np.array([np.mean(real_s <= t) for t in tau])
            idx  = np.argmin(np.abs(ap - bp))
            eer_t = tau[idx]

            bins = np.linspace(0, 1, 30)
            ax.hist(real_s, bins=bins, density=True, alpha=0.6,
                    color='#4CAF50', label=f'Bona Fide (n={len(real_s)})')
            ax.hist(fake_s, bins=bins, density=True, alpha=0.6,
                    color='#FF5722', label=f'Attack (n={len(fake_s)})')
            ax.axvline(eer_t, color='black', linestyle='--',
                       label=f'EER τ={eer_t:.3f}')

            ax.set_title(f'{method} — {ATK_LABEL[attack]}', fontsize=9)
            ax.set_xlabel('Bona Fide Score P(Real)', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 1)

    plt.tight_layout()
    fig_save(fig, 'score_cascade.png')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading results...")
    flid, base = load()

    print("\nGenerating eer_cascade.png ...")
    plot_eer_cascade(flid, base)

    print("Generating roc_cascade.png ...")
    plot_roc_cascade(flid, base)

    print("Generating score_cascade.png ...")
    plot_score_cascade(flid, base)

    print("\nDone.")


if __name__ == '__main__':
    main()
