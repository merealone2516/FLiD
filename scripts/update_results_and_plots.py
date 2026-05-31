#!/usr/bin/env python3
"""
Copy new leakage-free results into results/kfold/ and regenerate all plots.

Run AFTER:
    python -m flid.train_kfold --attack all
    python -m baseline.train_kfold --attack all

Usage:
    python scripts/update_results_and_plots.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO     = Path(__file__).resolve().parent.parent
OUT_DIR  = REPO / 'outputs' / 'kfold_results'
RES_DIR  = REPO / 'results'
KFOLD_DIR = RES_DIR / 'kfold'
PLOT_DIR  = RES_DIR / 'plots'

ATTACKS  = ['Face', 'Text', 'Both']
COLORS   = {'Face': '#2196F3', 'Text': '#4CAF50', 'Both': '#FF5722'}


# ─── Load results ────────────────────────────────────────────────────────────
def load_new_flid():
    p = OUT_DIR / 'kfold_results_docsplit.json'
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}\nRun: python -m flid.train_kfold --attack all")
    d = json.load(open(p))
    missing = [a for a in ATTACKS if a not in d]
    if missing:
        raise ValueError(
            f"Missing attacks {missing} in {p}\n"
            "Run: python -m flid.train_kfold --attack all")
    return d


def load_new_baseline():
    p = OUT_DIR / 'baseline_kfold_results.json'
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}\nRun: python -m baseline.train_kfold --attack all")
    return json.load(open(p))


# ─── Save clean results/kfold JSONs ──────────────────────────────────────────
def save_kfold_results(flid, baseline):
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)

    with open(KFOLD_DIR / 'flid_kfold_results.json', 'w') as f:
        json.dump(flid, f, indent=2)
    print(f"  Saved → results/kfold/flid_kfold_results.json")

    with open(KFOLD_DIR / 'baseline_kfold_results.json', 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f"  Saved → results/kfold/baseline_kfold_results.json")


# ─── Plot helpers ────────────────────────────────────────────────────────────
def get_summary(results, attack):
    """Return summary dict for a given attack key (handles Face/Text/Both or Face_attack etc)."""
    # Try exact key first, then with _attack suffix
    for key in [attack, f'{attack}_attack']:
        if key in results:
            return results[key]['summary']
    return None


def fig_save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → results/plots/{name}")


# ─── 1. Bar charts ───────────────────────────────────────────────────────────
def plot_bar(flid, baseline, metric, ylabel, title, fname, higher_better=True):
    x = np.arange(len(ATTACKS))
    w = 0.35

    flid_vals    = [get_summary(flid,     a)[metric]['mean']  for a in ATTACKS]
    flid_errs    = [get_summary(flid,     a)[metric]['std']   for a in ATTACKS]
    base_vals    = [get_summary(baseline, a)[metric]['mean']  for a in ATTACKS]
    base_errs    = [get_summary(baseline, a)[metric]['std']   for a in ATTACKS]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, flid_vals,  w, yerr=flid_errs,  label='FLiD (ours)',
                   capsize=4, color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + w/2, base_vals,  w, yerr=base_errs,  label='Baseline',
                   capsize=4, color='#FF5722', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{a} Attack' for a in ATTACKS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    fig_save(fig, fname)


# ─── 2. Per-fold metric plot ─────────────────────────────────────────────────
def plot_fold_metrics(flid, baseline):
    n_folds = len(flid[ATTACKS[0]]['folds'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, attack in zip(axes, ATTACKS):
        folds_x = range(1, n_folds + 1)

        flid_auc  = [f['auc']  for f in flid[attack]['folds']]
        base_key  = f'{attack}_attack'
        base_auc  = [f['auc']  for f in baseline[base_key]['folds']]

        ax.plot(folds_x, flid_auc, 'o-', color='#2196F3', label='FLiD', linewidth=2)
        ax.plot(folds_x, base_auc, 's--', color='#FF5722', label='Baseline', linewidth=2)
        ax.axhline(np.mean(flid_auc),  color='#2196F3', linestyle=':', alpha=0.6)
        ax.axhline(np.mean(base_auc),  color='#FF5722', linestyle=':', alpha=0.6)

        ax.set_title(f'{attack} Attack — AUC per fold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('AUC')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_save(fig, 'auc_per_fold.png')


# ─── 3. Summary table plot ───────────────────────────────────────────────────
def plot_summary_table(flid, baseline):
    metrics = ['auc', 'eer', 'accuracy', 'f1', 'bpcer10', 'bpcer20']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    labels = {
        'auc': 'AUC ↑', 'eer': 'EER % ↓', 'accuracy': 'Accuracy % ↑',
        'f1': 'F1 % ↑', 'bpcer10': 'BPCER@10 % ↓', 'bpcer20': 'BPCER@20 % ↓',
    }

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(ATTACKS))
        w = 0.35
        flid_v = [get_summary(flid,     a)[metric]['mean'] for a in ATTACKS]
        flid_e = [get_summary(flid,     a)[metric]['std']  for a in ATTACKS]
        base_v = [get_summary(baseline, a)[metric]['mean'] for a in ATTACKS]
        base_e = [get_summary(baseline, a)[metric]['std']  for a in ATTACKS]

        ax.bar(x - w/2, flid_v, w, yerr=flid_e, label='FLiD',     capsize=3,
               color='#2196F3', alpha=0.85)
        ax.bar(x + w/2, base_v, w, yerr=base_e, label='Baseline', capsize=3,
               color='#FF5722', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ATTACKS, rotation=15)
        ax.set_title(labels[metric])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('FLiD vs Baseline — All Metrics (5-fold CV, Document-Level Split)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_save(fig, 'all_metrics_summary.png')


# ─── 4. BPCER comparison ─────────────────────────────────────────────────────
def plot_bpcer(flid, baseline):
    bpcer_keys = ['bpcer10', 'bpcer20', 'bpcer50', 'bpcer100']
    x_labels   = ['10%', '20%', '50%', '100%']
    x = np.arange(len(bpcer_keys))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, attack in zip(axes, ATTACKS):
        fs = get_summary(flid,     attack)
        bs = get_summary(baseline, attack)

        flid_v = [fs[k]['mean'] for k in bpcer_keys]
        base_v = [bs[k]['mean'] for k in bpcer_keys]
        flid_e = [fs[k]['std']  for k in bpcer_keys]
        base_e = [bs[k]['std']  for k in bpcer_keys]

        w = 0.35
        ax.bar(x - w/2, flid_v, w, yerr=flid_e, label='FLiD',     capsize=3,
               color='#2196F3', alpha=0.85)
        ax.bar(x + w/2, base_v, w, yerr=base_e, label='Baseline', capsize=3,
               color='#FF5722', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f'BPCER@{l}' for l in x_labels])
        ax.set_ylabel('BPCER %')
        ax.set_title(f'{attack} Attack')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('BPCER at Operating Points — FLiD vs Baseline', fontsize=12)
    plt.tight_layout()
    fig_save(fig, 'bpcer_comparison.png')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading new results...")
    try:
        flid     = load_new_flid()
        baseline = load_new_baseline()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\nSaving updated results/kfold/ JSONs...")
    save_kfold_results(flid, baseline)

    print("\nGenerating plots...")
    plot_bar(flid, baseline, 'auc',     'AUC',         'AUC — FLiD vs Baseline',             'bar_auc.png')
    plot_bar(flid, baseline, 'eer',     'EER %',       'EER % — FLiD vs Baseline',           'bar_eer.png')
    plot_bar(flid, baseline, 'bpcer10', 'BPCER@10 %',  'BPCER@10 % — FLiD vs Baseline',      'bar_bpcer10.png')
    plot_fold_metrics(flid, baseline)
    plot_summary_table(flid, baseline)
    plot_bpcer(flid, baseline)

    print("\nDone. Summary:")
    print(f"{'Attack':<8} {'FLiD AUC':>10} {'Base AUC':>10} {'FLiD EER':>10} {'Base EER':>10}")
    print("-" * 52)
    for a in ATTACKS:
        fs = get_summary(flid,     a)
        bs = get_summary(baseline, a)
        print(f"{a:<8} {fs['auc']['mean']:>10.4f} {bs['auc']['mean']:>10.4f} "
              f"{fs['eer']['mean']:>10.2f}% {bs['eer']['mean']:>10.2f}%")


if __name__ == '__main__':
    main()
