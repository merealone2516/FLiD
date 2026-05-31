#!/usr/bin/env python3
"""
Generate detailed per-attack plots from saved kfold results:
  - Score distributions (Real vs Fake histograms)
  - EER threshold curves (APCER / BPCER vs threshold)
  - ROC curves (per fold + mean)
  - DET curves

Requires per-fold y_true and bf_scores saved by train_kfold.py.

Usage:
    python scripts/generate_detailed_plots.py
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
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

REPO     = Path(__file__).resolve().parent.parent
OUT_DIR  = REPO / 'outputs' / 'kfold_results'
PLOT_DIR = REPO / 'results' / 'plots'
ATTACKS  = ['Face', 'Text', 'Both']

FLID_COLOR = '#2196F3'
BASE_COLOR = '#FF5722'
REAL_COLOR = '#4CAF50'
FAKE_COLOR = '#FF5722'


# ─── Load ─────────────────────────────────────────────────────────────────────
def load(name):
    p = OUT_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return json.load(open(p))


def pool_scores(results, attack):
    """Pool y_true and bf_scores across all folds for an attack.
    FLiD stores scores in leakage_report; baseline stores them in folds."""
    key = attack if attack in results else f'{attack}_attack'
    all_y, all_s = [], []
    # Try leakage_report first (FLiD), then folds (baseline)
    for source in ['leakage_report', 'folds']:
        entries = results[key].get(source, [])
        for entry in entries:
            if 'y_true' in entry and 'bf_scores' in entry:
                all_y.extend(entry['y_true'])
                all_s.extend(entry['bf_scores'])
        if all_y:
            break
    return np.array(all_y), np.array(all_s)


def fig_save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → results/plots/{name}")


# ─── 1. Score distributions ───────────────────────────────────────────────────
def plot_scores(flid, attack, fname):
    y, s = pool_scores(flid, attack)
    if len(y) == 0:
        print(f"  Skipping {fname} — no per-fold scores saved")
        return

    real_s = s[y == 0]
    fake_s = s[y == 1]

    # EER threshold
    tau = np.linspace(0, 1, 5000)
    apcer = np.array([np.mean(fake_s > t) for t in tau])
    bpcer = np.array([np.mean(real_s <= t) for t in tau])
    eer_idx = np.argmin(np.abs(apcer - bpcer))
    eer_tau = tau[eer_idx]
    eer_val = (apcer[eer_idx] + bpcer[eer_idx]) / 2 * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(real_s, bins=bins, density=True, alpha=0.6, color=REAL_COLOR,
            label=f'Bona Fide (n={len(real_s)})')
    ax.hist(fake_s, bins=bins, density=True, alpha=0.6, color=FAKE_COLOR,
            label=f'Attack (n={len(fake_s)})')
    ax.axvline(eer_tau, color='black', linestyle='--',
               label=f'EER τ={eer_tau:.3f}')
    ax.set_xlabel('Bona Fide Score P(Real)')
    ax.set_ylabel('Density')
    ax.set_title(f'Score Distributions — {attack} Attack\nFLiD (MobileNetV3+MLP)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig_save(fig, fname)


def plot_scores_comparison(flid, baseline, attack, fname):
    """Side-by-side score distributions: baseline (left) vs FLiD (right)."""
    y_f, s_f = pool_scores(flid,     attack)
    y_b, s_b = pool_scores(baseline, attack)

    if len(y_f) == 0 or len(y_b) == 0:
        print(f"  Skipping {fname} — no per-fold scores saved")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Score Distributions — {attack} Attacks', fontweight='bold')

    for ax, y, s, title in [
        (axes[0], y_b, s_b, 'Gonzalez & Tapia (Baseline)'),
        (axes[1], y_f, s_f, 'Ours (MobileNetV3+MLP)'),
    ]:
        real_s = s[y == 0]
        fake_s = s[y == 1]
        tau = np.linspace(0, 1, 5000)
        apcer = np.array([np.mean(fake_s > t) for t in tau])
        bpcer = np.array([np.mean(real_s <= t) for t in tau])
        eer_idx = np.argmin(np.abs(apcer - bpcer))
        eer_tau = tau[eer_idx]

        bins = np.linspace(0, 1, 30)
        ax.hist(real_s, bins=bins, density=True, alpha=0.6, color=REAL_COLOR,
                label=f'Bona Fide (n={len(real_s)})')
        ax.hist(fake_s, bins=bins, density=True, alpha=0.6, color=FAKE_COLOR,
                label=f'Attack (n={len(fake_s)})')
        ax.axvline(eer_tau, color='black', linestyle='--',
                   label=f'EER τ={eer_tau:.3f}')
        ax.set_xlabel('Bona Fide Score P(Real)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig_save(fig, fname)


# ─── 2. EER curves ────────────────────────────────────────────────────────────
def plot_eer(flid, baseline, attack, fname):
    y_f, s_f = pool_scores(flid,     attack)
    y_b, s_b = pool_scores(baseline, attack)

    if len(y_f) == 0 or len(y_b) == 0:
        print(f"  Skipping {fname} — no per-fold scores saved")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'EER Analysis — {attack} Attacks', fontweight='bold')

    for ax, y, s, title in [
        (axes[0], y_b, s_b, 'Gonzalez & Tapia (Baseline)'),
        (axes[1], y_f, s_f, 'Ours (MobileNetV3+MLP)'),
    ]:
        real_s = s[y == 0]
        fake_s = s[y == 1]
        tau = np.linspace(0, 1, 5000)
        apcer = np.array([np.mean(fake_s > t) for t in tau]) * 100
        bpcer = np.array([np.mean(real_s <= t) for t in tau]) * 100
        eer_idx = np.argmin(np.abs(apcer - bpcer))
        eer_val = (apcer[eer_idx] + bpcer[eer_idx]) / 2
        eer_tau = tau[eer_idx]

        ax.plot(tau, apcer, color='red',  linewidth=2, label='APCER')
        ax.plot(tau, bpcer, color='blue', linewidth=2, label='BPCER')
        ax.axvline(eer_tau, color='green', linestyle='--', alpha=0.7)
        ax.axhline(eer_val, color='green', linestyle='--', alpha=0.7)
        ax.plot(eer_tau, eer_val, 'go', markersize=10,
                label=f'EER={eer_val:.1f}%\nτ={eer_tau:.3f}')
        ax.set_xlabel('Threshold (τ)')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    fig_save(fig, fname)


# ─── 3. ROC curves ────────────────────────────────────────────────────────────
def plot_roc(flid, baseline, attack, fname):
    y_f, s_f = pool_scores(flid,     attack)
    y_b, s_b = pool_scores(baseline, attack)

    if len(y_f) == 0 or len(y_b) == 0:
        print(f"  Skipping {fname} — no per-fold scores saved")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for y, s, label, color in [
        (y_b, s_b, 'Baseline',    BASE_COLOR),
        (y_f, s_f, 'FLiD (ours)', FLID_COLOR),
    ]:
        fpr, tpr, _ = roc_curve(y, s, pos_label=0)
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{label} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate (APCER)')
    ax.set_ylabel('True Positive Rate (1-BPCER)')
    ax.set_title(f'ROC Curve — {attack} Attack')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig_save(fig, fname)


# ─── 4. DET curves ────────────────────────────────────────────────────────────
def plot_det(flid, baseline, attack, fname):
    from scipy.special import ndtri

    y_f, s_f = pool_scores(flid,     attack)
    y_b, s_b = pool_scores(baseline, attack)

    if len(y_f) == 0 or len(y_b) == 0:
        print(f"  Skipping {fname} — no per-fold scores saved")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    tick_labels = ['0.1', '1', '5', '10', '20', '30', '50']

    for y, s, label, color in [
        (y_b, s_b, 'Baseline',    BASE_COLOR),
        (y_f, s_f, 'FLiD (ours)', FLID_COLOR),
    ]:
        fpr, fnr, _ = roc_curve(y, s, pos_label=0)
        fnr = np.clip(fnr, 1e-4, 1 - 1e-4)
        fpr = np.clip(fpr, 1e-4, 1 - 1e-4)
        ax.plot(ndtri(fpr), ndtri(fnr), color=color, linewidth=2, label=label)

    ax.set_xlabel('APCER (%)')
    ax.set_ylabel('BPCER (%)')
    ax.set_title(f'DET Curve — {attack} Attack')
    tick_vals = [ndtri(t) for t in ticks]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.legend()
    ax.grid(alpha=0.3)
    fig_save(fig, fname)


# ─── 5. Combined ROC all attacks ─────────────────────────────────────────────
def plot_roc_combined(flid, baseline):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('ROC Curves — FLiD vs Baseline', fontweight='bold')

    colors = {'Face': '#2196F3', 'Text': '#4CAF50', 'Both': '#9C27B0'}

    for ax, attack in zip(axes, ATTACKS):
        y_f, s_f = pool_scores(flid,     attack)
        y_b, s_b = pool_scores(baseline, attack)

        if len(y_f) == 0 or len(y_b) == 0:
            ax.text(0.5, 0.5, 'No scores saved', ha='center', va='center')
            ax.set_title(f'{attack} Attack')
            continue

        for y, s, label, color, ls in [
            (y_b, s_b, 'Baseline',    BASE_COLOR,       '--'),
            (y_f, s_f, 'FLiD (ours)', colors[attack],   '-'),
        ]:
            fpr, tpr, _ = roc_curve(y, s, pos_label=0)
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, linestyle=ls,
                    label=f'{label} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('APCER')
        ax.set_ylabel('1-BPCER')
        ax.set_title(f'{attack} Attack')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    plt.tight_layout()
    fig_save(fig, 'roc_all_combined.png')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading results...")
    try:
        flid     = load('kfold_results_docsplit.json')
        baseline = load('baseline_kfold_results.json')
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Check scores are present
    sample_attack = ATTACKS[0]
    y_test, _ = pool_scores(flid, sample_attack)
    if len(y_test) == 0:
        print("\nERROR: No per-fold scores found in results.")
        print("Re-run training first:")
        print("  python -m flid.train_kfold --attack all")
        sys.exit(1)

    print("\nGenerating score distribution plots...")
    for attack in ATTACKS:
        plot_scores_comparison(flid, baseline, attack,
                               f'scores_{attack.lower()}_attack.png')

    print("\nGenerating EER analysis plots...")
    for attack in ATTACKS:
        plot_eer(flid, baseline, attack,
                 f'eer_{attack.lower()}_attack.png')

    print("\nGenerating ROC curves...")
    for attack in ATTACKS:
        plot_roc(flid, baseline, attack,
                 f'roc_{attack.lower()}_attack.png')
    plot_roc_combined(flid, baseline)

    print("\nGenerating DET curves...")
    for attack in ATTACKS:
        plot_det(flid, baseline, attack,
                 f'det_{attack.lower()}_attack.png')

    print("\nAll plots saved to results/plots/")


if __name__ == '__main__':
    main()
