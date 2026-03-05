#!/usr/bin/env python3
"""
Generate backbone ablation comparison plots from 5-fold CV results.

Produces:
  1. Grouped bar chart: AUC by backbone × attack
  2. (Optional) Efficiency scatter: FLOPs vs mean AUC

Usage:
    python -m evaluation.plot_backbone_ablation
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).resolve().parent.parent / 'results' / 'ablation' / 'backbone_kfold_results.json'
OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / 'ablation'


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    backbones = list(data.keys())
    attacks = ['Face', 'Text', 'Both']
    n_bb = len(backbones)
    n_att = len(attacks)

    # Extract AUC means and stds
    auc_means = np.zeros((n_bb, n_att))
    auc_stds = np.zeros((n_bb, n_att))
    for i, bb in enumerate(backbones):
        for j, att in enumerate(attacks):
            key = f'{att}_attack'
            auc_means[i, j] = data[bb][key]['auc']['mean']
            auc_stds[i, j] = data[bb][key]['auc']['std']

    # ── Plot 1: Grouped bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_att)
    width = 0.22
    colors = ['#2196F3', '#FF9800', '#4CAF50']  # blue, orange, green

    for i, bb in enumerate(backbones):
        params_M = data[bb]['params_backbone'] / 1e6
        flops_M = data[bb]['flops_M']
        label = f'{bb}\n({params_M:.1f}M params, {flops_M:.0f}M FLOPs)'
        bars = ax.bar(x + (i - 1) * width, auc_means[i], width,
                      yerr=auc_stds[i], capsize=3, label=label,
                      color=colors[i], edgecolor='black', linewidth=0.5,
                      alpha=0.85)
        # Value labels on bars
        for bar, mean in zip(bars, auc_means[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    ax.set_xlabel('Attack Scenario', fontsize=12)
    ax.set_ylabel('AUC (5-Fold CV)', fontsize=12)
    ax.set_title('Backbone Ablation: 5-Fold Cross-Validation AUC', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a} Attack' for a in attacks], fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = OUT_DIR / 'backbone_ablation_auc.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.png')}")

    # ── Plot 2: Efficiency scatter (FLOPs vs mean AUC) ──
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    markers = ['o', 's', '^']
    for i, bb in enumerate(backbones):
        flops = data[bb]['flops_M']
        mean_auc = np.mean(auc_means[i])
        ax2.scatter(flops, mean_auc, s=120, marker=markers[i],
                    color=colors[i], edgecolor='black', linewidth=0.8,
                    zorder=5, label=bb)
        ax2.annotate(bb, (flops, mean_auc), fontsize=9,
                     xytext=(8, 8), textcoords='offset points')

    ax2.set_xlabel('FLOPs (M)', fontsize=12)
    ax2.set_ylabel('Mean AUC (across 3 attacks)', fontsize=12)
    ax2.set_title('Efficiency vs. Performance', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out2 = OUT_DIR / 'backbone_efficiency_scatter.pdf'
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    fig2.savefig(out2.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {out2}")
    print(f"Saved: {out2.with_suffix('.png')}")

    # ── Print LaTeX table ──
    print("\n% LaTeX table for paper:")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Backbone ablation study (5-fold CV). All backbones are frozen; only the lightweight MLP head is trained.}")
    print(r"\label{tab:backbone_ablation}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{l r r c c c}")
    print(r"\toprule")
    print(r"Backbone & Params & FLOPs & Face AUC & Text AUC & Both AUC \\")
    print(r"\midrule")
    for i, bb in enumerate(backbones):
        params = data[bb]['params_backbone'] / 1e6
        flops = data[bb]['flops_M']
        vals = []
        for att in attacks:
            m = data[bb][f'{att}_attack']['auc']['mean']
            s = data[bb][f'{att}_attack']['auc']['std']
            vals.append(f"{m:.3f}$\\pm${s:.3f}")
        # Bold the best per attack
        print(f"{bb} & {params:.1f}M & {flops:.0f}M & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


if __name__ == '__main__':
    main()
