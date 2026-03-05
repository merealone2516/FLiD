#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
# FLiD — Reproduce All Experiments
#
# Before running, ensure:
#   1. configs/paths.py has BASE set to your data directory
#   2. Dependencies are installed: pip install -r requirements.txt
#   3. Pre-extracted embeddings exist (see README.md)
# ════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

echo "══════════════════════════════════════════════════════════"
echo "  FLiD — Full Experiment Pipeline"
echo "══════════════════════════════════════════════════════════"

# ── Step 1: FLiD 5-fold CV ──
echo -e "\n[1/6] FLiD 5-fold cross-validation..."
python -m flid.train_kfold --attack all

# ── Step 2: Baseline 5-fold CV ──
echo -e "\n[2/6] Baseline 5-fold cross-validation..."
python -m baseline.train_kfold --attack all

# ── Step 3: Efficiency analysis ──
echo -e "\n[3/6] Efficiency analysis..."
python -m evaluation.efficiency

# ── Step 4: Fair YOLO vs Coord comparison ──
echo -e "\n[4/6] Fair YOLO vs coordinate crop comparison..."
python -m evaluation.fair_comparison

# ── Step 5: Full ablation study ──
echo -e "\n[5/6] Ablation study..."
python -m evaluation.ablation

# ── Step 6: Baseline single-run training (optional) ──
echo -e "\n[6/6] Baseline single-run training..."
python -m baseline.train --attack_type all

echo -e "\n══════════════════════════════════════════════════════════"
echo "  All experiments completed."
echo "  Results saved to outputs/"
echo "══════════════════════════════════════════════════════════"
