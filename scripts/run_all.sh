
set -euo pipefail
cd "$(dirname "$0")/.."

echo "══════════════════════════════════════════════════════════"
echo "  FLiD — Full Experiment Pipeline"
echo "══════════════════════════════════════════════════════════"

# ── Step 1: Fine-tune the YOLO11 field detector ──
echo -e "\n[1/6] Preparing YOLO11 dataset..."
python scripts/prepare_yolo_dataset.py

echo -e "\n[2/6] Fine-tuning YOLO11 field detector..."
python scripts/finetune_yolo11.py

# ── Step 2: Extract field embeddings (GT / YOLO11 / coarse) ──
echo -e "\n[3/6] Extracting embeddings (GT, YOLO11, coarse)..."
python scripts/extract_embeddings.py --attack all
python scripts/extract_yolo_embeddings.py --attack all
python scripts/extract_coarse_embeddings.py

# ── Step 3: 5-fold CV (produces the reported numbers) ──
echo -e "\n[4/6] FLiD Face/Text 5-fold CV (all backbones)..."
python scripts/run_kfold.py

echo -e "\n[5/6] FLiD Both cross-attack cascade (per-field min) + backbone ablation..."
python scripts/run_both_cascade.py
python scripts/perfield_ablation.py
python scripts/run_baseline.py

# ── Step 4: Figures ──
echo -e "\n[6/6] Generating figures..."
python scripts/generate_yolo11_plots.py
python scripts/plot_acc_prec_comparison.py

echo -e "\n══════════════════════════════════════════════════════════"
echo "  All experiments completed.  Results saved to results/"
echo "══════════════════════════════════════════════════════════"
