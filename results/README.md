# Pre-computed Results

This directory contains all pre-computed experimental results used in the
paper. These allow verification of reported numbers without re-running
experiments.

## Directory Structure

```
results/
├── kfold/                          # 5-Fold Stratified Cross-Validation
│   ├── flid_kfold_results.json     # FLiD (ours) — per-fold + bootstrap CIs
│   └── baseline_kfold_results.json # Gonzalez & Tapia baseline — per-fold
├── ablation/                       # Ablation & Efficiency Studies
│   ├── ablation_results.json       # All 5 ablation experiments
│   ├── efficiency_results.json     # Params, FLOPs, latency measurements
│   └── fair_yolo_comparison.json   # YOLO vs coordinate-crop comparison
└── plots/                          # Paper Figures (PNG)
    ├── roc_*.png                   # ROC curves (per-attack + combined)
    ├── eer_*.png                   # EER distribution plots
    ├── scores_*.png                # Score distribution histograms
    ├── det_*.png                   # DET curves
    └── bar_*.png                   # Bar charts (AUC, EER, BPCER@10)
```

## K-Fold Cross-Validation Results

### FLiD (Ours) — 5-Fold Stratified CV with Bootstrap CIs

| Attack | AUC | EER (%) | BPCER@10 | BPCER@20 | BPCER@50 |
|--------|-----|---------|----------|----------|----------|
| Face | 0.880 ± 0.054 | 18.05 ± 7.04 | 23.0 ± 12.1 | 45.0 ± 29.7 | 45.0 ± 29.7 |
| Text | 0.954 ± 0.016 | 11.61 ± 1.21 | 11.9 ± 2.2 | 21.2 ± 5.9 | 37.2 ± 16.9 |
| Both | 0.923 ± 0.036 | 15.16 ± 6.45 | 20.4 ± 11.1 | 35.1 ± 15.3 | 51.1 ± 22.3 |

### Baseline (Gonzalez & Tapia, MobileNetV2 from scratch) — 5-Fold CV

| Attack | AUC | EER (%) | BPCER@10 | BPCER@20 |
|--------|-----|---------|----------|----------|
| Face | 0.547 ± 0.086 | 47.45 ± 7.55 | 86.0 ± 17.4 | 93.0 ± 11.7 |
| Text | 0.541 ± 0.042 | 47.26 ± 3.52 | 87.5 ± 3.1 | 93.3 ± 3.8 |
| Both | 0.501 ± 0.073 | 50.50 ± 7.05 | 82.6 ± 10.4 | 90.3 ± 8.0 |

## Ablation Studies

Results in `ablation/ablation_results.json` cover five ablations:

1. **Whole-image vs ROI** — Full document vs face/text crops
2. **YOLO vs coordinate crops** — Learned detector vs fixed coordinates (5-run avg)
3. **Frozen vs fine-tuned backbone** — Frozen MobileNetV3 vs end-to-end
4. **Fusion vs single-field** — Concatenated embeddings vs individual fields
5. **Backbone choice** — MobileNetV3-Small vs EfficientNet-B0 vs ResNet-18

## Efficiency Comparison

| Method | Trainable Params | FLOPs | Latency (CPU) |
|--------|-----------------|-------|---------------|
| FLiD (Face) | 191K | 119M | 16.8 ms |
| Baseline | 2.55M | 2,503M | 37.0 ms |
| **Ratio** | **13× fewer** | **21× fewer** | **2.2× faster** |

## Plots

All 18 plots from the paper are in `plots/`:

| Plot | Filename | Description |
|------|----------|-------------|
| ROC (per-attack) | `roc_face_attack.png`, `roc_text_attack.png`, `roc_both_attacks.png` | ROC curves comparing FLiD vs baseline vs detectors |
| ROC (combined) | `roc_all_combined.png` | 3-panel combined ROC |
| EER (per-attack) | `eer_face_attack.png`, `eer_text_attack.png`, `eer_both_attacks.png` | EER distribution across k-fold |
| EER (combined) | `eer_combined_2x3.png` | 2×3 combined EER panel |
| Score dist. | `scores_face_attack.png`, `scores_text_attack.png`, `scores_both_attacks.png` | Score distributions per class |
| Score (combined) | `scores_combined_2x3.png` | 2×3 combined score panel |
| DET curves | `det_face_attack.png`, `det_text_attack.png`, `det_both_attacks.png` | Detection Error Tradeoff |
| Bar charts | `bar_auc.png`, `bar_eer.png`, `bar_bpcer10.png` | Summary comparison bars |

## Reproducing

To regenerate these results from scratch:
```bash
bash scripts/run_all.sh
```
Or run individual experiments — see the main [README](../README.md).
