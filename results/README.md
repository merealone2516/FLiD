# Pre-computed Results

All results below come from the **leakage-free** protocol: folds are split by
document identity (`StratifiedGroupKFold` on `face_id`), and early stopping /
LR scheduling use an inner identity-disjoint validation split of the training
fold — the held-out test fold is never used for model selection.

## Directory structure

```
results/
├── kfold/
│   ├── flid_kfold_face_yolo.json           # FLiD Face  — YOLO11 crops (deployable)
│   ├── flid_kfold_text_yolo.json           # FLiD Text  — YOLO11 crops (deployable)
│   ├── flid_kfold_both_yolo_cascade.json   # FLiD Both  — YOLO11 cross-attack cascade (per-field min)
│   ├── flid_kfold_text_coarse.json         # FLiD Text  — coarse single-box crop
│   ├── flid_kfold_both_coarse_cascade.json # FLiD Both  — coarse cross-attack cascade
│   ├── both_perfield_mobilenet_gt.json     # Per-field-min Both — GT crops
│   ├── both_perfield_mobilenet_yolo.json   # Per-field-min Both — YOLO11 crops
│   ├── flid_kfold_results.json             # FLiD GT crops — Face / Text / Both
│   └── baseline_kfold_results.json         # González & Tapia baseline (from scratch)
├── efficientnet_b0/                         # EfficientNet-B0 ablation JSONs (Table 2)
├── resnet50/                                # ResNet50 ablation JSONs (Table 2)
├── roc_yolo11.{png,pdf}                     # ROC curves (FLiD vs baseline)
├── score_yolo11.{png,pdf}                   # Score distributions
└── acc_prec_comparison.{png,pdf}            # vs TruFor / MMFusion / UniVAD
```

## Detection performance — deployable YOLO11 pipeline

| Attack | AUC            | EER (%)       | BPCER@10 (%) | Accuracy (%) |
|--------|----------------|---------------|--------------|--------------|
| Face   | 0.834 ± 0.077  | 24.68         | 43.00        | 75.2         |
| Text   | 0.926 ± 0.003  | 18.46         | 27.03        | 81.5         |
| Both   | 0.837 ± 0.055  | 23.20         | 46.85        | 76.8         |

## Baseline (González & Tapia — MobileNetV2 from scratch, full image)

| Attack | AUC           | EER (%) |
|--------|---------------|---------|
| Face   | 0.469 ± 0.107 | 53.36   |
| Text   | 0.532 ± 0.032 | 47.53   |
| Both   | 0.507 ± 0.063 | 51.18   |

## Backbone & crop-strategy ablation (AUC / EER%)

| Backbone          | Crop   | Face          | Text          | Both          |
|-------------------|--------|---------------|---------------|---------------|
| MobileNetV3-Small | GT     | 0.841 / 22.36 | 0.998 / 1.65  | 0.890 / 19.89 |
| MobileNetV3-Small | YOLO11 | 0.834 / 24.68 | 0.926 / 18.46 | 0.837 / 23.20 |
| MobileNetV3-Small | Coarse | 0.841 / 22.36 | 0.627 / 41.93 | 0.788 / 27.00 |
| EfficientNet-B0   | GT     | 0.839 / 22.86 | 1.000 / 1.09  | 0.744 / 33.18 |
| EfficientNet-B0   | YOLO11 | 0.834 / 26.77 | 0.928 / 18.34 | 0.800 / 26.08 |
| EfficientNet-B0   | Coarse | 0.839 / 22.86 | 0.695 / 37.76 | 0.723 / 35.06 |
| ResNet50          | GT     | 0.788 / 30.50 | 0.998 / 1.38  | 0.746 / 31.32 |
| ResNet50          | YOLO11 | 0.761 / 35.91 | 0.927 / 18.37 | 0.671 / 38.39 |
| ResNet50          | Coarse | 0.788 / 30.50 | 0.639 / 40.98 | 0.619 / 41.95 |

GT = manually annotated crops (upper bound); YOLO11 = automatic deployable
crops (main results); Coarse = single broad text-region crop (lower bound).
Coarse and GT share the same face crop.

## Reproducing

```bash
bash scripts/run_all.sh
```

Or run individual stages — see the main [README](../README.md).
