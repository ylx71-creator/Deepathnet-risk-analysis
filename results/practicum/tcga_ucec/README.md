# TCGA-UCEC DeePathNet experiments

This directory contains small, shareable summaries only. Patient-level data,
model checkpoints, fold assignments, and complete training outputs remain local
and are excluded from Git.

## FIGO stage prediction

The four-class target contains Stage I, II, III, and IV. All development
comparisons used the same 300 training and 76 validation patients; the 94-patient
test set was not reopened during tuning.

| Development setting | Validation Macro-F1 |
| --- | ---: |
| Original DeePathNet: dim 512, 16 heads, depth 2, lr 1e-5 | 0.198 |
| Train-only gene Z-score | 0.267 |
| Learning rate 1e-4 | 0.311 |
| Inverse-frequency class weighting | 0.279 |
| Compact DeePathNet: dim 256, 8 heads, depth 2, lr 1e-4; 3-seed mean (SD) | 0.304 (0.010) |
| Compact depth 1; 2-seed mean (SD) | 0.296 (0.009) |

The task remains strongly affected by class imbalance, especially for Stage II
and Stage IV. These development results do not establish final generalization.

## Molecular subtype prediction

The target contains CN_HIGH, CN_LOW, MSI, and POLE. Results below are paired
nested five-fold cross-validation estimates on 416 development patients using
3,012 RNA features. The 91-patient holdout was not read or evaluated.

| Model | Macro-F1 mean (SD) | Accuracy mean (SD) | Balanced accuracy mean (SD) |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.122 (0.002) | 0.322 (0.005) | 0.250 (0.000) |
| Logistic regression | **0.774 (0.068)** | **0.839 (0.043)** | **0.768 (0.065)** |
| MLP | 0.724 (0.044) | 0.803 (0.036) | 0.714 (0.038) |
| Compact DeePathNet | 0.675 (0.069) | 0.762 (0.048) | 0.674 (0.065) |

Compact DeePathNet used dim 256, 8 attention heads, depth 2, pathway dropout
0.5, Adam learning rate 1e-4, weight decay 1e-5, and batch size 64. It was lower
than logistic regression and the MLP in all five paired folds. These results do
not currently support attributing an advantage to pathway structure or the
Transformer; matched ablations are still required.

## Reproduction entry points

- `scripts/practicum/prepare_tcga_ucec_deepathnet_inputs.py`
- `scripts/practicum/train_tcga_ucec_stage.py`
- `scripts/practicum/prepare_tcga_ucec_subtypes.py`
- `scripts/practicum/screen_tcga_ucec_subtypes.py`
- `scripts/practicum/cv_tcga_ucec_subtypes.py`
