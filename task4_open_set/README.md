# Task 4: Open-Set Recognition

## Overview
CIFAR-10 (known, 10 classes) vs CIFAR-100 near/far unknowns.

**Near**: bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel
**Far**: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_vanilla_baseline.ipynb` | Train 10-class closed-set ResNet-18 |
| 2 | `2_posthoc_scores.ipynb` | MSP, MLS, Energy, Mahalanobis on frozen vanilla |
| 3 | `3_gcsc.ipynb` | Train with RandAugment, evaluate with MLS |
| 4 | `4_proser.ipynb` | Classifier + data placeholders, manifold mixup |
| 5 | `5_evaluation.ipynb` | Full OSR evaluation, AUROC, failure analysis |

## Source Modules
- `resnet_cifar.py` — CIFAR-adapted ResNet-18 (3×3 conv, no maxpool)
- `data.py` — CIFAR-10/100 loaders and unknown class definitions
- `scores.py` — MSP, MLS, Energy, Mahalanobis scores
- `proser.py` — PROSER training with manifold mixup
- `evaluate.py` — AUROC, threshold calibration, failure analysis
