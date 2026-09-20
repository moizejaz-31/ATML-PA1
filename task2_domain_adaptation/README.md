# Task 2: Unsupervised Domain Adaptation

## Overview
PACS: Photo, Art Painting, Cartoon (source) → Sketch (target with no labels).

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_source_only.ipynb` | Source-only ERM baseline (reused as Task 3 ERM) |
| 2 | `2_dan.ipynb` | DAN — MMD alignment (λ=1) |
| 3 | `3_dann.ipynb` | DANN — adversarial alignment with GRL |
| 4 | `4_cdan.ipynb` | CDAN — class-conditional adversarial |
| 5 | `5_evaluation.ipynb` | Final target evaluation + domain separability |
| 6 | `6_controlled_study.ipynb` | Alignment strength study |

## Source Modules
- `backbone.py` — ResNet-18 with feature extraction
- `discriminator.py` — Domain discriminator for DANN/CDAN
- `train.py` — Unified training loop
- `evaluate.py` — Evaluation and domain separability
