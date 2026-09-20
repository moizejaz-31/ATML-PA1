# Task 3: Domain Generalization

## Overview
PACS: Photo, Art Painting, Cartoon (source) → Sketch (UNSEEN target).
ERM baseline = Task 2 source-only checkpoint (NOT retrained).

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_erm_baseline.ipynb` | Load Task 2 source-only checkpoint |
| 2 | `2_dan_dg.ipynb` | DAN-DG — pairwise source MMD (λ=1) |
| 3 | `3_sam.ipynb` | SAM — sharpness-aware minimization (ρ=0.05) |
| 4 | `4_evaluation.ipynb` | Sketch evaluation + diagnostics |
| 5 | `5_controlled_study.ipynb` | Alignment/stability strength study |

## Source Modules
- `dan_dg.py` — Pairwise source-domain MMD
- `sam.py` — SAM optimizer wrapper
- `evaluate.py` — Source separability, sharpness proxy
