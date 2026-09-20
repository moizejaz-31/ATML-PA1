# Task 1: Inductive Biases & Feature Representations

## Overview
Compare ResNet-50, ViT-B/16, and CLIP under controlled image interventions.

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_setup_and_baselines.ipynb` | Dataset loading, linear head training, clean baseline |
| 2 | `2_color_bias.ipynb` | Grayscale + additional color intervention |
| 3 | `3_shape_vs_texture.ipynb` | AdaIN cue-conflict generation and shape bias |
| 4 | `4_spatial_sensitivity.ipynb` | Translation and patch shuffle experiments |
| 5 | `5_representation_analysis.ipynb` | Cosine stability + t-SNE/UMAP visualization |

## Source Modules
- `backbones.py` — Frozen ResNet-50, ViT-B/16, CLIP wrappers + linear heads
- `transforms.py` — Color, translation, patch shuffle transforms
- `cue_conflict.py` — AdaIN style transfer for shape vs texture
- `analysis.py` — Shape bias, cosine stability, representation viz
- `data.py` — STL-10 loader and evaluation subset selection
