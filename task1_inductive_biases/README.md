# Task 1: Inductive Biases & Feature Representations

This directory contains the code, notebooks, and empirical evaluation for **Task 1: Inductive Biases and Feature Representations** on the STL-10 dataset, comparing **ResNet-50**, **ViT-B/16**, and **CLIP ViT-B-32** under controlled visual interventions.

---

## Overview & Scientific Objectives

Different neural network architectures trained under distinct learning objectives encode fundamentally different visual representations even when their clean classification accuracies appear identical. 

In this task, we perform controlled interventions across four visual axes:
1. **Color (Chromatic Perturbations):** Evaluates sensitivity to color distribution shifts using Grayscale conversion and Hue rotation ($+90^\circ$).
2. **Shape vs. Texture (Cue Conflicts):** Synthesizes cue-conflict images via Adaptive Instance Normalization (AdaIN) style transfer across 6 class pairs (240 conflict images), measuring shape bias $B_{\text{shape}}$ and prediction coverage $\text{Cov}$.
3. **Spatial Translation:** Evaluates translation equivariance/invariance under cardinal spatial shifts ($16\text{px}$ and $32\text{px}$ with reflection padding).
4. **Spatial Patch Shuffling:** Shuffles a $4\times 4$ grid of $56\times 56\text{px}$ image patches, testing reliance on local textures versus global compositional geometry.

Additionally, every intervention is read out at two levels:
* **Feature Representation Stability ($I_T$):** Mean cosine similarity between the frozen backbone representations of clean and transformed images.
* **Prediction Consistency ($C_T$):** Fraction of images whose class prediction remains identical to the clean prediction.

---

## Evaluated Models & Backbones

All backbones are frozen and feature extractors are evaluated with a linear classification probe trained on a stratified 80/20 split of the official STL-10 training set under global seed `6304`:

| Model | Backbone Architecture | Pretraining Objective / Weights | Extracted Feature Dimension |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | Deep Convolutional Network | Supervised ImageNet-1K (`IMAGENET1K_V2`) | 2048-d (Global Average Pool) |
| **ViT-B/16** | Vision Transformer ($16\times 16$ patch) | Supervised ImageNet-1K (`IMAGENET1K_V1`) | 768-d (`[CLS]` token) |
| **CLIP Head** | Vision Transformer (ViT-B/32) | Contrastive Language-Image Pretraining (`openai`) | 512-d (Normalized Image Embedding) + Linear Probe |
| **CLIP Zero-Shot** | Vision Transformer (ViT-B/32) | Zero-shot text prompt cosine similarity (`openai`) | 512-d (Class text prompt templates) |

---

## Notebooks & Execution Sequence

All notebooks reside in `notebooks/` and should be executed in order:

| # | Notebook | Description & Outputs |
| :--- | :--- | :--- |
| **1** | `1_setup_and_baselines.ipynb` | Loads STL-10, extracts frozen representations, trains linear heads (AdamW, cosine schedule, 20 epochs), evaluates clean baseline accuracy and classifier softmax confidence. Caches evaluation indices (`cache/eval_indices.npy`). |
| **2** | `2_color_bias.ipynb` | Applies Grayscale conversion and Hue rotation ($+90^\circ$ in HSV space). Evaluates accuracy drops, prediction consistency $C_T$, and feature stability $I_T$. |
| **3** | `3_shape_vs_texture.ipynb` | Implements AdaIN style transfer between content and style images across 6 paired classes (`airplane-bird`, `car-truck`, `cat-dog`, etc.). Evaluates shape decisions ($N_s$), texture decisions ($N_t$), unclassified cases ($N_o$), shape bias, and coverage. |
| **4** | `4_spatial_sensitivity.ipynb` | Implements spatial translations (16px, 32px cardinal directions with reflection padding) and $4\times 4$ patch shuffling. Computes accuracy degradation and consistency curves. |
| **5** | `5_representation_analysis.ipynb` | Computes master cosine stability matrices, generates t-SNE / UMAP feature projections comparing clean and patch-shuffled representations, and outputs the master summary table. |

---

## Mathematical Formulations

### 1. Representation Stability ($I_T$)
The mean cosine similarity between the $L_2$-normalized representations of clean inputs $x_i$ and transformed inputs $T(x_i)$:
$$I_T = \frac{1}{N} \sum_{i=1}^N \frac{\langle f(x_i), f(T(x_i))\rangle}{\|f(x_i)\|_2 \|f(T(x_i))\|_2}$$

### 2. Prediction Consistency ($C_T$)
The fraction of inputs whose classification remains invariant under the intervention:
$$C_T = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\hat{y}(x_i) = \hat{y}(T(x_i))\right)$$

### 3. Shape Bias ($B_{\text{shape}}$) & Coverage ($\text{Cov}$)
Given $N_s$ shape-matching predictions, $N_t$ texture-matching predictions, and $N_o$ neither-matching predictions:
$$B_{\text{shape}} = \frac{N_s}{N_s + N_t}, \qquad \text{Cov} = \frac{N_s + N_t}{N_s + N_t + N_o}$$

---

## Key Empirical Findings

| Backbone | Clean Acc | Clean Conf | Grayscale Drop | Hue Drop | Shuffle Drop | Shape Bias | Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 97.0% | 0.93 | -3.6 pp | -3.2 pp | -8.6 pp | **60.8%** | 63.8% |
| **ViT-B/16** | 97.2% | 0.95 | -2.6 pp | -2.8 pp | -6.4 pp | **80.8%** | 71.7% |
| **CLIP Head** | 96.8% | 0.27* | -4.0 pp | -3.2 pp | -16.8 pp | **85.1%** | 64.2% |
| **CLIP Zero-Shot** | 93.4% | 0.93 | -3.8 pp | -3.0 pp | -15.8 pp | **82.2%** | 67.9% |

*\*Note on Confidence Artifact:* The low average confidence ($0.27$) of the CLIP linear probe is an artifact of training an unregularized linear head on unit-norm features without an explicit temperature scaler (like CLIP's native $100\times$ temperature multiplier), which artificially compresses the dynamic range of output logits.

---

## Directory Modules & Files

* `src/backbones.py`: Modular wrappers for torchvision ResNet-50, ViT-B/16, OpenCLIP ViT-B/32, and linear probe architectures.
* `src/data.py`: Deterministic STL-10 dataset loading and stratified subset sampling (`eval_indices.npy`).
* `src/transforms.py`: PyTorch transformation pipelines for Grayscale, HSV Hue rotation, translation, and patch shuffling.
* `src/cue_conflict.py`: PyTorch implementation of Adaptive Instance Normalization (AdaIN) style transfer.
* `src/analysis.py`: Metric evaluation routines ($I_T, C_T$, shape bias, coverage, t-SNE projections).
* `results/`: Contains machine-readable CSV result files (`task1_master_summary.csv`, `clean_baseline.csv`, `color_bias.csv`, `shape_bias.csv`, `patch_shuffle.csv`, `cosine_stability.csv`).
