# Learning Beyond IID: Inductive Biases, Domain Adaptation, Generalization, and Open-Set Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Random Seed: 6304](https://img.shields.io/badge/Seed-6304-green.svg)](shared/src/seed.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Course:** EE-5102 / CS-6304: Advanced Topics in Machine Learning  
> **Institution:** Department of Computer Science, Lahore University of Management Sciences  
> **Repository:** [https://github.com/moizejaz-31/ATML-PA1](https://github.com/moizejaz-31/ATML-PA1)  

---

## Executive Summary & Core Findings

Standard machine learning relies on the Independent and Identically Distributed (IID) assumption, presuming that training and evaluation environments share identical data distributions and closed label sets. This repository investigates empirical behavior, failure modes, and theoretical constraints when these assumptions are violated across four interconnected visual recognition tasks:

1. **Task 1: Inductive Biases & Feature Representations (STL-10):**
   * Pretrained Vision Transformers exhibit substantially higher shape bias (**80.8%** for ViT-B/16, **85.1%** for CLIP) than convolutional networks (**60.8%** for ResNet-50) under AdaIN cue conflicts.
   * Color perturbations (Grayscale, Hue rotation) produce minor drops ($<4.0\pp$), confirming color is treated as nuisance variation across all models.
   * Spatial patch shuffling exposes severe sensitivity in CLIP ($-16.8\pp$ drop) as non-aligned 56px patches break contrastive embeddings, while ViT-B/16 remains resilient ($-6.4\pp$).
   * Crucially, cue coverage is incomplete ($63.8\%\text{--}71.7\%$), meaning $\sim 1/3$ of predictions fall outside both shape and texture categories ($N_o$).

2. **Task 2: Unsupervised Domain Adaptation (PACS $\to$ Sketch):**
   * Multi-kernel Maximum Mean Discrepancy (DAN) bridges distribution shifts without target labels, lifting target accuracy from **59.2%** (Source-only) to **65.5%** (and **70.6%** at $\lambda=0.1$).
   * Adversarial alignment exhibits catastrophic negative transfer: DANN drops to **45.8%** and CDAN to **52.0%** due to discriminator underfitting ($51.5\%$ domain accuracy) and gradient conflict.
   * Over-alignment causes severe single-class collapse: setting $\lambda=10$ collapses the network to predicting `person` (4.07% accuracy).

3. **Task 3: Domain Generalization (PACS, Sketch Unseen):**
   * **Zero-target generalization outperforms target-aware adaptation:** Without observing any target data, multi-source moment matching (DAN-DG, **69.0%**) and Sharpness-Aware Minimization (SAM, **68.3%**) outperform target-aware DAN (**65.5%**).
   * Flattening loss minima (SAM, $\Delta_{\text{sharp}} = 0.091$) and enforcing multi-source invariance (DAN-DG, source separability $70.1\%$) provide two independent and effective pathways to out-of-distribution robustness.
   * Severe class imbalance (PACS `house` $n=80$ vs `horse` $n=816$) causes accuracy and macro-F1 to diverge: SAM drops house accuracy to $46.3\%$ ($-41.3\pp$), depressing macro-F1 while maintaining high overall accuracy.

4. **Task 4: Open-Set Recognition (CIFAR-10 Known, CIFAR-100 Unknown):**
   * **Stronger closed-set classifiers do not ensure open-set safety:** GCSC achieves superior closed-set accuracy (**94.95%** vs 94.81%), but increases near-unknown false acceptance by **3.6 pp** ($35.0\% \to 31.4\%$).
   * Post-hoc scoring functions (MLS, Energy) reject far unknowns effectively ($>89.5\%$ AUROC), but struggle against near unknowns ($\sim 80.1\%$ AUROC) due to semantic absorption.
   * **Convex Hull Theoretical Bound:** We formally prove that manifold mixup (PROSER) is geometrically confined to the convex hull of known training representations $\text{Conv}(\mathcal{S})$ and cannot populate exterior open space $\mathcal{O}_{\text{ext}}$, establishing a fundamental limitation of synthetic placeholder generation.

---

## Repository Architecture

```
pa1-beyond-iid/
├── README.md                           # Main repository overview & reproduction guide
├── requirements.txt                   # Frozen Python environment dependencies
├── setup.py                           # Installable package configuration
│
├── shared/                            # Shared PACS protocol & common infrastructure (Tasks 2 & 3)
│   ├── README.md                      # Detailed protocol & module documentation
│   ├── splits/
│   │   └── pacs_sketch_seed6304.json  # Pre-computed deterministic train/val splits
│   └── src/
│       ├── pacs_protocol.py           # Early stopping, optimizer, frozen BN, transforms
│       ├── pacs_data.py               # Multi-source and transductive target DataLoaders
│       ├── losses.py                  # Unbiased MK-MMD, Gradient Reversal Layer (GRL)
│       ├── metrics.py                 # Accuracy, macro-F1, per-class metrics
│       ├── seed.py                    # Global seed determinism (seed 6304)
│       ├── diagrams.py                # Publication-grade architecture & protocol figures
│       └── plotting.py                # Color-blind validated plotting utilities
│
├── task1_inductive_biases/            # Task 1: Inductive Biases & Feature Representations
│   ├── README.md                      # Detailed Task 1 documentation & results
│   ├── cache/                         # Deterministic evaluation indices (eval_indices.npy)
│   ├── notebooks/                     # 1_setup, 2_color, 3_shape, 4_spatial, 5_representation
│   ├── src/                           # Backbones, transforms, AdaIN cue-conflict, analysis
│   └── results/                       # Master summary CSVs & metric breakdowns
│
├── task2_domain_adaptation/           # Task 2: Unsupervised Domain Adaptation (UDA)
│   ├── README.md                      # Detailed Task 2 documentation & adversarial dynamics
│   ├── checkpoints/                   # Run configs & training histories (*_history.json)
│   ├── notebooks/                     # 1_source_only, 2_dan, 3_dann, 4_cdan, 5_eval, 6_study
│   ├── src/                           # ResNet-18, discriminators, unified UDA training loop
│   └── results/                       # Target evaluation CSVs & per-class delta tables
│
├── task3_domain_generalization/       # Task 3: Zero-Target Domain Generalization (DG)
│   ├── README.md                      # Detailed Task 3 documentation, SAM math & results
│   ├── checkpoints/                   # Model histories for ERM, DAN-DG, SAM
│   ├── notebooks/                     # 1_erm, 2_dan_dg, 3_sam, 4_evaluation, 5_study
│   ├── src/                           # Multi-source MMD, SAM optimizer wrapper, flatness
│   └── results/                       # DG main comparison, per-class Sketch metrics
│
└── task4_open_set/                    # Task 4: Open-Set Recognition (OSR)
    ├── README.md                      # Detailed Task 4 documentation & convex hull proof
    ├── checkpoints/                   # Checkpoint histories (Vanilla, GCSC, PROSER)
    ├── notebooks/                     # 1_vanilla, 2_scores, 3_gcsc, 4_proser, 5_evaluation
    ├── splits/                        # Deterministic CIFAR-10 split (cifar10_seed6304.json)
    ├── src/                           # CIFAR ResNet-18, PROSER mixup, post-hoc scores
    └── results/                       # OSR AUROC tables, 95% TPR thresholds, failure cases
```

---

## Environment Setup & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/moizejaz-31/ATML-PA1.git
cd ATML-PA1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages and editable repo package
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Setup
Data directories are kept outside git tracking (via `.gitignore`):
* **STL-10:** Automatically downloaded by torchvision upon running `task1_inductive_biases/notebooks/1_setup_and_baselines.ipynb`.
* **PACS:** Extract the official PACS dataset so images reside in:
  ```
  data/PACS/
  ├── photo/<class>/*.jpg
  ├── art_painting/<class>/*.jpg
  ├── cartoon/<class>/*.jpg
  └── sketch/<class>/*.jpg
  ```
* **CIFAR-10 & CIFAR-100:** Automatically downloaded via torchvision in `task4_open_set/notebooks/1_vanilla_baseline.ipynb` and `2_posthoc_scores.ipynb`.

---

## Reproduction Workflow

Each task is structured into sequenced, self-contained Jupyter notebooks that add the repository root to `sys.path`. Notebooks can be executed interactively or headlessly:

```bash
# Example headless execution
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook_path>
```

### Execution Order per Task:

| Task | Execution Pipeline |
| :--- | :--- |
| **Task 1** | `1_setup_and_baselines` $\to$ `2_color_bias` $\to$ `3_shape_vs_texture` $\to$ `4_spatial_sensitivity` $\to$ `5_representation_analysis` |
| **Task 2** | `1_source_only` $\to$ `2_dan` $\to$ `3_dann` $\to$ `4_cdan` $\to$ `5_evaluation` $\to$ `6_controlled_study` |
| **Task 3** | `1_erm_baseline` (loads Task 2 source-only) $\to$ `2_dan_dg` $\to$ `3_sam` $\to$ `4_evaluation` $\to$ `5_controlled_study` |
| **Task 4** | `1_vanilla_baseline` $\to$ `3_gcsc` $\to$ `4_proser` $\to$ `2_posthoc_scores` (first contact with unknowns) $\to$ `5_evaluation` |

---

## Experimental Protocol & Integrity Guarantees

* **Zero Target Leakage in UDA (Task 2):** Target domain (Sketch) labels are strictly omitted during training and checkpoint selection. Checkpoints are chosen exclusively based on **mean source-validation macro-F1**.
* **Zero Target Exposure in DG (Task 3):** Sketch images are completely inaccessible during training. DataLoaders in `task3_domain_generalization/src/train.py` strictly prohibit constructing a Sketch dataset. Sketch is evaluated only once post-hoc.
* **Closed-Set Integrity in OSR (Task 4):** CIFAR-100 images are never loaded during representation learning, classifier training, or threshold calibration. The operational rejection threshold $\tau$ is calibrated strictly on **CIFAR-10 validation data** at a fixed 95% TPR.
* **Frozen ImageNet Normalization:** Across Tasks 2 and 3, BatchNorm running statistics ($\mu, \sigma^2$) are frozen to ImageNet priors (`freeze_bn_stats`), preventing mixed source batches or unlabeled target batches from corrupting feature normalization.
* **Strict Global Determinism:** All experiments fix random seed **`6304`** across Python, NumPy, PyTorch CPU/CUDA, and cuDNN backends.

---

## Master Benchmark Results

### Task 1: Inductive Biases (STL-10)
| Model | Clean Acc | Clean Conf | Grayscale Drop | Hue Drop | Shuffle Drop | Shape Bias | Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 97.0% | 0.93 | $-3.6\pp$ | $-3.2\pp$ | $-8.6\pp$ | **60.8%** | 63.8% |
| **ViT-B/16** | 97.2% | 0.95 | $-2.6\pp$ | $-2.8\pp$ | $-6.4\pp$ | **80.8%** | 71.7% |
| **CLIP Head** | 96.8% | 0.27* | $-4.0\pp$ | $-3.2\pp$ | $-16.8\pp$ | **85.1%** | 64.2% |
| **CLIP Zero-Shot** | 93.4% | 0.93 | $-3.8\pp$ | $-3.0\pp$ | $-15.8\pp$ | **82.2%** | 67.9% |

### Task 2 & Task 3: PACS Sketch Adaptation & Generalization
| Method | Target Access | Mean Source F1 | Source Separability | Sharpness ($\Delta_{\text{sharp}}$) | Sketch Acc | Sketch F1 | $\Delta$ vs Base |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Source-only / ERM** | None | 93.8% | 89.7% | 0.355 | 59.2% | 66.1% | $0.0\pp$ |
| **DAN ($\lambda=1$)** | Unlabeled | 94.2% | 81.5% | — | 65.5% | 61.0% | $+6.3\pp$ |
| **DAN ($\lambda=0.1$)** | Unlabeled | 93.9% | 96.6% | — | **70.6%** | **69.7%** | $+11.4\pp$ |
| **DANN** | Unlabeled | 94.1% | 84.7% | — | 45.8% | 42.1% | $-13.4\pp$ |
| **CDAN** | Unlabeled | 93.9% | 87.2% | — | 52.0% | 48.6% | $-7.2\pp$ |
| **DAN-DG ($\lambda=1$)** | Zero | 93.9% | **70.1%** | 0.235 | **69.0%** | **70.4%** | $+9.8\pp$ |
| **SAM ($\rho=0.05$)** | Zero | **95.4%** | 88.4% | **0.091** | 68.3% | 67.0% | $+9.1\pp$ |

### Task 4: Open-Set Recognition (CIFAR-10 $\to$ CIFAR-100)
| Model (Score) | Closed-Set Acc | Near AUROC | Far AUROC | All AUROC | Known Accept at $\tau$ | Near Reject at $\tau$ | Far Reject at $\tau$ | FPR@95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla (MLS)** | 94.81% | 80.3% | 89.5% | 84.9% | 94.7% | **35.0%** | 53.5% | **55.8%** |
| **GCSC (MLS)** | **94.95%** | **81.4%** | **90.2%** | **85.8%** | 94.9% | 31.4% | **55.4%** | 56.6% |
| **PROSER (MLS)** | 94.35% | 80.1% | 87.5% | 83.8% | 94.7% | 32.0% | 48.5% | 59.8% |
| **PROSER (Placeholder)** | 94.35% | 78.5% | 88.3% | 83.4% | 94.8% | 32.0% | 51.8% | 58.1% |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
