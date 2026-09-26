# Task 4: Open-Set Recognition (CIFAR-10 Known, CIFAR-100 Unknown)

This directory contains the code, models, and evaluation suite for **Task 4: Open-Set Recognition (OSR)**, investigating how convolutional classifiers identify inputs from unseen semantic classes rather than overconfidently mapping them to known categories.

---

## Experimental Setup & Partitions

* **Known Classes (CIFAR-10):** All 10 classes (*airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*).
  - Training / Validation: Stratified 90/10 split of the official training set under seed `6304` (`splits/cifar10_seed6304.json`).
  - Known Test Evaluation: Official CIFAR-10 test set ($n=10,000$).
* **Unknown Classes (CIFAR-100):** Exactly 16 classes selected from the official CIFAR-100 test set ($800$ test images per class; CIFAR-100 training sets are **never loaded**):
  - **Near Unknowns (visually/semantically proximate):** *bus, pickup truck, motorcycle, tractor, wolf, fox, leopard, camel* (Total $n=6,400$).
  - **Far Unknowns (semantically dissimilar):** *bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe* (Total $n=6,400$).

---

## Evaluated Paradigms

All experiments utilize a custom **CIFAR ResNet-18** ($3\times 3$ stride-1 stem, no initial max-pooling, feature dimension $512$):

1. **Vanilla Baseline:** Optimized via standard empirical cross-entropy over known classes ($100$ epochs, SGD with momentum $0.9$, cosine annealing schedule).
2. **Good Closed-Set Classifier (GCSC) (Vaze et al., 2022):** Incorporates heavy RandAugment ($N=2, M=9$) data augmentation on top of standard random crop/flip to test the hypothesis that stronger closed-set classifiers automatically yield superior open-set rejection.
3. **PROSER (Zhou et al., 2021):** Extends the classifier head with $M=5$ dummy placeholder heads. Training alternates between:
   - Optimizing classifier placeholders on known instances (masking the true known logit to target dummy class $K+1$).
   - Synthesizing data placeholders via manifold mixup on intermediate `layer2` feature activations:
     $$\tilde{h} = \lambda h_i + (1-\lambda)h_j, \quad y_i \ne y_j, \quad \lambda \sim \text{Beta}(2, 2)$$
   - PROSER fine-tunes from the Vanilla checkpoint for $50$ epochs with learning rate $10^{-3}$.

---

## Theoretical Constraint: Convex Hull Bound

**Theoretical Guarantee (Proposition 1):**
> *Manifold mixup synthesizes representations strictly within the convex hull of known training representations $\text{Conv}(\mathcal{S})$. Consequently, synthetic data placeholders cannot populate the exterior open space $\mathcal{O}_{\text{ext}}$, leaving boundaries facing truly novel semantic space unsupported by synthetic data.*

---

## Post-Hoc Scoring Functions

We evaluate four post-hoc unknownness functions $u(x)$ on the frozen Vanilla model:
1. **Maximum Softmax Probability (MSP):**
   $$u_{\text{MSP}}(x) = 1 - \max_{k \in \{1,\dots,K\}} \sigma(z(x))_k$$
2. **Maximum Logit Score (MLS):**
   $$u_{\text{MLS}}(x) = -\max_{k \in \{1,\dots,K\}} z(x)_k$$
3. **Energy Score ($T=1$):**
   $$u_{\text{Energy}}(x) = -T \cdot \log \sum_{k=1}^K \exp\left(\frac{z(x)_k}{T}\right)$$
4. **Mahalanobis Feature Distance:**
   $$u_{\text{Mahal}}(x) = \min_{k \in \{1,\dots,K\}} (f(x) - \hat{\mu}_k)^\top \hat{\Sigma}^{-1} (f(x) - \hat{\mu}_k)$$
   using class-conditional centroids $\hat{\mu}_k$ and shared diagonal covariance $\hat{\Sigma}$ fitted on unaugmented known training features.
5. **PROSER Placeholder Score:**
   $$u_{\text{PROSER}}(x) = \frac{\exp(\max_{m=1}^M d_m(x))}{\sum_{k=1}^{10} \exp(z(x)_k) + \exp(\max_{m=1}^M d_m(x))}$$

---

## Operational Threshold Calibration ($\tau$)

To simulate real-world deployment, an operational threshold $\tau$ is calibrated on the **CIFAR-10 validation set** to achieve a fixed **95% True Positive Rate (TPR)** on known classes. An input is rejected as unknown if $u(x) > \tau$.

---

## Master Empirical Results

### Table 1: Post-Hoc Scoring Functions (Frozen Vanilla Model, CSA = 94.81%)

| Scoring Function | AUROC Near | AUROC Far | AUROC All | Known Accept at $\tau$ | Near Reject at $\tau$ | Far Reject at $\tau$ | FPR@95 (All) | Threshold $\tau$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MSP** | **81.8%** | 89.6% | 85.7% | 94.7% | 31.8% | 44.4% | 61.9% | 0.156 |
| **MLS** | 80.3% | 89.5% | 84.9% | 94.7% | **35.0%** | 53.5% | 55.8% | $-5.84$ |
| **Energy** | 80.3% | 89.5% | 84.9% | 94.6% | **35.0%** | **55.0%** | **55.0%** | $-6.03$ |
| **Mahalanobis** | 80.1% | **91.4%** | **85.7%** | 94.6% | 29.8% | 51.3% | 59.5% | $2812$ |

### Table 2: Model Comparison (Evaluated with MLS as Standard Score)

| Model (Score Metric) | Closed-Set Acc (CSA) | AUROC Near | AUROC Far | AUROC All | Known Accept | Near Reject | Far Reject | FPR@95 (All) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla (MLS)** | 94.81% | 80.3% | 89.5% | 84.9% | 94.7% | **35.0%** | 53.5% | **55.8%** |
| **GCSC (MLS)** | **94.95%** | **81.4%** | **90.2%** | **85.8%** | 94.9% | 31.4% | **55.4%** | 56.6% |
| **PROSER (MLS)** | 94.35% | 80.1% | 87.5% | 83.8% | 94.7% | 32.0% | 48.5% | 59.8% |
| **PROSER (Placeholder)** | 94.35% | 78.5% | 88.3% | 83.4% | 94.8% | 32.0% | 51.8% | 58.1% |

*Key Finding:* GCSC achieves the highest closed-set accuracy (94.95%), but near-unknown rejection drops by 3.6 pp (35.0% to 31.4%). Strong closed-set optimization expands known decision regions into adjacent open space, causing near unknowns to be falsely absorbed.

---

## Top Failure Modes (Semantic Absorption)

Under the calibrated Vanilla MLS threshold ($\tau = -5.84$):

| Group | Unknown Class | Assigned CIFAR-10 Class | Score $u_{\text{MLS}}$ | Semantic Plausibility | Root Cause |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Near** | `bus` | `truck` | $-12.62$ | Highly Plausible | High structural & vehicular part overlap. |
| **Near** | `fox` | `dog` | $-12.13$ | Highly Plausible | Shared canine facial structure and fur texture. |
| **Near** | `camel` | `cat` | $-11.43$ | Surprising | Quadruped silhouette matches quadruped class. |
| **Far** | `bottle` | `cat` | $-12.21$ | Surprising | Cylindrical shape aligns with sitting cat profile. |
| **Far** | `mushroom` | `bird` | $-11.73$ | Surprising | Texture and stalk curvature match bird perch. |
| **Far** | `keyboard` | `truck` | $-11.14$ | Surprising | Rectangular grid resembles truck grille/cargo. |

---

## Directory Modules & Files

* `notebooks/1_vanilla_baseline.ipynb`: Trains Vanilla ResNet-18, evaluates closed-set accuracy, caches known logits/features.
* `notebooks/2_posthoc_scores.ipynb`: Computes MSP, MLS, Energy, Mahalanobis; evaluates validation calibration and OSR AUROC.
* `notebooks/3_gcsc.ipynb`: Implements RandAugment ($N=2, M=9$) closed-set training.
* `notebooks/4_proser.ipynb`: Implements PROSER dual placeholder training (classifier + manifold mixup on `layer2`).
* `notebooks/5_evaluation.ipynb`: Multi-model comparison, threshold calibration, and semantic failure analysis.
* `src/resnet_cifar.py`: CIFAR ResNet-18 architecture with `pre_mix` / `forward_from_mid` split for manifold mixup.
* `src/proser.py`: Dual placeholder loss functions and dummy logit pooling.
* `src/scores.py`: Scoring function implementations (MSP, MLS, Energy, Mahalanobis, PROSER dummy head).
* `src/data.py`: Dataset loaders, split indexing, and RandAugment definitions.
* `src/evaluate.py`: ROC curve analysis, AUROC calculation, and 95% TPR threshold calibration.
* `results/`: Machine-readable results (`posthoc_scores_table.csv`, `model_comparison_table.csv`, `failure_cases_vanilla_mls.csv`, `*_csa.json`).
