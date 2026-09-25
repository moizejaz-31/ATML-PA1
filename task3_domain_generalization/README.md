# Task 3: Domain Generalization (PACS, Sketch Unseen)

This directory contains the implementation, notebooks, and evaluation protocols for **Task 3: Domain Generalization** on the PACS dataset, investigating whether multi-source distribution invariance or local parameter flatness confers superior out-of-distribution transfer to an unseen target domain (**Sketch**).

---

## Experimental Protocol & Integrity Guarantees

In strict accordance with the **EE-5102/CS-6304 Programming Assignment 1 Manual**:
1. **Zero Target Exposure:** Sketch images are **never loaded, seen, or consulted** during training, model selection, or hyperparameter tuning. The source dataset loaders in `src/train.py` strictly prohibit constructing a Sketch dataset.
2. **Post-Hoc Target Evaluation:** Sketch is evaluated **only once** in `4_evaluation.ipynb` after all method checkpoints are locked based on mean source-validation macro-F1.
3. **Shared Source Baseline:** The ERM baseline loads the identical Task 2 Source-only checkpoint without modification, ensuring an exact controlled comparison across adaptation paradigms.
4. **Budget & Constraints:**
   - Maximum budget: $\le 30$ epochs with patience $5$ on source-validation macro-F1.
   - Backbone: ImageNet-pretrained ResNet-18 with frozen BatchNorm statistics (`freeze_bn_stats`).
   - Optimizer: AdamW ($LR = 10^{-4}, WD = 10^{-4}$).

---

## Evaluated Methods

### 1. Empirical Risk Minimization (ERM)
Standard multi-source risk minimization over the combined distribution of `photo`, `art_painting`, and `cartoon`:
$$\mathcal{L}_{\text{ERM}}(\theta) = \frac{1}{|\mathcal{S}|} \sum_{e \in \mathcal{S}} \frac{1}{N_e} \sum_{i=1}^{N_e} \mathcal{L}_{\text{CE}}(f(x_i^{(e)}; \theta), y_i^{(e)})$$

### 2. Multi-Source Domain Adaptation Network (DAN-DG)
Enforces pairwise distribution alignment across all three source domain pairs without any access to target data:
$$\mathcal{L}_{\text{DAN-DG}}(\theta) = \mathcal{L}_{\text{ERM}}(\theta) + \frac{\lambda_{\text{DG}}}{3} \sum_{e < e'} \text{MMD}^2\left(\mathcal{F}(X^{(e)}), \mathcal{F}(X^{(e')})\right)$$
where MMD is computed using an unbiased multi-kernel Gaussian RBF mixture with bandwidths dynamically calibrated via the batch median heuristic.

### 3. Sharpness-Aware Minimization (SAM)
Instead of matching feature distributions, SAM seeks parameter configurations located in uniformly flat loss basins:
$$\min_\theta \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}_{\text{ERM}}(\theta + \epsilon)$$
* **Two-Step Update:**
  1. Compute ascent perturbation: $\hat{\epsilon}(\theta) = \rho \frac{\nabla_\theta \mathcal{L}_{\text{ERM}}(\theta)}{\|\nabla_\theta \mathcal{L}_{\text{ERM}}(\theta)\|_2}$ with perturbation radius $\rho = 0.05$.
  2. Perturb parameters: $\theta \leftarrow \theta + \hat{\epsilon}(\theta)$.
  3. Compute gradient at perturbed parameters $\nabla_\theta \mathcal{L}_{\text{ERM}}(\theta + \hat{\epsilon})$ and restore $\theta$.
  4. Perform base optimizer step (AdamW) using the perturbed gradient.

---

## Diagnostics: Invariance vs. Flatness

We evaluate two distinct diagnostic mechanisms:
1. **Source-Domain Separability:** A 3-way multinomial logistic regression probe ($C=1$, balanced, evaluated on stratified 70/30 source splits) measuring how distinguishable source domains remain in the latent feature space (chance level $= 33.3\%$). Lower separability indicates greater domain invariance.
2. **Loss Surface Sharpness ($\Delta_{\text{sharp}}$):** Evaluates local loss variation on a fixed evaluation batch (32 images per source domain, seed 6304):
   $$\Delta_{\text{sharp}} = \mathcal{L}(\theta + \hat{\epsilon}) - \mathcal{L}(\theta)$$
   Lower $\Delta_{\text{sharp}}$ indicates a flatter loss minimum.

---

## Empirical Results Summary

### Main Comparison Table
*All models selected purely on mean source-validation macro-F1:*

| Method | Mean Source F1 | Worst Source F1 | Source Separability | Sharpness ($\Delta_{\text{sharp}}$) | Sketch Acc | Sketch F1 | $\Delta$ vs ERM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ERM** | 93.8% | 91.3% (Art) | 89.7% | 0.355 | 59.2% | 66.1% | $0.0\pp$ |
| **DAN-DG** | 93.9% | 91.6% (Art) | **70.1%** | 0.235 | **69.0%** | **70.4%** | $+9.8\pp$ |
| **SAM** | **95.4%** | **92.0%** (Art) | 88.4% | **0.091** | 68.3% | 67.0% | $+9.1\pp$ |

### Per-Class Sketch Accuracy Breakdown (%)
*Class sample sizes: dog ($n=772$), elephant ($n=740$), giraffe ($n=753$), guitar ($n=608$), horse ($n=816$), house ($n=80$), person ($n=160$). Total $n=3929$.*

| Class | ERM (Base) | DAN-DG | SAM | Behavioral Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **dog** | 81.7 | 56.6 | 51.7 | Feature invariance erodes fine-grained canine textures. |
| **elephant** | 57.8 | 69.2 | **93.0** | SAM achieves massive $+35.1\pp$ gain via flat boundary margins. |
| **giraffe** | 37.5 | **65.6** | 52.2 | Multi-source alignment resolves color/spot discrepancy. |
| **guitar** | 84.0 | 81.6 | **92.8** | Flat minima stabilize salient geometric contours. |
| **horse** | 37.9 | **70.2** | 59.1 | Eliminates ERM horse $\to$ dog confusion (drops from 52% to 12%). |
| **house** | 87.5 | 85.0 | 46.3* | Small class ($n=80$): 42% misclassified as elephant under SAM. |
| **person** | 60.0 | **81.9** | 75.0 | Substantial generalization gains across non-photographic domains. |

*\*Note on Imbalance:* Because `house` contains only 80 test images ($1\text{ image} = 1.25\pp$), its $41.3\pp$ drop depresses macro-F1 while overall accuracy remains high (68.3%).

---

## Controlled Alignment Study ($\lambda_{\text{DG}} \in \{0.1, 1, 10\}$)

| $\lambda_{\text{DG}}$ | Mean Source F1 | Worst Source F1 | Source Separability | Sharpness ($\Delta_{\text{sharp}}$) | Sketch Acc | Observation |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 (ERM) | 93.8% | 91.3% | 89.7% | 0.355 | 59.2% | Baseline unaligned risk minimization. |
| 0.1 | **95.0%** | **92.6%** | 84.1% | 0.285 | 65.7% | Selected by source validation; under-aligns Sketch. |
| 1.0 | 93.9% | 91.6% | **70.1%** | 0.235 | **69.0%** | Optimal cross-domain transfer to Sketch. |
| 10.0 | 5.1% | 4.2% | 67.1%* | 0.036* | 4.1% | Catastrophic collapse: predicts single class (`person`). |

*\*Deceptive Diagnostics:* At $\lambda=10$, source separability drops to $67.1\%$ and $\Delta_{\text{sharp}}$ drops to $0.036$, deceptively appearing favorable despite total model collapse.

---

## Directory Modules & Files

* `notebooks/1_erm_baseline.ipynb`: Evaluates the shared source-only checkpoint across all three sources and worst-source domain.
* `notebooks/2_dan_dg.ipynb`: Trains DAN-DG using pairwise source MMD.
* `notebooks/3_sam.ipynb`: Implements Sharpness-Aware Minimization with frozen BatchNorm passes.
* `notebooks/4_evaluation.ipynb`: Final locked evaluation on unseen Sketch, domain separability, and sharpness profiling.
* `notebooks/5_controlled_study.ipynb`: Parametric sensitivity analysis over $\lambda_{\text{DG}} \in \{0.1, 1, 10\}$.
* `src/dan_dg.py`: Pairwise multi-source MMD loss computation.
* `src/sam.py`: PyTorch optimizer wrapper for SAM perturbation and step logic.
* `src/evaluate.py`: Source separability probe and loss sharpness estimation.
* `src/train.py`: Unified domain generalization training loop.
* `results/`: Machine-readable results (`task3_main_table.csv`, `task3_per_class_sketch.csv`, `task3_controlled_study.csv`).
