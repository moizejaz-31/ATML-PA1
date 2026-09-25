# Shared PACS Protocol & Utilities (Tasks 2 & 3)

This directory implements the shared experimental protocol, data infrastructure, training constraints, loss functions, and evaluation metrics common to **Task 2 (Unsupervised Domain Adaptation)** and **Task 3 (Domain Generalization)** on the PACS dataset, exactly adhering to the **EE-5102/CS-6304 Programming Assignment 1 Manual**.

---

## Purpose & Protocol Integrity

The PA1 specification mandates:
> *"Use one shared PACS protocol for Tasks 2 and 3, and keep method-specific losses separate from the common training loop."* (PA1 Manual, Section 2)
> *"Reuse the shared PACS data and split code from Task 2. Keep source-only model selection separate from the script that loads Sketch for final evaluation."* (PA1 Manual, Section 3)

By centralizing the dataset partitions, training parameters, early-stopping logic, and evaluation metrics into `shared/`, the codebase guarantees:
1. **Identical Source Baselines:** Both Task 2 (Source-only) and Task 3 (ERM) train on the exact same data partitions with identical optimizer hyperparameters and checkpoint selection criteria.
2. **Zero Target Leakage in Task 2:** Sketch images are provided strictly *without labels* during transductive adaptation. Target labels are never consulted during training or checkpoint selection.
3. **Zero Target Exposure in Task 3:** Sketch data is strictly withheld during multi-source training, validation, and hyperparameter selection. Sketch data is loaded only during post-hoc evaluation after checkpoints are finalized.
4. **Deterministic Reproducibility:** Global seed `6304` is enforced across all data loaders, model initializations, and batch sampling.

---

## Directory Layout

```
shared/
├── __init__.py                   # Package initialization
├── README.md                     # Architecture and protocol documentation (this file)
├── splits/
│   └── pacs_sketch_seed6304.json # Pre-computed train/val index splits for PACS under seed 6304
└── src/
    ├── __init__.py               # Subpackage export
    ├── seed.py                   # Global deterministic seed configuration & logging
    ├── pacs_protocol.py          # Training budget, optimizer, BN freezing, early stopping
    ├── pacs_data.py              # PyTorch datasets and multi-domain DataLoaders
    ├── losses.py                 # MK-MMD, Gradient Reversal Layer (GRL), domain loss
    ├── metrics.py                # Accuracy, macro-F1, per-class accuracy, confusion matrix
    ├── diagrams.py               # Method architecture and protocol pipeline schematics
    └── plotting.py               # Color-blind validated plotting styles and curves
```

---

## Module Breakdown

### 1. `shared/src/pacs_protocol.py`
Enforces the standardized training configuration required across all PACS experiments:
* **Pretrained Backbone:** TorchVision `ResNet-18` loaded with `ResNet18_Weights.IMAGENET1K_V1` and a 7-class linear classifier head.
* **BatchNorm Freezing (`freeze_bn_stats`):** Following standard UDA/DG practice, batch normalization layers have their running statistics ($\mu, \sigma^2$) frozen to ImageNet priors (`bn.eval()`), while affine parameters ($\gamma, \beta$) remain trainable. This prevents target batch statistics or mixed source distributions from corrupting feature normalization.
* **Optimization Budget:**
  - `MAX_EPOCHS = 30` with `PATIENCE = 5`.
  - Checkpoint selection criterion: **Mean Source-Validation Macro-F1**.
  - Optimizer: AdamW with backbone learning rate $10^{-4}$ and weight decay $10^{-4}$.
* **Standard Transforms:**
  - Training: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip()`, ImageNet normalization.
  - Evaluation: `Resize((224, 224))`, ImageNet normalization.

### 2. `shared/src/pacs_data.py`
Provides domain-aware data loading across the four PACS domains: `photo`, `art_painting`, `cartoon`, and `sketch` (7 classes: *dog, elephant, giraffe, guitar, horse, house, person*).
* **Multi-Source Training Loader:** Allocates balanced minibatches across sources ($N_{\text{src}} = 8$ images per domain $\times 3 = 24$ source samples per iteration).
* **Target Domain Loader (Task 2):** Streams $N_{\text{tgt}} = 24$ unlabeled Sketch samples per iteration to match source batch size.
* **Validation Split Loader:** Loads stratified source validation splits ($20\%$ per source domain) for metric logging and early stopping.

### 3. `shared/splits/pacs_sketch_seed6304.json`
Stores the pre-computed, deterministic file lists and stratified split indices for all PACS domains generated under random seed `6304`. Storing explicit indices prevents filesystem ordering discrepancies across operating systems (Linux vs. Windows vs. macOS).

### 4. `shared/src/losses.py`
Implements distribution alignment and adversarial objectives:
* **Multiple-Kernel MMD (`compute_mmd`):** Evaluates the biased Maximum Mean Discrepancy with a mixture of 5 Gaussian RBF kernels:
  $$\text{MMD}^2(X_s, X_t) = \frac{1}{n^2}\sum_{i,j} k(s_i, s_j) + \frac{1}{m^2}\sum_{i,j} k(t_i, t_j) - \frac{2}{nm}\sum_{i,j} k(s_i, t_j)$$
  with bandwidths $\sigma \in \{10^{-2}, 10^{-1}, 1, 10, 10^2\} \times \text{median heuristic}$.
* **Gradient Reversal Layer (`GradientReversalLayer`):** Autograd function that acts as an identity during forward propagation and scales gradients by $-\lambda_p$ during backward propagation:
  $$\lambda_p = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1, \quad p = \frac{\text{epoch}}{\text{max\_epochs}}$$

### 5. `shared/src/metrics.py`
Standard evaluation metrics for classification and distribution diagnostics:
* `macro_f1(y_true, y_pred)`: Unweighted mean of per-class F1-scores, critical for diagnosing failure modes under imbalanced test slices (e.g., PACS `house` $n=80$ vs `horse` $n=816$).
* `per_class_accuracy(y_true, y_pred, num_classes)`: Full diagonal breakdown of class accuracies.
* `prediction_consistency(p1, p2)`: Percentage of identical predictions across interventions.

### 6. `shared/src/seed.py`
Enforces global reproducibility across environments:
```python
from shared.src.seed import set_seed
set_seed(6304)  # Sets Python random, numpy.random, torch.manual_seed, torch.cuda.manual_seed_all
```

---

## Cross-Task Usage

### Task 2: Unsupervised Domain Adaptation
```python
from shared.src.pacs_protocol import get_resnet18, freeze_bn_stats, EarlyStopping
from shared.src.pacs_data import get_uda_loaders
from shared.src.losses import compute_mmd, GradientReversalLayer

model = get_resnet18(num_classes=7)
freeze_bn_stats(model)
train_loader, val_loader, test_loader = get_uda_loaders(target_domain="sketch")
```

### Task 3: Domain Generalization
```python
from shared.src.pacs_protocol import get_resnet18, freeze_bn_stats, EarlyStopping
from shared.src.pacs_data import get_dg_loaders
from shared.src.losses import compute_mmd

model = get_resnet18(num_classes=7)
freeze_bn_stats(model)
# Note: Sketch is NEVER passed to get_dg_loaders during training or selection
train_loader, val_loader = get_dg_loaders(exclude_domain="sketch")
```
