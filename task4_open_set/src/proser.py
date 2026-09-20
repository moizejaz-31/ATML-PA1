"""
PROSER: Classifier and Data Placeholders.

1. Classifier placeholders (β=1): dummy classifiers as strongest
   response when true class excluded.
2. Data placeholders (γ=0.1): manifold mixup after layer2, Beta(2,2),
   train mixed features toward dummies.

Init from Vanilla checkpoint. 5 dummies. Fine-tune 50 epochs.
SGD lr=1e-3, momentum=0.9, wd=5e-4, cosine decay, batch 128.
Split mini-batch: first half classifier-placeholder, second half data-placeholder.
"""

import torch
import numpy as np


def manifold_mixup(features, labels, alpha=2.0):
    """Mix features from different classes. lambda ~ Beta(alpha, alpha)."""
    B = features.size(0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(B)
    mixed = lam * features + (1 - lam) * features[perm]
    return mixed, lam, perm


# TODO: Implement PROSER training loop
# - classifier_placeholder_loss (β=1)
# - data_placeholder_loss (γ=0.1)
# - Combined training objective
