"""
Post-hoc novelty scores. Higher = more unknown.

MSP:  u = 1 - max softmax
MLS:  u = -max logit
Energy: u = -logsumexp(logits)
Mahalanobis: u = min_c (f-mu_c)^T Sigma^{-1} (f-mu_c)  [diagonal]
"""

import torch
import torch.nn.functional as F
import numpy as np


def msp(logits):
    return 1.0 - F.softmax(logits, dim=1).max(dim=1).values

def mls(logits):
    return -logits.max(dim=1).values

def energy(logits):
    return -torch.logsumexp(logits, dim=1)


def fit_mahalanobis(features, labels, num_classes=10, reg=1e-6):
    """Fit per-class means and shared diagonal covariance."""
    D = features.shape[1]
    means = np.zeros((num_classes, D))
    centered = []
    for c in range(num_classes):
        cf = features[labels == c]
        means[c] = cf.mean(0)
        centered.append(cf - means[c])
    var = np.var(np.concatenate(centered), axis=0) + reg
    return means, 1.0 / var


def mahalanobis(features, means, precision):
    """Min Mahalanobis distance across classes."""
    scores = np.full(len(features), np.inf)
    for c in range(len(means)):
        d = ((features - means[c]) ** 2 * precision).sum(axis=1)
        scores = np.minimum(scores, d)
    return scores
