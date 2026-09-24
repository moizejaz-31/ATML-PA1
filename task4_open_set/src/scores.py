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


def proser_score(logits, dummy):
    """PROSER placeholder score (reference implementation): probability assigned to the
    strongest dummy after a softmax over [known logits, max dummy logit]. Higher = more unknown."""
    logits, dummy = np.asarray(logits, dtype=np.float64), np.asarray(dummy, dtype=np.float64)
    aug = np.concatenate([logits, dummy.max(axis=1, keepdims=True)], axis=1)
    aug = aug - aug.max(axis=1, keepdims=True)
    p = np.exp(aug) / np.exp(aug).sum(axis=1, keepdims=True)
    return p[:, -1]


def all_posthoc_scores(logits, feats, maha_means, maha_precision):
    """MSP / MLS / Energy / Mahalanobis from the SAME saved logits and features (numpy in, numpy out)."""
    t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    return {
        "MSP": msp(t).numpy(),
        "MLS": mls(t).numpy(),
        "Energy": energy(t).numpy(),
        "Mahalanobis": mahalanobis(np.asarray(feats, dtype=np.float64), maha_means, maha_precision),
    }
