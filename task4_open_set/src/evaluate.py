"""
OSR evaluation for Task 4.

AUROC (near/far/all), CSA, FPR@95TPR, threshold calibration.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(known_scores, unknown_scores):
    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    s = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y, s)


def calibrate_threshold(val_scores, percentile=95):
    """95th percentile of unknownness on known validation set."""
    return np.percentile(val_scores, percentile)


def fpr_at_tpr(known_scores, unknown_scores, tpr=0.95):
    tau = np.percentile(known_scores, tpr * 100)
    return float(np.mean(unknown_scores <= tau)), tau
