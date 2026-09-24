"""
OSR evaluation for Task 4.

Convention: u(x) = unknownness (larger = more novel).
Threshold tau = 95th percentile of u on the CIFAR-10 *validation* set; accept x iff u(x) <= tau.
Reported: AUROC (known vs near / far / all), known-test acceptance rate,
near/far rejection rates, and FPR@95TPR = fraction of unknowns accepted at tau.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(known_scores, unknown_scores):
    """Unknowns are the positive class."""
    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    return float(roc_auc_score(y, np.concatenate([known_scores, unknown_scores])))


def roc(known_scores, unknown_scores):
    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([known_scores, unknown_scores]))
    return fpr, tpr


def calibrate_threshold(val_scores, percentile=95):
    """tau such that ~95% of known validation examples are accepted."""
    return float(np.percentile(val_scores, percentile))


def osr_report(val_u, test_u, near_u, far_u, closed_set_acc=None):
    tau = calibrate_threshold(val_u)
    all_u = np.concatenate([near_u, far_u])
    rep = {
        "CSA (%)": None if closed_set_acc is None else 100 * closed_set_acc,
        "AUROC near": 100 * auroc(test_u, near_u),
        "AUROC far": 100 * auroc(test_u, far_u),
        "AUROC all": 100 * auroc(test_u, all_u),
        "tau": tau,
        "Known test accept (%)": 100 * float(np.mean(test_u <= tau)),
        "Near reject (%)": 100 * float(np.mean(near_u > tau)),
        "Far reject (%)": 100 * float(np.mean(far_u > tau)),
        "FPR@95 near (%)": 100 * float(np.mean(near_u <= tau)),
        "FPR@95 far (%)": 100 * float(np.mean(far_u <= tau)),
        "FPR@95 all (%)": 100 * float(np.mean(all_u <= tau)),
    }
    return rep


def per_class_auroc(known_u, unknown_u, unknown_class_names):
    """AUROC of known test vs each individual unknown class."""
    names = np.asarray(unknown_class_names)
    return {c: 100 * auroc(known_u, unknown_u[names == c]) for c in dict.fromkeys(names)}
