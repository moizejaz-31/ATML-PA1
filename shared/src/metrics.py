"""
Shared evaluation metrics.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def accuracy(y_true, y_pred):
    return accuracy_score(to_np(y_true), to_np(y_pred))


def macro_f1(y_true, y_pred):
    return f1_score(to_np(y_true), to_np(y_pred), average="macro")


def per_class_accuracy(y_true, y_pred, class_names=None):
    y_true, y_pred = to_np(y_true), to_np(y_pred)
    result = {}
    for c in np.unique(y_true):
        mask = y_true == c
        name = class_names[c] if class_names else str(c)
        result[name] = float(np.mean(y_pred[mask] == c))
    return result


def prediction_consistency(preds_clean, preds_transformed):
    return float(np.mean(to_np(preds_clean) == to_np(preds_transformed)))
