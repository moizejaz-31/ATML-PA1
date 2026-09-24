"""
Evaluation and diagnostic utilities for Task 2 (Unsupervised Domain Adaptation).

Includes:
1. Classification metrics (Accuracy, Macro-F1, Per-class accuracy, Confusion Matrix)
2. Source validation evaluation (Photo, Art Painting, Cartoon)
3. Target evaluation (Sketch)
4. Domain Separability Score (Balanced Logistic Regression C=1, 70/30 split, seed 6304)
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 6304


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Compute accuracy, macro-F1, predictions, and ground-truth labels."""
    model = model.to(device)
    model.eval()
    all_preds, all_targets = [], []

    for x, y in dataloader:
        x = x.to(device)
        logits, _ = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return {
        'acc': float(acc),
        'f1': float(f1),
        'preds': all_preds,
        'targets': all_targets
    }


def evaluate_source_validation(model, val_loaders, device):
    """Evaluate model on all 3 source validation splits (Photo, Art, Cartoon)."""
    results = {}
    f1_list, acc_list = [], []

    for domain_name, loader in val_loaders.items():
        res = evaluate_model(model, loader, device)
        results[domain_name] = res
        f1_list.append(res['f1'])
        acc_list.append(res['acc'])

    results['mean_f1'] = float(np.mean(f1_list))
    results['mean_acc'] = float(np.mean(acc_list))
    return results


def evaluate_target(model, target_loader, device, num_classes=7):
    """Evaluate model on the target domain (Sketch)."""
    res = evaluate_model(model, target_loader, device)
    cm = confusion_matrix(res['targets'], res['preds'], labels=list(range(num_classes)))

    # Per-class accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)

    res['per_class_acc'] = per_class_acc.tolist()
    res['confusion_matrix'] = cm
    return res


@torch.no_grad()
def extract_bottleneck_features(model, dataloader, device):
    """Extract 512-dimensional bottleneck representations before classifier head."""
    model = model.to(device)
    model.eval()
    features, labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        _, f = model(x)
        features.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def compute_domain_separability(model, source_val_loaders, target_loader, device, seed=SEED):
    """
    Residual domain information (manual Step 5).

    - Freeze the backbone; collect equal numbers of source-validation and target features
    - 70/30 stratified split with seed 6304
    - Balanced logistic regression, C = 1 (features standardised on the training split,
      because feature scales differ by ~5x across methods and would otherwise change the
      effective strength of the C = 1 penalty)
    - Held-out accuracy = domain separability score (50% = chance)
    Target *images* are used, never target labels.
    """
    src_feats = np.concatenate([extract_bottleneck_features(model, l, device)[0]
                                for l in source_val_loaders.values()])
    tgt_feats, _ = extract_bottleneck_features(model, target_loader, device)
    return domain_separability_from_features(src_feats, tgt_feats, seed)


def domain_separability_from_features(src_feats, tgt_feats, seed=SEED):
    """Same probe as `compute_domain_separability`, on already-extracted features."""
    n = min(len(src_feats), len(tgt_feats))
    rng = np.random.RandomState(seed)
    si = rng.choice(len(src_feats), n, replace=False)
    ti = rng.choice(len(tgt_feats), n, replace=False)
    X = np.concatenate([src_feats[si], tgt_feats[ti]])
    y = np.array([0] * n + [1] * n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed))
    clf.fit(Xtr, ytr)
    return {
        "domain_separability": float(clf.score(Xte, yte)),
        "n_samples_per_domain": int(n),
        "test_samples": int(len(yte)),
    }


@torch.no_grad()
def collect_outputs(model, dataloader, device):
    """Logits, 512-d features and labels for every example in a loader (eval mode)."""
    model = model.to(device).eval()
    L, Fe, Y = [], [], []
    for x, y in dataloader:
        logits, f = model(x.to(device))
        L.append(logits.cpu()); Fe.append(f.cpu()); Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Fe).numpy(), torch.cat(Y).numpy()
