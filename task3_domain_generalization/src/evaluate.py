"""
Task 3 evaluation: source separability, sharpness proxy, Sketch evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


def source_separability(domain_feats, domain_labels, seed=SEED):
    """Multinomial logistic regression to predict P/A/C. Chance=33.3%."""
    Xtr, Xte, ytr, yte = train_test_split(
        domain_feats, domain_labels, test_size=0.3, random_state=seed, stratify=domain_labels
    )
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, multi_class="multinomial")
    clf.fit(Xtr, ytr)
    return clf.score(Xte, yte)


def sharpness_proxy(model, images, labels, rho=0.05):
    """Delta_sharp = L(theta+eps) - L(theta), eps = rho * grad/||grad||."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.zero_grad()
    for p in model.parameters(): p.requires_grad_(True)
    logits, _ = model(images)
    loss_base = criterion(logits, labels)
    loss_base.backward()
    grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
    old = {n: p.data.clone() for n, p in model.named_parameters() if p.grad is not None}
    for n, p in model.named_parameters():
        if p.grad is not None:
            p.data += rho * p.grad / (grad_norm + 1e-12)
    with torch.no_grad():
        loss_pert = criterion(model(images)[0], labels)
    for n, p in model.named_parameters():
        if n in old: p.data = old[n]
    return (loss_pert - loss_base).item()
