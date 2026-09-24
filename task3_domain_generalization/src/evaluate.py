"""
Task 3 diagnostics: source-domain separability and the common sharpness proxy.
Neither function touches Sketch.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task2_domain_adaptation.src.evaluate import extract_bottleneck_features

SEED = 6304


def source_separability(model, val_loaders, device, seed=SEED):
    """Multinomial LR (C=1) predicting Photo/Art/Cartoon from frozen features.

    Balanced: the same number of validation features from every source domain
    (the size of the smallest validation split). 70/30 stratified split, seed 6304.
    Features standardised on the training split (as in the Task 2 diagnostic).
    Chance = 33.3%.
    """
    feats = {d: extract_bottleneck_features(model, l, device)[0] for d, l in val_loaders.items()}
    n = min(len(v) for v in feats.values())
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i, d in enumerate(val_loaders):
        idx = rng.choice(len(feats[d]), n, replace=False)
        X.append(feats[d][idx]); y += [i] * n
    X, y = np.concatenate(X), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed))
    clf.fit(Xtr, ytr)                      # lbfgs => multinomial for 3 classes
    return {"source_separability": float(clf.score(Xte, yte)), "n_per_domain": int(n), "test_samples": int(len(yte))}


def fixed_sharpness_batch(val_loaders, per_domain=32, seed=SEED):
    """32 validation images from each source domain, chosen once with seed 6304."""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for d, loader in val_loaders.items():
        ds = loader.dataset
        for i in rng.choice(len(ds), per_domain, replace=False):
            x, y = ds[int(i)]
            xs.append(x); ys.append(y)
    return torch.stack(xs), torch.tensor(ys)


def sharpness_proxy(model, images, labels, rho=0.05):
    """Delta_sharp = L(theta + eps) - L(theta), eps = rho * grad / ||grad||_2, model in eval mode."""
    ce = nn.CrossEntropyLoss()
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    loss_base = ce(model(images)[0], labels)
    loss_base.backward()
    with torch.no_grad():
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        gnorm = torch.norm(torch.stack([g.norm() for g in grads]))
        eps = [rho * g / (gnorm + 1e-12) for g in grads]
        saved = [p.detach().clone() for p in params]
        for p, e in zip(params, eps):
            p.add_(e)
        loss_pert = ce(model(images)[0], labels)
        for p, s in zip(params, saved):     # restore the exact weights (theta + eps - eps != theta in float32)
            p.copy_(s)
    model.zero_grad(set_to_none=True)
    return {"loss": loss_base.item(), "loss_perturbed": loss_pert.item(),
            "delta_sharp": (loss_pert - loss_base).item(), "grad_norm": gnorm.item()}


# ----------------------------------------------------------------------------- extra source-side diagnostics
def pairwise_mmd_matrix(feats_by_domain, n=200, seed=SEED):
    """3x3 matrix of MMD^2 between source domains on frozen validation features (Task 2 kernel).
    `feats_by_domain`: dict domain -> (N, 512) numpy array. n features per domain, seeded."""
    from shared.src.losses import compute_mmd
    rng = np.random.RandomState(seed)
    doms = list(feats_by_domain)
    sub = {d: torch.as_tensor(feats_by_domain[d][rng.choice(len(feats_by_domain[d]), min(n, len(feats_by_domain[d])), replace=False)],
                              dtype=torch.float32) for d in doms}
    M = np.zeros((len(doms), len(doms)))
    for i, a in enumerate(doms):
        for j, b in enumerate(doms):
            if i < j:
                M[i, j] = M[j, i] = compute_mmd(sub[a], sub[b]).item()
    return M, doms


def random_filter_normalized_direction(model, seed=SEED):
    """Random direction with each filter rescaled to the norm of the matching weight filter
    (Li et al., 2018), so landscapes of differently-scaled models are comparable. BN/bias params get 0."""
    g = torch.Generator().manual_seed(seed)
    direction = []
    for p in model.parameters():
        d = torch.randn(p.shape, generator=g).to(p.device)
        if p.dim() <= 1:
            d.zero_()
        else:
            d = d * (p.detach().flatten(1).norm(dim=1) / (d.flatten(1).norm(dim=1) + 1e-10)).view(-1, *[1] * (p.dim() - 1))
        direction.append(d)
    return direction


@torch.no_grad()
def loss_along(model, images, labels, directions, coords):
    """CE on (images, labels) at theta + sum_k c_k d_k for every coordinate tuple in `coords` (eval mode)."""
    ce = nn.CrossEntropyLoss()
    model.eval()
    params = list(model.parameters())
    base = [p.detach().clone() for p in params]
    out = []
    for c in coords:
        for p, b, *ds in zip(params, base, *directions):
            p.copy_(b + sum(ci * d for ci, d in zip(c, ds)))
        out.append(ce(model(images)[0], labels).item())
    for p, b in zip(params, base):
        p.copy_(b)
    return np.array(out)
