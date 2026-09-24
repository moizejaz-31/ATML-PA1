"""
PROSER (Zhou et al., CVPR 2021): classifier placeholders + data placeholders.

Notation: z in R^K are the known-class logits, d in R^M the dummy logits
(K = 10, M = 5). Following the paper, the augmented classifier is
    f_hat(x) = [ z_1, ..., z_K, max_m d_m ]      (K+1 outputs; index K = "unknown")

Each mini-batch is split in two equal halves, in the order the PA1 manual specifies
("first half for classifier-placeholder training, second half for manifold-mixup data
placeholders"):

First half  -> ordinary CE + classifier placeholders:
    l_ce     = CE(f_hat(x), y)
    l_clf    = CE(f_hat(x) with the true logit z_y masked to -inf, K)   weight beta = 1
    i.e. once the ground-truth class is removed, the strongest dummy should be
    the largest remaining response.

Second half -> data placeholders (manifold mixup):
    h_i, h_j = phi_pre(x_i), phi_pre(x_j) (after layer2), y_i != y_j, lam ~ Beta(2, 2)
    h_tilde  = lam h_i + (1 - lam) h_j
    l_data   = CE(f_hat(h_tilde), K)                     weight gamma = 0.1

Total: l_ce + beta * l_clf + gamma * l_data. No CIFAR-100 image is ever used.
Reference implementation: github.com/zhoudw-zdw/CVPR21-Proser (adapted: the mixup
partner is restricted to a different class, as the PA1 manual requires, and the
dummy response is max-pooled as in the paper's Eq.).
"""

import numpy as np
import torch
import torch.nn.functional as F


def augmented_logits(logits, dummy):
    """[z, max_m d_m] -> (B, K+1)."""
    return torch.cat([logits, dummy.max(dim=1, keepdim=True).values], dim=1)


def different_class_partner(y, generator=None):
    """For each i, a random partner j in the batch with y_j != y_i (or -1 if impossible)."""
    B = y.size(0)
    perm = torch.randperm(B, generator=generator, device="cpu").to(y.device)
    partner = perm.clone()
    bad = y[partner] == y
    # Re-draw partners for same-class pairs a few times, then drop the rest.
    for _ in range(10):
        if not bad.any():
            break
        redraw = torch.randint(0, B, (int(bad.sum()),), generator=generator, device="cpu").to(y.device)
        partner[bad] = redraw
        bad = y[partner] == y
    partner[bad] = -1
    return partner


def proser_step_losses(model, x, y, beta=1.0, gamma=0.1, mix_alpha=2.0, rng=None, generator=None):
    """Returns (total_loss, dict of components, known_logits of the CE half, labels of the CE half)."""
    K = model.fc.out_features
    half = x.size(0) // 2
    x_ce, y_ce = x[:half], y[:half]          # first half: CE + classifier placeholders (manual)
    x_mix, y_mix = x[half:], y[half:]        # second half: manifold-mixup data placeholders

    # ---- data placeholders (second half) ----
    lam = float((rng or np.random).beta(mix_alpha, mix_alpha))
    h = model.pre_mix(x_mix)
    j = different_class_partner(y_mix, generator)
    keep = j >= 0
    if keep.any():
        h_tilde = lam * h[keep] + (1.0 - lam) * h[j[keep]]
        z_mix, _, d_mix = model.forward_from_mid(h_tilde)
        target_unknown = torch.full((h_tilde.size(0),), K, dtype=torch.long, device=x.device)
        l_data = F.cross_entropy(augmented_logits(z_mix, d_mix), target_unknown)
    else:                                   # no different-class pair in this half-batch (practically never)
        l_data = h.sum() * 0.0

    # ---- CE + classifier placeholders (first half) ----
    z, _, d = model(x_ce)
    out = augmented_logits(z, d)
    l_ce = F.cross_entropy(out, y_ce)
    masked = out.clone()
    masked[torch.arange(len(y_ce), device=x.device), y_ce] = -1e4   # fp16-safe 'minus infinity'
    l_clf = F.cross_entropy(masked, torch.full_like(y_ce, K))

    total = l_ce + beta * l_clf + gamma * l_data
    comps = {"ce": l_ce.item(), "clf_placeholder": l_clf.item(), "data_placeholder": l_data.item(),
             "lam": lam, "mixed_pairs": int(keep.sum())}
    return total, comps, z.detach(), y_ce
