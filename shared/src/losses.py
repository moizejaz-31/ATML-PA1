"""
Shared loss functions for Tasks 2 and 3.

- Multi-kernel MMD (used by DAN and DAN-DG)
- Gradient Reversal Layer (used by DANN and CDAN)
"""

import math
import torch
from torch.autograd import Function


# ─── MMD ───

def compute_mmd(source, target, bw_multipliers=(0.5, 1.0, 2.0)):
    """Unbiased multi-kernel MMD^2 between source and target features (U-statistic,
    Gretton et al., 2012): the k(x_i, x_i) self-similarity terms are excluded from the
    within-set averages, so the estimate is ~0 when both sets come from one distribution.

    Kernel: sum of RBF kernels with bandwidths = multiplier * median pairwise squared
    distance of the current combined batch (treated as a constant).

    Why unbiased: the biased V-statistic (plain .mean() including the diagonal) has a floor
    of ~0.30 for 8-vs-8 batches of the SAME domain (~0.10 at 24-vs-24). Minimising that floor
    rewards collapsing each batch's features; with DAN-DG's 8-per-domain pairs this killed
    the representation within ~10 steps (all images got the same prediction).
    """
    n, m = source.size(0), target.size(0)
    combined = torch.cat([source, target], dim=0)
    pw_sq = torch.cdist(combined, combined, p=2).pow(2)
    median_sq = torch.median(pw_sq[pw_sq > 0])
    bws = [mult * median_sq.item() for mult in bw_multipliers]
    K = sum(torch.exp(-pw_sq / (2 * b + 1e-8)) for b in bws)
    k_ss, k_tt, k_st = K[:n, :n], K[n:, n:], K[:n, n:]
    return ((k_ss.sum() - k_ss.diagonal().sum()) / (n * (n - 1))
            + (k_tt.sum() - k_tt.diagonal().sum()) / (m * (m - 1))
            - 2 * k_st.mean())


# ─── Gradient Reversal Layer ───

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradientReversalLayer(torch.nn.Module):
    """GRL with schedule: alpha(p) = 2/(1+exp(-10p)) - 1."""
    def __init__(self, max_alpha=1.0):
        super().__init__()
        self.max_alpha = max_alpha

    def forward(self, x, progress):
        alpha = (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0) * self.max_alpha
        return _GradReverse.apply(x, alpha)
