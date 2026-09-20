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
    """Multi-kernel MMD^2 between source and target features.
    Bandwidths = multiplier * median pairwise squared distance."""
    combined = torch.cat([source, target], dim=0)
    pw_sq = torch.cdist(combined, combined, p=2).pow(2)
    median_sq = torch.median(pw_sq[pw_sq > 0])
    bws = [m * median_sq.item() for m in bw_multipliers]

    def rbf_sum(x, y):
        d = torch.cdist(x, y, p=2).pow(2)
        return sum(torch.exp(-d / (2 * b + 1e-8)) for b in bws)

    return rbf_sum(source, source).mean() + rbf_sum(target, target).mean() - 2 * rbf_sum(source, target).mean()


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
