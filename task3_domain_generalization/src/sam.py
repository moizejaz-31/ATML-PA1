"""
SAM: Sharpness-Aware Minimization (Foret et al., 2021), standard non-adaptive form.

    min_theta max_{||eps||_2 <= rho} L_ERM(theta + eps)

Step 1: eps = rho * g / ||g||_2 (g = gradient at theta), move to theta + eps.
Step 2: compute the gradient at theta + eps, restore theta, and let the base
        optimizer (AdamW, same lr / wd as ERM) apply that gradient.
Two forward/backward passes per update. The caller keeps BatchNorm in eval mode
during both passes (frozen running statistics).
"""

import torch


class SAM:
    """Wraps a base optimizer (AdamW here)."""

    def __init__(self, base_optimizer, rho=0.05):
        assert rho >= 0
        self.base_optimizer = base_optimizer
        self.rho = rho
        self._eps = {}

    def _params(self):
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    yield p

    @torch.no_grad()
    def first_step(self):
        """theta <- theta + rho * g / ||g||."""
        grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in self._params()]), p=2)
        scale = self.rho / (grad_norm + 1e-12)
        self._eps = {}
        for p in self._params():
            e = p.grad * scale
            p.add_(e)
            self._eps[p] = e
        return grad_norm.item()

    @torch.no_grad()
    def second_step(self):
        """Restore theta and update it with the gradient computed at theta + eps."""
        for p, e in self._eps.items():
            p.sub_(e)
        self._eps = {}
        self.base_optimizer.step()

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
