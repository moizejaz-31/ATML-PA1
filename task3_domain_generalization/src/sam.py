"""
SAM: Sharpness-Aware Minimization.

min_theta max_{||eps||<=rho} L_ERM(theta + eps)

rho=0.05. Two forward/backward passes per step.
Frozen BN running stats during both passes.
"""

import torch


class SAM:
    """SAM optimizer wrapper."""

    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho

    @torch.no_grad()
    def first_step(self):
        """Compute and apply ascent perturbation."""
        grad_norm = torch.sqrt(sum(
            p.grad.norm() ** 2 for group in self.base_optimizer.param_groups
            for p in group["params"] if p.grad is not None
        ))
        self.old_params = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.old_params.append(p.data.clone())
                eps = self.rho * p.grad / (grad_norm + 1e-12)
                p.data.add_(eps)

    @torch.no_grad()
    def second_step(self):
        """Restore parameters and apply gradient update."""
        idx = 0
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.old_params[idx]
                idx += 1
        self.base_optimizer.step()
