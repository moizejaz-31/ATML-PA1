"""
Domain discriminator for DANN and CDAN.

256-unit hidden, ReLU, dropout 0.5, 2-class output.
"""

import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=None, hidden_dim=256, dropout=0.5, input_dim=None, hidden=None):
        super().__init__()
        dim = in_features if in_features is not None else input_dim
        if dim is None:
            raise ValueError("DomainDiscriminator requires in_features or input_dim")
        h_dim = hidden if hidden is not None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 2),
        )

    def forward(self, x):
        return self.net(x)
