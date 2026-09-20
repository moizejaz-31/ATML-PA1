"""
Domain discriminator for DANN and CDAN.

256-unit hidden, ReLU, dropout 0.5, 2-class output.
"""

import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)
