"""
CIFAR-adapted ResNet-18.

Changes from torchvision ResNet-18: 3x3 stride-1 first conv, no initial max-pool,
32x32 inputs. Random initialisation (no ImageNet weights).

PROSER variant: `num_dummy` extra linear "dummy" classifiers on the same 512-d
feature. `forward_from_mid` lets manifold mixup run layer3 -> head on mixed layer2
activations.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetCIFAR(nn.Module):
    FEAT_DIM = 512

    def __init__(self, num_classes=10, num_dummy=0):
        super().__init__()
        base = resnet18(weights=None, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(self.FEAT_DIM, num_classes)
        self.num_dummy = num_dummy
        self.dummy = nn.Linear(self.FEAT_DIM, num_dummy) if num_dummy > 0 else None

    def add_dummies(self, num_dummy):
        """Append randomly initialised dummy classifiers (PROSER)."""
        self.num_dummy = num_dummy
        self.dummy = nn.Linear(self.FEAT_DIM, num_dummy).to(self.fc.weight.device)

    def pre_mix(self, x):
        """phi_pre: input -> output of layer2."""
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer2(self.layer1(x))

    def post_mix(self, h):
        """layer3 -> layer4 -> pooled 512-d feature."""
        return self.avgpool(self.layer4(self.layer3(h))).flatten(1)

    def heads(self, f):
        logits = self.fc(f)
        if self.dummy is not None:
            return logits, f, self.dummy(f)
        return logits, f

    def forward(self, x):
        return self.heads(self.post_mix(self.pre_mix(x)))

    def forward_from_mid(self, h):
        return self.heads(self.post_mix(h))
