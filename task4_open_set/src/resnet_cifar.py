"""
CIFAR-adapted ResNet-18.

Changes from standard ResNet-18:
- 3×3 stride-1 conv (instead of 7×7 stride-2)
- No initial max-pooling
- Operates on 32×32 images
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetCIFAR(nn.Module):
    FEAT_DIM = 512

    def __init__(self, num_classes=10, num_dummy=0):
        super().__init__()
        base = resnet18(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(self.FEAT_DIM, num_classes)
        self.num_dummy = num_dummy
        if num_dummy > 0:
            self.dummies = nn.ModuleList([nn.Linear(self.FEAT_DIM, 1) for _ in range(num_dummy)])

    def extract_features(self, x, return_mid=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        mid = x if return_mid else None
        x = self.layer3(x)
        x = self.layer4(x)
        f = self.avgpool(x).flatten(1)
        return (f, mid) if return_mid else f

    def forward_from_mid(self, h):
        x = self.layer3(h)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x):
        f = self.extract_features(x)
        logits = self.fc(f)
        if self.num_dummy > 0:
            dummy = torch.cat([d(f) for d in self.dummies], dim=1)
            return logits, f, dummy
        return logits, f
