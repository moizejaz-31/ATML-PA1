"""
ResNet-18 backbone for Tasks 2 and 3.

Extracts 512-dim features. BN running stats frozen at ImageNet values.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    FEAT_DIM = 512

    def __init__(self, num_classes=7):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        self.avgpool = m.avgpool
        self.fc = nn.Linear(self.FEAT_DIM, num_classes)

    def extract_features(self, x):
        return self.avgpool(self.features(x)).flatten(1)

    def forward(self, x):
        f = self.extract_features(x)
        return self.fc(f), f
