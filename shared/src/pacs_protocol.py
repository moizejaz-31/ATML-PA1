"""
Shared training protocol for Tasks 2 and 3.

Defines preprocessing, augmentation, BatchNorm freezing,
optimizer settings, and early stopping.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Training constants
LR = 1e-4
WD = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
SRC_BATCH_PER_DOMAIN = 8
TGT_BATCH = 24
SEED = 6304


def get_resnet18(num_classes=7):
    """ResNet-18 with ImageNet weights and new head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_bn_stats(model):
    """Freeze BatchNorm running stats at ImageNet values.
    Scale/bias (gamma/beta) remain trainable."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)


class EarlyStopping:
    """Early stopping (higher metric = better)."""
    def __init__(self, patience=PATIENCE):
        self.patience = patience
        self.best = -float("inf")
        self.counter = 0

    def step(self, score):
        if score > self.best:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
