"""
PACS dataset loader.

4 domains: Photo, Art Painting, Cartoon, Sketch
7 classes: dog, elephant, giraffe, guitar, horse, house, person

Source domains: Photo, Art Painting, Cartoon
Target domain: Sketch
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 6304
PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
PACS_SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
PACS_TARGET_DOMAIN = "sketch"
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
NUM_CLASSES = 7


class PACSDataset(Dataset):
    """PACS dataset for a single domain."""

    def __init__(self, root, domain, transform=None):
        assert domain in PACS_DOMAINS
        self.root = root
        self.domain = domain
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(PACS_CLASSES)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        domain_dir = os.path.join(self.root, self.domain)
        for cls_name in sorted(os.listdir(domain_dir)):
            cls_dir = os.path.join(domain_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx.get(cls_name.lower())
            if label is None:
                continue
            for img in sorted(os.listdir(cls_dir)):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append((os.path.join(cls_dir, img), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def create_stratified_splits(dataset, train_ratio=0.8, seed=SEED):
    """Create stratified 80/20 train/val split."""
    labels = [l for _, l in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(range(len(labels)), labels))
    return train_idx.tolist(), val_idx.tolist()
