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

# Decoded images shared by every PACSDataset instance (train and eval views of the
# same domain would otherwise each hold their own copy). Keyed by file path.
_IMAGE_CACHE = {}


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
        if path not in _IMAGE_CACHE:
            _IMAGE_CACHE[path] = Image.open(path).convert("RGB")
        img = _IMAGE_CACHE[path]
        if self.transform:
            img = self.transform(img)
        return img, label


def create_stratified_splits(dataset, train_ratio=0.8, seed=SEED):
    """Create stratified 80/20 train/val split."""
    labels = [l for _, l in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(range(len(labels)), labels))
    return train_idx.tolist(), val_idx.tolist()


def get_pacs_dataloaders(data_root, split_file, source_batch_size=8, target_batch_size=24, eval_batch_size=32, num_workers=0):
    """Build train, val, and target dataloaders according to PA1 protocol."""
    import json
    from shared.src.pacs_protocol import TRAIN_TRANSFORM, EVAL_TRANSFORM

    with open(split_file, "r") as f:
        split_data = json.load(f)

    source_train_datasets = {}
    source_val_datasets = {}
    for domain in PACS_SOURCE_DOMAINS:
        full_train = PACSDataset(data_root, domain, transform=TRAIN_TRANSFORM)
        full_val = PACSDataset(data_root, domain, transform=EVAL_TRANSFORM)
        tr_idx = split_data[domain]["train_idx"]
        va_idx = split_data[domain]["val_idx"]
        source_train_datasets[domain] = Subset(full_train, tr_idx)
        source_val_datasets[domain] = Subset(full_val, va_idx)

    target_train_ds = PACSDataset(data_root, PACS_TARGET_DOMAIN, transform=TRAIN_TRANSFORM)
    target_eval_ds = PACSDataset(data_root, PACS_TARGET_DOMAIN, transform=EVAL_TRANSFORM)

    source_train_loaders = {
        d: torch.utils.data.DataLoader(ds, batch_size=source_batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        for d, ds in source_train_datasets.items()
    }
    source_val_loaders = {
        d: torch.utils.data.DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
        for d, ds in source_val_datasets.items()
    }
    target_train_loader = torch.utils.data.DataLoader(target_train_ds, batch_size=target_batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    target_eval_loader = torch.utils.data.DataLoader(target_eval_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    return source_train_loaders, source_val_loaders, target_train_loader, target_eval_loader

