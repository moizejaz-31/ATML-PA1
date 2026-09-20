"""
Dataset loading for Task 1.

STL-10: 10 classes, lower computational cost (recommended).
Stratified 80/20 train/val split (seed 6304).
Class-balanced 500-image test subset (seed 6304).
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 6304
STL10_CLASSES = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]


def get_base_transform(size=224):
    """224x224 RGB tensor (before model-specific normalization)."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


class STL10FastDataset(Dataset):
    """
    Fast in-memory STL-10 Dataset backed by pre-cached fast tensor files.
    Loads in <0.2s without downloading large archives.
    """
    def __init__(self, pt_path, transform=None):
        cached = torch.load(pt_path, weights_only=False)
        self.images = cached["images"]
        self.labels = np.array(cached["labels"], dtype=np.int64)
        self.transform = transform
        self.classes = STL10_CLASSES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_arr = self.images[idx]
        if isinstance(img_arr, np.ndarray):
            if img_arr.ndim == 3 and img_arr.shape[0] == 3:
                img_arr = np.transpose(img_arr, (1, 2, 0))
            img = Image.fromarray(img_arr)
        elif isinstance(img_arr, Image.Image):
            img = img_arr
        else:
            img = Image.fromarray(np.array(img_arr))

        target = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, target


def find_fast_file(filename, search_roots):
    """Search multiple standard relative and absolute paths for cached files."""
    for root in search_roots:
        candidate = os.path.normpath(os.path.join(root, filename))
        if os.path.exists(candidate):
            return candidate
    return None


def load_stl10(root="data/stl10"):
    """
    Load STL-10 dataset.
    Prioritizes fast pre-cached .pt files (from PA0/HuggingFace mirror).
    Falls back to torchvision STL10 automatically.
    """
    search_dirs = [
        root,
        os.path.join(os.path.dirname(root)),
        "data",
        "../data",
        "../../data",
        os.path.abspath("pa1-beyond-iid/data"),
        os.path.abspath("../PA0/atml-assignment0/data"),
        os.path.abspath("../../PA0/atml-assignment0/data"),
    ]
    
    train_fast = find_fast_file("stl10_train_fast.pt", search_dirs)
    test_fast = find_fast_file("stl10_test_fast.pt", search_dirs)

    base_tf = get_base_transform()

    if train_fast and test_fast:
        print(f"Loading fast pre-cached STL-10:")
        print(f"  Train: {train_fast}")
        print(f"  Test:  {test_fast}")
        train = STL10FastDataset(train_fast, transform=base_tf)
        test  = STL10FastDataset(test_fast, transform=base_tf)
        return train, test, STL10_CLASSES

    # Fallback to official torchvision STL10
    print(f"Fast cache not found. Using torchvision STL10 from '{root}'...")
    train = datasets.STL10(root, split="train", download=True, transform=base_tf)
    test = datasets.STL10(root, split="test", download=True, transform=base_tf)
    return train, test, STL10_CLASSES


def get_dataset_labels(dataset):
    """Fast label extraction avoiding full image loads."""
    if hasattr(dataset, "labels") and dataset.labels is not None:
        return np.array(dataset.labels)
    if hasattr(dataset, "targets") and dataset.targets is not None:
        return np.array(dataset.targets)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def make_train_val_split(dataset, seed=SEED):
    """Stratified 80/20 train/val split from training partition."""
    labels = get_dataset_labels(dataset)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    tr, va = next(sss.split(range(len(labels)), labels))
    return Subset(dataset, tr.tolist()), Subset(dataset, va.tolist())


def make_eval_subset(dataset, total=500, seed=SEED):
    """Class-balanced subset of test images."""
    labels = get_dataset_labels(dataset)
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    per_class = total // len(classes)
    selected = []
    for c in classes:
        idx = np.where(labels == c)[0]
        n = min(per_class, len(idx))
        selected.extend(rng.choice(idx, n, replace=False).tolist())
    return sorted(selected)
