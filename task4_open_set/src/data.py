"""
CIFAR-10 / CIFAR-100 data for Task 4 (Open-Set Recognition).

Known   : all 10 CIFAR-10 classes; stratified 90/10 split of the official training
          set (seed 6304) -> train / validation. Official test set = known test.
Unknown : fixed CIFAR-100 *test* classes (800 images per group). CIFAR-100 training
          images are never loaded.
"""

import json
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 6304
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CIFAR10_ROOT = os.path.join(REPO_ROOT, "data", "cifar10")
CIFAR100_ROOT = os.path.join(REPO_ROOT, "data", "cifar100")
SPLIT_FILE = os.path.join(REPO_ROOT, "task4_open_set", "splits", "cifar10_seed6304.json")

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
NEAR_UNKNOWNS = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_UNKNOWNS = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# GCSC = vanilla recipe + RandAugment inserted after crop/flip, before ToTensor/Normalize.
GCSC_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

TEST_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_or_load_split():
    """Stratified 90/10 split of CIFAR-10 train (seed 6304), cached as JSON."""
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            return json.load(f)
    labels = np.array(datasets.CIFAR10(CIFAR10_ROOT, train=True, download=False).targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr, va = next(sss.split(np.zeros(len(labels)), labels))
    split = {"seed": SEED, "train_idx": sorted(tr.tolist()), "val_idx": sorted(va.tolist())}
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f)
    return split


def get_cifar10(train_transform=TRAIN_TF):
    """(train [augmented], train [unaugmented, for Mahalanobis], val, test) datasets."""
    split = make_or_load_split()
    aug = datasets.CIFAR10(CIFAR10_ROOT, train=True, download=False, transform=train_transform)
    plain = datasets.CIFAR10(CIFAR10_ROOT, train=True, download=False, transform=TEST_TF)
    test = datasets.CIFAR10(CIFAR10_ROOT, train=False, download=False, transform=TEST_TF)
    return (Subset(aug, split["train_idx"]), Subset(plain, split["train_idx"]),
            Subset(plain, split["val_idx"]), test)


def get_cifar100_unknowns():
    """Near / far unknown subsets of the CIFAR-100 TEST split (800 images each)."""
    ds = datasets.CIFAR100(CIFAR100_ROOT, train=False, download=False, transform=TEST_TF)
    c2i = {n: i for i, n in enumerate(ds.classes)}
    targets = np.array(ds.targets)
    groups = {}
    for gname, names in [("near", NEAR_UNKNOWNS), ("far", FAR_UNKNOWNS)]:
        ids = [c2i[n] for n in names]                      # KeyError if a name is wrong
        idx = np.where(np.isin(targets, ids))[0].tolist()
        assert len(idx) == 800, (gname, len(idx))
        groups[gname] = Subset(ds, idx)
    return groups["near"], groups["far"], ds.classes


def make_loader(ds, train, batch_size=128, num_workers=None, seed=SEED):
    """Training loaders use 4 worker processes (augmentation is CPU-bound, esp. RandAugment);
    evaluation loaders run in-process (ToTensor/Normalize only) to keep memory bounded."""
    if num_workers is None:
        num_workers = 4 if train else 0
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=False, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0, generator=g if train else None)


def denorm(x):
    """Normalised CHW tensor -> HWC numpy image in [0, 1] for plotting."""
    m, s = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    return (x.detach().cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
