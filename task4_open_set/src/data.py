"""
CIFAR-10/100 data for Task 4.

Known: CIFAR-10 (10 classes), 90/10 train/val split (seed 6304).
Unknown: Fixed CIFAR-100 test classes (near + far).
"""

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 6304

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

NEAR_UNKNOWNS = ["bus","pickup_truck","motorcycle","tractor","wolf","fox","leopard","camel"]
FAR_UNKNOWNS = ["bottle","bowl","chair","clock","keyboard","mushroom","sunflower","wardrobe"]

TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])

GCSC_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])

TEST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])


def get_cifar10(root="data/cifar10"):
    full = datasets.CIFAR10(root, train=True, download=True, transform=TRAIN_TF)
    test = datasets.CIFAR10(root, train=False, download=True, transform=TEST_TF)
    labels = np.array(full.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr, va = next(sss.split(range(len(labels)), labels))
    val_ds = datasets.CIFAR10(root, train=True, download=False, transform=TEST_TF)
    return Subset(full, tr.tolist()), Subset(val_ds, va.tolist()), test


def get_cifar100_unknowns(root="data/cifar100"):
    ds = datasets.CIFAR100(root, train=False, download=True, transform=TEST_TF)
    c2i = {n: i for i, n in enumerate(ds.classes)}
    near_idx, far_idx = [], []
    for name in NEAR_UNKNOWNS:
        ci = c2i.get(name)
        if ci is not None:
            near_idx.extend([i for i, t in enumerate(ds.targets) if t == ci])
    for name in FAR_UNKNOWNS:
        ci = c2i.get(name)
        if ci is not None:
            far_idx.extend([i for i, t in enumerate(ds.targets) if t == ci])
    return Subset(ds, near_idx), Subset(ds, far_idx)
