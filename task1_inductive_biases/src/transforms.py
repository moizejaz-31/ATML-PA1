"""
Image transforms for Task 1.

All operate on 224x224 RGB tensors (C,H,W in [0,1]) BEFORE
model-specific normalization so same images reuse across models.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

SEED = 6304


def to_grayscale(img):
    """3-channel grayscale."""
    return TF.rgb_to_grayscale(img, num_output_channels=3)


def hue_rotation(img, degrees=90.0):
    factor = (degrees % 360) / 360.0
    if factor > 0.5: factor -= 1.0
    return TF.adjust_hue(img, factor)


def translate(img, delta, direction):
    """Translate with reflection padding + shifted crop."""
    if delta == 0: return img.clone()
    C, H, W = img.shape
    pads = {"up":(0,0,0,delta), "down":(0,0,delta,0), "left":(0,delta,0,0), "right":(delta,0,0,0)}
    starts = {"up":(delta,0), "down":(0,0), "left":(0,delta), "right":(0,0)}
    padded = F.pad(img.unsqueeze(0), pads[direction], mode="reflect").squeeze(0)
    y, x = starts[direction]
    return padded[:, y:y+H, x:x+W]


def shuffle_patches(img, grid=4, seed=SEED, idx=0):
    """4x4 non-identity patch permutation."""
    C, H, W = img.shape
    ph, pw = H // grid, W // grid
    rng = np.random.RandomState(seed + idx)
    n = grid * grid
    perm = np.arange(n)
    while np.all(perm == np.arange(n)):
        rng.shuffle(perm)
    patches = []
    for i in range(grid):
        for j in range(grid):
            patches.append(img[:, i*ph:(i+1)*ph, j*pw:(j+1)*pw])
    out = torch.zeros_like(img)
    for k, src in enumerate(perm):
        i, j = k // grid, k % grid
        out[:, i*ph:(i+1)*ph, j*pw:(j+1)*pw] = patches[src]
    return out
