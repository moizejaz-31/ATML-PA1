"""Global seed utilities. Seed = 6304."""

import os
import random
import numpy as np
import torch

SEED = 6304


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_line(msg):
    """print + optional append to $PA1_TRAIN_LOG (lets headless notebook runs be monitored)."""
    print(msg, flush=True)
    path = os.environ.get("PA1_TRAIN_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
