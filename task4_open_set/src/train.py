"""
Training and output extraction for Task 4.

Vanilla / GCSC : 10-class CIFAR ResNet-18 from random init, CE, SGD(0.1, m=0.9,
                 wd=5e-4), cosine decay, batch 128, CLOSED_SET_EPOCHS = 100 (manual), seed 6304.
PROSER         : init from the selected Vanilla checkpoint + 5 dummy classifiers,
                 PROSER_EPOCHS = 50 (manual), SGD(1e-3, m=0.9, wd=5e-4), cosine, batch 128.
Checkpoints are selected by CIFAR-10 validation accuracy only (known logits).

Mixed precision is used for speed only (it does not change the recipe): bf16 autocast on GPUs with
native bf16 (compute capability >= 8: RTX 30/40, L4, A100), fp16 autocast + GradScaler otherwise (e.g. T4).
"""

import copy
import os
import time

import numpy as np
import torch

from shared.src.seed import log_line
import torch.nn.functional as F

from task4_open_set.src.proser import proser_step_losses

def _native_bf16():
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:
        return False


_NATIVE_BF16 = _native_bf16()
AMP_DTYPE = torch.bfloat16 if _NATIVE_BF16 else torch.float16
AMP = dict(device_type="cuda", dtype=AMP_DTYPE)


def _make_scaler():
    """Loss scaling is only needed (and only enabled) for fp16."""
    return torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)

# Manual budgets: 100 epochs for Vanilla/GCSC, 50 for PROSER.
CLOSED_SET_EPOCHS = 100
PROSER_EPOCHS = 50


@torch.no_grad()
def val_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        with torch.autocast(**AMP):
            logits = model(x.to(device, non_blocking=True))[0]
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total += y.numel()
    return correct / total


def _sgd_cosine(model, lr, epochs, steps_per_epoch):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)  # per-step cosine
    return opt, sched


def train_closed_set(model, train_loader, val_loader, device, epochs=CLOSED_SET_EPOCHS, lr=0.1, save_path=None,
                     verbose=True, log_every=1):
    model = model.to(device).to(memory_format=torch.channels_last)
    opt, sched = _sgd_cosine(model, lr, epochs, len(train_loader))
    scaler = _make_scaler()
    log_line(f"mixed precision: {AMP_DTYPE}")
    hist = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_acc, best_epoch, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tl = tc = tn = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(**AMP):
                logits = model(x)[0]
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * y.size(0); tc += (logits.argmax(1) == y).sum().item(); tn += y.size(0)
        va = val_accuracy(model, val_loader, device)
        hist["train_loss"].append(tl / tn); hist["train_acc"].append(tc / tn)
        hist["val_acc"].append(va); hist["lr"].append(sched.get_last_lr()[0])
        if va > best_acc:
            best_acc, best_epoch = va, ep
            best_state = copy.deepcopy(model.state_dict())
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)
        if verbose and (ep % log_every == 0 or ep == epochs):
            log_line(f"ep {ep:3d} | loss {tl/tn:.3f} | train acc {tc/tn:.4f} | val acc {va:.4f}"
                  f"{' *' if best_epoch == ep else ''} | {time.time()-t0:.0f}s")
    model.load_state_dict(best_state)
    return {"model": model, "history": hist, "best_epoch": best_epoch, "best_val_acc": best_acc}


def train_proser(model, train_loader, val_loader, device, epochs=PROSER_EPOCHS, lr=1e-3, beta=1.0, gamma=0.1,
                 seed=6304, save_path=None, verbose=True, log_every=1):
    """`model` must already carry the vanilla weights and 5 dummy classifiers."""
    model = model.to(device).to(memory_format=torch.channels_last)
    opt, sched = _sgd_cosine(model, lr, epochs, len(train_loader))
    scaler = _make_scaler()
    log_line(f"mixed precision: {AMP_DTYPE}")
    rng = np.random.RandomState(seed)
    gen = torch.Generator().manual_seed(seed)
    keys = ["loss", "ce", "clf_placeholder", "data_placeholder"]
    hist = {k: [] for k in keys + ["train_acc", "val_acc", "dummy_top_share"]}
    best_acc, best_epoch, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        sums = {k: 0.0 for k in keys}; n = 0; tc = tn = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(**AMP):
                loss, comps, z, y_ce = proser_step_losses(model, x, y, beta=beta, gamma=gamma, rng=rng, generator=gen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums["loss"] += loss.item()
            for k in keys[1:]:
                sums[k] += comps[k]
            n += 1
            tc += (z.argmax(1) == y_ce).sum().item(); tn += y_ce.numel()
        va = val_accuracy(model, val_loader, device)
        for k in keys:
            hist[k].append(sums[k] / n)
        hist["train_acc"].append(tc / tn); hist["val_acc"].append(va)
        if va > best_acc:
            best_acc, best_epoch = va, ep
            best_state = copy.deepcopy(model.state_dict())
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)
        if verbose and (ep % log_every == 0 or ep == epochs):
            log_line(f"ep {ep:3d} | total {sums['loss']/n:.3f} | ce {sums['ce']/n:.3f} | clf-ph {sums['clf_placeholder']/n:.3f}"
                  f" | data-ph {sums['data_placeholder']/n:.3f} | val acc {va:.4f}{' *' if best_epoch == ep else ''}"
                  f" | {time.time()-t0:.0f}s")
    model.load_state_dict(best_state)
    return {"model": model, "history": hist, "best_epoch": best_epoch, "best_val_acc": best_acc}


@torch.no_grad()
def extract_outputs(model, loader, device):
    """Penultimate features (fp32), known logits and (PROSER) dummy logits for a loader."""
    model = model.to(device).eval()
    feats, logits, dummies, labels = [], [], [], []
    for x, y in loader:
        out = model(x.to(device))                       # fp32 so all scores share exact values
        logits.append(out[0].float().cpu()); feats.append(out[1].float().cpu())
        if len(out) == 3:
            dummies.append(out[2].float().cpu())
        labels.append(y)
    res = {"logits": torch.cat(logits).numpy(), "feats": torch.cat(feats).numpy(), "labels": torch.cat(labels).numpy()}
    if dummies:
        res["dummy"] = torch.cat(dummies).numpy()
    return res
