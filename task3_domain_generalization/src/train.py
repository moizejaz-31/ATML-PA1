"""
Training for Task 3 (Domain Generalization, Sketch UNSEEN).

One loop for ERM / DAN-DG / SAM with the Task 2 protocol:
ResNet-18 (ImageNet), frozen BN running stats, AdamW(1e-4, 1e-4), <= MAX_EPOCHS epochs,
early stopping (PATIENCE; 30 / 5 as in the manual) on mean source-validation macro-F1, 8 images per
source domain per update, seed 6304.

This module never builds a Sketch dataset: `get_source_loaders` only accepts the
three source domains. (ERM itself is NOT retrained - Task 3 reuses the Task 2
source-only checkpoint; `method="erm"` exists only so the SAM/DAN-DG code paths are
directly comparable to plain ERM.)
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from shared.src.pacs_data import PACSDataset, PACS_SOURCE_DOMAINS
from shared.src.seed import log_line
from shared.src.pacs_protocol import TRAIN_TRANSFORM, EVAL_TRANSFORM, freeze_bn_stats, EarlyStopping, MAX_EPOCHS, PATIENCE
from task2_domain_adaptation.src.evaluate import evaluate_source_validation
from task3_domain_generalization.src.dan_dg import pairwise_source_mmd
from task3_domain_generalization.src.sam import SAM

METHODS = ("erm", "dan_dg", "sam")


def get_source_loaders(data_root, split_file, batch_per_domain=8, eval_batch_size=64):
    """Source-only loaders (train + validation). Sketch is deliberately unreachable."""
    with open(split_file) as f:
        split = json.load(f)
    train_loaders, val_loaders = {}, {}
    for d in PACS_SOURCE_DOMAINS:
        assert d != "sketch"
        tr = torch.utils.data.Subset(PACSDataset(data_root, d, TRAIN_TRANSFORM), split[d]["train_idx"])
        va = torch.utils.data.Subset(PACSDataset(data_root, d, EVAL_TRANSFORM), split[d]["val_idx"])
        train_loaders[d] = torch.utils.data.DataLoader(tr, batch_size=batch_per_domain, shuffle=True, drop_last=True)
        val_loaders[d] = torch.utils.data.DataLoader(va, batch_size=eval_batch_size, shuffle=False)
    return train_loaders, val_loaders


def _next(iters, key, loader):
    try:
        return next(iters[key])
    except StopIteration:
        iters[key] = iter(loader)
        return next(iters[key])


def train_dg(model, source_loaders, val_loaders, device, method="erm", lambda_dg=1.0, rho=0.05,
             max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=1e-4, wd=1e-4, save_path=None, verbose=True):
    assert method in METHODS, method
    assert "sketch" not in source_loaders and "sketch" not in val_loaders
    model = model.to(device)
    base_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sam = SAM(base_opt, rho=rho) if method == "sam" else None
    ce = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience)
    domains = list(source_loaders)
    steps_per_epoch = max(len(l) for l in source_loaders.values())

    history = {k: [] for k in ["train_loss", "cls_loss", "mmd_loss", "sam_perturbed_loss", "sam_grad_norm",
                               "val_mean_f1", "val_mean_acc", "val_worst_f1"]}
    history["mmd_pairs"] = {}
    history["val_per_domain"] = {d: {"acc": [], "f1": []} for d in val_loaders}
    best_state, best_f1, best_epoch = copy.deepcopy(model.state_dict()), -float("inf"), 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        freeze_bn_stats(model)
        iters = {d: iter(l) for d, l in source_loaders.items()}
        acc = {"loss": 0.0, "cls": 0.0, "mmd": 0.0, "pert": 0.0, "gn": 0.0}
        pair_acc = {}

        for _ in range(steps_per_epoch):
            xs, ys = zip(*[_next(iters, d, source_loaders[d]) for d in domains])
            sizes = [x.size(0) for x in xs]
            x, y = torch.cat(xs).to(device), torch.cat(ys).to(device)

            logits, f = model(x)
            loss_cls = ce(logits, y)            # domain-balanced batch == mean of the 3 source risks
            loss_mmd = torch.zeros((), device=device)
            if method == "dan_dg":
                loss_mmd, pairs = pairwise_source_mmd(dict(zip(domains, torch.split(f, sizes))))
                for k, v in pairs.items():
                    pair_acc[k] = pair_acc.get(k, 0.0) + v.item()
            # lambda_DG/3 * sum over 3 pairs == lambda_DG * mean over pairs
            loss = loss_cls + lambda_dg * loss_mmd

            if method == "sam":
                sam.zero_grad()
                loss.backward()
                acc["gn"] += sam.first_step()
                loss_pert = ce(model(x)[0], y)  # second pass at theta + eps (BN still frozen)
                sam.zero_grad()
                loss_pert.backward()
                sam.second_step()
                acc["pert"] += loss_pert.item()
            else:
                base_opt.zero_grad(set_to_none=True)
                loss.backward()
                base_opt.step()

            acc["loss"] += loss.item()
            acc["cls"] += loss_cls.item()
            acc["mmd"] += loss_mmd.item()

        n = steps_per_epoch
        history["train_loss"].append(acc["loss"] / n)
        history["cls_loss"].append(acc["cls"] / n)
        history["mmd_loss"].append(acc["mmd"] / n)
        history["sam_perturbed_loss"].append(acc["pert"] / n if method == "sam" else float("nan"))
        history["sam_grad_norm"].append(acc["gn"] / n if method == "sam" else float("nan"))
        for k, v in pair_acc.items():
            history["mmd_pairs"].setdefault(k, []).append(v / n)

        val = evaluate_source_validation(model, val_loaders, device)
        f1s = [val[d]["f1"] for d in val_loaders]
        history["val_mean_f1"].append(val["mean_f1"])
        history["val_mean_acc"].append(val["mean_acc"])
        history["val_worst_f1"].append(float(min(f1s)))
        for d in val_loaders:
            history["val_per_domain"][d]["acc"].append(val[d]["acc"])
            history["val_per_domain"][d]["f1"].append(val[d]["f1"])

        improved = val["mean_f1"] > best_f1
        if improved:
            best_f1, best_epoch = val["mean_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)
        if verbose:
            msg = f"[{method}] ep {epoch:2d} | cls {history['cls_loss'][-1]:.3f}"
            if method == "dan_dg":
                msg += f" | mmd {history['mmd_loss'][-1]:.4f}"
            if method == "sam":
                msg += f" | L(theta+eps) {history['sam_perturbed_loss'][-1]:.3f}"
            msg += f" | val F1 mean {val['mean_f1']:.4f} worst {min(f1s):.4f}{' *' if improved else ''}"
            log_line(msg)
        if stopper.step(val["mean_f1"]):
            if verbose:
                log_line(f"Early stopping at epoch {epoch}; best epoch {best_epoch} (val F1 {best_f1:.4f})")
            break

    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": best_epoch, "best_f1": best_f1}
