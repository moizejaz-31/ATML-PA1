"""
Training engine for Task 2 (Unsupervised Domain Adaptation on PACS).

Every method runs through ONE loop (`train_uda`) so that initialization, source
sampling, augmentation, optimizer and budget are identical; only the alignment
term differs:

    source_only : L = CE(source)
    dan         : L = CE + lambda_MMD * MMD^2(F(x_s), F(x_t))          (fixed lambda)
    dann        : L = CE + CE_dom(D(GRL_alpha(F(x))))                   (unit weight)
    cdan        : L = CE + CE_dom(D(GRL_alpha(vec(F(x) ⊗ softmax(C(F(x)))))))

Protocol (ATML PA1 manual):
- BatchNorm running stats frozen at ImageNet values (gamma/beta trainable)
- AdamW (lr=1e-4, wd=1e-4), <= MAX_EPOCHS epochs, early stop after PATIENCE epochs
  without improvement in mean source-validation macro-F1 (30 / 5 as in the manual,
  set in shared/src/pacs_protocol.py)
- 8 images per source domain + 24 unlabeled target images per update
- GRL schedule alpha(p) = max_alpha * (2 / (1 + exp(-10 p)) - 1), p = step / total_steps
- DANN/CDAN: the discriminator's parameter group uses disc_lr_mult x lr (default 10x,
  the standard practice for the newly added adversarial head in the DANN/CDAN
  reference code). With one shared AdamW at 1e-4 the raw-feature adversarial game
  diverged (feature norms 1e5+, cls loss 1e3+ at alpha~0.1). That failure is visible
  from training losses alone, so the fix uses no target labels. Features are NOT
  L2-normalised before D: normalising hid the norm gap from D, and the classifier then
  mapped most target images to one class.
- DANN/CDAN stabilisers (label-free): with 10x discriminator lr alone, DANN still diverged
  once alpha passed ~0.6 (epoch 3: cls loss 553, feature norm 6e4). A ReLU discriminator's
  logits grow with the feature scale, so through the GRL the backbone can raise the domain
  loss without bound by inflating ALL features. Two remedies close that loop:
    (1) D's input is the feature divided by one detached scalar per step, the mean feature
        norm of the joint source+target batch. Uniform inflation no longer changes what D
        sees, and the reversed gradient shrinks as the scale grows. Unlike per-sample L2
        normalisation, source-vs-target norm DIFFERENCES stay visible to D. (Clamping D's
        logits was tried and rejected: it zeroes D's own gradient once it saturates, and D
        died at exactly ln 2 / 50% accuracy.) The classifier still uses the raw feature.
    (2) gradient-norm clipping (grad_clip) on the backbone+head and on D, which stops the
        sudden gradient spikes that Adam turns into oversized steps.
  Both apply only to the adversarial methods; objective, schedule and weights are unchanged.
- Target labels are NEVER read here. The target diagnostics logged per epoch
  (feature norm, predicted-class histogram, prediction entropy) use predictions only.
"""

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.src.seed import log_line
from shared.src.pacs_protocol import freeze_bn_stats, EarlyStopping, MAX_EPOCHS, PATIENCE
from shared.src.losses import compute_mmd, GradientReversalLayer
from task2_domain_adaptation.src.evaluate import evaluate_source_validation

METHODS = ("source_only", "dan", "dann", "cdan")


def grl_alpha(progress, max_alpha=1.0):
    """Ganin et al. (2016) schedule, scaled by max_alpha (controlled study)."""
    return max_alpha * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def _next(iters, key, loader):
    """Next batch from a loader, restarting (cycling) it when exhausted."""
    try:
        return next(iters[key])
    except StopIteration:
        iters[key] = iter(loader)
        return next(iters[key])


def cdan_features(f, logits):
    """Multilinear conditioning g = vec(f ⊗ p). Neither f nor p is detached (manual)."""
    p = F.softmax(logits, dim=1)
    return torch.bmm(p.unsqueeze(2), f.unsqueeze(1)).flatten(1)   # (B, C*512)


def train_uda(model, source_loaders, val_loaders, device, method="source_only",
              target_loader=None, discriminator=None, lambda_mmd=1.0, max_alpha=1.0,
              max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=1e-4, wd=1e-4, disc_lr_mult=10.0,
              grad_clip=1.0, save_path=None, verbose=True):
    """Unified training loop for Source-only / DAN / DANN / CDAN."""
    assert method in METHODS, method
    uses_target = method != "source_only"
    adversarial = method in ("dann", "cdan")
    if uses_target:
        assert target_loader is not None, f"{method} needs the unlabeled target loader"
    if adversarial:
        assert discriminator is not None, f"{method} needs a domain discriminator"

    model = model.to(device)
    groups = [{"params": model.parameters()}]
    if adversarial:
        discriminator = discriminator.to(device)
        groups.append({"params": discriminator.parameters(), "lr": lr * disc_lr_mult})
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    grl = GradientReversalLayer(max_alpha=max_alpha)
    stopper = EarlyStopping(patience=patience)

    steps_per_epoch = max(len(l) for l in source_loaders.values())
    total_steps = max_epochs * steps_per_epoch
    domains = list(source_loaders.keys())

    keys = ["train_loss", "cls_loss", "align_loss", "dom_acc", "alpha",
            "src_feat_norm", "tgt_feat_norm", "tgt_max_class_share", "tgt_pred_entropy",
            "val_mean_f1", "val_mean_acc"]
    history = {k: [] for k in keys}
    history["val_per_domain"] = {d: {"acc": [], "f1": []} for d in val_loaders}
    history["tgt_pred_hist"] = []

    best_state, best_f1, best_epoch = copy.deepcopy(model.state_dict()), -float("inf"), 0
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        freeze_bn_stats(model)                      # BN running stats stay at ImageNet values
        if adversarial:
            discriminator.train()
        iters = {d: iter(l) for d, l in source_loaders.items()}
        if uses_target:
            iters["__target__"] = iter(target_loader)

        acc = {k: 0.0 for k in ["loss", "cls", "align", "dcorrect", "dtotal", "fs", "ft", "ent"]}
        tgt_hist = np.zeros(7)

        for _ in range(steps_per_epoch):
            global_step += 1
            progress = global_step / total_steps

            xs, ys = zip(*[_next(iters, d, source_loaders[d]) for d in domains])
            x_s, y_s = torch.cat(xs).to(device), torch.cat(ys).to(device)
            n_s = x_s.size(0)
            if uses_target:
                x_t, _ = _next(iters, "__target__", target_loader)   # target labels discarded
                x = torch.cat([x_s, x_t.to(device)])
            else:
                x = x_s

            # One forward pass over source+target. BN is in eval mode, so this is
            # identical to two separate passes.
            logits, f = model(x)
            loss_cls = ce(logits[:n_s], y_s)
            loss_align = torch.zeros((), device=device)

            if method == "dan":
                loss_align = compute_mmd(f[:n_s], f[n_s:])
                loss = loss_cls + lambda_mmd * loss_align
            elif adversarial:
                # D sees the features divided by one detached scalar (the joint batch's mean norm):
                # relative source/target norm differences stay visible, uniform inflation does not pay.
                f_d = f / f.norm(dim=1).mean().detach().clamp_min(1e-6)
                d_in = f_d if method == "dann" else cdan_features(f_d, logits)
                d_logits = discriminator(grl(d_in, progress))
                d_labels = torch.cat([torch.zeros(n_s), torch.ones(x.size(0) - n_s)]).long().to(device)
                loss_align = ce(d_logits, d_labels)          # unit weight
                loss = loss_cls + loss_align
                acc["dcorrect"] += (d_logits.argmax(1) == d_labels).sum().item()
                acc["dtotal"] += d_labels.numel()
            else:
                loss = loss_cls

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if adversarial:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
            optimizer.step()

            acc["loss"] += loss.item()
            acc["cls"] += loss_cls.item()
            acc["align"] += loss_align.item()
            with torch.no_grad():
                acc["fs"] += f[:n_s].norm(dim=1).mean().item()
                if uses_target:
                    acc["ft"] += f[n_s:].norm(dim=1).mean().item()
                    p_t = F.softmax(logits[n_s:], dim=1)
                    acc["ent"] += -(p_t * p_t.clamp_min(1e-12).log()).sum(1).mean().item()
                    tgt_hist += np.bincount(p_t.argmax(1).cpu().numpy(), minlength=7)

        n = steps_per_epoch
        history["train_loss"].append(acc["loss"] / n)
        history["cls_loss"].append(acc["cls"] / n)
        history["align_loss"].append(acc["align"] / n)
        history["dom_acc"].append(acc["dcorrect"] / acc["dtotal"] if acc["dtotal"] else float("nan"))
        history["alpha"].append(grl_alpha(global_step / total_steps, max_alpha) if adversarial else float("nan"))
        history["src_feat_norm"].append(acc["fs"] / n)
        history["tgt_feat_norm"].append(acc["ft"] / n if uses_target else float("nan"))
        history["tgt_pred_entropy"].append(acc["ent"] / n if uses_target else float("nan"))
        share = tgt_hist / tgt_hist.sum() if tgt_hist.sum() else tgt_hist
        history["tgt_pred_hist"].append(share.tolist())
        history["tgt_max_class_share"].append(float(share.max()) if uses_target else float("nan"))

        val = evaluate_source_validation(model, val_loaders, device)
        history["val_mean_f1"].append(val["mean_f1"])
        history["val_mean_acc"].append(val["mean_acc"])
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
            if method == "dan":
                msg += f" | mmd {history['align_loss'][-1]:.3f}"
            if adversarial:
                msg += (f" | dom {history['align_loss'][-1]:.3f} acc {history['dom_acc'][-1]:.2f}"
                        f" | alpha {history['alpha'][-1]:.2f}")
            if uses_target:
                msg += (f" | |f_s| {history['src_feat_norm'][-1]:.1f} |f_t| {history['tgt_feat_norm'][-1]:.1f}"
                        f" | tgt max-class {history['tgt_max_class_share'][-1]:.2f}")
            msg += f" | val F1 {val['mean_f1']:.4f}{' *' if improved else ''}"
            log_line(msg)

        if stopper.step(val["mean_f1"]):
            if verbose:
                log_line(f"Early stopping at epoch {epoch}; best epoch {best_epoch} (val F1 {best_f1:.4f})")
            break

    model.load_state_dict(best_state)
    return {"model": model, "discriminator": discriminator, "history": history,
            "best_epoch": best_epoch, "best_f1": best_f1}


# Thin wrappers kept for notebook readability.
def train_source_only(model, source_loaders, val_loaders, device, **kw):
    return train_uda(model, source_loaders, val_loaders, device, method="source_only", **kw)


def train_dan(model, source_loaders, target_loader, val_loaders, device, lambda_mmd=1.0, **kw):
    return train_uda(model, source_loaders, val_loaders, device, method="dan",
                     target_loader=target_loader, lambda_mmd=lambda_mmd, **kw)


def train_dann(model, discriminator, source_loaders, target_loader, val_loaders, device, max_alpha=1.0, **kw):
    return train_uda(model, source_loaders, val_loaders, device, method="dann", target_loader=target_loader,
                     discriminator=discriminator, max_alpha=max_alpha, **kw)


def train_cdan(model, discriminator, source_loaders, target_loader, val_loaders, device, max_alpha=1.0, **kw):
    return train_uda(model, source_loaders, val_loaders, device, method="cdan", target_loader=target_loader,
                     discriminator=discriminator, max_alpha=max_alpha, **kw)
