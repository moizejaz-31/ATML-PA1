# PA1: Beyond IID – Advanced Topics in Machine Learning

**EE-5102/CS-6304 – Fall 2026**

## Overview

This repository contains experiments examining four aspects of learning beyond the IID, closed-set setting:

| Task | Topic | Dataset | Models |
|------|-------|---------|--------|
| 1 | Inductive Biases & Representations | STL-10 | ResNet-50, ViT-B/16, CLIP ViT-B-32 (frozen) |
| 2 | Unsupervised Domain Adaptation | PACS → Sketch | ResNet-18 (fine-tuned) |
| 3 | Domain Generalization | PACS (Sketch unseen) | ResNet-18 (fine-tuned) |
| 4 | Open-Set Recognition | CIFAR-10 / CIFAR-100 | CIFAR-ResNet-18 |

## Repository Structure

```
pa1-beyond-iid/
├── shared/                       # Shared PACS protocol (Tasks 2 & 3), seed, plotting
│   ├── src/
│   └── splits/                   # pacs_sketch_seed6304.json
├── task1_inductive_biases/       # Inductive biases & feature representations
│   ├── src/
│   └── notebooks/
├── task2_domain_adaptation/      # Unsupervised domain adaptation
│   ├── src/
│   └── notebooks/
├── task3_domain_generalization/  # Domain generalization
│   ├── src/
│   └── notebooks/
├── task4_open_set/               # Open-set recognition
│   ├── src/
│   └── notebooks/
└── report/
```

## Setup

```bash
pip install -r requirements.txt
```

Data (not committed, see `.gitignore`):
- `data/PACS/{photo,art_painting,cartoon,sketch}/<class>/*.jpg`
- `data/cifar10/`, `data/cifar100/`: downloaded by the first cells of `task4_open_set/notebooks/1_vanilla_baseline.ipynb` (CIFAR-10) and `2_posthoc_scores.ipynb` (CIFAR-100) via torchvision; skipped if already present. Only the CIFAR-100 **test** split is used.

## Reproducing each task

Notebooks are executed from their own folder (they add the repo root to `sys.path`). Headless:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook>
```

| Task | Order |
|------|-------|
| 2 | `1_source_only` → `2_dan` → `3_dann` → `4_cdan` → `5_evaluation` → `6_controlled_study` |
| 3 | (after Task 2 notebook 1) `1_erm_baseline` → `2_dan_dg` → `3_sam` → `4_evaluation` → `5_controlled_study` |
| 4 | `1_vanilla_baseline` → `3_gcsc` → `4_proser` → `2_posthoc_scores` → `5_evaluation` (unknowns only after all checkpoints are fixed) |

Optional: set the environment variable `PA1_TRAIN_LOG=<file>` to also append per-epoch training lines to a log file (handy for headless runs).

Every training notebook stores its checkpoint plus a `<method>_history.json` with the exact config, and
every reported number is written to a CSV/JSON in the task's `results/` folder.

## Training budgets (as the manual specifies)

| Where | Setting | Value |
|---|---|---|
| Tasks 2 & 3 (`shared/src/pacs_protocol.py`) | `MAX_EPOCHS` / `PATIENCE` | 30 / 5 |
| Task 4 (`task4_open_set/src/train.py`) | `CLOSED_SET_EPOCHS` (Vanilla, GCSC) | 100 |
| Task 4 | `PROSER_EPOCHS` | 50 |

## Global Seed: **6304**

## Key Constraints

- Task 2: Target labels never used during training/model selection (only label-free target diagnostics are logged)
- Task 3: Sketch images never loaded until final evaluation (`task3_domain_generalization/src/train.get_source_loaders` has no Sketch path)
- Task 4: CIFAR-100 images never used during training, checkpoint selection, score design or thresholds
- Tasks 2 & 3: BatchNorm running stats frozen at ImageNet values

## Attribution

- Gradient reversal follows Ganin et al. (2016); CDAN's multilinear conditioning follows Long et al. (2018).
  Using a 10× learning rate for the newly added discriminator mirrors the DANN/CDAN reference implementations.
- SAM follows Foret et al. (2021) (two-step non-adaptive update, written for this repo).
- PROSER losses follow Zhou et al. (2021) and the authors' code (github.com/zhoudw-zdw/CVPR21-Proser),
  adapted as described in `task4_open_set/README.md`.
- Plot colours follow a colour-blind-validated categorical palette (`shared/src/plotting.py`).
- Method schematics (Task 3 protocol, DAN-DG, SAM, CIFAR ResNet-18, OSR pipeline, PROSER) are drawn with matplotlib in `shared/src/diagrams.py`. The SAM panel uses a toy 1-D loss, labelled as an illustration.

## GitHub Repository

<!-- Insert link here -->
