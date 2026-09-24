# Task 4: Open-Set Recognition (CIFAR-10 known, CIFAR-100 unknown)

**Known:** all 10 CIFAR-10 classes; stratified 90/10 split of the official train set (seed 6304,
`splits/cifar10_seed6304.json`); official test set = known test.
**Unknown:** fixed CIFAR-100 *test* classes, 800 images per group.
- Near: bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel
- Far: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe

CIFAR-100 training images are never loaded.

**Data download:** CIFAR-10 is downloaded by section 0 of `1_vanilla_baseline.ipynb`, CIFAR-100 by section 1 of
`2_posthoc_scores.ipynb` (torchvision, official archives; skipped if already in `data/`).

## Notebooks: execution order matters

Unknowns may be evaluated only after every checkpoint and score definition is fixed, so run:

1. `1_vanilla_baseline.ipynb`: train Vanilla, CSA, cache known-split features/logits
2. `3_gcsc.ipynb`: Vanilla recipe + `RandAugment(2, 9)` after crop/flip
3. `4_proser.ipynb`: PROSER fine-tuned from Vanilla (5 dummies, β = 1, γ = 0.1, mixup after layer2)
4. `2_posthoc_scores.ipynb`: **first contact with CIFAR-100** (asserts all three checkpoints exist); MSP / MLS / Energy / Mahalanobis on Vanilla
5. `5_evaluation.ipynb`: Vanilla vs GCSC vs PROSER (MLS) + PROSER placeholder score; failure analysis

## Source modules

- `src/resnet_cifar.py`: CIFAR ResNet-18 (3×3 stride-1 stem, no max-pool); `pre_mix` / `forward_from_mid` for manifold mixup; optional dummy head
- `src/data.py`: splits, transforms (`TRAIN_TF`, `GCSC_TF`, `TEST_TF`), unknown groups, loaders
- `src/train.py`: `train_closed_set` (SGD 0.1 / 0.9 / 5e-4, per-step cosine, `CLOSED_SET_EPOCHS` = 100, batch 128), `train_proser` (`PROSER_EPOCHS` = 50, lr 1e-3), `extract_outputs`
- `src/proser.py`: PROSER losses. Augmented classifier [z, max_m d_m]; classifier-placeholder loss (true logit masked → target K+1); data-placeholder loss on layer2 mixup of **different-class** pairs, λ ~ Beta(2, 2); first half of each batch = CE + classifier placeholders, second half = data placeholders (the manual's order)
- `src/scores.py`: MSP, MLS, Energy, Mahalanobis (class means + shared diagonal covariance from unaugmented train features, +1e-6), PROSER placeholder score = softmax([z, max d])_{K+1}
- `src/evaluate.py`: AUROC, τ = 95th percentile of validation unknownness, acceptance/rejection, FPR@95TPR

## Notes

- **Budgets:** Vanilla and GCSC train for 100 epochs, PROSER for 50, as the manual specifies
  (`CLOSED_SET_EPOCHS`, `PROSER_EPOCHS` in `src/train.py`).

- Checkpoints are selected by CIFAR-10 validation accuracy on the 10 known logits only.
- Training uses bf16 autocast for speed; all cached logits/features are computed in fp32, so every
  score sees identical values.
- The PROSER reference implementation (github.com/zhoudw-zdw/CVPR21-Proser) pairs mixup examples by
  random permutation. Here the partner is re-drawn so y_i ≠ y_j, as the manual requires, and the dummy
  response is max-pooled as in the paper.
- Failure cases are tagged "plausible"/"surprising" with a class-semantic mapping written a priori
  (in `5_evaluation.ipynb`), not tuned on results.

## Result files (`results/`)

`posthoc_scores_table.csv` (required table 1), `model_comparison_table.csv` (required table 2),
`model_comparison_deltas.csv`, `failure_cases_vanilla_mls.csv`, `*_csa.json`, figures.
