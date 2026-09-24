# Task 2: Unsupervised Domain Adaptation (PACS → Sketch)

Sources (labeled): `photo`, `art_painting`, `cartoon`. Target (unlabeled during training): `sketch`.
Transductive UDA: all Sketch images are visible *without labels* during adaptation; Sketch labels are
read only after a checkpoint is locked.

## Notebooks (run in order)

| # | Notebook | What it produces |
|---|----------|------------------|
| 1 | `1_source_only.ipynb` | Source-only ERM (also the Task 3 ERM baseline), domain gap, per-class failures |
| 2 | `2_dan.ipynb` | DAN, multi-kernel MMD, fixed λ = 1 |
| 3 | `3_dann.ipynb` | DANN, gradient reversal, α(p) schedule (see "DANN fix" below) |
| 4 | `4_cdan.ipynb` | CDAN, discriminator on vec(f ⊗ p), same adversarial settings as DANN |
| 5 | `5_evaluation.ipynb` | Main table, domain separability, curves, per-class transfer, confusions, t-SNE |
| 6 | `6_controlled_study.ipynb` | DAN λ_MMD ∈ {0.1, 1, 10} |

Each training notebook has `FORCE_TRAIN`; set it to `False` to reload `checkpoints/<method>.pth` and
`checkpoints/<method>_history.json` (the history stores the full config).

## Source modules

- `src/backbone.py`: ResNet-18 (ImageNet V1) returning `(logits, 512-d feature)`
- `src/discriminator.py`: 256-unit hidden layer, ReLU, Dropout(0.5), 2-way output
- `src/train.py`: **one loop, `train_uda`, for all four methods** (same sampling, augmentation, optimizer and budget)
- `src/evaluate.py`: source/target metrics, domain-separability probe, output collection
- `../shared/src/`: PACS data + splits, protocol (transforms, frozen-BN, early stopping), MMD and GRL, plotting style

## Protocol (all methods)

ResNet-18 `IMAGENET1K_V1`, new 7-way head, full fine-tuning; Resize 256×256 → RandomCrop 224 + HFlip
(train), CenterCrop 224 (eval), ImageNet normalisation; **BatchNorm running statistics frozen** (BN
modules in eval mode after `model.train()`; γ/β trainable); AdamW lr 1e-4, wd 1e-4; ≤ 30 epochs; early
stopping after 5 epochs without improvement in mean source-validation macro-F1; 8 images per source
domain + 24 target images per update; seed 6304; stratified 80/20 source splits in
`shared/splits/pacs_sketch_seed6304.json`.

## Design decisions and why

- **Budget:** ≤ 30 epochs with patience 5, as the manual specifies, for every Task 2 and Task 3 run. It is set
  once in `shared/src/pacs_protocol.py` (`MAX_EPOCHS`, `PATIENCE`); the DANN/CDAN reversal schedule α(p) is defined
  over this budget.

- **DANN fix (the earlier 26% result).** The earlier implementation L2-normalised features before the
  discriminator, clamped its logits and clipped gradients. Checked without target labels, that checkpoint
  sent 86% of Sketch images to one class, and its target feature norm drifted *further* from the source
  norm (32 vs 73): the discriminator only saw directions, while the classifier uses raw features. Following
  the manual (raw 512-d feature, no normalisation) with a single AdamW at 1e-4 diverged at α≈0.1
  (feature norms 10⁵+). The fix gives the discriminator its own AdamW parameter group at **10× lr**, the
  standard treatment of the new adversarial head in the DANN/CDAN reference code. Everything else stays
  as specified. It was chosen from training losses, feature norms and target *prediction* histograms only
  (no target labels). CDAN uses the identical setting. Probe logs are summarised in `3_dann.ipynb`.
- **Second DANN/CDAN stabiliser.** With 10x discriminator lr alone, a full DANN run still diverged once the GRL
  strength passed ~0.6 (epoch 3: cls loss 553, feature norm ~6e4). The discriminator now receives F(x) divided by
  the joint batch's mean feature norm (detached), so inflating all features no longer pays; relative
  source/target norm differences stay visible. Gradient-norm clipping (1.0) is added. A stress test ramping
  alpha to 1 stayed stable with a live discriminator (accuracy 26-74%). Clamping D's logits was rejected: it
  zeroed D's gradient and D died at exactly 50%.
- **DAN λ is fixed at 1** (no warm-up ramp), as the manual specifies.
- **Unbiased MMD² estimator** (`shared/src/losses.compute_mmd`, shared with Task 3's DAN-DG): the
  self-similarity terms k(xᵢ, xᵢ) are excluded. The biased version has a positive floor even for two batches
  from the same domain (≈0.10 at 24 vs 24, ≈0.30 at 8 vs 8). With DAN-DG's 8-per-domain pairs, minimising
  that floor collapsed the features within ~10 steps (one prediction for every image). Kernel and
  bandwidths (0.5 / 1 / 2 × batch median) are unchanged.
- **Domain-separability probe:** equal numbers of source-val and Sketch features (1,213 each), seeded
  stratified 70/30 split, `LogisticRegression(C=1, class_weight='balanced')`. Features are
  **standardised on the training split**, because methods' feature norms differ by ~5×, which would
  otherwise change the effective strength of C = 1.
- **Label-free training diagnostics** logged every epoch: source/target feature norms, target
  predicted-class histogram and entropy. They use predictions, never target labels.
- The legacy checkpoints/results trained with `RandomResizedCrop` (before the protocol fix) are kept
  in `checkpoints/legacy_randomresizedcrop/` and `results/legacy_randomresizedcrop/` for reference only.

## Result files (`results/`)

`task2_main_table.csv` (required comparison table), `task2_per_class_target.csv`,
`task2_per_class_delta.csv`, `task2_controlled_study.csv`, `<method>_summary.csv`, plus all figures.
