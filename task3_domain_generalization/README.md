# Task 3: Domain Generalization (PACS, Sketch unseen)

Budget: same as Task 2, ≤ 30 epochs with patience 5 (`shared/src/pacs_protocol.py`), as the manual specifies.

Photo, Art Painting and Cartoon are the only data available to training, source-side diagnostics,
checkpoint selection and hyperparameter selection. **Sketch is loaded only in `4_evaluation.ipynb`**
(final evaluation) and in the last section of `5_controlled_study.ipynb` (analysis only).
`src/train.get_source_loaders` cannot construct a Sketch dataset.

## Notebooks (run in order, after Task 2 notebook 1)

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `1_erm_baseline.ipynb` | Loads the Task 2 Source-only checkpoint **unchanged**; per-domain / mean / worst source-val metrics |
| 2 | `2_dan_dg.ipynb` | DAN-DG: ERM + (λ_DG/3)·Σ_{e<e'} MMD²(F(X_e), F(X_e')), λ_DG = 1, Task 2 MMD kernel |
| 3 | `3_sam.ipynb` | SAM (non-adaptive, ρ = 0.05) wrapping AdamW (1e-4, 1e-4); frozen BN in both passes |
| 4 | `4_evaluation.ipynb` | Main table, source separability, sharpness proxy + loss profile, Sketch, per-class vs ERM and vs Task 2 DAN |
| 5 | `5_controlled_study.ipynb` | DAN-DG λ_DG ∈ {0.1, 1, 10} (expectations stated first; main comparison stays at λ = 1) |

## Source modules

- `src/train.py`: `train_dg` (ERM / DAN-DG / SAM, same loop and protocol as Task 2) and `get_source_loaders`
- `src/dan_dg.py`: mean pairwise source MMD using `shared/src/losses.compute_mmd` (bandwidths from each pair's batch)
- `src/sam.py`: SAM wrapper (ε = ρ g/‖g‖, restore, base step with the gradient at θ + ε)
- `src/evaluate.py`: source-domain separability (multinomial LR, C = 1, balanced, 70/30, seed 6304; chance 33.3%),
  fixed sharpness batch (32 val images per source, seed 6304), Δ_sharp = L(θ+ε) − L(θ) in eval mode

## Result files (`results/`)

`task3_main_table.csv` (required table incl. separability and Δ_sharp), `task3_per_class_sketch.csv`,
`task3_per_class_delta.csv`, `task3_controlled_study.csv`, `<method>_source_side.csv`, figures.
