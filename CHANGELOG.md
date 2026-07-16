# Changelog

## Unreleased — PRX alignment

Brings this repository up to parity with the internal pipeline that produced
the final revised figures on arXiv. The previous state of this repository is
preserved at the git tag `v1.0-arxiv-initial`.

- `FrameDataset`, `encode_dataset`, `rollout_latent`, `rollout_rms`,
  `rollout_theta_rmse_wrapped` gained an `nF_stride` parameter, now defaulting
  to `1` (maximally overlapping windows) instead of non-overlapping windows.
  Pass `nF_stride=num_frames` to recover the old behavior.
- Added `RandomFeatureProber` + `fit_ridge_head` (frozen random expansion,
  closed-form ridge-regression head) as the primary prober used to decode
  physical variables — this is what the shipped `figure_data/` was computed
  with, not the MLP `Prober`.
- `Prober` gained a `num_layers` argument (default `3`, backward compatible).
  A smaller instance (`hdim=128, num_layers=2`) is used as a comparison
  prober alongside the `RandomFeatureProber` in the sweep-vs-kz/N/nF figures.
- `checkpoints/` and `figure_data/` (including `sweep_metrics.csv`) were
  regenerated from the internal "PRX" sweep — see DETAILS.md "Provenance" for
  the exact source run for each checkpoint.
- Added `figure_data/sweep_metrics_mlp128x2.csv`, the companion MLP-128x2
  prober metrics needed to reproduce the "RF vs MLP compare" figures.
- Added `figure_data/prober_ablation.csv` and `reproduce_figures.ipynb`'s
  "Prober Ablation Table" section: a broader architecture ablation (4
  RF-ridge + 5 MLP configs, kz=2 and kz=8) than just the two probers used
  elsewhere, showing why `RF 512 crit softplus` and `MLP 128x2` were chosen.
- `training.py`'s CLI gained a `--nF_stride` flag (default `1`).
- `reproduce_figures.ipynb`'s Supplementary Figures section was overhauled to
  match the internal notebook that produced the final figures: added the
  nF-stride comparison figure, added the RF/MLP overlay to the kz=8 combined
  figure, and added the prober ablation table.
- Relicensed from MIT to GPL-3.0, effective from this commit forward. Earlier
  commits (including the `v1.0-arxiv-initial` tag) remain available under MIT.
- Filled in the real arXiv citation (arXiv:2604.24662) and marked the first
  three authors as equal contribution.
