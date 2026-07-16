# DySIB Pendulum

Code and data for the paper:

> **Information bottleneck for learning the phase space of dynamics from high-dimensional experimental data**
> K. Michael Martini\*, Eslam Abdelaleem\*, Paarth Gulati\*, and Ilya Nemenman. arXiv:2604.24662, 2026.
> [arxiv.org/abs/2604.24662](https://arxiv.org/abs/2604.24662)
>
> \*Equal contribution.

**Note:** this repository was recently updated to match the pipeline that produced the paper's final figures (it previously reflected an earlier, pre-revision snapshot). See [Provenance](#provenance) below for what changed, or the [`v1.0-arxiv-initial`](https://github.com/paarthgulati/DYSIB_Pendulum/releases/tag/v1.0-arxiv-initial) tag for the prior version.

---

## What this repository does

**DySIB** (**Dy**namical **S**ymmetric **I**nformation **B**ottleneck) is a method that learns a compressed, low-dimensional representation of a dynamical system's state directly from high-dimensional observations — without labels or prior knowledge of the physics. Given a sequence of frames showing the system evolving over time, DySIB finds the coordinates that carry maximal information about the future, and those coordinates turn out to correspond to the true physical variables of the system.

This repository demonstrates DySIB on a rigid pendulum: the input is a video of the pendulum (128×128 RGB frames), and the method discovers 2 latent dimensions that align with angle and angular velocity. The same training code can be applied to other dynamical systems by providing trajectory data in the same format (see [DETAILS.md](DETAILS.md) for how to adapt the data pipeline).

The repository provides:
- Pretrained model checkpoints (~4.5 MB, shipped in `checkpoints/`)
- Precomputed analysis arrays for every figure (~7 MB, shipped in `figure_data/`)
- A single notebook that regenerates every figure in the paper — no GPU, no data download required
- The full training code to retrain from scratch
- The raw pendulum video dataset (downloaded automatically on first use, only needed for retraining)

---

## Quickstart: reproduce the paper figures (no training needed)

```bash
git clone https://github.com/paarthgulati/DYSIB_Pendulum.git
cd DYSIB_Pendulum
pip install -r requirements.txt
jupyter lab reproduce_figures.ipynb
```

The notebook loads pretrained models and precomputed data files that ship with the repository. It runs in under a minute on any laptop — no GPU, no downloading data. Figures are saved to `Figures/`.

---

## Installation

Requires **Python ≥ 3.9**.

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended for training but is not needed to reproduce figures.

---

## Running the full pipeline from scratch

### Step 1 — Download and preprocess the data

```bash
python data.py --all
```

This fetches the [Chen et al. (2022)](https://zenodo.org/records/6653856) pendulum video dataset from Zenodo (~167 MB), extracts 1200 trajectories (60 frames each at 128×128 RGB), downsamples each frame to 28×28 grayscale, and saves `data/pendulum_preprocessed.npz` (~227 MB).

### Step 2 — Train a model

```bash
python training.py \
    --kz 2 --num_frames 2 --nF_stride 1 --samples_n 1000 \
    --rep 0 --epochs 300 \
    --out checkpoints/my_run.pt
```

Key options:

| Flag | Meaning | Paper default |
|------|---------|--------------|
| `--kz` | Latent dimension | 2 |
| `--num_frames` | Frames per input window | 2 |
| `--nF_stride` | Temporal offset between past/future window starts. 1 = maximally overlapping (used for the final paper figures); pass a value equal to `--num_frames` for non-overlapping windows | 1 |
| `--samples_n` | Training trajectories (max 1000) | 1000 |
| `--rep` | Random seed | varies |
| `--epochs` | Training epochs | 300 |
| `--gamma` | KL compression weight γ; together with `--alpha`, sets β = α/γ | 0.01 |
| `--alpha` | MI maximisation weight α | 1.0 |

Training prints mutual information (MI) and loss per epoch. On CPU, 300 epochs takes roughly 20–30 minutes; on a GPU it is much faster.

### Step 3 — Regenerate analysis files

```bash
python precompute_figure_data.py
```

This loads the checkpoints in `checkpoints/`, encodes the dataset, trains linear probers (a `RandomFeatureProber` fit by closed-form ridge regression — see [DETAILS.md](DETAILS.md)), and writes the precomputed arrays consumed by the notebook to `figure_data/`. To use your own trained checkpoint instead of the shipped ones, update the `CKPT_KZ2` / `CKPT_KZ8` / `CKPT_KZ2_N64` / `CKPT_KZ2_N256` paths at the top of `precompute_figure_data.py`.

### Step 4 — Plot the figures

```bash
jupyter lab reproduce_figures.ipynb
```

---

## Repository layout

```
DYSIB_Pendulum/
├── models.py                   DYSIB, Encoder, DeltaDecoder, Prober, FrameDataset
├── training.py                 Training loop + prober training + encoding + rollout + CLI
├── data.py                     Download + extract + preprocess + load + CLI
├── precompute_figure_data.py   Compute figure_data/ from checkpoints + data
├── reproduce_figures.ipynb     Generate all paper figures from figure_data/
├── requirements.txt
├── LICENSE                     GPL-3.0
│
├── checkpoints/                Trained model weights (shipped, ~4.5 MB total)
│   ├── kz2_nF2_str1_N1000_rep3.pt   Main embedding figures (kz=2, 1000 training trials)
│   ├── kz8_nF2_str1_N1000_rep1.pt   Intrinsic-dimensionality figure (kz=8)
│   ├── kz2_nF2_str1_N64_rep3.pt
│   └── kz2_nF2_str1_N256_rep3.pt    Scalability comparison (N = 64, 256)
│
├── figure_data/                Precomputed arrays for the notebook (shipped, ~7 MB)
│   ├── sweep_metrics.csv           RF-ridge prober metrics from the hyperparameter sweep
│   ├── sweep_metrics_mlp128x2.csv  Companion MLP-128x2 prober metrics (theta/omega), for the "RF vs MLP compare" figures
│   ├── prober_ablation.csv         Prober architecture ablation (4 RF-ridge + 5 MLP configs, kz=2 and kz=8)
│   ├── phase_portrait.npz
│   ├── embedding_kz2.npz
│   ├── embedding_vs_N.npz
│   ├── rollout_test.npz
│   ├── pca_kz8.npz
│   └── cylindrical_phys.npz
│
├── data/                       Local data cache (git-ignored, created on first run)
│   ├── single_pendulum.zip     Raw archive from Zenodo
│   ├── single_pendulum/        Extracted frames (1200 trajectories × 60 PNG files)
│   └── pendulum_preprocessed.npz  Preprocessed frames + physical labels (~227 MB)
│
└── Figures/                    Output directory for saved figures (git-ignored)
```

---

## Using the code programmatically

Every function used in the pipeline is importable directly:

```python
import torch
from models   import DYSIB, Prober, FrameDataset
from training import train_dysib, train_prober, encode_dataset, rollout_latent
import data as data_mod

# Load data
d = data_mod.load()                       # downloads + preprocesses on first call
frames = torch.from_numpy(d["frames"])    # (1200, 60, 784)

# Build and train a model
model = DYSIB(input_dim=784, dz=2)
model, history = train_dysib(model, loader, x_test, y_test, epochs=300)

# Encode the dataset and roll out latent dynamics (nF_stride defaults to 1,
# maximally overlapping windows — matches the paper's final figures)
zx, numD = encode_dataset(model, frames, num_frames=2)
rollout   = rollout_latent(model, zx, num_frames=2, numD=numD)
```

See [DETAILS.md](DETAILS.md) for full function signatures, architecture details, checkpoint format, and notes on extending the code.

---

## Citation

If you use this code or the trained models, please cite:

```bibtex
@article{martini2026dysib,
  title         = {Information bottleneck for learning the phase space of dynamics from high-dimensional experimental data},
  author        = {Martini, K. Michael and Abdelaleem, Eslam and Gulati, Paarth and Nemenman, Ilya},
  journal       = {arXiv preprint arXiv:2604.24662},
  eprint        = {2604.24662},
  archivePrefix = {arXiv},
  year          = {2026},
  note          = {K. M. Martini, E. Abdelaleem, and P. Gulati contributed equally to this work.},
}
```

The raw video dataset is from:

```bibtex
@article{chen2022discovering,
  title   = {Automated discovery of fundamental variables hidden in experimental data},
  author  = {Chen, Boyuan and Huang, Kuang and Raghupathi, Sunand and
             Chandratreya, Ishaan and Du, Qiang and Lipson, Hod},
  journal = {Nature Computational Science},
  year    = {2022},
}
```

---

## Provenance

The shipped checkpoints and `figure_data/` were regenerated from the internal
sweep that produced the final revised figures on arXiv (`nF_stride=1`
overlapping windows; `RandomFeatureProber` + closed-form ridge regression as
the primary prober, compared against an MLP-128×2 prober in the sweep
comparison figures — see [DETAILS.md](DETAILS.md)). The prior version of this
repository (non-overlapping windows, MLP-only prober) is preserved at the git
tag [`v1.0-arxiv-initial`](https://github.com/paarthgulati/DYSIB_Pendulum/releases/tag/v1.0-arxiv-initial)
for anyone who needs to reproduce the original numbers.

---

## License

GPL-3.0 — see [LICENSE](LICENSE). (Earlier versions of this repository, including the `v1.0-arxiv-initial` tag, were released under MIT; that license still applies to the code as it existed at that point.)
