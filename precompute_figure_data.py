"""
Precompute analysis arrays used by reproduce_figures.ipynb.

Loads the shipped checkpoints + preprocessed data, encodes, trains probers,
rolls out dynamics, runs intrinsic-dimensionality estimators, and writes
one .npz / .csv per figure into figure_data/.

The reproduction notebook consumes only these files (no torch required).

Usage:
    python precompute_figure_data.py              # regenerate everything
    python precompute_figure_data.py --only embedding_kz2 rollout_test
    python precompute_figure_data.py --force      # overwrite existing outputs

Outputs
-------
figure_data/sweep_metrics.csv       Flat table of (kz, nF, N, rep, test/train metrics), RF-ridge prober.
figure_data/sweep_metrics_mlp128x2.csv  Companion MLP-128x2 prober metrics (theta/omega only),
                                     for the "RF vs MLP compare" figures.
figure_data/prober_ablation.csv     Prober architecture ablation (4 RF-ridge + 5 MLP configs,
                                     kz=2 and kz=8, 10 reps), incl. precomputed trainable-param counts.
figure_data/phase_portrait.npz      theta_wrapped, omega, latent Z for kz=2 run.
figure_data/embedding_kz2.npz       Subsampled Z + physical vars for kz=2 embedding scatter.
figure_data/embedding_vs_N.npz      Z + theta for N ∈ {64, 256, 1000}.
figure_data/rollout_test.npz        RMSE curves, quiver field, fixed points, vis trajectories.
figure_data/pca_kz8.npz             SVD, PR, TwoNN, MLE(k), PC1/PC2 projection.
figure_data/cylindrical_phys.npz    theta_wrapped_deg, omega_deg_per_s.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import torch

import data as data_mod
from models import DYSIB, Prober, RandomFeatureProber
from training import (
    encode_dataset,
    fit_ridge_head,
    predict_prober,
    train_prober,
)

# Prober config for the primary (RF-ridge) prober — matches the sweep that
# produced the final arxiv figures.
RF_EXPAND_DIM   = 512
RF_WEIGHT_INIT  = "softplus_critical"
RF_RIDGE_ALPHA  = 1e-3

# ─── config ─────────────────────────────────────────────────────────────────
CKPT_DIR    = Path("checkpoints")
FIGDATA_DIR = Path("figure_data")

# Runs that have shipped checkpoints (figures that visualize embeddings).
# All four are sourced from the "PRX" sweep (nF_stride=1) that produced the
# final arxiv figures — see DETAILS.md "Provenance" section.
CKPT_KZ2          = CKPT_DIR / "my_run.pt"
CKPT_KZ8          = CKPT_DIR / "kz8_nF2_str1_N1000_rep1.pt"
CKPT_KZ2_N64      = CKPT_DIR / "kz2_nF2_str1_N64_rep3.pt"
CKPT_KZ2_N256     = CKPT_DIR / "kz2_nF2_str1_N256_rep3.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Source H5 files for sweep metrics — consulted ONCE, producing sweep_metrics.csv
# and sweep_metrics_mlp128x2.csv. Users cloning the repo read the shipped .csv
# files; these paths are only relevant if you want to re-aggregate your own
# sweep from the internal "PRX" HDF5 sweep files (nF_stride=1, RF-ridge +
# MLP-128x2 probers) that produced the final arxiv figures.
# Set to your own directory of sweep .h5 files if needed.
PRX_H5_DIR = Path("/path/to/your/results_prx")   # expects results_{kz,nf,N,N_kz8,nf_stride_nf}.h5

# The MLP-128x2 comparison prober is stored in a separate companion file per
# sweep axis (not one uniform directory) — traced directly from the notebook
# that produced the final figures (paper_figures_prx_compare.ipynb, cell 2).
PRX_MLP_H5_FILES = {
    "kz":  PRX_H5_DIR / "prober_mlp" / "probers_kz.h5",
    "nf":  PRX_H5_DIR / "prober_mlp" / "probers_nf.h5",
    "N":   PRX_H5_DIR / "prober_mlp_ep100" / "probers_N.h5",       # 100 epochs, sufficient for kz<=8
    "kz8": PRX_H5_DIR / "prober_mlp_kz8_ep200" / "probers_N.h5",   # 200 epochs — kz=8 needed longer to converge
}

# Prober architecture ablation (broader than the two probers used elsewhere):
# 4 RF-ridge variants (expand_dim/init/activation) + 5 MLP variants (hdim x
# num_layers), each evaluated at a fixed run (nF=2, N=1000, stride=1) for both
# kz=2 and kz=8. Source: paper_figures_prx_compare.ipynb cell 47.
PROBER_ABLATION_H5 = {
    2: {
        "ridge": PRX_H5_DIR / "prober_ablation" / "kz2" / "probers_kz_ridge.h5",
        "mlp":   PRX_H5_DIR / "prober_ablation" / "kz2" / "probers_kz.h5",
    },
    8: {
        "ridge": PRX_H5_DIR / "prober_ablation" / "kz8" / "probers_kz_ridge.h5",
        "mlp":   PRX_H5_DIR / "prober_ablation" / "kz8" / "probers_kz.h5",
    },
}


# ─── helpers ────────────────────────────────────────────────────────────────
def _load_checkpoint(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    p  = dict(ck["params"])
    # Older checkpoints (pre-PRX) predate nF_stride and were trained with
    # non-overlapping windows (nF_stride == num_frames), not the published
    # default of 1 used going forward.
    p.setdefault("nF_stride", p["num_frames"])
    # Provenance fields (run_name / source_h5) live alongside "params" in the
    # checkpoint dict, not inside it — fold them in here so callers have one
    # place to look. Falls back to the filename for checkpoints saved before
    # this provenance convention existed.
    p.setdefault("run_name", ck.get("run_name", path.stem))
    p.setdefault("source_h5", ck.get("source_h5"))
    model = DYSIB(
        input_dim    = int(p["input_dim"]),
        dz           = int(p["ndims"]),
        dzf          = 32,
        num_frames   = int(p["num_frames"]),
        gamma        = 0.01,
        alpha        = 1.0,
        hidden       = 256,
        decoder_hdim = 64,
    ).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, p


def _fvars_flat(phys_arrays, num_frames, nF_stride=1):
    """Flatten physical variables to match (trial, window) ordering from FrameDataset.

    Each trial contributes numD = T - nF_stride - num_frames + 1 windows, indexed by
    the FIRST past-window frame, which is at global frame index 0..numD-1.
    """
    T = phys_arrays["theta"].shape[1]
    numD = T - nF_stride - num_frames + 1
    def _flat(a):
        return a[:, :numD].reshape(-1).astype(np.float32)
    return {
        "theta":    _flat(phys_arrays["theta"]),
        "omega":    _flat(phys_arrays["vel_theta"]),
        "kinetic":  _flat(phys_arrays["kinetic_energy"]),
        "pot":      _flat(phys_arrays["potential_energy"]),
        "total":    _flat(phys_arrays["total_energy"]),
    }, numD


def _wrap_deg(theta_rad):
    """Wrap theta in radians to degrees in (-180, 180]."""
    d = np.rad2deg(theta_rad)
    return ((d + 180) % 360) - 180


# ─── tasks ──────────────────────────────────────────────────────────────────
def task_sweep_metrics(out, force=False):
    """Flatten per-run RF-ridge prober metrics across all PRX H5 sweep files into one CSV.

    Each H5 run stores:
      runs/<run_name>/
        .attrs             : ndims, num_frames, nF_stride, samples_n, rep,
                             gamma, epochs, batch_size, prober_type, prober_config
        probers/.attrs     : theta_{train,test}_deg, omega_{train,test}_rmse,
                             kinetic_{...}, pot_{...}   (RF-ridge prober)
        training/test_mi   : (epochs,) float32  — take the last value as final MI
        training/train_mi  : (epochs,) float32

    See sweep_metrics_mlp128x2.csv (task_sweep_metrics_mlp128x2) for the
    companion MLP-128x2 prober used in the "RF vs MLP compare" figures.
    """
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    try:
        import h5py
        import pandas as pd
    except ImportError as e:
        raise ImportError(f"h5py + pandas required for sweep_metrics: {e}")

    param_keys  = ["ndims", "num_frames", "nF_stride", "samples_n", "rep",
                   "gamma", "epochs", "batch_size"]
    prober_keys = ["theta_test_deg", "theta_train_deg",
                   "omega_test_rmse", "omega_train_rmse",
                   "kinetic_test_rmse", "kinetic_train_rmse",
                   "pot_test_rmse", "pot_train_rmse"]

    rows = []
    for h5 in sorted(PRX_H5_DIR.glob("*.h5")):
        try:
            with h5py.File(h5, "r") as f:
                runs = f.get("runs")
                if runs is None:
                    continue
                for run_name, grp in runs.items():
                    row = {"run_name": run_name, "source_h5": h5.name,
                           "prober_type": "rf_ridge"}
                    for k in param_keys:
                        if k in grp.attrs:
                            row[k] = grp.attrs[k]
                    if "probers" in grp:
                        for k in prober_keys:
                            if k in grp["probers"].attrs:
                                row[k] = grp["probers"].attrs[k]
                    if "training" in grp:
                        if "test_mi" in grp["training"]:
                            row["final_test_mi"]  = float(grp["training"]["test_mi"][-1])
                        if "train_mi" in grp["training"]:
                            row["final_train_mi"] = float(grp["training"]["train_mi"][-1])
                    rows.append(row)
        except OSError:
            continue

    df = pd.DataFrame(rows)
    # De-duplicate (same run appearing in multiple H5 files) — keep first.
    # nF_stride MUST be part of the key: a stride=1 run and a stride=num_frames
    # run can otherwise share identical (ndims, num_frames, samples_n, rep) —
    # e.g. results_nf.h5 (stride=1) vs results_nf_stride_nf.h5 (stride=num_frames) —
    # and are genuinely different runs, not duplicates of each other.
    dedup_keys = ["ndims", "num_frames", "nF_stride", "samples_n", "rep"]
    if set(dedup_keys).issubset(df.columns):
        df = df.drop_duplicates(subset=dedup_keys, keep="first").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  → {out}  ({len(df)} unique runs)")


def task_sweep_metrics_mlp128x2(out, force=False):
    """Flatten per-run MLP-128x2 comparison prober metrics into a companion CSV.

    Sourced from PRX_MLP_H5_FILES — a separate companion file per sweep axis
    (kz, nf, N, kz8), each storing theta/omega-only metrics (no kinetic/pot)
    under runs/<run_name>/probers/128x2/{theta,omega}.attrs. Used together with
    sweep_metrics.csv (prober_type='rf_ridge') to reproduce the "RF vs MLP
    compare" figures.
    """
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    try:
        import h5py
        import pandas as pd
    except ImportError as e:
        raise ImportError(f"h5py + pandas required for sweep_metrics_mlp128x2: {e}")

    param_keys = ["ndims", "num_frames", "nF_stride", "samples_n", "rep",
                  "gamma", "epochs", "batch_size"]

    rows = []
    for axis, h5 in PRX_MLP_H5_FILES.items():
        if not h5.exists():
            print(f"  [skip] {axis}: {h5} not found")
            continue
        with h5py.File(h5, "r") as f:
            runs = f.get("runs")
            if runs is None:
                continue
            for run_name, grp in runs.items():
                # parent/name (not the full path): keeps provenance unambiguous —
                # two axes share the basename probers_N.h5 — without leaking
                # internal absolute paths into the shipped CSV.
                row = {"run_name": run_name,
                       "source_h5": f"{h5.parent.name}/{h5.name}", "axis": axis,
                       "prober_type": "mlp_128x2"}
                for k in param_keys:
                    if k in grp.attrs:
                        row[k] = grp.attrs[k]
                pgrp = grp.get("probers/128x2")
                if pgrp is not None:
                    if "theta" in pgrp and "theta_test_deg" in pgrp["theta"].attrs:
                        row["theta_test_deg"] = float(pgrp["theta"].attrs["theta_test_deg"])
                    if "omega" in pgrp and "omega_test_rmse" in pgrp["omega"].attrs:
                        row["omega_test_rmse"] = float(pgrp["omega"].attrs["omega_test_rmse"])
                rows.append(row)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  → {out}  ({len(df)} rows across {len(PRX_MLP_H5_FILES)} axes)")


def _ablation_trainable_params(cfg_key, dz, output_dim=2):
    """Trainable parameter count for one ablation prober config.

    RF-ridge: only the linear head is fit (ridge regression) — the random
    expansion is frozen, so trainable = output_dim * expand_dim + output_dim.
    MLP: every parameter is trained with Adam — instantiate and count.
    Computed once here (not in the notebook) so reproduce_figures.ipynb never
    needs torch.
    """
    if cfg_key.startswith("rf_"):
        expand_dim = int(cfg_key.split("_")[1])
        return output_dim * expand_dim + output_dim
    hdim, num_layers = (int(x) for x in cfg_key.split("x"))
    m = Prober(dz=dz, output_dim=output_dim, hdim=hdim, num_layers=num_layers)
    return sum(p.numel() for p in m.parameters())


def task_prober_ablation(out, force=False):
    """Flatten the prober architecture ablation (4 RF-ridge + 5 MLP configs,
    kz=2 and kz=8, 10 reps each, fixed nF=2/N=1000/stride=1 run) into one CSV.

    Trainable-parameter counts are precomputed here (needs torch/Prober) so the
    notebook itself can stay torch-free.
    """
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    try:
        import h5py
        import pandas as pd
    except ImportError as e:
        raise ImportError(f"h5py + pandas required for prober_ablation: {e}")

    rows = []
    for kz, h5_paths in PROBER_ABLATION_H5.items():
        for prober_kind, h5_path in h5_paths.items():
            if not h5_path.exists():
                print(f"  [skip] kz={kz} {prober_kind}: {h5_path} not found")
                continue
            with h5py.File(h5_path, "r") as f:
                for run_name, run_grp in f.get("runs", {}).items():
                    rep = int(run_grp.attrs.get("rep", 0))
                    for cfg_key, cfg_grp in run_grp.get("probers", {}).items():
                        label = str(cfg_grp.attrs.get("prober_label", cfg_key)).replace(" [ridge]", "")
                        theta_v = omega_v = None
                        if "theta" in cfg_grp and "theta_test_deg" in cfg_grp["theta"].attrs:
                            theta_v = float(cfg_grp["theta"].attrs["theta_test_deg"])
                        if "omega" in cfg_grp and "omega_test_rmse" in cfg_grp["omega"].attrs:
                            omega_v = float(cfg_grp["omega"].attrs["omega_test_rmse"]) * (180.0 / np.pi)
                        rows.append({
                            "kz": kz, "rep": rep, "cfg_key": cfg_key, "prober_label": label,
                            "prober_kind": "RF-linear" if cfg_key.startswith("rf_") else "MLP",
                            "theta_test_deg": theta_v, "omega_test_deg_s": omega_v,
                            "trainable_params_theta": _ablation_trainable_params(cfg_key, dz=kz, output_dim=2),
                            "trainable_params_omega": _ablation_trainable_params(cfg_key, dz=kz, output_dim=1),
                        })

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  → {out}  ({len(df)} rows)")


def task_phase_portrait(out, data_npz, force=False):
    """Encode kz=2 model and bundle with wrapped theta / omega."""
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    model, p = _load_checkpoint(CKPT_KZ2)
    frames = torch.from_numpy(data_npz["frames"]).float()
    zx, numD = encode_dataset(model, frames, int(p["num_frames"]),
                              nF_stride=int(p["nF_stride"]), device=DEVICE)
    zx_by_trial = zx.numpy().reshape(frames.shape[0], numD, int(p["ndims"]))

    theta_rad = data_npz["theta"]           # (1200, 60)
    omega     = data_npz["vel_theta"]       # (1200, 60)
    theta_wrapped_deg = _wrap_deg(theta_rad)

    np.savez(
        out,
        theta_wrapped_deg = theta_wrapped_deg.astype(np.float32),
        omega_rad_per_s   = omega.astype(np.float32),
        zx_latent         = zx_by_trial.astype(np.float32),   # (1200, numD, 2)
        run_name          = np.array(p["run_name"]),
    )
    print(f"  → {out}  zx shape={zx_by_trial.shape}")


def task_embedding_kz2(out, data_npz, force=False, subsample=100_000, seed=0):
    """Encoded Z + physical variables (theta/omega/KE/PE/TE) for scatter plots."""
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    model, p = _load_checkpoint(CKPT_KZ2)
    frames = torch.from_numpy(data_npz["frames"]).float()
    zx, numD = encode_dataset(model, frames, int(p["num_frames"]),
                              nF_stride=int(p["nF_stride"]), device=DEVICE)
    zx_np = zx.numpy()

    phys_dict = {
        "theta":            data_npz["theta"],
        "vel_theta":        data_npz["vel_theta"],
        "kinetic_energy":   data_npz["kinetic_energy"],
        "potential_energy": data_npz["potential_energy"],
        "total_energy":     data_npz["total_energy"],
    }
    fv, _ = _fvars_flat(phys_dict, int(p["num_frames"]), nF_stride=int(p["nF_stride"]))

    rng = np.random.default_rng(seed)
    N = zx_np.shape[0]
    idx = rng.choice(N, size=min(subsample, N), replace=False)

    np.savez(
        out,
        Z            = zx_np[idx].astype(np.float32),
        theta_deg    = _wrap_deg(fv["theta"][idx]).astype(np.float32),
        omega_dps    = (fv["omega"][idx] * 180.0 / np.pi).astype(np.float32),
        kinetic      = fv["kinetic"][idx].astype(np.float32),
        pot          = fv["pot"][idx].astype(np.float32),
        total        = fv["total"][idx].astype(np.float32),
        idx          = idx.astype(np.int64),
        N_total      = np.array(N),
    )
    print(f"  → {out}  {len(idx):,} / {N:,} points")


def task_embedding_vs_N(out, data_npz, force=False, subsample=50_000, seed=0):
    """Three embeddings (N=64, 256, 1000) for the supp comparison figure."""
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    frames = torch.from_numpy(data_npz["frames"]).float()
    phys_dict = {k: data_npz[k] for k in ("theta", "vel_theta",
                                          "kinetic_energy", "potential_energy",
                                          "total_energy")}

    ckpts = {64: CKPT_KZ2_N64, 256: CKPT_KZ2_N256, 1000: CKPT_KZ2}
    out_d = {}
    for N_train, ck_path in ckpts.items():
        model, p = _load_checkpoint(ck_path)
        zx, numD = encode_dataset(model, frames, int(p["num_frames"]),
                                  nF_stride=int(p["nF_stride"]), device=DEVICE)
        zx_np = zx.numpy()
        fv, _ = _fvars_flat(phys_dict, int(p["num_frames"]), nF_stride=int(p["nF_stride"]))

        rng = np.random.default_rng(seed)
        idx = rng.choice(zx_np.shape[0],
                         size=min(subsample, zx_np.shape[0]), replace=False)
        out_d[f"Z_N{N_train}"]       = zx_np[idx].astype(np.float32)
        out_d[f"theta_deg_N{N_train}"] = _wrap_deg(fv["theta"][idx]).astype(np.float32)

    np.savez(out, **out_d)
    print(f"  → {out}  N ∈ {{64, 256, 1000}}")


def task_rollout_test(out, data_npz, force=False, prober_type="rf_ridge",
                      num_samples=40, quiver_n=30, n_vis=3, vis_seed=3):
    """Roll out kz=2 dynamics on the 200-trial test set; compute RMSE + quiver.

    prober_type : "rf_ridge" (default) fits a RandomFeatureProber via closed-form
                  ridge regression, matching the pipeline that produced the final
                  arxiv figures. "mlp" reproduces the original 1024x3 MLP+Adam
                  prober used by earlier revisions of this repo.
    """
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    model, p = _load_checkpoint(CKPT_KZ2)
    nf = int(p["num_frames"])
    dz = int(p["ndims"])
    nf_stride = int(p["nF_stride"])

    frames = torch.from_numpy(data_npz["frames"]).float()
    zx, numD = encode_dataset(model, frames, nf, nF_stride=nf_stride, device=DEVICE)
    zx_np = zx.numpy()

    # ── train θ prober on cos/sin targets (for wrapped RMSE decoding) ──────
    # data_npz["theta"] is (1200, 60). FrameDataset visits windows starting at
    # frame t=0..numD-1 per trial, trial as slow index. Must slice first,
    # THEN flatten — naive reshape(-1) is frame-major and mis-aligned.
    theta_flat = torch.from_numpy(
        data_npz["theta"][:, :numD].reshape(-1)
    ).float()
    N_train_trials = 1000
    train_slice = slice(0, N_train_trials * numD)
    test_slice  = slice(N_train_trials * numD, 1200 * numD)

    theta_tr_xy = torch.stack(
        [torch.cos(theta_flat[train_slice]), torch.sin(theta_flat[train_slice])], dim=1
    )
    theta_ts_xy = torch.stack(
        [torch.cos(theta_flat[test_slice]),  torch.sin(theta_flat[test_slice])],  dim=1
    )

    if prober_type == "rf_ridge":
        p_theta = RandomFeatureProber(
            dz=dz, output_dim=2, expand_dim=RF_EXPAND_DIM, weight_init=RF_WEIGHT_INIT,
        ).to(DEVICE)
        fit_ridge_head(p_theta, zx[train_slice], theta_tr_xy,
                       alpha=RF_RIDGE_ALPHA, device=DEVICE)
    elif prober_type == "mlp":
        p_theta = Prober(dz=dz, output_dim=2).to(DEVICE)
        p_theta, _, _ = train_prober(
            p_theta,
            zx[train_slice], theta_tr_xy,
            zx[test_slice],  theta_ts_xy,
            epochs=100, lr=1e-4, batch_size=1024, device=DEVICE,
        )
    else:
        raise ValueError(f"Unknown prober_type: {prober_type!r}")

    # ── rollout on test set ────────────────────────────────────────────────
    N_TEST    = 200
    zx_test   = zx.reshape(1200, numD, dz)[N_train_trials:N_train_trials + N_TEST].clone()
    theta_tst = theta_flat[test_slice].reshape(N_TEST, numD)

    # (numD - 1) rather than numD: guarantees n_rollout * nf_stride stays a valid
    # index into the numD encoded windows. With nf_stride=1, numD // nf_stride
    # divides exactly and would request one step past the last window.
    n_rollout = (numD - 1) // nf_stride
    x_frames  = np.arange(n_rollout + 1) * nf_stride

    # particles (stochastic rollout)
    pred_z = torch.zeros(n_rollout + 1, N_TEST, dz)
    pred_z[0] = zx_test[:, 0, :]
    xi = pred_z[0].repeat_interleave(num_samples, dim=0).to(DEVICE)
    with torch.no_grad():
        for s in range(n_rollout):
            delta, lv = model.delta_decoder(xi)
            xi = xi + delta + torch.exp(0.5 * lv) * torch.randn_like(xi)
            particles = xi.cpu().reshape(N_TEST, num_samples, dz)
            pred_z[s + 1] = particles.mean(dim=1)

    true_z_visited = torch.stack(
        [zx_test[:, k * nf_stride, :] for k in range(n_rollout + 1)], dim=0
    )  # (n_rollout+1, N_TEST, dz)

    # Latent RMSE
    latent_err       = torch.sqrt(((pred_z - true_z_visited) ** 2).sum(dim=2))
    latent_rmse_mean = latent_err.mean(dim=1).numpy()
    latent_rmse_sem  = (latent_err.std(dim=1) / np.sqrt(N_TEST)).numpy()

    # θ RMSE (wrapped, in degrees)
    pred_flat = pred_z.reshape(-1, dz)
    with torch.no_grad():
        pred_xy = p_theta(pred_flat.to(DEVICE)).cpu()
    pred_theta = torch.atan2(pred_xy[:, 1], pred_xy[:, 0]).reshape(n_rollout + 1, N_TEST)
    true_theta = torch.stack([theta_tst[:, k * nf_stride] for k in range(n_rollout + 1)], dim=0)
    dtheta = torch.atan2(torch.sin(pred_theta - true_theta),
                         torch.cos(pred_theta - true_theta))
    err_deg_sq    = torch.rad2deg(dtheta).pow(2)
    mse_per_step  = err_deg_sq.mean(dim=1)
    theta_err_mean = torch.sqrt(mse_per_step).numpy()
    theta_err_sem  = (err_deg_sq.std(dim=1) / (2 * torch.sqrt(mse_per_step) * np.sqrt(N_TEST))).numpy()

    # ── decoder quiver field on test-set region ────────────────────────────
    test_pts = zx_np[N_train_trials * numD:]
    z0, z1   = test_pts[:, 0], test_pts[:, 1]
    r0 = [np.percentile(z0, 1), np.percentile(z0, 99)]
    r1 = [np.percentile(z1, 1), np.percentile(z1, 99)]
    gx, gy = np.meshgrid(np.linspace(*r0, quiver_n), np.linspace(*r1, quiver_n))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        dxy, _ = model.delta_decoder(torch.from_numpy(grid).to(DEVICE))
    dxy = dxy.cpu().numpy()
    # normalize arrow lengths
    norms = np.linalg.norm(dxy, axis=1, keepdims=True)
    norms[norms == 0] = 1
    quiver_uv = dxy / norms

    # ── fixed points: find θ ≈ 0 and θ ≈ π in latent grid ──────────────────
    # Cheap: sweep fine grid, encode, find points that map to those angles.
    # Here we approximate as: mean latent code in data points where θ_raw near 0 / π.
    theta_test_flat = theta_tst.reshape(-1).numpy()
    zflat = test_pts
    wrap = ((theta_test_flat + np.pi) % (2 * np.pi)) - np.pi   # (-π, π]
    near0 = np.abs(wrap)         < 0.05
    nearpi = np.abs(np.abs(wrap) - np.pi) < 0.05
    fix_dn = zflat[near0].mean(axis=0)  if near0.any()  else None   # stable
    fix_up = zflat[nearpi].mean(axis=0) if nearpi.any() else None   # unstable

    # ── pick visualization trajectories (deterministic seed) ───────────────
    rng = np.random.default_rng(vis_seed)
    vis_idx = rng.choice(N_TEST, size=n_vis, replace=False)
    vis_true_z = zx_test[vis_idx, :, :2].numpy()         # (n_vis, numD, 2)
    vis_pred_z = pred_z[:, vis_idx, :2].numpy()          # (n_rollout+1, n_vis, 2)

    np.savez(
        out,
        x_frames         = x_frames.astype(np.int32),
        latent_rmse_mean = latent_rmse_mean.astype(np.float32),
        latent_rmse_sem  = latent_rmse_sem.astype(np.float32),
        theta_err_mean   = theta_err_mean.astype(np.float32),
        theta_err_sem    = theta_err_sem.astype(np.float32),
        quiver_xy        = grid.astype(np.float32),
        quiver_uv        = quiver_uv.astype(np.float32),
        fix_up           = (fix_up if fix_up is not None else np.full(2, np.nan)).astype(np.float32),
        fix_dn           = (fix_dn if fix_dn is not None else np.full(2, np.nan)).astype(np.float32),
        test_z_bg        = test_pts.astype(np.float32),  # background scatter
        r0               = np.array(r0, dtype=np.float32),
        r1               = np.array(r1, dtype=np.float32),
        vis_idx          = vis_idx.astype(np.int32),
        vis_true_z       = vis_true_z.astype(np.float32),
        vis_pred_z       = vis_pred_z.astype(np.float32),
        num_frames       = np.array(nf),
        nF_stride        = np.array(nf_stride),
        n_rollout        = np.array(n_rollout),
    )
    print(f"  → {out}  n_rollout={n_rollout}, N_TEST={N_TEST}")


def task_pca_kz8(out, data_npz, force=False, id_subsample=10_000, seed=1):
    """Encode kz=8 model; compute SVD, participation ratio, TwoNN, MLE(k), PCA top-2."""
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    model, p = _load_checkpoint(CKPT_KZ8)
    frames = torch.from_numpy(data_npz["frames"]).float()
    zx, numD = encode_dataset(model, frames, int(p["num_frames"]),
                              nF_stride=int(p["nF_stride"]), device=DEVICE)
    zx_np = zx.numpy()
    z_centered = zx_np - zx_np.mean(axis=0)
    dz8 = zx_np.shape[1]

    # SVD
    U, sv, Vt = np.linalg.svd(z_centered, full_matrices=False)
    pr       = float(sv.sum() ** 2 / (sv ** 2).sum())
    sv_norm  = sv / sv[0]
    var_exp  = sv ** 2 / (sv ** 2).sum()
    z_pc     = z_centered @ Vt[:2].T

    # Physical variables
    phys_dict = {k: data_npz[k] for k in ("theta", "vel_theta",
                                          "kinetic_energy", "potential_energy",
                                          "total_energy")}
    fv, _ = _fvars_flat(phys_dict, int(p["num_frames"]), nF_stride=int(p["nF_stride"]))
    theta_deg = _wrap_deg(fv["theta"])

    # Intrinsic dimensionality
    import skdim
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z_centered),
                     size=min(id_subsample, len(z_centered)), replace=False)
    Z_id = z_centered[idx]
    id_twonn = float(skdim.id.TwoNN().fit(Z_id).dimension_)

    def _lb_mle(X, k):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        T  = nn.kneighbors(X)[0][:, 1:]
        lr = np.log(np.clip(T[:, -1:] / np.clip(T[:, :-1], 1e-10, None),
                             1e-10, None))
        return float((1.0 / lr.mean(axis=1)).mean())
    K_curve      = np.arange(2, min(101, len(Z_id) // 10), 2, dtype=np.int32)
    id_mle_curve = np.array([_lb_mle(Z_id, int(k)) for k in K_curve], dtype=np.float32)

    np.savez(
        out,
        sv            = sv.astype(np.float32),
        sv_norm       = sv_norm.astype(np.float32),
        pr            = np.array(pr, dtype=np.float32),
        id_twonn      = np.array(id_twonn, dtype=np.float32),
        K_curve       = K_curve,
        id_mle_curve  = id_mle_curve,
        z_pc          = z_pc.astype(np.float32),
        theta_deg     = theta_deg.astype(np.float32),
        var_exp       = var_exp.astype(np.float32),
        dz            = np.array(dz8, dtype=np.int32),
        run_name      = np.array(p["run_name"]),
    )
    print(f"  → {out}  PR={pr:.2f}, TwoNN={id_twonn:.2f}")


def task_cylindrical_phys(out, data_npz, force=False):
    """Wrapped theta and omega arrays used by the cylindrical projection figure."""
    if out.exists() and not force:
        print(f"  [skip] {out}")
        return
    theta_wrapped_deg = _wrap_deg(data_npz["theta"])
    omega_dps         = data_npz["vel_theta"] * 180.0 / np.pi
    np.savez(
        out,
        theta_wrapped_deg = theta_wrapped_deg.astype(np.float32),
        omega_dps         = omega_dps.astype(np.float32),
    )
    print(f"  → {out}")


# ─── orchestration ──────────────────────────────────────────────────────────
TASKS = {
    "sweep_metrics":        ("sweep_metrics.csv",           task_sweep_metrics,           False),
    "sweep_metrics_mlp128x2": ("sweep_metrics_mlp128x2.csv", task_sweep_metrics_mlp128x2, False),
    "prober_ablation":      ("prober_ablation.csv",         task_prober_ablation,        False),
    "phase_portrait":    ("phase_portrait.npz",    task_phase_portrait,   True),
    "embedding_kz2":     ("embedding_kz2.npz",     task_embedding_kz2,    True),
    "embedding_vs_N":    ("embedding_vs_N.npz",    task_embedding_vs_N,   True),
    "rollout_test":      ("rollout_test.npz",      task_rollout_test,     True),
    "pca_kz8":           ("pca_kz8.npz",           task_pca_kz8,          True),
    "cylindrical_phys":  ("cylindrical_phys.npz",  task_cylindrical_phys, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(TASKS),
                    help="Run only the listed tasks")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs")
    args = ap.parse_args()

    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data lazily — only if a task needs it.
    selected = args.only or list(TASKS)
    needs_data = any(TASKS[t][2] for t in selected)
    data_npz = data_mod.load() if needs_data else None
    if data_npz is not None:
        print(f"Data: frames shape={data_npz['frames'].shape}, "
              f"theta shape={data_npz['theta'].shape}")

    for task_key in selected:
        out_name, fn, needs = TASKS[task_key]
        out_path = FIGDATA_DIR / out_name
        print(f"\n[{task_key}]")
        if needs:
            fn(out_path, data_npz, force=args.force)
        else:
            fn(out_path, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
