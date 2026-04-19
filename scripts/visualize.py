"""
visualize.py — Load checkpoints for all models and produce all §6 figures.

Usage
-----
    python scripts/visualize.py \
        --pinn      experiments/runs/pinn_seed0/checkpoints/best.pt \
        --pinf_f    experiments/runs/pinf_fourier_seed0/checkpoints/best.pt \
        --pinf_s    experiments/runs/pinf_siren_seed0/checkpoints/best.pt \
        [--n_traj 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import CubicSpline

from src.dataset import BurgersPDEBench
from src.models.pinn import build_pinn
from src.models.pinf import build_pinf
from src.evaluator import predict_field, spectral_metrics
from src.visualize import make_all_figures, plot_super_resolution
from scripts.evaluate import load_model_from_ckpt


def pick_representative_trajs(u_gt: np.ndarray, n: int = 3) -> list[int]:
    """Pick n trajectories spanning low / moderate / sharp shock complexity."""
    # Proxy: sort by max |u_x| and pick evenly spaced quantiles
    dx = 2.0 / (u_gt.shape[2] - 1)
    max_ux = np.abs(np.gradient(u_gt[:, -1, :], dx, axis=-1)).max(axis=-1)  # (B,)
    sorted_idx = np.argsort(max_ux)
    picks = [
        sorted_idx[len(sorted_idx) // 4],
        sorted_idx[len(sorted_idx) // 2],
        sorted_idx[-len(sorted_idx) // 10],
    ]
    return [int(p) for p in picks[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn",   required=True)
    parser.add_argument("--pinf_f", required=True, help="PINF-Fourier checkpoint")
    parser.add_argument("--pinf_s", required=True, help="PINF-SIREN checkpoint")
    parser.add_argument("--n_traj", type=int, default=3)
    parser.add_argument("--h5_path", default="../data/1D_Burgers_Sols_Nu0.01.hdf5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model_paths = {
        "PINN":        args.pinn,
        "PINF-Fourier": args.pinf_f,
        "PINF-SIREN":  args.pinf_s,
    }
    models = {}
    for name, ckpt_path in model_paths.items():
        model, _ = load_model_from_ckpt(Path(ckpt_path), device)
        models[name] = model
        print(f"Loaded {name}  params={model.count_parameters():,}")

    # Dataset
    h5_path = Path(__file__).parent.parent / args.h5_path
    val_ds  = BurgersPDEBench(h5_path, split="val", mode="field", normalize=True)
    val_loader = DataLoader(val_ds, batch_size=args.n_traj, shuffle=False, num_workers=2)
    batch = next(iter(val_loader))

    ic     = batch["ic"]         # (B, X)
    u_true = batch["u"].numpy()  # (B, T, X)
    x_norm = batch["x"][0].numpy()
    t_norm = batch["t"][0].numpy()

    # Physical grids (x is already physical; t needs rescaling)
    T_MAX  = 2.01
    x_phys = x_norm
    t_phys = (t_norm + 1.0) / 2.0 * T_MAX

    # Pick representative trajectories
    traj_ids = pick_representative_trajs(u_true, n=args.n_traj)
    ic_sel   = ic[traj_ids]
    u_true_sel = u_true[traj_ids]

    # Predict on original grid for each model
    u_dict: dict[str, np.ndarray] = {"GT": u_true_sel}
    for model_name, model in models.items():
        with torch.no_grad():
            u_pred = predict_field(
                model, ic_sel.to(device),
                torch.from_numpy(x_norm), torch.from_numpy(t_norm),
                device,
            ).cpu().numpy()
        u_dict[model_name] = u_pred

    # Spectral metrics (on all n_traj trajectories)
    spectra_dict = {}
    for model_name, model in models.items():
        spectra_dict[model_name] = spectral_metrics(
            u_dict[model_name], u_true_sel, t_norm
        )

    # Super-resolution (4× spatial)
    X_orig = x_norm
    X_sr   = np.linspace(x_norm[0], x_norm[-1], len(x_norm) * 4)
    u_dict_sr: dict[str, np.ndarray] = {}

    # GT spline at 4× spatial
    gt_sr = np.zeros((args.n_traj, len(t_norm), len(X_sr)), dtype=np.float32)
    for b in range(args.n_traj):
        for ti in range(len(t_norm)):
            cs = CubicSpline(X_orig, u_true_sel[b, ti])
            gt_sr[b, ti] = cs(X_sr)
    u_dict_sr["GT_spline"] = gt_sr

    for model_name, model in models.items():
        with torch.no_grad():
            u_sr = predict_field(
                model, ic_sel.to(device),
                torch.from_numpy(X_sr.astype(np.float32)),
                torch.from_numpy(t_norm),
                device,
            ).cpu().numpy()
        u_dict_sr[model_name] = u_sr

    # Make all figures
    make_all_figures(
        u_dict=u_dict,
        x_phys=x_phys,
        t_phys=t_phys,
        spectra_dict=spectra_dict,
        history_dict=None,   # convergence curves require pre-loaded log files
        u_dict_sr={k: v[0] for k, v in u_dict_sr.items()},  # first traj
        x_sr=X_sr,
    )


if __name__ == "__main__":
    main()
