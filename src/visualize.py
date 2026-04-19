"""
visualize.py — All required figures (spec §6).

Figures
-------
1. Space-time heatmap grid (3 traj × [GT | PINN | PINF-Fourier | PINF-SIREN])
2. Error heatmaps (|u_pred - u_true|, log scale)
3. Snapshot line plots at t ∈ {0.0, 0.5, 1.0, 1.5, 2.0}
4. Shock close-up (±0.1 around steepest gradient at t≈1.0)
5. Super-resolution comparison (4× spatial)
6. Error power spectrum (log-log, wavenumber vs power)
7. Convergence curves (loss + val-L2 vs epoch, 3 seeds)
8. Characteristic lines (dx/dt = u(x,t))

All PNG + PDF saved to results/figures/.
Colormaps: viridis for field values, RdBu_r for signed error.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Any

FIG_DIR = Path(__file__).parent.parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CMAP_FIELD = "viridis"
CMAP_ERROR = "RdBu_r"
DPI = 150


def _save(fig: plt.Figure, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 & 2: Space-time heatmap grid + error heatmaps
# ---------------------------------------------------------------------------
def plot_spacetime_grid(
    u_dict: dict[str, np.ndarray],   # {'GT': (3,T,X), 'PINN': ..., ...}
    x_phys: np.ndarray,               # (X,)
    t_phys: np.ndarray,               # (T,)
    name: str = "fig1_spacetime_grid",
):
    """3 trajectories × N models: heatmap of u(x,t)."""
    models = list(u_dict.keys())
    n_traj = next(iter(u_dict.values())).shape[0]
    n_cols = len(models)

    fig, axes = plt.subplots(
        n_traj, n_cols,
        figsize=(4 * n_cols, 3.5 * n_traj),
        squeeze=False,
    )

    # Shared colormap range for predictions (excluding GT for error maps)
    u_gt = u_dict[models[0]]
    vmin, vmax = u_gt.min(), u_gt.max()

    for col, model_name in enumerate(models):
        u = u_dict[model_name]   # (3, T, X)
        for row in range(n_traj):
            ax = axes[row][col]
            im = ax.imshow(
                u[row].T,
                origin="lower",
                aspect="auto",
                extent=[t_phys[0], t_phys[-1], x_phys[0], x_phys[-1]],
                cmap=CMAP_FIELD,
                vmin=vmin, vmax=vmax,
            )
            if row == 0:
                ax.set_title(model_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Traj {row+1}\nx", fontsize=8)
            ax.set_xlabel("t" if row == n_traj - 1 else "")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Space-time solutions u(x,t)", fontsize=12)
    plt.tight_layout()
    _save(fig, name)


def plot_error_heatmaps(
    u_pred_dict: dict[str, np.ndarray],   # {'PINN': (3,T,X), ...} — no GT key
    u_gt: np.ndarray,                     # (3, T, X)
    x_phys: np.ndarray,
    t_phys: np.ndarray,
    name: str = "fig2_error_heatmaps",
):
    """Absolute error |u_pred - u_true| on log scale."""
    models = list(u_pred_dict.keys())
    n_traj = u_gt.shape[0]
    n_cols = len(models)

    fig, axes = plt.subplots(n_traj, n_cols, figsize=(4 * n_cols, 3.5 * n_traj), squeeze=False)

    for col, model_name in enumerate(models):
        err = np.abs(u_pred_dict[model_name] - u_gt)   # (3, T, X)
        norm = mcolors.LogNorm(vmin=max(err.min(), 1e-6), vmax=err.max())
        for row in range(n_traj):
            ax = axes[row][col]
            im = ax.imshow(
                err[row].T,
                origin="lower",
                aspect="auto",
                extent=[t_phys[0], t_phys[-1], x_phys[0], x_phys[-1]],
                cmap="hot_r",
                norm=norm,
            )
            if row == 0:
                ax.set_title(f"|err| {model_name}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Traj {row+1}\nx", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Absolute error |u_pred − u_true|", fontsize=12)
    plt.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 3: Snapshot line plots
# ---------------------------------------------------------------------------
def plot_snapshots(
    u_dict: dict[str, np.ndarray],   # {'GT': (T,X), 'PINN': ...}
    x_phys: np.ndarray,               # (X,)
    t_phys: np.ndarray,               # (T,)
    t_query: list[float] = None,
    name: str = "fig3_snapshots",
):
    """Line plots of u(x) at fixed t values, all models overlaid."""
    if t_query is None:
        t_query = [0.0, 0.5, 1.0, 1.5, 2.0]

    n_cols = len(t_query)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(u_dict)))

    for ci, (model_name, u) in enumerate(u_dict.items()):
        for col, tq in enumerate(t_query):
            ti = int(np.argmin(np.abs(t_phys - tq)))
            ls = "-" if model_name == "GT" else "--"
            lw = 2.0 if model_name == "GT" else 1.2
            axes[col].plot(x_phys, u[ti], ls=ls, lw=lw, color=colors[ci], label=model_name)
            axes[col].set_title(f"t = {tq:.1f}", fontsize=10)
            axes[col].set_xlabel("x")
            if col == 0:
                axes[col].set_ylabel("u(x)")

    axes[0].legend(fontsize=8)
    fig.suptitle("Solution snapshots u(x, t)", fontsize=12)
    plt.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 4: Shock close-up
# ---------------------------------------------------------------------------
def plot_shock_closeup(
    u_dict: dict[str, np.ndarray],   # {'GT': (T,X), ...}
    x_phys: np.ndarray,
    t_phys: np.ndarray,
    t_target: float = 1.0,
    window: float = 0.1,
    name: str = "fig4_shock_closeup",
):
    """Zoom on the steepest gradient region at t≈t_target."""
    ti = int(np.argmin(np.abs(t_phys - t_target)))
    u_gt = u_dict[list(u_dict.keys())[0]]
    ux = np.gradient(u_gt[ti], x_phys)
    shock_x = x_phys[np.argmax(np.abs(ux))]

    x_mask = (x_phys >= shock_x - window) & (x_phys <= shock_x + window)
    x_zoom = x_phys[x_mask]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(u_dict)))
    for ci, (model_name, u) in enumerate(u_dict.items()):
        ls = "-" if model_name == "GT" else "--"
        ax.plot(x_zoom, u[ti][x_mask], ls=ls, lw=2, color=colors[ci], label=model_name)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(f"Shock close-up at t = {t_target:.1f}  (x ∈ [{shock_x-window:.2f}, {shock_x+window:.2f}])")
    ax.legend()
    plt.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 5: Super-resolution
# ---------------------------------------------------------------------------
def plot_super_resolution(
    u_dict_sr: dict[str, np.ndarray],  # {'GT_spline': (T,X_sr), 'PINN': ..., 'PINF': ...}
    x_sr: np.ndarray,
    t_phys: np.ndarray,
    t_target: float = 1.0,
    name: str = "fig5_super_resolution",
):
    """Overlay SR predictions vs spline GT at one t slice."""
    ti = int(np.argmin(np.abs(t_phys - t_target)))
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(u_dict_sr)))

    for ci, (label, u) in enumerate(u_dict_sr.items()):
        ls = "-" if "GT" in label else "--"
        ax.plot(x_sr, u[ti], ls=ls, lw=2, color=colors[ci], label=label)

    ax.set_xlabel("x (4× spatial resolution)")
    ax.set_ylabel("u(x)")
    ax.set_title(f"4× Spatial Super-resolution at t = {t_target:.1f}")
    ax.legend()
    plt.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 6: Error power spectrum
# ---------------------------------------------------------------------------
def plot_error_spectrum(
    spectra_dict: dict[str, dict],  # {'PINN': spectral_metrics_output, 'PINF': ...}
    t_label: str = "t_norm=0.00",
    name: str = "fig6_error_spectrum",
):
    """Log-log plot of error power spectrum vs wavenumber."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(spectra_dict)))

    for ci, (model_name, spectra) in enumerate(spectra_dict.items()):
        if t_label not in spectra:
            continue
        k     = spectra[t_label]["k"]
        power = spectra[t_label]["power"]
        mask  = k > 0
        ax.loglog(k[mask], power[mask], lw=2, color=colors[ci], label=model_name)

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("|ê(k)|²  (error power)")
    ax.set_title(f"Error power spectrum ({t_label})")
    ax.legend()
    plt.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 7: Convergence curves
# ---------------------------------------------------------------------------
def plot_convergence(
    history_dict: dict[str, list[dict]],  # model_name → list of per-seed dicts
    name: str = "fig7_convergence",
):
    """Training loss and val rel-L2 vs epoch, 3 seeds overlaid."""
    metrics_to_plot = ["loss/data", "loss/phys", "val/rel_l2"]
    ylabels = ["Data loss", "Physics loss", "Val rel-L2"]
    n_models = len(history_dict)
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(
        n_models, n_metrics,
        figsize=(5 * n_metrics, 3.5 * n_models),
        squeeze=False,
    )

    for row, (model_name, seed_histories) in enumerate(history_dict.items()):
        for col, (metric, ylabel) in enumerate(zip(metrics_to_plot, ylabels)):
            ax = axes[row][col]
            for seed_idx, history in enumerate(seed_histories):
                if metric not in history:
                    continue
                epochs = np.arange(1, len(history[metric]) + 1)
                ax.semilogy(epochs, history[metric], alpha=0.8, label=f"seed {seed_idx}")
            if row == 0:
                ax.set_title(ylabel, fontsize=10)
            if col == 0:
                ax.set_ylabel(model_name, fontsize=9)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7)

    fig.suptitle("Training convergence", fontsize=12)
    plt.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 8: Characteristic lines
# ---------------------------------------------------------------------------
def plot_characteristics(
    u_dict: dict[str, np.ndarray],   # {'GT': (T,X), 'PINN': ..., ...}
    x_phys: np.ndarray,               # (X,)
    t_phys: np.ndarray,               # (T,)
    n_chars: int = 40,
    name: str = "fig8_characteristics",
):
    """Lagrangian characteristic lines dx/dt = u(x,t) for each model."""
    n_models = len(u_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    dt = t_phys[1] - t_phys[0]
    x0_indices = np.linspace(0, len(x_phys) - 1, n_chars, dtype=int)

    for ax, (model_name, u) in zip(axes, u_dict.items()):
        for xi0 in x0_indices:
            x_char = [x_phys[xi0]]
            for ti in range(len(t_phys) - 1):
                xc = x_char[-1]
                # Interpolate u at current x position
                xi = np.interp(xc, x_phys, u[ti])
                x_next = xc + xi * dt
                x_next = np.clip(x_next, x_phys[0], x_phys[-1])
                x_char.append(x_next)
            ax.plot(x_char, t_phys, lw=0.6, alpha=0.7, color="steelblue")

        ax.set_xlabel("x")
        ax.set_title(model_name)
        if ax is axes[0]:
            ax.set_ylabel("t")

    fig.suptitle("Characteristic lines  dx/dt = u(x,t)", fontsize=12)
    plt.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Convenience: run all figures given pre-computed arrays
# ---------------------------------------------------------------------------
def make_all_figures(
    u_dict: dict[str, np.ndarray],   # {'GT': (3,T,X), 'PINN': ..., ...}
    x_phys: np.ndarray,
    t_phys: np.ndarray,
    spectra_dict: dict[str, dict] | None = None,
    history_dict: dict[str, list] | None = None,
    u_dict_sr: dict[str, np.ndarray] | None = None,
    x_sr: np.ndarray | None = None,
):
    gt_key = [k for k in u_dict if "GT" in k or "gt" in k.lower()][0]
    u_gt   = u_dict[gt_key]
    u_pred_dict = {k: v for k, v in u_dict.items() if k != gt_key}

    # Fig 1 — full grid
    plot_spacetime_grid(u_dict, x_phys, t_phys)

    # Fig 2 — error heatmaps
    plot_error_heatmaps(u_pred_dict, u_gt, x_phys, t_phys)

    # Fig 3 — snapshots (use first trajectory only)
    u_dict_traj0 = {k: v[0] for k, v in u_dict.items()}
    plot_snapshots(u_dict_traj0, x_phys, t_phys)

    # Fig 4 — shock close-up
    plot_shock_closeup(u_dict_traj0, x_phys, t_phys)

    # Fig 5 — SR (only if provided)
    if u_dict_sr is not None and x_sr is not None:
        plot_super_resolution(u_dict_sr, x_sr, t_phys)

    # Fig 6 — spectral (only if provided)
    if spectra_dict is not None:
        # Use t_norm ≈ 0 (closest to t_phys = 1.0)
        T_MAX = 2.01
        tq_norm = 1.0 / (T_MAX / 2.0) - 1.0
        t_label = min(
            next(iter(spectra_dict.values())).keys(),
            key=lambda k: abs(float(k.split("=")[1]) - tq_norm)
        )
        plot_error_spectrum(spectra_dict, t_label=t_label)

    # Fig 7 — convergence (only if provided)
    if history_dict is not None:
        plot_convergence(history_dict)

    # Fig 8 — characteristics
    plot_characteristics(u_dict_traj0, x_phys, t_phys)

    print(f"All figures saved to {FIG_DIR}")
