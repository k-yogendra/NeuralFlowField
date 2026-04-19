"""
evaluator.py — Full evaluation suite (spec §5).

Metrics
-------
§5.1  Accuracy      : rel-L2, RMSE, L∞, nRMSE
§5.2  Physical      : PDE residual L2, mass conservation, periodicity
§5.3  Super-res     : spatial (2×, 4×), temporal (2×, 4×), joint
§5.4  Shock region  : L2 in shock vs smooth regions (|u_x| > τ·max|u_x|)
§5.5  Spectral      : FFT of error field along x
§5.6  Efficiency    : param count, inference latency, FLOPs
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.interpolate import CubicSpline

from .losses.physics import physics_residual, NU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.no_grad()
def predict_field(
    model: nn.Module,
    ic: torch.Tensor,       # (B, X)
    x_grid: torch.Tensor,   # (X2,)  normalized
    t_grid: torch.Tensor,   # (T2,)  normalized
    device: torch.device,
    chunk: int = 8192,
) -> torch.Tensor:
    """Query model on a full (T2, X2) grid for a batch of ICs.

    Returns
    -------
    u_pred : (B, T2, X2)
    """
    B = ic.shape[0]
    T2 = len(t_grid)
    X2 = len(x_grid)

    x_exp = x_grid.view(1, 1, X2).expand(B, T2, X2)   # (B, T2, X2)
    t_exp = t_grid.view(1, T2, 1).expand(B, T2, X2)   # (B, T2, X2)

    x_flat = x_exp.reshape(B, T2 * X2).to(device)
    t_flat = t_exp.reshape(B, T2 * X2).to(device)
    ic_dev = ic.to(device)

    chunks = []
    for i in range(0, T2 * X2, chunk):
        chunks.append(model(x_flat[:, i:i+chunk], t_flat[:, i:i+chunk], ic_dev))
    return torch.cat(chunks, dim=1).reshape(B, T2, X2)


# ---------------------------------------------------------------------------
# §5.1  Accuracy metrics
# ---------------------------------------------------------------------------
def accuracy_metrics(u_pred: np.ndarray, u_true: np.ndarray) -> dict[str, float]:
    """Compute per-batch accuracy metrics and return their means.

    Parameters
    ----------
    u_pred, u_true : (B, T, X)
    """
    diff = u_pred - u_true
    norm_true = np.linalg.norm(u_true.reshape(u_true.shape[0], -1), axis=1)  # (B,)
    norm_diff = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)      # (B,)

    rel_l2 = (norm_diff / (norm_true + 1e-8)).mean()
    rmse   = np.sqrt((diff ** 2).mean())
    linf   = np.abs(diff).max(axis=(1, 2)).mean()

    u_range = u_true.max(axis=(1, 2)) - u_true.min(axis=(1, 2))   # (B,)
    nrmse   = (np.sqrt((diff ** 2).mean(axis=(1, 2))) / (u_range + 1e-8)).mean()

    return {
        "acc/rel_l2": float(rel_l2),
        "acc/rmse":   float(rmse),
        "acc/linf":   float(linf),
        "acc/nrmse":  float(nrmse),
    }


# ---------------------------------------------------------------------------
# §5.2  Physical consistency
# ---------------------------------------------------------------------------
def pde_residual_metrics(
    model: nn.Module,
    ic: torch.Tensor,       # (B, X)
    x_grid: torch.Tensor,   # (X_r,) normalized
    t_grid: torch.Tensor,   # (T_r,) normalized
    device: torch.device,
) -> dict[str, float]:
    """Evaluate PDE residual on a dense grid via autodiff."""
    B  = ic.shape[0]
    Tr = len(t_grid)
    Xr = len(x_grid)

    x_exp = x_grid.view(1, 1, Xr).expand(B, Tr, Xr).reshape(B, Tr * Xr).to(device)
    t_exp = t_grid.view(1, Tr, 1).expand(B, Tr, Xr).reshape(B, Tr * Xr).to(device)
    x_exp = x_exp.requires_grad_(True)
    t_exp = t_exp.requires_grad_(True)
    ic_dev = ic.to(device)

    r = physics_residual(model, x_exp, t_exp, ic_dev)   # (B, Tr*Xr)
    r_np = _to_np(r)

    return {
        "phys/pde_residual_l2_mean": float(np.sqrt((r_np ** 2).mean())),
        "phys/pde_residual_l2_max":  float(np.sqrt((r_np ** 2).max())),
    }


def mass_conservation_metrics(
    u_pred: np.ndarray,  # (B, T, X)
    x_phys: np.ndarray,  # (X,)  physical x in [-1, 1]
) -> dict[str, float]:
    """Max |∫u(x,t)dx − ∫u(x,0)dx| over t, averaged across batch."""
    mass_t0 = np.trapz(u_pred[:, 0, :], x_phys, axis=-1)  # (B,)
    drift = []
    for ti in range(u_pred.shape[1]):
        mass_ti = np.trapz(u_pred[:, ti, :], x_phys, axis=-1)
        drift.append(np.abs(mass_ti - mass_t0))              # (B,)
    drift_arr = np.stack(drift, axis=1)   # (B, T)
    return {
        "phys/mass_conservation_max_drift": float(drift_arr.max(axis=1).mean()),
    }


def periodicity_metrics(
    u_pred: np.ndarray,  # (B, T, X)
) -> dict[str, float]:
    """Mean |u(-1, t) − u(+1, t)| — uses first/last x grid cells."""
    diff = np.abs(u_pred[:, :, 0] - u_pred[:, :, -1])  # (B, T)
    return {"phys/periodicity_error": float(diff.mean())}


# ---------------------------------------------------------------------------
# §5.3  Super-resolution
# ---------------------------------------------------------------------------
def super_resolution_metrics(
    model: nn.Module,
    ic: torch.Tensor,         # (B, X)
    u_true: np.ndarray,       # (B, T, X)   — original (201, 1024) grid
    x_norm_orig: np.ndarray,  # (X,)   original normalized x
    t_norm_orig: np.ndarray,  # (T,)   original normalized t
    device: torch.device,
) -> dict[str, float]:
    """Evaluate SR at several upsampling factors; compare against spline GT."""
    results = {}

    def interp_gt_spatial(factor: int) -> np.ndarray:
        X_new = u_true.shape[2] * factor
        x_new = np.linspace(x_norm_orig[0], x_norm_orig[-1], X_new)
        out = np.zeros((u_true.shape[0], u_true.shape[1], X_new), dtype=np.float32)
        for b in range(u_true.shape[0]):
            for ti in range(u_true.shape[1]):
                cs = CubicSpline(x_norm_orig, u_true[b, ti])
                out[b, ti] = cs(x_new).astype(np.float32)
        return out, x_new

    def interp_gt_temporal(factor: int) -> np.ndarray:
        T_new = u_true.shape[1] * factor
        t_new = np.linspace(t_norm_orig[0], t_norm_orig[-1], T_new)
        out = np.zeros((u_true.shape[0], T_new, u_true.shape[2]), dtype=np.float32)
        for b in range(u_true.shape[0]):
            for xi in range(u_true.shape[2]):
                cs = CubicSpline(t_norm_orig, u_true[b, :, xi])
                out[b, :, xi] = cs(t_new).astype(np.float32)
        return out, t_new

    B = ic.shape[0]

    # Spatial SR
    for factor in [2, 4]:
        gt_sr, x_new = interp_gt_spatial(factor)
        x_t = torch.from_numpy(x_new.astype(np.float32))
        t_t = torch.from_numpy(t_norm_orig.astype(np.float32))
        with torch.no_grad():
            u_pred_sr = _to_np(predict_field(model, ic, x_t, t_t, device))
        metrics = accuracy_metrics(u_pred_sr, gt_sr)
        results[f"sr/spatial_{factor}x_rel_l2"] = metrics["acc/rel_l2"]

    # Temporal SR
    for factor in [2, 4]:
        gt_sr, t_new = interp_gt_temporal(factor)
        x_t = torch.from_numpy(x_norm_orig.astype(np.float32))
        t_t = torch.from_numpy(t_new.astype(np.float32))
        with torch.no_grad():
            u_pred_sr = _to_np(predict_field(model, ic, x_t, t_t, device))
        metrics = accuracy_metrics(u_pred_sr, gt_sr)
        results[f"sr/temporal_{factor}x_rel_l2"] = metrics["acc/rel_l2"]

    # Joint SR (4x temporal × 4x spatial)
    gt_sr_t, t_new = interp_gt_temporal(4)
    gt_sr_tx, x_new = interp_gt_spatial(4)
    # Interpolate gt_sr_t further in x
    X_new = u_true.shape[2] * 4
    T_new = u_true.shape[1] * 4
    gt_joint = np.zeros((B, T_new, X_new), dtype=np.float32)
    for b in range(B):
        for ti in range(T_new):
            cs = CubicSpline(x_norm_orig, gt_sr_t[b, ti])
            gt_joint[b, ti] = cs(x_new).astype(np.float32)
    x_t = torch.from_numpy(x_new.astype(np.float32))
    t_t = torch.from_numpy(t_new.astype(np.float32))
    with torch.no_grad():
        u_pred_joint = _to_np(predict_field(model, ic, x_t, t_t, device))
    metrics = accuracy_metrics(u_pred_joint, gt_joint)
    results["sr/joint_4x4x_rel_l2"] = metrics["acc/rel_l2"]

    return results


# ---------------------------------------------------------------------------
# §5.4  Shock vs smooth region
# ---------------------------------------------------------------------------
def shock_region_metrics(
    u_pred: np.ndarray,  # (B, T, X)
    u_true: np.ndarray,  # (B, T, X)
    dx: float,
    tau: float = 0.3,
) -> dict[str, float]:
    """L2 error separately in shock and smooth regions."""
    # Finite-difference u_x along x axis
    u_x = np.gradient(u_true, dx, axis=2)   # (B, T, X)
    max_ux = np.abs(u_x).max(axis=2, keepdims=True)  # (B, T, 1)
    shock_mask = np.abs(u_x) > tau * max_ux  # (B, T, X)

    diff_sq = (u_pred - u_true) ** 2

    def masked_rel_l2(mask):
        num = np.sqrt((diff_sq * mask).sum(axis=(1, 2)))
        den = np.sqrt((u_true ** 2 * mask).sum(axis=(1, 2))) + 1e-8
        return (num / den).mean()

    return {
        "shock/rel_l2_shock":  float(masked_rel_l2(shock_mask)),
        "shock/rel_l2_smooth": float(masked_rel_l2(~shock_mask)),
    }


# ---------------------------------------------------------------------------
# §5.5  Spectral analysis
# ---------------------------------------------------------------------------
def spectral_metrics(
    u_pred: np.ndarray,  # (B, T, X)
    u_true: np.ndarray,  # (B, T, X)
    t_norm: np.ndarray,  # (T,)  normalized
    t_query_norm: list[float] = None,
) -> dict[str, np.ndarray]:
    """Power spectrum of error field at fixed t values.

    Returns dict of wavenumber → power arrays for each query t.
    """
    if t_query_norm is None:
        T_MAX = 2.01
        # Normalized equivalents of t_phys ∈ {0.5, 1.0, 1.5}
        t_query_norm = [t / (T_MAX / 2.0) - 1.0 for t in [0.5, 1.0, 1.5]]

    error = u_pred - u_true    # (B, T, X)
    X = error.shape[2]
    k = np.fft.rfftfreq(X, d=1.0 / X)

    spectra = {}
    for tq in t_query_norm:
        ti = int(np.argmin(np.abs(t_norm - tq)))
        e_t = error[:, ti, :]              # (B, X)
        E_k = np.fft.rfft(e_t, axis=-1)   # (B, X//2+1)
        power = (np.abs(E_k) ** 2).mean(axis=0)   # (X//2+1,)
        label = f"t_norm={tq:.2f}"
        spectra[label] = {"k": k, "power": power}

    return spectra


# ---------------------------------------------------------------------------
# §5.6  Efficiency
# ---------------------------------------------------------------------------
def efficiency_metrics(
    model: nn.Module,
    ic: torch.Tensor,       # (1, X)
    x_grid: torch.Tensor,   # (X,)
    t_grid: torch.Tensor,   # (T,)
    device: torch.device,
    n_repeats: int = 10,
) -> dict[str, Any]:
    """Param count, inference latency, FLOPs."""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Inference latency
    model.eval()
    ic_dev = ic.to(device)
    T, X = len(t_grid), len(x_grid)
    x_flat = x_grid.view(1, 1, X).expand(1, T, X).reshape(1, T * X).to(device)
    t_flat = t_grid.view(1, T, 1).expand(1, T, X).reshape(1, T * X).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(x_flat, t_flat, ic_dev)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(x_flat, t_flat, ic_dev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / n_repeats * 1000.0

    # FLOPs via thop (optional)
    flops = None
    try:
        from thop import profile as thop_profile
        x_in = x_flat[:, :1024]
        t_in = t_flat[:, :1024]
        macs, _ = thop_profile(model, inputs=(x_in, t_in, ic_dev), verbose=False)
        flops = 2 * macs   # MACs → FLOPs
    except Exception:
        pass

    return {
        "efficiency/param_count": param_count,
        "efficiency/latency_ms":  latency_ms,
        "efficiency/flops":       flops,
    }


# ---------------------------------------------------------------------------
# Full evaluation entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run all §5 metrics on the validation set.

    Parameters
    ----------
    model      : trained model
    val_loader : DataLoader in field mode (returns ic, u, x, t)
    device     : compute device
    max_batches: limit for quick smoke tests (None = full val set)

    Returns
    -------
    dict of metric name → value
    """
    model.eval()
    all_metrics: dict[str, list] = {}

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        ic    = batch["ic"].to(device)          # (B, X)
        u_true_t = batch["u"]                   # (B, T, X)  keep on CPU
        x_norm = batch["x"][0].numpy()          # (X,)
        t_norm = batch["t"][0].numpy()          # (T,)

        # Predict on original grid
        x_t = batch["x"][0]   # (X,)
        t_t = batch["t"][0]   # (T,)
        u_pred_t = predict_field(model, ic, x_t, t_t, device)

        u_pred_np = _to_np(u_pred_t)
        u_true_np = u_true_t.numpy()

        # §5.1 Accuracy
        m = accuracy_metrics(u_pred_np, u_true_np)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

        # §5.2 Physical consistency
        dx_phys = (x_norm[-1] - x_norm[0]) / (len(x_norm) - 1)
        m = mass_conservation_metrics(u_pred_np, x_norm)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

        m = periodicity_metrics(u_pred_np)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

        # §5.4 Shock breakdown
        m = shock_region_metrics(u_pred_np, u_true_np, dx=dx_phys)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

    # Aggregate
    aggregated = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    # §5.5 Spectral (return arrays, not scalar — handled separately by visualizer)
    # Computed on last batch only as a representative sample
    aggregated["_spectral"] = spectral_metrics(u_pred_np, u_true_np, t_norm)

    return aggregated
