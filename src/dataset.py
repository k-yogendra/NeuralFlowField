"""
PDEBench 1D Burgers dataset loader for neural-field training.

Features
--------
- Fork-safe HDF5 handling (works with DataLoader num_workers > 0).
- Dataset-wide normalization stats computed on first use, cached to disk.
- Coordinate normalization baked in (x, t -> [-1, 1]).
- Two modes:
    "points": returns sampled (x,t,u) triples per trajectory -- for coord-based
              neural fields (SIREN / Fourier features / hash grid).
    "field":  returns full (T, X) solution grid -- for FNO / U-Net / sanity checks.

File: 1D_Burgers_Sols_Nu0.01.hdf5
  tensor:        (10000, 201, 1024) float32
  x-coordinate:  (1024,)   -- cell-centered on [-1, 1], periodic
  t-coordinate:  (201,)    -- [0, 2.01]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Stats: one-pass, chunked, cached to disk
# ---------------------------------------------------------------------------
def compute_or_load_stats(
    h5_path: Union[str, Path],
    cache_path: Optional[Union[str, Path]] = None,
    chunk: int = 200,
    verbose: bool = True,
) -> dict:
    """Compute (mean, std, min, max) over the full `tensor` field.

    Uses float64 accumulators. ~30s one-time cost over 2 * 10^9 values.
    Caches to <h5_path>.stats.npz unless an explicit cache_path is given.
    """
    h5_path = Path(h5_path)
    if cache_path is None:
        cache_path = h5_path.with_suffix(h5_path.suffix + ".stats.npz")
    cache_path = Path(cache_path)

    if cache_path.exists():
        s = np.load(cache_path)
        return {k: float(s[k]) for k in s.files}

    if verbose:
        print(f"[stats] computing over {h5_path.name} (one-time)...")

    s1 = 0.0            # sum
    s2 = 0.0            # sum of squares
    n = 0
    u_min, u_max = np.inf, -np.inf

    with h5py.File(h5_path, "r") as f:
        tensor = f["tensor"]
        N = tensor.shape[0]
        for i in range(0, N, chunk):
            x = tensor[i:i + chunk].astype(np.float64)
            s1 += x.sum()
            s2 += (x * x).sum()
            n += x.size
            u_min = min(u_min, float(x.min()))
            u_max = max(u_max, float(x.max()))

    mean = s1 / n
    var = max(s2 / n - mean * mean, 0.0)
    std = float(np.sqrt(var))

    stats = dict(
        mean=float(mean), std=std,
        u_min=u_min, u_max=u_max, n=int(n),
    )
    np.savez(cache_path, **stats)
    if verbose:
        print(f"[stats] mean={stats['mean']:+.5f}  std={stats['std']:.5f}  "
              f"range=[{stats['u_min']:+.4f}, {stats['u_max']:+.4f}]  "
              f"cached -> {cache_path.name}")
    return stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BurgersPDEBench(Dataset):
    """
    PDEBench 1D Burgers dataset.

    Parameters
    ----------
    h5_path     : path to 1D_Burgers_Sols_Nu*.hdf5
    split       : "train" | "val" | "all"
    train_size  : number of trajectories in train split (default 9000, convention)
    mode        : "points" | "field"
    n_points    : number of (x,t) samples per trajectory in "points" mode
    normalize   : if True, standardize u to zero mean / unit std (dataset-wide)
    seed        : base RNG seed (each DataLoader worker gets a derived seed)

    Returns (per __getitem__)
    -------
    mode="points":
        ic      (X,)      float32   initial condition at 1024 grid points
        coords  (N, 2)    float32   (x_norm, t_norm) in [-1, 1]^2
        values  (N,)      float32   u at those coords (normalized if normalize=True)
        idx     int                 original trajectory index in the h5 file

    mode="field":
        ic      (X,)      float32
        u       (T, X)    float32   full solution (normalized if normalize=True)
        x       (X,)      float32   x grid, in [-1, 1]
        t       (T,)      float32   t grid, normalized to [-1, 1]
        idx     int
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        split: str = "train",
        train_size: int = 9000,
        mode: str = "points",
        n_points: int = 4096,
        normalize: bool = True,
        seed: int = 0,
    ):
        assert split in ("train", "val", "all")
        assert mode in ("points", "field")
        self.h5_path = str(h5_path)
        self.mode = mode
        self.n_points = int(n_points)
        self.normalize = bool(normalize)
        self.seed = int(seed)

        # --- Stats (cached) ---
        self.stats = compute_or_load_stats(self.h5_path) if normalize else None

        # --- Read small metadata arrays eagerly, close file ---
        with h5py.File(self.h5_path, "r") as f:
            tensor = f["tensor"]
            N, self.T, self.X = tensor.shape
            x_coord = f["x-coordinate"][:].astype(np.float32)    # already [-1, 1]
            t_coord = f["t-coordinate"][:].astype(np.float32)    # [0, 2.01]

        # PDEBench quirk: t-coordinate has T+1 entries (cell edges), tensor
        # has T time slices. Truncate to match. Same guard for x.
        if t_coord.shape[0] != self.T:
            t_coord = t_coord[:self.T]
        if x_coord.shape[0] != self.X:
            x_coord = x_coord[:self.X]

        # --- Normalized coordinate grids ---
        self.x_norm = x_coord                                    # (X,)  already [-1, 1]
        t_max = float(t_coord.max())
        self.t_norm = (t_coord / t_max * 2.0 - 1.0).astype(np.float32)   # -> [-1, 1]

        # --- Split indices ---
        if split == "train":
            self.indices = np.arange(0, train_size)
        elif split == "val":
            self.indices = np.arange(train_size, N)
        else:
            self.indices = np.arange(0, N)

        # --- Lazy per-worker state ---
        self._h5 = None
        self._tensor = None
        self._rng = np.random.default_rng(self.seed)

    # --- Fork-safe HDF5 handling --------------------------------------------
    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._tensor = self._h5["tensor"]

    def __getstate__(self):
        # Drop unpicklable h5 handles before the worker fork.
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_tensor"] = None
        return state

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    # --- Normalization helpers ----------------------------------------------
    def normalize_u(self, u: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return u
        return (u - self.stats["mean"]) / self.stats["std"]

    def denormalize_u(self, u):
        if not self.normalize:
            return u
        return u * self.stats["std"] + self.stats["mean"]

    # --- Length / item ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        self._open_h5()
        traj_idx = int(self.indices[idx])
        u_full = self._tensor[traj_idx].astype(np.float32)       # (T, X)
        ic = u_full[0].copy()                                     # (X,)

        if self.normalize:
            u_full = self.normalize_u(u_full)
            ic = self.normalize_u(ic)

        if self.mode == "field":
            return {
                "ic":  torch.from_numpy(ic),
                "u":   torch.from_numpy(u_full),
                "x":   torch.from_numpy(self.x_norm),
                "t":   torch.from_numpy(self.t_norm),
                "idx": traj_idx,
            }

        # mode == "points": uniform random (t, x) sampling from the trajectory
        t_idx = self._rng.integers(0, self.T, size=self.n_points)
        x_idx = self._rng.integers(0, self.X, size=self.n_points)
        coords = np.stack([self.x_norm[x_idx], self.t_norm[t_idx]], axis=-1)  # (N, 2)
        values = u_full[t_idx, x_idx]                                         # (N,)

        return {
            "ic":     torch.from_numpy(ic),
            "coords": torch.from_numpy(coords.astype(np.float32)),
            "values": torch.from_numpy(values.astype(np.float32)),
            "idx":    traj_idx,
        }


# ---------------------------------------------------------------------------
# Worker init: each DataLoader worker gets an independent RNG
# ---------------------------------------------------------------------------
def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    base_seed = (info.seed + worker_id) % (2 ** 32)
    info.dataset._rng = np.random.default_rng(base_seed)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H5 = "../data/1D_Burgers_Sols_Nu0.01.hdf5"     # relative to src/ (data/ is inside NeuralFlowField/)

    train_ds = BurgersPDEBench(H5, split="train", mode="points", n_points=4096)
    val_ds   = BurgersPDEBench(H5, split="val",   mode="points", n_points=4096)
    print(f"train: {len(train_ds)} trajectories   val: {len(val_ds)}")
    print(f"stats: {train_ds.stats}")

    train_dl = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    batch = next(iter(train_dl))
    print("\n[points] batch:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:7s}: shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:7s}: {v if not hasattr(v, 'shape') else tuple(v.shape)}")

    # Field mode: one-off full trajectory
    field_ds = BurgersPDEBench(H5, split="val", mode="field")
    sample = field_ds[0]
    print("\n[field] single sample:")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:7s}: shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:7s}: {v}")