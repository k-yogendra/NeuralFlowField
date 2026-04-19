"""
modal_train.py — Run PINN / PINF training on Modal cloud GPUs.

Setup (one-time)
----------------
1. pip install modal
2. modal token new          # authenticate
3. modal volume create pinn-data
4. modal volume put pinn-data /path/to/1D_Burgers_Sols_Nu0.01.hdf5 /data/
5. modal volume put pinn-data /path/to/1D_Burgers_Sols_Nu0.01.hdf5.stats.npz /data/

Usage
-----
# Single run:
    modal run scripts/modal_train.py --config configs/pinn.yaml --seed 0

# Full comparison (9 runs) — runs in parallel on Modal:
    modal run scripts/modal_train.py::run_all

# Ablations:
    modal run scripts/modal_train.py::run_ablations

# Download results after training:
    modal volume get pinn-results /runs ./experiments/runs
    modal volume get pinn-results /figures ./results/figures
    modal volume get pinn-results /metrics ./results/metrics

GPU options (set GPU env var or edit gpu= below):
    A10G  — 24 GB VRAM, $1.10/h  (recommended: good price/perf, fits full batch)
    A100  — 40 GB VRAM, $3.04/h  (fastest, use for ablations if time-constrained)
    T4    — 16 GB VRAM, $0.59/h  (budget option; may need smaller batch)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app and volumes
# ---------------------------------------------------------------------------
app = modal.App("NeuralFlowField")

# Persistent volume for the 7.7 GB HDF5 dataset (upload once, reuse forever)
data_vol = modal.Volume.from_name("pinn-data", create_if_missing=True)

# Persistent volume for checkpoints, figures, and metrics
results_vol = modal.Volume.from_name("pinn-results", create_if_missing=True)

DATA_DIR    = Path("/data")
RESULTS_DIR = Path("/results")
CODE_DIR    = Path("/code")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy>=1.24",
        "scipy>=1.11",
        "h5py>=3.9",
        "omegaconf>=2.3",
        "einops>=0.7",
        "matplotlib>=3.7",
        "tensorboard>=2.14",
        "thop>=0.1.1",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu121",
    )
)

# ---------------------------------------------------------------------------
# Helper: copy code into container at runtime
# ---------------------------------------------------------------------------
# We mount the local code directory so changes are picked up without rebuilding
# the image. Alternatively, bake code into the image for fully reproducible runs.
code_mount = modal.Mount.from_local_dir(
    Path(__file__).parent.parent,   # NeuralFlowField/ directory
    remote_path=str(CODE_DIR),
    condition=lambda p: not any(
        skip in p for skip in ["__pycache__", ".pt", ".pyc", "experiments/"]
    ),
)

# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=modal.gpu.A10G(),          # change to modal.gpu.A100() for faster runs
    timeout=60 * 60 * 14,          # 14-hour max (200-epoch run is ~10–12 h)
    volumes={
        str(DATA_DIR):    data_vol,
        str(RESULTS_DIR): results_vol,
    },
    mounts=[code_mount],
    cpu=4,
    memory=16384,
)
def train(config: str, seed: int = 0) -> dict:
    """Train a single model with the given config file and seed.

    Parameters
    ----------
    config : config file name relative to NeuralFlowField/configs/ (e.g. 'pinn.yaml')
    seed   : random seed

    Returns
    -------
    dict with best_val_l2 and run_dir path on the results volume
    """
    import sys
    sys.path.insert(0, str(CODE_DIR))

    from omegaconf import OmegaConf
    from pathlib import Path
    import torch

    # Point h5_path at the volume-mounted data
    config_path = CODE_DIR / "configs" / config
    base_path   = CODE_DIR / "configs" / "base.yaml"

    base = OmegaConf.load(base_path)
    exp  = OmegaConf.load(config_path)
    for cfg in [base, exp]:
        if "defaults" in cfg:
            cfg = OmegaConf.masked_copy(cfg, [k for k in cfg if k != "defaults"])

    cfg = OmegaConf.merge(base, exp)

    # Override paths to use volume mounts
    cfg.dataset.h5_path = str(DATA_DIR / "1D_Burgers_Sols_Nu0.01.hdf5")
    cfg.logging.out_dir  = str(RESULTS_DIR / "runs")
    cfg.seed = seed

    print(OmegaConf.to_yaml(cfg))

    # --- Imports ---
    from torch.utils.data import DataLoader
    from src.dataset import BurgersPDEBench, worker_init_fn
    from src.models.pinn import build_pinn
    from src.models.pinf import build_pinf
    from src.trainer import Trainer, set_seed

    set_seed(seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")

    h5_path = cfg.dataset.h5_path
    dcfg    = cfg.dataset

    train_ds = BurgersPDEBench(h5_path, split="train", train_size=dcfg.train_size,
                                mode="points", n_points=dcfg.n_points,
                                normalize=True, seed=seed)
    val_ds   = BurgersPDEBench(h5_path, split="val", train_size=dcfg.train_size,
                                mode="field", normalize=True, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=dcfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=dcfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(train_ds)} trajectories  |  Val: {len(val_ds)} trajectories")

    # Build model
    model_cfg  = OmegaConf.to_container(cfg.model, resolve=True)
    model_name = model_cfg.get("name", "pinn")
    if model_name == "pinn":
        model = build_pinn(model_cfg)
    else:
        model = build_pinf(model_cfg)
    print(f"Model: {model_name}  |  params = {model.count_parameters():,}")

    config_stem = Path(config).stem
    run_dir = Path(cfg.logging.out_dir) / f"{config_stem}_seed{seed}"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=OmegaConf.to_container(cfg, resolve=True),
        run_dir=run_dir,
        device=device,
    )
    trainer.train()

    # Commit results to the volume so they persist after the container exits
    results_vol.commit()

    return {
        "config":       config,
        "seed":         seed,
        "best_val_l2":  trainer.best_val_l2,
        "run_dir":      str(run_dir),
    }


# ---------------------------------------------------------------------------
# Evaluate + save metrics JSON on Modal
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=modal.gpu.A10G(),
    timeout=60 * 60 * 2,
    volumes={
        str(DATA_DIR):    data_vol,
        str(RESULTS_DIR): results_vol,
    },
    mounts=[code_mount],
    cpu=4,
    memory=16384,
)
def evaluate(run_name: str) -> dict:
    """Load best checkpoint from a run and compute all §5 metrics."""
    import sys, json, torch
    sys.path.insert(0, str(CODE_DIR))

    from pathlib import Path
    from torch.utils.data import DataLoader
    from src.dataset import BurgersPDEBench
    from src.evaluator import evaluate as eval_fn, efficiency_metrics
    from scripts.evaluate import load_model_from_ckpt

    ckpt_path = RESULTS_DIR / "runs" / run_name / "checkpoints" / "best.pt"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model_from_ckpt(ckpt_path, device)

    dcfg   = cfg.get("dataset", {})
    h5_path = DATA_DIR / "1D_Burgers_Sols_Nu0.01.hdf5"
    val_ds  = BurgersPDEBench(h5_path, split="val",
                               train_size=dcfg.get("train_size", 9000),
                               mode="field", normalize=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    metrics  = eval_fn(model, val_loader, device)
    spectral = metrics.pop("_spectral", None)

    sample   = next(iter(val_loader))
    eff      = efficiency_metrics(model, sample["ic"][:1], sample["x"][0], sample["t"][0], device)
    metrics.update(eff)

    def _clean(v):
        import numpy as np
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.floating, np.integer)): return float(v)
        return v
    metrics = {k: _clean(v) for k, v in metrics.items() if v is not None}

    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.json").write_text(json.dumps(metrics, indent=2))
    results_vol.commit()

    return metrics


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(config: str = "pinn.yaml", seed: int = 0):
    """Single run entry point.

    modal run scripts/modal_train.py --config pinn.yaml --seed 0
    """
    result = train.remote(config=config, seed=seed)
    print(f"\nDone: {result}")


@app.local_entrypoint()
def run_all():
    """Launch all 9 main training runs in parallel on Modal.

    modal run scripts/modal_train.py::run_all
    """
    experiments = [
        ("pinn.yaml",         0),
        ("pinn.yaml",         1),
        ("pinn.yaml",         2),
        ("pinf_fourier.yaml", 0),
        ("pinf_fourier.yaml", 1),
        ("pinf_fourier.yaml", 2),
        ("pinf_siren.yaml",   0),
        ("pinf_siren.yaml",   1),
        ("pinf_siren.yaml",   2),
    ]

    print(f"Launching {len(experiments)} training runs in parallel...")
    # starmap dispatches all runs simultaneously — Modal schedules them across GPUs
    results = list(train.starmap(experiments))

    print("\n=== Results ===")
    for r in results:
        print(f"  {r['config']:25s} seed={r['seed']}  best_val_L2={r['best_val_l2']:.4f}")

    # Evaluate all runs
    run_names = [f"{Path(cfg).stem}_seed{seed}" for cfg, seed in experiments]
    print(f"\nEvaluating {len(run_names)} checkpoints...")
    metrics_list = list(evaluate.map(run_names))

    print("\n=== Evaluation Summary ===")
    for name, m in zip(run_names, metrics_list):
        print(f"  {name:35s}  rel_L2={m.get('acc/rel_l2', float('nan')):.4f}")


@app.local_entrypoint()
def run_ablations():
    """Launch all 15 ablation runs in parallel.

    modal run scripts/modal_train.py::run_ablations
    """
    ablations = [
        ("ablations/fourier_sigma_1.yaml",  0),
        ("ablations/fourier_sigma_1.yaml",  1),
        ("ablations/fourier_sigma_1.yaml",  2),
        ("ablations/fourier_sigma_5.yaml",  0),
        ("ablations/fourier_sigma_5.yaml",  1),
        ("ablations/fourier_sigma_5.yaml",  2),
        ("ablations/fourier_sigma_10.yaml", 0),
        ("ablations/fourier_sigma_10.yaml", 1),
        ("ablations/fourier_sigma_10.yaml", 2),
        ("ablations/siren_w0_10.yaml",      0),
        ("ablations/siren_w0_10.yaml",      1),
        ("ablations/siren_w0_10.yaml",      2),
        ("ablations/siren_w0_30.yaml",      0),
        ("ablations/siren_w0_30.yaml",      1),
        ("ablations/siren_w0_30.yaml",      2),
    ]

    print(f"Launching {len(ablations)} ablation runs in parallel...")
    results = list(train.starmap(ablations))
    print("\n=== Ablation Results ===")
    for r in results:
        print(f"  {r['config']:40s} seed={r['seed']}  best_val_L2={r['best_val_l2']:.4f}")