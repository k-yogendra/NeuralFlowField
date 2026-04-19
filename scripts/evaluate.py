"""
evaluate.py — Load a checkpoint and run all §5 metrics.

Usage
-----
    python scripts/evaluate.py <checkpoint.pt> [--config <config.yaml>] [--max_batches N]

Outputs
-------
    results/metrics/<run_name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import BurgersPDEBench
from src.models.pinn import build_pinn
from src.models.pinf import build_pinf
from src.evaluator import evaluate, efficiency_metrics
from src.trainer import load_checkpoint


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["cfg"]

    model_cfg  = cfg.get("model", {})
    model_name = model_cfg.get("name", "pinn")

    if model_name == "pinn":
        model = build_pinn(model_cfg)
    elif model_name == "pinf":
        model = build_pinf(model_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit val batches (for quick smoke tests)")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = load_model_from_ckpt(ckpt_path, device)
    print(f"Model params: {model.count_parameters():,}")

    # Dataset
    dcfg    = cfg.get("dataset", {})
    h5_path = Path(__file__).parent.parent / dcfg.get("h5_path", "data/1D_Burgers_Sols_Nu0.01.hdf5")
    val_ds  = BurgersPDEBench(
        h5_path, split="val", train_size=dcfg.get("train_size", 9000),
        mode="field", normalize=True,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    # Core evaluation
    print("Running evaluation...")
    metrics = evaluate(model, val_loader, device, max_batches=args.max_batches)

    # Remove non-serialisable spectral arrays from the scalar JSON
    spectral = metrics.pop("_spectral", None)

    # Efficiency metrics
    sample = next(iter(val_loader))
    ic_1  = sample["ic"][:1]
    x_g   = sample["x"][0]
    t_g   = sample["t"][0]
    eff   = efficiency_metrics(model, ic_1, x_g, t_g, device)
    metrics.update(eff)

    # Convert any remaining non-serialisable values
    def _clean(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v
    metrics = {k: _clean(v) for k, v in metrics.items() if v is not None}

    # Save
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent.parent / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ckpt_path.parent.parent.name}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics written to {out_path}")

    # Print summary
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<40s} {v:.6f}")


if __name__ == "__main__":
    main()
