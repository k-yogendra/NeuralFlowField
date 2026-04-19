"""
train.py — Training entry point for PINN / PINF.

Does NOT require Hydra (compatible with Python 3.14+).
Configs are plain YAML files loaded via OmegaConf.
Command-line overrides use dot-notation: key=value or nested.key=value.

Usage
-----
    python scripts/train.py --config configs/smoke_pinn.yaml
    python scripts/train.py --config configs/pinn.yaml seed=1
    python scripts/train.py --config configs/pinf_fourier.yaml train.epochs=200 seed=0

Outputs go to experiments/runs/<config_stem>_seed<N>/
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Allow imports from NeuralFlowField/
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader

from src.dataset import BurgersPDEBench, worker_init_fn
from src.models.pinn import build_pinn
from src.models.pinf import build_pinf
from src.trainer import Trainer, set_seed


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_BASE_CFG = Path(__file__).parent.parent / "configs" / "base.yaml"


def load_config(config_path: str, overrides: list[str]) -> DictConfig:
    """Load base + experiment config, then apply CLI overrides."""
    base = OmegaConf.load(_BASE_CFG)
    exp  = OmegaConf.load(config_path)

    # Remove Hydra 'defaults' key — not needed outside Hydra
    if "defaults" in exp:
        exp = OmegaConf.masked_copy(exp, [k for k in exp if k != "defaults"])
    if "defaults" in base:
        base = OmegaConf.masked_copy(base, [k for k in base if k != "defaults"])

    cfg = OmegaConf.merge(base, exp)

    # Apply dot-notation CLI overrides: key=value or nested.key=value
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        # Try to parse as number/bool, fall back to string
        try:
            val = OmegaConf.create({key: int(val)})[key]
        except (ValueError, Exception):
            try:
                val = OmegaConf.create({key: float(val)})[key]
            except (ValueError, Exception):
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
        override = OmegaConf.from_dotlist([f"{key}={val}"])
        cfg = OmegaConf.merge(cfg, override)

    return cfg


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(cfg: DictConfig):
    model_cfg  = OmegaConf.to_container(cfg.model, resolve=True)
    model_name = model_cfg.get("name", "pinn")

    if model_name == "pinn":
        model = build_pinn(model_cfg)
        print(f"[model] PINN  |  params = {model.count_parameters():,}")
        return model

    elif model_name == "pinf":
        model = build_pinf(model_cfg)
        enc_type = model_cfg.get("encoding", {}).get("type", "?")
        print(f"[model] PINF-{enc_type}  |  params = {model.count_parameters():,}")
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PINN or PINF on 1D Burgers")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args, extra = parser.parse_known_args()   # extra = list of key=value overrides

    cfg = load_config(args.config, extra)
    print(OmegaConf.to_yaml(cfg))

    # Seeding
    set_seed(cfg.seed, deterministic=True)

    # Paths
    code_dir = Path(__file__).parent.parent
    h5_path  = code_dir / cfg.dataset.h5_path

    # Run output directory
    config_stem = Path(args.config).stem
    run_dir = code_dir / cfg.logging.out_dir / f"{config_stem}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save merged config
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Git SHA for reproducibility
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(code_dir)
        ).decode().strip()
        (run_dir / "git_sha.txt").write_text(sha)
    except Exception:
        pass

    # Dataset
    dcfg = cfg.dataset
    train_ds = BurgersPDEBench(
        h5_path, split="train", train_size=dcfg.train_size,
        mode="points", n_points=dcfg.n_points, normalize=True, seed=cfg.seed,
    )
    val_ds = BurgersPDEBench(
        h5_path, split="val", train_size=dcfg.train_size,
        mode="field", normalize=True, seed=cfg.seed,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=dcfg.batch_size, shuffle=True,
        num_workers=dcfg.num_workers, pin_memory=pin,
        worker_init_fn=worker_init_fn if dcfg.num_workers > 0 else None,
        persistent_workers=dcfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=dcfg.batch_size, shuffle=False,
        num_workers=dcfg.num_workers, pin_memory=pin,
        persistent_workers=dcfg.num_workers > 0,
    )

    print(f"Train: {len(train_ds)} trajectories  |  Val: {len(val_ds)} trajectories")

    # Model
    model = build_model(cfg)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=OmegaConf.to_container(cfg, resolve=True),
        run_dir=run_dir,
    )
    trainer.train()
    print(f"\nCheckpoints saved to: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
