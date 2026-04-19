# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research comparison of **Physics-Informed Neural Fields (PINF)** vs. standard **Physics-Informed Neural Networks (PINN)** on the 1D viscous Burgers equation (ν=0.01). The scientific question: does Fourier/SIREN input encoding mitigate spectral bias that causes PINNs to fail on sharp-gradient flows?

Dataset: PDEBench `1D_Burgers_Sols_Nu0.01.hdf5` — 10,000 trajectories of shape `(T=201, X=1024)`. Splits are fixed: train 0–8999, val 9000–9999. **Do not modify `src/dataset.py`** — it is validated and frozen.

## Commands

### Setup
```bash
conda env create -f environment.yml
conda activate neuralflowfield
```

### Training
```bash
# Single run
python scripts/train.py --config configs/pinn.yaml seed=0
python scripts/train.py --config configs/pinf_fourier.yaml seed=0
python scripts/train.py --config configs/pinf_siren.yaml seed=0

# Smoke test (fast validation)
python scripts/train.py --config configs/smoke_pinn.yaml

# CLI overrides use dot-notation: key=value
python scripts/train.py --config configs/pinn.yaml train.epochs=50 seed=1

# Full 9-run comparison (3 models × 3 seeds)
bash scripts/run_all.sh
```

### Evaluation
```bash
python scripts/evaluate.py experiments/runs/pinn_seed0/checkpoints/best.pt
python scripts/evaluate.py experiments/runs/pinn_seed0/checkpoints/best.pt --max_batches 10
```

### Visualization
```bash
python scripts/visualize.py \
    --pinn   experiments/runs/pinn_seed0/checkpoints/best.pt \
    --pinf_f experiments/runs/pinf_fourier_seed0/checkpoints/best.pt \
    --pinf_s experiments/runs/pinf_siren_seed0/checkpoints/best.pt
```

### Tests
```bash
pytest tests/                          # all tests
pytest tests/test_physics_loss.py -v  # physics residual correctness
pytest tests/test_models.py -v        # forward/backward smoke tests
pytest tests/test_encoders.py -v      # shape tests
```

## Architecture

### Model hierarchy (`src/models/`)
- **`base.py`** — `BaseModel` (abstract) + `TrunkMLP`. All models share the same forward interface: `forward(x, t, ic) → u_pred` where `x, t: (B, N)` normalized to `[-1,1]`, `ic: (B, 1024)`. Subclasses implement `encode_coords(coords)` only.
- **`encoders.py`** — `ICEncoder` (1D CNN: channels [32,64,64], kernel 5, stride 2, GlobalAveragePool → latent z ∈ R^128); `FourierEncoding` (random Fourier features, frozen B matrix); `SIRENEncoding` (sinusoidal activations with ω₀ init).
- **`pinn.py`** — `PINN`: identity coordinate encoding (raw x, t → trunk). Vanilla Raissi-style baseline.
- **`pinf.py`** — `PINF`: swappable encoding selected by config (`fourier` or `siren`).

Both models are **conditional/operator-style** — they take IC as input and generalize across trajectories. Per-instance training is not the comparison mode.

**Parameter-count matching is mandatory**: PINN and PINF must match within ±5%. Adjust hidden width if needed.

### Losses (`src/losses/`)
- **`data.py`** — MSE on N=4096 sampled `(x,t)` points per trajectory.
- **`physics.py`** — Burgers PDE residual via `torch.autograd.grad` with `create_graph=True`. Coordinates are normalized; chain rule is applied: `u_t_phys = u_t_norm / (T_MAX/2)` where `T_MAX=2.01`. x_norm = x_phys (no correction). Use `physics_residual()` for raw residuals, `physics_loss()` for the MSE scalar.
- **`balancing.py`** — `GradNormBalancer` implements both fixed-weight mode and adaptive gradient-norm balancing (Wang et al. 2021). Controlled by `train.use_grad_balance` in config.

### Training (`src/trainer.py`)
Adam (lr=1e-3 → 1e-5 cosine, 200 epochs), batch size 16 trajectories. Optional L-BFGS fine-tune (`lbfgs_finetune_epochs`). Checkpoints saved every 25 epochs + best-val. Logging via TensorBoard or W&B (set `logging.backend`).

### Configuration
Configs are plain YAML files loaded via OmegaConf (not Hydra directly — the train script handles composition manually). `configs/base.yaml` is the root; experiment configs override it. Outputs go to `experiments/runs/<config_stem>_seed<N>/`. Each run saves merged config + git SHA for reproducibility.

## Key Constraints

- **Physics loss requires `create_graph=True`** — second derivatives (`u_xx`) need it.
- **Collocation points must have `requires_grad=True`** before being passed to `physics_residual`.
- **Coordinate convention**: both x and t are passed normalized to `[-1,1]`. The physics loss applies the chain rule internally. Do not pass physical-domain coordinates.
- **Dataset splits are fixed** — train_size=9000, val=indices 9000–9999. The `worker_init_fn` from `dataset.py` handles per-worker RNG seeding.
- **Deterministic runs**: `set_seed(seed, deterministic=True)` sets `torch.backends.cudnn.deterministic=True` and disables `benchmark`.

## Output Locations

- Checkpoints: `experiments/runs/<run_name>/checkpoints/`
- Metrics JSON: `results/metrics/`
- Figures (PNG + PDF): `results/figures/`
- Tables: `results/tables/`
- TensorBoard logs: `experiments/runs/<run_name>/logs/`