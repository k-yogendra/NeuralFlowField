# Physics-Informed Neural Field vs. PINN on 1D Burgers Equation

A reproducible comparison of **Physics-Informed Neural Networks (PINN)** and **Physics-Informed Neural Fields (PINF)** on the 1D viscous Burgers equation, using the PDEBench dataset. The core scientific question is whether a structured coordinate encoding (Fourier features / SIREN) mitigates the spectral bias that limits vanilla PINNs on sharp-gradient flows.

---

## Scientific Question

> Does a coordinate-based **neural field** (MLP with Fourier / SIREN input encoding) outperform a **vanilla PINN** (Raissi-style) on the 1D Burgers equation, under identical data, compute budget, and physics loss?

Three axes of comparison:
1. **Prediction accuracy** — relative L2, RMSE, L∞, nRMSE
2. **Physical consistency** — PDE residual, mass conservation, periodicity
3. **Generalization** — spatial and temporal super-resolution (up to 4×)

---

## Governing Equation

```
u_t + u · u_x = ν · u_xx,    x ∈ [-1, 1],   t ∈ [0, 2.01],   ν = 0.01
```

Periodic BC: `u(-1, t) = u(+1, t)`.

---

## Repository Structure

```
code/
├── src/
│   ├── dataset.py              # PDEBench HDF5 loader (points & field modes)
│   ├── models/
│   │   ├── encoders.py         # ICEncoder (CNN), FourierEncoding, SIRENEncoding
│   │   ├── base.py             # BaseModel + TrunkMLP (shared interface)
│   │   ├── pinn.py             # Vanilla PINN (no coordinate encoding)
│   │   └── pinf.py             # PINF (Fourier or SIREN encoding, param-matched)
│   ├── losses/
│   │   ├── data.py             # MSE data loss
│   │   ├── physics.py          # PDE residual via autograd (u_t + u·u_x - ν·u_xx)
│   │   └── balancing.py        # Fixed / adaptive gradient-norm balancing
│   ├── trainer.py              # Training loop, checkpointing, TensorBoard/W&B
│   ├── evaluator.py            # All §5 metrics (accuracy, physics, SR, spectral)
│   └── visualize.py            # All §6 figures (PNG + PDF)
├── configs/
│   ├── base.yaml               # Shared defaults (dataset, optimizer, logging)
│   ├── pinn.yaml               # Full PINN training (server)
│   ├── pinf_fourier.yaml       # Full PINF-Fourier training (server)
│   ├── pinf_siren.yaml         # Full PINF-SIREN training (server)
│   ├── local_pinn.yaml         # Local GPU run (500 traj, 10 epochs)
│   ├── local_pinf_fourier.yaml
│   ├── local_pinf_siren.yaml
│   ├── smoke_pinn.yaml         # Quick smoke test (100 traj, 5 epochs)
│   ├── smoke_pinf_fourier.yaml
│   ├── smoke_pinf_siren.yaml
│   └── ablations/
│       ├── fourier_sigma_1.yaml
│       ├── fourier_sigma_5.yaml
│       ├── fourier_sigma_10.yaml
│       ├── siren_w0_10.yaml
│       └── siren_w0_30.yaml
├── scripts/
│   ├── train.py                # Training entry point (no Hydra dependency)
│   ├── evaluate.py             # Load checkpoint → metrics JSON
│   ├── visualize.py            # Load checkpoints → all figures
│   └── run_all.sh              # Full reproduction script (server)
├── tests/
│   ├── test_encoders.py        # Shape / grad / init tests for all encoders
│   ├── test_models.py          # Forward / backward / param-count tests
│   └── test_physics_loss.py    # PDE residual on known analytic solutions
├── experiments/runs/           # Training outputs (checkpoints, TensorBoard logs)
├── results/
│   ├── figures/                # PNG + PDF figures
│   ├── metrics/                # JSON metric files per run
│   └── tables/                 # CSV / Markdown result tables
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Setup

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate pinn-pinf
```

### Option B — pip

```bash
pip install -r requirements.txt
```

**Tested with:** Python 3.14.1 · PyTorch 2.9.1+cu130 · CUDA 13.0

> **Note:** Hydra config management requires Python ≤ 3.13. The training script
> (`scripts/train.py`) uses plain OmegaConf and works on any Python version.

### Dataset

Download the PDEBench 1D Burgers dataset and place it at:

```
NeuralFlowField/data/1D_Burgers_Sols_Nu0.01.hdf5
```

Dataset: [PDEBench on HuggingFace](https://huggingface.co/datasets/pdebench/PDEBench) — file `1D_Burgers_Sols_Nu0.01.hdf5`  
Shape: `(10000, 201, 1024)` float32 — 10,000 trajectories, 201 time steps, 1024 spatial points.  
Stats (pre-computed): mean ≈ −0.012, std ≈ 0.635, range [−3.35, +3.41].

---

## Models

All models share the same interface:

```python
u_pred = model(x, t, ic)
# x, t : (B, N)  normalized coordinates in [-1, 1]
# ic   : (B, X)  normalized initial condition (X = 1024)
# out  : (B, N)  predicted solution
```

Both models are **conditional** — they take the IC as input via a shared CNN encoder, enabling generalization across trajectories (operator-style learning).

| Component | PINN | PINF-Fourier | PINF-SIREN |
|---|---|---|---|
| IC encoder | 1D CNN [32,64,64], k=5, GAP → z∈R¹²⁸ | same | same |
| Coord encoding | none (raw x, t) | γ(v) = [sin(2πBv), cos(2πBv)], B∼N(0,σ²), frozen | sin(ω₀ · Wx + b), Sitzmann init |
| Trunk | 6 × 128, tanh | 6 × 128, tanh | 6 × 128, tanh |
| Params | ~139K | ~140K (±0.9%) | ~140K (±0.1%) |

Parameter counts are matched within ±5% (spec requirement).

---

## Training

### Smoke test — verify everything runs (< 2 min)

```bash
cd NeuralFlowField/
python scripts/train.py --config configs/smoke_pinn.yaml
python scripts/train.py --config configs/smoke_pinf_fourier.yaml
python scripts/train.py --config configs/smoke_pinf_siren.yaml
```

### Local GPU run — verify convergence trend (GTX 1650, ~10 min/model)

```bash
python scripts/train.py --config configs/local_pinn.yaml
python scripts/train.py --config configs/local_pinf_fourier.yaml
python scripts/train.py --config configs/local_pinf_siren.yaml
```

### Full training — Modal cloud (recommended, 9 runs in parallel)

See [Modal section](#modal-cloud-training) below.

### Full training — local server (200 epochs, 9000 trajectories)

```bash
bash scripts/run_all.sh
# or individually, with seed override:
python scripts/train.py --config configs/pinn.yaml seed=0
python scripts/train.py --config configs/pinn.yaml seed=1
python scripts/train.py --config configs/pinn.yaml seed=2
python scripts/train.py --config configs/pinf_fourier.yaml seed=0
python scripts/train.py --config configs/pinf_siren.yaml seed=0
```

### CLI overrides

Any config key can be overridden from the command line using `key=value` or `nested.key=value`:

```bash
python scripts/train.py --config configs/pinn.yaml seed=1 train.epochs=50 train.lambda_phys=0.5
```

### Training details

| Setting | Value |
|---|---|
| Optimizer | Adam, β=(0.9, 0.999) |
| Learning rate | 1e-3 → 1e-5, cosine decay |
| Epochs | 200 |
| Batch size | 16 trajectories |
| Data points per step | 16 × 4096 = 65,536 |
| Collocation points | 16 × 2048 = 32,768 |
| Seeds | {0, 1, 2} |
| Optional | L-BFGS fine-tune (final 20 epochs), adaptive loss balancing |

### Physics loss

The PDE residual is computed via automatic differentiation:

```
r(x, t) = u_t + u · u_x − ν · u_xx
L_phys  = mean(r²)    over collocation points (no GT labels needed)
```

Chain rule is applied for the `[-1,1] → [0, 2.01]` t-normalization.

---

## Evaluation

```bash
python scripts/evaluate.py experiments/runs/local_pinn_seed0/checkpoints/best.pt
# optional: limit val batches for speed
python scripts/evaluate.py path/to/best.pt --max_batches 50
```

Outputs a JSON to `results/metrics/<run_name>.json`. Metrics computed:

| Category | Metrics |
|---|---|
| **Accuracy** | Relative L2, RMSE, L∞, nRMSE |
| **Physical** | Mass conservation drift, periodicity error |
| **Shock breakdown** | Relative L2 in shock vs. smooth regions (τ=0.3) |
| **Efficiency** | Param count, inference latency (ms/trajectory) |

### Local PINN result (10 epochs, 500 train trajectories)

| Metric | Value |
|---|---|
| rel-L2 | 0.615 |
| RMSE | 0.399 |
| L∞ | 1.455 |
| nRMSE | 0.167 |
| Mass drift | 0.007 |
| Periodicity error | 0.003 |
| Shock rel-L2 | 0.635 |
| Smooth rel-L2 | 0.616 |
| Inference latency | 450 ms (CPU) |

> Full results (200 epochs, 3 seeds × 3 models) to be added after server runs.

---

## Visualization

```bash
python scripts/visualize.py \
    --pinn   experiments/runs/pinn_seed0/checkpoints/best.pt \
    --pinf_f experiments/runs/pinf_fourier_seed0/checkpoints/best.pt \
    --pinf_s experiments/runs/pinf_siren_seed0/checkpoints/best.pt \
    --n_traj 3
```

Figures saved to `results/figures/` (PNG + PDF):

| Figure | Description |
|---|---|
| `fig1_spacetime_grid` | u(x,t) heatmaps: 3 trajectories × [GT \| PINN \| PINF-F \| PINF-S] |
| `fig2_error_heatmaps` | \|u_pred − u_true\| on log scale |
| `fig3_snapshots` | Line plots of u(x) at t ∈ {0, 0.5, 1.0, 1.5, 2.0} |
| `fig4_shock_closeup` | Zoom on steepest gradient region at t ≈ 1.0 |
| `fig5_super_resolution` | 4× spatial SR comparison vs. spline GT |
| `fig6_error_spectrum` | Log-log error power spectrum vs. wavenumber k |
| `fig7_convergence` | Loss and val rel-L2 vs. epoch (3 seeds overlaid) |
| `fig8_characteristics` | Lagrangian characteristics dx/dt = u(x,t) |

---

## Ablations

Fourier σ and SIREN ω₀ ablations are in `configs/ablations/`. Run via:

```bash
python scripts/train.py --config configs/ablations/fourier_sigma_1.yaml seed=0
python scripts/train.py --config configs/ablations/fourier_sigma_5.yaml seed=0
python scripts/train.py --config configs/ablations/fourier_sigma_10.yaml seed=0
python scripts/train.py --config configs/ablations/siren_w0_10.yaml seed=0
python scripts/train.py --config configs/ablations/siren_w0_30.yaml seed=0
```

---

## Tests

```bash
cd NeuralFlowField/
python -m pytest tests/ -v
# 35 tests, all passing
```

| Test file | What it checks |
|---|---|
| `test_encoders.py` | Output shapes, frozen Fourier buffer, SIREN init range, grad flow |
| `test_models.py` | Forward/backward, ±5% param-count match, IC-conditional output |
| `test_physics_loss.py` | PDE residual = 0 on zero/constant solutions; correct scaling with amplitude |

---

## TensorBoard

```bash
tensorboard --logdir experiments/runs/
```

Logged per epoch: `loss/data`, `loss/phys`, `loss/total`, `val/rel_l2`, `val/rmse`, `lr`, `grad_norm`, `wall_clock`.

---

## Hardware Notes

| Config | GPU | Time/epoch | Total |
|---|---|---|---|
| `smoke_*.yaml` | GTX 1650 | ~12 s | ~1 min |
| `local_*.yaml` | GTX 1650 | ~54 s | ~10 min |
| `*.yaml` (full) | GTX 1650 | ~3–6 h | not feasible |
| `*.yaml` (full) | A100 / V100 | ~2–5 min | ~7–17 h (×9 runs) |

The bottleneck is the physics loss: `create_graph=True` stores the full computation graph for `u_xx` (second-order autodiff) over all collocation points. VRAM usage scales with `batch_size × n_colloc`.

---

## Modal Cloud Training

`scripts/modal_train.py` wraps the training pipeline for [Modal](https://modal.com) — a serverless GPU cloud. All 9 main runs are dispatched **in parallel** so the wall-clock time equals one run, not nine.

### One-time setup

```bash
pip install modal
modal token new                  # authenticate in browser

# Create persistent volumes (data uploaded once, reused across all runs)
modal volume create pinn-data
modal volume create pinn-results

# Upload the 7.7 GB HDF5 dataset
modal volume put pinn-data /path/to/1D_Burgers_Sols_Nu0.01.hdf5 /data/
modal volume put pinn-data /path/to/1D_Burgers_Sols_Nu0.01.hdf5.stats.npz /data/
```

### Run experiments

```bash
cd NeuralFlowField/

# Test a single run first
modal run scripts/modal_train.py --config pinn.yaml --seed 0

# All 9 main runs in parallel (9 GPUs, wall-clock = 1 run)
modal run scripts/modal_train.py::run_all

# All 15 ablation runs in parallel
modal run scripts/modal_train.py::run_ablations
```

### Download results

```bash
modal volume get pinn-results /runs     ./experiments/runs
modal volume get pinn-results /metrics  ./results/metrics
modal volume get pinn-results /figures  ./results/figures
```

### Cost estimate (A10G, $1.10/hr)

| Workload | Runs | Est. time/run | Est. total cost |
|---|---|---|---|
| Main experiments | 9 | ~10–12 h | ~$119 |
| Ablations | 15 | ~5–6 h | ~$91 |
| **Grand total** | 24 | | **~$210** |

Well within $280 budget, with ~$70 buffer.  
Switch to `modal.gpu.A100()` in `modal_train.py` for ~3× faster runs at ~3× cost (~$180 total).

> **Tip:** Run a single seed first (`seed=0`) to confirm convergence before launching all 3 seeds.

---

## Reference

Zhao et al. 2024, *Physics of Fluids* — "Review of physics-informed neural networks: scope, applications, challenges."  
Tancik et al. 2020 — Fourier Features (random feature encoding).  
Sitzmann et al. 2020 — SIREN (implicit neural representations with periodic activations).  
Wang et al. 2021 — Gradient-norm balancing for PINN training.  
PDEBench — Takamoto et al. 2022.# NeuralFlowField
