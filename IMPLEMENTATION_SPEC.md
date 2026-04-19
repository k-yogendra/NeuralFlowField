# Physics-Informed Neural Field vs. PINN on 1D Burgers — Implementation Specification

**Target implementer:** Claude (Code).
**Audience of results:** research scholar producing publishable comparison.
**Status of dataset:** `dataset.py` already implemented and validated against `1D_Burgers_Sols_Nu0.01.hdf5` (PDEBench).

---

## 0. Scientific question

Does a coordinate-based **physics-informed neural field (PINF)** — plain MLP backbone wrapped in a structured input encoding (Fourier features / SIREN) — outperform a **standard physics-informed neural network (PINN)** on the 1D viscous Burgers equation, under identical data, identical compute budget, and identical physics loss?

The hypothesis is that the encoding in a PINF mitigates the MLP spectral bias that the project review paper (Zhao et al. 2024, *Phys. Fluids*) identifies as a primary failure mode of PINNs on sharp-gradient and multi-scale flows.

**Three axes of comparison (all required):**
1. **Prediction accuracy** — relative L2, RMSE, L∞ on held-out initial conditions.
2. **Physical consistency** — PDE residual, mass conservation, periodicity.
3. **Generalization** — spatial and temporal super-resolution at test time.

---

## 1. Governing equation and dataset

**Equation.** Viscous Burgers:

```
u_t + u · u_x = ν · u_xx,     x ∈ [-1, 1],   t ∈ [0, 2.01],   ν = 0.01
```

Periodic BC: `u(-1, t) = u(+1, t)`.

**Dataset.** PDEBench `1D_Burgers_Sols_Nu0.01.hdf5`.
- 10,000 trajectories, each `(T=201, X=1024)` float32.
- Loader in `dataset.py` provides `"points"` mode (for neural-field training) and `"field"` mode (for full-grid evaluation).
- Stats cached: `mean ≈ -0.012`, `std ≈ 0.635`, `range ≈ [-3.35, +3.41]`.
- Splits: train 0–8999, val 9000–9999 (fixed, do not shuffle).

Coordinates are pre-normalized to `[-1, 1]` for both x and t inside the Dataset.

---

## 2. Models

Both models share the same interface:

```python
class BaseModel(nn.Module):
    def forward(self, x, t, ic):           # predict u
        # x:  (B, N)  float32   in [-1, 1]
        # t:  (B, N)  float32   in [-1, 1]
        # ic: (B, X=1024)       normalized
        # returns: (B, N)
```

Both are **conditional** — they take the initial condition as input via an IC encoder so they can generalize across trajectories. Per-instance (non-conditional) training is **not** the comparison mode; we are comparing operator-style learning.

### 2.1 PINN baseline (`src/models/pinn.py`)

- **IC encoder:** 1D CNN, 3 layers, channels [32, 64, 64], kernel 5, stride 2, GlobalAveragePool → latent `z ∈ R^128`.
- **Trunk:** fully-connected MLP with `tanh` activations.
  - Input: `[x, t, z]` → concatenated to `R^(2 + 128)`.
  - Hidden: 6 layers × 128 units.
  - Output: scalar `u`.
- **No positional/Fourier encoding on (x, t).** This is the deliberate "vanilla PINN" baseline that the project review paper describes as the Raissi-style architecture.

### 2.2 Physics-Informed Neural Field (`src/models/pinf.py`)

Identical IC encoder and trunk width/depth as PINN, but with a **swappable input encoding** on `(x, t)`:

- **Fourier features** (`encoding=fourier`):
  - `γ(v) = [sin(2π · B · v), cos(2π · B · v)]`, `B ∈ R^(m × 2)` sampled from `N(0, σ²)`, frozen.
  - Hyperparams: `m = 64` (so encoded dim = 128), `σ ∈ {1, 5, 10}` — ablate.
- **SIREN** (`encoding=siren`):
  - Sinusoidal activations with ω₀ applied to first layer, 1.0 elsewhere.
  - Specialized init per Sitzmann et al. 2020.
  - Hyperparam: `ω₀ ∈ {10, 30}` — ablate. Start with 30.
- **(Optional, stretch goal)** hash-grid (`encoding=hashgrid`): multiresolution hash encoding (Müller et al. 2022). Gotcha: second derivatives are noisy or zero through hash interpolation — physics residual loss breaks. Only enable if dropping the physics loss for this variant is acceptable, or use finite-difference residuals.

### 2.3 Parameter-count matching (mandatory)

After constructing both models, count trainable parameters. They must match within **±5%**. If PINF has more params due to the encoding layer, compensate by reducing PINN hidden width or PINF hidden width symmetrically. Report final param counts.

---

## 3. Losses

### 3.1 Data loss (both models)

On each batch, sample `N=4096` random `(x_i, t_i)` per trajectory (already handled by the dataset's `"points"` mode):

```
L_data = (1/BN) · Σ |u_pred(x_i, t_i) - u_true(x_i, t_i)|²
```

### 3.2 Physics loss (both models)

Collocation points: sample an additional `M=2048` random `(x_c, t_c)` per trajectory (no ground-truth labels needed). Compute via `torch.autograd.grad`:

```
r(x, t) = u_t + u · u_x − ν · u_xx
L_phys = (1/BM) · Σ r(x_c, t_c)²
```

Use `create_graph=True` during training. Derivatives are taken in **physical coordinates**, not normalized coordinates — apply chain rule for the `[-1,1] → [0, 2.01]` rescaling on t (and analogous for x, which already matches physical domain).

### 3.3 Total loss and weighting

```
L = λ_data · L_data + λ_phys · L_phys
```

- Initial weights: `λ_data = 1.0`, `λ_phys = 0.1`.
- Enable **gradient-norm balancing** (Wang et al. 2021, "Respecting causality is all you need for training physics-informed neural networks" — ref. 231 in project review) as a training-time option. Implement both fixed-weight and adaptive-weight modes; compare in an ablation.

### 3.4 Periodicity loss (optional, both models)

Add `L_bc = (1/B) · Σ |u_pred(-1, t_k) - u_pred(+1, t_k)|²` on a small set of random `t_k`. Default off; enable as ablation.

---

## 4. Training protocol

### 4.1 Fairness constraints

- Same IC encoder architecture for both models.
- Same trunk depth and width (modulo the ±5% param rule above).
- Same optimizer, same schedule, same batch size, same epochs, same physics-loss weights.
- Same seed set across both models within a run; different seeds across runs.

### 4.2 Optimizer and schedule

- **Adam**, `lr = 1e-3`, `β = (0.9, 0.999)`.
- Cosine decay to `1e-5` over 200 epochs.
- Optional L-BFGS fine-tune for the final 20 epochs (project review notes L-BFGS is standard end-stage for PINNs — ref. 96, 97). Report results with and without.
- **Batch size:** 16 trajectories × 4096 points = 65,536 `(x,t)` training points per step.
- **Epochs:** 200. Target ≥95% of final val L2 by epoch 150.

### 4.3 Seeds and repeats

- **Seeds:** `{0, 1, 2}`. Report mean ± std for all metrics.
- Deterministic dataloader (the `worker_init_fn` in `dataset.py` handles this).

### 4.4 Checkpointing and logging

- Save model every 25 epochs + best-val checkpoint.
- Log to **TensorBoard** (default) and optionally **Weights & Biases** (toggle via config).
- Logged quantities per epoch: `L_data`, `L_phys`, `L_total`, `val_rel_L2`, `val_RMSE`, `grad_norm`, `lr`, wall-clock.

---

## 5. Evaluation protocol

All metrics computed on the 1000 val trajectories at the **full (201, 1024) grid** unless stated otherwise.

### 5.1 Accuracy

| Metric | Definition | Notes |
|---|---|---|
| **Relative L2** | `‖u_pred − u_true‖₂ / ‖u_true‖₂` per trajectory, then averaged | Primary headline metric. |
| **RMSE** | `sqrt(mean((u_pred − u_true)²))` | Absolute scale. |
| **L∞** | `max |u_pred − u_true|` | Shock-sensitive. |
| **nRMSE** | `RMSE / (max(u_true) − min(u_true))` | For comparison with PDEBench tables. |

### 5.2 Physical consistency

- **PDE residual L2:** evaluate `r(x, t) = u_t + u·u_x − ν·u_xx` on a dense `(256, 1024)` grid via autodiff. Report `‖r‖₂` mean and max across val set.
- **Mass conservation:** for each trajectory, `|∫ u(x, t) dx − ∫ u(x, 0) dx|` as a function of t (trapezoidal rule on the grid). Report max drift over t ∈ [0, 2.01]. For periodic Burgers this should be ≈ 0.
- **Periodicity:** `|u_pred(−1, t) − u_pred(+1, t)|` averaged over a grid of t values.
- **(Stretch)** Energy dissipation: `dE/dt = −ν · ∫ (u_x)² dx` with `E(t) = ½ ∫ u² dx` — the model's predicted dissipation rate should match the PDE's.

### 5.3 Generalization (super-resolution)

The dataset provides grid `(201, 1024)`. Evaluate each model at:
- **Spatial SR:** `(201, 2048)` and `(201, 4096)` — interpolate ground truth via cubic spline for comparison; report relative L2 vs. GT.
- **Temporal SR:** `(401, 1024)` and `(801, 1024)` — query the model at intermediate t values. Compare against a reference solution obtained by re-running a high-resolution numerical solver on the same ICs (or, as a proxy, cubic-spline interpolation of the 201-step ground truth).
- **Joint SR:** `(801, 4096)`.

This is the single most important test for the neural-field claim. Expect PINN to degrade more than PINF at high SR factors if the spectral-bias hypothesis holds.

### 5.4 Shock-region vs smooth-region breakdown

- Define shock mask: `|u_x| > τ · max|u_x|` per trajectory, with `τ = 0.3`.
- Report relative L2 separately in shock and smooth regions.
- Plot error-vs-`|u_x|` scatter (bucketed) for both models.

### 5.5 Spectral analysis

- Compute FFT of the **error field** `e(x, t) = u_pred − u_true` along x for each t.
- Plot log-log power spectrum `|ê(k, t)|²` vs. wavenumber `k`, averaged across val trajectories, at fixed t = 0.5, 1.0, 1.5.
- Hypothesis to confirm or reject: PINN error is concentrated at high k (spectral bias); PINF error spectrum is flatter.

### 5.6 Efficiency

- **Parameter count.**
- **Training wall-clock** per epoch and total.
- **Inference latency** for a full (201, 1024) field on a single GPU. Batch size 1, report ms per trajectory.
- **FLOPs** per forward pass (use `fvcore` or `thop`).

---

## 6. Visualizations

All figures exported as high-res PNG + PDF, saved to `results/figures/`. Use `matplotlib` with a consistent colormap (`viridis` for field values, `RdBu_r` for signed error, centered at zero).

### 6.1 Required figures

1. **Space-time heatmap grid.** For 3 representative test trajectories (low-complexity, moderate shock, sharp shock):
   - 3 rows × 4 cols: `[ground truth | PINN | PINF-Fourier | PINF-SIREN]`.
   - Colorbar shared across ground truth / predictions; separate colorbar for error columns.
2. **Error heatmaps.** Same 3 trajectories, `|u_pred − u_true|`, shared log-scale colormap.
3. **Snapshot comparison.** For one trajectory, line plots of `u(x)` at `t ∈ {0.0, 0.5, 1.0, 1.5, 2.0}` overlaying ground truth and all models.
4. **Shock close-up.** Zoom on the steepest-gradient region at `t ≈ 1.0`, ±0.1 in x. Same overlay.
5. **Super-resolution.** At 4× spatial resolution, show ground truth (spline interpolation), PINN, PINF. Highlight the reconstructed shock width.
6. **Error spectrum.** Log-log plot of `|ê(k)|²` vs `k`, averaged across val set, at `t = 1.0`. One line per model.
7. **Convergence curves.** Training `L_data`, `L_phys`, and val `rel_L2` vs. epoch, one panel per model, three seeds overlaid.
8. **Characteristic lines.** For one trajectory, plot `dx/dt = u(x,t)` characteristics computed from both the ground truth and each model. Shock formation is where characteristics collide — this visualizes whether models get the shock location right. Substitute for "streamline plots" (streamlines don't apply in 1D).

### 6.2 Not applicable (explicitly excluded)

- **Streamline plots.** Not meaningful for a 1D scalar field.
- **Divergence heatmaps.** Divergence-free constraint is a 2D/3D NS condition, not a Burgers one.
- **Vorticity fields.** Same reason.

If this project extends to 2D Navier–Stokes later (e.g., via The Well `shear_flow` dataset), those visualizations become applicable and should be added then.

---

## 7. Code structure

```
project/
├── configs/
│   ├── base.yaml                  # shared config (dataset, seeds, logging)
│   ├── pinn.yaml                  # PINN-specific
│   ├── pinf_fourier.yaml          # PINF with Fourier features
│   ├── pinf_siren.yaml            # PINF with SIREN
│   └── ablations/
│       ├── fourier_sigma_1.yaml
│       ├── fourier_sigma_5.yaml
│       ├── fourier_sigma_10.yaml
│       ├── siren_w0_10.yaml
│       └── siren_w0_30.yaml
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # ALREADY EXISTS — do not modify
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py            # IC encoder (1D CNN), input encodings (Fourier, SIREN)
│   │   ├── base.py                # BaseModel with shared forward interface
│   │   ├── pinn.py                # vanilla PINN
│   │   └── pinf.py                # PINF, encoding chosen by config
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── data.py                # MSE on sampled points
│   │   ├── physics.py             # residual computation via autograd
│   │   └── balancing.py           # gradient-norm balancing (Wang et al. 2021)
│   ├── trainer.py                 # training loop, checkpointing, logging
│   ├── evaluator.py               # all metrics from §5
│   └── visualize.py               # all figures from §6
├── scripts/
│   ├── train.py                   # python scripts/train.py +experiment=pinn seed=0
│   ├── evaluate.py                # load checkpoint, produce metrics JSON
│   ├── visualize.py               # load checkpoint, produce figures
│   └── run_all.sh                 # reproduce full comparison
├── experiments/
│   └── runs/                      # Hydra creates timestamped dirs here
├── results/
│   ├── tables/                    # CSV / Markdown tables
│   ├── figures/                   # PNG + PDF
│   └── metrics/                   # JSON per-run
├── tests/
│   ├── test_physics_loss.py       # verify residual on known solution
│   ├── test_encoders.py           # shape tests
│   └── test_models.py             # forward/backward smoke tests
├── requirements.txt
├── environment.yml
└── README.md
```

**Frameworks:**
- PyTorch ≥ 2.1.
- Hydra for config management.
- W&B for logging.
- `einops` for tensor rearrangement.
- `h5py` for dataset I/O (already a dep).

---

## 8. Configuration management

Use **Hydra** with composition. Example `configs/base.yaml`:

```yaml
defaults:
  - _self_
  - model: ???          # overridden per experiment

seed: 0
dataset:
  h5_path: data/1D_Burgers_Sols_Nu0.01.hdf5
  n_points: 4096
  n_colloc: 2048
  train_size: 9000
  batch_size: 16
  num_workers: 4
train:
  epochs: 200
  lr: 1.0e-3
  lr_final: 1.0e-5
  lambda_data: 1.0
  lambda_phys: 0.1
  use_grad_balance: false
  lbfgs_finetune_epochs: 0
logging:
  backend: W&B
  log_every: 50
  ckpt_every: 25
```

Example `configs/pinf_fourier.yaml`:

```yaml
defaults:
  - base

model:
  name: pinf
  ic_encoder:
    channels: [32, 64, 64]
    kernel: 5
    latent_dim: 128
  encoding:
    type: fourier
    n_features: 64
    sigma: 5.0
  trunk:
    hidden_dim: 128
    n_layers: 6
    activation: tanh
```

---

## 9. Reproducibility

- Pin all package versions in `requirements.txt` / `environment.yml`.
- Seed Python, NumPy, PyTorch (CPU + CUDA), and the dataset's per-worker RNG.
- Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` during final result runs (may slow training; fine for publication runs).
- Every experiment writes to its Hydra output dir: config, git SHA, full metrics JSON, checkpoints.
- `scripts/run_all.sh` reproduces the full comparison end-to-end in a fixed order on a single consumer GPU (NVIDIA GeForce GTX 1650).

---

## 10. Deliverables

1. **Trained checkpoints** for PINN, PINF-Fourier, PINF-SIREN — 3 seeds each = 9 checkpoints.
2. **Metrics table** (`results/tables/main.md`) with all §5 metrics, mean ± std across seeds, for all three models.
3. **Ablation tables** for Fourier σ ∈ {1, 5, 10} and SIREN ω₀ ∈ {10, 30}.
4. **All figures** in §6.1.
5. **`README.md`** with one-command reproduction instructions.
6. **`REPORT.md`** (~ 3–5 pages) summarizing: setup, headline numbers, which model won on each axis, where each failed, and interpretation in terms of the spectral-bias hypothesis.

---

## 11. Scope and non-goals

**In scope:** 1D Burgers, ν = 0.01, PDEBench dataset, conditional/operator-style training, PINN vs PINF architectural comparison.

**Out of scope for this phase:**
- 2D or 3D flows. Deferred to a follow-up phase using The Well's `shear_flow` dataset.
- Other viscosities. If time permits, cross-ν generalization is a useful robustness test: train on ν=0.01, eval on PDEBench ν=0.001 — requires downloading the `Nu0.001` file.
- Neural operators (FNO, DeepONet) as additional baselines. Valuable for context but not required by the scientific question as stated. If added, they come after the core PINN vs PINF comparison is done.
- Inverse problems. The project review dedicates substantial discussion to inverse PINNs; they're a separate research direction.
- Hash-grid PINF (Instant-NGP style). Listed as a stretch goal in §2.2 because physics-residual autodiff through hash interpolation is broken; adapting the pipeline for finite-difference residuals is a project of its own.

---

## 12. Implementation order (suggested)

1. Encoders + base model (`src/models/encoders.py`, `base.py`). Smoke test forward pass.
2. PINN (`pinn.py`). Train briefly, confirm it fits one trajectory.
3. Data loss + physics loss (`losses/`). Unit-test physics loss on an analytic solution (e.g., `u = sin(πx)·exp(-ν π² t)` for the heat equation, or a Cole–Hopf solution to Burgers).
4. Trainer (`trainer.py`). Train PINN on 100 trajectories as a smoke test, check val L2 decreases.
5. PINF with Fourier encoding. Match param count. Train 100 trajectories.
6. Evaluator (`evaluator.py`) — implement all §5 metrics, test on PINN checkpoint.
7. Visualizer (`visualize.py`) — all §6 figures.
8. Full training runs (3 seeds × 3 models = 9 runs).
9. Ablations.
10. Assemble report.

Stop at any numbered step if something breaks and report back before continuing.
