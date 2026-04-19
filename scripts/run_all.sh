#!/usr/bin/env bash
# run_all.sh — Full reproducible comparison: 3 models × 3 seeds = 9 runs.
# Usage: bash scripts/run_all.sh
# Assumes: conda env activated, cwd = NeuralFlowField/

set -euo pipefail

SEEDS=(0 1 2)
MODELS=(pinn pinf_fourier pinf_siren)
CKPT_ROOT="experiments/runs"

echo "=== PINN vs PINF: full training run ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"

# ----------------------------------------------------------------
# Phase 1: Training
# ----------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Training ${MODEL}  seed=${SEED}"
        python scripts/train.py \
            --config-name "${MODEL}" \
            seed="${SEED}" \
            logging.backend=tensorboard \
            hydra.run.dir="${CKPT_ROOT}/${MODEL}_seed${SEED}"
    done
done

# ----------------------------------------------------------------
# Phase 2: Evaluation
# ----------------------------------------------------------------
echo ""
echo "=== Evaluating all checkpoints ==="
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${CKPT_ROOT}/${MODEL}_seed${SEED}/checkpoints/best.pt"
        if [ -f "${CKPT}" ]; then
            echo "Evaluating ${MODEL} seed=${SEED}"
            python scripts/evaluate.py "${CKPT}"
        else
            echo "WARNING: checkpoint not found: ${CKPT}"
        fi
    done
done

# ----------------------------------------------------------------
# Phase 3: Figures (use seed=0 checkpoints)
# ----------------------------------------------------------------
echo ""
echo "=== Generating figures (seed=0) ==="
python scripts/visualize.py \
    --pinn   "${CKPT_ROOT}/pinn_seed0/checkpoints/best.pt" \
    --pinf_f "${CKPT_ROOT}/pinf_fourier_seed0/checkpoints/best.pt" \
    --pinf_s "${CKPT_ROOT}/pinf_siren_seed0/checkpoints/best.pt"

# ----------------------------------------------------------------
# Phase 4: Ablations (Fourier σ and SIREN ω₀)
# ----------------------------------------------------------------
echo ""
echo "=== Ablations ==="
ABLATIONS=(fourier_sigma_1 fourier_sigma_5 fourier_sigma_10 siren_w0_10 siren_w0_30)
for ABL in "${ABLATIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Ablation ${ABL}  seed=${SEED}"
        python scripts/train.py \
            --config-dir configs/ablations \
            --config-name "${ABL}" \
            seed="${SEED}" \
            hydra.run.dir="${CKPT_ROOT}/ablation_${ABL}_seed${SEED}"
    done
done

echo ""
echo "=== All done. Results in results/ ==="
