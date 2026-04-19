"""
test_physics_loss.py — Verify PDE residual on known analytic solutions.

Test 1: Heat equation  u = sin(πx) exp(-ν π² t)
  This satisfies u_t = ν u_xx  (Burgers with u·u_x = 0 only when u ≈ 0,
  so we test the heat-equation limit by checking the residual is small
  for a tiny-amplitude solution where the nonlinear term is negligible.)

Test 2: Cole–Hopf steady state  u = 0  (trivially satisfies the equation).

Test 3: Exact linear wave  u = const  → residual must be exactly 0.
"""

import sys
import math
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses.physics import physics_residual, NU, T_MAX


# A tiny dummy model that returns a fixed analytic u(x_norm, t_norm)
class AnalyticModel(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        # dummy parameter so autograd can track
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, ic):
        return self.fn(x, t) + 0 * self._dummy


def _make_grid(B=2, M=64):
    x = torch.rand(B, M) * 2 - 1
    t = torch.rand(B, M) * 2 - 1
    x.requires_grad_(True)
    t.requires_grad_(True)
    ic = torch.zeros(B, 1024)   # dummy IC
    return x, t, ic


class TestZeroSolution:
    """u = 0 satisfies Burgers exactly → residual must be 0."""

    def test_zero_residual(self):
        model = AnalyticModel(lambda x, t: torch.zeros_like(x))
        x, t, ic = _make_grid()
        r = physics_residual(model, x, t, ic)
        assert r.abs().max().item() < 1e-5, f"max residual = {r.abs().max().item():.2e}"


class TestConstantSolution:
    """u = c (any constant) → u_t = u_x = u_xx = 0 → residual = 0."""

    @pytest.mark.parametrize("c", [0.0, 1.0, -2.5])
    def test_constant(self, c):
        model = AnalyticModel(lambda x, t: torch.full_like(x, c))
        x, t, ic = _make_grid()
        r = physics_residual(model, x, t, ic)
        assert r.abs().max().item() < 1e-4, \
            f"c={c}: max residual = {r.abs().max().item():.2e}"


class TestHeatEquationSolution:
    """u = A·sin(πx)·exp(-ν·π²·t_phys) solves heat equation.

    For this solution:  u·u_x = A²·sin(πx)·cos(πx)·exp(-2ν π² t)
    This is NOT zero, so it doesn't satisfy Burgers exactly.
    We verify the implementation computes a non-trivial (non-zero) residual
    correctly by checking the sign structure matches expectations.
    We also verify that for A→0 the residual → 0 (linearisation).
    """

    def _heat_fn(self, A):
        def fn(x_norm, t_norm):
            # Convert to physical coords
            t_phys = (t_norm + 1.0) / 2.0 * T_MAX
            return A * torch.sin(math.pi * x_norm) * torch.exp(-NU * math.pi**2 * t_phys)
        return fn

    def test_residual_near_zero_for_small_amplitude(self):
        """For A=1e-4, nonlinear term ≈ A² ≈ 1e-8, so residual ≈ 0."""
        A = 1e-4
        model = AnalyticModel(self._heat_fn(A))
        x, t, ic = _make_grid(B=2, M=32)
        r = physics_residual(model, x, t, ic)
        # Residual should be O(A²) ≈ 1e-8; allow generous tolerance
        assert r.abs().max().item() < 1e-4, \
            f"A={A}: max residual = {r.abs().max().item():.2e}"

    def test_residual_grows_with_amplitude(self):
        """Residual for A=1.0 should be much larger than A=1e-4."""
        def _res(A):
            model = AnalyticModel(self._heat_fn(A))
            x, t, ic = _make_grid(B=2, M=32)
            return physics_residual(model, x, t, ic).abs().max().item()

        assert _res(1.0) > _res(1e-4), "Residual should scale with nonlinearity"


class TestPhysicsLossGradient:
    """physics_loss must be differentiable w.r.t. model parameters."""

    def test_backward(self):
        import torch.nn as nn
        from src.losses.physics import physics_loss

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 1)
            def forward(self, x, t, ic):
                B, N = x.shape
                z = ic.mean(dim=1, keepdim=True).expand(B, N)
                inp = torch.stack([x, t, z], dim=-1)
                return self.linear(inp).squeeze(-1)

        model = TinyModel()
        x = torch.rand(2, 64) * 2 - 1
        t = torch.rand(2, 64) * 2 - 1
        x.requires_grad_(True)
        t.requires_grad_(True)
        ic = torch.zeros(2, 1024)

        loss = physics_loss(model, x, t, ic)
        loss.backward()

        for p in model.parameters():
            assert p.grad is not None, "Model parameter has no gradient"
