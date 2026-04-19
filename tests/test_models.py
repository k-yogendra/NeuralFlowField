"""
test_models.py — Forward/backward smoke tests for PINN and PINF.

Also verifies the ±5% parameter-count matching constraint.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pinn import build_pinn, PINN
from src.models.pinf import build_pinf, PINF

B, N, X = 4, 128, 1024

PINN_CFG = {
    "name": "pinn",
    "ic_encoder": {"channels": [32, 64, 64], "kernel": 5, "latent_dim": 128},
    "trunk":      {"hidden_dim": 128, "n_layers": 6, "activation": "tanh"},
}

PINF_FOURIER_CFG = {
    "name": "pinf",
    "ic_encoder": {"channels": [32, 64, 64], "kernel": 5, "latent_dim": 128},
    "encoding":   {"type": "fourier", "n_features": 64, "sigma": 5.0},
    "trunk":      {"hidden_dim": 128, "n_layers": 6, "activation": "tanh"},
}

PINF_SIREN_CFG = {
    "name": "pinf",
    "ic_encoder": {"channels": [32, 64, 64], "kernel": 5, "latent_dim": 128},
    "encoding":   {"type": "siren", "omega_0": 30.0},
    "trunk":      {"hidden_dim": 128, "n_layers": 6, "activation": "tanh"},
}


def _fake_batch():
    x  = torch.rand(B, N) * 2 - 1
    t  = torch.rand(B, N) * 2 - 1
    ic = torch.randn(B, X)
    return x, t, ic


class TestPINN:
    def setup_method(self):
        self.model = build_pinn(PINN_CFG)

    def test_forward_shape(self):
        x, t, ic = _fake_batch()
        out = self.model(x, t, ic)
        assert out.shape == (B, N), out.shape

    def test_forward_dtype(self):
        x, t, ic = _fake_batch()
        out = self.model(x, t, ic)
        assert out.dtype == torch.float32

    def test_backward(self):
        x, t, ic = _fake_batch()
        out = self.model(x, t, ic)
        out.sum().backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_param_count_positive(self):
        assert self.model.count_parameters() > 0

    def test_output_changes_with_ic(self):
        x, t, ic = _fake_batch()
        ic2 = torch.randn(B, X)
        out1 = self.model(x, t, ic)
        out2 = self.model(x, t, ic2)
        assert not torch.allclose(out1, out2), "Output should depend on IC"


class TestPINFFourier:
    def setup_method(self):
        self.pinn  = build_pinn(PINN_CFG)
        self.model = build_pinf(PINF_FOURIER_CFG, pinn_param_count=self.pinn.count_parameters())

    def test_forward_shape(self):
        x, t, ic = _fake_batch()
        out = self.model(x, t, ic)
        assert out.shape == (B, N), out.shape

    def test_backward(self):
        x, t, ic = _fake_batch()
        out = self.model(x, t, ic)
        out.sum().backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_param_count_within_5pct(self):
        n_pinn = self.pinn.count_parameters()
        n_pinf = self.model.count_parameters()
        ratio  = abs(n_pinf - n_pinn) / n_pinn
        assert ratio <= 0.05, \
            f"Param count mismatch: PINN={n_pinn}, PINF-Fourier={n_pinf}, ratio={ratio:.3f}"

    def test_fourier_features_deterministic(self):
        """Same model + same input → same output (Fourier B is frozen)."""
        x, t, ic = _fake_batch()
        out1 = self.model(x, t, ic)
        out2 = self.model(x, t, ic)
        assert torch.allclose(out1, out2)


class TestPINFSIREN:
    def setup_method(self):
        self.pinn  = build_pinn(PINN_CFG)
        self.model = build_pinf(PINF_SIREN_CFG, pinn_param_count=self.pinn.count_parameters())

    def test_forward_shape(self):
        x, t, ic = _fake_batch()
        out = self.model(x, t, ic)
        assert out.shape == (B, N), out.shape

    def test_backward(self):
        x, t, ic = _fake_batch()
        out = self.model(x, t, ic)
        out.sum().backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_param_count_within_5pct(self):
        n_pinn = self.pinn.count_parameters()
        n_pinf = self.model.count_parameters()
        ratio  = abs(n_pinf - n_pinn) / n_pinn
        assert ratio <= 0.05, \
            f"Param count mismatch: PINN={n_pinn}, PINF-SIREN={n_pinf}, ratio={ratio:.3f}"

    def test_siren_uses_sin_activation(self):
        from src.models.encoders import SIRENEncoding
        assert isinstance(self.model.encoding, SIRENEncoding)


class TestModelInterface:
    """Both models must implement the BaseModel forward interface."""

    @pytest.mark.parametrize("model", [
        build_pinn(PINN_CFG),
        build_pinf(PINF_FOURIER_CFG),
        build_pinf(PINF_SIREN_CFG),
    ])
    def test_interface(self, model):
        x, t, ic = _fake_batch()
        out = model(x, t, ic)
        assert out.shape == (B, N)
        assert out.dtype == torch.float32
