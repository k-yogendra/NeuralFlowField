"""
test_encoders.py — Shape/dtype/gradient tests for encoders.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoders import ICEncoder, FourierEncoding, SIRENEncoding


B, X, N = 4, 1024, 32


class TestICEncoder:
    def setup_method(self):
        self.enc = ICEncoder(channels=[32, 64, 64], kernel=5, latent_dim=128)

    def test_output_shape(self):
        ic = torch.randn(B, X)
        z  = self.enc(ic)
        assert z.shape == (B, 128), z.shape

    def test_output_dtype(self):
        ic = torch.randn(B, X)
        z  = self.enc(ic)
        assert z.dtype == torch.float32

    def test_gradient_flows(self):
        ic = torch.randn(B, X, requires_grad=True)
        z  = self.enc(ic)
        z.sum().backward()
        assert ic.grad is not None


class TestFourierEncoding:
    def setup_method(self):
        self.enc = FourierEncoding(n_features=64, sigma=5.0, in_dim=2)

    def test_output_shape(self):
        coords = torch.randn(B, N, 2)
        out = self.enc(coords)
        assert out.shape == (B, N, 128), out.shape

    def test_no_learnable_params(self):
        n_params = sum(p.numel() for p in self.enc.parameters())
        assert n_params == 0, f"FourierEncoding should have no learnable params, got {n_params}"

    def test_buffer_frozen(self):
        # B matrix must be registered as a buffer, not a parameter
        assert hasattr(self.enc, "B")
        assert "B" not in {n for n, _ in self.enc.named_parameters()}

    def test_output_bounded(self):
        # sin/cos output must be in [-1, 1]
        coords = torch.randn(B, N, 2) * 10
        out = self.enc(coords)
        assert out.abs().max().item() <= 1.0 + 1e-5

    def test_gradient_flows_through(self):
        coords = torch.randn(B, N, 2, requires_grad=True)
        out = self.enc(coords)
        out.sum().backward()
        assert coords.grad is not None


class TestSIRENEncoding:
    def setup_method(self):
        self.enc = SIRENEncoding(in_dim=2, hidden_dim=128, omega_0=30.0)

    def test_output_shape(self):
        coords = torch.randn(B, N, 2)
        out = self.enc(coords)
        assert out.shape == (B, N, 128), out.shape

    def test_has_learnable_params(self):
        n_params = sum(p.numel() for p in self.enc.parameters() if p.requires_grad)
        assert n_params > 0

    def test_gradient_flows(self):
        coords = torch.randn(B, N, 2, requires_grad=True)
        out = self.enc(coords)
        out.sum().backward()
        assert coords.grad is not None

    def test_weight_init_range(self):
        w = self.enc.linear.weight.data
        in_dim = w.shape[1]
        bound  = 1.0 / in_dim
        assert w.abs().max().item() <= bound + 1e-6, \
            f"Init weight out of range: max={w.abs().max():.4f} bound={bound:.4f}"
