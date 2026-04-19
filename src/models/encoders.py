"""
encoders.py — building blocks shared by PINN and PINF.

Contents
--------
ICEncoder        : 1D CNN that encodes the initial condition into a latent vector z.
FourierEncoding  : Random Fourier Features (Tancik et al. 2020) for (x, t) coordinates.
SIRENEncoding    : First-layer sinusoidal encoding (Sitzmann et al. 2020).
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# IC Encoder
# ---------------------------------------------------------------------------
class ICEncoder(nn.Module):
    """1D CNN that maps an initial condition u(x, 0) → latent z ∈ R^latent_dim.

    Architecture
    ------------
    3 × Conv1d(in→out, kernel, stride=2) + BatchNorm + ReLU
    → GlobalAveragePool
    → Linear(last_channel, latent_dim)

    Default: channels=[32,64,64], kernel=5, latent_dim=128  (spec §2.1)
    Input  : (B, X)  — normalized IC at X=1024 grid points
    Output : (B, latent_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] = None,
        kernel: int = 5,
        latent_dim: int = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 64]

        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=2, padding=kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)        # → (B, c_in, 1)
        self.proj = nn.Linear(c_in, latent_dim)

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ic : (B, X)  float32

        Returns
        -------
        z  : (B, latent_dim)
        """
        x = ic.unsqueeze(1)          # (B, 1, X)
        x = self.conv(x)             # (B, C_last, X')
        x = self.pool(x).squeeze(-1) # (B, C_last)
        return self.proj(x)          # (B, latent_dim)


# ---------------------------------------------------------------------------
# Fourier Feature Encoding
# ---------------------------------------------------------------------------
class FourierEncoding(nn.Module):
    """Random Fourier Features for 2D coordinates (x, t).

    γ(v) = [sin(2π B v), cos(2π B v)]   where  B ∈ R^{m×d}  ~ N(0, σ²)

    Input  : (B, N, 2)  — (x_norm, t_norm) pairs
    Output : (B, N, 2m) — concatenated sin/cos features

    Parameters
    ----------
    n_features : m, number of frequency components (output dim = 2m)
    sigma      : std of the Gaussian from which B is sampled (controls bandwidth)
    in_dim     : input coordinate dimension (2 for (x, t))
    """

    def __init__(self, n_features: int = 64, sigma: float = 5.0, in_dim: int = 2):
        super().__init__()
        self.n_features = n_features
        self.out_dim = 2 * n_features
        B = torch.randn(n_features, in_dim) * sigma
        self.register_buffer("B", B)   # frozen — not a parameter

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : (..., in_dim)

        Returns
        -------
        encoded : (..., 2 * n_features)
        """
        proj = (2.0 * math.pi * coords) @ self.B.T   # (..., m)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # (..., 2m)


# ---------------------------------------------------------------------------
# SIREN Encoding (first-layer sinusoidal)
# ---------------------------------------------------------------------------
class SIRENEncoding(nn.Module):
    """SIREN-style first layer: Linear(in_dim → hidden_dim) with sin activation.

    Uses the specialized weight initialization from Sitzmann et al. 2020:
      - First layer : U(-1/in_dim, 1/in_dim) then scaled by ω₀
      - Subsequent  : U(-√(6/n), √(6/n))  (not applicable here; only one layer)

    The output is passed through sin(ω₀ · Wx + b).

    Input  : (..., in_dim)
    Output : (..., hidden_dim)

    Parameters
    ----------
    in_dim     : 2  (x, t)
    hidden_dim : must equal trunk input width so trunk receives consistent shape
    omega_0    : ω₀, controls frequency of the sinusoidal representation
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        in_dim = self.linear.weight.shape[1]
        bound = 1.0 / in_dim
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : (..., in_dim)

        Returns
        -------
        encoded : (..., hidden_dim)
        """
        return torch.sin(self.omega_0 * self.linear(coords))
