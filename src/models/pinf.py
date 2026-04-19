"""
pinf.py — Physics-Informed Neural Field (PINF).

Identical IC encoder and trunk to PINN, but (x, t) passes through a
structured input encoding (Fourier features or SIREN) before the trunk.

Architecture (spec §2.2)
------------------------
IC encoder : ICEncoder (same as PINN)
Encoding   : FourierEncoding  →  (B, N, 2m)   [m = n_features, out_dim = 2m]
           : SIRENEncoding    →  (B, N, hidden_dim)
Trunk      : 6 hidden layers × 128 units, tanh
Input      : [encoding(x,t), z]

Parameter-count matching (spec §2.3)
-------------------------------------
After construction, `build_pinf` checks that |N_pinf - N_pinn| / N_pinn ≤ 0.05.
If PINF is larger, trunk hidden_dim is reduced iteratively.
If PINF is smaller (rare), PINN trunk hidden_dim should be reduced instead —
caller is responsible for that symmetry.
"""

import torch
import torch.nn as nn

from .encoders import ICEncoder, FourierEncoding, SIRENEncoding
from .base import BaseModel, TrunkMLP


def build_pinf(cfg: dict, pinn_param_count: int | None = None) -> "PINF":
    """Construct a PINF from a config dict.

    Expected cfg keys:
        ic_encoder.channels    : [32, 64, 64]
        ic_encoder.kernel      : 5
        ic_encoder.latent_dim  : 128
        encoding.type          : 'fourier' | 'siren'
        encoding.n_features    : 64        (Fourier only)
        encoding.sigma         : 5.0       (Fourier only)
        encoding.omega_0       : 30.0      (SIREN only)
        trunk.hidden_dim       : 128
        trunk.n_layers         : 6
        trunk.activation       : 'tanh'

    Parameters
    ----------
    pinn_param_count : if provided, adjusts trunk hidden_dim so that
                       |N_pinf - pinn_param_count| / pinn_param_count ≤ 0.05.
    """
    ic_cfg  = cfg.get("ic_encoder", {})
    enc_cfg = cfg.get("encoding", {})
    trunk_cfg = cfg.get("trunk", {})

    latent_dim  = ic_cfg.get("latent_dim", 128)
    hidden_dim  = trunk_cfg.get("hidden_dim", 128)
    n_layers    = trunk_cfg.get("n_layers", 6)
    activation  = trunk_cfg.get("activation", "tanh")
    enc_type    = enc_cfg.get("type", "fourier")

    ic_encoder = ICEncoder(
        channels=ic_cfg.get("channels", [32, 64, 64]),
        kernel=ic_cfg.get("kernel", 5),
        latent_dim=latent_dim,
    )

    if enc_type == "fourier":
        n_features = enc_cfg.get("n_features", 64)
        sigma      = enc_cfg.get("sigma", 5.0)
        encoding   = FourierEncoding(n_features=n_features, sigma=sigma)
        coord_dim  = encoding.out_dim   # 2 * n_features
    elif enc_type == "siren":
        omega_0   = enc_cfg.get("omega_0", 30.0)
        encoding  = SIRENEncoding(in_dim=2, hidden_dim=hidden_dim, omega_0=omega_0)
        coord_dim = hidden_dim
    else:
        raise ValueError(f"Unknown encoding type: {enc_type!r}. Choose 'fourier' or 'siren'.")

    trunk_in_dim = coord_dim + latent_dim

    if pinn_param_count is not None:
        hidden_dim = _match_hidden_dim(
            pinn_param_count=pinn_param_count,
            ic_encoder=ic_encoder,
            encoding=encoding,
            trunk_in_dim=trunk_in_dim,
            n_layers=n_layers,
            activation=activation,
            latent_dim=latent_dim,
            enc_type=enc_type,
        )

    trunk = TrunkMLP(
        in_dim=trunk_in_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
    )

    model = PINF(ic_encoder=ic_encoder, trunk=trunk, encoding=encoding)

    if pinn_param_count is not None:
        ratio = abs(model.count_parameters() - pinn_param_count) / pinn_param_count
        if ratio > 0.05:
            import warnings
            warnings.warn(
                f"PINF param count {model.count_parameters()} differs from PINN "
                f"{pinn_param_count} by {ratio*100:.1f}% (> 5%). "
                "Consider adjusting hidden_dim manually."
            )

    return model


def _match_hidden_dim(
    pinn_param_count: int,
    ic_encoder: nn.Module,
    encoding: nn.Module,
    trunk_in_dim: int,
    n_layers: int,
    activation: str,
    latent_dim: int,
    enc_type: str,
) -> int:
    """Binary-search trunk hidden_dim so PINF params are within ±5% of PINN."""
    ic_params  = sum(p.numel() for p in ic_encoder.parameters() if p.requires_grad)
    enc_params = sum(p.numel() for p in encoding.parameters() if p.requires_grad)

    def count_trunk(h: int) -> int:
        t = TrunkMLP(in_dim=trunk_in_dim, hidden_dim=h, n_layers=n_layers, activation=activation)
        return sum(p.numel() for p in t.parameters() if p.requires_grad)

    target = pinn_param_count - ic_params - enc_params

    lo, hi = 16, 512
    best_h = 128
    for _ in range(20):
        mid = (lo + hi) // 2
        if count_trunk(mid) < target:
            lo = mid + 1
        else:
            hi = mid
        best_h = (lo + hi) // 2

    return best_h


class PINF(BaseModel):
    """Physics-Informed Neural Field with swappable coordinate encoding."""

    def __init__(
        self,
        ic_encoder: nn.Module,
        trunk: TrunkMLP,
        encoding: nn.Module,   # FourierEncoding or SIRENEncoding
    ):
        super().__init__(ic_encoder=ic_encoder, trunk=trunk)
        self.encoding = encoding

    def encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply the chosen encoding to raw (x, t) coordinates.

        Parameters
        ----------
        coords : (B, N, 2)

        Returns
        -------
        (B, N, coord_feat_dim)
        """
        return self.encoding(coords)
