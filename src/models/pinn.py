"""
pinn.py — Vanilla PINN baseline (Raissi-style).

No positional encoding on (x, t).  Raw coordinates are concatenated with the
IC latent and passed into the trunk MLP.

Architecture (spec §2.1)
------------------------
IC encoder : ICEncoder (channels=[32,64,64], kernel=5, latent_dim=128)
Trunk      : 6 hidden layers × 128 units, tanh
Input      : [x, t, z]  →  R^(2 + 128) = R^130
"""

import torch
import torch.nn as nn

from .encoders import ICEncoder
from .base import BaseModel, TrunkMLP


def build_pinn(cfg: dict) -> "PINN":
    """Construct a PINN from a config dict (mirrors the Hydra config structure).

    Expected cfg keys (with defaults):
        ic_encoder.channels    : [32, 64, 64]
        ic_encoder.kernel      : 5
        ic_encoder.latent_dim  : 128
        trunk.hidden_dim       : 128
        trunk.n_layers         : 6
        trunk.activation       : 'tanh'
    """
    ic_cfg = cfg.get("ic_encoder", {})
    trunk_cfg = cfg.get("trunk", {})

    latent_dim = ic_cfg.get("latent_dim", 128)
    hidden_dim = trunk_cfg.get("hidden_dim", 128)
    n_layers   = trunk_cfg.get("n_layers", 6)
    activation = trunk_cfg.get("activation", "tanh")

    ic_encoder = ICEncoder(
        channels=ic_cfg.get("channels", [32, 64, 64]),
        kernel=ic_cfg.get("kernel", 5),
        latent_dim=latent_dim,
    )

    # Trunk input: raw coords (x, t) = dim 2, plus latent z = latent_dim
    trunk_in_dim = 2 + latent_dim
    trunk = TrunkMLP(
        in_dim=trunk_in_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
    )

    return PINN(ic_encoder=ic_encoder, trunk=trunk)


class PINN(BaseModel):
    """Vanilla physics-informed neural network.

    encode_coords simply returns the raw (x, t) coordinates — no encoding.
    """

    def encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Identity encoding: return raw (x, t) as-is.

        Parameters
        ----------
        coords : (B, N, 2)

        Returns
        -------
        (B, N, 2)
        """
        return coords
