"""
base.py — shared trunk MLP and forward interface for PINN and PINF.

Both models inherit BaseModel and only differ in how they encode (x, t)
before feeding into the trunk.
"""

import torch
import torch.nn as nn


class TrunkMLP(nn.Module):
    """Fully-connected MLP with tanh activations.

    Parameters
    ----------
    in_dim     : dimension of the concatenated input [encoding(x,t), z]
    hidden_dim : width of each hidden layer
    n_layers   : number of hidden layers
    out_dim    : output dimension (1 for scalar u)
    activation : 'tanh' (default) or 'sin' (SIREN inner layers)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 6,
        out_dim: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.SiLU

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act_fn()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseModel(nn.Module):
    """Abstract base class.  Subclasses must implement `encode_coords`.

    Forward interface (spec §2)
    ---------------------------
    x   : (B, N)  float32   normalized x ∈ [-1, 1]
    t   : (B, N)  float32   normalized t ∈ [-1, 1]
    ic  : (B, X)  float32   normalized initial condition (X = 1024)

    Returns
    -------
    u_pred : (B, N)  float32
    """

    def __init__(self, ic_encoder: nn.Module, trunk: TrunkMLP):
        super().__init__()
        self.ic_encoder = ic_encoder
        self.trunk = trunk

    def encode_coords(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Map (x, t) → feature vector.  Shape: (B, N, coord_feat_dim).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,   # (B, N)
        t: torch.Tensor,   # (B, N)
        ic: torch.Tensor,  # (B, X)
    ) -> torch.Tensor:     # (B, N)
        B, N = x.shape

        # 1. Encode IC → latent z
        z = self.ic_encoder(ic)                    # (B, latent_dim)
        z_exp = z.unsqueeze(1).expand(B, N, -1)    # (B, N, latent_dim)

        # 2. Stack coordinates → (B, N, 2)
        coords = torch.stack([x, t], dim=-1)       # (B, N, 2)

        # 3. Encode coordinates
        coord_feat = self.encode_coords(coords)     # (B, N, coord_feat_dim)

        # 4. Concatenate and pass through trunk
        inp = torch.cat([coord_feat, z_exp], dim=-1)   # (B, N, coord_feat_dim + latent_dim)
        out = self.trunk(inp)                           # (B, N, 1)
        return out.squeeze(-1)                          # (B, N)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
