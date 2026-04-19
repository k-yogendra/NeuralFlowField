"""
data.py — Supervised data loss.

L_data = (1 / B*N) * Σ |u_pred(x_i, t_i) - u_true(x_i, t_i)|²
"""

import torch
import torch.nn.functional as F


def data_loss(
    u_pred: torch.Tensor,   # (B, N)
    u_true: torch.Tensor,   # (B, N)
) -> torch.Tensor:
    """Mean squared error between predicted and true solution values.

    Parameters
    ----------
    u_pred : (B, N)  model output at sampled (x, t) points
    u_true : (B, N)  ground-truth u at the same points

    Returns
    -------
    scalar tensor
    """
    return F.mse_loss(u_pred, u_true)
