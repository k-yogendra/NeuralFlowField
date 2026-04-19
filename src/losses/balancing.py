"""
balancing.py — Gradient-norm balancing for multi-objective PINN training.

Reference: Wang et al. 2021, "Understanding and mitigating gradient pathologies
in physics-informed neural networks."

Two modes
---------
fixed    : λ_data and λ_phys are constant (from config).
adaptive : at each step, rescale λ_phys so that
               ||∇_{θ} L_phys|| ≈ ||∇_{θ} L_data||
           using an exponential moving average to smooth the estimates.

Usage
-----
    balancer = GradNormBalancer(mode='adaptive', alpha=0.9)
    lambda_data, lambda_phys = balancer.step(
        loss_data, loss_phys, model_params, optimizer
    )
    loss = lambda_data * loss_data + lambda_phys * loss_phys
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""

import torch
from typing import Iterable


class GradNormBalancer:
    """Adaptive or fixed loss weight manager.

    Parameters
    ----------
    mode          : 'fixed' | 'adaptive'
    lambda_data   : fixed weight for data loss
    lambda_phys   : initial weight for physics loss
    alpha         : EMA decay for grad-norm estimates (adaptive mode)
    eps           : small constant to avoid division by zero
    """

    def __init__(
        self,
        mode: str = "fixed",
        lambda_data: float = 1.0,
        lambda_phys: float = 0.1,
        alpha: float = 0.9,
        eps: float = 1e-8,
    ):
        assert mode in ("fixed", "adaptive"), f"Unknown mode: {mode!r}"
        self.mode = mode
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.alpha = alpha
        self.eps = eps

        # EMA estimates of ||∇L||
        self._ema_data: float | None = None
        self._ema_phys: float | None = None

    def _grad_norm(
        self,
        loss: torch.Tensor,
        params: Iterable[torch.nn.Parameter],
    ) -> float:
        """Compute ||∇_{params} loss||_2 without accumulating into .grad."""
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        total = 0.0
        for g in grads:
            if g is not None:
                total += g.detach().norm().item() ** 2
        return total ** 0.5

    def step(
        self,
        loss_data: torch.Tensor,
        loss_phys: torch.Tensor,
        model_params: list[torch.nn.Parameter],
    ) -> tuple[float, float]:
        """Compute and optionally update weights.

        Parameters
        ----------
        loss_data    : scalar tensor (data loss, graph attached)
        loss_phys    : scalar tensor (physics loss, graph attached)
        model_params : list of parameters to differentiate through

        Returns
        -------
        (lambda_data, lambda_phys)  — current weights
        """
        if self.mode == "fixed":
            return self.lambda_data, self.lambda_phys

        # Adaptive: compute grad norms
        norm_data = self._grad_norm(loss_data, model_params)
        norm_phys = self._grad_norm(loss_phys, model_params)

        # Update EMA
        if self._ema_data is None:
            self._ema_data = norm_data
            self._ema_phys = norm_phys
        else:
            self._ema_data = self.alpha * self._ema_data + (1 - self.alpha) * norm_data
            self._ema_phys = self.alpha * self._ema_phys + (1 - self.alpha) * norm_phys

        # Rescale lambda_phys to balance: λ_phys * ||∇L_phys|| ≈ λ_data * ||∇L_data||
        if self._ema_phys > self.eps:
            self.lambda_phys = self.lambda_data * (self._ema_data / (self._ema_phys + self.eps))

        return self.lambda_data, self.lambda_phys

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "lambda_data": self.lambda_data,
            "lambda_phys": self.lambda_phys,
            "ema_data": self._ema_data,
            "ema_phys": self._ema_phys,
        }

    def load_state_dict(self, d: dict):
        self.mode = d["mode"]
        self.lambda_data = d["lambda_data"]
        self.lambda_phys = d["lambda_phys"]
        self._ema_data = d.get("ema_data")
        self._ema_phys = d.get("ema_phys")
