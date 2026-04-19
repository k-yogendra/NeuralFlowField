"""
physics.py — PDE residual loss for the 1D viscous Burgers equation.

Governing equation (physical coordinates):
    u_t + u * u_x = ν * u_xx,    ν = 0.01

Coordinate convention
----------------------
The dataset normalizes:
    x_norm ∈ [-1, 1]   ↔  x_phys ∈ [-1, 1]   (identity — same domain)
    t_norm ∈ [-1, 1]   ↔  t_phys ∈ [0, 2.01]

Chain rule for t:
    t_phys = (t_norm + 1) / 2 * T_max,  T_max = 2.01
    ∂u/∂t_phys = ∂u/∂t_norm  *  (2 / T_max)

x_norm = x_phys (domain is already [-1, 1]), so no chain-rule correction needed for x.

Usage
-----
    x_c, t_c : collocation points, already require_grad=True
    model     : callable (x, t, ic) → u_pred
    ic        : (B, X) IC tensor (no grad needed here)
"""

import torch

NU = 0.01
T_MAX = 2.01   # physical t ∈ [0, T_MAX]


def physics_residual(
    model,
    x_c: torch.Tensor,   # (B, M)  normalized, requires_grad=True
    t_c: torch.Tensor,   # (B, M)  normalized, requires_grad=True
    ic: torch.Tensor,    # (B, X)
) -> torch.Tensor:
    """Compute the Burgers PDE residual at collocation points.

    r = u_t_phys + u * u_x_phys - ν * u_xx_phys

    Returns
    -------
    r : (B, M)  residual values
    """
    # Ensure grads flow through x_c and t_c
    if not x_c.requires_grad:
        x_c = x_c.requires_grad_(True)
    if not t_c.requires_grad:
        t_c = t_c.requires_grad_(True)

    u = model(x_c, t_c, ic)   # (B, M)

    # --- First derivatives in normalized coords ---
    ones = torch.ones_like(u)

    grads = torch.autograd.grad(
        u, [x_c, t_c],
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    # If the model output doesn't depend on x_c or t_c (e.g. constant model),
    # autograd returns None — treat as zero gradient.
    u_x_norm = grads[0] if grads[0] is not None else torch.zeros_like(x_c)
    u_t_norm = grads[1] if grads[1] is not None else torch.zeros_like(t_c)

    # --- Second derivative w.r.t. x (x_norm = x_phys, so no scaling) ---
    # Only differentiate if u_x_norm has a grad_fn (i.e. it was computed via autograd).
    # If it was replaced with a zero tensor (constant/zero model), u_xx = 0 as well.
    if u_x_norm.requires_grad or u_x_norm.grad_fn is not None:
        u_xx_norm = torch.autograd.grad(
            u_x_norm, x_c,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if u_xx_norm is None:
            u_xx_norm = torch.zeros_like(x_c)
    else:
        u_xx_norm = torch.zeros_like(x_c)

    # --- Chain rule: normalized → physical time ---
    # dt_phys/dt_norm = T_MAX / 2
    dt_phys_dt_norm = T_MAX / 2.0
    u_t_phys  = u_t_norm  / dt_phys_dt_norm   # ∂u/∂t_phys = ∂u/∂t_norm * (dt_norm/dt_phys)
    # x_norm == x_phys, so dx_phys/dx_norm = 1 (no correction)
    u_x_phys  = u_x_norm
    u_xx_phys = u_xx_norm

    residual = u_t_phys + u * u_x_phys - NU * u_xx_phys  # (B, M)
    return residual


def physics_loss(
    model,
    x_c: torch.Tensor,   # (B, M)
    t_c: torch.Tensor,   # (B, M)
    ic: torch.Tensor,    # (B, X)
) -> torch.Tensor:
    """Mean squared PDE residual.

    L_phys = (1 / B*M) * Σ r(x_c, t_c)²

    Returns
    -------
    scalar tensor
    """
    r = physics_residual(model, x_c, t_c, ic)
    return (r ** 2).mean()


def periodicity_loss(
    model,
    t_k: torch.Tensor,   # (B, K)  random normalized t values
    ic: torch.Tensor,    # (B, X)
) -> torch.Tensor:
    """Boundary condition: u(-1, t) == u(+1, t) (periodic).

    L_bc = (1 / B*K) * Σ |u_pred(-1, t_k) - u_pred(+1, t_k)|²

    Returns
    -------
    scalar tensor
    """
    B, K = t_k.shape
    x_left  = torch.full((B, K), -1.0, dtype=t_k.dtype, device=t_k.device)
    x_right = torch.full((B, K), +1.0, dtype=t_k.dtype, device=t_k.device)
    u_left  = model(x_left,  t_k, ic)   # (B, K)
    u_right = model(x_right, t_k, ic)   # (B, K)
    return ((u_left - u_right) ** 2).mean()
