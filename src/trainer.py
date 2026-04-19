"""
trainer.py — Training loop for PINN and PINF.

Responsibilities
----------------
- Adam + cosine LR decay (1e-3 → 1e-5, 200 epochs).
- Optional L-BFGS fine-tune for final N epochs.
- Batch: 16 trajectories × 4096 data points + 2048 collocation points.
- Checkpoint every 25 epochs + best-val-L2 checkpoint.
- Logging: TensorBoard (default) or W&B.
- Full reproducibility seeding.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .losses.data import data_loss
from .losses.physics import physics_loss, periodicity_loss
from .losses.balancing import GradNormBalancer


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logger wrapper
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, backend: str, log_dir: str, project: str = "NeuralFlowField", run_name: str = "run"):
        self.backend = backend
        self._step = 0

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        elif backend == "wandb":
            import wandb
            wandb.init(project=project, name=run_name, dir=log_dir)
            self.writer = None
        else:
            raise ValueError(f"Unknown logging backend: {backend!r}")

    def log(self, metrics: dict[str, float], step: int | None = None):
        s = step if step is not None else self._step
        if self.backend == "tensorboard":
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, s)
        elif self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=s)
        self._step = s + 1

    def close(self):
        if self.backend == "tensorboard" and self.writer is not None:
            self.writer.close()
        elif self.backend == "wandb":
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_rel_l2: float,
    cfg: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_rel_l2": val_rel_l2,
            "cfg": cfg,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Validation metric
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_val_rel_l2(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Relative L2 and RMSE on the full (T, X) grid using field-mode loader.

    Parameters
    ----------
    max_batches : if set, evaluate only this many batches (useful for smoke tests).
    """
    rel_l2_list, rmse_list = [], []
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        ic   = batch["ic"].to(device)           # (B, X)
        u    = batch["u"].to(device)            # (B, T, X)
        x_g  = batch["x"].to(device)           # (B, X)  — already normalized
        t_g  = batch["t"].to(device)           # (B, T)  — already normalized

        B, T, X = u.shape
        # Build meshgrid of (x, t) for all (T, X) query points
        x_exp = x_g.unsqueeze(1).expand(B, T, X)  # (B, T, X)
        t_exp = t_g.unsqueeze(2).expand(B, T, X)  # (B, T, X)

        x_flat = x_exp.reshape(B, T * X)
        t_flat = t_exp.reshape(B, T * X)

        # Forward in chunks to avoid OOM
        chunk = 1024 * 16
        u_pred_chunks = []
        for i in range(0, T * X, chunk):
            u_pred_chunks.append(model(x_flat[:, i:i+chunk], t_flat[:, i:i+chunk], ic))
        u_pred = torch.cat(u_pred_chunks, dim=1).reshape(B, T, X)

        u_true = u
        diff = u_pred - u_true
        rel_l2 = (diff.norm(dim=(1, 2)) / (u_true.norm(dim=(1, 2)) + 1e-8)).mean().item()
        rmse   = diff.pow(2).mean().sqrt().item()
        rel_l2_list.append(rel_l2)
        rmse_list.append(rmse)

    return float(np.mean(rel_l2_list)), float(np.mean(rmse_list))


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        run_dir: str | Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tcfg = cfg.get("train", {})
        self.epochs          = tcfg.get("epochs", 200)
        self.lr              = tcfg.get("lr", 1e-3)
        self.lr_final        = tcfg.get("lr_final", 1e-5)
        self.lambda_data     = tcfg.get("lambda_data", 1.0)
        self.lambda_phys     = tcfg.get("lambda_phys", 0.1)
        self.use_grad_balance = tcfg.get("use_grad_balance", False)
        self.use_periodicity = tcfg.get("use_periodicity_loss", False)
        self.lambda_bc       = tcfg.get("lambda_bc", 0.01)
        self.lbfgs_epochs    = tcfg.get("lbfgs_finetune_epochs", 0)
        self.val_max_batches = tcfg.get("val_max_batches", None)  # None = evaluate full val set
        self.n_colloc        = cfg.get("dataset", {}).get("n_colloc", 2048)

        lcfg = cfg.get("logging", {})
        self.log_every  = lcfg.get("log_every", 50)
        self.ckpt_every = lcfg.get("ckpt_every", 25)

        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - self.lbfgs_epochs,
            eta_min=self.lr_final,
        )
        self.balancer = GradNormBalancer(
            mode="adaptive" if self.use_grad_balance else "fixed",
            lambda_data=self.lambda_data,
            lambda_phys=self.lambda_phys,
            alpha=tcfg.get("grad_balance_alpha", 0.9),
        )

        self.logger = Logger(
            backend=lcfg.get("backend", "tensorboard"),
            log_dir=str(self.run_dir / "logs"),
            project=lcfg.get("wandb_project", "NeuralFlowField"),
            run_name=self.run_dir.name,
        )

        self.best_val_l2 = float("inf")
        self._global_step = 0

    # ------------------------------------------------------------------
    def _colloc_points(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample M random collocation points in [-1,1]² per trajectory."""
        x_c = torch.rand(batch_size, self.n_colloc, device=self.device) * 2 - 1
        t_c = torch.rand(batch_size, self.n_colloc, device=self.device) * 2 - 1
        x_c.requires_grad_(True)
        t_c.requires_grad_(True)
        return x_c, t_c

    # ------------------------------------------------------------------
    def _train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        ic     = batch["ic"].to(self.device)        # (B, X)
        coords = batch["coords"].to(self.device)    # (B, N, 2)
        values = batch["values"].to(self.device)    # (B, N)

        x_d = coords[..., 0]   # (B, N)
        t_d = coords[..., 1]   # (B, N)

        B = ic.shape[0]

        # -- Data loss --
        u_pred = self.model(x_d, t_d, ic)
        l_data = data_loss(u_pred, values)

        # -- Physics loss --
        x_c, t_c = self._colloc_points(B)
        l_phys = physics_loss(self.model, x_c, t_c, ic)

        # -- Get weights (fixed or adaptive) --
        lam_d, lam_p = self.balancer.step(
            l_data, l_phys,
            list(self.model.parameters()),
        )

        loss = lam_d * l_data + lam_p * l_phys

        # -- Optional BC loss --
        l_bc = torch.tensor(0.0, device=self.device)
        if self.use_periodicity:
            t_k = torch.rand(B, 32, device=self.device) * 2 - 1
            l_bc = periodicity_loss(self.model, t_k, ic)
            loss = loss + self.lambda_bc * l_bc

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
        self.optimizer.step()

        return {
            "loss/total": loss.item(),
            "loss/data":  l_data.item(),
            "loss/phys":  l_phys.item(),
            "loss/bc":    l_bc.item() if self.use_periodicity else 0.0,
            "lambda/data": lam_d,
            "lambda/phys": lam_p,
            "grad_norm":  grad_norm,
        }

    # ------------------------------------------------------------------
    def train(self):
        t0_total = time.time()

        for epoch in range(1, self.epochs - self.lbfgs_epochs + 1):
            t0_epoch = time.time()
            epoch_metrics: dict[str, list] = {}

            for batch in self.train_loader:
                step_metrics = self._train_step(batch)
                for k, v in step_metrics.items():
                    epoch_metrics.setdefault(k, []).append(v)
                self._global_step += 1

            self.scheduler.step()

            # -- Aggregate epoch metrics --
            agg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
            agg["lr"] = self.scheduler.get_last_lr()[0]
            agg["wall_clock"] = time.time() - t0_total

            # -- Validation --
            val_rel_l2, val_rmse = compute_val_rel_l2(
                self.model, self.val_loader, self.device,
                max_batches=self.val_max_batches,
            )
            agg["val/rel_l2"] = val_rel_l2
            agg["val/rmse"]   = val_rmse

            epoch_time = time.time() - t0_epoch
            print(
                f"[epoch {epoch:3d}/{self.epochs}]  "
                f"loss={agg['loss/total']:.4e}  "
                f"data={agg['loss/data']:.4e}  "
                f"phys={agg['loss/phys']:.4e}  "
                f"val_L2={val_rel_l2:.4f}  "
                f"lr={agg['lr']:.2e}  "
                f"t={epoch_time:.1f}s"
            )

            if epoch % self.log_every == 0 or epoch == 1:
                self.logger.log({f"epoch/{k}": v for k, v in agg.items()}, step=epoch)

            # -- Checkpointing --
            if epoch % self.ckpt_every == 0:
                save_checkpoint(
                    self.run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                    self.model, self.optimizer, self.scheduler, epoch, val_rel_l2, self.cfg,
                )

            if val_rel_l2 < self.best_val_l2:
                self.best_val_l2 = val_rel_l2
                save_checkpoint(
                    self.run_dir / "checkpoints" / "best.pt",
                    self.model, self.optimizer, self.scheduler, epoch, val_rel_l2, self.cfg,
                )

        # -- Optional L-BFGS fine-tune --
        if self.lbfgs_epochs > 0:
            self._lbfgs_finetune()

        self.logger.close()
        print(f"\nTraining done. Best val rel-L2: {self.best_val_l2:.4f}")

    # ------------------------------------------------------------------
    def _lbfgs_finetune(self):
        print(f"\n[L-BFGS] Fine-tuning for {self.lbfgs_epochs} epochs...")
        lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        for epoch in range(1, self.lbfgs_epochs + 1):
            def closure():
                lbfgs.zero_grad()
                # Use one batch for closure
                batch = next(iter(self.train_loader))
                ic     = batch["ic"].to(self.device)
                coords = batch["coords"].to(self.device)
                values = batch["values"].to(self.device)
                x_d = coords[..., 0]
                t_d = coords[..., 1]
                B = ic.shape[0]

                u_pred = self.model(x_d, t_d, ic)
                l_data = data_loss(u_pred, values)
                x_c, t_c = self._colloc_points(B)
                l_phys = physics_loss(self.model, x_c, t_c, ic)
                loss = self.lambda_data * l_data + self.lambda_phys * l_phys
                loss.backward()
                return loss

            lbfgs.step(closure)
            val_rel_l2, _ = compute_val_rel_l2(self.model, self.val_loader, self.device)
            print(f"  [L-BFGS epoch {epoch}] val_L2={val_rel_l2:.4f}")

            if val_rel_l2 < self.best_val_l2:
                self.best_val_l2 = val_rel_l2
                save_checkpoint(
                    self.run_dir / "checkpoints" / "best.pt",
                    self.model, None, None,
                    self.epochs + epoch, val_rel_l2, self.cfg,
                )
