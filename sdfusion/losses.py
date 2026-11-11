"""Loss terms of the composite objective (Section III-B.4, Eq. 14)."""
from __future__ import annotations

import torch


def pricing_loss(r_hat: torch.Tensor, r: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared pricing error over observed firm-months."""
    err = (r_hat - torch.nan_to_num(r)) ** 2
    return (err * valid).sum() / valid.sum().clamp_min(1.0)


def orthogonality_loss(F: torch.Tensor) -> torch.Tensor:
    """|| F'F / T - I ||_F^2."""
    T, K = F.shape
    gram = F.T @ F / T
    return ((gram - torch.eye(K, device=F.device, dtype=F.dtype)) ** 2).sum()


def instrument_loss(lam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Loadings should vary with characteristics.

    ( ||Lambda_{n,t} - mean_n Lambda|| - ||x_{n,t} - mean_n x|| )^2, averaged
    over firms and months. lam: (B, n, K), x: (B, n, P).
    """
    dl = (lam - lam.mean(dim=1, keepdim=True)).norm(dim=-1)
    dx = (x - x.mean(dim=1, keepdim=True)).norm(dim=-1)
    return ((dl - dx) ** 2).mean()


def cvar_loss(
    r_hat: torch.Tensor, r: torch.Tensor, valid: torch.Tensor, alpha: float = 0.05
) -> torch.Tensor:
    """Tail-weighted squared error on the lower alpha-quantile of realised returns."""
    r_valid = r[valid]
    if r_valid.numel() == 0:
        return r_hat.new_zeros(())
    q = torch.quantile(r_valid.detach(), alpha)
    tail = valid & (torch.nan_to_num(r, nan=float("inf")) <= q)
    err = (r_hat - torch.nan_to_num(r)) ** 2
    return (err * tail).sum() / (alpha * valid.sum().clamp_min(1.0))
