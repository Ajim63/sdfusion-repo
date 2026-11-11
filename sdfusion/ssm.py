"""Selective state-space (Mamba) blocks.

Two interchangeable backends are provided:

* ``mamba_ssm``  - the official CUDA implementation (``pip install mamba-ssm``).
  This is the backend to use for full-size experiments.
* ``torch``      - a dependency-free PyTorch implementation of the selective
  state-space recursion in Section III-B of the paper. It runs on CPU and GPU
  but scans the sequence with a Python loop, so it is slow on long sequences.
  It exists so that the code can be run and inspected without a GPU.

``backend="auto"`` picks ``mamba_ssm`` when it is importable and falls back to
``torch`` otherwise.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional dependency
    from mamba_ssm import Mamba as _Mamba

    HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover - depends on the environment
    _Mamba = None
    HAS_MAMBA_SSM = False


class SelectiveSSM(nn.Module):
    """Pure-PyTorch selective state-space layer.

    For an input sequence ``u_1..u_L`` the layer computes

        delta_t = softplus(W_delta u_t)
        A_bar_t = exp(delta_t * A),      B_bar_t = delta_t * B_t
        h_t     = A_bar_t * h_{t-1} + B_bar_t * u_t
        y_t     = C_t h_t + D * u_t

    where ``A`` and ``D`` are shared parameters and ``B_t``, ``C_t`` are linear
    functions of ``u_t`` (the "selective" part).
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        # A is kept negative through -exp(A_log) so the recursion is stable.
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        # Initialise the step size so that softplus(bias) lies in [1e-3, 1e-1].
        dt = torch.exp(
            torch.rand(d_model) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, D)
        bsz, length, _ = x.shape
        u = F.silu(self.in_proj(x))
        delta = F.softplus(self.dt_proj(u))  # (B, L, D)
        A = -torch.exp(self.A_log)  # (D, n)
        Bm = self.B_proj(u)  # (B, L, n)
        Cm = self.C_proj(u)  # (B, L, n)

        h = u.new_zeros(bsz, self.d_model, self.d_state)
        ys = []
        for t in range(length):
            dt = delta[:, t].unsqueeze(-1)  # (B, D, 1)
            A_bar = torch.exp(dt * A)  # (B, D, n)
            B_bar = dt * Bm[:, t].unsqueeze(1)  # (B, D, n)
            h = A_bar * h + B_bar * u[:, t].unsqueeze(-1)
            y = (h * Cm[:, t].unsqueeze(1)).sum(-1) + self.D * u[:, t]
            ys.append(y)
        return self.out_proj(torch.stack(ys, dim=1))


def make_ssm(d_model: int, d_state: int = 16, backend: str = "auto") -> nn.Module:
    """Return a sequence layer mapping (B, L, D) -> (B, L, D)."""
    if backend not in {"auto", "mamba_ssm", "torch"}:
        raise ValueError(f"unknown backend: {backend}")
    if backend == "mamba_ssm" and not HAS_MAMBA_SSM:
        raise ImportError("backend='mamba_ssm' requested but mamba-ssm is not installed")
    if backend in {"auto", "mamba_ssm"} and HAS_MAMBA_SSM:
        return _Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
    return SelectiveSSM(d_model, d_state)


class MambaBlock(nn.Module):
    """Pre-norm residual block around one selective state-space layer."""

    def __init__(self, d_model: int, d_state: int = 16, backend: str = "auto"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = make_ssm(d_model, d_state, backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class BiMambaBlock(nn.Module):
    """Bidirectional block: a forward scan and a backward scan, then fused."""

    def __init__(self, d_model: int, d_state: int = 16, backend: str = "auto"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = make_ssm(d_model, d_state, backend)
        self.bwd = make_ssm(d_model, d_state, backend)
        self.fuse = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        f = self.fwd(z)
        b = torch.flip(self.bwd(torch.flip(z, dims=[1])), dims=[1])
        return x + self.fuse(torch.cat([f, b], dim=-1))
