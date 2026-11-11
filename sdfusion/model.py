"""SDFusion model components.

* :class:`TemporalFactorEncoder`  - latent factor realisations for the training
  months of an estimation window (Section III-B.1).
* :class:`LoadingModule`          - characteristic-driven factor loadings with a
  bidirectional cross-sectional scan (Section III-B.2).
* :class:`DiffusionHead`          - conditional DDPM over next-month factors
  (Section III-B.3).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import BiMambaBlock, MambaBlock


def sinusoidal_encoding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal encoding. positions: (L,) -> (L, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=positions.device) / max(half, 1)
    )
    args = positions.float().unsqueeze(-1) * freqs
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        enc = F.pad(enc, (0, 1))
    return enc


# --------------------------------------------------------------------------- #
# Temporal factor encoder
# --------------------------------------------------------------------------- #
class TemporalFactorEncoder(nn.Module):
    """Maps learnable factor embeddings and the macro state to factors F (T, K).

    Each of the ``n_months`` training months of the estimation window owns a
    learnable embedding per factor (``F_init``). The embeddings are combined
    with a positional encoding and the macro state, passed through ``n_layers``
    Mamba blocks along time, projected to a scalar per factor and whitened.
    """

    def __init__(
        self,
        n_months: int,
        n_factors: int,
        n_macro: int,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 5,
        backend: str = "auto",
        whiten_eps: float = 1e-4,
    ):
        super().__init__()
        self.n_months = n_months
        self.n_factors = n_factors
        self.d_model = d_model
        self.whiten_eps = whiten_eps
        self.F_init = nn.Parameter(torch.randn(n_months, n_factors, d_model))
        self.in_proj = nn.Linear(d_model + n_macro, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, backend) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Parameter(torch.randn(n_factors, d_model) / math.sqrt(d_model))

    def forward(self, macro: torch.Tensor, whiten: bool = True) -> torch.Tensor:
        """macro: (T, M) standardised macro state of the training months."""
        T = macro.shape[0]
        if T != self.n_months:
            raise ValueError(f"expected {self.n_months} months, got {T}")
        pos = sinusoidal_encoding(torch.arange(T, device=macro.device), self.d_model)
        tok = self.F_init + pos.unsqueeze(1)  # (T, K, d)
        m = macro.unsqueeze(1).expand(-1, self.n_factors, -1)  # (T, K, M)
        u = self.in_proj(torch.cat([tok, m], dim=-1))  # (T, K, d)
        h = u.transpose(0, 1)  # (K, T, d): one sequence per factor
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        f = (h * self.head.unsqueeze(1)).sum(-1).transpose(0, 1)  # (T, K)
        return self.whiten(f) if whiten else f

    def whiten(self, f: torch.Tensor) -> torch.Tensor:
        """F <- (F - mean) L^{-T} with L L^T = Cov(F), so that Cov(F) = I."""
        fc = f - f.mean(dim=0, keepdim=True)
        cov = fc.T @ fc / max(f.shape[0] - 1, 1)
        eye = torch.eye(cov.shape[0], device=f.device, dtype=f.dtype)
        scale = cov.diagonal().mean().detach().clamp_min(1e-8)
        L = torch.linalg.cholesky(cov + self.whiten_eps * scale * eye)
        return torch.linalg.solve_triangular(L, fc.T, upper=False).T

    @torch.no_grad()
    def roll(self, k: int = 1) -> None:
        """Shift the per-month embeddings when the window moves forward k months.

        The oldest ``k`` months are dropped; the embeddings of the ``k`` months
        that enter the training block are initialised from the most recent one.
        """
        k = min(k, self.n_months - 1)
        if k <= 0:
            return
        last = self.F_init[-1:].clone()
        self.F_init[:-k] = self.F_init[k:].clone()
        self.F_init[-k:] = last + 0.01 * torch.randn_like(self.F_init[-k:])


# --------------------------------------------------------------------------- #
# Cross-sectional loading module
# --------------------------------------------------------------------------- #
class LoadingModule(nn.Module):
    """Characteristics (B, n, P) -> loadings (B, n, K).

    Firms are expected to arrive already sorted by the economic ordering key
    (industry, size, identifier); see ``sdfusion/data.py``.
    """

    def __init__(
        self,
        n_chars: int,
        n_factors: int,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 5,
        backend: str = "auto",
        gate_hidden: int | None = None,
    ):
        super().__init__()
        gate_hidden = gate_hidden or max(n_chars // 2, 4)
        self.gate = nn.Sequential(
            nn.Linear(n_chars, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, n_chars)
        )
        self.embed = nn.Linear(n_chars, d_model)
        self.blocks = nn.ModuleList(
            [BiMambaBlock(d_model, d_state, backend) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_factors)
        )

    def channel_weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Channel attention: w_t = sigmoid(W2 ReLU(W1 xbar_t)), xbar_t a masked mean."""
        m = mask.float()
        xbar = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # (B, P)
        return torch.sigmoid(self.gate(xbar))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, perm_prob: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x, mask: (B, n, P). Returns loadings (B, n, K) and gate weights (B, P)."""
        w = self.channel_weights(x, mask)
        z = self.embed(x * w.unsqueeze(1))  # (B, n, d)

        # Random permutation augmentation of the firm ordering (training only).
        perm = None
        if self.training and perm_prob > 0 and torch.rand(()).item() < perm_prob:
            perm = torch.randperm(z.shape[1], device=z.device)
            z = z[:, perm]

        for blk in self.blocks:
            z = blk(z)

        if perm is not None:
            z = z[:, torch.argsort(perm)]
        return self.head(self.norm(z)), w


# --------------------------------------------------------------------------- #
# Conditional latent diffusion head
# --------------------------------------------------------------------------- #
class DiffusionHead(nn.Module):
    """Conditional DDPM for p(f_{t+1} | f_{<=t}, m_t) in the K-dim factor space.

    The conditioning vector is ``c_t = [GRU(f_{t-H+1..t}) || m_t]``. The macro
    part is dropped with probability ``p_drop`` during training so that
    classifier-free guidance can be applied at sampling time.
    """

    def __init__(
        self,
        n_factors: int,
        n_macro: int,
        hidden: int = 64,
        history: int = 12,
        train_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.history = history
        self.train_steps = train_steps
        self.p_drop = p_drop
        self.hidden = hidden

        self.gru = nn.GRU(n_factors, hidden, batch_first=True)
        self.null_macro = nn.Parameter(torch.zeros(n_macro))
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        in_dim = n_factors + hidden + hidden + n_macro
        self.eps_net = nn.Sequential(
            nn.Linear(in_dim, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, n_factors),
        )

        betas = torch.linspace(beta_start, beta_end, train_steps)
        self.register_buffer("alphas_bar", torch.cumprod(1.0 - betas, dim=0))

    # -- helpers ----------------------------------------------------------- #
    def _summary(self, hist: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(hist)  # hist: (B, H, K)
        return h[-1]

    def _eps(self, f_s, s, summary, macro):
        t_emb = self.time_mlp(sinusoidal_encoding(s, self.hidden))
        return self.eps_net(torch.cat([f_s, t_emb, summary, macro], dim=-1))

    # -- training ---------------------------------------------------------- #
    def loss(
        self,
        f0: torch.Tensor,
        hist: torch.Tensor,
        macro: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Denoising loss. f0: (B, K), hist: (B, H, K), macro: (B, M)."""
        B = f0.shape[0]
        dev = f0.device
        s = torch.randint(0, self.train_steps, (B,), generator=generator).to(dev)
        noise = torch.randn(f0.shape, generator=generator).to(dev)
        ab = self.alphas_bar[s].unsqueeze(-1)
        f_s = ab.sqrt() * f0 + (1.0 - ab).sqrt() * noise

        if self.training and self.p_drop > 0:
            drop = (torch.rand(B, generator=generator) < self.p_drop).to(dev)
            macro = torch.where(drop.unsqueeze(-1), self.null_macro.expand_as(macro), macro)

        pred = self._eps(f_s, s, self._summary(hist), macro)
        return F.mse_loss(pred, noise)

    # -- sampling ---------------------------------------------------------- #
    @torch.no_grad()
    def sample(
        self,
        hist: torch.Tensor,
        macro: torch.Tensor,
        n_samples: int = 100,
        steps: int = 10,
        guidance: float = 1.0,
        eta: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw factor samples. hist: (B, H, K), macro: (B, M) -> (J, B, K).

        Uses a strided (DDIM-style) sampler with ``steps`` denoising steps;
        ``eta=1`` gives ancestral (stochastic) sampling, ``eta=0`` is deterministic.
        """
        B = hist.shape[0]
        dev = hist.device
        J = n_samples
        summary = self._summary(hist).repeat(J, 1)  # (J*B, hidden)
        m_cond = macro.repeat(J, 1)
        m_null = self.null_macro.expand_as(m_cond)

        x = torch.randn((J * B, self.n_factors), generator=generator).to(dev)
        ts = torch.linspace(self.train_steps - 1, 0, steps).round().long().tolist()
        for i, t in enumerate(ts):
            s = torch.full((J * B,), t, device=dev, dtype=torch.long)
            eps = self._eps(x, s, summary, m_cond)
            if guidance != 1.0:
                eps_u = self._eps(x, s, summary, m_null)
                eps = eps_u + guidance * (eps - eps_u)
            ab_t = self.alphas_bar[t]
            ab_prev = self.alphas_bar[ts[i + 1]] if i + 1 < len(ts) else x.new_tensor(1.0)
            x0 = ((x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-10.0, 10.0)
            sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            dir_xt = (1.0 - ab_prev - sigma**2).clamp_min(0.0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + dir_xt
            if i + 1 < len(ts):
                x = x + sigma * torch.randn(x.shape, generator=generator).to(dev)
        return x.view(J, B, self.n_factors)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class SDFusion(nn.Module):
    def __init__(self, n_chars: int, n_macro: int, n_train_months: int, cfg: dict):
        super().__init__()
        K = cfg["K"]
        d = cfg["d_model"]
        n = cfg.get("d_state", 16)
        backend = cfg.get("backend", "auto")
        dcfg = cfg["diffusion"]
        self.encoder = TemporalFactorEncoder(
            n_train_months, K, n_macro, d, n, cfg["temporal_layers"], backend
        )
        self.loading = LoadingModule(n_chars, K, d, n, cfg["cross_layers"], backend)
        self.diffusion = DiffusionHead(
            K,
            n_macro,
            hidden=dcfg["hidden"],
            history=dcfg["history"],
            train_steps=dcfg["train_steps"],
            p_drop=dcfg["p_drop"],
        )

    def deterministic_parameters(self):
        return list(self.encoder.parameters()) + list(self.loading.parameters())
