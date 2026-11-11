"""Staged training and rolling-window out-of-sample forecasting.

Estimation window (per roll):

    | <------ train_months ------> | <-- val_months --> | test month(s) |

* Factors of the *training* months come from the temporal encoder.
* Factors of validation and test months are never trained. Once a month's
  returns are realised, its factor is recovered by a cross-sectional ridge
  regression of returns on the estimated loadings (the factor-mimicking
  portfolio return), and is appended to the factor history.
* The forecast for a month is r_hat = Lambda' E[f], where E[f] is the mean of
  J samples from the diffusion head conditioned on the factor history and the
  current macro state. Only information dated before the target month is used.
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from . import losses
from .model import SDFusion


def _warmup_cosine(step: int, total: int, warmup_frac: float) -> float:
    warm = max(int(total * warmup_frac), 1)
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(total - warm, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


class Trainer:
    def __init__(self, cfg: dict, panel, device: torch.device):
        self.cfg = cfg
        self.panel = panel
        self.device = device
        self.wcfg = cfg["window"]
        self.tcfg = cfg["train"]
        self.lcfg = cfg["loss"]
        self.dcfg = cfg["model"]["diffusion"]
        self.n_train = int(self.wcfg["train_months"])
        self.n_val = int(self.wcfg["val_months"])
        self.rng = np.random.default_rng(self.tcfg.get("seed", 0))
        self.model = SDFusion(
            panel.n_chars, panel.n_macro, self.n_train, cfg["model"]
        ).to(device)
        self.steps_per_epoch = max(
            math.ceil(self.n_train / int(self.tcfg["t_chunk"])), 1
        )

    # ------------------------------------------------------------------ #
    # window bookkeeping
    # ------------------------------------------------------------------ #
    def set_window(self, start: int) -> None:
        self.start = start
        self.train_idx = np.arange(start, start + self.n_train)
        self.val_idx = np.arange(start + self.n_train, start + self.n_train + self.n_val)
        # Macro variables are standardised with training-month statistics only.
        m = self.panel.macro
        mu = np.nanmean(m[self.train_idx], axis=0)
        sd = np.nanstd(m[self.train_idx], axis=0) + 1e-8
        z = np.clip(np.nan_to_num((m - mu) / sd), -5.0, 5.0)
        self.macro = torch.tensor(z, dtype=torch.float32, device=self.device)

    def _t(self, a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # building blocks
    # ------------------------------------------------------------------ #
    def train_factors(self) -> torch.Tensor:
        return self.model.encoder(self.macro[self.train_idx])

    def _history(self, seq: torch.Tensor, positions) -> torch.Tensor:
        """History windows. seq: (L, K). For each position p return seq[p-H:p],
        left-padded with zeros, shape (len(positions), H, K)."""
        H = self.model.diffusion.history
        pad = torch.cat([seq.new_zeros(H, seq.shape[1]), seq], dim=0)
        return torch.stack([pad[p : p + H] for p in positions], dim=0)

    @torch.no_grad()
    def _loadings(self, t: int):
        """Full cross-section loadings for pair t (evaluation mode)."""
        it = self.panel.month(t, self.tcfg.get("eval_n_sub"), self.rng)
        x = self._t(it.x).unsqueeze(0)
        mask = self._t(it.mask, torch.bool).unsqueeze(0)
        lam, w = self.model.loading(x, mask)
        return it, lam[0], w[0]

    @staticmethod
    def _realized_factor(lam: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-sectional ridge regression of returns on loadings."""
        valid = torch.isfinite(y)
        L, r = lam[valid], y[valid]
        K = L.shape[1]
        gram = L.T @ L
        ridge = 1e-4 * gram.diagonal().mean().clamp_min(1e-8)
        eye = torch.eye(K, device=L.device, dtype=L.dtype)
        return torch.linalg.solve(gram + ridge * eye, L.T @ r)

    @torch.no_grad()
    def _months_pass(self, months):
        """Loadings, returns and realised factors for a list of pairs."""
        out = []
        for t in months:
            it, lam, w = self._loadings(int(t))
            y = self._t(it.y)
            out.append(dict(t=int(t), it=it, lam=lam, y=y, w=w, f=self._realized_factor(lam, y)))
        return out

    # ------------------------------------------------------------------ #
    # losses on a training batch
    # ------------------------------------------------------------------ #
    def _deterministic_losses(self, F: torch.Tensor, perm_prob: float, with_cvar: bool):
        c = min(int(self.tcfg["t_chunk"]), self.n_train)
        s0 = int(self.rng.integers(0, self.n_train - c + 1))
        pos = np.arange(s0, s0 + c)
        x, mask, y = self.panel.batch(self.train_idx[pos], int(self.tcfg["n_sub"]), self.rng)
        x, mask, y = self._t(x), self._t(mask, torch.bool), self._t(y)
        lam, _ = self.model.loading(x, mask, perm_prob)
        r_hat = (lam * F[pos].unsqueeze(1)).sum(-1)  # (B, n)
        valid = torch.isfinite(y)
        out = {
            "pricing": losses.pricing_loss(r_hat, y, valid),
            "orth": losses.orthogonality_loss(F),
            "inst": losses.instrument_loss(lam, x),
        }
        if with_cvar:
            out["cvar"] = losses.cvar_loss(r_hat, y, valid, self.lcfg["cvar_alpha"])
        return out

    def _diffusion_loss(self, F: torch.Tensor, generator=None) -> torch.Tensor:
        pos = range(self.n_train)
        hist = self._history(F, pos)
        return self.model.diffusion.loss(F, hist, self.macro[self.train_idx], generator)

    # ------------------------------------------------------------------ #
    # validation metrics (one per stage)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def val_fit(self) -> float:
        """Stage 1: cross-sectional fit of realised returns on loadings."""
        self.model.eval()
        sse, n = 0.0, 0
        for d in self._months_pass(self.val_idx):
            v = torch.isfinite(d["y"])
            res = d["y"][v] - d["lam"][v] @ d["f"]
            sse += float((res**2).sum())
            n += int(v.sum())
        return sse / max(n, 1)

    @torch.no_grad()
    def val_diffusion(self, F_train: torch.Tensor, val_pass) -> float:
        """Stage 2: denoising loss on realised validation-month factors."""
        self.model.eval()
        f_val = torch.stack([d["f"] for d in val_pass])
        seq = torch.cat([F_train, f_val], dim=0)
        pos = range(self.n_train, self.n_train + self.n_val)
        hist = self._history(seq, pos)
        g = torch.Generator().manual_seed(1234)
        vals = [
            float(self.model.diffusion.loss(f_val, hist, self.macro[self.val_idx], g))
            for _ in range(8)
        ]
        return float(np.mean(vals))

    @torch.no_grad()
    def _forecast(self, F_train: torch.Tensor, known, targets, seed: int | None = None):
        """Predict returns for ``targets`` (list of month dicts from _months_pass).

        ``known`` are month dicts whose factors are already realised and precede
        the targets. Realised factors of earlier targets are added to the
        history of later ones, which is information available at that time.
        """
        fs = [F_train] + [d["f"].unsqueeze(0) for d in known + targets]
        seq = torch.cat(fs, dim=0)
        first = self.n_train + len(known)
        pos = range(first, first + len(targets))
        hist = self._history(seq, pos)
        macro = self.macro[[d["t"] for d in targets]]
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        samples = self.model.diffusion.sample(
            hist,
            macro,
            n_samples=int(self.dcfg["n_samples"]),
            steps=int(self.dcfg["sample_steps"]),
            guidance=float(self.dcfg["guidance"]),
            eta=float(self.dcfg["eta"]),
            generator=g,
        )  # (J, B, K)
        return samples

    @torch.no_grad()
    def val_predictive(self) -> float:
        """Stage 3: out-of-sample style MSE on the validation months."""
        self.model.eval()
        F_train = self.train_factors()
        val_pass = self._months_pass(self.val_idx)
        f_hat = self._forecast(F_train, [], val_pass, seed=1234).mean(0)  # (B, K)
        sse, n = 0.0, 0
        for d, f in zip(val_pass, f_hat):
            v = torch.isfinite(d["y"])
            sse += float(((d["y"][v] - d["lam"][v] @ f) ** 2).sum())
            n += int(v.sum())
        return sse / max(n, 1)

    # ------------------------------------------------------------------ #
    # generic stage runner with early stopping
    # ------------------------------------------------------------------ #
    def _run_stage(self, name, params, step_fn, val_fn, epochs, val_every):
        if epochs <= 0:
            return
        opt = torch.optim.AdamW(
            params, lr=float(self.tcfg["lr"]), weight_decay=float(self.tcfg["weight_decay"])
        )
        total = epochs * self.steps_per_epoch
        warm = float(self.tcfg.get("warmup_frac", 0.05))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: _warmup_cosine(s, total, warm)
        )
        best, best_state, bad, step = float("inf"), None, 0, 0
        patience = int(self.tcfg["patience"])
        t0 = time.time()
        for ep in range(epochs):
            self.model.train()
            last = 0.0
            for _ in range(self.steps_per_epoch):
                loss = step_fn(step, total)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(self.tcfg["grad_clip"]))
                opt.step()
                sched.step()
                step += 1
                last = float(loss.detach())
            if (ep + 1) % val_every == 0 or ep == epochs - 1:
                v = val_fn()
                improved = v < best - 1e-9
                if improved:
                    best, bad = v, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    bad += 1
                print(
                    f"  [{name}] epoch {ep + 1:4d}/{epochs}  train {last:.5f}  "
                    f"val {v:.5f}{' *' if improved else ''}  ({time.time() - t0:.0f}s)"
                )
                if bad >= patience:
                    print(f"  [{name}] early stop at epoch {ep + 1}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------------------------------------------------ #
    # the three stages
    # ------------------------------------------------------------------ #
    def stage1(self, epochs: int) -> None:
        """Deterministic SDF learning: encoder + loadings, diffusion untouched."""
        lam1, lam2 = self.lcfg["lambda_orth"], self.lcfg["lambda_inst"]
        p0 = float(self.tcfg.get("perm_prob", 0.0))

        def step(s, total):
            perm = p0 * max(0.0, 1.0 - s / max(total, 1))  # annealed to zero
            L = self._deterministic_losses(self.train_factors(), perm, with_cvar=False)
            return L["pricing"] + lam1 * L["orth"] + lam2 * L["inst"]

        self._run_stage(
            "stage1",
            self.model.deterministic_parameters(),
            step,
            self.val_fit,
            epochs,
            int(self.tcfg["val_every"]),
        )

    def stage2(self, epochs: int) -> None:
        """Generative factor learning: diffusion head on frozen factors."""
        self.model.eval()
        with torch.no_grad():
            F_train = self.train_factors()
            val_pass = self._months_pass(self.val_idx)

        def step(s, total):
            return self._diffusion_loss(F_train)

        self._run_stage(
            "stage2",
            list(self.model.diffusion.parameters()),
            step,
            lambda: self.val_diffusion(F_train, val_pass),
            epochs,
            1,
        )

    def stage3(self, epochs: int, name: str = "stage3") -> None:
        """Joint refinement under the full objective (Eq. 14)."""
        lc = self.lcfg

        def step(s, total):
            F = self.train_factors()
            L = self._deterministic_losses(F, 0.0, with_cvar=True)
            return (
                L["pricing"]
                + lc["lambda_orth"] * L["orth"]
                + lc["lambda_inst"] * L["inst"]
                + lc["lambda_diff"] * self._diffusion_loss(F)
                + lc["lambda_cvar"] * L["cvar"]
            )

        self._run_stage(
            name,
            list(self.model.parameters()),
            step,
            self.val_predictive,
            epochs,
            int(self.tcfg["val_every"]),
        )

    # ------------------------------------------------------------------ #
    # out-of-sample prediction
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, test_idx):
        """Forecast the pairs in ``test_idx`` (which follow the validation block)."""
        self.model.eval()
        F_train = self.train_factors()
        val_pass = self._months_pass(self.val_idx)
        test_pass = self._months_pass(test_idx)
        samples = self._forecast(F_train, val_pass, test_pass)  # (J, B, K)
        f_hat = samples.mean(0)

        pred_rows, factor_rows, gate_rows = [], [], []
        for b, d in enumerate(test_pass):
            month = str(self.panel.target_months[d["t"]])
            r_draws = samples[:, b] @ d["lam"].T  # (J, n) systematic return draws
            pred_rows.append(
                pd.DataFrame(
                    {
                        "id": d["it"].ids,
                        "month": month,
                        "r_hat": (d["lam"] @ f_hat[b]).cpu().numpy(),
                        "r_hat_q05": torch.quantile(r_draws, 0.05, dim=0).cpu().numpy(),
                        "r_hat_q95": torch.quantile(r_draws, 0.95, dim=0).cpu().numpy(),
                        "r_true": d["y"].cpu().numpy(),
                    }
                )
            )
            row = {"month": month}
            row.update({f"f_hat_{k + 1}": float(v) for k, v in enumerate(f_hat[b])})
            row.update({f"f_real_{k + 1}": float(v) for k, v in enumerate(d["f"])})
            factor_rows.append(row)
            g = {"month": month}
            g.update({c: float(v) for c, v in zip(self.panel.char_cols, d["w"])})
            gate_rows.append(g)
        return pd.concat(pred_rows), pd.DataFrame(factor_rows), pd.DataFrame(gate_rows)

    def roll(self, k: int) -> None:
        self.model.encoder.roll(k)


# ---------------------------------------------------------------------- #
# rolling driver
# ---------------------------------------------------------------------- #
def _append_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_rolling(cfg: dict, panel, device: torch.device) -> None:
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    tr = Trainer(cfg, panel, device)
    w, t = cfg["window"], cfg["train"]
    span = tr.n_train + tr.n_val
    step = int(w.get("step_months", 1))
    if len(panel) <= span:
        raise ValueError(
            f"panel has {len(panel)} usable months but the estimation window needs {span}"
        )
    starts = list(range(0, len(panel) - span, step))
    if w.get("max_windows"):
        starts = starts[: int(w["max_windows"])]

    ckpt_path = os.path.join(out, "checkpoint.pt")
    files = {k: os.path.join(out, f"{k}.csv") for k in ("predictions", "factors", "gates")}
    done = -1
    if cfg.get("resume") and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        tr.model.load_state_dict(ck["model"])
        done = ck["window"]
        print(f"[run] resuming after window {done + 1}/{len(starts)}")
    else:
        for f in files.values():
            if os.path.exists(f):
                os.remove(f)

    n_par = sum(p.numel() for p in tr.model.parameters())
    print(f"[run] {len(starts)} window(s), {n_par:,} parameters, device={device}")

    for i, s in enumerate(starts):
        if i <= done:
            continue
        tr.set_window(s)
        test_idx = np.arange(s + span, min(s + span + step, len(panel)))
        print(
            f"[window {i + 1}/{len(starts)}] train {panel.months[s]}..{panel.months[s + tr.n_train - 1]}"
            f"  val ..{panel.months[s + span - 1]}"
            f"  test target {panel.target_months[test_idx[0]]}..{panel.target_months[test_idx[-1]]}"
        )
        if i == 0:
            tr.stage1(int(t["epochs_stage1"]))
            tr.stage2(int(t["epochs_stage2"]))
            tr.stage3(int(t["epochs_stage3"]))
        else:
            tr.roll(step)  # warm start from the previous window
            tr.stage3(int(t["finetune_epochs"]), name="finetune")

        preds, factors, gates = tr.predict(test_idx)
        _append_csv(preds, files["predictions"])
        _append_csv(factors, files["factors"])
        _append_csv(gates, files["gates"])
        torch.save({"model": tr.model.state_dict(), "window": i}, ckpt_path)

    print(f"[run] done. Outputs in {out}")
