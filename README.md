# SDFusion

PyTorch implementation of **SDFusion: A Conditional Stochastic Discount Factor
with Selective State-Space Dynamics and Latent Factor Diffusion**.

SDFusion estimates a conditional latent-factor model of stock returns,

    r[n, t+1] = Lambda[n, t]' f[t+1] + e[n, t+1]

with three components:

| Component | What it does | Code |
|---|---|---|
| Cross-Sectional Loading Module | maps firm characteristics to factor loadings with channel attention and a bidirectional Mamba scan across firms | `sdfusion/model.py: LoadingModule` |
| Temporal Factor Encoder | learns latent factor realisations with Mamba blocks along time, conditioned on the macroeconomic state, followed by whitening | `sdfusion/model.py: TemporalFactorEncoder` |
| Conditional Latent Diffusion Module | models the distribution of next-month factors given factor history and the macro state | `sdfusion/model.py: DiffusionHead` |

## Repository layout

```
sdfusion/
  ssm.py        selective state-space (Mamba) blocks, two backends
  model.py      loading module, temporal encoder, diffusion head
  losses.py     pricing, orthogonality, instrument and CVaR terms
  data.py       builds the panel from the three CSV files
  trainer.py    three-stage training and the rolling-window driver
train.py        main entry point
evaluate.py     out-of-sample R^2, RMSE, MAE, long-short portfolio
configs/
  default.yaml  settings of the paper (240-month window, K = 10, ...)
  smoke.yaml    tiny configuration for a quick CPU check
scripts/
  make_synthetic_data.py   random data in the expected layout
docs/
  algorithm.tex            pseudocode (LaTeX)
data/
  README.md     input file format
```

## Installation

```bash
pip install -r requirements.txt
# recommended for full-size runs (CUDA GPU required):
pip install mamba-ssm
```

Python 3.10 or newer. Without `mamba-ssm` the code falls back to a pure-PyTorch
selective state-space layer (`backend: torch`), which implements the same
recursion but scans sequences in a Python loop. It is fine for small tests and
too slow for cross-sections of several thousand firms.

## Data

The CRSP and Compustat data are licensed through WRDS and are not included.
Provide three CSV files (see [`data/README.md`](data/README.md) for details):

| File | Columns |
|---|---|
| `characteristics.csv` | `permno`, `date`, one column per characteristic |
| `returns.csv` | `permno`, `date`, `ret` (monthly excess return) |
| `macro.csv` | `date`, one column per macroeconomic variable |

Characteristics and macro variables dated month *t* are paired with the return
of month *t+1*. Characteristics are rank-normalised within each month and
missing entries are masked.

## Quick start (no WRDS access needed)

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python train.py --config configs/smoke.yaml
python evaluate.py --run outputs/smoke
```

This runs the whole pipeline in under a minute on a laptop CPU. The synthetic
data are random, so the resulting numbers are meaningless


## Running on the real panel

```bash
python train.py --config configs/default.yaml \
    --chars data/characteristics.csv \
    --returns data/returns.csv \
    --macro data/macro.csv \
    --out outputs/run1

python evaluate.py --run outputs/run1
```

Useful options: `--max-windows N` (stop after N windows), `--resume` (continue
from `checkpoint.pt` after an interruption), `--seed`, `--device`.

### Outputs

| File | Content |
|---|---|
| `predictions.csv` | `id`, `month`, `r_hat` (point forecast), `r_hat_q05`, `r_hat_q95` (quantiles of the systematic component across factor draws), `r_true` |
| `factors.csv` | forecast and realised latent factors for each test month |
| `gates.csv` | channel-attention weight of each characteristic for each test month |
| `metrics.json`, `long_short.csv` | written by `evaluate.py` |
| `checkpoint.pt`, `config_used.json` | model state and the configuration of the run |

`evaluate.py` reports two R^2 measures, since they differ and are often
confused: the pooled out-of-sample R^2 against a zero forecast,
`1 - sum (r - r_hat)^2 / sum r^2`, and the average monthly cross-sectional R^2.

## Training and evaluation protocol

1. **Estimation window.** 240 months: the first 216 for training, the last 24
   for validation and early stopping.
2. **Three stages** on the first window: (1) encoder and loadings on
   `L_pricing + l1 L_orth + l2 L_inst`; (2) diffusion head on `L_diff` with the
   rest frozen; (3) joint refinement on the full objective including `L_CVaR`.
3. **Forecast.** Sample `J` factor paths from the diffusion head for the month
   after the window, average them, and combine with the loadings:
   `r_hat = Lambda' mean(f)`.
4. **Roll.** Move the window forward one month, warm-start from the previous
   parameters, fine-tune on the joint objective, forecast again.

Pseudocode is in [`docs/algorithm.tex`](docs/algorithm.tex) and summarised below.

```
for each test month tau:
    train block, validation block <- the 240 months before tau
    standardise macro variables with training-block statistics
    if first window:  stage 1 -> stage 2 -> stage 3
    else:             shift factor embeddings by one month, fine-tune (stage 3)

    F          <- TemporalEncoder(F_init, macro[train])          # whitened, T x K
    for t in validation block:                                   # realised factors
        f[t+1] <- ridge regression of r[., t+1] on Lambda[., t]
    c          <- [ GRU(last H factors) || macro[tau] ]
    f_hat^(j)  <- denoise(noise, c)  for j = 1..J                # classifier-free guidance
    r_hat[n]   <- Lambda[n, tau]' mean_j f_hat^(j)
```


## License

Released under the MIT License (see `LICENSE`).
