"""Generate a small synthetic panel in the expected CSV format.

The CRSP/Compustat data used in the paper are licensed and cannot be
redistributed. This script creates random data with the same layout so that the
full pipeline can be run end to end. The numbers it produces carry no economic
meaning.

Example
-------
python scripts/make_synthetic_data.py --out data/synthetic
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--firms", type=int, default=300)
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--chars", type=int, default=8)
    p.add_argument("--macro", type=int, default=4)
    p.add_argument("--factors", type=int, default=3)
    p.add_argument("--missing", type=float, default=0.05, help="share of missing characteristics")
    p.add_argument("--signal", type=float, default=0.01, help="strength of macro -> factor predictability")
    p.add_argument("--start", default="2000-01")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    N, T, P, M, K = a.firms, a.months, a.chars, a.macro, a.factors
    months = pd.period_range(a.start, periods=T, freq="M")

    # persistent macro state and factors whose mean depends on it
    macro = np.zeros((T, M))
    for t in range(1, T):
        macro[t] = 0.9 * macro[t - 1] + rng.normal(scale=0.3, size=M)
    G = rng.normal(scale=a.signal, size=(M, K))
    factors = np.zeros((T, K))
    for t in range(1, T):
        factors[t] = macro[t - 1] @ G + rng.normal(scale=0.04, size=K)

    # persistent characteristics, loadings that depend on them
    W = rng.normal(size=(P, K)) / np.sqrt(P)
    x = rng.normal(size=(N, P))
    industry = rng.integers(0, 10, size=N)
    first = rng.integers(0, T // 3, size=N)  # staggered listing dates
    last = T - rng.integers(0, T // 4, size=N)

    char_rows, ret_rows = [], []
    for t in range(T):
        x = 0.95 * x + np.sqrt(1 - 0.95**2) * rng.normal(size=(N, P))
        alive = (first <= t) & (t < last)
        beta = np.tanh(x @ W)
        if t + 1 < T:
            r_next = beta @ factors[t + 1] + rng.normal(scale=0.10, size=N)
            nxt = alive & (t + 1 < last)
            ret_rows.append(
                pd.DataFrame(
                    {"permno": 10000 + np.flatnonzero(nxt), "date": str(months[t + 1]), "ret": r_next[nxt]}
                )
            )
        obs = x.copy()
        obs[rng.random(size=obs.shape) < a.missing] = np.nan
        df = pd.DataFrame(obs[alive], columns=[f"c{j + 1}" for j in range(P)])
        df.insert(0, "date", str(months[t]))
        df.insert(0, "permno", 10000 + np.flatnonzero(alive))
        df["industry"] = industry[alive]
        df["mktcap"] = np.exp(x[alive, 0] + 5.0)
        char_rows.append(df)

    os.makedirs(a.out, exist_ok=True)
    pd.concat(char_rows).to_csv(os.path.join(a.out, "characteristics.csv"), index=False)
    pd.concat(ret_rows).to_csv(os.path.join(a.out, "returns.csv"), index=False)
    mdf = pd.DataFrame(macro, columns=[f"m{j + 1}" for j in range(M)])
    mdf.insert(0, "date", [str(m) for m in months])
    mdf.to_csv(os.path.join(a.out, "macro.csv"), index=False)
    print(f"wrote characteristics.csv, returns.csv, macro.csv to {a.out}")


if __name__ == "__main__":
    main()
