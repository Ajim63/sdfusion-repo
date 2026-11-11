"""Evaluate out-of-sample forecasts written by train.py.

Reads <run>/predictions.csv and reports

* pooled out-of-sample R^2 against a zero forecast:  1 - sum (r - r_hat)^2 / sum r^2
* average monthly cross-sectional R^2 (benchmark: that month's cross-sectional mean)
* RMSE and MAE (pooled over firm-months)
* equal-weighted long-short decile portfolio: mean, volatility, Sharpe ratio

Example
-------
python evaluate.py --run outputs/run1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def forecast_metrics(df: pd.DataFrame) -> dict:
    err = df["r_true"] - df["r_hat"]
    r2_zero = 1.0 - float((err**2).sum()) / float((df["r_true"] ** 2).sum())

    def _cs_r2(g):
        e = g["r_true"] - g["r_hat"]
        d = g["r_true"] - g["r_true"].mean()
        return 1.0 - float((e**2).sum()) / max(float((d**2).sum()), 1e-12)

    cs = df.groupby("month")[["r_true", "r_hat"]].apply(_cs_r2)
    return {
        "n_firm_months": int(len(df)),
        "n_months": int(df["month"].nunique()),
        "r2_oos_zero_benchmark": r2_zero,
        "r2_cross_sectional_avg": float(cs.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
    }


def long_short(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for month, g in df.groupby("month"):
        if len(g) < 2 * n_bins:
            continue
        rank = g["r_hat"].rank(method="first")
        bins = pd.qcut(rank, n_bins, labels=False)
        top = g.loc[bins == n_bins - 1, "r_true"].mean()
        bot = g.loc[bins == 0, "r_true"].mean()
        rows.append({"month": month, "long": top, "short": bot, "long_short": top - bot})
    ls = pd.DataFrame(rows)
    if not ls.empty:
        ls["cum_long_short"] = ls["long_short"].cumsum()
    return ls


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", required=True, help="output directory of train.py")
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.run, "predictions.csv"))
    df = df[np.isfinite(df["r_true"]) & np.isfinite(df["r_hat"])]
    metrics = forecast_metrics(df)

    ls = long_short(df, args.bins)
    if not ls.empty:
        mu, sd = ls["long_short"].mean(), ls["long_short"].std(ddof=1)
        metrics.update(
            {
                "long_short_mean_monthly": float(mu),
                "long_short_vol_monthly": float(sd),
                "long_short_sharpe_annualized": float(np.sqrt(12) * mu / sd) if sd > 0 else None,
            }
        )
        ls.to_csv(os.path.join(args.run, "long_short.csv"), index=False)

    with open(os.path.join(args.run, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    for k, v in metrics.items():
        print(f"{k:32s} {v:.6f}" if isinstance(v, float) else f"{k:32s} {v}")


if __name__ == "__main__":
    main()
