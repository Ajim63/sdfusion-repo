"""Panel construction from separate characteristics, returns and macro CSV files.

Expected inputs (long format, one row per firm-month):

characteristics CSV : <id_col>, <date_col>, char_1, ..., char_P
                      (optional columns used only to order firms: industry, size)
returns CSV         : <id_col>, <date_col>, <return_col>      (monthly excess return)
macro CSV           : <date_col>, macro_1, ..., macro_M        (one row per month)

Timing convention. A "pair" t consists of the characteristics and macro state
observed at month t and the excess return realised in month t+1. Characteristics
dated t must therefore be known at the end of month t. If your returns file
already stores the next-month return on the row dated t, set
``returns_are_forward: true`` in the config.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _to_month(s: pd.Series) -> pd.PeriodIndex:
    """Parse dates given as 198001, 19800131, '1980-01-31', ... into monthly periods."""
    if pd.api.types.is_numeric_dtype(s):
        txt = s.astype("int64").astype(str)
        n = txt.str.len().iloc[0]
        fmt = {6: "%Y%m", 8: "%Y%m%d"}.get(int(n))
        dt = pd.to_datetime(txt, format=fmt) if fmt else pd.to_datetime(txt)
    else:
        dt = pd.to_datetime(s)
    return pd.PeriodIndex(dt, freq="M")


@dataclass
class MonthData:
    ids: np.ndarray  # (n,)  firm identifiers, in scan order
    x: np.ndarray  # (n, P) normalised characteristics, missing entries set to 0
    mask: np.ndarray  # (n, P) True where the characteristic is observed
    y: np.ndarray  # (n,)  next-month excess return, NaN if unobserved


class Panel:
    """Month-by-month (ragged) storage of the firm panel."""

    def __init__(self, months, macro, items, char_cols, macro_cols):
        self.months = list(months)  # characteristic month of each pair
        self.target_months = [m + 1 for m in self.months]
        self.macro = macro.astype(np.float32)  # (T, M), raw values
        self.items: list[MonthData] = items
        self.char_cols = list(char_cols)
        self.macro_cols = list(macro_cols)

    def __len__(self) -> int:
        return len(self.months)

    @property
    def n_chars(self) -> int:
        return len(self.char_cols)

    @property
    def n_macro(self) -> int:
        return len(self.macro_cols)

    def month(self, t: int, n_sub: int | None = None, rng=None) -> MonthData:
        """Full cross-section of pair t, optionally subsampled (order preserved)."""
        it = self.items[t]
        if n_sub is None or len(it.ids) <= n_sub:
            return it
        rng = rng or np.random.default_rng()
        idx = np.sort(rng.choice(len(it.ids), size=n_sub, replace=False))
        return MonthData(it.ids[idx], it.x[idx], it.mask[idx], it.y[idx])

    def batch(self, month_idx, n_sub: int, rng):
        """Stack several months into (B, n, P) arrays with a common n.

        Each month contributes ``n = min(n_sub, smallest cross-section)`` firms,
        drawn at random and kept in scan order, so no padding is needed.
        """
        n = min([n_sub] + [len(self.items[t].ids) for t in month_idx])
        xs, ms, ys = [], [], []
        for t in month_idx:
            it = self.items[t]
            idx = np.sort(rng.choice(len(it.ids), size=n, replace=False))
            xs.append(it.x[idx])
            ms.append(it.mask[idx])
            ys.append(it.y[idx])
        return np.stack(xs), np.stack(ms), np.stack(ys)


def load_panel(cfg: dict) -> Panel:
    id_col = cfg.get("id_col", "permno")
    date_col = cfg.get("date_col", "date")
    ret_col = cfg.get("return_col", "ret")
    industry_col = cfg.get("industry_col")
    size_col = cfg.get("size_col")
    exclude = set(cfg.get("exclude_cols") or [])
    normalize = cfg.get("normalize", "rank")
    min_firms = int(cfg.get("min_firms_per_month", 50))

    # ---- characteristics ------------------------------------------------- #
    chars = pd.read_csv(cfg["characteristics_csv"])
    chars["_month"] = _to_month(chars[date_col])
    chars = chars.drop_duplicates(subset=[id_col, "_month"], keep="last")

    char_cols = cfg.get("characteristic_cols")
    if not char_cols:
        reserved = {id_col, date_col, "_month"} | exclude
        if industry_col:
            reserved.add(industry_col)
        char_cols = [c for c in chars.columns if c not in reserved]
    chars[char_cols] = chars[char_cols].apply(pd.to_numeric, errors="coerce")

    # Ordering key for the cross-sectional scan: industry, then size (descending),
    # then identifier. Uses raw values, before any normalisation.
    if industry_col:
        chars["_ind"] = pd.factorize(chars[industry_col], sort=True)[0]  # NaN -> -1
    else:
        chars["_ind"] = 0
    if size_col:
        chars["_size"] = -pd.to_numeric(chars[size_col], errors="coerce").fillna(-np.inf)
    else:
        chars["_size"] = 0.0

    # ---- returns --------------------------------------------------------- #
    rets = pd.read_csv(cfg["returns_csv"], usecols=[id_col, date_col, ret_col])
    rets["_month"] = _to_month(rets[date_col])
    if not cfg.get("returns_are_forward", False):
        rets["_month"] = rets["_month"] - 1  # return of month t+1 is the target of pair t
    rets = rets.drop_duplicates(subset=[id_col, "_month"], keep="last")
    rets[ret_col] = pd.to_numeric(rets[ret_col], errors="coerce")
    chars = chars.merge(
        rets[[id_col, "_month", ret_col]].rename(columns={ret_col: "_y"}),
        on=[id_col, "_month"],
        how="left",
    )

    # ---- macro ----------------------------------------------------------- #
    macro = pd.read_csv(cfg["macro_csv"])
    macro["_month"] = _to_month(macro[date_col])
    macro_cols = cfg.get("macro_cols") or [
        c for c in macro.columns if c not in {date_col, "_month"}
    ]
    macro = macro.drop_duplicates("_month").set_index("_month")[macro_cols]
    macro = macro.apply(pd.to_numeric, errors="coerce")

    # ---- cross-sectional normalisation ---------------------------------- #
    mask_df = chars[char_cols].notna()
    if normalize == "rank":
        # rank within month, mapped to (-0.5, 0.5]
        chars[char_cols] = chars.groupby("_month")[char_cols].rank(pct=True) - 0.5
    elif normalize == "zscore":
        g = chars.groupby("_month")[char_cols]
        chars[char_cols] = (chars[char_cols] - g.transform("mean")) / (
            g.transform("std") + 1e-8
        )
    elif normalize != "none":
        raise ValueError(f"unknown normalize option: {normalize}")

    chars = chars.sort_values(["_month", "_ind", "_size", id_col], kind="stable")
    mask_df = mask_df.loc[chars.index]

    x_all = chars[char_cols].to_numpy(dtype=np.float32)
    m_all = mask_df.to_numpy(dtype=bool)
    x_all = np.where(m_all, x_all, 0.0).astype(np.float32)
    y_all = chars["_y"].to_numpy(dtype=np.float32)
    id_all = chars[id_col].to_numpy()
    month_all = chars["_month"].to_numpy()

    # contiguous blocks per month (data are sorted by month)
    change = np.flatnonzero(month_all[1:] != month_all[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(month_all)]])

    months, items, macro_rows = [], [], []
    for s, e in zip(starts, ends):
        mth = month_all[s]
        if mth not in macro.index:
            continue
        keep = m_all[s:e].any(axis=1)  # drop firms with no characteristics at all
        y = y_all[s:e][keep]
        if keep.sum() < min_firms or np.isfinite(y).sum() < min_firms:
            continue
        months.append(mth)
        items.append(MonthData(id_all[s:e][keep], x_all[s:e][keep], m_all[s:e][keep], y))
        macro_rows.append(macro.loc[mth].to_numpy(dtype=np.float32))

    if not months:
        raise ValueError("no usable months: check column names, dates and file paths")

    gaps = [i for i in range(1, len(months)) if months[i] != months[i - 1] + 1]
    if gaps:
        print(
            f"[data] warning: {len(gaps)} gap(s) in the monthly sequence, "
            f"first after {months[gaps[0] - 1]}"
        )

    panel = Panel(months, np.stack(macro_rows), items, char_cols, macro_cols)
    sizes = [len(it.ids) for it in items]
    print(
        f"[data] {len(panel)} months ({months[0]} to {months[-1]}), "
        f"{panel.n_chars} characteristics, {panel.n_macro} macro variables, "
        f"{int(np.mean(sizes))} firms per month on average"
    )
    return panel
