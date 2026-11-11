"""Train SDFusion with a rolling window and write out-of-sample forecasts.

Example
-------
python train.py --config configs/default.yaml \
    --chars data/characteristics.csv --returns data/returns.csv --macro data/macro.csv \
    --out outputs/run1
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from sdfusion import load_panel, run_rolling


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--chars", help="characteristics CSV (overrides the config)")
    p.add_argument("--returns", help="returns CSV (overrides the config)")
    p.add_argument("--macro", help="macro CSV (overrides the config)")
    p.add_argument("--out", help="output directory (overrides the config)")
    p.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None, help="stop after this many windows")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.chars:
        cfg["data"]["characteristics_csv"] = args.chars
    if args.returns:
        cfg["data"]["returns_csv"] = args.returns
    if args.macro:
        cfg["data"]["macro_csv"] = args.macro
    if args.out:
        cfg["output_dir"] = args.out
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.max_windows is not None:
        cfg["window"]["max_windows"] = args.max_windows
    cfg["resume"] = bool(args.resume)

    seed = int(cfg["train"].get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    panel = load_panel(cfg["data"])
    run_rolling(cfg, panel, device)


if __name__ == "__main__":
    main()
