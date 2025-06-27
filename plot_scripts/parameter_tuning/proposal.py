#!/usr/bin/env python3
"""plot.py
Generate side‑by‑side heatmaps of
  • highest validation accuracy achieved (left)
  • overall compression metric (right)
for every (batch_compression, token_compression) configuration contained in a
results directory produced by your split‑learning experiments.

Usage
-----
$ python plot.py --results_dir /path/to/results \
                --output /path/to/heatmaps.png

Dependencies: Python ≥3.8, pandas, numpy, matplotlib.
Install with: pip install pandas matplotlib numpy
"""

from __future__ import annotations
import argparse
import json
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

PARAM_DIR_RE = re.compile(
    r"params=\{'batch_compression': ([0-9.]+), 'token_compression': ([0-9.]+)\}")


def extract_metrics(directory: str) -> Tuple[float, float, float, float]:
    """Return (batch_c, token_c, max_val_acc, compression) for *directory*.

    Raises ValueError if the directory name or JSON file is malformed.
    """
    match = PARAM_DIR_RE.fullmatch(os.path.basename(directory))
    if not match:
        raise ValueError("Directory name does not match expected pattern: " + directory)

    batch_c = float(match.group(1))
    token_c = float(match.group(2))

    json_path = os.path.join(directory, "training_results.json")
    with open(json_path, "r") as f:
        results = json.load(f)

    val_accuracies = results.get("Val accuracies", [])
    if not val_accuracies:
        raise ValueError(f"No validation accuracies in {json_path}")
    max_val_acc = max(val_accuracies)

    compression = results.get("Compression")
    if compression is None:
        # Try alternative keys, if any
        compression = results.get("Compression ratio")
        if compression is None:
            raise ValueError(f"No compression metric in {json_path}")

    return batch_c, token_c, max_val_acc, compression


def gather_dataframe(results_dir: str) -> pd.DataFrame:
    """Scan *results_dir* and build a DataFrame with metrics for each run."""
    data = []
    for entry in os.scandir(results_dir):
        if not entry.is_dir():
            continue
        try:
            metrics = extract_metrics(entry.path)
            data.append(metrics)
        except (ValueError, FileNotFoundError) as exc:
            # Skip malformed folders but notify in console.
            print(f"[warn] {exc}")

    if not data:
        raise RuntimeError("No valid experiment folders found in " + results_dir)

    return pd.DataFrame(
        data,
        columns=["batch_compression", "token_compression", "val_acc", "compression"],
    )


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a pivoted DataFrame indexed by batch_compression (rows) and
    token_compression (cols) for *metric* values."""
    pivot = (
        df.pivot(index="batch_compression", columns="token_compression", values=metric)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return pivot


def make_heatmaps(df: pd.DataFrame, output_file: str) -> None:
    """Create and save the two‑panel heatmap figure."""
    acc_grid = pivot_metric(df, "val_acc")
    comp_grid = pivot_metric(df, "compression")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left: validation accuracy ------------------------------------------------
    im0 = axes[0].imshow(acc_grid, origin="lower", aspect="auto",  norm=PowerNorm(gamma=2.5))
    axes[0].set_title("Max Validation Accuracy")
    axes[0].set_xlabel("Token Compression")
    axes[0].set_ylabel("Batch Compression")
    axes[0].set_xticks(range(len(acc_grid.columns)))
    axes[0].set_xticklabels(acc_grid.columns)
    axes[0].set_yticks(range(len(acc_grid.index)))
    axes[0].set_yticklabels(acc_grid.index)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Right: compression -------------------------------------------------------
    im1 = axes[1].imshow(comp_grid, origin="lower", aspect="auto")
    axes[1].set_title("Total Compression")
    axes[1].set_xlabel("Token Compression")
    axes[1].set_ylabel("Batch Compression")
    axes[1].set_xticks(range(len(comp_grid.columns)))
    axes[1].set_xticklabels(comp_grid.columns)
    axes[1].set_yticks(range(len(comp_grid.index)))
    axes[1].set_yticklabels(comp_grid.index)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Compression Parameters Tuning on Food-101 on 10 epochs", fontsize=16)


    # Get axis positions corresponding to token/batch values
    token_vals = np.array(acc_grid.columns.values, dtype=float)
    batch_vals = np.array(acc_grid.index.values, dtype=float)

    # Compute line: b = sqrt(t)
    token_vals_dense = np.linspace(token_vals.min(), token_vals.max(), 500)
    batch_line_dense = token_vals_dense ** (1/4)

    # Filter out-of-bound points
    valid = (batch_line_dense >= batch_vals.min()) & (batch_line_dense <= batch_vals.max())
    token_vals_dense = token_vals_dense[valid]
    batch_line_dense = batch_line_dense[valid]

    # Interpolate: map real values to fractional indices
    x = np.interp(token_vals_dense, token_vals, np.arange(len(token_vals)))
    y = np.interp(batch_line_dense, batch_vals, np.arange(len(batch_vals)))

    # Plot on both heatmaps
 
    axes[0].plot(x, y, color="red", linestyle="--", linewidth=2, label=r"euristic: $b = \sqrt[4]{t}$")
    axes[0].legend()

    

    plt.savefig(output_file, dpi=300)
    print(f"[info] Figure saved to {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    df = gather_dataframe("/home/federico/Desktop/Split_Learning/results/proposal_parameter_tuning/food-101/deit_tiny_patch16_224.fb_in1k/proposal/communication=clean/")
    make_heatmaps(df, "plots/proposal_parameter_tuning/heatmaps")


if __name__ == "__main__":
    main()
