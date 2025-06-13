# #!/usr/bin/env python3
# """plot.py
# Generate side‑by‑side heatmaps of
#   • highest validation accuracy achieved (left)
#   • overall compression metric (right)
# for every (batch_compression, token_compression) configuration contained in a
# results directory produced by your split‑learning experiments.

# Usage
# -----
# $ python plot.py --results_dir /path/to/results \
#                 --output /path/to/heatmaps.png

# Dependencies: Python ≥3.8, pandas, numpy, matplotlib.
# Install with: pip install pandas matplotlib numpy
# """

# from __future__ import annotations
# import argparse
# import json
# import os
# import re
# from typing import Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import PowerNorm
# # ──────────────────────────────────────────────────────────────────────────────
# # Helper functions
# # ──────────────────────────────────────────────────────────────────────────────

# PARAM_DIR_RE = re.compile(
#     r"params=\{'batch_compression': ([0-9.]+), 'token_compression': ([0-9.]+)\}")


# def extract_metrics(directory: str) -> Tuple[float, float, float, float]:
#     """Return (batch_c, token_c, max_val_acc, compression) for *directory*.

#     Raises ValueError if the directory name or JSON file is malformed.
#     """
#     match = PARAM_DIR_RE.fullmatch(os.path.basename(directory))
#     if not match:
#         raise ValueError("Directory name does not match expected pattern: " + directory)

#     batch_c = float(match.group(1))
#     token_c = float(match.group(2))

#     json_path = os.path.join(directory, "training_results.json")
#     with open(json_path, "r") as f:
#         results = json.load(f)

#     val_accuracies = results.get("Val accuracies", [])
#     if not val_accuracies:
#         raise ValueError(f"No validation accuracies in {json_path}")
#     max_val_acc = max(val_accuracies)

#     compression = results.get("Compression")
#     if compression is None:
#         # Try alternative keys, if any
#         compression = results.get("Compression ratio")
#         if compression is None:
#             raise ValueError(f"No compression metric in {json_path}")

#     return batch_c, token_c, max_val_acc, compression


# def gather_dataframe(results_dir: str) -> pd.DataFrame:
#     """Scan *results_dir* and build a DataFrame with metrics for each run."""
#     data = []
#     for entry in os.scandir(results_dir):
#         if not entry.is_dir():
#             continue
#         try:
#             metrics = extract_metrics(entry.path)
#             data.append(metrics)
#         except (ValueError, FileNotFoundError) as exc:
#             # Skip malformed folders but notify in console.
#             print(f"[warn] {exc}")

#     if not data:
#         raise RuntimeError("No valid experiment folders found in " + results_dir)

#     return pd.DataFrame(
#         data,
#         columns=["batch_compression", "token_compression", "val_acc", "compression"],
#     )


# def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
#     """Return a pivoted DataFrame indexed by batch_compression (rows) and
#     token_compression (cols) for *metric* values."""
#     pivot = (
#         df.pivot(index="batch_compression", columns="token_compression", values=metric)
#         .sort_index(axis=0)
#         .sort_index(axis=1)
#     )
#     return pivot


# def make_heatmaps(df: pd.DataFrame, output_file: str) -> None:
#     """Create and save the two‑panel heatmap figure."""
#     acc_grid = pivot_metric(df, "val_acc")
#     comp_grid = pivot_metric(df, "compression")

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

#     # Left: validation accuracy ------------------------------------------------
#     im0 = axes[0].imshow(acc_grid, origin="lower", aspect="auto",  norm=PowerNorm(gamma=2.5))
#     axes[0].set_title("Max Validation Accuracy")
#     axes[0].set_xlabel("Token Compression")
#     axes[0].set_ylabel("Batch Compression")
#     axes[0].set_xticks(range(len(acc_grid.columns)))
#     axes[0].set_xticklabels(acc_grid.columns)
#     axes[0].set_yticks(range(len(acc_grid.index)))
#     axes[0].set_yticklabels(acc_grid.index)
#     fig.colorbar(im0, ax=axes[0], shrink=0.8)

#     # Right: compression -------------------------------------------------------
#     im1 = axes[1].imshow(comp_grid, origin="lower", aspect="auto")
#     axes[1].set_title("Total Compression")
#     axes[1].set_xlabel("Token Compression")
#     axes[1].set_ylabel("Batch Compression")
#     axes[1].set_xticks(range(len(comp_grid.columns)))
#     axes[1].set_xticklabels(comp_grid.columns)
#     axes[1].set_yticks(range(len(comp_grid.index)))
#     axes[1].set_yticklabels(comp_grid.index)
#     fig.colorbar(im1, ax=axes[1], shrink=0.8)

#     fig.suptitle("Compression Parameters Tuning on Food-101 on 10 epochs", fontsize=16)


#     # Get axis positions corresponding to token/batch values
#     token_vals = np.array(acc_grid.columns.values, dtype=float)
#     batch_vals = np.array(acc_grid.index.values, dtype=float)

#     # Compute line: b = sqrt(t)
#     token_vals_dense = np.linspace(token_vals.min(), token_vals.max(), 500)
#     batch_line_dense = token_vals_dense ** (1/4)

#     # Filter out-of-bound points
#     valid = (batch_line_dense >= batch_vals.min()) & (batch_line_dense <= batch_vals.max())
#     token_vals_dense = token_vals_dense[valid]
#     batch_line_dense = batch_line_dense[valid]

#     # Interpolate: map real values to fractional indices
#     x = np.interp(token_vals_dense, token_vals, np.arange(len(token_vals)))
#     y = np.interp(batch_line_dense, batch_vals, np.arange(len(batch_vals)))

#     # Plot on both heatmaps
 
#     axes[0].plot(x, y, color="red", linestyle="--", linewidth=2, label=r"euristic: $b = \sqrt[4]{t}$")
#     axes[0].legend()

    

#     plt.savefig(output_file, dpi=300)
#     print(f"[info] Figure saved to {output_file}")


# # ──────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ──────────────────────────────────────────────────────────────────────────────


# def main() -> None:
#     df = gather_dataframe("/home/federico/Desktop/Split_Learning/results/proposal_parameter_tuning/food-101/deit_tiny_patch16_224.fb_in1k/proposal/communication=clean/")
#     make_heatmaps(df, "plots/proposal_parameter_tuning/heatmaps")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3

import os
import json
import matplotlib.pyplot as plt

# Define methods and their folder names
methods = ["proposal", "quantization", "Random_Top_K", "base"]

def load_results(root_dir):
    results = {m: {"compression": [], "max_val_accuracy": [], "val_accuracies": [], "communication": []} for m in methods}
    for method in methods:
        method_path = os.path.join(root_dir, method, "communication=clean")
        if not os.path.exists(method_path):
            print(f"[WARNING] Skipping missing method folder: {method_path}")
            continue

        for subdir in os.listdir(method_path):
            if not subdir.startswith("params="):
                continue

            json_path = os.path.join(method_path, subdir, "training_results.json")
            if not os.path.exists(json_path):
                print(f"[WARNING] Missing file: {json_path}")
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                compression = data.get("Compression", None)
                val_accuracies = data.get("Val accuracies", [])
                communication_cost = data.get("Communication cost", [])

                if compression is None or not val_accuracies:
                    print(f"[WARNING] Missing data in {json_path}")
                    continue

                max_val_acc = max(val_accuracies)
                results[method]["communication"].append(communication_cost)
                results[method]["compression"].append(compression)
                results[method]["max_val_accuracy"].append(max_val_acc)
                results[method]["val_accuracies"].append(val_accuracies)
            except Exception as e:
                print(f"[ERROR] Failed to process {json_path}: {e}")

    return results

def plot_results(results, output_path):
    plt.figure(figsize=(10, 6))
    for method, data in results.items():


        if method == "base":
            # Plot a horizontal line at the base accuracy
            base_acc = data["max_val_accuracy"][0]
            plt.hlines(base_acc, xmin=0, xmax=0.65, color ="red", linestyles="--", label="base (no compression)")
        else:
            sorted_data = sorted(zip(data["compression"], data["max_val_accuracy"]))
            x_sorted, y_sorted = zip(*sorted_data)
            plt.plot(x_sorted, y_sorted, marker="o", label=method, linewidth=1.5)

    
    plt.xlim(0, 0.5)
    plt.xlabel("Compression ")
    plt.ylabel("Max Validation Accuracy")
    plt.title("Compression vs Accuracy on Food 101")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + "accuracy_vs_compression")
    print(f"[INFO] Plot saved to: {output_path}")


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_communication(results, output_path):
    """
    Plot communication vs. accuracy for each method.
    
    Args:
        results (dict): Dictionary containing method results.
        output_path (str): Directory to save the plots.
        compressions_to_plot (dict or None): Dict mapping methods to a list of compressions.
                                             If a method is not listed or list is empty/None, plot all.
    """

    compressions_to_plot = {
    "proposal": [0.01, 0.05, 0.1, 0.2],
    "quantization": [0.03125, 0.09375, 0.1875, 0.28125],  # <- will plot all for quantization
    "Random_Top_K": [0.013750000000000002,0.06875, 0.1375, 0.275, 0.4125],  # <- will plot all for Random_Top_K
}



    methods = [m for m in results if m != "base"]
    num_methods = len(methods)

    # Get global min/max compression for consistent colormap
    all_compressions = [
        c for m in methods 
        for c in (compressions_to_plot.get(m) or results[m]["compression"])
    ]
    global_min = min(all_compressions)
    global_max = max(all_compressions)

    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
    cmap = cm.get_cmap("viridis")

    fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 6), sharey=True)

    if num_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        data = results[method]

        allowed = compressions_to_plot.get(method, None)
        if not allowed:
            allowed = None

        if allowed is None:
            filtered = list(zip(data["compression"], data["communication"], data["val_accuracies"]))
        else:
            allowed_set = set(allowed)
            filtered = [
                (c, comm, acc)
                for c, comm, acc in zip(data["compression"], data["communication"], data["val_accuracies"])
                if c in allowed_set
            ]

        if not filtered:
            print(f"[WARNING] No matching compressions for method '{method}'")
            continue

        compressions, communications, accuracies = zip(*filtered)
        sorted_indices = np.argsort(compressions)

        for idx in sorted_indices:
            compression = compressions[idx]
            communication = communications[idx]
            val_accuracy = accuracies[idx]
            color = cmap(norm(compression))
            ax.plot(communication, val_accuracy, marker="o", label=f"{compression:.3f}", color=color, linewidth=1.5, alpha=0.7)

        # Plot base
        base_comm = results["base"]["communication"][0]
        base_acc = results["base"]["val_accuracies"][0]
        ax.plot(base_comm, base_acc, marker="x", label="1 (base)", color="red", linewidth=2)

        ax.set_xlabel("Communication")
        ax.set_title(f"{method}")
        ax.grid(True)
        ax.legend(title="Compression", fontsize="small")

    axes[0].set_ylabel("Validation Accuracy")

    # Add horizontal colorbar (below all plots)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])  # [left, bottom, width, height]
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label="Compression")

    plt.tight_layout(rect=[0, 0.15, 1, 1])  # leave space at bottom for colorbar
    save_path = f"{output_path}/accuracy_vs_communication_all_methods.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Combined plot saved to: {save_path}")



def main():
    results = load_results("/home/federico/Desktop/Split_Learning/results/baselines/food-101/deit_tiny_patch16_224.fb_in1k")
    plot_results(results, "plots/baselines/")
    plot_communication(results, "plots/baselines/")


if __name__ == "__main__":
    main()
