


import os
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def extract_params_from_folder(folder_name):
    match = re.match(r"params=\{'q': (\d+), 'r': \d+, 'l': (\d+)\}", folder_name)
    if match:
        q = int(match.group(1))
        l = int(match.group(2))
        return q, l
    return None, None


def load_results(results_dir):
    val_acc_dict = {}
    compression_dict = {}

    for folder in os.listdir(results_dir):
        full_path = os.path.join(results_dir, folder)
        if not os.path.isdir(full_path):
            continue

        q, l = extract_params_from_folder(folder)
        if q is None or l is None:
            continue

        json_path = os.path.join(full_path, "training_results.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        val_acc = data["Val accuracies"][-1]
        compression = data["Compression"]

        val_acc_dict[(q, l)] = val_acc
        compression_dict[(q, l)] = compression

    return val_acc_dict, compression_dict


def create_heatmap(data_dict, x_vals, y_vals, title, ax):
    heatmap = np.full((len(y_vals), len(x_vals)), np.nan)

    for i, l in enumerate(y_vals):
        for j, q in enumerate(x_vals):
            if (q, l) in data_dict:
                heatmap[i, j] = data_dict[(q, l)]

    im = ax.imshow(heatmap, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels(x_vals, rotation=45)
    ax.set_yticklabels(y_vals)
    ax.set_xlabel("q")
    ax.set_ylabel("l")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)


def plot_heatmaps(results_dir, output_dir):
    val_acc_dict, compression_dict = load_results(results_dir)

    q_vals = sorted(set(q for q, _ in val_acc_dict.keys()))
    l_vals = sorted(set(l for _, l in val_acc_dict.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Fedlite parameter tuning for 1 epoch in food_101", fontsize=18)
    create_heatmap(val_acc_dict, q_vals, l_vals, "Validation Accuracy", axes[0])
    create_heatmap(compression_dict, q_vals, l_vals, "Compression", axes[1])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "heatmaps.png"))
    plt.close()


if __name__ == "__main__":
    plot_heatmaps("/home/federico/Desktop/Split_Learning/results/fedlite_parameters_tuning/food-101/deit_tiny_patch16_224.fb_in1k/fedlite/communication=clean", "plots/fedlite_parameter_tuning")
