# !/usr/bin/env python3

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors




def load_results(root_dir):

    methods = ["proposal", "quantization", "Random_Top_K", "base"]

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
    dataset= output_path.split("/")[-2]
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
    plt.title("Compression vs Accuracy on " + dataset)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + "/" + "accuracy_vs_compression")
    print(f"[INFO] Plot saved to: {output_path}")


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
    "quantization": [0.03125, 0.09375, 0.1875, 0.28125],  
    "Random_Top_K": [0.013593750000000002, 0.013750000000000002,0.06796875, 0.06875, 0.1359375, 0.1375, 0.271875, 0.275, 0.4078125, 0.4125],  
}

    dataset= output_path.split("/")[-2]

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
    fig.suptitle("Accuracy vs Communication on " + dataset)

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
    save_path = output_path + "/" + "accuracy_vs_communication_all_methods.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Combined plot saved to: {save_path}")



def main():

    results_baselines_path = "/home/federico/Desktop/Split_Learning/results/baselines"
    baselines_plot_path = "plots/baselines"

    datasets = ["cifar100", "food-101"]
    models = ["deit_tiny_patch16_224.fb_in1k"]

    for model in models:
        for dataset in datasets:
            results_path = results_baselines_path + "/" + dataset + "/" + model
            plot_path = baselines_plot_path + "/" + dataset + "/" + model
            os.makedirs(plot_path, exist_ok=True)


            # Load results 
            results = load_results(results_path)

            plot_results(results, plot_path)
            plot_communication(results, plot_path)


if __name__ == "__main__":
    main()

