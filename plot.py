import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def plot_metrics_vs_epochs(results_root = "results", plots_root = "plots/baselines/metrics_vs_epochs",
                            datasets = ["food-101", "flowers-102"],
                              models = ["deit_tiny_patch16_224.fb_in1k","deit_small_patch16_224.fb_in1k"],
                                methods = ["only_forward_split_learning_baseline", "simple_split_learning_baseline", "send_raw_data_baseline"]): 
    
    for model in models:
        for dataset in datasets:
            for method in methods:

                # Make results and plots paths 
                results_path = results_root + "/" + dataset + "/" + model + "/" + method
                plots_path = plots_root + "/" + dataset + "/" + model + "/" + method
                
                os.makedirs(plots_path, exist_ok=True)
                snr_folders = [str(i) for i in range(-10, 11)]  
                train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
                
                for snr in snr_folders:
                    result_file = os.path.join(results_path, "snr=" + snr, "training_results.json")
                    if os.path.exists(result_file):
                        with open(result_file, "r") as f:
                            data = json.load(f)
                        train_losses.append(data["Train losses"])
                        train_accuracies.append(data["Train accuracies"])
                        val_losses.append(data["Val losses"])
                        val_accuracies.append(data["Val accuracies"])
                
                snr_values = [float(snr) for snr in snr_folders]
                norm = mcolors.Normalize(vmin=min(snr_values), vmax=max(snr_values))
                cmap = plt.cm.viridis  
                
                def plot_metric(metric_data, ylabel, title, filename):
                    plt.figure(figsize=(10, 6))
                    for i, metric in enumerate(metric_data):
                        plt.plot(range(1, len(metric) + 1), metric, marker='o', color=cmap(norm(snr_values[i])))
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    plt.colorbar(sm, ax=plt.gca(), label="SNR Value")
                    plt.xlabel("Epoch")
                    plt.ylabel(ylabel)
                    plt.title(title)
                    plt.grid(True)
                    plt.tight_layout()

                    if "Loss" in ylabel:
                        plt.ylim(0,5)
                    elif "Accuracy" in ylabel:
                        plt.ylim(0,1)
                    plt.savefig(os.path.join(plots_path, filename))
                    plt.close()
                
                plot_metric(train_accuracies, "Training Accuracy", "Training Accuracy vs Epochs for Different SNRs", "train_accuracy_vs_epochs.png")
                plot_metric(train_losses, "Training Loss", "Training Loss vs Epochs for Different SNRs", "train_loss_vs_epochs.png")
                plot_metric(val_accuracies, "Validation Accuracy", "Validation Accuracy vs Epochs for Different SNRs", "val_accuracy_vs_epochs.png")
                plot_metric(val_losses, "Validation Loss", "Validation Loss vs Epochs for Different SNRs", "val_loss_vs_epochs.png")
    
    print("Epoch-based plots saved in the 'plots' folder.")


def plot_metrics_vs_snr(results_path = "results", plots_path = "plots/baselines", dataset = "food-101", model = "deit_small_patch16_224.fb_in1k"):
    
    # Make results and plots paths 
    results_path += "/" + dataset + "/" + model
    plots_path += "/" + dataset + "/" + model

    os.makedirs(plots_path, exist_ok=True)
    experiments = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]

    metrics = {}
    
    for exp in experiments:
        metrics[exp] = {"snr_values": [], "train_losses": [], "train_accuracies": [], "val_losses": [], "val_accuracies": [], "comm_costs": []}
        
        for snr in range(-10, 11):
            json_file = os.path.join(results_path, exp, "snr=" + str(snr), "training_results.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                metrics[exp]["snr_values"].append(snr)
                metrics[exp]["train_losses"].append(data["Train losses"][-1])
                metrics[exp]["train_accuracies"].append(data["Train accuracies"][-1])
                metrics[exp]["val_losses"].append(data["Val losses"][-1])
                metrics[exp]["val_accuracies"].append(data["Val accuracies"][-1])
                metrics[exp]["comm_costs"] = data["Communication cost"]
        
    def plot_metric(y_key, ylabel, title, filename, colors):
        plt.figure(figsize=(10, 6))
        for i, (exp, data) in enumerate(metrics.items()):
            plt.plot(data["snr_values"], data[y_key], marker='o', label=exp, color=colors[i % len(colors)])
        plt.xlabel("SNR Value")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, filename))
        plt.close()

    def plot_cost(y_key, ylabel, title, filename, colors):
        plt.figure(figsize=(10, 6))
        for i, (exp, data) in enumerate(metrics.items()):
            plt.plot(range(5), data[y_key], marker='o', label=exp, color=colors[i % len(colors)])
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, filename))
        plt.close()
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']
    
    plot_metric("train_accuracies", "Training Accuracy", "Training Accuracy vs SNR", "train_accuracy_vs_snr.png", colors)
    plot_metric("train_losses", "Training Loss", "Training Loss vs SNR", "train_loss_vs_snr.png", colors)
    plot_metric("val_accuracies", "Validation Accuracy", "Validation Accuracy vs SNR", "val_accuracy_vs_snr.png", colors)
    plot_metric("val_losses", "Validation Loss", "Validation Loss vs SNR", "val_loss_vs_snr.png", colors)
    plot_cost("comm_costs", "Communication Cost", "Communication Cost ", "comm_cost.png", colors)
    
    print("SNR-based comparison plots saved in the 'plots/last_epoch' folder.")


# Create comparison plots between experiments 
def comparison_between_experiments_plot(results_root="results", plots_root="plots/baselines_comparison"):
    datasets = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']

    metrics_keys = [
        ("train_losses", "Train losses"),
        ("train_accuracies", "Train accuracies"),
        ("val_losses", "Val losses"),
        ("val_accuracies", "Val accuracies")
    ]

    for metric_key, metric_label in metrics_keys:
        # Determine the number of rows and columns based on datasets and models
        num_rows = len(datasets)
        num_cols = len(os.listdir(os.path.join(results_root, datasets[0])))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 10))
        fig.suptitle(f"Comparison of {metric_label} across Datasets and Models", fontsize=16)

        for i, dataset in enumerate(datasets):
            dataset_path = os.path.join(results_root, dataset)
            models = [m for m in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, m))]

            for j, model in enumerate(models):
                model_path = os.path.join(dataset_path, model)
                methods = [method for method in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, method))]

                metrics = {}

                for method in methods:
                    metrics[method] = {"snr_values": [], metric_key: []}

                    for snr in range(-10, 11):
                        json_file = os.path.join(model_path, method, f"snr={snr}", "training_results.json")
                        if os.path.exists(json_file):
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            metrics[method]["snr_values"].append(snr)
                            metrics[method][metric_key].append(data[metric_label][-1])

                ax = axs[i, j]
                for k, (method, data) in enumerate(metrics.items()):
                    ax.plot(data["snr_values"], data[metric_key], marker='o', label=method, color=colors[k % len(colors)])
                ax.set_title(f"{dataset} - {model}")
                ax.set_xlabel("SNR Value")
                ax.set_ylabel(metric_label)
                ax.grid(True)
                ax.set_ylim(0, 1)
                if "loss" in metric_key:
                    ax.set_ylim(0, 5)  # Adjust as necessary depending on loss scale

                if i == 0 and j == len(models) - 1:
                    ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
        os.makedirs(plots_root, exist_ok=True)
        plt.savefig(os.path.join(plots_root, f"{metric_key}_comparison.png"))
        plt.close()

    print(f"SNR-based comparison plots saved in the '{plots_root}' folder.")

