import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import hydra 


@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")



def plot_metrics_vs_epochs(cfg): 

    results_path = cfg.core.results_path #fix this 
    plots_path = cfg.core.plots_path
     
    os.makedirs(plots_path, exist_ok=True)
    snr_folders = [str(i) for i in range(-10, 11)]  
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    
    for snr in snr_folders:
        result_file = os.path.join(results_path, snr, "training_results.json")
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
            plt.plot(range(1, 21), metric, marker='o', color=cmap(norm(snr_values[i])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label="SNR Value")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, filename))
        plt.close()
    
    plot_metric(train_accuracies, "Training Accuracy", "Training Accuracy vs Epochs for Different SNRs", "train_accuracy_vs_epochs.png")
    plot_metric(train_losses, "Training Loss", "Training Loss vs Epochs for Different SNRs", "train_loss_vs_epochs.png")
    plot_metric(val_accuracies, "Validation Accuracy", "Validation Accuracy vs Epochs for Different SNRs", "val_accuracy_vs_epochs.png")
    plot_metric(val_losses, "Validation Loss", "Validation Loss vs Epochs for Different SNRs", "val_loss_vs_epochs.png")
    
    print("Epoch-based plots saved in the 'plots' folder.")


def plot_metrics_vs_snr(results_path="results", plots_path="plots/baselines"):  
    os.makedirs(plots_path, exist_ok=True)
    
    experiments = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]
    
    metrics = {}
    
    for exp in experiments:
        metrics[exp] = {"snr_values": [], "train_losses": [], "train_accuracies": [], "val_losses": [], "val_accuracies": [], "comm_costs": []}
        
        for snr in range(-10, 11):
            json_file = os.path.join(results_path, exp, str(snr), "training_results.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                metrics[exp]["snr_values"].append(snr)
                metrics[exp]["train_losses"].append(data["Train losses"][-1])
                metrics[exp]["train_accuracies"].append(data["Train accuracies"][-1])
                metrics[exp]["val_losses"].append(data["Val losses"][-1])
                metrics[exp]["val_accuracies"].append(data["Val accuracies"][-1])
                metrics[exp]["comm_costs"].append(data["Communication cost"])
    
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
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']
    
    plot_metric("train_accuracies", "Training Accuracy", "Training Accuracy vs SNR", "train_accuracy_vs_snr.png", colors)
    plot_metric("train_losses", "Training Loss", "Training Loss vs SNR", "train_loss_vs_snr.png", colors)
    plot_metric("val_accuracies", "Validation Accuracy", "Validation Accuracy vs SNR", "val_accuracy_vs_snr.png", colors)
    plot_metric("val_losses", "Validation Loss", "Validation Loss vs SNR", "val_loss_vs_snr.png", colors)
    plot_metric("comm_costs", "Communication Cost", "Communication Cost vs SNR", "comm_cost_vs_snr.png", colors)
    
    print("SNR-based comparison plots saved in the 'plots/last_epoch' folder.")

