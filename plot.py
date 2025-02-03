import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_metrics_vs_epochs(results_path="results", plots_path="plots"):  
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

def plot_metrics_vs_snr(results_path="results", plots_path="plots/last_epoch"):  
    os.makedirs(plots_path, exist_ok=True)
    snr_values, train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], [], []
    
    for snr in range(-10, 11):
        json_file = os.path.join(results_path, str(snr), "training_results.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            snr_values.append(snr)
            train_losses.append(data["Train losses"][-1])
            train_accuracies.append(data["Train accuracies"][-1])
            val_losses.append(data["Val losses"][-1])
            val_accuracies.append(data["Val accuracies"][-1])
    
    def plot_metric(x, y, ylabel, title, filename, color):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', label=title, color=color)
        plt.xlabel("SNR Value")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, filename))
        plt.close()
    
    plot_metric(snr_values, train_accuracies, "Training Accuracy", "Training Accuracy vs SNR", "train_accuracy_vs_snr.png", 'b')
    plot_metric(snr_values, train_losses, "Training Loss", "Training Loss vs SNR", "train_loss_vs_snr.png", 'r')
    plot_metric(snr_values, val_accuracies, "Validation Accuracy", "Validation Accuracy vs SNR", "val_accuracy_vs_snr.png", 'g')
    plot_metric(snr_values, val_losses, "Validation Loss", "Validation Loss vs SNR", "val_loss_vs_snr.png", 'm')
    
    print("SNR-based plots saved in the 'plots/last_epoch' folder.")

# Example usage
# plot_metrics_vs_epochs()
# plot_metrics_vs_snr()
