import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def collect_results_grouped_by_method(results_root):
    methods = defaultdict(lambda: defaultdict(dict))
    for snr_folder in os.listdir(results_root):
        if not snr_folder.startswith("snr="):
            continue
        snr_value = snr_folder.split("=")[1]
        snr_path = os.path.join(results_root, snr_folder)
        for method in os.listdir(snr_path):
            method_path = os.path.join(snr_path, method)
            if not os.path.isdir(method_path):
                continue
            for param_folder in os.listdir(method_path):
                if not param_folder.startswith("params="):
                    continue
                param_value = param_folder.split("=")[1]
                result_path = os.path.join(method_path, param_folder, "training_results.json")
                if os.path.exists(result_path):
                    with open(result_path, "r") as f:
                        data = json.load(f)
                    methods[method][snr_value][param_value] = data
    return methods

def plot_metric_evolution(metrics_dict, metric_key, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    for param, data in metrics_dict.items():
        if metric_key in data:
            plt.plot(data[metric_key], label=f"params={param}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_final_comparison(results, metric_key, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    for param in sorted({p for snr in results for p in results[snr]}):
        snrs = []
        values = []
        for snr in sorted(results.keys(), key=lambda x: float(x)):
            if param in results[snr] and metric_key in results[snr][param]:
                snrs.append(snr)
                values.append(results[snr][param][metric_key][-1])
        if snrs:
            plt.plot(snrs, values, marker='o', label=f"params={param}")
    plt.title(title)
    plt.xlabel("SNR")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_method(method_name, method_results, output_root):
    save_base = os.path.join(output_root, method_name)
    os.makedirs(save_base, exist_ok=True)

    all_params = {p for snr in method_results for p in method_results[snr]}

    if len(all_params) > 1:
        for snr, param_dict in method_results.items():
            save_dir = os.path.join(save_base, "different_snr", f"snr={snr}")
            os.makedirs(save_dir, exist_ok=True)

            plot_metric_evolution(param_dict, "Train accuracies",
                                  f"{method_name} - Train Accuracy @ SNR={snr}", "Accuracy",
                                  os.path.join(save_dir, "train_acc.png"))

            plot_metric_evolution(param_dict, "Val accuracies",
                                  f"{method_name} - Val Accuracy @ SNR={snr}", "Accuracy",
                                  os.path.join(save_dir, "val_acc.png"))

            plot_metric_evolution(param_dict, "Train losses",
                                  f"{method_name} - Train Loss @ SNR={snr}", "Loss",
                                  os.path.join(save_dir, "train_loss.png"))

            plot_metric_evolution(param_dict, "Val losses",
                                  f"{method_name} - Val Loss @ SNR={snr}", "Loss",
                                  os.path.join(save_dir, "val_loss.png"))

            plot_metric_evolution(param_dict, "Communication cost",
                                  f"{method_name} - Communication Cost @ SNR={snr}", "Cumulative Cost",
                                  os.path.join(save_dir, "comm_cost.png"))

    # Always do comparison across SNRs
    save_cmp = os.path.join(save_base, "comparison")
    os.makedirs(save_cmp, exist_ok=True)

    plot_final_comparison(method_results, "Train accuracies",
                          f"{method_name} - Final Train Accuracy", "Accuracy",
                          os.path.join(save_cmp, "train_acc.png"))

    plot_final_comparison(method_results, "Val accuracies",
                          f"{method_name} - Final Val Accuracy", "Accuracy",
                          os.path.join(save_cmp, "val_acc.png"))

    plot_final_comparison(method_results, "Train losses",
                          f"{method_name} - Final Train Loss", "Loss",
                          os.path.join(save_cmp, "train_loss.png"))

    plot_final_comparison(method_results, "Val losses",
                          f"{method_name} - Final Val Loss", "Loss",
                          os.path.join(save_cmp, "val_loss.png"))

    plot_final_comparison(method_results, "Communication cost",
                          f"{method_name} - Final Comm. Cost", "Cumulative Cost",
                          os.path.join(save_cmp, "comm_cost.png"))

def main():
    results_root = "/home/federico/Desktop/Split_Learning/results/Static/flowers-102/deit_tiny_patch16_224.fb_in1k"
    output_root = "plots/Static"

    print("📂 Parsing results...")
    all_methods = collect_results_grouped_by_method(results_root)

    for method_name, method_results in all_methods.items():
        print(f"📈 Plotting for method: {method_name}")
        plot_method(method_name, method_results, output_root)

if __name__ == "__main__":
    main()
