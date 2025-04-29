import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def load_last_value(json_path, measure):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data[measure][-1] if measure in data else None
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def collect_results(base_path, dataset="flowers-102", backbone="deit_tiny_patch16_224.fb_in1k", measure="Val losses"):
    results = defaultdict(lambda: defaultdict(dict))  # method -> snr -> k -> value
    
    root = os.path.join(base_path, dataset, backbone)
    for method in os.listdir(root):
        method_path = os.path.join(root, method)
        if not os.path.isdir(method_path):
            continue
        for snr_dir in os.listdir(method_path):
            if not snr_dir.startswith("snr="):
                continue
            snr = float(snr_dir.split("=")[1])
            snr_path = os.path.join(method_path, snr_dir)
            for k_dir in os.listdir(snr_path):
                if not k_dir.startswith("k="):
                    continue
                k = float(k_dir.split("=")[1])
                param_folder = "params=None" if method == "classic_split_learning" else "params={'K': 9}"
                json_path = os.path.join(snr_path, k_dir, param_folder, "training_results.json")
                value = load_last_value(json_path, measure)
                if value is not None:
                    results[method][snr][k] = value
    return results

def plot_results(results, measure):
    os.makedirs("plots", exist_ok=True)

    for method, snr_dict in results.items():
        plt.figure(figsize=(8, 6))
        for snr, k_dict in sorted(snr_dict.items()):
            ks = sorted(k_dict.keys())
            vals = [k_dict[k] for k in ks]
            plt.plot(ks, vals, marker='o', label=f"SNR={snr}")
        plt.title(f"{method} - {measure} (Last Epoch)")
        plt.xlabel("Compression Ratio (k/n)")
        plt.ylabel(measure)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{method.replace('/', '_')}_{measure.replace(' ', '_')}_curve.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", type=str, help="Name of the measure to plot, e.g., 'Val losses'")
    parser.add_argument("--base_path", type=str, default="results/baselines", help="Base results folder")
    args = parser.parse_args()

    results = collect_results(args.base_path, measure=args.measure)
    plot_results(results, args.measure)