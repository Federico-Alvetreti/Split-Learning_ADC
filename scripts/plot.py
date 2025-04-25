import os
import json
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_experiments(snr, x_metric, y_metric, group_by=None,
                     base_dir='/home/federico/Desktop/Split_Learning/results/baselines/flowers-102/deit_tiny_patch16_224.fb_in1k/ours'):
    """
    Plot (and save) experiments for a given SNR, using interpolation for grouping by communication cost,
    and place a vertically-stacked legend below the plot.

    Args:
        snr (int or str): SNR value, e.g. 0, -10, 10.
        x_metric (str): 'epoch' or 'communication_cost'
        y_metric (str): 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'communication_cost'
        group_by (str or None): parameter name to group by; if None, plot each experiment separately.
        base_dir (str): path to the 'ours' directory containing 'snr=...' folders.
    """
    # map y_metric → JSON key
    metric_map = {
        'train_loss': 'Train losses',
        'train_accuracy': 'Train accuracies',
        'val_loss': 'Val losses',
        'val_accuracy': 'Val accuracies',
        'communication_cost': 'Communication cost'
    }
    y_key = metric_map[y_metric]

    snr_dir = os.path.join(base_dir, f'snr={snr}')
    if not os.path.isdir(snr_dir):
        raise ValueError(f"SNR directory not found: {snr_dir}")

    # Load all experiments
    experiments = []
    for entry in os.listdir(snr_dir):
        if not entry.startswith('params='): continue
        params = ast.literal_eval(entry[len('params='):])
        path = os.path.join(snr_dir, entry, 'training_results.json')
        if not os.path.isfile(path): continue
        with open(path) as f:
            data = json.load(f)
        y_vals = data[y_key]
        x_vals = (list(range(1, len(y_vals)+1)) if x_metric=='epoch'
                  else data['Communication cost'])
        experiments.append({'params': params, 'x': np.array(x_vals), 'y': np.array(y_vals)})

    if not experiments:
        raise ValueError(f"No experiments for SNR={snr}")

    # Prepare plotting data
    plot_data = []
    if group_by:
        # Group experiments by parameter
        groups = {}
        for e in experiments:
            key = e['params'].get(group_by)
            if key is None:
                raise ValueError(f"Parameter '{group_by}' missing in some runs.")
            groups.setdefault(key, []).append(e)

        for key, runs in groups.items():
            if x_metric == 'communication_cost':
                # Interpolate onto a common grid
                all_x = np.concatenate([r['x'] for r in runs])
                grid_x = np.linspace(all_x.min(), all_x.max(), num=200)
                interp_ys = [np.interp(grid_x, r['x'], r['y'], left=np.nan, right=np.nan)
                             for r in runs]
                y_avg = np.nanmean(np.vstack(interp_ys), axis=0)
                plot_data.append((f"{group_by}={key}", grid_x, y_avg))
            else:
                # Align by shortest epoch length
                min_len = min(len(r['y']) for r in runs)
                xs = np.vstack([r['x'][:min_len] for r in runs])
                ys = np.vstack([r['y'][:min_len] for r in runs])
                plot_data.append((f"{group_by}={key}", xs.mean(0), ys.mean(0)))
    else:
        # No grouping: each experiment is its own line
        plot_data = [(str(e['params']), e['x'], e['y']) for e in experiments]

    # Plot
    plt.figure(figsize=(8,5))
    for label, xs, ys in plot_data:
        plt.plot(xs, ys, label=label)
    plt.xlabel('Epoch' if x_metric=='epoch' else 'Communication Cost')
    y_labels = {
        'train_loss': 'Train Loss',
        'train_accuracy': 'Train Accuracy',
        'val_loss': 'Validation Loss',
        'val_accuracy': 'Validation Accuracy',
        'communication_cost': 'Communication Cost'
    }
    plt.ylabel(y_labels[y_metric])
    plt.title(f"SNR {snr}: {y_labels[y_metric]} vs {'Epochs' if x_metric=='epoch' else 'Comm Cost'}")

    # Place vertically-stacked legend below the plot
    plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.2),
               ncol=1,
               fontsize='small')
    plt.grid(True)

    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.30)

    # Save
    output_dir = "/home/federico/Desktop/Split_Learning/plots/ours"
    save_folder = os.path.join(output_dir, f"snr_{snr}")
    os.makedirs(save_folder, exist_ok=True)
    fname = f"{snr}_{x_metric}_vs_{y_metric}" + (f"_by_{group_by}" if group_by else "") + ".png"
    save_path = os.path.join(save_folder, fname)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot as: {save_path}")

    # Show
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot training curves with interpolation and vertically-stacked legend")
    parser.add_argument('--snr',       required=True, help="e.g. 0, -10, 10")
    parser.add_argument('--x_metric',  required=True,
                        choices=['epoch','communication_cost'],
                        help="x-axis metric")
    parser.add_argument('--y_metric',  required=True,
                        choices=['train_loss','train_accuracy',
                                 'val_loss','val_accuracy','communication_cost'],
                        help="y-axis metric")
    parser.add_argument('--group_by',  help="parameter name to average over")
    args = parser.parse_args()

    plot_experiments(
        snr=args.snr,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        group_by=args.group_by
    )