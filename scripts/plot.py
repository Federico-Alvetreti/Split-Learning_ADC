# #!/usr/bin/env python3
# import os
# import json
# import argparse
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns  # newly added for heatmaps

# def load_all_results(results_dir, comm_threshold=None):
#     """
#     Traverse 'classic_split_learning', 'ours', 'JPEG' under results_dir.
#     Extract full validation series and sample at the minimum communication cost
#     per compression (excluding 'JPEG' in the first pass).
#     Returns records with keys: method, snr, compression,
#     series_val_loss, series_val_accuracy, val_loss, val_accuracy.
#     """
#     methods = ['classic_split_learning', 'ours']
#     comm_mins_per_comp = defaultdict(list)

#     # First pass to find per-compression communication cost minimum (excluding JPEG)
#     for method in methods:
#         method_dir = os.path.join(results_dir, method)
#         if not os.path.isdir(method_dir):
#             continue
#         for snr_folder in os.listdir(method_dir):
#             if not snr_folder.startswith('snr='):
#                 continue
#             for comp_folder in os.listdir(os.path.join(method_dir, snr_folder)):
#                 if not comp_folder.startswith('compression='):
#                     continue
#                 try:
#                     compression = float(comp_folder.split('=')[1])
#                 except ValueError:
#                     continue
#                 comp_path = os.path.join(method_dir, snr_folder, comp_folder)
#                 for params_folder in os.listdir(comp_path):
#                     subdir = os.path.join(comp_path, params_folder)
#                     tr_json = os.path.join(subdir, 'training_results.json')
#                     if not os.path.isfile(tr_json):
#                         continue
#                     with open(tr_json, 'r') as f:
#                         data = json.load(f)
#                     comm_series = data.get("Communication cost", [])
#                     if comm_series:
#                         comm_mins_per_comp[compression].append(comm_series[-1])

#     # Compute minimum communication per compression
#     comp_thresholds = {
#         comp: min(vals) for comp, vals in comm_mins_per_comp.items()
#     }

#     # Second pass to extract records (this time including JPEG)
#     records = []
#     for method in ['classic_split_learning', 'JPEG', 'ours']:
#         method_dir = os.path.join(results_dir, method)
#         if not os.path.isdir(method_dir):
#             continue
#         for snr_folder in os.listdir(method_dir):
#             if not snr_folder.startswith('snr='):
#                 continue
#             try:
#                 snr = float(snr_folder.split('=')[1])
#             except ValueError:
#                 continue
#             for comp_folder in os.listdir(os.path.join(method_dir, snr_folder)):
#                 if not comp_folder.startswith('compression='):
#                     continue
#                 try:
#                     compression = float(comp_folder.split('=')[1])
#                 except ValueError:
#                     continue
#                 comp_path = os.path.join(method_dir, snr_folder, comp_folder)
#                 for params_folder in os.listdir(comp_path):
#                     subdir = os.path.join(comp_path, params_folder)
#                     tr_json = os.path.join(subdir, 'training_results.json')
#                     if not os.path.isfile(tr_json):
#                         continue
#                     with open(tr_json, 'r') as f:
#                         data = json.load(f)
#                     val_losses  = data.get('Val losses', [])
#                     val_accs    = data.get('Val accuracies', [])
#                     comm_series = data.get("Communication cost", [])
#                     if not val_losses or not val_accs:
#                         continue

#                     # Pick index based on per-compression threshold
#                     threshold = comm_threshold if comm_threshold is not None else comp_thresholds.get(compression, None)
#                     if threshold is not None and comm_series and len(comm_series) == len(val_losses):
#                         idx = next((i for i, c in enumerate(comm_series) if c >= threshold), len(val_losses)-1)
#                     else:
#                         idx = len(val_losses) - 1

#                     records.append({
#                         'method': method,
#                         'snr': snr,
#                         'compression': compression,
#                         'series_val_loss': val_losses[:idx+1],
#                         'series_val_accuracy': val_accs[:idx+1],
#                         'val_loss': val_losses[idx],
#                         'val_accuracy': val_accs[idx]
#                     })
#     return records


# def plot_metric_vs_compression(records, metric, output_dir=None):
#     """Plot compression vs metric for each SNR and method in fixed order."""
#     snrs = sorted({r['snr'] for r in records})
#     method_order = ['ours', 'classic_split_learning', 'JPEG']
#     fig, axes = plt.subplots(1, len(snrs), figsize=(5 * len(snrs), 4), squeeze=False)

#     for j, snr in enumerate(snrs):
#         ax = axes[0][j]
#         snr_records = [r for r in records if r['snr'] == snr]
#         for method in method_order:
#             method_records = sorted(
#                 [r for r in snr_records if r['method'] == method],
#                 key=lambda r: r['compression']
#             )
#             if not method_records:
#                 continue
#             compressions = [r['compression'] for r in method_records]
#             metrics      = [r[metric]        for r in method_records]
#             ax.plot(compressions, metrics, marker='o', label=method)

#         ax.set_title(f'SNR={snr}')
#         ax.set_xlabel('Compression rate')
#         if metric == 'val_accuracy':
#             ax.set_ylim(0, 1)
#             ax.set_ylabel('Val Accuracy')
#         else:
#             ax.set_ylabel('Val Loss')
#         ax.grid(True)
#         ax.set_xscale('log')
#         ax.legend()

#     plt.tight_layout()
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, f'{metric}_vs_compression.png'))
#     else:
#         plt.show()


# def plot_epoch_curves_grid(records, metric, output_dir=None):
#     """
#     Grid of per-epoch curves: rows=SNR, cols=compression.
#     Each subplot overlays methods: classic_split_learning (baseline), JPEG, and ours.
#     """
#     snrs = sorted({r['snr'] for r in records})
#     ks   = sorted({r['compression'] for r in records})
#     n_rows, n_cols = len(snrs), len(ks)
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharey=True)
#     fig.subplots_adjust(top=0.88, left=0.12, right=0.95, hspace=0.4, wspace=0.3)

#     # column headers
#     for j, k in enumerate(ks):
#         ax = axes[0][j]
#         x0, x1 = ax.get_position().x0, ax.get_position().x1
#         fig.text((x0 + x1)/2, 0.93, f'k/N = {k}', ha='center', va='bottom', fontsize='large')

#     # row headers
#     for i, snr in enumerate(snrs):
#         ax = axes[i][0]
#         y0, y1 = ax.get_position().y0, ax.get_position().y1
#         fig.text(0.07, (y0 + y1)/2, f'SNR = {snr}', ha='left', va='center', rotation='vertical', fontsize='large')

#     for i, snr in enumerate(snrs):
#         for j, k in enumerate(ks):
#             ax = axes[i][j]
#             for method, label in [
#                 ('ours', 'Ours'),
#                 ('classic_split_learning', 'Classic SL'),
#                 ('JPEG', 'JPEG')
#             ]:
#                 rec = next((r for r in records
#                             if r['method']==method and r['snr']==snr and r['compression']==k),
#                            None)
#                 if not rec:
#                     continue
#                 series = rec['series_val_accuracy'] if metric == 'val_accuracy' else rec['series_val_loss']
#                 line, = ax.plot(series, label=label)

#                 # extend baseline dotted to match our length
#                 if method in ('classic_split_learning', 'JPEG'):
#                     ours_rec = next((rr for rr in records
#                                      if rr['method']=='ours' and rr['snr']==snr and rr['compression']==k),
#                                     None)
#                     if ours_rec:
#                         ours_len = len(ours_rec['series_val_accuracy'] if metric=='val_accuracy' else ours_rec['series_val_loss'])
#                         base_len = len(series)
#                         if base_len < ours_len:
#                             last = series[-1]
#                             ax.hlines(
#                                 last,
#                                 base_len - 1,
#                                 ours_len - 1,
#                                 linestyles='dotted',
#                                 label='_nolegend_',
#                                 color=line.get_color()
#                             )

#             ax.set_title(f'SNR={snr}, k/N={k}')
#             if i == n_rows - 1:
#                 ax.set_xlabel('Epoch')
#             if j == 0:
#                 ax.set_ylabel('Val Accuracy' if metric=='val_accuracy' else 'Val Loss')
#             ax.grid(True)
#             ax.legend()

#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         fname = f'epoch_grid_{metric}.png'
#         fig.savefig(os.path.join(output_dir, fname))
#     else:
#         plt.show()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap

# def plot_delta_heatmap_continuous(records,
#                                   method_a='ours',
#                                   method_b='classic_split_learning',
#                                   output_dir=None,
#                                   cmap='RdYlGn'):
#     """
#     Continuous diverging heatmap of method_a - method_b.
#     Each cell shows:
#       <method_b val_accuracy>
#       <+/- delta = ours - classic>
#     """
#     # collect unique grid axes
#     snr_vals = sorted({r['snr'] for r in records})
#     kn_vals  = sorted({r['compression'] for r in records})
#     n, m = len(snr_vals), len(kn_vals)

#     # prepare matrices
#     mat_delta = np.zeros((n, m), dtype=float)
#     mat_base  = np.zeros((n, m), dtype=float)

#     # fill them
#     for i, snr in enumerate(snr_vals):
#         for j, kn in enumerate(kn_vals):
#             ra = next((r['val_accuracy'] for r in records
#                        if r['snr']==snr and r['compression']==kn and r['method']==method_a), 0.0)
#             rb = next((r['val_accuracy'] for r in records
#                        if r['snr']==snr and r['compression']==kn and r['method']==method_b), 0.0)
#             mat_delta[i, j] = ra - rb
#             mat_base[i, j]  = rb

#      # symmetric color limits
#     max_abs = np.max(np.abs(mat_delta))

#     fig, ax = plt.subplots(figsize=(6,5))
#     # draw the heatmap WITHOUT annotations
#     sns.heatmap(
#         mat_delta,
#         annot=False,       # ← turn off the built-in annot
#         cmap=cmap,
#         center=0,
#         vmin=-max_abs,
#         vmax=+max_abs,
#         linewidths=0.8,
#         linecolor='white',
#         xticklabels=[f"{k:.0e}" for k in kn_vals],
#         yticklabels=snr_vals,
#         cbar_kws={'label': "Gained Accuracy"}
#     )

#     # now manually overlay two texts per cell
#     n, m = mat_delta.shape
#     for i in range(n):
#         for j in range(m):
#             base = mat_base[i, j]
#             delta = mat_delta[i, j]

#             if delta > 0:
#                 delta_plot = f"+{abs(delta):.2f}"
#             else:
#                 delta_plot = f"-{abs(delta):.2f}"
#             # small baseline at lower half
#             ax.text(
#                 j + 0.5, i + 0.7,
#                 f"({base:.2f})",
#                 ha='center', va='center',
#                 fontsize=8,
#                 color='black'
#             )
#             # larger delta at upper half
#             ax.text(
#                 j + 0.5, i + 0.5,
#                 delta_plot,
#                 ha='center', va='center',
#                 fontsize=10,
#                 fontweight='medium',
#                 color='black'
#             )

#     ax.invert_yaxis()
#     ax.set_xlabel("k/N")
#     ax.set_ylabel("SNR (dB)")
#     plt.title("Gained Accuracy using our method \n compared to Classic Split-Learning")

#     plt.tight_layout()

#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, "delta_heatmap_continuous.png"))
#     else:
#         plt.show()


# def plot_final_accuracy_heatmap(records,
#                                 method_a='ours',
#                                 method_b='classic_split_learning',
#                                 output_dir=None,
#                                 cmap='RdYlGn'):
#     """
#     Diverging heatmap (method_a - method_b) as background color,
#     with each cell showing:
#       <method_a val_accuracy>
#       (<+/- delta = ours - classic>)
#     """
#     # collect unique grid axes
#     snr_vals = sorted({r['snr'] for r in records})
#     kn_vals  = sorted({r['compression'] for r in records})
#     n, m = len(snr_vals), len(kn_vals)

#     # prepare matrices
#     mat_delta = np.zeros((n, m), dtype=float)
#     mat_acc   = np.zeros((n, m), dtype=float)

#     # fill them
#     for i, snr in enumerate(snr_vals):
#         for j, kn in enumerate(kn_vals):
#             ra = next((r['val_accuracy'] for r in records
#                        if r['snr']==snr and r['compression']==kn and r['method']==method_a), 0.0)
#             rb = next((r['val_accuracy'] for r in records
#                        if r['snr']==snr and r['compression']==kn and r['method']==method_b), 0.0)
#             mat_acc[i, j]   = ra
#             mat_delta[i, j] = ra - rb

#     # symmetric color limits on the delta
#     max_abs = np.max(np.abs(mat_acc))

#     fig, ax = plt.subplots(figsize=(6,5))
#     sns.heatmap(
#         mat_acc,
#         annot=False,       # no default annot
#         cmap=cmap,
#         center=0,
#         vmin=0,
#         vmax=+max_abs,
#         linewidths=0.8,
#         linecolor='white',
#         xticklabels=[f"{k:.0e}" for k in kn_vals],
#         yticklabels=snr_vals,
#         cbar_kws={'label': "Val Accuracy"}
#     )

#     # overlay annotations
#     for i in range(n):
#         for j in range(m):
#             acc   = mat_acc[i, j]
#             delta = mat_delta[i, j]
#             # format delta with sign
#             sign = '+' if delta >= 0 else '-'
#             delta_str = f"{sign}{abs(delta):.2f}"

#             # large accuracy at cell center
#             ax.text(
#                 j + 0.5, i + 0.6,
#                 f"{acc:.2f}",
#                 ha='center', va='center',
#                 fontsize=10,
#                 fontweight='bold',
#                 color='white'
#             )
#             # smaller delta just below
#             ax.text(
#                 j + 0.5, i + 0.3,
#                 f"({delta_str})",
#                 ha='center', va='center',
#                 fontsize=8,
#                 color='white'
#             )

#     ax.invert_yaxis()
#     ax.set_xlabel("k/N")
#     ax.set_ylabel("SNR (dB)")
#     plt.title("Validation Accuracy of Our Method\nand Δ vs. Classic Split-Learning")

#     plt.tight_layout()

#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, "accuracy_with_delta_heatmap.png"), dpi=300)
#     else:
#         plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from matplotlib.patches import Patch

# def plot_stacked_vs_baseline(records,
#                              method_a='ours',
#                              method_b='classic_split_learning',
#                              output_dir=None,
#                              spacing=3.5,
#                              bar_width=0.55,
#                              bar_depth=0.55):
#     """
#     Side-by-side 3D bar panels:
#       • Left:  baseline + Δ (ours−baseline) stacked
#       • Right: baseline only
#     A single shared legend and suptitle.
#     """
#     # 1) Collect sorted unique SNRs and compression rates
#     snr_vals = sorted({r['snr'] for r in records})
#     kn_vals  = sorted({r['compression'] for r in records})

#     # 2) Build matrices of final val accuracies
#     def build_matrix(method):
#         M = np.zeros((len(snr_vals), len(kn_vals)), dtype=float)
#         for i, snr in enumerate(snr_vals):
#             for j, kn in enumerate(kn_vals):
#                 rec = next(
#                     (r for r in records
#                      if r['snr']==snr and
#                         r['compression']==kn and
#                         r['method']==method),
#                     None
#                 )
#                 M[i, j] = rec['val_accuracy'] if rec else 0.0
#         return M

#     A = build_matrix(method_a)  # ours
#     B = build_matrix(method_b)  # classic
#     D = A - B                    # delta

#     # 3) Create figure and two 3D subplots
#     fig = plt.figure(figsize=(14, 6))
#     ax2 = fig.add_subplot(1, 2, 1, projection='3d')
#     ax1 = fig.add_subplot(1, 2, 2, projection='3d')
#     ax1.view_init(elev=25, azim=-146)
#     ax2.view_init(elev=25, azim=-146)
#     # 4) Draw the bars
#     for i in range(len(snr_vals)):
#         for j in range(len(kn_vals)):
#             x = j * spacing
#             y = i * spacing
#             base = B[i, j]
#             delta = max(D[i, j], 0)

#             # Left panel: baseline
#             ax1.bar3d(
#                 x, y, 0,
#                 bar_width, bar_depth, base,
#                 color='C1', alpha=0.7
#             )
#             # Left panel: delta on top
#             ax1.bar3d(
#                 x, y, base,
#                 bar_width, bar_depth, delta,
#                 color='C0', alpha=0.7
#             )

#             # Right panel: baseline only
#             ax2.bar3d(
#                 x, y, 0,
#                 bar_width, bar_depth, base,
#                 color='C1', alpha=0.7
#             )

#     # 5) Configure axes for both plots
#     for ax in (ax1, ax2):
#         ax.set_xticks(np.arange(len(kn_vals)) * spacing)
#         ax.set_xticklabels([f"{k:.0e}" for k in kn_vals], rotation=20)
#         ax.set_xlabel("k/N", rotation=0, labelpad=10)
#         ax.set_yticks(np.arange(len(snr_vals)) * spacing)
#         ax.set_yticklabels(snr_vals)
#         ax.set_ylabel("SNR (dB)", rotation=0, labelpad=10)
#         ax.set_zlim(0, 1)
#         ax.zaxis.set_rotate_label(False)
#         ax.set_zlabel("Val Accuracy", rotation=90, labelpad=10)
#         ax.set_zlabel("Val Accuracy")

#     # 6) Shared legend and super-title
#     legend_handles = [
#         Patch(color='C1', label='Baseline Accuracy'),
#         Patch(color='C0', label='Gained Accuracy'),
#     ]
#     fig.legend(
#         handles=legend_handles,
#         loc='upper center',
#         ncol=2,
#         frameon=False,
#         bbox_to_anchor=(0.5, 0.95)
#     )
#     fig.suptitle("Accuracy gains using our method", y=0.99)

#     # 7) Layout & save/show
#     plt.tight_layout(rect=(0, 0, 1, 0.92))
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         fig.savefig(os.path.join(output_dir, "stacked_vs_baseline_3d.png"))
#     else:
#         plt.show()


# def main():
#     parser = argparse.ArgumentParser(
#         description="Plot compression vs metrics for split-learning methods"
#     )
#     parser.add_argument(
#         '--results_dir',
#         type=str,
#         default='/home/federico/Desktop/Split_Learning/results/baselines/imagenette/deit_tiny_patch16_224.fb_in1k/',
#         help='Root directory containing method subfolders'
#     )
#     parser.add_argument(
#         '--output_dir',
#         type=str,
#         default='plots/imagenette',
#         help='Directory to save output plots'
#     )
#     parser.add_argument(
#         '--comm_threshold',
#         type=float,
#         default=None,
#         help='Communication cost threshold at which to sample metrics'
#     )
#     args = parser.parse_args()

#     records = load_all_results(args.results_dir, comm_threshold=args.comm_threshold)
#     if not records:
#         print('No valid records found under', args.results_dir)
#         return

#     # compression vs accuracy & loss
#     plot_metric_vs_compression(records, 'val_accuracy', output_dir=args.output_dir)
#     plot_metric_vs_compression(records, 'val_loss',     output_dir=args.output_dir)

#     # epoch-curves grids
#     plot_epoch_curves_grid(records, 'val_accuracy', output_dir=args.output_dir)
#     plot_epoch_curves_grid(records, 'val_loss',     output_dir=args.output_dir)

#     # final-accuracy heatmaps: ours, classic, and delta
#     plot_delta_heatmap_continuous(records, output_dir=args.output_dir)
#     plot_stacked_vs_baseline(records, output_dir=args.output_dir)
#     plot_final_accuracy_heatmap(records, output_dir=args.output_dir)

# if __name__ == '__main__':
#     main()



#  ######################### COMPRESSION ABLATION PLOTS  ######################################
# import os
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# import numpy as np
# import os

# # Collect data 
# def collect_combinations_and_accuracies(root_dir):
#     combinations = []
#     accuracies = []

#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         if "training_results.json" in filenames:
#             base = os.path.basename(dirpath)
#             if base.startswith("params="):
#                 params_str = base[len("params="):]
#                 try:
#                     params = eval(params_str)
#                     token_c = float(params["token_compression"])
#                     batch_c = float(params["batch_compression"])
#                     encoder_c = float(params["encoder_compression"])
#                 except Exception:
#                     continue

#                 try:
#                     with open(os.path.join(dirpath, "training_results.json"), "r") as f:
#                         data = json.load(f)
#                         val_acc = max(data["Val accuracies"])
#                 except Exception:
#                     continue

#                 combinations.append((token_c, batch_c, encoder_c))
#                 accuracies.append(val_acc)

#     return combinations, accuracies

# def plot_3d_heatmap(combinations, accuracies, output_path="plots/ablation/surface_colored_by_accuracy.png"):
#     # Unpack data
#     Token_Compression, Batch_Compression, Encoder_Compression = zip(*combinations)

#     # Make numpy 
#     Token_Compression = np.array(Token_Compression)
#     Batch_Compression = np.array(Batch_Compression)
#     Encoder_Compression = np.array(Encoder_Compression)
#     accuracies = np.array(accuracies)

#     # Get compression level
#     k = np.mean(Token_Compression * Batch_Compression * Encoder_Compression)

#     # Create interpolation grid
#     t_lin = np.logspace(-3, 0, 50)
#     b_lin = np.logspace(-3, 0, 50)
#     t_mesh, b_mesh = np.meshgrid(t_lin, b_lin)
#     e_mesh = k / (t_mesh * b_mesh)

#     # Mask out extreme C values
#     e_mesh = np.ma.masked_where(np.abs(e_mesh) > 0.5, e_mesh)

#     # Interpolate accuracies   
#     interpolated_accuracies = griddata(
#         points=(Token_Compression, Batch_Compression, Encoder_Compression),
#         values=accuracies,
#         xi=(t_mesh, b_mesh, e_mesh),  fill_value=0,
#         method='nearest')
    
#     # Plot
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Normalize colors 
#     norm   = Normalize(vmin=accuracies.min(), vmax=accuracies.max())
#     colors = plt.cm.viridis(norm(interpolated_accuracies))
#     mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
#     fig.colorbar(mappable, ax=ax,shrink=0.5, aspect=10, label='Final Val Accuracy')

#     ax.plot_surface(
#         t_mesh, b_mesh, e_mesh,
#         facecolors=colors,
#         rstride=1,
#         cstride=1,
#         linewidth=0.1,
#         edgecolor='k',
#         antialiased=True,
#         alpha=1.0,
#         shade=False)
    
#     # Labels and view
#     ax.set_xlabel('Token Compression', labelpad=10)
#     ax.set_ylabel('Batch Compression', labelpad=10)
#     ax.set_zlabel('Encoder Compression', labelpad=10)

#     ax.set_title('Ablation study on Compression Components with fixed SNR=-10, k/N = 0.001, and no gradient stopping. ')
#     ax.view_init(elev=15, azim=70)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def plot_heatmap_2d(x_vals, y_vals, value, xlabel, ylabel, title, output_path, log = False):

#     plt.figure(figsize=(8, 6))
#     plt.scatter(x_vals, y_vals, c=value, cmap='viridis', s=100, edgecolors='k')
#     plt.colorbar(label='Validation Accuracy')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     max_value = max(max(x_vals),max(y_vals))
#     if log:
#         plt.xscale('log')
#         plt.yscale('log')
#     plt.xlim(0, max_value * 1.1)
#     plt.ylim(0, max_value * 1.1)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def generate_all_heatmaps(combinations, accuracies):
#     os.makedirs("plots/ablation", exist_ok=True)

#     batch_compressions = []
#     encoder_compressions = []
#     token_compressions = []
#     val_accuracies = []

#     for (token, batch, enc), acc in zip(combinations, accuracies):
#         token_compressions.append(token)
#         batch_compressions.append(batch)
#         encoder_compressions.append(enc)
#         val_accuracies.append(acc)



#     plot_heatmap_2d(batch_compressions, encoder_compressions, val_accuracies,
#                     xlabel='Batch Compression',
#                     ylabel='Encoder Compression',
#                     title=f'Val Accuracy',
#                     output_path=f'plots/ablation/heatmap_batch_encoder.png')
    
#     plot_heatmap_2d(token_compressions, encoder_compressions, val_accuracies,
#                     xlabel='Token Compression',
#                     ylabel='Encoder Compression',
#                     title=f'Val Accuracy',
#                     output_path=f'plots/ablation/heatmap_token_encoder.png')
    
#     plot_heatmap_2d(batch_compressions, token_compressions, val_accuracies,
#                     xlabel='Batch Compression',
#                     ylabel='Token Compression',
#                     title=f'Val Accuracy',
#                     output_path=f'plots/ablation/heatmap_batch_token.png')

# def plot_alpha_beta_heatmap(combinations, accuracies, output_path="plots/ablation/heatmap_alpha_beta.png"):
#     alphas = []
#     betas = []

#     for (token, batch, encoder) in combinations:
#         alphas.append(batch / encoder)
#         betas.append(token / encoder)

#     plot_heatmap_2d(
#         x_vals=alphas,
#         y_vals=betas,
#         value=accuracies,
#         xlabel="Alpha = Batch / Encoder",
#         ylabel="Beta = Token / Encoder",
#         title="Val Accuracy vs α, β",
#         output_path=output_path,
#         log = True
#     )

# if __name__ == "__main__":
#     results_dir = "results/ablation/compressions/imagenette/deit_tiny_patch16_224.fb_in1k/ours/snr=-10/compression=0.001"
#     combinations, accuracies = collect_combinations_and_accuracies(results_dir)
#     plot_3d_heatmap(combinations, accuracies)
#     generate_all_heatmaps(combinations, accuracies)
#     plot_alpha_beta_heatmap(combinations, accuracies)



import argparse
import json
import os
import re
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.colors import PowerNorm

import numpy as np

# === EDIT THIS: Where plots will go ===
OUTPUT_DIR = "plots/compression_analysis"


def _extract_rates(path: str) -> Tuple[float, float, float]:
    m_batch = re.search(r"batch_compression_rate':\s*([0-9.]+)", path)
    m_token = re.search(r"token_compression_rate':\s*([0-9.]+)", path)
    m_comp = re.search(r"compression=([0-9.]+)", path)
    return float(m_token.group(1)), float(m_batch.group(1)), float(m_comp.group(1))


def _gather_runs(root):
    xs, ys, zs, accs = [], [], [], []

    for dirpath, _, filenames in os.walk(root):
        if "training_results.json" not in filenames:
            continue
        token, batch, comp = _extract_rates(dirpath)
        with open(os.path.join(dirpath, "training_results.json")) as fh:
            data = json.load(fh)
        val_acc = data.get("Val accuracies") or data.get("Val Accuracies")
        acc = max(val_acc)

        if batch != 0.5 and batch!= 0.541667:
            xs.append(token)
            ys.append(batch)
            zs.append(comp)
            accs.append(acc)

    if not xs:
        raise RuntimeError(f"No valid runs found under {root}")
    return np.array(xs), np.array(ys), np.array(zs), np.array(accs)


def _plot_3d(xs, ys, zs, accs, out_path):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    norm = PowerNorm(gamma=2, vmin=accs.min(), vmax=accs.max())
    sc = ax.scatter(xs, ys, zs, c=accs, cmap='viridis', norm=norm)

    ax.set_xlabel("Token compression rate")
    ax.set_ylabel("Batch compression rate")
    ax.set_zlabel("Overall compression")
    ax.view_init(elev=0, azim=-135)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Final Val Accuracy')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"3D scatter saved ➜ {out_path}")

from matplotlib.colors import PowerNorm
def _plot_heatmaps(xs, ys, accs, comps, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Define grid
    unique_x = np.sort(np.unique(xs))
    unique_y = np.sort(np.unique(ys))
    X, Y = np.meshgrid(unique_x, unique_y)

    def make_grid(values):
        grid = np.full(X.shape, np.nan)
        for x, y, v in zip(xs, ys, values):
            xi = np.where(unique_x == x)[0][0]
            yi = np.where(unique_y == y)[0][0]
            grid[yi, xi] = v
        return grid

    acc_grid = make_grid(accs)
    comp_grid = make_grid(comps)

    acc_norm = PowerNorm(gamma=2, vmin=np.nanmin(acc_grid), vmax=np.nanmax(acc_grid))
    comp_norm = PowerNorm(gamma=1, vmin=np.nanmin(comp_grid), vmax=np.nanmax(comp_grid))

    extent = [unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()]

    im0 = axes[0].imshow(acc_grid, cmap="viridis", norm=acc_norm, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Token compression rate")
    axes[0].set_ylabel("Batch compression rate")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(comp_grid, cmap="viridis", norm=comp_norm, origin="lower", aspect="auto", extent=extent)
    axes[1].set_title("Overall Compression")
    axes[1].set_xlabel("Token compression rate")
    axes[1].set_ylabel("Batch compression rate")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Add line: batch = sqrt(token)
    x_line = np.linspace(unique_x.min(), unique_x.max(), 500)
    y_line = np.sqrt(x_line)
    for ax in axes:
        ax.plot(x_line, y_line, 'r--', linewidth=1.5, label=r'$\mathrm{batch} = \sqrt{\mathrm{token}}$')
        ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Heatmaps saved ➜ {out_path}")

def main():
    output_dir = "plots/hyperparameter_tuning"
    root = "/home/federico/Desktop/Split_Learning/results/hyperparameters_tuning/imagenette/deit_tiny_patch16_224.fb_in1k/ours/snr=10/"
    os.makedirs(output_dir, exist_ok=True)

    xs, ys, zs, accs = _gather_runs(root)

    _plot_3d(xs, ys, zs, accs, os.path.join(output_dir, "3d_scatter.png"))
    _plot_heatmaps(xs, ys, accs, zs, os.path.join(output_dir, "side_by_side_heatmaps.png"))


if __name__ == "__main__":
    main()

