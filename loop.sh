#!/usr/bin/env bash
set -euo pipefail

# Sweep values for each parameter
energy_thresholds=(0.9 0.7)
compressions=(0.9 0.7 0.5)
freeze_probs=(0.3 0.5 0.7)
update_freqs=(3 10)
snrs=(-10 0 10)

# Loop over every combination
for et in "${energy_thresholds[@]}"; do
  for comp in "${compressions[@]}"; do
    for fp in "${freeze_probs[@]}"; do
      for uf in "${update_freqs[@]}"; do
        for snr in "${snrs[@]}"; do
          echo "→ Running: energy_threshold=$et, compression=$comp, freeze_probability=$fp, update_frequency=$uf, snr=$snr"
          python main.py \
            method.parameters.energy_threshold="$et" \
            method.parameters.compression="$comp" \
            method.parameters.freeze_probability="$fp" \
            method.parameters.update_frequency="$uf" \
            hyperparameters.snr="$snr"
        done
      done
    done
  done
done
