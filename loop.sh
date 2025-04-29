#!/usr/bin/env bash
set -euo pipefail

# Sweep values for each parameter
snrs=(-10 -5 0  5 10)
autoencoder_compressions=(0.5 0.4 0.3 0.2 0.1)
methods=(delta classic_split_learning)

# Loop over every combination
for comp in "${autoencoder_compressions[@]}"; do
  for snr in "${snrs[@]}"; do
    for method in "${methods[@]}"; do
      echo "→ Running: method=$method, snr=$snr, compression=$comp"
      python main.py \
        method="$method" \
        communication.encoder.output_size="$comp" \
        hyperparameters.snr="$snr"
    done
  done
done
