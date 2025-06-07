#!/usr/bin/env bash
set -euo pipefail

# Define arrays correctly (no spaces around '=' and use parentheses properly)
batch_compressions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
token_compressions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

# Grid-search the two hyper-parameters
for batch_compression in "${batch_compressions[@]}"; do
  for token_compression in "${token_compressions[@]}"; do
    python main.py \
      method.parameters.token_compression="$token_compression" \
      method.parameters.batch_compression="$batch_compression" 
  done
done
