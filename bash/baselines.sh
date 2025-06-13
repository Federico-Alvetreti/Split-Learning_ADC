#!/usr/bin/env bash
set -euo pipefail

# # ------------ Proposal ----------- 
# compressions=(0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

# for compression in "${compressions[@]}"; do
#     python main.py \
#         method.parameters.compression="$compression" \
#         method="proposal"
# done

# ------------ Quantization -----------
numbers_of_bits=(1 3 5 6 9 12 15 18 21 24 27 30 32)

for n_bits in "${numbers_of_bits[@]}"; do
    python main.py \
        method.parameters.n_bits="$n_bits" \
        method="quantization"
done

# # ------------ Random Top-k -----------
# rates=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

# for rate in "${rates[@]}"; do
#     python main.py \
#         method.parameters.rate="$rate" \
#         method="random_top_k"
# done
