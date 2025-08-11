#!/usr/bin/env bash
set -euo pipefail

model=${1:-"deit_tiny_patch16_224"}
dataset=${2:-"cifar_100"}

# Define arrays correctly (no spaces around '=' and use parentheses properly)
batch_compressions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
token_compressions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

#srun python main_dev.py method='proposal' dataset=${2} model=${1} method.parameters.token_compression=${3} method.parameters.token_compression=${4} hyperparameters.experiment_name=search

# Grid-search the two hyper-parameters
for bp in "${batch_compressions[@]}"; do
  for tp in "${token_compressions[@]}"; do
    echo $tp $bp
    sbatch slurm/proposal_hyp_search.sh $model $dataset $tp $bp
  done
done
