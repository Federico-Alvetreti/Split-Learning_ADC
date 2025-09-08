#!/usr/bin/env bash

# bash slurm/ablations/batch_size.sh deit_tiny_patch16_224 food_101 [2, 4, 8, 16, 32] c3-sl
# bash slurm/ablations/batch_size.sh deit_small_patch16_224 food_101 [2, 4, 8, 16, 32] c3-sl
# bash slurm/ablations/batch_size.sh deit_small_patch16_224 cifar_100 [2, 4, 8, 16, 32] c3-sl
# bash slurm/ablations/batch_size.sh deit_small_patch16_224 cifar_100 [2, 4, 8, 16, 32] c3-sl


model=${1:-"deit_tiny_patch16_224"}
dataset=${2:-"cifar_100"}
compression=${3:-0.5}
method=${4:-"proposal"}

#models=("deit_tiny_patch16_224" "deit_small_patch16_224")
#datasets=("cifar_100" "food_101")

pool=(16 32 64 128 256)
for ppl in "${pool[@]}"; do
    sbatch  slurm/main_batch_size.sh $method $model $dataset $compression $ppl
done
