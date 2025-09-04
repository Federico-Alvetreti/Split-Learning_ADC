#!/usr/bin/env bash

models=("deit_tiny_patch16_224" "deit_small_patch16_224")
datasets=("cifar_100" "food_101")
datasets=("imagenette")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        Rs=(2 4 8 16 32)
        for R in "${Rs[@]}"; do
            sbatch  slurm/main.sh "c3-sl" $model $dataset $R
          done
    done
done