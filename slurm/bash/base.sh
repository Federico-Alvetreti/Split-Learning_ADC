#!/usr/bin/env bash

models=("deit_tiny_patch16_224" "deit_small_patch16_224")
datasets=("cifar_100" "food_101")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
#        python main.py\
#            method="base"\
#            dataset="$dataset"\
#            model="$model"
#
          sbatch  slurm/one_parameter.sh "base" $model $dataset
        sbatch
    done
done
