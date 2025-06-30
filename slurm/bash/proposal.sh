#!/usr/bin/env bash

models=("deit_tiny_patch16_224" "deit_small_patch16_224")
datasets=("cifar_100" "food_101")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        compressions=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)
        for compression in "${compressions[@]}"; do
            python main.py \
                method.parameters.compression="$compression" \
                method="proposal"\
                dataset="$dataset"\
                model="$model"
        done
    done
done