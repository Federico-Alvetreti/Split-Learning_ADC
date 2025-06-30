#!/usr/bin/env bash

models=("deit_tiny_patch16_224" "deit_small_patch16_224")
datasets=("cifar_100" "food_101")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        rates=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)
        for rate in "${rates[@]}"; do
            python main.py \
                method.parameters.rate="$rate" \
                method="top_k"\
                dataset="$dataset"\
                model="$model"
        done
    done
done
