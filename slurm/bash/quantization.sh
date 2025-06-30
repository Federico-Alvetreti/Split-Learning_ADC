#!/usr/bin/env bash

models=("deit_tiny_patch16_224" "deit_small_patch16_224")
datasets=("cifar_100" "food_101")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        numbers_of_bits=(1 3 5 6 9 12 15)
        for n_bits in "${numbers_of_bits[@]}"; do
            python main.py \
                method.parameters.n_bits="$n_bits" \
                method="quantization"\
                dataset="$dataset"\
                model="$model"
        done
    done
done