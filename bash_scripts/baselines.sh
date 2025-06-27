#!/usr/bin/env bash
set -euo pipefail

# Get models and datsets 
models=("deit_tiny_patch16_224" "deit_small_patch16_224")
datasets=("cifar_100" "food_101")

# For each combination of dataset and model 
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do

         # ------------ Base ----------- 
        python main.py\
            method="base"\
            dataset="$dataset"\
            model="$model"


        # ------------ C3-SL -----------
        Rs=(2 4 8 16 32)

        for R in "${Rs[@]}"; do
            python main.py \
                method.parameters.R="$R" \
                method="c3-sl"\
                dataset="$dataset"\
                model="$model"
        done

        # ------------ Bottlenet -----------
        compressions=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)

        for compression in "${compressions[@]}"; do
            python main.py \
                method.parameters.compression="$compression" \
                method="bottlenet"\
                dataset="$dataset"\
                model="$model"
        done


        # ------------ Quantization -----------
        numbers_of_bits=(1 3 5 6 9 12 15)

        for n_bits in "${numbers_of_bits[@]}"; do
            python main.py \
                method.parameters.n_bits="$n_bits" \
                method="quantization"\
                dataset="$dataset"\
                model="$model"
        done

        # ------------ Top-k -----------
        rates=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)

        for rate in "${rates[@]}"; do
            python main.py \
                method.parameters.rate="$rate" \
                method="top_k"\
                dataset="$dataset"\
                model="$model"
        done


        # ------------ Random Top-k -----------
        rates=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)

        for rate in "${rates[@]}"; do
            python main.py \
                method.parameters.rate="$rate" \
                method="random_top_k"\
                dataset="$dataset"\
                model="$model"
        done

        # ------------ Proposal ----------- 
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
