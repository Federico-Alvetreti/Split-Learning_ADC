#!/usr/bin/env bash

models=("deit_tiny_patch16_224" "deit_small_patch16_224")
datasets=("cifar_100" "food_101")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        Rs=(2 4 8 16 32)
        for R in "${Rs[@]}"; do
          python main.py \
              method.parameters.R="$R" \
              method="c3-sl"\
              dataset="$dataset"\
              model="$model"
          done
    done
done