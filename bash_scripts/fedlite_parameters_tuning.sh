#!/usr/bin/env bash

L=(2 4 8 16 32 64)
Q=(591 1182) #2364 4728 9456 18912

for l in "${L[@]}"; do
    for q in "${Q[@]}"; do
        echo "Running with l=$l, q=$q"
        if ! python main.py method.parameters.l="$l" method.parameters.q="$q" method="fedlite"; then
            echo "❌ Run failed with l=$l, q=$q"
        fi
    done
done