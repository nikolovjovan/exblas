#!/bin/bash

for t in 1 2 4 8
do
    echo "Running test with $t threads..."
    export OMP_NUM_THREADS=$t
    ./run >output_$t.log
done