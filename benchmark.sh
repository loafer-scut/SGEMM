#!/bin/bash
set -euo pipefail

SIZES=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680 8192 8704 9216 9728 10240)
ARCH=${ARCH:-sm_120}
mkdir -p build results

echo "GPU : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "NVCC: $(nvcc --version | tail -1)"

printf "N,min_ms,avg_ms,gflops\n" > results/v1_naive.csv
printf "N,min_ms,avg_ms,gflops\n" > results/v2_tiling.csv
printf "N,min_ms,avg_ms,gflops\n" > results/v3_regtile.csv

for SIZE in "${SIZES[@]}"; do
    nvcc -O3 -arch=$ARCH -DDSIZE=$SIZE naive.cu          -o build/v1_$SIZE
    nvcc -O3 -arch=$ARCH -DDSIZE=$SIZE shared_memory.cu  -o build/v2_$SIZE
    nvcc -O3 -arch=$ARCH -DDSIZE=$SIZE register_tiling.cu -o build/v3_$SIZE
    ./build/v3_$SIZE | grep '^RES' | sed 's/^RES,//' >> results/v3_regtile.csv
    ./build/v1_$SIZE | grep '^RES' | sed 's/^RES,//' >> results/v1_naive.csv
    ./build/v2_$SIZE | grep '^RES' | sed 's/^RES,//' >> results/v2_tiling.csv
    echo "  N=$SIZE done"
done
echo "完成:results/v1_naive.csv / results/v2_tiling.csv"