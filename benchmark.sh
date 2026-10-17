#!/bin/bash
# run_benchmark.sh


SIZES=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680 8192 8704 9216 9728 10240)
CSV_FILE="benchmark_results.csv"

echo "DSIZE,Naive_ms,Shared_ms" > $CSV_FILE
echo "Starting 20-point benchmark..."

for SIZE in "${SIZES[@]}"; do
    echo "====================================="
    echo "Testing DSIZE = $SIZE"
    

    nvcc -DDSIZE=$SIZE naive.cu -o naive_$SIZE
    nvcc -DDSIZE=$SIZE shared_memory.cu -o shared_$SIZE
    
 
    NAIVE_OUT=$(./naive_$SIZE)
    NAIVE_TIME=$(echo "$NAIVE_OUT" | grep "kernel takes" | awk '{print $(NF-1)}')
    

    SHARED_OUT=$(./shared_$SIZE)
    SHARED_TIME=$(echo "$SHARED_OUT" | grep "kernel takes" | awk '{print $(NF-1)}')
    
    echo "$SIZE,$NAIVE_TIME,$SHARED_TIME" >> $CSV_FILE
    echo "  -> Naive: $NAIVE_TIME ms | Shared: $SHARED_TIME ms"
done

echo "====================================="
echo "Benchmark completed! Results saved to $CSV_FILE"