#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU  : %s\n", p.name);
    printf("CC   : sm_%d%d\n", p.major, p.minor);
    printf("SM   : %d\n", p.multiProcessorCount);
    printf("L2   : %.1f MB (%d bytes)\n", p.l2CacheSize/1048576.0, p.l2CacheSize);
    printf("VRAM : %.1f GB\n", p.totalGlobalMem/1073741824.0);
    int clk = 0;
    if (cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0) == cudaSuccess)
        printf("FP32 : %.1f TFLOPS @ %.0f MHz (理论峰值)\n",
               p.multiProcessorCount*128.0*2.0*clk*1e3/1e12, clk/1000.0);
    else
        printf("FP32 : clock rate unavailable\n");
    return 0;
}