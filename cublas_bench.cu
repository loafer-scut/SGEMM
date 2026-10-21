#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef DSIZE
#define DSIZE 1024
#endif

int main()
{
    const int N = DSIZE;
    const size_t n2 = (size_t)N * N;

    float *h_A = new float[n2], *h_B = new float[n2], *h_C = new float[n2];
    unsigned int seed = 12345u;
    for (size_t i = 0; i < n2; i++) {
        seed = seed * 1664525u + 1013904223u;  h_A[i] = (seed >> 8) / 16777216.0f - 0.5f;
        seed = seed * 1664525u + 1013904223u;  h_B[i] = (seed >> 8) / 16777216.0f - 0.5f;
        h_C[i] = 0.f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n2 * sizeof(float));
    cudaMalloc((void**)&d_B, n2 * sizeof(float));
    cudaMalloc((void**)&d_C, n2 * sizeof(float));
    cudaMemcpy(d_A, h_A, n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n2 * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);      //force the cublas use FP32 instead of TF32

    const float alpha = 1.f, beta = 0.f;  //we need to calculate C=A*B , so the alpha is 1.0 and beta is 0

    // cublas uses column as the main sequence , we need to transform the matrix
    auto run = [&]() {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_B, N, d_A, N, &beta, d_C, N);
    };

    for (int w = 0; w < 3; w++) run();               
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float best = 1e30f, total = 0.f;
    const int ITERS = 20;
    for (int it = 0; it < ITERS; it++) {
        cudaEventRecord(s);
        run();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, s, e);
        if (ms < best) best = ms;
        total += ms;
    }
    double gf = 2.0 * N * N * (double)N / (best * 1e6);
    printf("cuBLAS N=%d  min=%.3f ms  avg=%.3f ms  %.1f GFLOPS\n", N, best, total / ITERS, gf);
    printf("RES,%d,%.4f,%.4f,%.1f\n", N, best, total / ITERS, gf);

    cudaMemcpy(h_C, d_C, n2 * sizeof(float), cudaMemcpyDeviceToHost);

    
    if (N <= 256) {
        double max_err = 0.0, max_ref = 0.0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double ref = 0.0;
                for (int k = 0; k < N; k++)
                    ref += (double)h_A[(size_t)i * N + k] * h_B[(size_t)k * N + j];
                double d = fabs((double)h_C[(size_t)i * N + j] - ref);
                if (d > max_err) max_err = d;
                if (fabs(ref) > max_ref) max_ref = fabs(ref);
            }
        printf("check: max_abs_err=%.3e  rel_err=%.3e  ->  %s\n",
               max_err, max_err / max_ref, (max_err / max_ref < 1e-4) ? "PASS" : "FAIL");
    }

    cublasDestroy(h);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}