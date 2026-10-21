#include<stdio.h>
#include<cuda_runtime.h>
#include<math.h>

#ifndef DSIZE
#define DSIZE 1024
#endif

//CheckErrors Module
#define cudaCheckErrors(msg)\
    do {\
        cudaError_t __err=cudaGetLastError();\
        if(__err!=cudaSuccess){\
            fprintf(stderr,"Fatal error:%s (%s at %s:%d)\n",\
                msg,cudaGetErrorString(__err),\
                __FILE__,__LINE__);\
                fprintf(stderr,"*** FAILED - ABORTING\n");\
                exit(1);\
        }\
    }while(0)

const int BM=128,BN=128;
const int BK  = 32;        //shared memory tiling size
const int TT  = 8;         //per thread is responsible for calculating four elements
const int NTM = BM / TT;   //the number of threads in first dimension of the block
const int NTN = BN / TT;     //the number of threads in second dimension of the block
const int PAD = 0;         //padding could eliminate the bank conflict

__global__ void regtile_mmul(const float* A, const float* B, float* C, int ds) {
    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];

    const int tid    = threadIdx.y * NTN + threadIdx.x;   // 0..255
    const int base_m = blockIdx.y * BM;    //the start id of row
    const int base_n = blockIdx.x * BN;    //the start id of col

    float acc[TT][TT];
    #pragma unroll
    //initialize
    for (int i = 0; i < TT; i++)
        #pragma unroll
        for (int j = 0; j < TT; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < ds; k0 += BK) {
        //Memory Coalescing
        
        for (int e = 0; e < BM * BK / (NTN * NTM); e++) {
            //per thread take one element in every turn
            int idx = e * (NTM * NTN) + tid;
            int ra = idx / BK, ca = idx % BK;
            int rb = idx / BN, cb = idx % BN;
            As[ra][ca] = A[(base_m + ra) * ds + k0 + ca];
            Bs[rb][cb] = B[(k0 + rb) * ds + base_n + cb];
            //every turn takes 16 elements
        }
        __syncthreads();

        
        for (int k = 0; k < BK; k++) {
            float a[TT], b[TT];
            #pragma unroll
            for (int i = 0; i < TT; i++) a[i] = As[threadIdx.y * TT + i][k];
            #pragma unroll
            for (int j = 0; j < TT; j++) b[j] = Bs[k][threadIdx.x * TT + j];
            #pragma unroll
            for (int i = 0; i < TT; i++)
                #pragma unroll
                for (int j = 0; j < TT; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TT; i++)
        #pragma unroll
        for (int j = 0; j < TT; j++)
            C[(base_m + threadIdx.y * TT + i) * ds + base_n + threadIdx.x * TT + j] = acc[i][j];
}

int main()
{
    float *h_A,*h_B,*h_C,*d_A,*d_B,*d_C;

    h_A=new float[DSIZE*DSIZE];
    h_B=new float[DSIZE*DSIZE];
    h_C=new float[DSIZE*DSIZE];

    unsigned int seed = 12345u;
    for(int i=0;i<DSIZE*DSIZE;i++){
        seed = seed*1664525u + 1013904223u;  h_A[i] = (seed>>8)/16777216.0f - 0.5f;
        seed = seed*1664525u + 1013904223u;  h_B[i] = (seed>>8)/16777216.0f - 0.5f;
        h_C[i] = 0.f;
    }

    cudaMalloc(&d_A,DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_B,DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_C,DSIZE*DSIZE*sizeof(float));

    cudaCheckErrors("cudaMalloc errors");

    cudaMemcpy(d_A,h_A,DSIZE*DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,DSIZE*DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D errors");

    dim3 block(NTM,NTN);
    dim3 grid((DSIZE+BM-1)/BM,(DSIZE+BN-1)/BN);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int w=0;w<3;w++) regtile_mmul<<<grid,block>>>(d_A,d_B,d_C,DSIZE);
    cudaCheckErrors("warmup launch");
    cudaDeviceSynchronize();

    const int ITERS = 20;
    float best = 1e30f, total = 0.f;
    for(int it=0;it<ITERS;it++){
        cudaEventRecord(start);
        regtile_mmul<<<grid,block>>>(d_A,d_B,d_C,DSIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms,start,stop);
        if(ms < best) best = ms;
        total += ms;
    }
    printf("register tiling N=%d  min=%.3f ms  avg=%.3f ms  %.1f GFLOPS\n",
           DSIZE, best, total/ITERS, 2.0*DSIZE*DSIZE*DSIZE/(best*1e6));
    printf("RES,%d,%.4f,%.4f,%.1f\n", DSIZE, best, total/ITERS,
           2.0*DSIZE*DSIZE*DSIZE/(best*1e6));

    cudaMemcpy(h_C,d_C,DSIZE*DSIZE*sizeof(float),cudaMemcpyDeviceToHost);
    
    cudaCheckErrors("kernel execution failure or cudaMemcpy D2H errors");
       if (DSIZE <= 256) {
        double max_err = 0.0, max_ref = 0.0;
        for (int i = 0; i < DSIZE; i++)
            for (int j = 0; j < DSIZE; j++) {
                double ref = 0.0;
                for (int k = 0; k < DSIZE; k++)
                    ref += (double)h_A[i*DSIZE+k] * h_B[k*DSIZE+j];
                double e = fabs((double)h_C[i*DSIZE+j] - ref);
                if (e > max_err) max_err = e;
                if (fabs(ref) > max_ref) max_ref = fabs(ref);
            }
        printf("check: max_abs_err=%.3e  rel_err=%.3e  ->  %s\n",
               max_err, max_err/max_ref, (max_err/max_ref < 1e-4) ? "PASS" : "FAIL");
    } else {
    }
    return 0;
}