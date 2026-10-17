#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>

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


const int block_size=32;
const float A_val=3.0f;
const float B_val=2.0f;

__global__ void shared_memory_mmul(const float*A,const float*B,float *C,int ds){

    //Cache tiles in Shared Memory to reduce global memory access
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    int idy=threadIdx.y+blockIdx.y*blockDim.y;

    //Boundary check to prevent out-of-bounds access
    if((idx<ds)&&(idy<ds)){
        float temp=0;
        //tiling the matrix into several slices and the number of it is ds/block_size
        for(int i=0;i<ds/block_size;i++){
            As[threadIdx.y][threadIdx.x]=A[idy*ds+(threadIdx.x+block_size*i)];
            Bs[threadIdx.y][threadIdx.x]=B[(i*block_size+threadIdx.y)*ds+idx];

            //make sure synchronized data loading on shared memory in case read wrong data
            __syncthreads();

            //achieve multiple 
            for(int k=0;k<block_size;k++)
                temp+=As[threadIdx.y][k]*Bs[k][threadIdx.x];
            //make sure all the threads finished the task , so that we can start the next round
            __syncthreads();
        }
        C[idy*ds+idx]=temp;
    }

}

int main()
{
    float *h_A,*h_B,*h_C,*d_A,*d_B,*d_C;

    h_A=new float[DSIZE*DSIZE];
    h_B=new float[DSIZE*DSIZE];
    h_C=new float[DSIZE*DSIZE];

    for(int i=0;i<DSIZE*DSIZE;i++)
    {
        h_A[i]=A_val;
        h_B[i]=B_val;
        h_C[i]=0;
    }

    cudaMalloc(&d_A,DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_B,DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_C,DSIZE*DSIZE*sizeof(float));

    cudaCheckErrors("cudaMalloc errors");

    cudaMemcpy(d_A,h_A,DSIZE*DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,DSIZE*DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D errors");

    dim3 block(block_size,block_size);
    dim3 grid((DSIZE+block.x-1)/block.x,(DSIZE+block.y-1)/block.y);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int w=0;w<3;w++) shared_memory_mmul<<<grid,block>>>(d_A,d_B,d_C,DSIZE);
    cudaCheckErrors("warmup launch");
    cudaDeviceSynchronize();

    const int ITERS = 20;
    float best = 1e30f, total = 0.f;
    for(int it=0;it<ITERS;it++){
        cudaEventRecord(start);
        shared_memory_mmul<<<grid,block>>>(d_A,d_B,d_C,DSIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms,start,stop);
        if(ms < best) best = ms;
        total += ms;
    }
    printf("Tiling N=%d  min=%.3f ms  avg=%.3f ms  %.1f GFLOPS\n",
           DSIZE, best, total/ITERS, 2.0*DSIZE*DSIZE*DSIZE/(best*1e6));
    printf("RES,%d,%.4f,%.4f,%.1f\n", DSIZE, best, total/ITERS,
           2.0*DSIZE*DSIZE*DSIZE/(best*1e6));

    cudaMemcpy(h_C,d_C,DSIZE*DSIZE*sizeof(float),cudaMemcpyDeviceToHost);
    
    cudaCheckErrors("kernel execution failure or cudaMemcpy D2H errors");
    for(int i=0;i<DSIZE*DSIZE;i++)if(h_C[i]!=A_val*B_val*DSIZE){
        printf("mismatch at index %d,was %f,should be %f\n",i,h_C[i],A_val*B_val*DSIZE);
        return -1;
    }
    printf("Success!\n");
    return 0;
}