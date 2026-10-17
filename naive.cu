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

__global__ void naive_mmul(const float *A,const float *B,float* C,int ds){

    //naive version 
    int idx=threadIdx.x+blockIdx.x*blockDim.x;//first dimension
    int idy=threadIdx.y+blockIdx.y*blockDim.y;//second dimension
    
    //two dimension->one dimension
    //idy refers to row id , ds refers to the number of columns and i refers to every id in a row
    //idx refers to every id ina column , ds refers to the number of rows and i refers to the id of row

    if((idx<ds)&&(idy<ds)){
        float temp=0;
        for(int i=0;i<ds;i++){
            temp+=A[idy*ds+i]*B[i*ds+idx];
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

    for(int i=0;i<DSIZE*DSIZE;i++){
        h_A[i]=A_val;
        h_B[i]=B_val;
        h_C[i]=0;
    }

    cudaMalloc((void**)&d_A,DSIZE*DSIZE*sizeof(float));
    cudaMalloc((void**)&d_B,DSIZE*DSIZE*sizeof(float));
    cudaMalloc((void**)&d_C,DSIZE*DSIZE*sizeof(float));

    cudaCheckErrors("cudaMalloc failure");


    cudaMemcpy(d_A,h_A,DSIZE*DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,DSIZE*DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    dim3 block(block_size,block_size); //To make the struct of block as 2D
    dim3 grid((DSIZE+block.x-1)/block.x,(DSIZE+block.y-1)/block.y); //To ceil how many blocks do we need and make the struct of grid as 2D

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    for(int w=0;w<3;w++) naive_mmul<<<grid,block>>>(d_A,d_B,d_C,DSIZE);
    cudaCheckErrors("warmup launch");
    cudaDeviceSynchronize();

    const int ITERS = 20;
    float best = 1e30f, total = 0.f;
    for(int it=0;it<ITERS;it++){
        cudaEventRecord(start);
        naive_mmul<<<grid,block>>>(d_A,d_B,d_C,DSIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms,start,stop);
        if(ms < best) best = ms;
        total += ms;
    }
    printf("Naive N=%d  min=%.3f ms  avg=%.3f ms  %.1f GFLOPS\n",
           DSIZE, best, total/ITERS, 2.0*DSIZE*DSIZE*DSIZE/(best*1e6));
    printf("RES,%d,%.4f,%.4f,%.1f\n", DSIZE, best, total/ITERS,
           2.0*DSIZE*DSIZE*DSIZE/(best*1e6));
    cudaMemcpy(h_C,d_C,DSIZE*DSIZE*sizeof(float),cudaMemcpyDeviceToHost);

    cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
    for(int i=0;i<DSIZE*DSIZE;i++) {
        if(h_C[i]!=A_val*B_val*DSIZE){
            printf("mismatch at index %d: %f,should be %f\n",i,h_C[i],A_val*B_val*DSIZE);
            return -1;
        }
    }
    printf("Success!\n");

    delete[] h_A;delete[] h_B;delete[] h_C;
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);

    return 0;

}