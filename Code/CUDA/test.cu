#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda.h>
#include <iostream>
#include "baseline.cu"

#define NOUT_PER_THREADS 1

__global__ void Test(uint64_t *DStates, uint32_t *DOuts){

    extern __shared__ uint64_t BStates[];

    /// Read states to shared mem.
    BStates[threadIdx.x] = DStates[blockIdx.x*blockDim.x + threadIdx.x];
    __syncthreads();
    
    #pragma unroll 
    for(int i=0;i<NOUT_PER_THREADS;i++){
        DOuts[(blockIdx.x*blockDim.x + threadIdx.x)*NOUT_PER_THREADS+i] = pcg32_64(BStates[threadIdx.x],blockIdx.x*blockDim.x + threadIdx.x);
    }
    
    
    __syncthreads();
    ///Save states back:
    DStates[blockIdx.x*blockDim.x + threadIdx.x] = BStates[threadIdx.x];


};



int main(int argc, char *argv[]){


    unsigned int BlockSize_x = 256;
    unsigned int GridSize_x = 256;

    /// each thread will have one state.
    uint64_t* HStates = (uint64_t*)malloc(sizeof(uint64_t)*BlockSize_x*GridSize_x);
    uint64_t* DStates;

    uint32_t* HOuts = (uint32_t*)malloc(sizeof(uint32_t)*BlockSize_x*GridSize_x*NOUT_PER_THREADS);
    uint32_t* DOuts;

    /// Allocate device mem.
    if(cudaMalloc((void**)&DStates, sizeof(uint64_t)*BlockSize_x*GridSize_x)){
        fprintf(stderr,"ERROR, couldn't allocate Device Mem.%s","\n");
        exit(1);
    }
    if(cudaMalloc((void**)&DOuts, sizeof(uint32_t)*BlockSize_x*GridSize_x*NOUT_PER_THREADS)){
        fprintf(stderr,"ERROR, couldn't allocate Device Mem.%s","\n");
        exit(1);
    }


    ///Initialize, all the threads use same seed, the streams are attached to unique tids.
    ///Maximum total threads that can use same seed is limited with 2^63 for 64-bits states.
    ///Each stream geneartes unique RNGS with period 2^64
    uint64_t seed = 99;
    for(unsigned int i=0;i<BlockSize_x*GridSize_x;i++)
        HStates[i] = seed;


    ///Move State -> Dev.
    cudaMemcpy(DStates,HStates,sizeof(uint64_t)*BlockSize_x*GridSize_x,cudaMemcpyHostToDevice);

    
    ///Launch Kernel:
    for(unsigned int i=0;i<10000;i++)
        Test<<<GridSize_x,BlockSize_x,sizeof(uint64_t)*BlockSize_x>>>(DStates,DOuts);

    printf("Done.%s","\n");

    ///Get Result -> Loc.
    cudaMemcpy(HOuts,DOuts,sizeof(uint32_t)*BlockSize_x*GridSize_x*NOUT_PER_THREADS,cudaMemcpyDeviceToHost);
    

    free(HStates);
    free(HOuts);
    cudaFree(DStates);
    cudaFree(DOuts);    
    



    






}





