#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

__global__ void exclusive_prefix_sum(int* input, int* output, int N) {
  int tid = threadIdx.x;
  int idx = blockDim.x * blockIdx.x + threadIdx.x; 

  extern __shared__ int smem[]; 
  if (idx < N) { 
    smem[tid] = input[idx];
  } else { 
    smem[tid] = 0; 
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) { 
    if ((tid+1) % (stride * 2) == 0) { 
      smem[tid] += smem[tid-stride];
    }
    __syncthreads();
  }

  if (tid == (blockDim.x-1)) { 
    smem[tid] = 0; 
  }

  for (int stride = (blockDim.x/2); stride >= 1; stride /= 2) { 
    if ((tid+1) % (stride * 2) == 0) { 
      int intermediate = smem[tid]; 
      smem[tid] += smem[tid-stride];
      smem[tid-stride] = intermediate; 
    }
    __syncthreads();
  }

  if (idx < N) {
    output[idx] = smem[tid]; 
  } 
}

int main() {
  int N = 1024;
  int *input;
  int *output;
  cudaMalloc(&input, N * sizeof(int));
  cudaMalloc(&output, N * sizeof(int));
  cudaMemset(output, 0, N * sizeof(int)); 

  int *host_input = (int*) malloc(N*sizeof(int));
  for (int i = 0; i < N; i++) {
    host_input[i] = i; 
  }

  cudaMemcpy(input, host_input, N*sizeof(int), cudaMemcpyHostToDevice); 
  
  int threadsPerBlock = 1024;
  int blocksPerGrid = ((N + (threadsPerBlock - 1)) / threadsPerBlock);

  exclusive_prefix_sum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(input, output, N);

  cudaDeviceSynchronize();

  int *host_output = (int*)malloc(N * sizeof(int));
  cudaMemcpy(host_output, output, N*sizeof(int), cudaMemcpyDeviceToHost);

  int prefix_sum = 0; 
  for (int i = 0; i < N; i++) {
    if (prefix_sum != host_output[i]) {
      printf("sum for index is %d, expected to be %d", host_output[i], prefix_sum);
    }
    prefix_sum += i; 
  }
  return 0; 
}
