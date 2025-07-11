#include <cuda_runtime.h>
#include <iostream>

__global__ void matrix_multiplication_kernel(int* A, int* B, int* C, int M, int K, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x; 
  int idy = blockDim.y * blockIdx.y + threadIdx.y; 

  if (idx < N && idy < M) {
    int s = 0; 
    for (int i = 0; i < K; i++) {
      s += A[idy * K + i] * B[i * N + idx];
    }
    C[idy * N + idx] = s; 
  }
}

int main() {
  int M = 64; 
  int K = 64;
  int N = 64;

  // we have two matricies with dimensions (M, K) and (K, N) respectively
  // A: M x K
  // B: K x N
  // output: M x N

  int* A = (int*) malloc(sizeof(int) * (M*K));
  for (int i = 0; i < (M*K); i++) {
    A[i] = i;
  }

  int* B = (int*) malloc(sizeof(int) * (K*N));
  for (int i = 0; i < (K*N); i++) {
    B[i] = i;
  }

  // Load arrays into GPUs 
  int* A_device;
  cudaMalloc(&A_device, sizeof(int) * (M*K));
  cudaMemcpy(A_device, A, sizeof(int) * (M*K), cudaMemcpyHostToDevice);
  int* B_device;
  cudaMalloc(&B_device, sizeof(int) * (K*N));
  cudaMemcpy(B_device, B, sizeof(int) * (K*N), cudaMemcpyHostToDevice);
  int* C_device;
  cudaMalloc(&C_device, sizeof(int) * (M*N));
  cudaMemset(C_device, 0, (M*N) * sizeof(int));   
  
  int* output_reference_solution = (int*) malloc(sizeof(int) * (M*N));

  // outer loop is going through all the rows of matrix A
  // inner loop is going through all the columns of matrix B 
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int dotProd = 0;
      for (int k = 0; k < K; k++){
      	dotProd += A[i * N + k] * B[k * N + j];
      }
      output_reference_solution[i * M + j] = dotProd; 
    }
  }
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
		     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_device,B_device,C_device,M,K,N);
  cudaDeviceSynchronize();

  int* host_output = (int*) malloc(sizeof(int) * (M*N));

  cudaMemcpy(host_output, C_device, sizeof(int) * (M*N), cudaMemcpyDeviceToHost);

  for (int i = 0; i < (M*N); i++) {
    if (output_reference_solution[i] = host_output[i]) {
      printf("index %d is %d, but actual value should be %d", i, host_output[i], output_reference_solution[i]);
    }
  }
  
  return 0; 
}
