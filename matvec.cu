#include <iostream>
#include <cassert>
#include <ostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matvec(int n, float* a, float* b, float* c)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if (col < n)
  {
    for(int i = 0; i < n; i++)
    {
      c[col] += a[i*n + col] * b[col];
    }
  }
}


int main()
{
  int N = 1024;
  int BLOCK_SIZE=32;
  float* a = new float[N*N];
  float* b = new float[N];
  float* c = new float[N];
  for (int i = 0; i<N; i++)
  {
    b[i] = i;
    a[i*N + i] = 2;
  }
  float* a_d;
  float* b_d;
  float* c_d;

  cudaMalloc((void**) &a_d, N*N*sizeof(float));
  cudaMalloc((void**) &b_d, N*sizeof(float));
  cudaMalloc((void**) &c_d, N*sizeof(float));

  cudaMemcpy(a_d, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(N/(float)BLOCK_SIZE),1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  matvec<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);

  
  for (int i = 0; i<N; i++)
  {
    assert(c[i] == b[i]*2);
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  return 0;
}
