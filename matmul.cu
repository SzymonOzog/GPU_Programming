#include <iostream>
#include <cassert>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matmul_elem(int w, int h, float* a, float* b, float* c)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < w && column < h)
  {
    for(int i = 0; i < w; i++)
    {
      c[row*w+column] += a[row*w + i] * b[i*w + column];
    }
  }
}

__global__ void matmul_row(int w, int h, float* a, float* b, float* c)
{
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < w)
  {
    for(int i = 0; i < w; i++)
    {
      for(int j = 0; j < w; j++)
      {
        c[row*w+j] += a[row*h + i] * b[i*w + j];
      }
    }
  }
}

__global__ void matmul_col(int w, int h, float* a, float* b, float* c)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  if (column < h)
  {
    for(int i = 0; i < w; i++)
    {
      for(int j = 0; j < w; j++)
      {
        c[j*w+column] += a[j*w + i] * b[i*w + column];
      }
    }
  }
}

int main()
{
  int N = 1024;
  int BLOCK_SIZE=32;
  float* a = new float[N*N];
  float* b = new float[N*N];
  float* c = new float[N*N];
  float* d = new float[N*N];
  float* e = new float[N*N];
  for (int i = 0; i<N; i++)
  {
    for (int j = 0; j<N; j++)
    {
      if (i == j)
      {
        a[i*N + j] = 2;
      }
      b[i*N + j] = i+j;
    }
  }
  float* a_d;
  float* b_d;
  float* c_d;
  float* d_d;
  float* e_d;

  cudaMalloc((void**) &a_d, N*N*sizeof(float));
  cudaMalloc((void**) &b_d, N*N*sizeof(float));
  cudaMalloc((void**) &c_d, N*N*sizeof(float));
  cudaMalloc((void**) &d_d, N*N*sizeof(float));
  cudaMalloc((void**) &e_d, N*N*sizeof(float));

  cudaMemcpy(a_d, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  matmul_elem<<<dimGrid, dimBlock>>>(N, N, a_d, b_d, c_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(c, c_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  dim3 dimGrid_r(1, ceil(N/(float)BLOCK_SIZE), 1);
  dim3 dimBlock_r(1, BLOCK_SIZE, 1);

  matmul_row<<<dimGrid_r, dimBlock_r>>>(N, N, a_d, b_d, d_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(d, d_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  dim3 dimGrid_c(ceil(N/(float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock_c(BLOCK_SIZE, 1, 1);

  matmul_col<<<dimGrid_c, dimBlock_c>>>(N, N, a_d, b_d, e_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(e, e_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i<N; i++)
  {
    for (int j = 0; j<N; j++)
    {
      assert(c[i*N+j] == b[i*N+j]*2);
      assert(d[i*N+j] == b[i*N+j]*2);
      assert(e[i*N+j] == b[i*N+j]*2);
    }
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  cudaFree(e_d);
  return 0;
}
