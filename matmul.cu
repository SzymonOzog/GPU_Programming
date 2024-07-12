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

__global__ void matmul_elem(int n, float* a, float* b, float* c)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < n && column < n)
  {
    float dot_prod = 0.f;
    for(int i = 0; i < n; i++)
    {
      dot_prod += a[row*n + i] * b[i*n + column];
    }
    c[row*n+column] = dot_prod;
  }
}

__global__ void matmul_elem_onedim(int n, float* a, float* b, float* c)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int row = idx/n;
  int column = idx%n;
  if (row < n && column < n)
  {
    float dot_prod = 0.f;
    for(int i = 0; i < n; i++)
    {
      dot_prod += a[row*n + i] * b[i*n + column];
    }
    c[row*n+column] = dot_prod;
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

  cudaMalloc((void**) &a_d, N*N*sizeof(float));
  cudaMalloc((void**) &b_d, N*N*sizeof(float));
  cudaMalloc((void**) &c_d, N*N*sizeof(float));
  cudaMalloc((void**) &d_d, N*N*sizeof(float));

  cudaMemcpy(a_d, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  matmul_elem<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(c, c_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  dim3 dimGrid_r(ceil(N*N/(float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock_r(BLOCK_SIZE, 1, 1);

  matmul_elem_onedim<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(d, d_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i<N; i++)
  {
    for (int j = 0; j<N; j++)
    {
      assert(c[i*N+j] == b[i*N+j]*2);
      assert(d[i*N+j] == d[i*N+j]*2);
    }
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  return 0;
}
