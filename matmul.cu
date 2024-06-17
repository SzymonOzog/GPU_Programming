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
    float acc = 0.f;
    for(int i = 0; i < w; i++)
    {
      acc += a[row*h + i] * b[i*w + column];
    }
    c[row*w+column] = acc;
  }
}

int main()
{
  int N = 1024;
  int BLOCK_SIZE=32;
  float* a = new float[N*N];
  float* b = new float[N*N];
  float* c = new float[N*N];
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

  cudaMalloc((void**) &a_d, N*N*sizeof(float));
  cudaMalloc((void**) &b_d, N*N*sizeof(float));
  cudaMalloc((void**) &c_d, N*N*sizeof(float));

  cudaMemcpy(a_d, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  std::cout<<dimGrid.x<<" "<<dimGrid.y<<" "<<dimGrid.z<<std::endl;
  std::cout<<dimBlock.x<<" "<<dimBlock.y<<" "<<dimBlock.z<<std::endl;

  matmul_elem<<<dimGrid, dimBlock>>>(N, N, a_d, b_d, c_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(c, c_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i<N; i++)
  {
    for (int j = 0; j<N; j++)
    {
      assert(c[i*N+j] == b[i*N+j]*2);
    }
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  return 0;
}
