#include <iostream>

__global__ void add(int n , float* a, float* b, float* c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}

int main()
{
  int N = 4096;
  int BLOCK_SIZE=256;
  float* a = new float[N];
  float* b = new float[N];
  float* c = new float[N];
  for (int i = 0; i<N; i++)
  {
    a[i] = i;
    b[i] = 2*i;
  }
  float* a_d;
  float* b_d;
  float* c_d;

  cudaMalloc((void**) &a_d, N*sizeof(float));
  cudaMalloc((void**) &b_d, N*sizeof(float));
  cudaMalloc((void**) &c_d, N*sizeof(float));

  cudaMemcpy(a_d, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice);

  add<<<ceil(N/(float)BLOCK_SIZE), BLOCK_SIZE>>>(N, a_d, b_d, c_d);

  cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i<10; i++)
  {
    std::cout<<a[i]<<" "<<b[i]<<" "<<c[i]<<std::endl;
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  return 0;
}
