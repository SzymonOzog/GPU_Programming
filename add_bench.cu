#include <cmath>
#include <iostream>
#include <chrono>

#define BENCH_STEPS 400

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA error %d:  %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
  for (int p = 0; p<25; p++)
  {
    int N = std::pow(2, p);
    int BLOCK_SIZE=1024;
    float* a = new float[N];
    float* b = new float[N];
    float* c = new float[N];
    float* c2 = new float[N];
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

    double gpu_time=0.0;
    for (int i = 0; i<BENCH_STEPS; i++)
    {
      auto start_time = std::chrono::system_clock::now();
      add<<<ceil(N/(float)BLOCK_SIZE), BLOCK_SIZE>>>(N, a_d, b_d, c_d);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      double final_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start_time).count();
      gpu_time += final_time;
    }

    double cpu_time=0.0;
    for (int i = 0; i<BENCH_STEPS; i++)
    {
      auto start_time = std::chrono::system_clock::now();
      for (int j = 0; j<N; j++)
      {
        c2[j] = a[j] + b[j];
      }
      double final_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start_time).count();
      cpu_time += final_time;
    }

    std::cout<<"p = "<<p<<" cpu time: "<<cpu_time<<" gpu time: "<<gpu_time<<std::endl;

    cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
  }
  return 0;
}
