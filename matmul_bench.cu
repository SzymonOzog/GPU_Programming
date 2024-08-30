#include <iostream>
#include <cassert>
#include <chrono>
#include <random>

#define TILE_WIDTH 32
#define BENCH_STEPS 3
#define TIMINGS 8
#define START 8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))
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

__global__ void tiled_matmul(int n, float* a, float* b, float* c)
{
  __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

  int column = blockIdx.x*TILE_WIDTH + threadIdx.x;
  int row = blockIdx.y*TILE_WIDTH + threadIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float dot_prod = 0.f;
  for (int tile_offset = 0; tile_offset<n; tile_offset+=TILE_WIDTH)
  {
    int a_chk = tile_offset+tx < n && row < n;
    a_tile[ty][tx] = a_chk ? a[row*n + tile_offset+tx] : 0.f;

    int b_chk = (tile_offset+ty) < n && column < n;
    b_tile[ty][tx] = b_chk ? b[(tile_offset+ty)*n + column] : 0.f;

    __syncthreads();
    for(int i = 0; i < TILE_WIDTH; i++)
    {
      dot_prod += a_tile[ty][i] * b_tile[i][tx];
    }
    __syncthreads();
  }

  if (row < n && column < n)
  {
    c[row*n+column] = dot_prod;
  }
}

float get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // range [0, 1)
    return dis(e);
}

void cpu_matmul(int n, float* a, float* b, float*c)
{
  for (int i = 0; i<n; i++)
  {
    for (int j = 0; j<n; j++)
    {
      float dot_product = 0.f;
      for (int k = 0; k<n; k++)
      {
        dot_product += a[i*n + k] * b[k*n + j];
      }
      c[i*n+j] = dot_product; 
    }
  }
}

int main()
{
  float mt[TIMINGS];
  float tt[TIMINGS];
  float ct[TIMINGS];
  float* a_d;
  float* b_d;
  float* c_d;
  float* d_d;

  long max_N = std::pow<long, long>(2, START+TIMINGS-1);
  cudaMalloc((void**) &a_d, max_N*max_N*sizeof(float));
  cudaMalloc((void**) &b_d, max_N*max_N*sizeof(float));
  cudaMalloc((void**) &c_d, max_N*max_N*sizeof(float));
  cudaMalloc((void**) &d_d, max_N*max_N*sizeof(float));

  float* a = new float[max_N * max_N];
  float* b = new float[max_N * max_N];
  float* c = new float[max_N * max_N];

  for (int p = START; p<START+TIMINGS; p++)
  {
    long N = std::pow<long, long>(2, p);
    int BLOCK_SIZE=32;

    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    double matmul_time=0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      // CLEAR CACHE
      cudaMemset(a_d, 1, max_N*max_N*sizeof(float));
      cudaMemset(b_d, 1, max_N*max_N*sizeof(float));
      cudaMemset(c_d, 1, max_N*max_N*sizeof(float));
      cudaMemset(d_d, 1, max_N*max_N*sizeof(float));
      auto start_time = std::chrono::system_clock::now();
      matmul_elem<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      double final_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start_time).count();
      if (i != -1) // one warmup run
      {
        matmul_time += final_time;
      }
    }

    dimGrid = dim3(ceil(N/(float)TILE_WIDTH), ceil(N/(float)TILE_WIDTH), 1);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);

    double tiled_time=0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      // CLEAR CACHE
      cudaMemset(a_d, 1, max_N*max_N*sizeof(float));
      cudaMemset(b_d, 1, max_N*max_N*sizeof(float));
      cudaMemset(c_d, 1, max_N*max_N*sizeof(float));
      cudaMemset(d_d, 1, max_N*max_N*sizeof(float));
      auto start_time = std::chrono::system_clock::now();
      tiled_matmul<<<dimGrid, dimBlock>>>(N, a_d, b_d, d_d);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      double final_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start_time).count();
      if (i != -1) // one warmup run
      {
        tiled_time += final_time;
      }
    }

    double cpu_time=0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      // CLEAR CACHE
      memset(a, 1, max_N*max_N*sizeof(float));
      memset(b, 1, max_N*max_N*sizeof(float));
      memset(c, 1, max_N*max_N*sizeof(float));
      auto start_time = std::chrono::system_clock::now();
      cpu_matmul(N, a, b, c);
      double final_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start_time).count();
      if (i != -1) // one warmup run
      {
        cpu_time += final_time;
      }
    }
    std::cout<<"n = "<<N<<" matmul time: "<<matmul_time/BENCH_STEPS<<" tiled time: "<<tiled_time/BENCH_STEPS<<" cpu time: "<<cpu_time/BENCH_STEPS<<std::endl;

    mt[p-START] = matmul_time/BENCH_STEPS;
    tt[p-START] = tiled_time/BENCH_STEPS;
    ct[p-START] = cpu_time/BENCH_STEPS;
  }
  float* c_h = new float[max_N*max_N];
  float* d_h = new float[max_N*max_N];
  cudaMemcpy(c_h, c_d, max_N*max_N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_h, d_d, max_N*max_N*sizeof(float), cudaMemcpyDeviceToHost);
  float tolerance = 1e-6;
  for (int i = 0; i < max_N*max_N; i++)
  {
    ASSERT(abs(c[i] - d_h[i]) < tolerance, "failed at %d, %f, %f\n", i, c[i], d_h[i]);
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);

  std::cout<<"normal_times = [";
  for (int i = 0; i<TIMINGS; i++)
  {
    std::cout<<mt[i]<<", ";
  }
  std::cout<<"]"<<std::endl;

  std::cout<<"tiled_times = [";
  for (int i = 0; i<TIMINGS; i++)
  {
    std::cout<<tt[i]<<", ";
  }
  std::cout<<"]"<<std::endl;

  std::cout<<"cpu_times = [";
  for (int i = 0; i<TIMINGS; i++)
  {
    std::cout<<ct[i]<<", ";
  }
  std::cout<<"]"<<std::endl;
  return 0;
}
