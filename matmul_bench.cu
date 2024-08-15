#include <iostream>
#include <cassert>
#include <chrono>
#include <random>

#define TILE_WIDTH 16
#define BENCH_STEPS 1
#define TIMINGS 14
#define START 2

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
    int a_x_offset = tile_offset + tx;
    a_tile[ty][tx] = a_x_offset < n ? a[row*n + a_x_offset] : 0.f;

    int b_y_offset = tile_offset + ty;
    b_tile[ty][tx] = b_y_offset < n ? b[b_y_offset*n + column] : 0.f;

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

int main()
{
  float mt[TIMINGS];
  float tt[TIMINGS];
  float* a_d;
  float* b_d;
  float* c_d;
  float* d_d;

  long max_N = std::pow<long, long>(2, START+TIMINGS-1);
  cudaMalloc((void**) &a_d, max_N*max_N*sizeof(float));
  cudaMalloc((void**) &b_d, max_N*max_N*sizeof(float));
  cudaMalloc((void**) &c_d, max_N*max_N*sizeof(float));
  cudaMalloc((void**) &d_d, max_N*max_N*sizeof(float));
  for (int p = START; p<START+TIMINGS; p++)
  {
    long N = std::pow<long, long>(2, p);
    int BLOCK_SIZE=32;


    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    double matmul_time=0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
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
    std::cout<<"n = "<<N<<" matmul time: "<<matmul_time/BENCH_STEPS<<" tiled time: "<<tiled_time/BENCH_STEPS<<std::endl;

    mt[p-START] = matmul_time/BENCH_STEPS;
    tt[p-START] = tiled_time/BENCH_STEPS;
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
  return 0;
}
