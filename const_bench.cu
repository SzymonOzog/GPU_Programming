#include <iomanip>
#include <iostream>
#include <cassert>

#define BLOCK_SIZE 1024
#define CONST_SIZE 16384
#define BENCH_STEPS 1000
#define TIMINGS 14
#define START 10
#define ACCESSES 10

#define access (threadIdx.x * dist) % CONST_SIZE

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

void clear_l2() {
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2; // just to be extra safe (cache is not necessarily strict LRU)
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

__constant__ float c_mem[CONST_SIZE];

__global__ void add(int n , float* a, float* b, float* c, int dist)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int y = access;
  if (i < n-ACCESSES)
  {
    for(int x = 0; x<ACCESSES; x++)
    {
      c[i] = a[i] + b[y+x];
    }
  }
}

__global__ void add_const(int n , float* a, float* c, int dist)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int y = access;
  if (i < n-ACCESSES)
  {
    for(int x = 0; x<ACCESSES; x++)
    {
      c[i] = a[i] + c_mem[y+x];
    }
  }
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
  cudaMalloc((void**) &a_d, max_N*sizeof(float));
  cudaMalloc((void**) &b_d, CONST_SIZE*sizeof(float));
  cudaMalloc((void**) &c_d, max_N*sizeof(float));
  cudaMalloc((void**) &d_d, max_N*sizeof(float));

  float* cmemset = new float[max_N];
  cudaMemset(a_d, 1, max_N*sizeof(float));
  cudaMemset(b_d, 1, CONST_SIZE*sizeof(float));
  memset(cmemset, 1, CONST_SIZE*sizeof(float));
  cudaMemcpyToSymbol(c_mem, cmemset, CONST_SIZE*sizeof(float));
  cudaMemset(d_d, 1, max_N*sizeof(float));

  for (int distance = 1; distance<17; distance++)
  {
    for (int p = START; p<START+TIMINGS; p++)
    {
      cudaEvent_t start, stop;
      float time;
      gpuErrchk(cudaEventCreate(&start));
      gpuErrchk(cudaEventCreate(&stop));
      long N = std::pow<long, long>(2, p);

      dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), 1, 1);
      dim3 dimBlock(BLOCK_SIZE, 1, 1);

      double add_time=0.0;
      for (int i = -1; i<BENCH_STEPS; i++)
      {
        clear_l2();
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaEventRecord(start));
        add<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d, distance);
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&time, start, stop));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        if (i != -1) // one warmup run
        {
          add_time += time / BENCH_STEPS;
        }
      }

      double const_time=0.0;
      for (int i = -1; i<BENCH_STEPS; i++)
      {
        clear_l2();
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaEventRecord(start));
        add_const<<<dimGrid, dimBlock>>>(N, a_d, d_d, distance);
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&time, start, stop));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        if (i != -1) // one warmup run
        {
          const_time += time / BENCH_STEPS;
        }
      }

      mt[p-START] = add_time;
      tt[p-START] = const_time;
      gpuErrchk(cudaEventDestroy(start));
      gpuErrchk(cudaEventDestroy(stop));
    }
    float* c_h = new float[max_N];
    float* d_h = new float[max_N];
    cudaMemcpy(c_h, c_d, max_N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_h, d_d, max_N*sizeof(float), cudaMemcpyDeviceToHost);
    float tolerance = 1e-6;
    for (int i = 0; i < max_N; i++)
    {
      ASSERT(abs(c_h[i] - d_h[i]) < tolerance, "failed at %d, %f, %f\n", i, c_h[i], d_h[i]);
    }
    std::cout<<"ratio"<<distance<<" = [";
    for (int i = 0; i<TIMINGS; i++)
    {
      std::cout<<std::fixed<<std::setprecision(3)<<tt[i]/mt[i]<<", ";
    }
    std::cout<<"]"<<std::endl;
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  return 0;
}

