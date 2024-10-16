#include <iomanip>
#include <iostream>
#include <cassert>

#define BLOCK_SIZE 32 
#define BENCH_STEPS 4000
#define MAX_OFFSET 129 
 
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

__global__ void copy(int n , float* in, float* out, int offset)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i + offset] = in[i + offset];
  }
}

int main()
{
  double timings[MAX_OFFSET];
  float* in_d;
  float* out_d;

  long N = std::pow<long, long>(2, 20);

  for (int o = -1; o<MAX_OFFSET; o++)
  {
    //one warmup run
    int offset = std::max(o, 0);
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    cudaMalloc((void**) &in_d, (N+offset)*sizeof(float));
    cudaMalloc((void**) &out_d, (N+offset)*sizeof(float));
    float time = 0.f;
    double run_time = 0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      copy<<<dimGrid, dimBlock>>>(N, in_d, out_d, offset);
      gpuErrchk(cudaEventRecord(stop));
      gpuErrchk(cudaEventSynchronize(stop));
      gpuErrchk(cudaEventElapsedTime(&time, start, stop));
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      if (i != -1) // one warmup run
      {
        run_time += time / BENCH_STEPS;
      }
    }

    timings[offset] = run_time;
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }
  std::cout<<"timings"<<" = [";
  for (int i = 0; i<MAX_OFFSET; i++)
  {
    std::cout<<std::fixed<<std::setprecision(6)<<timings[i]<<", ";
  }
  std::cout<<"]"<<std::endl;
  cudaFree(in_d);
  cudaFree(out_d);
  return 0;
}

