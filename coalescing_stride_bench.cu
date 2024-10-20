#include <iomanip>
#include <iostream>
#include <cassert>

#define BLOCK_SIZE 32
#define BENCH_STEPS 100
#define MAX_STRIDE 15
#define BLOCKS 84*10

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

__global__ void copy(int n , float* in, float* out, int stride)
{
  unsigned long i = (blockIdx.x*blockDim.x + threadIdx.x)*stride;
  if (i < n)
  {
    out[i] = in[i];
  }
  else
  {
    printf("skip load \n");
  }
}

int main()
{
  double timings[MAX_STRIDE+1];
  float* in_d;
  float* out_d;

  long N = std::pow<long, long>(2, 31);

  float* out_h = new float[N];
  float* in_h = new float[N];
  gpuErrchk(cudaMalloc((void**) &out_d, N*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &in_d, N*sizeof(float)));
  for (int s = -1; s<=MAX_STRIDE; s++)
  {
    int stride = std::pow(2, std::max(0, s));
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));


    dim3 dimGrid(BLOCKS, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    float time = 0.f;
    double run_time = 0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      cudaMemset(in_d, 1, N*sizeof(float));
      cudaMemset(out_d, 0, N*sizeof(float));
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      copy<<<dimGrid, dimBlock>>>(N, out_d, in_d, stride);
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

    std::cout<<stride<<" "<<run_time<<std::endl;
    if (s >= 0)
    {
      timings[s] = run_time;
    }
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }
  std::cout<<"timings"<<" = [";
  for (int i = 0; i<=MAX_STRIDE; i++)
  {
    std::cout<<std::fixed<<std::setprecision(6)<<timings[i]<<", ";
  }
  std::cout<<"]"<<std::endl;
  cudaFree(in_d);
  cudaFree(out_d);
  return 0;
}
