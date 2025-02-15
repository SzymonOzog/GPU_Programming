#include <iomanip>
#include <iostream>
#include <cassert>
#include <cublas_v2.h>

#define BLOCK_SIZE 128 
#define BENCH_STEPS 1000
 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))

using datatype = half;
using datatype_vec = half2;

// using datatype = float;
// using datatype_vec = float4;

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

__global__ void copy(int n , datatype* in, datatype* out)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = in[i];
  }
}

__global__ void copyf4(int n , datatype_vec* in, datatype_vec* out)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = in[i];
  }
}

__global__ void copy_loop(int n , datatype* in, datatype* out, int max_size)
{
  unsigned long i = blockIdx.x * blockDim.x;
  for (int idx = i * max_size; idx < (i+blockDim.x)*max_size; idx+=blockDim.x)
  {
      if (idx<n)
      {
        out[idx+threadIdx.x] = in[idx+threadIdx.x];
      }
  }
}

__global__ void copy_loop_datatype4(int n , datatype_vec* in, datatype_vec* out, int max_size)
{
  unsigned long i = blockIdx.x * blockDim.x;
  for (int idx = i * max_size; idx < (i+blockDim.x)*max_size; idx+=blockDim.x)
  {
      if (idx<n)
      {
        out[idx+threadIdx.x] = in[idx+threadIdx.x];
      }
  }
}

int main()
{
  datatype* in_d;
  datatype* out_d;
  datatype* out2_d;

  long N = std::pow<long, long>(2, 25);

    //one warmup run
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    cudaMalloc((void**) &in_d, N*sizeof(datatype));
    datatype* cp = new datatype[N];
    for (int i = 0; i < N; i++)
    {
        cp[i] = (float)N;
    }
    cudaMemcpy(in_d, cp, N*sizeof(datatype), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &out_d, N*sizeof(datatype));
    cudaMemset(out_d, 0, N*sizeof(datatype));

    cudaMalloc((void**) &out2_d, N*sizeof(datatype));
    cudaMemset(out2_d, 0, N*sizeof(datatype));
    float time = 0.f;
    double run_time = 0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      copy<<<dimGrid, dimBlock>>>(N, in_d, out_d);
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

    std::cout<<"regular time "<<run_time<<std::endl;

    dimGrid = dim3(ceil(N/(float)(BLOCK_SIZE*4)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    time = 0.f;
    run_time = 0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      copyf4<<<dimGrid, dimBlock>>>(N/4, reinterpret_cast<datatype_vec*>(in_d), reinterpret_cast<datatype_vec*>(out_d));
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

    std::cout<<"vectorized time "<<run_time<<std::endl;

    int loop_size = 1024;
    dimGrid = dim3(ceil(N/(float)(BLOCK_SIZE*loop_size)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    time = 0.f;
    run_time = 0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      copy_loop<<<dimGrid, dimBlock>>>(N, in_d, out2_d, loop_size);
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

    datatype* out_h = new datatype[N];
    datatype* out2_h = new datatype[N];
    cudaMemcpy(out_h, out_d, N*sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(out2_h, out2_d, N*sizeof(datatype), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
      ASSERT(out_h[i] == out2_h[i], "failed at copy loop %d, %f, %f\n", i, (float)out_h[i], (float)out2_h[i]);
    }

    loop_size = loop_size/4;
    std::cout<<"loop time "<<run_time<<std::endl;
    dimGrid = dim3(ceil(N/(float)(BLOCK_SIZE*4*loop_size)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    time = 0.f;
    run_time = 0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      copy_loop_datatype4<<<dimGrid, dimBlock>>>(N/4, reinterpret_cast<datatype_vec*>(in_d), reinterpret_cast<datatype_vec*>(out2_d), loop_size);
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
    std::cout<<"loop time vectorized "<<run_time<<std::endl;

    cudaMemcpy(out_h, out_d, N*sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(out2_h, out2_d, N*sizeof(datatype), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
      ASSERT(out_h[i] == out2_h[i], "failed at  copy loopa datatype 4 %d, %f, %f\n", i, (float)out_h[i], (float)out2_h[i]);
    }

  cudaFree(in_d);
  cudaFree(out_d);
  return 0;
}

