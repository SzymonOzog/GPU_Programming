#include <cuda_fp16.h>
#include <stdint.h>
using dtype = float;

static __device__ __forceinline__ void mma(uint32_t* A, uint32_t* B, uint32_t* ACC)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(ACC[0]), "+r"(ACC[1])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

 __global__ void mma_kern(
         const dtype* __restrict__ a,
         const dtype* __restrict__ b,
         dtype* __restrict__ c
         ) {
        // asm volatile(".pragma \"mma_throughput\";");
        //asm volatile(".pragma \"enable_smem_spilling\";");
        extern __shared__ char* inp[];
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            *reinterpret_cast<float4*>(inp+k) = *reinterpret_cast<const float4*>(a+k);
            *reinterpret_cast<float4*>(inp+k+1024) = *reinterpret_cast<const float4*>(b+k);
        }

        uint32_t acc[2];
        uint32_t a_t[4];
        uint32_t b_t[2];
        for (int k = 0; k < 1024; k += 8) {
            for (int i = 0; i < 4; i++) {
                a_t[i] = reinterpret_cast<uint32_t*>(inp + k)[i];
            }
            for (int i = 0; i < 2; i++) {
                b_t[i] = reinterpret_cast<uint32_t*>(inp + k + 1024)[i];
            }
            mma(a_t, b_t, acc);
        }
        c[threadIdx.x] = reinterpret_cast<dtype*>(acc)[0];
        c[threadIdx.x + blockDim.x] = reinterpret_cast<dtype*>(acc)[1];
 }

 int main() {
     dtype* a = nullptr;
     dtype* b = nullptr;
     dtype* c = nullptr;

     cudaMalloc(&a, 1024 * sizeof(dtype));
     cudaMalloc(&b, 1024 * sizeof(dtype));
     cudaMalloc(&c, 1024 * sizeof(dtype));

     mma_kern<<<1,1>>>(a,b,c);
     cudaDeviceSynchronize();

     cudaFree(a);
     cudaFree(b);
     cudaFree(c);
     return 0;
 }

