#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
// #define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&value)[0])
#define HALF2(value) (reinterpret_cast<__half2*>(&value)[0])
#define LD128(value) (reinterpret_cast<float4*>(&value)[0])


#define CHECK_TORCH_TYPE(tensor, type)                      \
if(tensor.options().dtype() != type){                       \
    throw std::runtime_error("tensor type must be"#type);   \
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return __half2float(val);
}

template<const int THREAD_NUM>
__device__ __forceinline__ float block_reduce_sum_f32_f32(float val) {
    __shared__ float shm[THREAD_NUM / WARP_SIZE];
    val = __expf(val);
    int warp_idx = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    val = warp_reduce_sum_f32<32>(val);
    if(lane == 0)
        shm[warp_idx] = val;
    __syncthreads();
    if(threadIdx.x < 32)
        val = warp_reduce_sum_f32<32>(lane < (THREAD_NUM / WARP_SIZE)? shm[lane] : 0.0f);
    return val;
}

template<const int THREAD_NUM>
__global__ void softmax_kernel_f32(float* a, float* b, float* total, int N){
    int tid = threadIdx.x + THREAD_NUM * blockIdx.x;
    float reg = a[tid];
    float sum = block_reduce_sum_f32_f32<THREAD_NUM>(reg);
    if(threadIdx.x == 0){
        atomicAdd(total, sum);
    }
    __syncthreads();
    b[tid] = __expf(reg) / total[0];
}


#define TORCH_BINDING_SOFTMAX(TYPE_INPUT, TYPE_TORCH, ELEMENT_TYPE, ACCESS_NUM)   \
void softmax_##TYPE_INPUT(torch::Tensor a, torch::Tensor b){             \
    CHECK_TORCH_TYPE(a, TYPE_TORCH); \
    CHECK_TORCH_TYPE(b, TYPE_TORCH); \
    auto options = torch::TensorOptions().dtype((TYPE_TORCH)).device(torch::kCUDA, 0);\
    auto total = torch::zeros({1}, options);                                       \
    const int n = a.dim();          \
    int N = 1; \
    for(int i = 0; i < n; i++){ \
        N *= a.size(i); \
    } \
    const int thread_num = 256; \
    dim3 block(thread_num); \
    dim3 grid((N + thread_num - 1) / thread_num / ACCESS_NUM); \
    softmax_kernel_##TYPE_INPUT<thread_num><<<grid, block>>>( \
        reinterpret_cast<ELEMENT_TYPE*>(a.data_ptr()), \
        reinterpret_cast<ELEMENT_TYPE*>(b.data_ptr()), \
        reinterpret_cast<ELEMENT_TYPE*>(total.data_ptr()), \
        N \
    ); \
}

TORCH_BINDING_SOFTMAX(f32,    torch::kFloat32,       float,              1)
// TORCH_BINDING_SOFTMAX(f32x4,  torch::kFloat32,       float,              4)


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
m.def(STRINGFY(func), func);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32)
//   TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32x4)
//   TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f16)
//   TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f16x8)
}