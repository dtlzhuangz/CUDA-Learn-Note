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
__global__ void dot_prod_kernel_f32_f32(float* a, float* b, float* c, int N){
    __shared__ float shm[THREAD_NUM];
    int tid = threadIdx.x + blockIdx.x * THREAD_NUM;
    float prod = 0.0;
    if(tid < N)
        prod = a[tid] * b[tid];

    int lane = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;
    if(tid < N){
        shm[warp_idx] = warp_reduce_sum_f32(prod);
    }
    __syncthreads();
    
    float t = threadIdx.x < THREAD_NUM / WARP_SIZE ? shm[lane] : 0.0f;
    if(warp_idx == 0)
        t = warp_reduce_sum_f32(t);

    if(threadIdx.x == 0)
        atomicAdd(c, t);
}

template<const int THREAD_NUM>
__global__ void dot_prod_kernel_f32_f32x4(float* a, float* b, float* c, int N){
    __shared__ float shm[THREAD_NUM];
    int tid = (threadIdx.x + blockIdx.x * THREAD_NUM) * 4;
    float prod = 0.0;
    float4 reg_a = FLOAT4(a[tid]);
    float4 reg_b = FLOAT4(b[tid]);
    if(tid < N)
        prod = reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w;

    int lane = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;
    if(tid < N){
        shm[warp_idx] = warp_reduce_sum_f32(prod);
    }
    __syncthreads();
    
    float t = threadIdx.x < THREAD_NUM / WARP_SIZE ? shm[lane] : 0.0f;
    if(warp_idx == 0)
        t = warp_reduce_sum_f32(t);

    if(threadIdx.x == 0)
        atomicAdd(c, t);
}

template<const int THREAD_NUM>
__global__ void dot_prod_kernel_f16_f16(half* a, half* b, float* c, int N){
    __shared__ float shm[THREAD_NUM / WARP_SIZE];
    int tid = threadIdx.x + blockIdx.x * THREAD_NUM;
    float prod = 0.0;
    if(tid < N)
        prod = a[tid] * b[tid];

    int lane = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;
    if(tid < N){
        shm[warp_idx] = warp_reduce_sum_f16_f32(prod);
    }
    __syncthreads();
    
    float t = threadIdx.x < THREAD_NUM / WARP_SIZE ? shm[lane] : 0.0f;
    if(warp_idx == 0)
        t = warp_reduce_sum_f32(t);

    if(threadIdx.x == 0)
        atomicAdd(c, t);
}

template<const int THREAD_NUM>
__global__ void dot_prod_kernel_f16_f16x8(half* a, half* b, float* c, int N){
    __shared__ float shm[THREAD_NUM / WARP_SIZE];
    int tid = (threadIdx.x + blockIdx.x * THREAD_NUM) * 8;
    const half z = __float2half(0.0);
    half prod = z;
    half reg_A[8], reg_B[8];
    LD128(reg_A[0]) = LD128(a[tid]);
    LD128(reg_B[0]) = LD128(b[tid]);

    #pragma unroll 
    for(int i = 0; i < 8; i += 2){
        half2 v = __hmul2(HALF2(reg_A[i]), HALF2(reg_B[i]));
        prod += (i + tid < N ? v.x + v.y : z);
    }
    

    int lane = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;
    if(tid < N){
        shm[warp_idx] = warp_reduce_sum_f16_f32(prod);
    }
    __syncthreads();

    float t = threadIdx.x < THREAD_NUM / WARP_SIZE ? shm[lane] : 0.0f;
    // if(threadIdx.x < 32) printf("%d %d %f\n", blockIdx.x, threadIdx.x, t);
    if(warp_idx == 0)
        t = warp_reduce_sum_f32(t);
    if(threadIdx.x == 0)
        atomicAdd(c, t);
}


#define TORCH_BINDING_DOT_PROD(TYPE_INPUT, TYPE_ACT, TYPE_TORCH, ELEMENT_TYPE, ACCESS_NUM)   \
torch::Tensor dot_prod_##TYPE_INPUT##_##TYPE_ACT(torch::Tensor a, torch::Tensor b){             \
    CHECK_TORCH_TYPE(a, TYPE_TORCH); \
    CHECK_TORCH_TYPE(b, TYPE_TORCH); \
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device( \
    torch::kCUDA, 0); \
    auto prod = torch::zeros({1}, options); \
    const int n = a.dim(); \
    int N = 1; \
    for(int i = 0; i < n; i++){ \
        N *= a.size(i); \
    } \
    dim3 block(256); \
    dim3 grid((N + 256 - 1) / 256 / ACCESS_NUM); \
    dot_prod_kernel_##TYPE_INPUT##_##TYPE_ACT<256><<<grid, block>>>( \
        reinterpret_cast<ELEMENT_TYPE*>(a.data_ptr()), \
        reinterpret_cast<ELEMENT_TYPE*>(b.data_ptr()), \
        reinterpret_cast<float*>(prod.data_ptr()), \
        N \
    ); \
    return prod; \
}

TORCH_BINDING_DOT_PROD(f32,        f32,  torch::kFloat32,       float,              1)
TORCH_BINDING_DOT_PROD(f32,        f32x4,  torch::kFloat32,       float,              4)
TORCH_BINDING_DOT_PROD(f16,        f16,  torch::kHalf,       half,              1)
TORCH_BINDING_DOT_PROD(f16,        f16x8,  torch::kHalf,       half,              8)


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
m.def(STRINGFY(func), func);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f16x8)
}