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


__global__ void elementwise_add_kernel_f16(half* a, half* b, half* c, int N){
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    c[idx] = b[idx] + a[idx];
}


__global__ void elementwise_add_kernel_f16x2(half* a, half* b, half* c, int N){
    int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * 2;
    half2 res = __hadd2(HALF2(a[idx]), HALF2(b[idx]));
    c[idx] = res.x;
    c[idx + 1] = res.y;
}


__global__ void elementwise_add_kernel_f16x8(half* a, half* b, half* c, int N){
    int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * 8;
    half reg_a[8], reg_b[8], reg_c[8];
    LD128(reg_a[0]) = LD128(a[idx]);
    LD128(reg_b[0]) = LD128(b[idx]);
    #pragma unroll
    for(int i = 0; i < 8; i += 2){
        HALF2(reg_c[i]) = __hadd2(HALF2(reg_a[i]), HALF2(reg_b[i]));
    }
    LD128(c[idx]) = LD128(reg_c[0]);
}


#define ELEMENT_WISE_TORCH_BINDING(INPUT_TYPE, TORCH_TYPE, ELEMENT_TYPE, NUM_ELEMENTS)      \
void elementwise_add_##INPUT_TYPE(torch::Tensor a, torch::Tensor b, torch::Tensor c){       \
    CHECK_TORCH_TYPE(a, TORCH_TYPE);                                                        \
    CHECK_TORCH_TYPE(b, TORCH_TYPE);                                                        \
    const int n = a.dim();                                                                  \
    int N = 1;                                                                              \
    for(int i = 0; i < n; i++){                                                             \
        N *= a.size(i);                                                                     \
    }                                                                                       \
    if(n == 2){                                                                             \
        int s = a.size(0);                                                                  \
        int k = a.size(1);                                                                  \
        if(k / NUM_ELEMENTS <= 1024){                                                       \
            dim3 block(k / NUM_ELEMENTS);                                                   \
            dim3 grid(s);                                                                   \
            elementwise_add_kernel_##INPUT_TYPE<<<grid, block>>>(         \
                reinterpret_cast<ELEMENT_TYPE*>(a.data_ptr()),                              \
                reinterpret_cast<ELEMENT_TYPE*>(b.data_ptr()),                              \
                reinterpret_cast<ELEMENT_TYPE*>(c.data_ptr()),                              \
                N                                                                           \
            );                                                                              \
        }                                                                                   \
        else{                                                                               \
            dim3 block(256 / NUM_ELEMENTS);                                                 \
            dim3 grid((N + 256 - 1) / 256);                                                 \
            elementwise_add_kernel_##INPUT_TYPE<<<grid, block>>>(       \
                reinterpret_cast<ELEMENT_TYPE*>(a.data_ptr()),                              \
                reinterpret_cast<ELEMENT_TYPE*>(b.data_ptr()),                              \
                reinterpret_cast<ELEMENT_TYPE*>(c.data_ptr()),                              \
                N                                                                           \
            );                                                                              \
            }                                                                               \
        }                                                                                   \
    else{                                                                                   \
        dim3 block(256 / NUM_ELEMENTS);                                                     \
        dim3 grid((N + 256 - 1) / 256);                                                     \
        elementwise_add_kernel_##INPUT_TYPE<<<grid, block>>>(           \
            reinterpret_cast<ELEMENT_TYPE*>(a.data_ptr()),                                  \
            reinterpret_cast<ELEMENT_TYPE*>(b.data_ptr()),                                  \
            reinterpret_cast<ELEMENT_TYPE*>(c.data_ptr()),                                  \
            N                                                                               \
        );                                                                                  \
    }                                                                                       \
}

ELEMENT_WISE_TORCH_BINDING(f16, torch::kHalf, half, 1)
ELEMENT_WISE_TORCH_BINDING(f16x2, torch::kHalf, half, 2)
ELEMENT_WISE_TORCH_BINDING(f16x8, torch::kHalf, half, 8)


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
m.def(STRINGFY(func), func);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)

}