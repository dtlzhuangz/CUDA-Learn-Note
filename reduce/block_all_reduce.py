import torch
import time 
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='block_all_reduce_lib', 
           sources=['block_all_reduce.cu'], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ], 
           extra_cflags=['-std=c++17'])


def run_benchmark(perf_func: callable, values: torch.Tensor, tag: str, 
                  warmup: int = 10, iters: int = 1000):
    # if perf_func.__name__ == torch.sum.__name__:
    #     values = values.float() # for precision
    for i in range(warmup):
        out = perf_func(values) # warmup
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.item()
    if tag.startswith("i8"):
        print(f"{out_info:>17}: {out_val:<15}, time:{mean_time:.8f}ms")
    else:
        print(f"{out_info:>17}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


print("-" * 80)
N_ELEMENTS = 256*92*16
values = torch.randn((N_ELEMENTS)).cuda().float()
run_benchmark(lib.block_all_reduce_sum_f32_acc_with_f32,   values, "f32f32")
run_benchmark(lib.block_all_reduce_sum_f32x4_acc_with_f32, values, "f32x4f32")
run_benchmark(torch.sum, values, "f32f32_th")

print("-" * 80)
values_half = values.half()
run_benchmark(lib.block_all_reduce_sum_f16_acc_with_f16,   values_half, "f16f16")
run_benchmark(lib.block_all_reduce_sum_f16_acc_with_f32,   values_half, "f16f32")
run_benchmark(lib.block_all_reduce_sum_f16x2_acc_with_f32, values_half, "f16x2f32")
run_benchmark(lib.block_all_reduce_sum_f16x2_acc_with_f16, values_half, "f16x2f16")
run_benchmark(torch.sum, values_half, "f16f16_th")

print("-" * 80)
values_bf16 = values.bfloat16()
run_benchmark(lib.block_all_reduce_sum_bf16_acc_with_bf16,   values_bf16, "bf16bf16")
run_benchmark(lib.block_all_reduce_sum_bf16_acc_with_f32,    values_bf16, "bf16f32")
run_benchmark(lib.block_all_reduce_sum_bf16x2_acc_with_f32,  values_bf16, "bf16x2f32")
run_benchmark(lib.block_all_reduce_sum_bf16x2_acc_with_bf16, values_bf16, "bf16x2bf16")
run_benchmark(torch.sum, values_bf16, "bf16bf16_th")

print("-" * 80)
values_f8e4m3 = values.to(dtype=torch.float8_e4m3fn)
run_benchmark(lib.block_all_reduce_sum_fp8_e4m3_acc_with_f16, values_f8e4m3, "f8e4m3f16")
run_benchmark(torch.sum, values_f8e4m3.half(), "f8e4m3f16_th") # torch.sum not support fp8

print("-" * 80)
values_f8e5m2 = values.to(dtype=torch.float8_e5m2)
run_benchmark(lib.block_all_reduce_sum_fp8_e5m2_acc_with_f16, values_f8e5m2, "f8e5m2f16")
run_benchmark(torch.sum, values_f8e5m2.half(), "f8e5m2f16_th") # torch.sum not support fp8

print("-" * 80)
values_i8 = torch.ones(N_ELEMENTS, dtype=torch.int8).cuda()
run_benchmark(lib.block_all_reduce_sum_i8_acc_with_i32, values_i8, "i8i32")
run_benchmark(torch.sum, values_i8, "i8i32_th")
print("-" * 80)