import numpy as np
import timeit
import time
import torch

def benchmark_numpy_transpose(n, m, num_runs=10):
    # 创建一个简单的测试数组而不是随机值
    setup_code = f"""
import numpy as np
# 使用简单的形状初始化，避免随机初始化的开销
arr = np.ones(({n}, {m}, 8), dtype=np.int32)
# 稍微修改一些值以验证转置
for i in range(min(10, {n})):
    for j in range(min(10, {m})):
        arr[i, j] = np.arange(i*10+j, i*10+j+8)
"""
    
    # 测试代码 - 使用.copy()强制执行转置
    test_code = "np.transpose(arr, axes=(1, 0, 2)).copy()"
    
    # 计算数组大小（以字节为单位）
    element_size = np.dtype(np.int32).itemsize
    array_size_bytes = n * m * 8 * element_size
    
    # 理论内存操作
    theoretical_memory_operations = 2 * array_size_bytes  # 读取+写入
    
    print(f"NumPy Array dimensions: {n}x{m}x8 (int32)")
    print(f"Total array size: {array_size_bytes / (1024*1024):.2f} MB")
    
    # 预先执行一次
    locals_dict = {}
    exec(setup_code, globals(), locals_dict)
    
    # 首次预热（不计入测量）
    exec(test_code, globals(), locals_dict)
    
    # 测量执行时间
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        exec(test_code, globals(), locals_dict)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / num_runs
    
    # 计算理论内存带宽
    bandwidth = theoretical_memory_operations / (avg_time * 1024 * 1024 * 1024)  # GB/s
    
    print(f"Average time to transpose a {n}x{m}x8 NumPy array over {num_runs} runs: {avg_time:.6f} seconds")
    print(f"Theoretical memory bandwidth: {bandwidth:.2f} GB/s")

def benchmark_pytorch_transpose(n, m, num_runs=10, device='cpu'):
    # 创建PyTorch张量 - 不使用随机初始化
    setup_code = f"""
import torch
device = '{device}'
# 使用简单的形状初始化，避免随机初始化的开销
tensor = torch.ones(({n}, {m}, 8), dtype=torch.int32, device=device)
# 稍微修改一些值以验证转置
for i in range(min(10, {n})):
    for j in range(min(10, {m})):
        tensor[i, j] = torch.arange(i*10+j, i*10+j+8, dtype=torch.int32, device=device)
"""
    
    # 测试代码 - 使用.clone()强制执行转置
    test_code = "tensor.permute(1, 0, 2).contiguous()"
    
    # 计算张量大小（以字节为单位）
    element_size = 4  # torch.int32的字节数
    tensor_size_bytes = n * m * 8 * element_size
    
    # 理论内存操作
    theoretical_memory_operations = 2 * tensor_size_bytes
    
    print(f"PyTorch Tensor dimensions: {n}x{m}x8 (int32) on {device}")
    print(f"Total tensor size: {tensor_size_bytes / (1024*1024):.2f} MB")
    
    # 预先执行代码以设置环境
    exec(setup_code)
    locals_dict = {}
    exec(setup_code, globals(), locals_dict)
    
    # 首次预热（不计入测量）
    exec(test_code, globals(), locals_dict)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 手动测量时间，避免timeit的额外开销
    times = []
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        exec(test_code, globals(), locals_dict)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / num_runs
    
    # 计算理论内存带宽
    bandwidth = theoretical_memory_operations / (avg_time * 1024 * 1024 * 1024)  # GB/s
    
    print(f"Average time to transpose a {n}x{m}x8 PyTorch tensor over {num_runs} runs: {avg_time:.6f} seconds")
    print(f"Theoretical memory bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    print(f"Benchmark started by: tie-pilot-qxw")
    print(f"Date and Time: 2025-03-30 08:19:22 (UTC)")
    print("\n")
    
    # 设置较小的运行次数对于大规模测试
    small_runs = 20
    medium_runs = 10
    large_runs = 5
    
    print("=" * 50)
    print("NUMPY BENCHMARK")
    print("=" * 50)
    
    # print("Small array test (100x100x8):")
    # benchmark_numpy_transpose(100, 100, small_runs)
    
    # print("\nMedium array test (1000x1000x8):")
    # benchmark_numpy_transpose(1000, 1000, medium_runs)
    
    print("\nLarge array test (2000x2000x8):")
    # benchmark_numpy_transpose(2**4, 2**26, large_runs)
    
    print("\n" + "=" * 50)
    print("PYTORCH CPU BENCHMARK")
    print("=" * 50)
    
    # print("Small tensor test (100x100x8):")
    # benchmark_pytorch_transpose(100, 100, small_runs)
    
    # print("\nMedium tensor test (1000x1000x8):")
    # benchmark_pytorch_transpose(1000, 1000, medium_runs)
    
    print("\nLarge tensor test (2000x2000x8):")
    benchmark_pytorch_transpose(2**4, 2**26, large_runs)
    
    # # 如果有CUDA设备，也测试GPU
    # if torch.cuda.is_available():
    #     print("\n" + "=" * 50)
    #     print("PYTORCH CUDA BENCHMARK")
    #     print("=" * 50)
        
    #     print("Small tensor test (100x100x8):")
    #     benchmark_pytorch_transpose(100, 100, small_runs, device='cuda')
        
    #     print("\nMedium tensor test (1000x1000x8):")
    #     benchmark_pytorch_transpose(1000, 1000, medium_runs, device='cuda')
        
    #     print("\nLarge tensor test (2000x2000x8):")
    #     benchmark_pytorch_transpose(2000, 2000, large_runs, device='cuda')