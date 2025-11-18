import torch
import time
from config import Config

def clear_memory():
    """清理GPU内存缓存"""
    if Config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def measure_time(func, *args, **kwargs):
    """测量函数执行时间（包含CUDA同步）"""
    clear_memory()
    # 热身
    for _ in range(Config.WARMUP_ITER):
        func(*args, **kwargs)
    # 测量
    start = time.time()
    for _ in range(Config.MEASURE_ITER):
        func(*args, **kwargs)
    if Config.DEVICE == "cuda":
        torch.cuda.synchronize()  # 确保CUDA操作完成
    end = time.time()
    avg_time = (end - start) / Config.MEASURE_ITER
    return avg_time  # 平均时间（秒）

def measure_memory_usage(func, *args, **kwargs):
    """测量函数执行时的内存使用峰值（MB）"""
    clear_memory()
    if Config.DEVICE == "cuda":
        start_mem = torch.cuda.max_memory_allocated() / (1024**2)
        func(*args, **kwargs)
        torch.cuda.synchronize()
        end_mem = torch.cuda.max_memory_allocated() / (1024**2)
        peak_mem = end_mem - start_mem
    else:
        # CPU内存测量（简单近似）
        import psutil
        process = psutil.Process()
        start_mem = process.memory_info().rss / (1024**2)
        func(*args, **kwargs)
        end_mem = process.memory_info().rss / (1024**2)
        peak_mem = end_mem - start_mem
    clear_memory()
    return peak_mem