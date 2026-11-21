

## 快速开始

### 1. 环境准备

在带 Ascend 910B4 的宿主机上，确保 CANN、torch、torch_npu、
tilelang-ascend 可用：

```bash
cd /root/FlashAttention_AutoTune_910B
python3 -c "import torch, torch_npu, tilelang; print(torch.__version__, tilelang.__version__)"
npu-smi info
```

### 2. 使用已编译内核

仓库已把性能模型预筛出的 21 个 top-3 内核配置编译为 `.so`，放在
`artifacts/compiled_kernels/`。在 910B4 宿主机上无需重新编译，可直接加载：

```bash
# 查看可用的预编译内核
python3 tools/run_precompiled_kernel.py --list

# 运行单个内核并测时
python3 tools/run_precompiled_kernel.py \
  --artifact artifacts/compiled_kernels/fa_seq4096_bm64_bn128_s2/artifact.json \
  --benchmark --warmup 1 --repeat 2

# 只测试历史最优内核，或测试全部 top-3 内核
python3 tools/benchmark_precompiled_kernels.py
python3 tools/benchmark_precompiled_kernels.py --all
```

详细说明见 `artifacts/compiled_kernels/README.md`。

## 
