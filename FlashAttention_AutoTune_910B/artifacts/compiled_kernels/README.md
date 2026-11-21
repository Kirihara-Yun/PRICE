# 预编译 FlashAttention 内核

本目录保存 `results/predictions/fa_dsv4_topk.json` 中每个序列长度 top-3
内核配置对应的离线编译产物。共 7 个序列长度、每长度 3 个候选，合计 21 个
内核；每个内核目录包含：

```text
fa_seq<seq>_bm<block_M>_bn<block_N>_s<num_stages>/
├── artifact.json          # 加载与复现所需元数据
├── kernel.so              # 已编译动态库
├── kernel_source.c        # TileLang 生成的全局内核源码
└── kernel_wrapper.cpp     # cython 运行包装源码
```

## 1. 前置条件

- 在 Ascend 910B/910B4 宿主机上运行内核，确保 `torch_npu` 可用：

  ```bash
  npu-smi info
  python3 -c "import torch, torch_npu, tilelang; print(torch.__version__)"
  ```

- 重新编译时才需要本机 CANN 与 `bisheng` 编译器。

## 2. 直接运行已编译内核

查看全部预编译内核：

```bash
cd /root/FlashAttention_AutoTune_910B
python3 tools/run_precompiled_kernel.py --list
```

运行单个内核并测量时延：

```bash
python3 tools/run_precompiled_kernel.py \
  --artifact artifacts/compiled_kernels/fa_seq4096_bm64_bn128_s2/artifact.json \
  --benchmark --warmup 1 --repeat 2
```

加入数值校验：

```bash
python3 tools/run_precompiled_kernel.py \
  --artifact artifacts/compiled_kernels/fa_seq4096_bm64_bn128_s2/artifact.json \
  --benchmark --check
```

## 3. 批量时延测试

只测试历史最优配置（每个 seq_len 一个内核）：

```bash
python3 tools/benchmark_precompiled_kernels.py
```

测试全部 21 个 top-3 预编译内核，并自动取每个 seq_len 时延最低者：

```bash
python3 tools/benchmark_precompiled_kernels.py --all
```

输出：

```text
results/benchmark/fa_precompiled_latency.json
results/benchmark/fa_precompiled_latency.csv
results/benchmark/fa_precompiled_latency.md
```

## 4. 重新编译

```bash
python3 tools/compile_selected_kernels.py
```

可选参数：

```text
--prediction  预测文件，默认 results/predictions/fa_dsv4_topk.json
--seq-list    只编译指定 seq_len
--target      默认 ascendc
--platform    910B 默认 A2
```

## 5. 配置说明

每个序列长度的 top-3 配置来自性能模型预筛，格式为：

```text
block_M x block_N, num_stages
```

例如 `fa_seq4096_bm64_bn128_s2` 表示 `S=4096`、
`block_M=64`、`block_N=128`、`num_stages=2`。
