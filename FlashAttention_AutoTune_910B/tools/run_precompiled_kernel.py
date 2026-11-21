#!/usr/bin/env python3
"""加载 ``artifacts/compiled_kernels`` 中的预编译内核并运行/测时。

使用预编译 .so 时不再重复 TileLang 编译，只加载动态库、构造运行包装器：

    python3 tools/run_precompiled_kernel.py --list
    python3 tools/run_precompiled_kernel.py \\
        --artifact artifacts/compiled_kernels/fa_seq4096_bm64_bn128_s2/artifact.json
    python3 tools/run_precompiled_kernel.py \\
        --artifact artifacts/compiled_kernels/fa_seq4096_bm64_bn128_s2/artifact.json \\
        --benchmark --warmup 1 --repeat 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "src" / "kernels"
ARTIFACT_ROOT = ROOT / "artifacts" / "compiled_kernels"


def load_fa_module():
    if str(KERNEL_DIR) not in sys.path:
        sys.path.insert(0, str(KERNEL_DIR))
    import flash_attn_tilelang_ds_v4 as fa_module
    return fa_module


def make_prim_func(fa_module, manifest: dict):
    cfg = manifest["config"]
    return fa_module.flash_attention_fwd.__jit_impl__.func(
        batch=manifest["batch"],
        seq_len=manifest["seq_len"],
        heads_q=manifest["heads"],
        heads_kv=manifest["heads"],
        dim=manifest["dim"],
        block_M=cfg["block_M"],
        block_N=cfg["block_N"],
        num_stages=cfg["fa_num_stages"],
    )


def load_artifact(manifest_path: Path):
    import tilelang
    from tilelang.engine.lower import extrac_params

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fa_module = load_fa_module()
    prim_func = make_prim_func(fa_module, manifest)
    params = extrac_params(prim_func)

    artifact_dir = manifest_path.parent
    global_source = (artifact_dir / manifest["kernel_global_source"]).read_text(
        encoding="utf-8"
    )
    lib_path = str(artifact_dir / manifest["kernel_lib"])

    kernel = tilelang.JITKernel.from_database(
        func=prim_func,
        kernel_global_source=global_source,
        kernel_lib_path=lib_path,
        params=params,
        target=manifest["target"],
        target_host=manifest.get("target_host"),
        platform=manifest["platform"],
        out_idx=manifest["out_idx"],
        workspace_idx=manifest["workspace_idx"],
        auto_gm_idx=manifest.get("auto_gm_idx", []),
        execution_backend=manifest["execution_backend"],
        pass_configs=fa_module.pass_configs,
        compile_flags=manifest.get("compile_flags"),
    )
    return kernel, manifest


def list_artifacts():
    index_path = ARTIFACT_ROOT / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for item in index["artifacts"]:
            print(item["path"])
        return
    for path in sorted(ARTIFACT_ROOT.glob("*/artifact.json")):
        print(path.relative_to(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--artifact", type=Path,
                        default=ARTIFACT_ROOT /
                                "fa_seq4096_bm64_bn128_s2" /
                                "artifact.json")
    parser.add_argument("--benchmark", action="store_true",
                        help="使用 do_bench 测量时延")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--check", action="store_true",
                        help="与 ref_flash_attn 做数值校验")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_artifacts()
        return

    import torch
    import torch_npu

    torch.set_default_device("npu")
    kernel, manifest = load_artifact(args.artifact)
    S = manifest["seq_len"]
    H = manifest["heads"]
    D = manifest["dim"]
    B = manifest["batch"]

    q = torch.randn((B, H, S, D), dtype=torch.float16)
    k = torch.randn((B, H, S, D), dtype=torch.float16)
    v = torch.randn((B, H, S, D), dtype=torch.float16)
    torch.npu.synchronize()

    out = kernel(q, k, v)
    torch.npu.synchronize()
    print(f"loaded: {args.artifact.name} -> {args.artifact.parent}")
    print(f"shape: B={B} H={H} S={S} D={D}")
    print(f"config: {manifest['config']}")
    print(f"output shape: {tuple(out.shape)}, dtype={out.dtype}")

    if args.check:
        fa_module = load_fa_module()
        ref = fa_module.ref_flash_attn(q, k, v)
        torch.npu.synchronize()
        max_error = (ref - out).abs().max().item()
        torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
        print(f"check: passed, max_abs_error={max_error:.6e}")

    if args.benchmark:
        from tilelang.profiler import do_bench

        ms = do_bench(
            lambda: kernel(q, k, v),
            _n_warmup=args.warmup,
            _n_repeat=args.repeat,
            return_mode="median",
        )
        print(f"latency: {ms:.4f} ms")


if __name__ == "__main__":
    main()
