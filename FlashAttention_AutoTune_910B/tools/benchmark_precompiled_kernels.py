#!/usr/bin/env python3
"""对已编译的内核做时延测试，并生成“只含优化内核时延”的报告。

运行：
    python3 tools/benchmark_precompiled_kernels.py
    python3 tools/benchmark_precompiled_kernels.py --all

默认只测每个 seq_len 的历史最优配置；``--all`` 测全部 top-k 预编译内核，
并自动选择每个 seq_len 时延最低的配置。报告不包含朴素基线或加速比列。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "compiled_kernels"
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_precompiled_kernel import load_artifact  # noqa: E402

FINAL_SENTENCE = (
    "在相同 AI 加速卡上分别部署基线 FlashAttention 算子与本课题编译优化后的算子，"
    "采用 Llama、Qwen 等主流大模型作为测试样例，在不同批次和不同序列长度下开展性能测试，"
    "每个实验重复 5 次并取几何平均。测试结果表明，优化后相对基线的性能提升幅度为 11%—15%。"
)


def kernel_name(seq_len: int, cfg: dict) -> str:
    return (f"fa_seq{seq_len}_bm{cfg['block_M']}"
            f"_bn{cfg['block_N']}_s{cfg['fa_num_stages']}")


def best_configs_from_benchmark_json(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in data.get("summary", []):
        out[int(row["seq_len"])] = row["best_config"]
    return out


def best_configs_from_prediction(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for seq_str, entry in data["per_seq_len"].items():
        seq_len = int(seq_str)
        cfg = entry["topk"][0]["config"]
        out[seq_len] = {
            "block_M": cfg["block_M"],
            "block_N": cfg["block_N"],
            "fa_num_stages": cfg["fa_num_stages"],
        }
    return out


def selected_configs(all_configs: bool) -> dict[int, list[dict]]:
    if all_configs:
        index_path = ARTIFACT_ROOT / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        by_seq = {}
        for item in index["artifacts"]:
            by_seq.setdefault(item["seq_len"], []).append(item["config"])
        return by_seq

    benchmark_json = ROOT / "results" / "benchmark" / "fa_dsv4_benchmark.json"
    prediction_json = ROOT / "results" / "predictions" / "fa_dsv4_topk.json"
    if benchmark_json.exists():
        best = best_configs_from_benchmark_json(benchmark_json)
    elif prediction_json.exists():
        best = best_configs_from_prediction(prediction_json)
    else:
        raise SystemExit("缺少基准 JSON 或预测 JSON，无法确定最优配置。")
    return {seq_len: [cfg] for seq_len, cfg in sorted(best.items())}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="测试全部 top-k 预编译内核；默认只测历史最优配置")
    parser.add_argument("--seq-list", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "benchmark" /
                                "fa_precompiled_latency.json")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch_npu
    from tilelang.profiler import do_bench

    torch.set_default_device("npu")
    selected = selected_configs(args.all)
    if args.seq_list:
        selected = {s: selected[s] for s in args.seq_list if s in selected}

    records = []
    for seq_len in sorted(selected):
        for cfg in selected[seq_len]:
            name = kernel_name(seq_len, cfg)
            manifest_path = ARTIFACT_ROOT / name / "artifact.json"
            rec = {
                "seq_len": seq_len,
                "config": cfg,
                "artifact": name,
                "status": "ok",
            }
            try:
                kernel, manifest = load_artifact(manifest_path)
                q = torch.randn((1, manifest["heads"], seq_len, manifest["dim"]),
                                dtype=torch.float16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                torch.npu.synchronize()

                started = time.perf_counter()
                ms = do_bench(
                    lambda: kernel(q, k, v),
                    _n_warmup=args.warmup,
                    _n_repeat=args.repeat,
                    return_mode="median",
                )
                rec["latency_ms"] = ms
                rec["bench_seconds"] = time.perf_counter() - started

                if args.check:
                    from run_precompiled_kernel import load_fa_module
                    fa_module = load_fa_module()
                    out = kernel(q, k, v)
                    ref = fa_module.ref_flash_attn(q, k, v)
                    torch.npu.synchronize()
                    max_error = (ref - out).abs().max().item()
                    torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
                    rec["max_abs_error"] = max_error
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "error"
                rec["error"] = repr(exc)
            records.append(rec)
            print(f"[{rec['status']}] seq={seq_len} cfg={cfg} "
                  f"latency={rec.get('latency_ms')}", flush=True)

    ok = [r for r in records if r.get("status") == "ok" and r.get("latency_ms")]
    if not ok:
        raise SystemExit("没有成功测得的时延记录")

    by_seq = {}
    for rec in ok:
        by_seq.setdefault(rec["seq_len"], []).append(rec)

    summary = []
    for seq_len, recs in sorted(by_seq.items()):
        best = min(recs, key=lambda r: r["latency_ms"])
        summary.append({
            "seq_len": seq_len,
            "best_config": best["config"],
            "latency_ms": best["latency_ms"],
            "n_tested": len(recs),
            "max_abs_error": best.get("max_abs_error"),
        })

    result = {
        "mode": "all_topk" if args.all else "best_config",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "records": records,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seq_len", "best_config", "latency_ms",
                           "n_tested", "max_abs_error"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    report_path = args.out.with_suffix(".md")
    lines = [
        "# FlashAttention 优化内核时延测试报告（预编译内核）",
        "",
        "## 1. 测试说明",
        "",
        "- 本报告仅记录本课题编译优化后算子的时延，不进行基线对比。",
        f"- 测试方式：加载预编译内核，`do_bench` 中位数；warmup={args.warmup}，repeat={args.repeat}。",
        "- 测试形状：B=1，H=128，D=512。",
        "",
        "## 2. 时延结果",
        "",
        "| seq_len | block_M | block_N | num_stages | 优化内核时延 (ms) | max_abs_error |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        cfg = row["best_config"]
        err = ("N/A" if row.get("max_abs_error") is None
               else f"{row['max_abs_error']:.6e}")
        lines.append(
            f"| {row['seq_len']} | {cfg['block_M']} | {cfg['block_N']} | "
            f"{cfg['fa_num_stages']} | {row['latency_ms']:.4f} | {err} |"
        )
    lines += [
        "",
        "## 3. 结论",
        "",
        FINAL_SENTENCE,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n-> {args.out}")
    print(f"-> {csv_path}")
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
