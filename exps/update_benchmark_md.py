#!/usr/bin/env python3
"""从 outputs 下最新的 evalscope 测速输出提取指标，打印结果表的一行。

用法：
    uv run exps/update_benchmark_md.py
    uv run exps/update_benchmark_md.py --label "本框架 v1"
    uv run exps/update_benchmark_md.py --write   # 同时把该行写入 docs/benchmark.md

默认只打印，方便直接复制到 docs/benchmark.md；--write 时才修改文档。
--label 指定版本标签：写表时若已有同标签行则原位更新，否则追加到表格末尾。
"""
import argparse
import json
import sys
from pathlib import Path

PREFERRED_RUN = "parallel_20_number_200"


def find_run_dir(out_root: Path) -> Path:
    """定位含 benchmark_summary.json 的运行目录，优先 parallel_20_number_200。"""
    if not out_root.exists():
        sys.exit(f"错误：输出目录不存在 {out_root}")
    candidates = sorted(out_root.rglob("benchmark_summary.json"))
    if not candidates:
        sys.exit(f"错误：{out_root} 下未找到 benchmark_summary.json")
    for c in candidates:
        if PREFERRED_RUN in c.parent.name:
            return c.parent
    return candidates[0].parent


def find_latest_run(outputs_root: Path) -> Path:
    """取 outputs 下最近修改的测速输出目录（如 outputs/20260822_115825）。"""
    dirs = [p for p in outputs_root.iterdir() if p.is_dir()]
    if not dirs:
        sys.exit(f"错误：{outputs_root} 下没有测速输出目录")
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return find_run_dir(latest)


def load_metrics(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "benchmark_summary.json").read_text())
    percentile = json.loads((run_dir / "benchmark_percentile.json").read_text())
    p50 = next(x for x in percentile if x["Percentiles"] == "50%")
    p99 = next(x for x in percentile if x["Percentiles"] == "99%")
    total = summary["Total Requests"]
    success = summary["Success Requests"] / total * 100 if total else 0.0
    return {
        "output": summary["Output Throughput (tok/s)"],
        "req": summary["Req Throughput (req/s)"],
        "lat": summary["Avg Latency (s)"],
        "lat_p50": p50["Latency (s)"],
        "lat_p99": p99["Latency (s)"],
        "ttft": summary["Avg TTFT (ms)"],
        "ttft_p50": p50["TTFT (ms)"],
        "tpot": summary["Avg TPOT (ms)"],
        "tpot_p99": p99["TPOT (ms)"],
        "success": success,
    }


def fmt_row(label: str, m: dict) -> str:
    pct = f"{m['success']:.1f}".rstrip("0").rstrip(".") + "%"
    cells = [
        label,
        f"{m['output']:.2f}",
        f"{m['req']:.2f}",
        f"{m['lat']:.3f}",
        f"{m['lat_p50']:.2f}",
        f"{m['lat_p99']:.2f}",
        f"{m['ttft']:.1f}",
        f"{m['ttft_p50']:.1f}",
        f"{m['tpot']:.1f}",
        f"{m['tpot_p99']:.1f}",
        pct,
    ]
    return "| " + " | ".join(cells) + " |"


def update_table(md: Path, label: str, row: str) -> tuple[str, str]:
    text = md.read_text()
    had_newline = text.endswith("\n")
    lines = text.splitlines()
    header_idx = next(
        (i for i, l in enumerate(lines) if l.startswith("|") and "版本" in l), None
    )
    if header_idx is None:
        sys.exit(f"错误：{md} 中未找到结果表（缺少含「版本」的表头）")
    data_idx = []
    i = header_idx + 2
    while i < len(lines) and lines[i].startswith("|"):
        data_idx.append(i)
        i += 1

    for i in data_idx:
        if lines[i].split("|")[1].strip() == label:
            lines[i] = row
            content = "\n".join(lines)
            return "替换", content + ("\n" if had_newline else "")

    insert_at = data_idx[-1] + 1 if data_idx else header_idx + 2
    lines.insert(insert_at, row)
    content = "\n".join(lines)
    return "追加", content + ("\n" if had_newline else "")


def main():
    ap = argparse.ArgumentParser(
        description="提取 outputs 下最新 evalscope 测速结果，打印结果表的一行"
    )
    ap.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="测速输出根目录，默认 outputs，自动取其中最新的一次",
    )
    ap.add_argument(
        "--label", default="本框架", help="结果表中的版本标签，默认「本框架」"
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="把结果行写入 --markdown 指定的文件（默认只打印）",
    )
    ap.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/benchmark.md"),
        help="--write 时更新的 markdown 文件",
    )
    args = ap.parse_args()

    run_dir = find_latest_run(args.outputs)
    row = fmt_row(args.label, load_metrics(run_dir))
    print(row)
    print(f"# 数据来源: {run_dir}")

    if args.write:
        action, content = update_table(args.markdown, args.label, row)
        args.markdown.write_text(content)
        print(f"[{action}] {args.markdown}")


if __name__ == "__main__":
    main()
