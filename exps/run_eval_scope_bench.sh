#!/bin/bash
# evalscope 压测脚本。
# 每次运行的输出目录固定为 outputs/<时间戳>/，目录里除 evalscope 结果外，
# 还在开跑前存档一份实验现场（服务端配置、git 状态、GPU 快照、服务进程），
# 保证表格里的每一行结果都能复现当时的参数与环境。
#
# 用法：MODEL_PATH 必须指向 Qwen3-0.6B 权重目录（见 docs/benchmark.md）。
set -euo pipefail

RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="outputs/$RUN_TS"
mkdir -p "$OUT_DIR"

# ---- 存档实验现场（开跑前抓，配置中途被改也能对比出来）----
{
  echo "captured_at: $(date '+%F %T')"
  echo
  echo "== git =="
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "commit: $(git rev-parse HEAD)"
  echo
  echo "-- git status --"
  git status --short
  echo
  echo "-- git diff（不含 uv.lock）--"
  git diff -- . ':(exclude)uv.lock'
} > "$OUT_DIR/git_state.txt"

cp examples/configs/batch2.yaml "$OUT_DIR/"
cp examples/configs/batch2.local.yaml "$OUT_DIR/" 2>/dev/null || true

{
  echo "== nvidia-smi =="
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv
  echo
  echo "== 占用显存的进程（防孤儿进程干扰）=="
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
} > "$OUT_DIR/nvidia_smi.txt" 2>&1

# 当前在跑的服务（命令行里能看到 env 覆盖和启动参数）
ps -eo pid,etime,cmd | grep -E "examples/server\.py|vllm serve" | grep -v grep \
  > "$OUT_DIR/server_process.txt" || true

# ---- 压测本体 ----
evalscope perf \
  --url http://localhost:8887/v1/chat/completions \
  --model qwen3-0.6b \
  --api openai \
  --parallel 20 \
  --number 200 \
  --dataset random \
  --min-prompt-length 512 \
  --max-prompt-length 512 \
  --min-tokens 512 \
  --max-tokens 512 \
  --stream \
  --extra-args '{"ignore_eos":true,"enable_thinking":false}' \
  --outputs-dir "$OUT_DIR" \
  --no-timestamp \
  --tokenizer-path $MODEL_PATH
