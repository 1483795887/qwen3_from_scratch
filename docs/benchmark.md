# 性能测试

使用 [evalscope](https://github.com/modelscope/evalscope) 压测本框架与 vLLM 的 OpenAI 兼容 API，在相同负载下对比吞吐与延迟。

## 环境准备

vLLM 与本框架的依赖互不兼容，需要另起一个独立的虚拟环境，仅安装 vllm 与 evalscope：

```bash
# 在项目根目录创建独立虚拟环境
uv venv .venv-vllm
uv pip install --python .venv-vllm vllm evalscope \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

测速脚本通过环境变量 `MODEL_PATH` 定位 Qwen3-0.6B 权重目录，先设置好：

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
```

## 1. 启动 vLLM 服务

激活测速环境，先启动 vLLM（`exps/start_vllm.sh`，监听 `0.0.0.0:8887`）：

```bash
source .venv-vllm/bin/activate
bash exps/start_vllm.sh
```

## 2. 运行 evalscope 压测

另开终端，激活同样的环境后执行 `exps/run_eval_scope_bench.sh`：

```bash
source .venv-vllm/bin/activate
bash exps/run_eval_scope_bench.sh
```

负载为 **20 并发 × 200 请求**，每条请求 512 input + 512 output tokens，流式输出（`--extra-args '{"ignore_eos":true}'` 保证每条固定输出 512 tokens）。结果输出到 `outputs/<时间戳>/qwen3-0.6b/performance_summary.txt`。

## 3. 测本框架

启动方式见 [examples.md](examples.md) 的「OpenAI 兼容服务器」一章，使用真实模型：

```bash
uv run examples/server.py --config_path examples/configs/batch2_example.yaml --model qwen3-0.6b --use_real_model
```

服务同样监听 `0.0.0.0:8887`，evalscope 的 `--url` 无需修改。两个服务共用端口且 12GB 显存一次只能跑一个，测本框架前先停掉 vLLM。

## 测试结果

负载：20 并发 × 200 请求，512 input + 512 output tokens，流式输出，`ignore_eos`。表格每行是一次完整测速，本框架每次改进后追加一行。

| 版本 | Output 吞吐 (tok/s) | 请求吞吐 (req/s) | 平均延迟 (s) | 延迟 p50 (s) | 延迟 p99 (s) | TTFT avg (ms) | TTFT p50 (ms) | TPOT avg (ms) | TPOT p99 (ms) | 成功率 |
|---|---|---|---|---|---|---|---|---|---|---|
| vLLM 0.15.1（基准） | 1439.41 | 2.81 | 7.109 | 7.10 | 7.30 | 314.8 | 344.8 | 13.3 | 13.8 | 100% |
| 基础(6d721b803b4a8f325340af6d1862be1cb7f72e2b) | 428.68 | 0.9178 | 21.779 | 21.71 | 22.71 | 22.2 | 20.7 | 60.0 | 381.3 | 100% |
| 71aea938e4708342f725f00a19e6092df07cb450 | 638.66 | 1.37 | 14.559 | 14.51 | 15.11 | 20.8 | 21.0 | 39.2 | 195.5 | 100% |
| 16c5e826f6805165c38d766139e5d8371231ac2b | 712.81 | 1.49 | 13.397 | 13.42 | 14.21 | 21.9 | 19.6 | 33.9 | 321.7 | 100% |
| 049da8c311852443b3cf1b4d4604223b02fe73ea | 714.90 | 1.40 | 14.313 | 14.35 | 14.71 | 13.5 | 13.4 | 28.0 | 28.8 | 100% |

> 049da8c311852443b3cf1b4d4604223b02fe73ea 之前的测试结果结果没有返回usage，深度思考部分没有被统计进去，吞吐量估算偏低，但相对的改进是有效的

vLLM 基准为单次测速（总时长 71.14s，共生成 102,400 tokens），原始数据在本机 `outputs/<时间戳>/qwen3-0.6b/` 下（该目录不入库）。
