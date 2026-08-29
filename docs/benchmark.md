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

脚本会在开跑前把实验现场存档到 `outputs/<时间戳>/`，用于事后复现每次测速对应的参数与环境（`batch2.yaml` 不入 git，且可能在中途被改过）：

- `batch2.yaml` — 服务端配置副本
- `git_state.txt` — 当前分支、commit、未提交改动（`git diff`，不含 uv.lock）
- `nvidia_smi.txt` — GPU 时钟/温度/显存与占用显存的进程（排查孤儿进程干扰）
- `server_process.txt` — 当时在跑的服务进程及其启动命令（可能为空，如测 vLLM 时）

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
| 4edde7d76db8f5e4b59b4eaaf4036206199a2384 | 726.26 | 1.42 | 14.089 | 14.10 | 14.38 | 14.1 | 12.0 | 27.5 | 28.1 | 100% |
| 3ba6d5dee98d848295792f5012ec0f974c3e8bd2 | 796.20 | 1.56 | 12.850 | 12.92 | 13.71 | 12.4 | 12.2 | 25.1 | 26.8 | 100% |
| f8cd8100eb64204ff6dc394702e7db2e3ec4737b | 863.98 | 1.69 | 11.843 | 11.88 | 12.29 | 11.1 | 10.3 | 23.1 | 24.0 | 100% |
| 8fcdd87d16e980564cced4cdf380cd9e7f3ca580 | 980.60 | 1.92 | 10.433 | 10.39 | 11.08 | 15.2 | 12.3 | 20.4 | 21.6 | 100% |
| 2ebeac0af312640aa4f28466d0b14176a670b649 | 1089.31 | 2.13 | 9.394 | 9.35 | 10.20 | 13.8 | 11.6 | 18.4 | 19.9 | 100% |
| f557033f85c172e5292f2190085c34e222bd49c6 | 1092.26 | 2.13 | 9.368 | 9.35 | 9.88 | 196.2 | 189.3 | 17.9 | 18.3 | 100% |
| 776978d | 1396.37 | 2.73 | 7.328 | 7.32 | 7.74 | 176.3 | 168.9 | 14.0 | 14.3 | 100% |
| abe7ff2 | 1423.28 | 2.78 | 7.189 | 7.19 | 7.53 | 177.3 | 171.2 | 13.7 | 14.0 | 100% |

> 049da8c311852443b3cf1b4d4604223b02fe73ea 之前的测试结果结果没有返回usage，深度思考部分没有被统计进去，吞吐量估算偏低，但相对的改进是有效的

> f8cd8100eb64204ff6dc394702e7db2e3ec4737b 及之后使用 90% 的显存水位，预留10%给解码使用

> f557033f85c172e5292f2190085c34e222bd49c6 之前的TTFT计算都是不正确的，直接忽略

vLLM 基准为单次测速（总时长 71.14s，共生成 102,400 tokens），原始数据在本机 `outputs/<时间戳>/qwen3-0.6b/` 下（该目录不入库）。
