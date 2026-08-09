# LLMEngine.generate_stream 性能指标内嵌：yield StreamChunk 替代 str

将 `LLMEngine.generate_stream` 的返回类型从 `AsyncIterator[str]` 改为 `AsyncIterator[StreamChunk]`，每个 yield 单元内嵌 `PerfMetrics` 性能指标快照。指标全部基于 consumer 侧 wall-clock 测量（TTFT、Effective TPS），不从 `ModelWorker` 进程传回时间戳。

## Considered Options

- **改 yield 类型为富对象**（采纳）：`generate_stream` yield `StreamChunk(delta, metrics)`，每个 chunk 带运行时指标。优点：指标随流自然交付，调用方无需额外代码。缺点：breaking change，现有调用方需改用 `.delta`。
- **新增并行方法 `generate_stream_with_metrics`**：原方法不变，另加带指标版本。零破坏性，但两条路径维护成本高，且原方法永远缺少可观测性。
- **回调 / 上下文对象**：`generate_stream` 保持 yield `str`，调用方传入 callback(metrics)。不破坏迭代器协议，但推拉范式混合，且 callback 时序与 yield 时序的关系不直观。
- **流结束后返回汇总**：stream 跑完后从 engine 读 `last_request_metrics`。最简单，但无法实时获取指标，且多请求并发时状态管理复杂。

## 指标定义的权衡

TTFT 和 TPS 均采用 **wall-clock**（consumer 侧测量）而非 **compute-side**（Worker 进程内测量）：

- Wall-clock TTFT 包含 asyncio 排队、tokenize、Scheduler 等待、`mp.Queue` IPC、GPU 计算的全部开销——这是调用方真实感受到的延迟。
- Wall-clock Effective TPS 包含批处理干扰（其他请求共享 GPU）和 IPC 开销——这是调用方实际获得的吞吐。
- Compute-side 指标更纯净（只反映模型性能），但需要跨进程传时间戳，实现复杂度高，且不是调用方关心的数字。

本项目目标是调用方可观测性，不是模型性能剖析（后者由 `examples/benchmark.py` + `torch.profiler` 覆盖）。因此选 wall-clock。

## Consequences

- **Breaking change**：`generate_stream` 的 yield 类型从 `str` 变为 `StreamChunk`。所有调用方（如 `examples/llm_runner.py`）需改为 `chunk.delta` 而非直接用迭代变量。
- **仅 LLMEngine**：`BatchRunner.generate_stream` 保持 yield `str` 不变。两个引擎接口不一致，但 `BatchRunner` 的性能剖析由 `examples/benchmark.py` 外部计时覆盖，场景不同。
- **指标不含 compute-side 数据**：调用方无法从 `PerfMetrics` 区分「GPU 计算慢」和「IPC/调度慢」。如需细分，需另建 Worker 进程内的剖析机制（out of scope）。
- **多请求干扰**：并发请求的 Effective TPS 会因批处理共享 GPU 而降低。这是 wall-clock 指标的固有特性，反映真实服务吞吐。
- **第一个 chunk 的 `tps` 为 `0.0`**：只有 prefill 没有 decode，TPS 无意义。调用方需处理此边界。
