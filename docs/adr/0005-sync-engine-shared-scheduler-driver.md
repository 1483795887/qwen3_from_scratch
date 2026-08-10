# SyncEngine：进程内同步推理，复用共享调度驱动

新增 `SyncEngine`（进程内同步推理引擎），与服务路径 `LLMEngine` 共享调度循环与推理组件，但省去运行线程、子进程和 `mp.Queue` IPC。`LLMEngine.run` 的调度循环本体被提取为 `SchedulerDriver`（`schedule → worker forward → post_process 回填`），两条路径共用，仅 `worker_forward` 注入不同：服务路径走 `mp.Queue` 往返，同步路径直调 `ModelWorker.forward`。

## Context

本地 Python 场景下，服务架构（调度线程 `threading.Thread` + 推理进程 `mp.Process`）需要一个 async 驱动、显式 `close()`，且每个 decode 步都存在 `mp.Queue` 序列化 - 反序列化开销，TTFT/TPS 被 IPC 污染，忘记 `close` 会残留进程。用户要的是「单个 python 调用、进程结束自然退出、测速走 paged_attn + packed 输入」，同时保留服务架构供后续 FastAPI。

## Considered Options

- **直接在现有 `LLMEngine` 上加同步糖衣**（未采纳）：服务代码不动，包一层 `asyncio.run` 的同步 wrapper。改动最少，但 `mp.Queue` IPC 的每步往返开销原封不动，「降低数据传输耗时」的核心诉求落空。
- **用 `BatchRunner` 作为本地路径**（未采纳）：`BatchRunner` 本来就是进程内同步，但只支持 `PreAllocatedKVCache` 的 BHSD Batch 模式，与项目 `batch2.yaml` 的 `self_attn: paged_attn`（`PagedKVCache` + `VarLenPagedAttn` 撑起 SHD/Packed）不兼容，测速路径会与实际服务不一致。
- **新建独立同步引擎，照抄调度逻辑**（未采纳）：同步路径与 `LLMEngine.run` 各维护一份调度代码，复用度低，后续修 bug 需两处同步。
- **提取共享调度驱动 `SchedulerDriver` + 同步直驱 `SyncEngine`**（采纳）：调度循环只写一份；`SyncEngine` 当前进程内直接实例化 `ModelWorker`（模型加载 + Paged cache + `forward`），无线程、无进程、无 IPC，进程结束自然退出。

## Consequences

- `examples/llm_runner.py`（服务、异步、需 `close`）与新增的同步示例并存，覆盖两场景。
- `BatchRunner` 保留，仅供 `examples/basic_generation.py` 简单演示；不作为 Paged/批处理测速路径。
- 同步 `generate_stream` 是普通 Python generator（非 async），yield 语义与 async 版的 `StreamChunk` 一致；每条 `metrics` 仍走 consumer 侧 wall-clock（继承 ADR 0004），但不再包含 IPC 开销，因此数值会更接近纯调度+GPU。
- `SyncEngine` 无 `close`：无线程、无子进程，不提供等于无需清理；进程结束即退出。若后续要加 FastAPI，服务路径 `LLMEngine` 保持可用。

## 状态

Accepted