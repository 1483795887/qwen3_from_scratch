# Qwen3 From Scratch — 推理架构

从零实现的 Qwen3 模型推理框架，支持 Batch 单请求和 Packed 多请求变长两种推理模式。

## Language

### 推理模式

**BatchRunner**:
单请求推理引擎，继承自原 `InferenceEngine`。使用 BHSD 张量布局、`PreAllocatedKVCache`、SDPA/Triton flash attention。接口：`prefill` / `step` / `generate` / `generate_stream`。
_Avoid_: InferenceEngine, InferenceSession

**AsyncLLM**:
高层异步推理引擎，面向调用方。内部驱动后台 `mp.Process`（`EngineCoreProc`，含 `EngineCore` + `Scheduler` + `PagedKVCache`）+ ZMQ PUSH/PULL IPC 完成多请求推理。接口：`generate_stream`（async，yield `StreamChunk`）/ `warmup` / `close`。
_Avoid_: InferenceEngine, Engine, LLMServer, LLMEngine（已废弃）

**LLM**:
进程内同步推理引擎，封装 `EngineCore` 与 `InProcClient`，所有推理逻辑在当前进程完成。接口全部同步：`generate` / `batch_generate` / `warmup`。无需 `close`，python 进程结束即自然退出。
_Avoid_: LocalEngine, DirectEngine, SyncLLMEngine, SyncEngine（已废弃）

**BatchRunner**:
单请求 Batch 模式引擎（保留作 `examples/basic_generation.py` 的简单示例）。使用 BHSD 张量布局、`PreAllocatedKVCache`、SDPA/Triton flash attention，不兼容 `paged_attn` 组件，不走 Packed/批处理路径。新功能以 `LLM` / `AsyncLLM` 为准。
_Avoid_: InferenceEngine, InferenceSession

**PackedRunner**:
多请求变长推理引擎（待实现）。使用 SHD 张量布局、`PagedKVCache`、`VarLenPagedAttn`。接口：`add_request` / `step` / `generate`。
_Avoid_: PagedInferenceEngine

**Runner**:
泛指 `BatchRunner` 或 `PackedRunner`。各自独立构建模型、配置组件、管理 context，不共享接口基类。

### 张量布局

**BHSD**:
`(Batch, Head, Seq, Dim)` 四维布局。Batch 模式使用，attention 输入经 transpose 后为此格式。
_Avoid_: BSHD, BHSD transposed

**SHD**:
`(Seq, Head, Dim)` 三维布局，又称 THD。Packed 模式使用，多条序列的 token 拍平在 Seq 维（T = total_tokens）。
_Avoid_: THD（与 SHD 同义，统一用 SHD）

### KV Cache

**PreAllocatedKVCache**:
Batch 模式专用。预分配 `(B, max_len, H, D)` 连续内存，按 `cache_position` 顺序写入。
_Avoid_: SimpleKVCache（已废弃）

**PagedKVCache**:
Packed 模式专用。按 `block_size` 分页分配，`slot_mapping` 映射 token → 物理槽位，`block_tables` 映射序列 → 物理页。通过 `get(layer_idx)` 返回整层缓存，attention 间接寻址读取。
_Avoid_: block pool, page table（这些是实现细节，不是领域术语）

**BlockManager**:
`PackedRunner` 的页面管理器。负责物理页的分配、回收、构建 `block_tables`。不关心请求状态和调度策略。

**Scheduler**:
`PackedRunner` 的调度器。管理请求队列，决定每步处理哪些请求（prefill/decode/分段 prefill），构建 `StepMetadata`。

**SchedulerDriver**:
共享调度驱动。封装「入队新序列 → `Scheduler.schedule()` → worker 推理 → `post_process` 回填」的调度循环本体，供旧 `LLMEngine`（进程路径，`worker_forward` 走 `mp.Queue` 往返）与旧 `SyncEngine`（进程内直调 `ModelWorker.forward`）共用。新架构中 `AsyncLLM` 使用 `EngineCoreProc`（子进程内置调度循环），`LLM` 使用 `InProcClient` 直接驱动 `EngineCore.step()`，均不再依赖 `SchedulerDriver`。
_Avoid_: EngineLoop, Driver（太宽）, SyncLoop

**StepMetadata**:
一步前向所需的全部张量元数据，由 Scheduler 构建。包含 `input_ids`、`position_ids`、`slot_mapping`、`block_tables`、`cum_seq_lens_q/kv`。

### 位置编码

**RotaryEmbedding**:
预计算 cos/sin buffer 的 RoPE 模块。`__init__` 时按 `max_position_embeddings` 一次性计算 `cos_sin_cache (max_position, 2, D)`，`register_buffer` 持有。`forward(positions, query, key)` 按位置索引应用。
_Avoid_: PythonRope, MyRope（这些是 apply 逻辑的包装，不是预计算的持有者）

**get_rope**:
`lru_cache` 装饰的工厂函数。相同参数只创建一个 `RotaryEmbedding` 实例，全局共享。所有 RoPE 消费者（`PythonRope`、`MyRope`、`FusedSelfAttention`、`PagedSelfAttention`）通过它获取预计算 buffer。

### 组件配置

**ComponentConfig**:
组件实现的运行时加载决策。由 `name`（注册名，如 `"base"`/`"my_op"`）和 `kwargs`（实现特定参数）组成。`ModelConfig` 持有 6 个 `ComponentConfig` 字段（`self_attn`/`mlp`/`norm`/`attn`/`rope`/`decoder_layer`），默认全 `"base"`（HuggingFace 兼容）。
组件覆写发生在 `ModelLoader.load` 的 API 层，叠加在从 `config.json` 读出的架构参数之上——`config.json` 只存架构参数（dims/layers/heads），不存组件实现选择。
`ModelLoader.load` 接受 `components: Optional[Dict[str, ComponentConfig]]`，逐字段覆盖 `load_from_file` 读出的默认值，并对字段名和实现名做严格校验（fail fast）。
组件默认值当前为 "base"。ADR 0001 要求 BatchRunner 使用 `FusedSelfAttention`（注册名 `"my_op"`），该默认值待未来配置文件机制落地后切换。
_Avoid_: 组件实现持久化进 config.json（架构参数与组件实现分离）

**ComponentFactory**:
组件注册表 + 工厂。`@register(component_type, name)` 装饰器注册实现类，`create(component_type, config, ...)` 按 `ComponentConfig.name` 查表实例化。是 `ComponentConfig` 的实现机制，不是领域概念。

### 配置加载

**BatchConfig**:
Batch 模式的多模型配置文件。YAML 格式，顶层含全局 `generation` 默认值和 `models` 列表。通过 `load_batch_config(config_path)` 加载，返回 `BatchConfig` 实例。加载时全量校验所有模型条目（name 唯一、path 存在、components 注册名有效、max_len 合法等）。
_Avoid_: config.json（模型架构参数文件，不混淆）

**ModelEntry**:
`BatchConfig.models` 列表中的一个模型条目，未合并状态。持有 `name`（唯一标识符）、`path`（模型目录）、`device`、`max_len`、`components`（组件覆写）、`generation`（`Optional`，模型级覆盖）。
_Avoid_: model config（混淆 ModelConfig）

**ResolvedModelEntry**:
`BatchConfig.get_model(name)` 返回的已合并对象。全局 `GenerationDefaults` 与模型级 `GenerationOverrides` 已深度合并，所有字段必填无歧义。Runner 构建只消费 `ResolvedModelEntry`，不关心合并过程。
_Avoid_: ModelEntry（未合并，不直接用于构建 Runner）

**GenerationDefaults / GenerationOverrides**:
全局默认 / 模型级覆盖，字段相同（`temperature` / `top_k` / `top_p` / `do_sample` / `max_new_tokens`）。合并方式为深度合并——模型级只覆盖显式声明的字段，未声明的继承全局。优先级链：运行时参数 > 模型级覆盖 > 全局默认 > 模型目录的 `generation_config.json`。

**PackedConfig**:
Packed 模式的多模型配置文件（待实现）。与 `BatchConfig` 完全分离，不共享基类。将含 `kv_cache_memory_utilization` 等 Packed 模式专用推理参数。
_Avoid_: BatchConfig（两种配置各自独立）

### 性能指标

**StreamChunk**:
`AsyncLLM.generate_stream` 的 yield 单元。包含 `delta`（解码文本片段）、`req_id`、`prompt_tokens`、`generated_tokens`。
_Avoid_: RequestResult（旧 LLMEngine 内部消息传递单元，已废弃）, Chunk, TokenResult

**PerfMetrics**:
单次请求的运行时性能指标快照（旧 `LLMEngine` / `SyncEngine` 使用，随旧文件一起删除，未来可在新架构下重新引入）。字段：`ttft`（首词延迟，秒）、`token_count`（已生成 token 数）、`tps`（解码阶段 running 平均 tokens/sec）、`total_elapsed`（从 `generate_stream` 调用到此 chunk 的总耗时，秒）。全部基于 consumer 侧 wall-clock 测量。
_Avoid_: PerfStats, Metrics, TimingInfo

**BatchPerfMetrics**:
旧 `SyncEngine.batch_generate` 返回的批量汇总指标（随旧文件一起删除，未来可在新架构下重新引入）。含 `per_request`（每请求 `PerfMetrics`，顺序与输入一致）、`num_requests`、`total_tokens`、`total_elapsed`（最后一个请求结束时间 - 批量调用开始时间）、`aggregate_tps`（`total_tokens / total_elapsed`，整批吞吐）。
_Avoid_: BatchMetrics, PerfSummary

**Wall-clock TTFT**:
从调用方发出请求到收到第一个 `StreamChunk.delta` 的时间。异步路径（AsyncLLM）包含 asyncio Queue 排队、tokenize、Scheduler 等待、ZMQ IPC、GPU prefill 计算、回传的全部开销；同步路径（LLM）无 IPC，主要是 tokenize + Scheduler + GPU prefill。均为调用方真实感受到的延迟。
_Avoid_: Compute TTFT（仅 Worker 进程内的纯计算时间，本项目不采集）, Prefill time

**Effective TPS**:
解码阶段 running 平均速度 = `(token_count - 1) / (最后一个 delta 时间 - 第一个 delta 时间)`。包含批处理干扰（其他请求占用 GPU）和 IPC 开销。第一个 chunk 的 TPS 为 `0.0`（尚无 decode 阶段）。
_Avoid_: Pure decode TPS（仅模型 forward 时间的倒数，本项目不采集）, Instantaneous TPS

### 核心原则

**模型纯计算**:
模型内部（`Qwen3.forward`、各 attention 层、RoPE）不检查 context 字段是否为空再兜底构建。所有 context 字段由 Runner 在调 `model()` 前预设好。缺失即 `assert` 失败，不内联补建。
_Avoid_: fallback, lazy init, 兜底

**Runner 拥有栈**:
每个 Runner 独立构建整个技术栈——引擎循环、`ComponentFactory` 组件配置、模型实例。通过修改 `ModelConfig` 组件字段选择不同 attention/cache 实现。模型内部无 Batch/Packed 模式分发。

**合法业务开关 vs 防御性兜底**:
`if ctx.use_cache` 是合法业务开关（训练 vs 推理），保留。`if ctx.position_embeddings is None` 是防御性兜底（模型替外部擦屁股），删除。区别：前者是两个一等行为的分叉，后者是外部未尽责时的补丁。

### KV 来源判定

**block_tables 判定**:
`VarLenPagedAttn` 根据 `ctx.block_tables` 是否为空判定 KV 来源。空 → 首次 prefill，直接用传入的 k/v。非空 → 从 `PagedKVCache` 读全量 KV。先 `kv_cache.update()` 写入再从 cache 读。
_Avoid_: is_prefill 判定（已废弃，后续连续批处理时可能以其他形式回来）

### Triton Paged Attention

**flash_attn_varlen_func**:
Triton 分页注意力的 prefill 入口，host 函数，接口对齐 flash-attn（同名同参）。负责多请求变长打包 + 分页/连续 KV 读取 + flash 注意力数值。Triton 专属加载器（`load_paged_memory` / `load_contiguous_memory`）是其内部实现，不是领域概念。
_Avoid_: 直接叫 paged_attn（那是文件/模块名，语义太宽）

**block_table**:
Triton 接口参数（单数），形状 `(B, ceil(max_seqlen_k / block_size))`。与领域内 `block_tables`（复数，`ModelContext` / `BlockManager`）同物，接口层对齐 flash-attn 的命名。
_Avoid_: 在 Triton 接口签名里写 block_tables（与 flash-attn 不一致）

**分段 prefill**:
请求已有缓存 KV（`generated_len > 0`，`cu_seqlens_q != cu_seqlens_kv`）时继续追加 prefill token。因果掩码按绝对位置计算：query 位置 `pos` 只能 attend `kv < pos`。与首 prefill（`generated_len = 0`）是两种不同的 prefill 形态。
