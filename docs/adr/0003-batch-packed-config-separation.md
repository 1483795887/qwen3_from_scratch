# Batch 与 Packed 配置文件完全分离

`BatchConfig` 和 `PackedConfig` 是两套独立的数据类和加载函数，不共享基类。这与 ADR 0001（双 Runner 分离）一脉相承。

## Considered Options

- **共享基类**：模型路径、组件配置等共同字段抽到基类（如 `ModelEntry`），Batch 和 Packed 各自继承并加自己的推理字段。优点：DRY。缺点：引入"哪些字段共有、哪些子类特有"的耦合讨论，而两种 Runner 的消费方式完全不同（BatchRunner 预分配连续内存，PackedRunner 要算显存预算 + 分页）。
- **完全分离**（采纳）：`BatchConfig` / `PackedConfig` 各自独立，各自的字段、各自的加载函数。与"Runner 拥有栈、各自独立构建、不共享接口基类"的核心原则一致。
- **单一文件 + 模式开关**：配置文件里有 `mode: "batch" | "packed"` 字段，框架按 mode 走不同分支。缺点：一个文件混入两种模式的字段，校验逻辑要按模式分支，增加了不必要的复杂度。

## Consequences

- `BatchConfig` 当前只含 Batch 模式需要的字段（`max_len` 等），不含 `kv_cache_memory_utilization` 等 Packed 专用参数。
- `PackedConfig` 待 PackedRunner 实现时同步设计，将含显存预算、`block_size` 等 Packed 专用参数。
- 两种配置文件的加载函数（`load_batch_config` / `load_packed_config`）完全独立，不共享代码路径。
- 如果未来发现两种配置有大量重复逻辑，可以再考虑提取公共工具函数（不是基类），但当前不预设。
