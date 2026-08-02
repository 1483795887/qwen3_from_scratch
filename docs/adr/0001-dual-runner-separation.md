# 双 Runner 分离：BatchRunner 与 PackedRunner

将推理架构完全分离为 `BatchRunner`（单请求 BHSD）和 `PackedRunner`（多请求变长 SHD），各自构建独立的模型实例和组件配置。模型内部不做 Batch/Packed 模式分发。这与 `data/docs/paged_attention_migration.md` Phase 1 原方案（将 `PagedSelfAttention` 合并进 `SelfAttention`，用 `ctx.slot_mapping` 在 attention 层做运行时分发）相反。

## Considered Options

- **合并到单类 + 运行时分发**（原方案）：`SelfAttention` 内部 `if ctx.slot_mapping is not None` 选择 batch 或 packed 路径。优点：一个模型类服务所有场景。缺点：每个 attention 层背负两套逻辑，`ModelContext` 的每个字段都隐含模式语义，测试和调试时必须同时理解两套路径。
- **双 Runner 分离**（采纳）：`BatchRunner` 配置 `FusedSelfAttention` + `PreAllocatedKVCache`，`PackedRunner` 配置 `PagedSelfAttention` + `PagedKVCache`。模型内部无模式分发。缺点：两个模型实例不共享 attention 代码路径（但 q_proj/o_proj/norm/rope/transformer_block 等共享）。

## Consequences

- `SelfAttention` / `FusedSelfAttention` 保持纯 Batch，`PagedSelfAttention` 保持纯 Packed，互不干涉。
- `Qwen3.forward` 变纯管道（`embedding → layers → norm → output`），不构建 `position_ids`、不持有 `self.rope`。
- `ModelContext` 保留为胖 dataclass（不拆类型），但删除 `position_embeddings`、`is_prefill`、`num_tokens` 等无用字段。Batch/Packed 各自只读自己需要的字段。
- 迁移文档 Phase 1 的"合并"计划作废，Phase 2-3 的 `BlockManager` / `Scheduler` / `PackedRunner` 仍然适用。
- `if ctx.use_cache`（训练 vs 推理）保留——这是合法业务开关，不是模式分发。
