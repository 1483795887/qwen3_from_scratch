# Triton Paged Attention 接口对齐 flash-attn

Triton 分页注意力的 prefill 入口命名为 `flash_attn_varlen_func`，参数名用 `block_table`（单数），形状 `(B, max_seqlen_k // block_size)`，与 flash-attn 官方接口完全一致。项目本身不依赖 flash-attn（pyproject 无此依赖，测试参照物是 torch SDPA），对齐的是接口而非实现。

## Considered Options

- **项目自有命名**：如 `varlen_paged_attn(q, k, v, ..., block_tables)`，参数名用领域内复数 `block_tables`（与 `ModelContext` / `BlockManager` 一致）。优点：与项目自身术语统一，不暗示外部依赖。缺点：无法直接移植 flash-attn 的使用方式，调用方需要重新学习一套参数；未来若引入真 flash-attn 对比或替换，命名形成噪音。
- **对齐 flash-attn（采纳）**：同名同参，调用写法可直接抄 flash-attn 文档。缺点：`block_table` 单数与领域复数 `block_tables` 并存，`flash_attn_varlen_func` 名字暗示外部库。已通过 `CONTEXT.md` 词条明确二者关系（`block_table` 是 Triton 接口层命名，`block_tables` 是领域层命名），避免未来被"改回项目命名"。

## Consequences

- 测试可直接照抄 flash-attn 的调用签名构造输入。
- `CONTEXT.md` 记录 `block_table`（单数，接口参数）与 `block_tables`（复数，领域概念）的同物关系。
- decode 入口 `flash_attn_with_kvcache` 后续实现时沿用同一对齐原则。
