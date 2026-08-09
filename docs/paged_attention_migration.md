# PagedAttention 迁移待办

> 目标：在现有 Batch 单请求代码基础上，实现 PagedAttention + 多请求变长推理。
>
> ⚠️ **架构方向已变更（见 [ADR-0001](../../docs/adr/0001-dual-runner-separation.md)）**：
> 原方案为"所有模块同时兼容 Batch 和 Packed，通过 `slot_mapping` 一键切换"。
> 新方案为"双 Runner 分离"——`BatchRunner` 和 `PackedRunner` 各自构建独立模型和组件配置，
> 模型内部不做模式分发。下方 Phase 1 的合并计划已作废，Phase 2-3 的 BlockManager / Scheduler / PackedRunner 仍然适用。
>
> 新增前置工作：RoPE 预计算重构（`get_rope` + `RotaryEmbedding`）、`ModelContext` 字段清理、
> `Qwen3.forward` 变纯管道。详见 [CONTEXT.md](../../CONTEXT.md)。

---

## 模式判定

| 状态 | `use_cache` | `slot_mapping` | 模式 | 输入形状 |
|------|-------------|----------------|------|---------|
| 训练 | `False` | `None` | Batch（无 cache） | `(B, S, H)` |
| 单请求推理 | `True` | `None` | Batch（PreAllocatedKVCache） | `(B, S, H)` |
| 多请求 Paged 推理 | `True` | `(T,)` tensor | Packed（PagedKVCache） | `(T, H)` |

判定开关（不加新字段，从现有字段推断）：

```python
is_packed = ctx.slot_mapping is not None
```

---

## 各模块改动总览

| 模块 | 改动 | 说明 |
|------|------|------|
| `rope.py` — `build_cos_sin_embed` | ✏️ 改 | einsum → outer，统一输出 `(N, D)` |
| `rope.py` — `PythonRope.forward` | ✏️ 改 | 用 `x.dim()` 判断 reshape 方式 |
| `rope.py` — `MyRope.forward` | ✏️ 小改 | 非 4D 输入 fallback 到 Python 路径 |
| `qwen3.py` — `Qwen3.forward` | ✏️ 小改 | position_ids 构建加 packed 分支 |
| `self_attn.py` — `SelfAttention` | ✏️ 改 | 新增 `_forward_packed`，`__init__` 加 `self.paged_attn` |
| `self_attn.py` — `PagedSelfAttention` | 🗑️ 废弃 | 逻辑合并进 `SelfAttention` |
| `norm.py` | ✅ 不动 | rms_norm 只操作最后一维 |
| `feedback.py` | ✅ 不动 | Linear + SiLU 只操作最后一维 |
| `transformer_block.py` | ✅ 不动 | 链式调用，布局由子模块处理 |
| `attn.py` — `TorchVarLenPagedAttn` | ✅ 不动 | 已实现，作为 packed 路径的 attention |
| `attn.py` — `TorchGQA` / `MyAttnFlash` | ✅ 不动 | Batch-only，packed 不调用 |
| `context.py` | ✅ 不动 | 字段已齐备 |
| 训练代码 `examples/train/` | ✅ 不动 | 独立于推理路径 |
| `paged_cache.py` | 🔜 后续优化 | Python for 循环 → 向量化 scatter |

---

## Phase 0：RoPE 双模式统一

> 前置条件：无。这是所有后续工作的基础。

### Task 0.1 — `build_cos_sin_embed` 统一输出 `(N, D)`

**文件**：`src/qwen3_from_scratch/models/rope.py`

**当前问题**：

```python
# einsum 硬性要求 position_ids 为 2D (1, S)，packed 模式 (T,) 会报错
freqs = torch.einsum("bj,bk->bjk", position_ids, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)  # (1, S, D)
```

**目标**：flatten position_ids，用 `torch.outer` 替代 einsum，输出统一为 `(N, D)`。

```python
def build_cos_sin_embed(self, dtype, position_ids):
    inv_freq = 1.0 / (
        self.base_freq ** (
            torch.arange(0, self.head_dim, 2, device=position_ids.device).float()
            / self.head_dim
        )
    )                                              # (D//2,) 去掉 unsqueeze
    pos = position_ids.reshape(-1).float()         # (N,) (1,S) 和 (T,) 都 flatten
    freqs = torch.outer(pos, inv_freq)             # (N, D//2)
    emb = torch.cat([freqs, freqs], dim=-1)        # (N, D)
    return PositionEmbeddings(emb.cos().to(dtype), emb.sin().to(dtype))
```

**验收**：

- [ ] Batch 模式 `position_ids=(1, S)` → `cos.shape == (S, D)`
- [ ] Packed 模式 `position_ids=(T,)` → `cos.shape == (T, D)`
- [ ] 现有训练跑通（`use_cache=False`）
- [ ] 现有 `test_rope.py` 通过

### Task 0.2 — `PythonRope.forward` 按维度 reshape

**文件**：`src/qwen3_from_scratch/models/rope.py`

**当前问题**：

```python
seq_len = x.shape[2]                           # 假设 4D BHSD，3D THD 会取到 H
emb_cos = ctx.position_embeddings.cos_embed[None, :, :]  # 假设 cos 是 3D (1,S,D)
```

**目标**：用 `x.dim()` 判断，cos 始终是 2D `(N, D)`，按需加维度。

```python
def forward(self, x):
    ctx = get_forward_context()

    # fallback：上层未预构建时自动构建（训练路径）
    if ctx.position_embeddings is None:
        if ctx.position_ids is None:
            if x.dim() == 4:          # BHSD
                ctx.position_ids = torch.arange(x.shape[2], device=x.device).unsqueeze(0)
            elif x.dim() == 3:        # THD
                ctx.position_ids = torch.arange(x.shape[0], device=x.device)
        ctx.position_embeddings = self.build_cos_sin_embed(x.dtype, ctx.position_ids)

    cos = ctx.position_embeddings.cos_embed        # (N, D) 永远 2D
    sin = ctx.position_embeddings.sin_embed

    if x.dim() == 4:                               # BHSD: (B, H, S, D)
        cos_e = cos[None, None, :, :]              # (1, 1, S, D)
        sin_e = sin[None, None, :, :]
    elif x.dim() == 3:                             # THD: (T, H, D)
        cos_e = cos[:, None, :]                    # (T, 1, D)
        sin_e = sin[:, None, :]
    else:
        raise ValueError(f"Unexpected x.dim()={x.dim()}, expected 3 or 4")

    if self.rope_type == "neox":
        return (x * cos_e) + (self._rotate_half_neox(x) * sin_e)
    elif self.rope_type == "normal":
        return (x * cos_e) + (self._rotate_normal(x) * sin_e)
    else:
        raise ValueError(f"Unknown RoPE type: {self.rope_type}")
```

**验收**：

- [ ] 4D 输入 `(B, H, S, D)` → 输出同形状，数值与改动前一致
- [ ] 3D 输入 `(T, H, D)` → 输出同形状
- [ ] `_rotate_half_neox` / `_rotate_normal` 不需改（只操作 `dim=-1`）

### Task 0.3 — `MyRope.forward` 非 4D fallback

**文件**：`src/qwen3_from_scratch/models/rope.py`

**原因**：Triton `neox_rope` kernel 硬编码取 `Q.shape[-2]` 作为 seq 维，THD 输入会取到 H 而非 T。

```python
def forward(self, x):
    ctx = get_forward_context()
    if self.rope_type == "normal" or not x.is_cuda:
        return super().forward(x)
    if x.dim() != 4:                    # ← 新增：packed 模式走 Python
        return super().forward(x)
    # 4D BHSD on CUDA: triton 路径
    cos = ctx.position_embeddings.cos_embed
    sin = ctx.position_embeddings.sin_embed
    cos_e = cos[None, None, :, :]
    sin_e = sin[None, None, :, :]
    from qwen3_from_scratch.kernels.triton.rope import neox_rope
    return neox_rope(x, cos_e, sin_e)
```

**验收**：

- [ ] CUDA 上 4D 输入仍走 Triton kernel
- [ ] CUDA 上 3D 输入 fallback 到 `PythonRope.forward`

### Task 0.4 — 清理下游对 cos.dim() 的假设

**文件**：`src/qwen3_from_scratch/models/self_attn.py`

改动后 `cos_embed` 永远是 2D，以下代码中的 `if cos.dim() == 3: cos = cos[0]` 变为死代码：

- `FusedSelfAttention._forward_pytorch` (line ~132)
- `FusedSelfAttention.forward` (line ~198)
- `PagedSelfAttention._forward_pytorch` (line ~308)

**处理方式**：可暂不删除（不影响正确性），但在 Phase 1 重构 `PagedSelfAttention` 时一并清理。

---

## Phase 1：~~Qwen3.forward + SelfAttention 双模式~~ [已作废]

> ⚠️ 本阶段原方案（将 `PagedSelfAttention` 合并进 `SelfAttention`，用 `ctx.slot_mapping` 做运行时分发）已被 ADR-0001 取代。
> 新方案下 `SelfAttention` / `FusedSelfAttention` 保持纯 Batch，`PagedSelfAttention` 保持纯 Packed，互不合并。
>
> 替代工作（已纳入新计划）：
> - `Qwen3.forward` 变纯管道（删 `position_ids` 构建、删 `position_embeddings` 构建、删 `self.rope`）
> - `PythonRope` / `MyRope` 兜底回退改为 `assert`，内部调 `get_rope` 获取预计算 cos/sin
> - `FusedSelfAttention._build_rope_embeddings` 删除，内联 RoPE 改为调 `get_rope`
> - `ModelContext` 删除 `position_embeddings`、`is_prefill`、`num_tokens`、`PositionEmbeddings`
> - 训练代码内联构建 `position_ids`（`Qwen3.forward` 不再代劳）

### Task 1.1 — `Qwen3.forward` 增加 packed 分支

**文件**：`src/qwen3_from_scratch/models/qwen3.py`

**当前问题**：`idx.shape[1]` 假设 2D 输入，packed 模式 `idx` 是 `(T,)`。

```python
def forward(self, idx):
    ctx = get_forward_context()
    tok_embd = self.tok_embd(idx)

    if ctx.slot_mapping is None:        # batch 模式：自己算 position_ids
        ctx.position_ids = torch.arange(
            ctx.cache_position,
            ctx.cache_position + idx.shape[1],
            dtype=torch.long, device=tok_embd.device,
        ).unsqueeze(0)
    # packed 模式：position_ids 由调度器预计算，已在 ctx 中

    ctx.position_embeddings = self.rope.build_cos_sin_embed(
        ctx.dtype, ctx.position_ids
    )
    x = tok_embd
    for layer in self.trf_blocks:
        x = layer(x)
    x = self.final_norm(x)
    logits = self.output_head(x)
    return logits
```

**验收**：

- [ ] `idx=(B, S)` → 正常，与改动前一致
- [ ] `idx=(T,)` + `ctx.slot_mapping` 已设 → 正常
- [ ] `idx=(T,)` + `ctx.slot_mapping=None` → 走 batch 分支，`idx.shape[1]` 报错（预期行为，不应这样调用）
- [ ] 训练不受影响

### Task 1.2 — `SelfAttention` 增加 `_forward_packed`

**文件**：`src/qwen3_from_scratch/models/self_attn.py`

**改动**：

1. `__init__` 增加 `self.paged_attn`
2. `forward` 增加模式分发
3. 新增 `_forward_packed` 方法
4. 原有逻辑提取为 `_forward_batch`（不改动逻辑）

```python
@ComponentFactory.register("self_attn", "base")
class SelfAttention(nn.Module):
    def __init__(self, config, name, layer_idx=0, **kwargs):
        super().__init__()
        # ... 现有 q/k/v/o_proj, q_norm, k_norm, rope ...
        self.gqa = ComponentFactory.create("attn", config, layer_idx=layer_idx)
        # 新增
        from qwen3_from_scratch.models.attn import VarLenPagedAttn
        self.paged_attn = VarLenPagedAttn(config, layer_idx=layer_idx)

    def forward(self, x):
        ctx = get_forward_context()
        if ctx.slot_mapping is not None:
            return self._forward_packed(x, ctx)
        return self._forward_batch(x, ctx)

    def _forward_batch(self, x, ctx):
        # === 原有代码，原封搬过来 ===
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)
        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        q = self.rope(q).to(x.dtype)
        k = self.rope(k).to(x.dtype)
        if ctx.use_cache:
            k, v = ctx.kv_cache.update(
                k.transpose(1, 2), v.transpose(1, 2),
                self.layer_idx, ctx.cache_position,
            )
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        o = self.gqa(q, k, v).transpose(1, 2).reshape(*input_shape, -1)
        return self.o_proj(o)

    def _forward_packed(self, x, ctx):
        T = x.shape[0]
        D = self.config.head_dim
        H_q = self.config.num_attention_heads
        H_kv = self.config.num_key_value_heads

        # projection + reshape — 无 transpose
        q = self.q_norm(self.q_proj(x).view(T, H_q, D))  # (T, H_q, D)
        k = self.k_norm(self.k_proj(x).view(T, H_kv, D))  # (T, H_kv, D)
        v = self.v_proj(x).view(T, H_kv, D)  # (T, H_kv, D)

        # RoPE — 统一调用 self.rope，与 batch 模式相同
        q = self.rope(q).to(x.dtype)
        k = self.rope(k).to(x.dtype)

        # KV Cache — 写入 paged cache + 取出整层缓存
        ctx.kv_cache.update(k, v, self.layer_idx, ctx.cache_position)
        k_cache, v_cache = ctx.kv_cache.get(self.layer_idx)

        # Paged attention
        o = self.paged_attn(q, k_cache, v_cache)  # (T, H_q, D)
        o = o.reshape(T, -1)
        return self.o_proj(o)
```

**验收**：

- [ ] Batch 模式：输出与改动前 `SelfAttention` 完全一致
- [ ] Packed 模式：手动构建 context（slot_mapping / block_tables / cum_seq_lens），输出与 `TorchVarLenPagedAttn` 直接调用一致
- [ ] RoPE 在 packed 模式下通过 `self.rope(q)` 调用，不再内联

### Task 1.3 — 废弃 `PagedSelfAttention`

**文件**：`src/qwen3_from_scratch/models/self_attn.py`

`PagedSelfAttention` 的功能已合并进 `SelfAttention._forward_packed`。

- [ ] 确认无外部引用（`test_paged_attn.py` 等直接测试 `TorchVarLenPagedAttn`，不依赖此类）
- [ ] 删除或注释标记 deprecated
- [ ] 如有测试引用，改为使用 `SelfAttention` + packed context

---

## Phase 2：BlockManager + Scheduler

> 前置条件：Phase 1 完成。此阶段不改模型代码，纯新增。

### Task 2.1 — BlockManager（页面管理器）

**新建文件**：`src/qwen3_from_scratch/inference/block_manager.py`

```python
class BlockManager:
    """管理 PagedKVCache 的物理页面分配与回收"""

    def __init__(self, num_pages: int, block_size: int):
        self.block_size = block_size
        self.num_pages = num_pages
        self.free_blocks: list[int] = list(range(num_pages))
        self.allocated: dict[int, list[int]] = {}  # seq_id -> [block_ids]

    def can_allocate(self, num_tokens: int) -> bool:
        """是否有足够空闲页容纳 num_tokens"""
        ...

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """为序列分配页面，返回 block_id 列表"""
        ...

    def append_block(self, seq_id: int) -> int:
        """序列需要更多空间时追加一个页"""
        ...

    def free(self, seq_id: int):
        """序列结束，释放其所有页面"""
        ...

    def get_block_tables(self, seq_ids: list[int]) -> torch.Tensor:
        """构建 (num_seqs, max_blocks) 的 block_tables 张量"""
        ...
```

**验收**：

- [ ] 分配/回收正确，free_blocks 数量一致
- [ ] 同一 seq_id 重复 allocate 报错或追加
- [ ] 内存不足时 `can_allocate` 返回 False

### Task 2.2 — Scheduler（调度器）

**新建文件**：`src/qwen3_from_scratch/inference/scheduler.py`

```python
@dataclass
class Request:
    request_id: int
    prompt_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    is_finished: bool = False

@dataclass
class StepMetadata:
    """一步前向所需的全部元数据"""
    input_ids: torch.Tensor              # (total_tokens,)
    position_ids: torch.Tensor           # (total_tokens,)
    slot_mapping: torch.Tensor           # (total_tokens,)
    block_tables: torch.Tensor           # (num_seqs, max_blocks)
    cum_seq_lens_q: torch.Tensor         # (num_seqs + 1,)
    cum_seq_lens_kv: torch.Tensor        # (num_seqs + 1,)
    active_requests: list[Request]
    is_prefill_step: bool

class Scheduler:
    def __init__(self, block_manager: BlockManager, max_num_seqs: int):
        ...

    def add_request(self, request: Request):
        """添加新请求到 waiting 队列"""
        ...

    def has_pending(self) -> bool:
        """是否还有未完成的请求"""
        ...

    def schedule(self) -> StepMetadata:
        """
        策略（先做简单的 prefill/decode 分离）：
        1. 如果 waiting 非空且资源够，做 prefill 步（可批量多个 prompt）
        2. 否则做 decode 步（所有 running 请求各 1 token）
        3. 构建 StepMetadata 中的所有张量
        """
        ...

    def update(self, step_metadata: StepMetadata, next_tokens: dict[int, int]):
        """
        前向后调用：
        - 更新每个 request 的 generated_ids
        - 检查 EOS / max_tokens → 标记 finished → 释放页面
        - decode 步可能需要 append_block
        """
        ...
```

**验收**：

- [ ] 单请求 prefill：`cum_seq_lens_q == cum_seq_lens_kv`，`slot_mapping` 连续
- [ ] 多请求 prefill：不同长度 prompt 打包，`cum_seq_lens` 正确
- [ ] 单请求 decode：`cum_seq_lens_q = [0, 1]`，`cum_seq_lens_kv = [0, current_len]`
- [ ] 多请求 decode：各请求不同长度，`cum_seq_lens` 正确
- [ ] EOS 触发 free，页面回收
- [ ] `slot_mapping` 与 `block_tables` 一致：`slot = block_tables[seq, token//block_size] * block_size + token % block_size`

### Task 2.3 — `slot_mapping` 构建逻辑

在 Scheduler 中实现，需与 `PagedKVCache._update_var_len` 的读取逻辑对齐：

```python
def _build_slot_mapping(self, requests, seq_lens_kv):
    """
    对每个 request 的每个 token：
      slot = block_tables[seq_idx, token // block_size] * block_size + token % block_size
    padding token 用 -1（PagedKVCache 会跳过）
    """
    ...
```

**验收**：

- [ ] `PagedKVCache.update` 写入的位置与 `TorchVarLenPagedAttn._load_kv` 读取的位置一致
- [ ] `-1` slot 被 `PagedKVCache._update_var_len` 正确跳过

---

## Phase 3：多请求生成引擎

> 前置条件：Phase 2 完成。

### Task 3.1 — PagedInferenceEngine

**新建文件**：`src/qwen3_from_scratch/inference/engine.py`

```python
class PagedInferenceEngine:
    def __init__(self, model, config: ModelConfig, block_size=16, mem_size=...):
        self.model = model
        self.config = config
        self.block_size = block_size
        num_blocks = PagedKVCache.get_block_num(
            mem_size, config.num_hidden_layers,
            config.num_key_value_heads, config.head_dim,
            block_size=block_size,
        )
        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.block_manager)
        self.kv_cache = PagedKVCache(num_blocks, config.num_hidden_layers,
                                      config.num_key_value_heads, config.head_dim,
                                      block_size=block_size)
        self._context = ModelContext(
            use_cache=True,
            kv_cache=self.kv_cache,
            block_size=block_size,
        )

    def add_request(self, request_id, prompt_ids):
        self.scheduler.add_request(Request(request_id, prompt_ids))

    def step(self) -> dict[int, int]:
        """执行一步前向，返回 {request_id: next_token_id}"""
        if not self.scheduler.has_pending():
            return {}

        meta = self.scheduler.schedule()

        # 填充 context
        self._context.slot_mapping = meta.slot_mapping
        self._context.block_tables = meta.block_tables
        self._context.cum_seq_lens_q = meta.cum_seq_lens_q
        self._context.cum_seq_lens_kv = meta.cum_seq_lens_kv
        self._context.position_ids = meta.position_ids
        self._context.cache_position = 0  # packed 模式不用
        set_forward_context(self._context)

        # 前向
        with torch.no_grad():
            logits = self.model(meta.input_ids)  # (total_tokens, vocab)

        # 每个 request 取最后一个 token 的 logits
        results = {}
        next_tokens = {}
        for i, req in enumerate(meta.active_requests):
            q_start = int(meta.cum_seq_lens_q[i])
            q_end = int(meta.cum_seq_lens_q[i + 1])
            last_token_logits = logits[q_end - 1]  # 该 request 的最后一个 q token
            next_token = sample(last_token_logits, temperature, top_k)
            next_tokens[req.request_id] = next_token
            results[req.request_id] = next_token

        self.scheduler.update(meta, next_tokens)
        return results

    def generate(self, requests: dict[int, list[int]], max_new_tokens=...):
        """便捷方法：批量添加请求并生成到全部完成"""
        for rid, prompt in requests.items():
            self.add_request(rid, prompt)
        while self.scheduler.has_pending():
            self.step()
```

**验收**：

- [ ] 单请求：输出与 `InferenceSession.generate_from_ids` 逐 token 一致（贪婪解码）
- [ ] 多请求：各请求独立生成，长度不同，提前结束的请求不影响其他
- [ ] 页面在请求结束后被回收，可被新请求复用

### Task 3.2 — 端到端测试

**新建文件**：`test/test_paged_engine.py`

- [ ] 单请求 prefill + decode，对比 `InferenceSession` 输出
- [ ] 2 请求不同长度 prompt，各自贪婪解码，对比单请求结果
- [ ] 3 请求，中间一个先结束，其余继续
- [ ] 页面耗尽场景（小 mem_size）行为正确（等待或报错）

---

## Phase 4：性能优化（后续）

> 前置条件：Phase 3 功能正确。非阻塞，可独立进行。

### Task 4.1 — PagedKVCache.update 向量化

**文件**：`src/qwen3_from_scratch/inference/kv_cache/paged_cache.py`

当前 Python for 循环逐 token 写入：

```python
for i in range(k.shape[0]):
    slot = slot_mapping[i]
    block_id, slot_id = slot // self.block_size, slot % self.block_size
    self.k_cache[layer_idx, block_id, slot_id] = k[i]
```

改为向量化 scatter：

```python
valid = slot_mapping != -1
slots = slot_mapping[valid]
block_ids = slots // self.block_size
slot_ids = slots % self.block_size
self.k_cache[layer_idx, block_ids, slot_ids] = k[valid]
self.v_cache[layer_idx, block_ids, slot_ids] = v[valid]
```

### Task 4.2 — Triton Paged Attention Kernel

**新建文件**：`src/qwen3_from_scratch/kernels/triton/paged_attn.py`

参考现有 `flash_attention` kernel 模板，增加 paged KV 读取（通过 `block_tables` 间接寻址）。`grouped_gemm`（`kernels/triton/gemm.py`）可作为变长 batch matmul 的参考。

### Task 4.3 — fused_kvcache stub 实现

**文件**：`src/qwen3_from_scratch/kernels/triton/fused/fused_kvcache.py`

当前只有签名。目标是 fuse QK rms_norm + RoPE + KV-cache write 到一个 kernel。

### Task 4.4 — PagedSelfAttention CUDA 路径

如果需要 CUDA 上跑 packed 模式（当前 `_forward_packed` 只能 CPU），需要：
- Triton paged attention kernel（Task 4.2）
- 或在 `_forward_packed` 中对 projection 部分用 Triton linear，attention 部分用上述 kernel

---

## 不改动清单（明确记录）

| 组件 | 不改原因 |
|------|---------|
| `examples/train/` | 训练用 base 组件 + `use_cache=False`，与推理路径完全独立 |
| `norm.py` 全部 | `rms_norm` 只操作 `dim=-1`，`(B,S,H)` 和 `(T,H)` 行为一致 |
| `feedback.py` 全部 | `nn.Linear` + SiLU + residual 只操作 `dim=-1` |
| `transformer_block.py` 全部 | 链式调用 `norm → attn → +residual → norm → mlp → +residual`，布局由子模块处理 |
| `InferenceSession` + `PreAllocatedKVCache` | 单请求简单推理向后兼容 |
| `generate()` 函数 | 单请求场景仍可用 |
| `TorchGQA` / `MyAttnFlash` / `MyAttn` | Batch-only attention，packed 模式不调用 |
| `FusedSelfAttention` CUDA 路径 | Triton kernel 硬编码 BHSD，packed 不走此路径 |
| `context.py` | 字段已齐备（`block_tables`, `slot_mapping`, `cum_seq_lens_q/kv`, `block_size`） |
| `TorchPagedAttn` / `TorchVarLenPagedAttn` | 已实现并测试通过，作为 packed 路径的 attention |

---

## 统一数据流图

```
                         position_ids
                     ┌──────────────────┐
                     │                  │
               Batch (1, S)        Packed (T,)
                     │                  │
                     ▼                  ▼
          build_cos_sin_embed (flatten + outer)
                     │                  │
                     ▼                  ▼
                 cos (N, D)         cos (N, D)     ← 统一格式
                     │                  │
                     ▼                  ▼
               forward(q)           forward(q)
              q is BHSD            q is THD
              dim()==4             dim()==3
                     │                  │
                     ▼                  ▼
          cos[None,None,:,:]     cos[:,None,:]
           (1,1,S,D)              (T,1,D)
                     │                  │
                     ▼                  ▼
            q * cos + rotate(q) * sin   ← 同一行代码
                     │                  │
                     ▼                  ▼
    SelfAttention._forward_batch   SelfAttention._forward_packed
      ├ projection  q/k/v_proj        ├ projection  q/k/v_proj (相同)
      ├ reshape     view+transpose    ├ reshape     view (无transpose)
      ├ rope        self.rope(q)      ├ rope        self.rope(q) (相同)
      ├ cache       PreAllocated      ├ cache       PagedKVCache
      ├ attention   self.gqa (BHSD)   ├ attention   self.paged_attn (SHD)
      └ o_proj      o_proj            └ o_proj      o_proj (相同)
```
