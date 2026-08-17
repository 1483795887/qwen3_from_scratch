"""Batch 模式的多模型配置加载。

从 YAML 配置文件加载多个模型的配置，包括模型路径、组件覆写、采样参数。
与 PackedConfig 完全分离，不共享基类（ADR 0003）。

配置文件结构：
    generation:          # 全局默认，所有模型继承
        temperature: 0.85
        top_k: 40
        ...
    models:              # 模型列表
        - name: "qwen3-0.6b"
          path: "/path/to/model"
          device: "cpu"
          max_len: 2048
          components:
            mlp: "moe"               # 简写
            attn:                     # 展开
              name: "my_op"
              kwargs:
                scale: 1.0
          generation:                # 可选，覆盖全局
            temperature: 0.7
"""

import os
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional

import torch
import yaml

from .config import ComponentConfig
from .config import load_from_file as load_model_config

# ── dtype 解析 ────────────────────────────────

_DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}

# 反向映射：torch.dtype → 规范字符串（用于保存）
_DTYPE_TO_STR: Dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _parse_dtype(value: Any) -> torch.dtype:
    """将 YAML 中的 dtype 配置解析为 torch.dtype。

    接受：
      - str:  "bf16", "bfloat16", "fp16", "float16", "fp32", "float32" 等
      - torch.dtype: 直接返回
    """
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _DTYPE_ALIASES:
            raise ValueError(
                f"不支持的 dtype '{value}'。"
                f"有效值: {sorted(_DTYPE_ALIASES.keys())}"
            )
        return _DTYPE_ALIASES[key]
    raise ValueError(
        f"kv_cache_dtype 必须是 str 或 torch.dtype，得到: "
        f"{type(value).__name__}"
    )


# ── 采样参数 ──────────────────────────────────


@dataclass
class GenerationDefaults:
    """全局采样默认值。所有字段必填。"""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 100


@dataclass
class GenerationOverrides:
    """模型级采样覆盖。所有字段 Optional，None 表示继承全局。"""

    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    max_new_tokens: Optional[int] = None


# ── 模型条目 ──────────────────────────────────


@dataclass
class ModelEntry:
    """BatchConfig.models 列表中的一个条目，未合并状态。

    generation 字段是 Optional——可能为 None（继承全局）。
    Runner 构建不应直接消费 ModelEntry，应通过 get_model 获取 ResolvedModelEntry。
    """

    name: str
    path: str
    device: str = "cpu"
    max_len: int = 2048
    dtype: torch.dtype = torch.bfloat16
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    generation: Optional[GenerationOverrides] = None


@dataclass
class ResolvedModelEntry:
    """已合并的模型条目。generation 必填，无歧义。

    由 BatchConfig.get_model() 返回，Runner 构建只消费此类型。
    """

    name: str
    path: str
    device: str
    max_len: int
    dtype: torch.dtype
    components: Dict[str, ComponentConfig]
    generation: GenerationDefaults


@dataclass
class SchedulerDefaults:
    max_num_seqs: int = 32
    max_num_tokens: int = 3000
    block_size: int = 16
    gpu_utilization: float = 0.5
    enable_prefix_cache: bool = True
    chunked_prefill_size: int = 512


# ── 顶层配置 ──────────────────────────────────


@dataclass
class BatchConfig:
    """Batch 模式的多模型配置。

    通过 load_batch_config() 加载，加载时全量校验所有模型条目。
    """

    generation: GenerationDefaults = field(default_factory=GenerationDefaults)
    models: List[ModelEntry] = field(default_factory=list)
    scheduler: SchedulerDefaults = field(default_factory=SchedulerDefaults)
    kv_cache_dtype: torch.dtype = torch.bfloat16

    def get_model(self, name: str) -> ResolvedModelEntry:
        """按 name 查找模型，返回已合并的 ResolvedModelEntry。"""
        for m in self.models:
            if m.name == name:
                merged = _merge_generation(self.generation, m.generation)
                return ResolvedModelEntry(
                    name=m.name,
                    path=m.path,
                    device=m.device,
                    dtype=m.dtype,
                    max_len=m.max_len,
                    components=dict(m.components),
                    generation=merged,
                )
        available = self.list_model_names()
        raise ValueError(f"模型 '{name}' 未找到。可用: {available}")

    def list_model_names(self) -> List[str]:
        """返回所有模型的 name 列表。"""
        return [m.name for m in self.models]


# ── 合并 ──────────────────────────────────────


def _merge_generation(
    defaults: GenerationDefaults,
    overrides: Optional[GenerationOverrides],
) -> GenerationDefaults:
    """深度合并：模型级覆盖全局，未覆盖的字段继承全局默认。"""
    if overrides is None:
        return replace(defaults)
    merged = replace(defaults)
    for f in fields(GenerationOverrides):
        val = getattr(overrides, f.name)
        if val is not None:
            setattr(merged, f.name, val)
    return merged


# ── 组件配置解析 ──────────────────────────────


def _parse_component(value: Any) -> ComponentConfig:
    """解析 YAML 中的组件配置，支持简写和展开两种格式。

    简写: mlp: "moe"              → ComponentConfig(name="moe")
    展开: mlp: {name: "my_op", kwargs: {scale: 1.0}}
    """
    if isinstance(value, str):
        return ComponentConfig(name=value)
    if isinstance(value, dict):
        if "name" not in value:
            raise ValueError(
                f"组件配置展开格式必须含 'name' 字段，得到: {value}"
            )
        return ComponentConfig(
            name=value["name"],
            kwargs=value.get("kwargs", {}),
        )
    raise ValueError(
        f"组件配置值必须是 str（简写）或 dict（展开），得到: "
        f"{type(value).__name__}"
    )


# ── YAML 加载 ────────────────────────────────


def load_batch_config(config_path: str) -> BatchConfig:
    """从 YAML 文件加载 BatchConfig，加载时全量校验。

    校验内容：
      - models 列表非空
      - name 唯一
      - path 存在且含 config.json
      - components 字段名和实现名在 ComponentFactory._registry 中有效
      - max_len > 0
      - max_len ≤ max_position_embeddings（超出则警告+截断）
      - device 合法
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"配置文件为空: {config_path}")

    # 解析全局 generation
    gen_raw = raw.get("generation", {}) or {}
    generation = GenerationDefaults(
        temperature=gen_raw.get("temperature", 1.0),
        top_k=gen_raw.get("top_k", 0),
        top_p=gen_raw.get("top_p", 1.0),
        do_sample=gen_raw.get("do_sample", False),
        max_new_tokens=gen_raw.get("max_new_tokens", 100),
    )

    # 解析调度配置
    scheduler_raw = raw.get("scheduler", {}) or {}
    scheduler = SchedulerDefaults(
        max_num_seqs=scheduler_raw.get("max_num_seqs", 32),
        max_num_tokens=scheduler_raw.get("max_num_tokens", 3000),
        block_size=scheduler_raw.get("block_size", 16),
        gpu_utilization=scheduler_raw.get("gpu_utilization", 0.5),
        enable_prefix_cache=scheduler_raw.get("enable_prefix_cache", True),
        chunked_prefill_size=scheduler_raw.get("chunked_prefill_size", 512),
    )

    # 解析 KVCache dtype
    kv_cache_dtype = _parse_dtype(raw.get("kv_cache_dtype", "bfloat16"))

    # 解析 models
    models_raw = raw.get("models", [])
    if not isinstance(models_raw, list):
        raise ValueError("'models' 必须是列表")

    models: List[ModelEntry] = []
    for i, m in enumerate(models_raw):
        if not isinstance(m, dict):
            raise ValueError(f"models[{i}] 必须是字典")

        # 必填字段
        for key in ("name", "path"):
            if key not in m:
                raise ValueError(
                    f"models[{i}] (name={m.get('name', '?')}): "
                    f"缺少必填字段 '{key}'"
                )

        # 组件配置
        components_raw = m.get("components", {}) or {}
        if not isinstance(components_raw, dict):
            raise ValueError(f"模型 {m['name']}: components 必须是字典")
        components = {
            k: _parse_component(v) for k, v in components_raw.items()
        }

        # 模型级 generation 覆盖
        gen_override: Optional[GenerationOverrides] = None
        gen_m = m.get("generation")
        if gen_m:
            gen_override = GenerationOverrides(
                temperature=gen_m.get("temperature"),
                top_k=gen_m.get("top_k"),
                top_p=gen_m.get("top_p"),
                do_sample=gen_m.get("do_sample"),
                max_new_tokens=gen_m.get("max_new_tokens"),
            )

        models.append(
            ModelEntry(
                name=m["name"],
                path=m["path"],
                device=m.get("device", "cpu"),
                dtype=_parse_dtype(m.get("dtype", "bfloat16")),
                max_len=m.get("max_len", 2048),
                components=components,
                generation=gen_override,
            )
        )

    config = BatchConfig(
        generation=generation,
        models=models,
        scheduler=scheduler,
        kv_cache_dtype=kv_cache_dtype,
    )
    _validate(config)
    return config


# ── 校验 ──────────────────────────────────────


def _validate(config: BatchConfig) -> None:
    """全量校验配置，任一错误立即抛出 ValueError。"""
    if not config.models:
        raise ValueError("models 列表不能为空")

    # name 唯一
    names = [m.name for m in config.models]
    seen: set = set()
    duplicates: set = set()
    for n in names:
        if n in seen:
            duplicates.add(n)
        seen.add(n)
    if duplicates:
        raise ValueError(f"模型 name 不唯一: {sorted(duplicates)}")

    # 逐模型校验
    from .factory import ComponentFactory

    for m in config.models:
        # path 存在
        if not os.path.isdir(m.path):
            raise ValueError(f"模型 {m.name}: path 不存在或不是目录: {m.path}")

        # config.json 存在
        config_json = os.path.join(m.path, "config.json")
        if not os.path.isfile(config_json):
            raise ValueError(f"模型 {m.name}: 目录下无 config.json: {m.path}")

        # components 字段名 + 实现名有效
        for comp_name, comp_conf in m.components.items():
            if comp_name not in ComponentFactory._registry:
                raise ValueError(
                    f"模型 {m.name}: 未知组件字段 '{comp_name}'。"
                    f"有效: {list(ComponentFactory._registry.keys())}"
                )
            if comp_conf.name not in ComponentFactory._registry[comp_name]:
                raise ValueError(
                    f"模型 {m.name}: 未知 {comp_name} 实现 "
                    f"'{comp_conf.name}'。"
                    f"已注册: "
                    f"{list(ComponentFactory._registry[comp_name].keys())}"
                )

        # max_len > 0
        if m.max_len <= 0:
            raise ValueError(
                f"模型 {m.name}: max_len 必须大于 0，当前: {m.max_len}"
            )

        # max_len ≤ max_position_embeddings（警告+截断）
        model_cfg = load_model_config(config_json)
        max_pos = model_cfg.max_position_embeddings
        if m.max_len > max_pos:
            print(
                f"Warning: 模型 {m.name}: max_len={m.max_len} 超过 "
                f"max_position_embeddings={max_pos}，"
                f"截断为 {max_pos}"
            )
            m.max_len = max_pos

        # device 合法
        try:
            torch.device(m.device)
        except (ValueError, RuntimeError) as e:
            raise ValueError(
                f"模型 {m.name}: 无效 device '{m.device}': {e}"
            ) from e
