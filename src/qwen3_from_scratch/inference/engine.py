import json
import os
from typing import Collection, Dict, Iterator, List, Optional, Union

import jinja2
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from qwen3_from_scratch.factory.config import ComponentConfig, GenerationConfig
from qwen3_from_scratch.inference.context import ModelContext, set_forward_context
from qwen3_from_scratch.inference.kv_cache.pre_allocated_kv_cache import (
    PreAllocatedKVCache,
)
from qwen3_from_scratch.inference.model_loader import ModelLoader
from qwen3_from_scratch.inference.sampler import (
    GreedySampler,
    Sampler,
    TemperatureSampler,
    TopKSampler,
)


class BatchRunner:
    """单请求 Batch 推理引擎。

    提供三种推理模式：
      - generate(prompt)          → str           同步，返回完整文本
      - generate_stream(prompt)   → Iterator[str] 异步，逐词元 yield 解码文本
      - prefill(ids) + step(id)   → int           手动控制：先 prefill 再逐步 step

    引擎内部持有 tokenizer 和可选 chat_template，
    generate/generate_stream 接受 str（raw text）或 list[dict]（messages，走 chat template）。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        sampler: Sampler,
        max_len: int = 2048,
        chat_template: Optional[str] = None,
        eos_ids: Union[int, Collection[int], None] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.max_len = max_len
        self.chat_template = chat_template
        self.device = next(model.parameters()).device

        self.eos_ids = self._normalize_eos(eos_ids)

        self._context: Optional[ModelContext] = None
        self._seq_len: int = 0
        self._init_kv_cache()

    # ── 便捷构造 ──────────────────────────────────

    @classmethod
    def from_path(
        cls,
        model_path: str,
        device: str = "cpu",
        sampler: Optional[Sampler] = None,
        max_len: int = 2048,
        components: Optional[Dict[str, ComponentConfig]] = None,
    ) -> "BatchRunner":
        """从模型路径一键构建引擎。

        自动加载 model / tokenizer / chat_template / generation_config。
        如果不传 sampler，则根据 gen_config 自动选择。
        """
        model = ModelLoader.load(model_path, device, components=components)
        tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        chat_template = _load_chat_template(model_path)
        gen_config = GenerationConfig.load_from_file(
            os.path.join(model_path, "generation_config.json")
        )
        if sampler is None:
            sampler = _sampler_from_gen_config(gen_config)
        return cls(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            max_len=max_len,
            chat_template=chat_template,
            eos_ids=gen_config.eos_token_id,
        )

    # ── public API ────────────────────────────────

    def prefill(self, prompt_ids: torch.Tensor) -> List[int]:
        """处理整个 prompt，建立 KV cache 上下文，返回第一个生成词元（长度 B 的 list）。

        每次调用都会重置 KV cache，适合开始新的生成。
        """
        self._init_kv_cache()
        set_forward_context(self._context)

        prompt_ids = prompt_ids.to(self.device)
        self._context.cache_position = 0
        self._context.position_ids = torch.arange(
            0, prompt_ids.shape[1], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(prompt_ids)
        logits = logits[:, -1, :]  # [B, vocab]
        next_ids = self.sampler(logits)  # [B, 1]

        self._seq_len = prompt_ids.shape[1]
        return next_ids[:, 0].tolist()  # length B

    def step(self, token_ids: List[int]) -> List[int]:
        """单步 decode：输入 B 个词元，输出 B 个下一个词元。

        必须先调用 prefill 建立上下文。
        """
        self._context.cache_position = self._seq_len
        self._context.position_ids = torch.arange(
            self._seq_len, self._seq_len + 1, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        token_tensor = torch.tensor([token_ids], device=self.device)  # [B, 1]
        with torch.no_grad():
            logits = self.model(token_tensor)
        logits = logits[:, -1, :]  # [B, vocab]
        next_ids = self.sampler(logits)  # [B, 1]

        self._seq_len += 1
        return next_ids[:, 0].tolist()  # length B

    def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 100,
        eos_ids: Union[int, Collection[int], None] = None,
    ) -> Iterator[str]:
        """异步：逐词元 yield 解码文本。"""
        eos = self.eos_ids if eos_ids is None else self._normalize_eos(eos_ids)
        ids = self._encode(prompt)

        first_ids = self.prefill(ids)
        if first_ids[0] in eos:
            return
        yield self.tokenizer.decode(first_ids, skip_special_tokens=False)

        cur = first_ids
        for _ in range(max_new_tokens - 1):
            nxt = self.step(cur)
            if nxt[0] in eos:
                break
            yield self.tokenizer.decode(nxt, skip_special_tokens=False)
            cur = nxt

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 100,
        eos_ids: Union[int, Collection[int], None] = None,
    ) -> str:
        """同步：返回完整文本。"""
        return "".join(
            self.generate_stream(prompt, max_new_tokens, eos_ids)
        )

    # ── 内部方法 ──────────────────────────────────

    def _init_kv_cache(self):
        """构建 ModelContext + PreAllocatedKVCache，重置序列长度。"""
        self._context = ModelContext()
        self._context.use_cache = True
        self._context.dtype = torch.bfloat16
        self._context.kv_cache = PreAllocatedKVCache(
            self.max_len, self.model.config.num_hidden_layers
        )
        self._seq_len = 0

    def _encode(self, prompt: Union[str, List[Dict[str, str]]]) -> torch.Tensor:
        """prompt → token id tensor [1, S]。

        str  → 直接 tokenize
        list → jinja2 chat template 渲染后 tokenize
        """
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list):
            text = self._apply_chat_template(prompt)
        else:
            raise TypeError(
                f"prompt must be str or list[dict], got {type(prompt)}"
            )
        ids = self.tokenizer.encode(text)
        return torch.tensor([ids.ids])

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """用 jinja2 渲染 chat template，无 template 时简单拼接。"""
        if self.chat_template is None:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        env = jinja2.Environment()
        template = env.from_string(self.chat_template)
        return template.render(messages=messages)

    @staticmethod
    def _normalize_eos(
        eos_ids: Union[int, Collection[int], None]
    ) -> set:
        if eos_ids is None:
            return set()
        if isinstance(eos_ids, int):
            return {eos_ids}
        return set(eos_ids)


# ── 模块级辅助函数 ───────────────────────────────


def _load_chat_template(model_path: str) -> Optional[str]:
    """从 tokenizer_config.json 读取 chat_template 字段。"""
    path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("chat_template", None)
    return None


def _sampler_from_gen_config(gen_config: GenerationConfig) -> Sampler:
    """根据 GenerationConfig 自动选择 Sampler 类型。"""
    if gen_config.temperature > 0.0 and gen_config.top_k > 0:
        return TopKSampler(
            top_k=gen_config.top_k, temperature=gen_config.temperature
        )
    elif gen_config.temperature > 0.0:
        return TemperatureSampler(temperature=gen_config.temperature)
    else:
        return GreedySampler()
