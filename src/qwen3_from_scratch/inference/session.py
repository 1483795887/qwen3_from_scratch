import os
from typing import Iterable, Union, Collection, List, Dict, Any, Optional

import torch
import jinja2
from tokenizers import Tokenizer
from torch.nn import Module

from qwen3_from_scratch.factory.config import GenerationConfig, load_from_file
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.inference.generate import generate as _generate
from qwen3_from_scratch.inference.kv_cache.pre_allocated_kv_cache import PreAllocatedKVCache
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file


class InferenceSession:
    """Encapsulates model loading, generation config, chat template, and text generation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_len: int = 2048,
    ):
        load_env_file()

        self.model_path = model_path
        self.device = device
        self.max_len = max_len

        # Load configs
        self.model_config = load_from_file(os.path.join(model_path, "config.json"))
        self.gen_config = GenerationConfig.load_from_file(
            os.path.join(model_path, "generation_config.json")
        )

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))

        # Load chat_template from tokenizer_config.json
        self._load_chat_template()

        # Load model
        self.model = self._load_model()

        # Initialize KV cache
        self._init_kv_cache()

    def _load_chat_template(self):
        """Load chat_template from tokenizer_config.json."""
        import json

        tokenizer_config_path = os.path.join(self.model_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.chat_template = config.get("chat_template", None)
        else:
            self.chat_template = None

    def _load_model(self) -> Module:
        """Load the model with weights."""
        loader = ParameterLoader()
        loader.load(self.model_path)
        model = Qwen3(config=self.model_config)
        model.load_state(loader)
        model.to(self.device)
        return model

    def _init_kv_cache(self):
        """Initialize the KV cache for generation."""
        self._context = ModelContext()
        self._context.use_cache = True
        self._context.dtype = torch.bfloat16
        self._context.kv_cache = PreAllocatedKVCache(
            self.max_len, self.model_config.num_hidden_layers
        )

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Apply the chat template to format messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional arguments for the template (e.g., tools, tools用过).

        Returns:
            Formatted prompt string.
        """
        if self.chat_template is None:
            # Fallback: simple concatenation
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        env = jinja2.Environment()
        template = env.from_string(self.chat_template)

        # Build template kwargs
        template_kwargs = {"messages": messages, **kwargs}

        return template.render(**template_kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_ids: Optional[Union[int, Collection[int]]] = None,
        stream: bool = True,
    ) -> Iterable[str] | str:
        """
        Generate text from a plain prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate (default: self.max_len).
            temperature: Sampling temperature (default: from gen_config).
            top_k: Top-k sampling parameter (default: from gen_config).
            eos_ids: EOS token IDs to stop at (default: from gen_config).
            stream: Whether to yield tokens incrementally.

        Returns:
            Generated text (str if stream=False, Iterable[str] if stream=True).
        """
        inputs = self.tokenizer.encode(prompt)
        return self.generate_from_ids(
            torch.tensor([inputs.ids]),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_ids=eos_ids,
            stream=stream,
        )

    def generate_from_messages(
        self,
        messages: Union[List[Dict[str, str]], str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_ids: Optional[Union[int, Collection[int]]] = None,
        stream: bool = True,
        **chat_template_kwargs,
    ) -> Iterable[str] | str:
        """
        Generate text from chat messages using the chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                      or a single string (auto-wrapped as user message).
            max_new_tokens: Maximum tokens to generate (default: self.max_len).
            temperature: Sampling temperature (default: from gen_config).
            top_k: Top-k sampling parameter (default: from gen_config).
            eos_ids: EOS token IDs to stop at (default: from gen_config).
            stream: Whether to yield tokens incrementally.
            **chat_template_kwargs: Additional kwargs for the chat template.

        Returns:
            Generated text (str if stream=False, Iterable[str] if stream=True).
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        prompt = self.apply_chat_template(messages, **chat_template_kwargs)
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_ids=eos_ids,
            stream=stream,
        )

    def generate_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_ids: Optional[Union[int, Collection[int]]] = None,
        stream: bool = True,
    ) -> Iterable[str] | str:
        """
        Generate text from token IDs.

        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len].
            max_new_tokens: Maximum tokens to generate (default: self.max_len).
            temperature: Sampling temperature (default: from gen_config).
            top_k: Top-k sampling parameter (default: from gen_config).
            eos_ids: EOS token IDs to stop at (default: from gen_config).
            stream: Whether to yield tokens incrementally.

        Returns:
            Generated text (str if stream=False, Iterable[str] if stream=True).
        """
        max_new_tokens = max_new_tokens or self.max_len
        temperature = temperature if temperature is not None else self.gen_config.temperature
        top_k = top_k if top_k is not None else self.gen_config.top_k
        eos_ids = eos_ids if eos_ids is not None else self.gen_config.eos_token_id

        # Reset context for new generation
        self._init_kv_cache()

        return _generate(
            self.model,
            input_ids,
            max_new_tokens,
            context=self._context,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            eos_ids=eos_ids,
            tokenizer=self.tokenizer,
            device=self.device,
            stream=stream,
        )
