import asyncio
from pathlib import Path

from qwen3_from_scratch.inference.llm.async_llm import AsyncLLM
from qwen3_from_scratch.inference.llm.llm_base import GenerateParams


def get_config_path():
    return Path(__file__).parent / "configs" / "batch2.yaml"


async def main():
    # engine = LLMEngine(get_config_path(), "qwen3-0.6b")
    engine = AsyncLLM(get_config_path(), "qwen3-0.6b")
    await engine.warmup()
    async for chunk in engine.generate_stream(
        [{"role": "user", "content": "介绍一下自己"}],
        GenerateParams(max_new_tokens=400, ignore_eos=True),
    ):
        print(chunk.delta, end="")

    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
