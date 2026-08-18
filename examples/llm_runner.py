import asyncio
from pathlib import Path

from qwen3_from_scratch.inference.llm_engine import LLMEngine


def get_config_path():
    return Path(__file__).parent / "configs" / "batch2.yaml"


async def main():
    engine = LLMEngine(get_config_path(), "qwen3-0.6b")
    await engine.warmup()
    async for chunk in engine.generate_stream(
        [{"role": "user", "content": "介绍一下自己"}], max_new_tokens=400, ignore_eos=True
    ):
        print(chunk.delta, end="")

    print("\n\n--- 性能指标 ---")
    print(f"首词延迟 (TTFT): {chunk.metrics.ttft:.4f}s")
    print(f"生成词元数: {chunk.metrics.token_count}")
    print(f"平均速度: {chunk.metrics.tps:.2f} tokens/s")
    print(f"总耗时: {chunk.metrics.total_elapsed:.4f}s")
    engine.close()


if __name__ == "__main__":
    asyncio.run(main())
