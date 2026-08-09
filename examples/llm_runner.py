from qwen3_from_scratch.inference.llm_engine import LLMEngine
import asyncio
from pathlib import Path

def get_config_path():
    return Path(__file__).parent / "configs" / "batch2.yaml"

async def main():
    engine = LLMEngine(get_config_path(), "qwen3-0.6b")
    async for delta in engine.generate_stream([{"role": "user", "content": "介绍一下自己"}]):
        print(delta, end='')
    engine.close()

if __name__ == '__main__':
    asyncio.run(main())
