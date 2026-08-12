"""
需要执行
uv pip install fastapi fastapi-openai-compat uvicorn
"""

import json

from fastapi import FastAPI
from contextlib import asynccontextmanager
import argparse
from qwen3_from_scratch.inference.llm_engine import (
    LLMEngine,
    StreamChunk,
    PerfMetrics,
)
from fastapi_openai_compat import (
    CompletionResult,
    Message,
    Choice,
    ChatCompletion,
    create_openai_router,
)
import time

from qwen3_from_scratch.inference.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="配置文件地址", required=True)
    parser.add_argument("--model", help="模型名称", required=True)
    parser.add_argument(
        "--use_real_model",
        action="store_true",
        help="开启真实运行",
        default=False,
    )
    return parser.parse_args()


class FakeEngine:
    def __init__(self, config_path: str, model_name: str):
        logger.info(f"init {config_path} {model_name}")

    async def warmup(self):
        logger.info("warmup")

    async def generate_stream(self, messages: dict, max_new_tokens: int):
        content = "hello world"
        for c in content.split():
            yield StreamChunk(c, PerfMetrics(0, 0, 0, 0))


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    config_path: str,
    model_name: str,
    use_real_model: bool = False,
):
    engine = (
        LLMEngine(config_path, model_name)
        if use_real_model
        else FakeEngine(config_path, model_name)
    )

    await engine.warmup()
    app.state.engine = engine
    yield
    app.state.engine = None
    engine.close()


def create_app(config_path: str, mode_name: str, use_real_model: bool):
    app = FastAPI(
        lifespan=lambda app_inst: lifespan(
            app_inst, config_path, mode_name, use_real_model
        )
    )

    async def completions(
        model: str, messages: list[dict], body: dict
    ) -> CompletionResult:
        logger.info(json.dumps(messages, ensure_ascii=False))
        yield ChatCompletion(
            id="id-123",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    delta=Message(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        max_tokens = body.get("max_tokens")

        async for item in app.state.engine.generate_stream(
            messages, max_new_tokens=max_tokens
        ):
            yield ChatCompletion(
                id="id-123",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    Choice(
                        index=0,
                        delta=Message(content=item.delta, role="assistant"),
                        finish_reason=None,
                    )
                ],
            )
        yield ChatCompletion(
            id="id-123",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    delta=Message(role="assistant"),
                    finish_reason="stop",
                )
            ],
        )

    def list_models() -> list[str]:
        return app.state.engine.config.list_model_names()

    router = create_openai_router(
        list_models=list_models, run_completion=completions
    )
    app.include_router(router)
    return app


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    app = create_app(args.config_path, args.model, args.use_real_model)
    uvicorn.run(app, host="0.0.0.0", port=8889)
