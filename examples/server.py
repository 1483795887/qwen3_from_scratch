"""
OpenAI 兼容 API 服务器，支持工具调用（tool calling）。

需要执行
uv pip install fastapi fastapi-openai-compat uvicorn

工具调用说明：
- 请求体里带 ``tools``（OpenAI function-calling 格式）时，会透传给 Qwen3 的
  chat template，模型据此在输出中生成形如 {"name": ..., "arguments": ...}
  的工具调用。
- 服务器在流式文本里解析 TOOL_CALL_OPEN / TOOL_CALL_CLOSE 标记，把其中的
  JSON 转成 OpenAI 的 ``tool_calls`` 字段返回，``finish_reason`` 置为
  ``tool_calls``。
- 同时把 Qwen3 的思维块（THINK_OPEN ... THINK_CLOSE）路由到
  ``reasoning_content``，避免污染正文。
- 默认带 ``tools`` 时关闭思维（``enable_thinking=False``），输出更干净；可在
  请求体里用 ``enable_thinking: true`` 覆盖（也支持 chat_template_kwargs）。
- 同时支持流式（``stream: true``，SSE）与非流式（默认，单个 JSON）两种模式。
"""

import json
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4
import argparse
import time

from fastapi import FastAPI

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

from qwen3_from_scratch.inference.logger import get_logger

logger = get_logger(__name__)

# Qwen3 输出中使用的特殊标记。它们在词表里被标记为 special=false，引擎解码时
# skip_special_tokens=True 也会保留；且每个标记是单个 token，引擎逐 token
# 解码时会作为一个完整 chunk 到达，故可用相等判断识别。
# 这里用拼接构造，避免源码里出现会被误解析的字面量标记。
# 不少CodeAgent遇到这个关键词都会出现异常
TOOL_CALL_OPEN = "<" + "tool_call" + ">"
TOOL_CALL_CLOSE = "</" + "tool_call" + ">"
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


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


# ── 工具调用流式解析 ──────────────────────────────────────────────


@dataclass
class ParseEvent:
    """解析器每次 feed 产出的事件。"""

    kind: str  # "content" | "reasoning" | "tool_call"
    text: str = ""
    tool_call: dict | None = None  # {"id", "name", "arguments"}


def _load_json_lenient(text: str):
    """宽松 JSON 解析：先整体解析，失败则抽取第一个 {...} 再试。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


class ToolCallStreamParser:
    """把 Qwen3 流式输出（token 级 delta）解析成 OpenAI 语义事件。

    状态机：
      - content    : 普通文本 -> content
      - reasoning  : THINK_OPEN ... THINK_CLOSE 内 -> reasoning_content
      - tool_call  : TOOL_CALL_OPEN ... TOOL_CALL_CLOSE 内 -> 累积 JSON，
                     闭合后产出 tool_call

    依赖前提：工具调用/思维标记是词表中的单 token，引擎逐 token 解码时它们
    会作为完整字符串到达，因此用相等判断即可识别。
    """

    def __init__(self):
        self._state = "content"
        self._tool_buffer = ""

    def feed(self, delta: str) -> list[ParseEvent]:
        events: list[ParseEvent] = []
        if delta == THINK_OPEN:
            self._state = "reasoning"
            return events
        if delta == THINK_CLOSE:
            self._state = "content"
            return events
        if delta == TOOL_CALL_OPEN:
            self._state = "tool_call"
            self._tool_buffer = ""
            return events
        if delta == TOOL_CALL_CLOSE:
            tc = self._parse_tool_call(self._tool_buffer)
            if tc is not None:
                events.append(ParseEvent(kind="tool_call", tool_call=tc))
            elif self._tool_buffer:
                # JSON 解析失败，原样作为 content 返回，避免丢内容
                events.append(
                    ParseEvent(kind="content", text=self._tool_buffer)
                )
            self._state = "content"
            self._tool_buffer = ""
            return events

        if self._state == "tool_call":
            self._tool_buffer += delta
        elif self._state == "reasoning":
            if delta:
                events.append(ParseEvent(kind="reasoning", text=delta))
        else:
            if delta:
                events.append(ParseEvent(kind="content", text=delta))
        return events

    @staticmethod
    def _parse_tool_call(raw: str) -> dict | None:
        text = raw.strip()
        if not text:
            return None
        obj = _load_json_lenient(text)
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        arguments = obj.get("arguments", {})
        # OpenAI 规范：arguments 是 JSON 字符串
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "id": f"call_{uuid4().hex[:24]}",
            "name": obj["name"],
            "arguments": arguments,
        }


# ── 引擎 ──────────────────────────────────────────────────────────


class FakeEngine:
    """无模型环境下的假引擎，用于联调工具调用链路。"""

    def __init__(self, config_path: str, model_name: str):
        logger.info(f"init {config_path} {model_name}")

    async def warmup(self):
        logger.info("warmup")

    async def generate_stream(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        if tools:
            # 模拟一次工具调用：逐块吐出 TOOL_CALL_OPEN / JSON / TOOL_CALL_CLOSE
            first = tools[0]
            fn = (
                first.get("function", first) if isinstance(first, dict) else {}
            )
            name = fn.get("name", "get_weather")
            payload = json.dumps(
                {"name": name, "arguments": {"city": "Beijing"}},
                ensure_ascii=False,
            )
            for piece in [
                TOOL_CALL_OPEN,
                " " + payload + " ",
                TOOL_CALL_CLOSE,
            ]:
                yield StreamChunk(piece, PerfMetrics(0, 0, 0, 0))
            return
        content = "hello world"
        for c in content.split():
            yield StreamChunk(c, PerfMetrics(0, 0, 0, 0))

    def close(self):
        pass


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


def create_app(config_path: str, model_name: str, use_real_model: bool):
    app = FastAPI(
        lifespan=lambda app_inst: lifespan(
            app_inst, config_path, model_name, use_real_model
        )
    )

    def _gen_id() -> str:
        return f"{model_name}-{uuid4().hex[:24]}"

    def _chunk(
        resp_id: str,
        model: str,
        *,
        delta: Message,
        finish_reason: str | None = None,
    ) -> ChatCompletion:
        return ChatCompletion(
            id=resp_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(index=0, delta=delta, finish_reason=finish_reason)
            ],
        )

    def _build_engine_stream(model: str, messages: list[dict], body: dict):
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        max_tokens = body.get("max_tokens")
        # 默认：带 tools 时关闭思维，输出更干净；否则沿用模型默认（思维开启）
        enable_thinking: bool | None = False if tools else None
        ctk = body.get("chat_template_kwargs") or {}
        if "enable_thinking" in ctk:
            enable_thinking = ctk["enable_thinking"]
        if "enable_thinking" in body:
            enable_thinking = body["enable_thinking"]
        return app.state.engine.generate_stream(
            messages,
            max_new_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
        )

    async def completions(
        model: str, messages: list[dict], body: dict
    ) -> CompletionResult:
        logger.info(json.dumps(messages, ensure_ascii=False))
        if body.get("stream", False):
            # 返回异步生成器，router 会包成 SSE
            return _stream_completion(model, messages, body)
        return await _nonstream_completion(model, messages, body)

    async def _stream_completion(
        model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[ChatCompletion, None]:
        resp_id = _gen_id()
        # 首块：角色
        yield _chunk(resp_id, model, delta=Message(role="assistant"))

        parser = ToolCallStreamParser()
        has_tool_calls = False
        tool_index = 0
        gen = _build_engine_stream(model, messages, body)
        async for item in gen:
            for ev in parser.feed(item.delta):
                if ev.kind == "content" and ev.text:
                    yield _chunk(
                        resp_id,
                        model,
                        delta=Message(role="assistant", content=ev.text),
                    )
                elif ev.kind == "reasoning" and ev.text:
                    yield _chunk(
                        resp_id,
                        model,
                        delta=Message(
                            role="assistant", reasoning_content=ev.text
                        ),
                    )
                elif ev.kind == "tool_call":
                    has_tool_calls = True
                    tc = ev.tool_call
                    yield _chunk(
                        resp_id,
                        model,
                        delta=Message(
                            role="assistant",
                            tool_calls=[
                                {
                                    "index": tool_index,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"],
                                    },
                                }
                            ],
                        ),
                    )
                    tool_index += 1

        finish_reason = "tool_calls" if has_tool_calls else "stop"
        yield _chunk(
            resp_id,
            model,
            delta=Message(role="assistant"),
            finish_reason=finish_reason,
        )

    async def _nonstream_completion(
        model: str, messages: list[dict], body: dict
    ) -> ChatCompletion:
        resp_id = _gen_id()
        parser = ToolCallStreamParser()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict] = []
        gen = _build_engine_stream(model, messages, body)
        async for item in gen:
            for ev in parser.feed(item.delta):
                if ev.kind == "content":
                    content_parts.append(ev.text)
                elif ev.kind == "reasoning":
                    reasoning_parts.append(ev.text)
                elif ev.kind == "tool_call":
                    tc = ev.tool_call
                    tool_calls.append(
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                    )

        content = "".join(content_parts) or None
        reasoning = "".join(reasoning_parts) or None
        finish_reason = "tool_calls" if tool_calls else "stop"
        message = Message(
            role="assistant",
            content=content,
            reasoning_content=reasoning,
            tool_calls=tool_calls or None,
        )
        return ChatCompletion(
            id=resp_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(index=0, message=message, finish_reason=finish_reason)
            ],
        )

    def list_models() -> list[str]:
        return [model_name]

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
