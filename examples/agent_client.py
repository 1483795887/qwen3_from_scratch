# uv pip install openai

import asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

# -------------------------- 模拟工具实现 --------------------------
def get_weather(city: str):
    """模拟天气工具"""
    return f"【模拟工具返回】{city} 当前温度 26℃，晴天"

tool_map = {
    "get_weather": get_weather
}

async def run_stream_agent(base_url: str, model: str, user_query: str):
    client = AsyncOpenAI(
        api_key="dummy",
        base_url='http://127.0.0.1:8889/v1'
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市当前天气",
                "parameters": {
                    "type": "object",
                    "required": ["city"],
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    }
                }
            }
        }
    ]

    messages = [{"role": "user", "content": user_query}]
    max_round = 3  # 防止死循环，最多多轮次数

    for round_idx in range(max_round):
        print(f"\n===== 第{round_idx+1}轮请求模型 =====")
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True
        )

        collected_tool_calls = []
        content_buf = ""

        async for chunk in resp:
            chunk: ChatCompletionChunk
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 普通文本流打印
            if delta.content:
                content_buf += delta.content
                print(delta.content, end="", flush=True)

            # 流式分片组装tool_calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    while len(collected_tool_calls) <= idx:
                        collected_tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    slot = collected_tool_calls[idx]
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function.name:
                        slot["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["function"]["arguments"] += tc.function.arguments

        print("\n")
        if not collected_tool_calls:
            print("✅本轮无工具调用，对话结束")
            break

        print(" 检测到工具调用：", collected_tool_calls)
        # 把assistant工具调用消息加入上下文
        assistant_msg = {
            "role": "assistant",
            "tool_calls": collected_tool_calls
        }
        messages.append(assistant_msg)

        # 逐个执行工具
        for call in collected_tool_calls:
            func_name = call["function"]["name"]
            args_str = call["function"]["arguments"]
            print(f"🔧执行工具 {func_name}, args={args_str}")
            import json
            try:
                args = json.loads(args_str)
                func = tool_map[func_name]
                tool_result = func(**args)
            except Exception as e:
                tool_result = f"工具调用异常: {str(e)}"

            # 添加tool返回消息
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_result
            })
        # 进入下一轮，模型会看到工具返回继续生成回答

async def main():
    await run_stream_agent(
        base_url="http://127.0.0.1:8889/v1",
        model="qwen3-0.6b",
        user_query="调用工具帮我查一下今天北京的天气" # 需要明确要求调用工具，否则可能不会调用工具
    )

if __name__ == "__main__":
    asyncio.run(main())