import os

from qwen3_from_scratch.factory.config import ComponentConfig
from qwen3_from_scratch.inference.engine import BatchRunner
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()

# 从魔搭 (modelscope.cn/models/gongjy/minimind-3-moe)
# 或 HuggingFace (huggingface.co/jingyaogong/minimind-3-moe) 下载模型，
# 将下载路径设置到环境变量 MINIMIND_MODEL_PATH（或在 .env 中配置）
MODEL_PATH = os.environ.get("MINIMIND_MODEL_PATH")
if MODEL_PATH is None:
    raise RuntimeError(
        "请设置环境变量 MINIMIND_MODEL_PATH 为 minimind-3-moe 的下载路径\n"
        "下载地址：\n"
        "  魔搭:   https://www.modelscope.cn/models/gongjy/minimind-3-moe\n"
        "  HuggingFace: https://huggingface.co/jingyaogong/minimind-3-moe"
    )


def main():
    engine = BatchRunner.from_path(
        MODEL_PATH,
        device="cpu",
        max_len=2048,
        components={"mlp": ComponentConfig("moe")},
    )

    for token in engine.generate_stream(
        [{"role": "user", "content": "你有什么特长？"}],
        max_new_tokens=2048,
        temperature=0.85,
    ):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
