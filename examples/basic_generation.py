import os

from qwen3_from_scratch.inference.engine import InferenceEngine
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()


def main():
    model_path = os.environ.get("MODEL_PATH")
    engine = InferenceEngine.from_path(model_path, device="cpu", max_len=2048)

    for token in engine.generate_stream(
        [{"role": "user", "content": "介绍一下你自己"}],
        max_new_tokens=400,
    ):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
