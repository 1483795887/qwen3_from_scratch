import os

from qwen3_from_scratch.inference.session import InferenceSession
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()


def main():
    model_path = os.environ.get("MODEL_PATH")
    session = InferenceSession(model_path, device="cpu", max_len=2048)

    result = session.generate_from_messages(
        "介绍一下你自己",
        max_new_tokens=400,
        temperature=0.7,
        stream=True,
    )
    for token in result:
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
