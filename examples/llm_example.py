from qwen3_from_scratch.inference.engine.llm import LLM


def get_config_path():
    from pathlib import Path

    return Path(__file__).parent / "configs" / "batch2.yaml"


def main():
    llm = LLM(get_config_path(), "qwen3-0.6b", log_interval=10)
    llm.warmup()

    print(
        llm.generate(
            [{"role": "user", "content": "介绍一下你自己"}],
            enable_thinking=True,
        )
    )

    for res in llm.batch_generate(
        [
            [{"role": "user", "content": "介绍一下你自己"}],
            [{"role": "user", "content": "你有哪些技能"}],
            [{"role": "user", "content": "1+1等于几"}],
        ]
    ):
        print(res)


if __name__ == "__main__":
    main()
