"""从 YAML 配置文件加载多模型并运行推理。

演示 BatchConfig → get_model → from_model_entry 的两步加载流程。
使用前请修改 examples/configs/batch.yaml 中的模型路径。
"""

from qwen3_from_scratch.factory.batch_config import load_batch_config
from qwen3_from_scratch.inference.engine import BatchRunner

CONFIG_PATH = "examples/configs/batch.yaml"


def main():
    # 第一步：加载配置文件，全量校验
    config = load_batch_config(CONFIG_PATH)

    print(f"配置文件加载成功，可用模型: {config.list_model_names()}\n")

    # 第二步：选择模型，获取已合并的 ResolvedModelEntry
    model_name = "qwen3-0.6b"
    entry = config.get_model(model_name)
    print(
        f"已选择模型: {entry.name}\n"
        f"  路径: {entry.path}\n"
        f"  设备: {entry.device}\n"
        f"  max_len: {entry.max_len}\n"
        f"  采样参数: temperature={entry.generation.temperature}, "
        f"max_new_tokens={entry.generation.max_new_tokens}\n"
    )

    # 第三步：从 ResolvedModelEntry 构建引擎
    engine = BatchRunner.from_model_entry(entry)

    # 也可以一步到位：BatchRunner.from_config(CONFIG_PATH, model_name)

    # 推理——不传 max_new_tokens 时用配置文件的默认值
    for token in engine.generate_stream(
        [{"role": "user", "content": "你有什么特长？"}],
    ):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
