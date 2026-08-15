# 使用样本

`examples/` 目录下的示例程序，覆盖单模型推理、多模型批量、性能基准、量化研究等场景。运行前请先完成 [启动指南](quickstart.md) 中的编译与配置。

| 文件 | 说明 |
| --- | --- |
| [basic_generation.py](../examples/basic_generation.py) | 最简单的模型推理示例，通过 `BatchRunner` 单模型生成，可修改提示词查看整体运行情况 |
| [llm_runner.py](../examples/llm_runner.py) | 基于 `LLMEngine` 的推理引擎示例，支持流式生成与多模型批量配置，详见 [启动指南](quickstart.md) |
| [run_from_config.py](../examples/run_from_config.py) | 从 YAML 配置文件加载多模型并运行推理，演示 `BatchConfig → get_model → from_model_entry` 的两步加载流程 |
| [run_minimind.py](../examples/run_minimind.py) | 运行 minimind-3-moe 小模型，需要设置环境变量 `MINIMIND_MODEL_PATH` |
| [sync_engine_example.py](../examples/sync_engine_example.py) | `SyncEngine` 同步推理引擎示例，单请求流式生成并打印性能指标 |
| [benchmark.py](../examples/benchmark.py) | 性能基准测试，对 `BatchRunner` 做 warmup 后测速 |
| [qwen3_quant_study.py](../examples/qwen3_quant_study.py) | Qwen3 量化研究脚本 |
| [train/](../examples/train/) | 训练相关示例：数据集转换（`convert_to_jsonl.py`）与训练脚本（`train.py`） |

## 模型路径配置

不同示例的模型来源：

- 基于 Qwen3 的示例读取环境变量 `MODEL_PATH`（在 `.env` 中配置）
- `run_minimind.py` 读取环境变量 `MINIMIND_MODEL_PATH`，模型可从[魔搭](https://modelscope.cn/models/gongjy/minimind-3-moe)或 [HuggingFace](https://huggingface.co/jingyaogong/minimind-3-moe) 下载
- 基于配置文件的示例（`run_from_config.py`、`llm_runner.py`）将 `examples/configs/*.yaml` 中的 `path` 改为自己的模型下载路径
