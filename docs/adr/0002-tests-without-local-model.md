# 测试不依赖本地模型文件：预置配置 + 随机构造 + opt-in 真实权重

测试默认不再读取本地 `MODEL_PATH`。模型配置使用预置 `ModelConfig()`（Qwen3-0.6B 形状），并由 `ModelConfig.to_transformers_config()` 推导 transformers `Qwen3Config`，二者一一对应（单一数据源）。与 transformers 的数值对比用例随机初始化参数（需要时经 bf16 表达，如 paged attention 用例），不再加载真实权重。仅「验证真实权重加载」的用例经 `real_model_path` / `real_model_config` fixture opt-in，未设 `MODEL_PATH` 则整体跳过。`examples/`、`exps/` 维持 `MODEL_PATH` 不变。

## Considered Options

- **维持现状**：所有用例依赖 `MODEL_PATH` 读取 config.json 和权重。优点：数值对比使用真实权重，保真度最高。缺点：本地没有模型文件时几乎全部测试无法运行，CI 无从谈起。
- **预置配置 + 随机构造 + opt-in 真实权重**（采纳）：shape / 机制用例与真实模型解耦，CI 可全量运行；真实权重加载的正确性由开发者本地设置 `MODEL_PATH` 时验证。缺点：真实权重下的端到端精度不在 CI 内覆盖。

## Consequences

- `test/conftest.py` 默认 fixture 不再依赖 `MODEL_PATH`；新增 `real_model_path` / `real_model_config`（真实权重用例专用，未设置则 skip）。
- `test_qwen3.py` 的完整模型用例使用专属 `_SMALL_MODEL_CONFIG` 缩小配置以提速。
- `ModelConfig.to_transformers_config()` 成为 ModelConfig ↔ Qwen3Config 对应关系的唯一入口。
- 依赖真实权重的用例收敛为 2 个纯加载验证（`test_ffn_load`、`test_parameter_loading`，后者恒 skip）；其余 shape / 对比用例均使用预置配置 + 随机构造参数。
