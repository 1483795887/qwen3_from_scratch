# 配置文件格式选择 YAML

多模型加载配置文件使用 YAML 格式，而非 JSON 或 TOML。

## Considered Options

- **JSON**：与现有 `config.json` / `generation_config.json` 一致，无新依赖。但不支持注释，多模型列表嵌套时可读性差。
- **YAML**（采纳）：支持注释、锚点（可复用配置块）、更简洁的嵌套表达。推理框架惯例（vLLM、TGI 均用 YAML）。`transformers` 已间接依赖 PyYAML，仅需显式声明。
- **TOML**：Python 原生支持（`tomllib`），但多模型列表（数组里嵌套表）的表达不如 YAML 直观。

## Consequences

- `pyproject.toml` 需显式添加 `pyyaml` 依赖（此前通过 `transformers` 间接依赖）。
- 配置文件支持 `#` 注释，便于调参时标注意图。
- 组件配置支持两种写法：简写（`mlp: "moe"`）和展开（`mlp: {name: "my_op", kwargs: {scale: 1.0}}`），加载时按值类型判断。
