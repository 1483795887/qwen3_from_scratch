# Qwen3 From Scratch

一个从零开始实现的 Qwen3 大语言模型，包含手写 CUDA 算子、性能优化和完整推理流程。

## 文档导航

- [📖 技术博客](docs/blog.md) — 从零开始写 Qwen3 系列文章
- [🚀 启动指南](docs/quickstart.md) — 编译、配置与运行
- [🧪 使用样本](docs/examples.md) — 示例程序说明

## 技术实现

### 模型架构
- 完整实现 Qwen3 模型：Embedding、Attention、MLP、RMS Norm
- 支持加载 HuggingFace 官方预训练权重
- 模块化设计：通过 ComponentFactory 切换不同实现

### CUDA 算子优化
- **RMS Norm 融合算子**：融合归一化+缩放+偏置，减少显存访问
- **KV Cache 优化**：优化自回归推理的显存占用和速度
- **更多算子开发中**：Flash Attention (Triton)、Rotary Embedding 等

### 项目结构
```
qwen3_from_scratch/
├── src/qwen3_from_scratch/  # Python模型实现
│   ├── models/              # 模型架构
│   ├── layers/              # 各层实现
│   └── kernels/             # CUDA算子
├── csrc/                    # C++/CUDA源码
├── examples/                # 使用示例
├── test/                    # 测试用例
└── exps/                    # 性能实验
    └── reports/             # 性能报告
```

## 验证方法
基于 `transformers` 库作为基准，在相同输入和相同参数的条件下，对比输出结果的一致性，验证各组件实现的正确性。

每个组件会使用 ComponentFactory 进行创建，基于 ModelConfig 配置每个组件的参数，包括具体实现、参数等

每个测试会对不同组件、cpu和cuda都运行，如果不支持cuda会自动跳过

## 引用
- [transformers库](https://github.com/huggingface/transformers)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [minimind](https://github.com/jingyaogong/minimind)
