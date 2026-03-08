# Qwen3 From Scratch

## 概述
本项目是一个用于深入研究大语言模型基本原理的实践项目。通过从零开始手写实现 Qwen3 模型，深入理解大模型的架构设计、推理机制和性能优化方法。

## 主要功能
- **模型实现**：完整实现 Qwen3 模型架构
- **参数加载**：支持加载官方预训练权重并进行推理
- **KVCache 优化**：实现 KVCache 机制以提升推理效率
- **算子实现**：使用 C++/CUDA 手写核心算子
- **性能评估**：测量模型的性能指标和推理准确性

## 验证方法
基于 `transformers` 库作为基准，在相同输入和相同参数的条件下，对比输出结果的一致性，验证各组件实现的正确性。

## 编译
本项目分为python代码和C++/Cuda算子代码，前者通过uv控制，后者通过cmake控制

首先使用 `uv sync` 安装依赖并生成虚拟环境，至少需要 `torch` 库，然后使用 `uv pip install -e .`安装python项目，这样才能使用 `from qwen3_from_scratch` 引用代码

安装完依赖后使用 `cmake -B build` 进行 cmake 配置，它会使用 uv 获取 torch、python 等库的安装路径，然后使用 `cmake --build build`启动编译，编译完成后会在 `src/qwen3_from_scratch/kernels` 下生成一个 ops 的动态链接库的软链接，直接使用 `from qwen3_from_scratch.kernels import ops` 即可导入使用

## 启动
需要自己从Hugging Face或者魔搭上下载Qwen3的模型，复制 `.env.example` 为 `.env`，设置Qwen3模型的路径

启动入口主要有两个：
- test 下的测试用例，使用 `uv run pytest` 可以启动
- examples/basic_generation.py，一个简单的模型推理例子，可以修改提示词查看模型整体的运行情况

# 引用
- [transformers库](https://github.com/huggingface/transformers)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)