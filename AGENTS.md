# 概述
这是一个基于 pytorch 和 triton 的 qwen3 模型

# 环境
使用 uv ，而不是直接使用 python 和 pip

使用 uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple 加速安装

triton 只能在 gpu 上运行，涉及它的测试需要在开头判断cuda，跳过不支持 cuda 的环境