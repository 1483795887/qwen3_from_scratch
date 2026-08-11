import torch

from qwen3_from_scratch.inference.context import get_forward_context
from qwen3_from_scratch.inference.model_worker import ModelWorker
from qwen3_from_scratch.inference.sequence import Sequence


def test_build_context_decode_position_is_last_token_index():
    """回归: decode 步当前输入 token 已被 post_process 追加进 token_ids,
    其位置 = len(seq) - 1 (而不是 len(seq))。

    旧的 len(seq) 使每个 decode token 的 rope 位置都比实际大 1,
    单独劣化生成质量 (非分歧源, 但 single/batch 同样受影响)。
    """
    # 不加载真实模型, 只测 context 构建: 跳过 __init__, 只给 device
    worker = ModelWorker.__new__(ModelWorker)
    worker.device = torch.device("cpu")
    seqs = [
        Sequence(list(range(5)), max_new_tokens=10),
        Sequence(list(range(3)), max_new_tokens=10),
    ]
    for idx, seq in enumerate(seqs):
        # 模拟调度后状态: 已分配页 + 一轮 post_process 追加生成的 token
        seq.block_tables = [idx]
        seq.token_ids.append(100)
    worker.build_context_decode(seqs)
    context = get_forward_context()
    assert context.position_ids.tolist() == [len(s) - 1 for s in seqs]
