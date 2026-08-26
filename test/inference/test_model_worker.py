import torch

from qwen3_from_scratch.inference.context import get_forward_context
from qwen3_from_scratch.inference.model_runner.model_worker import ModelWorker
from qwen3_from_scratch.inference.sequence import Sequence


def _make_worker() -> ModelWorker:
    """不加载真实模型，只测 context 构建：跳过 __init__，只给 device。"""
    worker = ModelWorker.__new__(ModelWorker)
    worker.device = torch.device("cpu")
    return worker


def test_build_context_decode_position_is_last_token_index():
    """回归: decode 步当前输入 token 已被 post_process 追加进 token_ids,
    其位置 = len(seq) - 1 (而不是 len(seq))。
    """
    worker = _make_worker()
    seqs = [
        Sequence(list(range(5)), max_new_tokens=10),
        Sequence(list(range(3)), max_new_tokens=10),
    ]
    for idx, seq in enumerate(seqs):
        # 模拟调度后状态: 已分配页 + 一轮 post_process 追加生成的 token
        seq.block_tables = [idx]
        seq.token_ids.append(100)
        seq.cached_len = len(seq.prompts)
        seq.num_tokens = 1
    worker.build_context(seqs)
    context = get_forward_context()
    assert context.position_ids.tolist() == [len(s) - 1 for s in seqs]


def test_build_context_chunked_prefill_positions_start_at_cached_len():
    """分段 prefill：第二段的 positions 从 cached_len 起，q/kv 长度正确。"""
    worker = _make_worker()
    seq = Sequence(list(range(5)), max_new_tokens=10)
    seq.block_tables = [0]
    seq.cached_len = 2  # 前 2 token 已缓存
    seq.num_tokens = 2  # 本段处理 token[2:4]
    worker.build_context([seq])
    context = get_forward_context()
    assert context.position_ids.tolist() == [2, 3]
    assert context.cum_seq_lens_q.tolist() == [0, 2]
    assert context.cum_seq_lens_kv.tolist() == [0, 4]  # 2 缓存 + 2 新写


def test_build_context_mixed_batch():
    """混合 batch：decode 与 prefill 分段共存。"""
    worker = _make_worker()
    decode_seq = Sequence(list(range(5)), max_new_tokens=10)
    decode_seq.block_tables = [0]
    decode_seq.token_ids.append(100)
    decode_seq.cached_len = 5
    decode_seq.num_tokens = 1

    prefill_seq = Sequence(list(range(8)), max_new_tokens=10)
    prefill_seq.block_tables = [1]
    prefill_seq.cached_len = 3
    prefill_seq.num_tokens = 3  # 处理 token[3:6]

    worker.build_context([decode_seq, prefill_seq])
    context = get_forward_context()
    # decode 位置 5，prefill 位置 3,4,5
    assert context.position_ids.tolist() == [5, 3, 4, 5]
    assert context.cum_seq_lens_q.tolist() == [0, 1, 4]
    # decode kv=6（token_ids 长度），prefill kv=6（3 缓存 + 3 新写）
    assert context.cum_seq_lens_kv.tolist() == [0, 6, 12]


def test_build_context_full_hit_decode_uses_last_prompt_token():
    """全命中 decode：输入取 token_ids[-1]（=prompts[-1]），不写 KV。"""
    worker = _make_worker()
    seq = Sequence(list(range(5)), max_new_tokens=10)
    seq.block_tables = [0]
    seq.cached_len = 5  # 全命中，prompt 全在 KV
    seq.num_tokens = 1
    worker.build_context([seq])
    context = get_forward_context()
    assert context.position_ids.tolist() == [4]
    assert context.cum_seq_lens_q.tolist() == [0, 1]
    assert context.cum_seq_lens_kv.tolist() == [0, 5]
    # 全命中时无需写 KV，slot_mapping 为空
    assert context.slot_mapping.numel() == 0
    # 输入是最后一个 prompt token
    inputs = worker.build_inputs([seq])
    assert inputs.tolist() == [4]
