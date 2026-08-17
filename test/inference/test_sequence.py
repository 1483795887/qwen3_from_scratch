from qwen3_from_scratch.inference.sequence import Sequence


def test_num_tokens_defaults_to_zero():
    seq = Sequence([1, 2, 3], max_new_tokens=10)
    assert seq.num_tokens == 0


def test_is_prefill_true_before_any_prefill():
    # cached_len = 0 < len(prompts)
    seq = Sequence([1, 2, 3], max_new_tokens=10)
    assert seq.is_prefill


def test_is_prefill_true_during_partial_prefill():
    # 0 < cached_len < len(prompts)：分段 prefill 进行中
    seq = Sequence([1, 2, 3, 4, 5], max_new_tokens=10)
    seq.cached_len = 2
    assert seq.is_prefill


def test_is_prefill_false_after_full_prefill():
    # cached_len == len(prompts)：prompt 已填满，进入 decode
    seq = Sequence([1, 2, 3], max_new_tokens=10)
    seq.cached_len = 3
    assert not seq.is_prefill


def test_is_prefill_false_during_decode():
    # cached_len > len(prompts)：decode 阶段
    seq = Sequence([1, 2, 3], max_new_tokens=10)
    seq.cached_len = 4
    assert not seq.is_prefill


def test_generated_lens_excludes_prompts():
    seq = Sequence([1, 2, 3], max_new_tokens=10)
    assert seq.generated_lens == 0
    seq.token_ids.append(9)
    seq.token_ids.append(8)
    assert seq.generated_lens == 2
