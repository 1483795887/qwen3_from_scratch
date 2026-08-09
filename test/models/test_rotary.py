"""RotaryEmbedding + get_rope 预计算模块测试。"""

import torch

from qwen3_from_scratch.models.rotary import get_rope


class TestRotaryEmbedding:
    def test_cos_sin_cache_shape(self):
        """cos_sin_cache 应为 (max_position, 1, head_dim)。

        inv_freq 长度 = head_dim//2，cos/sin 各 (max_pos, head_dim//2)，
        cat 后 (max_pos, head_dim)。
        """
        rotary = get_rope(128, 128, 40960, 100000.0)
        assert rotary.cos_sin_cache.shape == (40960, 1, 128)

    def test_cos_sin_cache_values(self):
        """cos 部分与手动计算 torch.outer(arange, inv_freq).cos() 一致。"""
        head_dim = 128
        max_pos = 1024
        base = 100000.0

        rotary = get_rope(head_dim, head_dim, max_pos, base)

        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        t = torch.arange(max_pos, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        expected_cos = freqs.cos()

        actual_cos = rotary.cos_sin_cache[:, 0, : head_dim // 2]
        assert torch.allclose(actual_cos, expected_cos, atol=1e-5)

    def test_cos_sin_cache_sin_values(self):
        """sin 部分与手动计算一致。"""
        head_dim = 128
        max_pos = 1024
        base = 100000.0

        rotary = get_rope(head_dim, head_dim, max_pos, base)

        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        t = torch.arange(max_pos, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        expected_sin = freqs.sin()

        actual_sin = rotary.cos_sin_cache[:, 0, head_dim // 2 :]
        assert torch.allclose(actual_sin, expected_sin, atol=1e-5)


class TestGetRopeCache:
    def test_same_params_return_same_object(self):
        """相同参数调用 get_rope 两次返回同一对象。"""
        r1 = get_rope(128, 128, 512, 100000.0)
        r2 = get_rope(128, 128, 512, 100000.0)
        assert r1 is r2

    def test_different_params_return_different_object(self):
        """不同参数返回不同对象。"""
        r1 = get_rope(128, 128, 512, 100000.0)
        r2 = get_rope(64, 64, 512, 100000.0)
        assert r1 is not r2
