"""ModelContext 字段清理测试。"""

import pytest

from qwen3_from_scratch.inference.context import ModelContext


def test_model_context_no_position_embeddings():
    """ModelContext 不应再包含 position_embeddings 字段。"""
    ctx = ModelContext()
    assert not hasattr(ctx, "position_embeddings")


def test_model_context_no_is_prefill():
    """ModelContext 不应再包含 is_prefill 字段。"""
    ctx = ModelContext()
    assert not hasattr(ctx, "is_prefill")


def test_model_context_no_num_tokens():
    """ModelContext 不应再包含 num_tokens 字段。"""
    ctx = ModelContext()
    assert not hasattr(ctx, "num_tokens")


def test_position_embeddings_class_removed():
    """PositionEmbeddings 类应已删除，不可导入。"""
    with pytest.raises(ImportError):
        from qwen3_from_scratch.inference.context import (
            PositionEmbeddings,  # noqa: F401
        )


def test_model_context_retains_required_fields():
    """清理后仍保留必要字段。"""
    ctx = ModelContext()
    assert hasattr(ctx, "dtype")
    assert hasattr(ctx, "use_cache")
    assert hasattr(ctx, "kv_cache")
    assert hasattr(ctx, "position_ids")
    assert hasattr(ctx, "cache_position")
    assert hasattr(ctx, "block_tables")
    assert hasattr(ctx, "block_size")
    assert hasattr(ctx, "cum_seq_lens_q")
    assert hasattr(ctx, "cum_seq_lens_kv")
    assert hasattr(ctx, "slot_mapping")
