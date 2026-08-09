"""BatchConfig 配置加载、校验、合并、组件解析的单元测试。

不依赖真实模型文件——用 tmp_path 创建假模型目录 + 假 config.json。
集成测试（from_config → Runner 推理）单独标记 skip。
"""

import json
import os

import pytest

from qwen3_from_scratch.factory.batch_config import (
    GenerationDefaults,
    GenerationOverrides,
    ResolvedModelEntry,
    _merge_generation,
    _parse_component,
    load_batch_config,
)

# ── 合并逻辑 ──────────────────────────────────


class TestMergeGeneration:
    """_merge_generation 深度合并测试。"""

    def test_none_overrides_returns_defaults_copy(self):
        """overrides=None 时返回 defaults 的副本，不是同一对象。"""
        defaults = GenerationDefaults(temperature=0.85, top_k=40)
        merged = _merge_generation(defaults, None)
        assert merged.temperature == 0.85
        assert merged.top_k == 40
        assert merged is not defaults

    def test_partial_override_only_overrides_set_fields(self):
        """模型级只覆盖显式声明的字段，其余继承全局。"""
        defaults = GenerationDefaults(
            temperature=0.85,
            top_k=40,
            top_p=1.0,
            do_sample=True,
            max_new_tokens=2048,
        )
        overrides = GenerationOverrides(temperature=0.7)
        merged = _merge_generation(defaults, overrides)
        assert merged.temperature == 0.7
        assert merged.top_k == 40
        assert merged.top_p == 1.0
        assert merged.do_sample is True
        assert merged.max_new_tokens == 2048

    def test_full_override_overrides_all(self):
        """所有字段都覆盖时，完全替换。"""
        defaults = GenerationDefaults()
        overrides = GenerationOverrides(
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=512,
        )
        merged = _merge_generation(defaults, overrides)
        assert merged.temperature == 0.5
        assert merged.top_k == 10
        assert merged.top_p == 0.9
        assert merged.do_sample is True
        assert merged.max_new_tokens == 512

    def test_none_values_do_not_override(self):
        """Optional 字段值为 None 时不覆盖（即'未声明'语义）。"""
        defaults = GenerationDefaults(temperature=0.85)
        overrides = GenerationOverrides(
            temperature=None,
            top_k=99,
        )
        merged = _merge_generation(defaults, overrides)
        assert merged.temperature == 0.85
        assert merged.top_k == 99


# ── 组件解析 ──────────────────────────────────


class TestParseComponent:
    """_parse_component 简写/展开两种格式测试。"""

    def test_shorthand_string(self):
        """str 值 → ComponentConfig(name=str)，无 kwargs。"""
        cc = _parse_component("moe")
        assert cc.name == "moe"
        assert cc.kwargs == {}

    def test_expanded_with_name_only(self):
        """dict 含 name、无 kwargs → kwargs 默认空。"""
        cc = _parse_component({"name": "my_op"})
        assert cc.name == "my_op"
        assert cc.kwargs == {}

    def test_expanded_with_kwargs(self):
        """dict 含 name + kwargs → 完整解析。"""
        cc = _parse_component({"name": "my_op", "kwargs": {"scale": 1.0}})
        assert cc.name == "my_op"
        assert cc.kwargs == {"scale": 1.0}

    def test_expanded_missing_name_raises(self):
        """dict 不含 name → ValueError。"""
        with pytest.raises(ValueError, match="name"):
            _parse_component({"kwargs": {}})

    def test_invalid_type_raises(self):
        """非 str/dict → ValueError。"""
        with pytest.raises(ValueError, match="str"):
            _parse_component(123)


# ── YAML 加载与校验 ──────────────────────────


def _to_yaml_path(path):
    """将路径转为 YAML 安全格式（正斜杠，避免双引号中转义问题）。"""
    return str(path).replace("\\", "/")


def _make_fake_model_dir(tmp_path, name="fake_model", max_pos=40960):
    """在 tmp_path 下创建假的模型目录（含 config.json）。"""
    model_dir = tmp_path / name
    model_dir.mkdir(exist_ok=True)
    config = {
        "vocab_size": 151936,
        "hidden_size": 1024,
        "hidden_act": "silu",
        "num_hidden_layers": 28,
        "max_position_embeddings": max_pos,
        "num_key_value_heads": 8,
        "num_attention_heads": 16,
        "head_dim": 128,
        "intermediate_size": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 100000,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return _to_yaml_path(model_dir)


def _write_yaml(tmp_path, content):
    """写入临时 YAML 文件，返回路径。"""
    p = tmp_path / "batch.yaml"
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestLoadBatchConfig:
    """load_batch_config 加载与校验测试。"""

    def test_basic_load(self, tmp_path):
        """正常加载：全局 generation + 1 个模型。"""
        model_path = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
generation:
  temperature: 0.85
  top_k: 40
  max_new_tokens: 2048
models:
  - name: "test-model"
    path: "{model_path}"
    device: "cpu"
    max_len: 2048
""",
        )
        config = load_batch_config(yaml_path)
        assert config.generation.temperature == 0.85
        assert config.generation.max_new_tokens == 2048
        assert len(config.models) == 1
        assert config.models[0].name == "test-model"

    def test_get_model_returns_resolved(self, tmp_path):
        """get_model 返回 ResolvedModelEntry，generation 已合并。"""
        model_path = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
generation:
  temperature: 0.85
  top_k: 40
  max_new_tokens: 2048
models:
  - name: "m1"
    path: "{model_path}"
    max_len: 2048
    generation:
      temperature: 0.5
""",
        )
        config = load_batch_config(yaml_path)
        resolved = config.get_model("m1")
        assert isinstance(resolved, ResolvedModelEntry)
        # 覆盖生效
        assert resolved.generation.temperature == 0.5
        # 未覆盖的继承全局
        assert resolved.generation.top_k == 40
        assert resolved.generation.max_new_tokens == 2048

    def test_get_model_not_found(self, tmp_path):
        """get_model 查找不存在的 name → ValueError 含可用列表。"""
        model_path = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{model_path}"
    max_len: 2048
""",
        )
        config = load_batch_config(yaml_path)
        with pytest.raises(ValueError, match="m1"):
            config.get_model("nonexistent")

    def test_list_model_names(self, tmp_path):
        """list_model_names 返回所有 name。"""
        p1 = _make_fake_model_dir(tmp_path, "m1")
        p2 = _make_fake_model_dir(tmp_path, "m2")
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p1}"
    max_len: 2048
  - name: "m2"
    path: "{p2}"
    max_len: 1024
""",
        )
        config = load_batch_config(yaml_path)
        assert config.list_model_names() == ["m1", "m2"]

    def test_empty_file_raises(self, tmp_path):
        """空 YAML → ValueError。"""
        yaml_path = _write_yaml(tmp_path, "")
        with pytest.raises(ValueError, match="空"):
            load_batch_config(yaml_path)

    def test_empty_models_raises(self, tmp_path):
        """models 为空列表 → ValueError。"""
        yaml_path = _write_yaml(
            tmp_path,
            """
generation:
  temperature: 0.85
models: []
""",
        )
        with pytest.raises(ValueError, match="空"):
            load_batch_config(yaml_path)

    def test_duplicate_names_raises(self, tmp_path):
        """name 重复 → ValueError。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "dup"
    path: "{p}"
    max_len: 2048
  - name: "dup"
    path: "{p}"
    max_len: 1024
""",
        )
        with pytest.raises(ValueError, match="不唯一"):
            load_batch_config(yaml_path)

    def test_missing_path_raises(self, tmp_path):
        """path 不存在 → ValueError。"""
        yaml_path = _write_yaml(
            tmp_path,
            """
models:
  - name: "m1"
    path: "/nonexistent/path"
    max_len: 2048
""",
        )
        with pytest.raises(ValueError, match="path"):
            load_batch_config(yaml_path)

    def test_missing_config_json_raises(self, tmp_path):
        """path 存在但无 config.json → ValueError。"""
        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir(exist_ok=True)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{_to_yaml_path(empty_dir)}"
    max_len: 2048
""",
        )
        with pytest.raises(ValueError, match="config.json"):
            load_batch_config(yaml_path)

    def test_max_len_zero_raises(self, tmp_path):
        """max_len ≤ 0 → ValueError。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p}"
    max_len: 0
""",
        )
        with pytest.raises(ValueError, match="max_len"):
            load_batch_config(yaml_path)

    def test_max_len_exceeds_position_truncated(self, tmp_path):
        """max_len > max_position_embeddings → 警告 + 截断。"""
        p = _make_fake_model_dir(tmp_path, max_pos=1024)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p}"
    max_len: 4096
""",
        )
        config = load_batch_config(yaml_path)
        assert config.models[0].max_len == 1024

    def test_invalid_device_raises(self, tmp_path):
        """非法 device → ValueError。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p}"
    max_len: 2048
    device: "not_a_device"
""",
        )
        with pytest.raises(ValueError, match="device"):
            load_batch_config(yaml_path)

    def test_components_shorthand_parsed(self, tmp_path):
        """组件简写格式正确解析为 ComponentConfig。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p}"
    max_len: 2048
    components:
      mlp: "moe"
""",
        )
        config = load_batch_config(yaml_path)
        assert config.models[0].components["mlp"].name == "moe"
        assert config.models[0].components["mlp"].kwargs == {}

    def test_components_expanded_parsed(self, tmp_path):
        """组件展开格式正确解析。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p}"
    max_len: 2048
    components:
      attn:
        name: "my_op"
        kwargs:
          scale: 1.0
""",
        )
        config = load_batch_config(yaml_path)
        cc = config.models[0].components["attn"]
        assert cc.name == "my_op"
        assert cc.kwargs == {"scale": 1.0}

    def test_components_invalid_field_raises(self, tmp_path):
        """组件字段名不在 ComponentFactory._registry → ValueError。

        注意：此测试需要 ComponentFactory 已初始化。
        若 registry 为空（无 @register 调用），此测试可能不触发。
        当前项目注册在模块导入时完成，应可用。
        """
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p}"
    max_len: 2048
    components:
      nonexistent_field: "base"
""",
        )
        with pytest.raises(ValueError, match="nonexistent_field"):
            load_batch_config(yaml_path)

    def test_missing_required_field_raises(self, tmp_path):
        """缺必填字段（name/path/max_len）→ ValueError。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - path: "{p}"
    max_len: 2048
""",
        )
        with pytest.raises(ValueError, match="name"):
            load_batch_config(yaml_path)

    def test_no_generation_block_uses_defaults(self, tmp_path):
        """无 generation 块 → 用 GenerationDefaults 内置默认值。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
models:
  - name: "m1"
    path: "{p}"
    max_len: 2048
""",
        )
        config = load_batch_config(yaml_path)
        assert config.generation.temperature == 1.0
        assert config.generation.do_sample is False
        assert config.generation.max_new_tokens == 100

    def test_model_generation_none_inherits_global(self, tmp_path):
        """模型级无 generation → ResolvedModelEntry.generation == 全局。"""
        p = _make_fake_model_dir(tmp_path)
        yaml_path = _write_yaml(
            tmp_path,
            f"""
generation:
  temperature: 0.5
  max_new_tokens: 999
models:
  - name: "m1"
    path: "{p}"
    max_len: 2048
""",
        )
        config = load_batch_config(yaml_path)
        resolved = config.get_model("m1")
        assert resolved.generation.temperature == 0.5
        assert resolved.generation.max_new_tokens == 999


# ── 集成测试（依赖真实模型）──────────────────


class TestFromConfigIntegration:
    """from_config → Runner 完整流程集成测试。

    依赖 MODEL_PATH 环境变量指向真实模型目录。
    无模型时跳过。
    """

    @pytest.fixture
    def real_model_path(self):
        from qwen3_from_scratch.utils.env import load_env_file

        load_env_file()
        path = os.environ.get("MODEL_PATH")
        if not path or not os.path.isdir(path):
            pytest.skip("MODEL_PATH 未设置或不存在，跳过集成测试")
        return path.replace("\\", "/")
