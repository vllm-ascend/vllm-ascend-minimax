# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""CPU checks for the integrated BF16 DSpark path, without importing vLLM."""

import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from pydantic.dataclasses import dataclass


@pytest.fixture
def native_profile():
    source = {
        "text_config": {
            "hidden_size": 6144,
            "num_hidden_layers": 60,
            "vocab_size": 200064,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "dense_intermediate_size": 12288,
            "rope_theta": 10000000,
            "rotary_dim": 64,
            "quantization_config": {"quant_method": "ascend"},
        }
    }
    fixture = Path(__file__).parent / "fixtures/minimax_m31_bf16_dspark_header.json"
    return source, json.loads(fixture.read_text())


def test_native_config_preserves_the_target_and_uses_bf16(adapter_modules, native_profile):
    source, header = native_profile
    original = json.dumps(source, sort_keys=True)
    config = adapter_modules.weights.native_draft_config(source, header)
    assert config["architectures"] == ["DSparkMiniMaxDraftModel"]
    assert config["num_hidden_layers"] == 5
    assert config["target_layer_ids"] == [3, 17, 31, 45, 59]
    assert config["dspark_ffn_hidden_size"] == 12288
    assert config["markov_rank"] == 256
    assert config["n_predict"] == 7
    assert config["sliding_window"] == 4096
    assert config["mask_token_id"] == 200058
    assert config["num_local_experts"] == 0
    assert "quantization_config" not in config
    assert json.dumps(source, sort_keys=True) == original


def test_native_fallback_rejects_an_unrelated_target(adapter_modules, native_profile):
    source, header = native_profile
    source["text_config"]["num_hidden_layers"] = 61
    with pytest.raises(ValueError, match="only supports the MiniMax M3.1 Flash profile"):
        adapter_modules.weights.native_draft_config(source, header)


@pytest.fixture
def official_draft_config():
    # MiniMaxAI/MiniMax-M3.1-Flash, revision 54e4e69, dspark/config.json.
    path = Path(__file__).parent / "fixtures/minimax_m31_dspark_config.json"
    return json.loads(path.read_text())


def test_official_draft_config_matches_the_actual_bf16_schema(adapter_modules, native_profile, official_draft_config):
    source, header = native_profile
    original = json.dumps(official_draft_config, sort_keys=True)
    config = adapter_modules.weights.native_draft_config(source, header, official_draft_config)
    assert config["target_layer_ids"] == [3, 17, 31, 45, 59]
    assert config["num_hidden_layers"] == 5
    assert config["n_predict"] == 7
    assert config["intermediate_size"] == 12288
    assert config["num_local_experts"] == 0
    assert config["dtype"] == "bfloat16"
    assert "quantization_config" not in config
    assert json.dumps(official_draft_config, sort_keys=True) == original


def test_explicit_draft_training_parameters_override_the_fallback(
    adapter_modules, native_profile, official_draft_config
):
    source, header = native_profile
    draft = official_draft_config["text_config"]
    draft.update(
        dspark_target_layer_ids=[0, 10, 20, 30, 40],
        dspark_block_size=3,
        dspark_noise_token_id=123,
        sliding_window=512,
        dspark_hidden_norm_type="rms",
    )
    config = adapter_modules.weights.native_draft_config(source, header, official_draft_config)
    assert config["target_layer_ids"] == [0, 10, 20, 30, 40]
    assert config["n_predict"] == config["dflash_config"]["block_size"] == 3
    assert config["mask_token_id"] == config["dflash_config"]["mask_token_id"] == 123
    assert config["sliding_window"] == config["dflash_config"]["sliding_window"] == 512
    assert config["dspark_hidden_norm_type"] == "rms"


def test_incompatible_draft_config_is_rejected(adapter_modules, native_profile, official_draft_config):
    source, header = native_profile
    official_draft_config["text_config"]["dspark_markov_rank"] = 128
    with pytest.raises(ValueError, match="Tensor shape mismatch"):
        adapter_modules.weights.native_draft_config(source, header, official_draft_config)


def test_main_config_cannot_be_mistaken_for_a_draft_config(adapter_modules, native_profile):
    source, header = native_profile
    source["architectures"] = ["MiniMaxM3SparseForConditionalGeneration"]
    with pytest.raises(ValueError, match="not a target model config"):
        adapter_modules.weights.native_draft_config(source, header, source)


def test_hf_parser_sentinel_does_not_require_checkpoint_metadata(adapter_modules):
    sentinel = SimpleNamespace(model_type="dummy_minimax_m3_vl")
    assert (
        adapter_modules.weights.draft_hf_override(sentinel, model_root="/missing/checkpoint", target_override={})
        is sentinel
    )


def test_native_shard_needs_no_prepared_config_or_index(adapter_modules, native_profile, tmp_path):
    _, header = native_profile
    draft = tmp_path / "dspark"
    draft.mkdir()
    encoded = json.dumps(header).encode()
    shard = draft / "model-dspark-00001.safetensors"
    shard.write_bytes(len(encoded).to_bytes(8, "little") + encoded)
    assert adapter_modules.weights.read_draft_header(tmp_path) == header
    duplicate = draft / "model-dspark-00002.safetensors"
    duplicate.write_bytes(shard.read_bytes())
    with pytest.raises(ValueError, match="Duplicate DSpark tensor across shards"):
        adapter_modules.weights.read_draft_header(tmp_path)


@pytest.fixture
def adapter_modules(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    helper_path = root / "vllm_ascend/patch/minimax_m3_dspark.py"
    spec = importlib.util.spec_from_file_location("minimax_dspark_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.minimax_m3_dspark", helper)
    yield SimpleNamespace(weights=helper, guard=helper)
    sys.modules.pop("minimax_dspark_model_under_test", None)


@pytest.fixture
def checkpoint():
    config = {
        "hidden_size": 4,
        "head_dim": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 5,
        "vocab_size": 6,
        "dspark_ffn_hidden_size": 8,
        "dspark_markov_rank": 2,
        "dspark_target_layer_ids": [1, 3],
        "dspark_block_size": 4,
        "dspark_markov_head_type": "low_rank",
        "dspark_noise_token_id": 5,
        "dspark_hybrid_context_fusion": False,
        "rope_theta": 10000,
        "rotary_dim": 2,
    }
    shapes = {
        "fc.weight": (4, 8),
        "hidden_norm.weight": (4,),
        "final_norm.weight": (4,),
        "markov_head.markov_w1.weight": (6, 2),
        "markov_head.markov_w2.weight": (6, 2),
        "confidence_head.proj.weight": (1, 6),
        "confidence_head.proj.bias": (1,),
    }
    for layer in range(5):
        prefix = f"layers.{layer}.decoder_layer."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (4,),
                prefix + "post_attention_layernorm.weight": (4,),
                prefix + "self_attn.q_norm.weight": (2,),
                prefix + "self_attn.k_norm.weight": (2,),
                prefix + "self_attn.q_proj.weight": (4, 4),
                prefix + "self_attn.k_proj.weight": (2, 4),
                prefix + "self_attn.v_proj.weight": (2, 4),
                prefix + "self_attn.o_proj.weight": (4, 4),
                prefix + "mlp.gate_proj.weight": (8, 4),
                prefix + "mlp.up_proj.weight": (8, 4),
                prefix + "mlp.down_proj.weight": (4, 8),
            }
        )
    tensors = {
        "language_model.model.dspark." + name: torch.arange(math.prod(shape), dtype=torch.bfloat16).reshape(shape)
        for name, shape in shapes.items()
    }
    return config, tensors


def _header(tensors):
    return {name: {"shape": list(value.shape), "dtype": "BF16"} for name, value in tensors.items()}


def test_native_bf16_loading_preserves_tensor_identity_and_values(adapter_modules, checkpoint):
    config, tensors = checkpoint
    snapshots = {name: tensor.clone() for name, tensor in tensors.items()}
    loaded = dict(adapter_modules.weights.iter_bf16_weights(iter(tensors.items()), config))
    assert len(loaded) == 62
    assert loaded["norm.weight"] is tensors["language_model.model.dspark.final_norm.weight"]
    assert (
        loaded["layers.0.self_attn.q_proj.weight"]
        is tensors["language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight"]
    )
    for name, tensor in tensors.items():
        normalized = adapter_modules.weights.map_weight_name(name)
        assert loaded[normalized] is tensor
        assert loaded[normalized].data_ptr() == tensor.data_ptr()
        torch.testing.assert_close(tensor, snapshots[name], rtol=0, atol=0)


def test_model_loader_passes_bf16_weights_to_the_existing_dspark_loader(adapter_modules, checkpoint, monkeypatch):
    class DSparkLoader(torch.nn.Module):
        def load_weights(self, weights):
            self.received = dict(weights)
            return set(self.received)

    modules = {
        "vllm.model_executor.layers.layernorm": SimpleNamespace(GemmaRMSNorm=object, RMSNorm=object),
        "vllm.model_executor.models.qwen3_dflash": SimpleNamespace(
            DFlashQwen3Attention=object,
            DFlashQwen3DecoderLayer=torch.nn.Module,
            DFlashQwen3ForCausalLM=object,
            _resolve_layer_attention=Mock(),
        ),
        "vllm.model_executor.models.qwen3_dspark": SimpleNamespace(Qwen3DSparkModel=torch.nn.Module),
        "vllm_ascend.models.minimax_m3.minimax_m3": SimpleNamespace(MiniMaxM3MLP=object),
        "vllm_ascend.models.qwen3_dspark": SimpleNamespace(AscendQwen3DSparkForCausalLM=DSparkLoader),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    model_path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_minimax_m3_dspark.py"
    spec = importlib.util.spec_from_file_location("minimax_dspark_model_under_test", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.MiniMaxFlashDSparkForCausalLM.__new__(model_module.MiniMaxFlashDSparkForCausalLM)
    torch.nn.Module.__init__(model)
    config, tensors = checkpoint
    model.config = SimpleNamespace(to_dict=lambda: config)
    loaded = model.load_weights(iter(tensors.items()))
    assert len(loaded) == 62
    for name, tensor in tensors.items():
        assert model.received[adapter_modules.weights.map_weight_name(name)] is tensor


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float16, torch.float32])
def test_non_bf16_draft_is_rejected_without_conversion(adapter_modules, checkpoint, dtype):
    config, tensors = checkpoint
    name = "language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight"
    tensors[name] = tensors[name].to(dtype)
    with pytest.raises(ValueError, match="Expected native BF16 DSpark tensor"):
        list(adapter_modules.weights.iter_bf16_weights(tensors.items(), config))


def test_legacy_scale_tensors_are_rejected(adapter_modules, checkpoint):
    config, tensors = checkpoint
    name = "language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight_scale_inv"
    tensors[name] = torch.ones((4, 1), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Tensor schema mismatch"):
        list(adapter_modules.weights.iter_bf16_weights(tensors.items(), config))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "mapped_duplicate", "shape", "prefix"])
def test_bf16_schema_still_rejects_invalid_checkpoints(adapter_modules, checkpoint, fault):
    config, tensors = checkpoint
    entries = list(tensors.items())
    if fault == "missing":
        entries.pop()
    elif fault == "duplicate":
        entries.append(entries[0])
    elif fault == "mapped_duplicate":
        entries.append(
            ("language_model.model.dspark.norm.weight", tensors["language_model.model.dspark.final_norm.weight"])
        )
    elif fault == "shape":
        entries[0] = (entries[0][0], torch.zeros(1, dtype=torch.bfloat16))
    else:
        entries[0] = ("other_model.fc.weight", entries[0][1])
    with pytest.raises(ValueError):
        list(adapter_modules.weights.iter_bf16_weights(iter(entries), config))


def _core_post_init(self):
    if self.method == "dspark" and (
        "Qwen3DSparkModel" not in self.draft_model_config.architectures
        and "K3DSparkModel" not in self.draft_model_config.architectures
    ):
        self.draft_model_config.architectures = ["DSparkDraftModel"]
    self.validated = True


@pytest.fixture
def core_post_init():
    # The patch mutates the function object; each test needs its own code object.
    function = FunctionType(_core_post_init.__code__, _core_post_init.__globals__, _core_post_init.__name__)
    function.__module__ = "vllm.config.speculative"
    return function


@pytest.mark.parametrize(
    "architecture,method,expected",
    [
        ("DSparkMiniMaxDraftModel", "dspark", "DSparkMiniMaxDraftModel"),
        ("Qwen3DSparkModel", "dspark", "Qwen3DSparkModel"),
        ("K3DSparkModel", "dspark", "K3DSparkModel"),
        ("UnknownDraft", "dspark", "DSparkDraftModel"),
        ("UnknownDraft", "eagle3", "UnknownDraft"),
    ],
)
def test_automatic_guard_changes_only_the_minimax_architecture(
    adapter_modules, core_post_init, architecture, method, expected
):
    adapter_modules.guard._allow_minimax_draft(core_post_init)
    config = SimpleNamespace(method=method, draft_model_config=SimpleNamespace(architectures=[architecture]))
    core_post_init(config)
    assert config.draft_model_config.architectures == [expected]
    assert config.validated


def test_automatic_guard_is_idempotent(adapter_modules, core_post_init):
    adapter_modules.guard._allow_minimax_draft(core_post_init)
    first_code = core_post_init.__code__
    adapter_modules.guard._allow_minimax_draft(core_post_init)
    assert core_post_init.__code__ is first_code


def test_automatic_guard_reaches_an_already_compiled_pydantic_validator(adapter_modules, core_post_init):
    @dataclass
    class SpeculativeConfig:
        method: str
        draft_model_config: Any
        validated: bool = False

        __post_init__ = core_post_init

    before = SpeculativeConfig(
        method="dspark", draft_model_config=SimpleNamespace(architectures=["DSparkMiniMaxDraftModel"])
    )
    assert before.draft_model_config.architectures == ["DSparkDraftModel"]
    adapter_modules.guard._allow_minimax_draft(core_post_init)
    after = SpeculativeConfig(
        method="dspark", draft_model_config=SimpleNamespace(architectures=["DSparkMiniMaxDraftModel"])
    )
    assert after.draft_model_config.architectures == ["DSparkMiniMaxDraftModel"]
    assert after.validated


def test_changed_core_guard_fails_without_modifying_the_method(adapter_modules):
    def unknown_core(self):
        self.validated = True

    before = unknown_core.__code__
    with pytest.raises(RuntimeError, match="Core version changed"):
        adapter_modules.guard._allow_minimax_draft(unknown_core)
    assert unknown_core.__code__ is before


def test_native_architecture_is_registered_by_the_standard_plugin(adapter_modules, monkeypatch):
    registry = SimpleNamespace(register_model=Mock())
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/__init__.py"
    spec = importlib.util.spec_from_file_location("ascend_model_registration", path)
    registration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registration)
    registration.register_model()
    registry.register_model.assert_any_call(
        "DSparkMiniMaxDraftModel", "vllm_ascend.patch.worker.patch_minimax_m3_dspark:MiniMaxFlashDSparkForCausalLM"
    )
