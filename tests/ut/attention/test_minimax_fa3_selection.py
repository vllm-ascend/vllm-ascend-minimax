# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.selector import AttentionSelectorConfig

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import NPUPlatform

with patch.dict(sys.modules, {"flash_attn_npu_3": MagicMock()}):
    from vllm_ascend.attention import flash_attention_v3 as fa3


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    "architecture", ["MiniMaxM3SparseForConditionalGeneration", "DSparkMiniMaxDraftModel", "Qwen3ForCausalLM"]
)
def test_minimax_fa3_hardware_selection(device_type, installed, architecture):
    target = SimpleNamespace(hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForConditionalGeneration"]))
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        speculative_config=SimpleNamespace(method="dspark", target_model_config=target),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        quant_config=None,
    )
    selector = AttentionSelectorConfig(head_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto", block_size=128)
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch("vllm_ascend.platform._validate_fa3_backend", return_value=False),
        patch("vllm_ascend.platform.get_ascend_device_type", return_value=device_type),
        patch("vllm_ascend.platform.get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch("vllm_ascend.platform.util.find_spec", return_value=object() if installed else None),
    ):
        backend = NPUPlatform.get_attn_backend_cls(None, selector)
    if device_type == AscendDeviceType._310P:
        expected = "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310"
    elif device_type in (AscendDeviceType.A2, AscendDeviceType.A3) and installed and architecture != "Qwen3ForCausalLM":
        expected = "vllm_ascend.attention.flash_attention_v3.AscendFlashAttentionBackend"
    else:
        expected = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
    assert backend == expected


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_minimax_fa3_scheduler_api(causal, padded):
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.capture_sizes = set()
    builder.capture_query_len = None
    builder.block_size = 128
    builder.max_num_reqs = 3
    scale = 0.123
    spec = (16, 1, 128, torch.bfloat16, scale, 0.0, 4096)
    builder.scheduler_specs = {spec}
    builder.scheduler_buffers = {}
    offsets = torch.tensor([0, 8, 8] if padded else [0, 8, 11], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=8 if padded else 11,
        num_input_tokens=11,
        max_query_len=8,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor([137, 73], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.arange(11),
        causal=causal,
    )

    # Match the keyword-only subset of the v0.4.2.post2 public scheduler API.
    def scheduler(
        *,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype,
        cu_seqlens_q,
        page_size,
        causal,
        window_size,
        softcap,
        softmax_scale,
        num_splits,
    ):
        assert softcap == 0.0
        assert softmax_scale == scale
        assert window_size == (4095, 0 if causal else 4095)
        assert cache_seqlens.numel() == batch_size
        assert cu_seqlens_q.numel() == batch_size + 1
        return torch.tensor([1], dtype=torch.uint8)

    with patch.object(fa3, "get_scheduler_metadata", side_effect=scheduler):
        metadata = builder.build(0, common)
    assert metadata.scheduler_metadata[spec].tolist() == [1]
    assert metadata.query_start_loc.numel() == (2 if padded else 3)
    assert metadata.block_tables.shape[0] == (1 if padded else 2)


def test_minimax_fa3_graph_metadata_keeps_storage_across_batch_sizes():
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.capture_sizes = {16, 24}
    builder.capture_query_len = 8
    builder.max_num_reqs = 3
    builder.block_size = 128
    spec = (16, 1, 128, torch.bfloat16, 0.123, 0.0, 4096)
    builder.scheduler_specs = {spec}
    builder.scheduler_buffers = {}
    offsets = torch.tensor([0, 8, 16, 16], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=16,
        num_input_tokens=24,
        max_query_len=8,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor([137, 73, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
        slot_mapping=torch.arange(24),
        causal=True,
    )

    def scheduler(**kwargs):
        # The 950 API appends an int32 cumulative-KV entry per request.
        return torch.full((3584 + 4 * (kwargs["batch_size"] + 1),), kwargs["batch_size"], dtype=torch.uint8)

    addresses = []
    with patch.object(fa3, "get_scheduler_metadata", side_effect=scheduler):
        for batch_size in (2, 1, 3, 2):
            offsets.copy_(torch.tensor([8 * min(i, batch_size) for i in range(4)], dtype=torch.int32))
            common.num_actual_tokens = batch_size * 8
            metadata = builder.build(0, common)
            tiling = metadata.scheduler_metadata[spec]
            addresses.append(tiling.data_ptr())
            assert tiling.numel() == 3584 + 4 * (batch_size + 1)
            assert torch.all(tiling == batch_size)
        common.num_input_tokens = 16
        other_bucket = builder.build(0, common)
    assert len(set(addresses)) == 1
    assert len(builder.scheduler_buffers) == 2
    assert other_bucket.scheduler_metadata[spec].data_ptr() != addresses[0]
