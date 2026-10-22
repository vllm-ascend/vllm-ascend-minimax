# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.quantization.methods.modelopt_bf16 import (
    Mxfp8BF16LinearMethod,
    Nvfp4BF16MoEMethod,
    dequantize_mxfp8,
    dequantize_nvfp4,
)


def test_nvfp4_low_nibble_order_and_global_divisor():
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
    scale = torch.tensor([[0.5]], dtype=torch.float8_e4m3fn)
    actual = dequantize_nvfp4(packed, scale, torch.tensor(2.0))
    expected = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]], dtype=torch.bfloat16) / 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mxfp8_e8m0_groups_are_independent():
    weight = torch.tensor([[1.0] * 32 + [-2.0] * 32], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[128, 126]], dtype=torch.uint8)
    actual = dequantize_mxfp8(weight.view(torch.uint8), scale)
    expected = torch.tensor([[2.0] * 32 + [-1.0] * 32], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_float_weights_cannot_be_misread_as_serialized_bytes():
    with pytest.raises(ValueError, match="serialized FP8"):
        dequantize_mxfp8(torch.zeros(2, 32), torch.zeros(2, 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="packed uint8"):
        dequantize_nvfp4(torch.zeros(2, 16), torch.zeros(2, 2, dtype=torch.uint8), torch.tensor(1.0))


def test_reserved_mxfp8_scale_propagates_nan():
    weight = torch.ones(1, 32).to(torch.float8_e4m3fn)
    assert dequantize_mxfp8(weight, torch.tensor([[255]], dtype=torch.uint8)).isnan().all()


@pytest.mark.parametrize("dequantizer,packed_width,scale_width", [(dequantize_mxfp8, 32, 2), (dequantize_nvfp4, 16, 1)])
def test_reject_incompatible_scale_shape(dequantizer, packed_width, scale_width):
    args = [torch.zeros(2, packed_width, dtype=torch.uint8), torch.zeros(2, scale_width, dtype=torch.uint8)]
    if dequantizer is dequantize_nvfp4:
        args.append(torch.tensor(1.0))
    with pytest.raises(ValueError):
        dequantizer(*args)


def make_moe(tp_rank=0, local_expert=True):
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(tp_size=2, tp_rank=tp_rank)
    layer._map_global_expert_id_to_local_expert_id = lambda _: 0 if local_expert else -1
    method = Nvfp4BF16MoEMethod.__new__(Nvfp4BF16MoEMethod)
    config = SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=False))
    with (
        patch("vllm_ascend.quantization.methods.modelopt_bf16.get_current_vllm_config", return_value=config),
        patch(
            "vllm_ascend.quantization.methods.modelopt_bf16.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=False),
        ),
    ):
        method.create_weights(layer, 1, 32, 16, torch.bfloat16)
    return method, layer


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("projection", ["w1", "w2", "w3"])
def test_nvfp4_tp_loading_shards_packed_weights(rank, projection):
    _, layer = make_moe(tp_rank=rank)
    is_down = projection == "w2"
    param = layer.w2_weight if is_down else layer.w13_weight
    source = torch.arange(32 * 16, dtype=torch.int32).to(torch.uint8).reshape(32, 16)
    param.weight_loader(param, source, "experts.weight", projection, 0)
    actual = param[0] if is_down else param[0].chunk(2, dim=0)[projection == "w3"]
    expected = source[:, rank * 8 : (rank + 1) * 8] if is_down else source[rank * 16 : (rank + 1) * 16]
    torch.testing.assert_close(actual, expected)


def test_nvfp4_gate_and_up_global_scales_remain_distinct():
    method, layer = make_moe()
    param = layer.w13_weight_global_scale
    param.weight_loader(param, torch.tensor([2.0]), "global_scale", "w1", 0)
    param.weight_loader(param, torch.tensor(4.0), "global_scale", "w3", 0)
    assert param.tolist() == [[2.0, 4.0]]
    layer.w13_weight.data.fill_(0x22)  # FP4 1.0
    layer.w13_weight_scale.data.fill_(56)  # E4M3 1.0
    actual = method._weight(layer, "w13")
    assert actual.shape == (1, 32, 32)
    torch.testing.assert_close(actual[..., :16], torch.full((1, 32, 16), 0.5, dtype=torch.bfloat16))
    torch.testing.assert_close(actual[..., 16:], torch.full((1, 32, 16), 0.25, dtype=torch.bfloat16))


def test_nvfp4_nonlocal_expert_is_not_loaded():
    _, layer = make_moe(local_expert=False)
    param = layer.w2_weight
    assert param.weight_loader(param, torch.zeros(32, 16, dtype=torch.uint8), "weight", "w2", 8, True) is False


def test_missing_nvfp4_expert_cannot_silently_keep_initialized_weights():
    method, layer = make_moe()
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="safetensors"))
    with (
        patch("vllm_ascend.quantization.methods.modelopt_bf16.get_current_vllm_config", return_value=config),
        pytest.raises(ValueError, match="Incomplete NVFP4 checkpoint"),
    ):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_nvfp4_reject_invalid_global_scale(value):
    _, layer = make_moe()
    param = layer.w2_weight_global_scale
    with pytest.raises(ValueError, match="finite positive"):
        param.weight_loader(param, torch.tensor(value), "global_scale", "w2", 0)


def test_mxfp8_linear_load_preserves_fp8_bits_and_bf16_compute():
    method = Mxfp8BF16LinearMethod()
    layer = torch.nn.Module()

    def loader(param, weight):
        param.data.copy_(weight)

    method.create_weights(layer, 32, [2], 32, 2, torch.bfloat16, weight_loader=loader)
    weight = torch.full((2, 32), 0.5, dtype=torch.float8_e4m3fn)
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale.weight_loader(layer.weight_scale, torch.full((2, 1), 128, dtype=torch.uint8))
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="safetensors"))
    with patch("vllm_ascend.quantization.methods.modelopt_bf16.get_current_vllm_config", return_value=config):
        method.process_weights_after_loading(layer)
    x = torch.ones(3, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(method.apply(layer, x), torch.full((3, 2), 32.0, dtype=torch.bfloat16))
