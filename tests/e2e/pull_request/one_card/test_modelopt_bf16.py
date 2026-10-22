# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.quantization.methods.modelopt_bf16 import (
    Nvfp4BF16MoEMethod,
    dequantize_mxfp8,
    dequantize_nvfp4,
)


@pytest.mark.parametrize("rows,k", [(2, 32), (128, 128), (128, 6144)])
def test_mxfp8_bf16_dequantization_on_npu(rows, k):
    weight = (torch.randn(rows, k) * 0.1).to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.randint(124, 130, (rows, k // 32), dtype=torch.uint8)
    expected = dequantize_mxfp8(weight, scale)
    actual = dequantize_mxfp8(weight.npu(), scale.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows,k", [(2, 32), (128, 128), (128, 6144)])
def test_nvfp4_bf16_dequantization_on_npu(rows, k):
    weight = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8)
    scale = torch.randint(1, 100, (rows, k // 16), dtype=torch.uint8)
    global_scale = torch.tensor(128.0)
    expected = dequantize_nvfp4(weight, scale, global_scale)
    actual = dequantize_nvfp4(weight.npu(), scale.npu(), global_scale.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_bf16_grouped_mlp_matches_cpu():
    layer = torch.nn.Module()
    for prefix, rows, k, shards in (("w13", 32, 32, 2), ("w2", 32, 16, 1)):
        setattr(layer, f"{prefix}_weight", torch.randint(0, 256, (2, rows, k // 2), dtype=torch.uint8, device="npu"))
        setattr(layer, f"{prefix}_weight_scale", torch.full((2, rows, k // 16), 56, dtype=torch.uint8, device="npu"))
        setattr(layer, f"{prefix}_weight_global_scale", torch.full((2, shards), 32.0, device="npu"))
    method = Nvfp4BF16MoEMethod.__new__(Nvfp4BF16MoEMethod)
    x = torch.randn(6, 32, dtype=torch.bfloat16, device="npu")
    inputs = SimpleNamespace(
        layer=layer,
        hidden_states=x,
        group_list_type=0,
        group_list=torch.tensor([3, 6], dtype=torch.int64, device="npu"),
    )
    gate, up = method.apply_gmm1(inputs).chunk(2, dim=-1)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    actual = method.apply_gmm2(inputs, hidden, None).cpu()
    w13 = method._weight(layer, "w13").cpu()
    w2 = method._weight(layer, "w2").cpu()
    x_cpu = x.cpu()
    expected = []
    for expert in range(2):
        gate_up = (x_cpu[expert * 3 : (expert + 1) * 3].float() @ w13[expert].float()).to(torch.bfloat16)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        expected.append((hidden.float() @ w2[expert].float()).to(torch.bfloat16))
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0.02, atol=0.001)
