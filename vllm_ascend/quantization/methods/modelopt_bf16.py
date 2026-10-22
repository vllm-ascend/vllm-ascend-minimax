# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in BF16 computation for MXFP8 linears and packed CT NVFP4 experts.

This compatibility path does not require native FP8/FP4 matrix instructions.
Linear weights are expanded once. Expert weights stay packed between calls;
one local projection is expanded for each grouped matmul. This costs time and
temporary HBM and is intended for correctness and initial service bring-up.
"""

from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod


def fp8_e4m3_to_float(raw: torch.Tensor) -> torch.Tensor:
    """Decode bytes without invoking an FP8 device cast on A3."""
    if raw.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("E4M3 data must contain serialized FP8 values or their raw bytes")
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().to(raw.device)
    return table[raw.view(torch.uint8).long()]


def dequantize_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.uint8:
        raise ValueError("MXFP8 E8M0 scales must be stored as uint8 bytes")
    if weight.shape[-1] % 32 or scale.shape != (*weight.shape[:-1], weight.shape[-1] // 32):
        raise ValueError("MXFP8 requires one E8M0 scale per 32 weights")
    values = fp8_e4m3_to_float(weight).unflatten(-1, (-1, 32))
    exponents = scale.view(torch.uint8).float() - 127
    factors = torch.where(scale == 255, float("nan"), torch.pow(2.0, exponents)).unsqueeze(-1)
    return (values * factors).flatten(-2).to(torch.bfloat16)


def dequantize_nvfp4(weight: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """CT NVFP4 stores low nibble first and the global scale as a divisor."""
    if weight.dtype != torch.uint8:
        raise ValueError("NVFP4 weights must be packed uint8 bytes")
    if weight.shape[-1] % 8 or scale.shape != (*weight.shape[:-1], weight.shape[-1] // 8):
        raise ValueError("NVFP4 requires one E4M3 scale per 16 unpacked weights")
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=weight.device,
    )
    values = table[codes.long()].unflatten(-1, (-1, 16))
    factors = fp8_e4m3_to_float(scale).unsqueeze(-1)
    return ((values * factors).flatten(-2) / global_scale).to(torch.bfloat16)


def _byte_loader(loader):
    @wraps(loader)
    def load(param, loaded_weight, *args, **kwargs):
        if loaded_weight.dtype == torch.float8_e4m3fn:
            loaded_weight = loaded_weight.view(torch.uint8)
        elif loaded_weight.dtype != torch.uint8:
            raise ValueError(f"Expected serialized FP8/byte tensor, got {loaded_weight.dtype}")
        return loader(param, loaded_weight, *args, **kwargs)

    return load


class Mxfp8BF16LinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % 32:
            raise ValueError("BF16 MXFP8 compatibility requires TP reduction shards aligned to 32 weights")
        output_size_per_partition = sum(output_partition_sizes)
        loader = _byte_loader(extra_weight_attrs.pop("weight_loader"))
        for name, shape in (
            ("weight", (output_size_per_partition, input_size_per_partition)),
            ("weight_scale", (output_size_per_partition, input_size_per_partition // 32)),
        ):
            param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.uint8), requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, {**extra_weight_attrs, "input_dim": 1, "output_dim": 0, "weight_loader": loader})

    def process_weights_after_loading(self, layer):
        if get_current_vllm_config().load_config.load_format == "dummy":
            random_weight = torch.randn(layer.weight.shape, device="cpu") * 0.02
            layer.weight.data.copy_(random_weight.to(torch.float8_e4m3fn).view(torch.uint8).to(layer.weight.device))
            layer.weight_scale.data.fill_(127)
        layer.weight.data = dequantize_mxfp8(layer.weight, layer.weight_scale)

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class Nvfp4BF16MoEMethod(AscendUnquantizedFusedMoEMethod):
    @property
    def supports_eplb(self):
        return False

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs
    ):
        config = get_current_vllm_config()
        if config.parallel_config.enable_eplb or get_ascend_config().enable_fused_mc2:
            raise ValueError("NVFP4 BF16 compatibility does not support EPLB or fused MC2")
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError("NVFP4 BF16 compatibility requires hidden and TP intermediate sizes aligned to 16")
        self.intermediate_size = intermediate_size_per_partition
        for prefix, rows, cols, shards in (
            ("w13", 2 * intermediate_size_per_partition, hidden_size, 2),
            ("w2", hidden_size, intermediate_size_per_partition, 1),
        ):
            shapes = {
                "weight": ((num_experts, rows, cols // 2), torch.uint8),
                "weight_scale": ((num_experts, rows, cols // 16), torch.uint8),
                "weight_global_scale": ((num_experts, shards), torch.float32),
                "input_global_scale": ((num_experts, shards), torch.float32),
            }
            for suffix, (shape, dtype) in shapes.items():
                param = torch.nn.Parameter(torch.ones(shape, dtype=dtype), requires_grad=False)
                param.is_nvfp4_packed = suffix == "weight"
                param.nvfp4_loaded_shards = set()
                param.weight_loader = self._loader(layer, prefix, suffix)
                layer.register_parameter(f"{prefix}_{suffix}", param)

    def _loader(self, layer, prefix, suffix):
        def load(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            if shard_id not in (("w1", "w3") if prefix == "w13" else ("w2",)):
                raise ValueError(f"Invalid NVFP4 projection shard {shard_id} for {prefix}")
            local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            target = param.data[local_id]
            if suffix.endswith("global_scale"):
                if loaded_weight.numel() != 1:
                    raise ValueError(f"{weight_name} requires one global scale per projection")
                if not torch.isfinite(loaded_weight).all() or not (loaded_weight > 0).all():
                    raise ValueError(f"{weight_name} requires a finite positive global scale")
                target[0 if shard_id != "w3" else 1].copy_(loaded_weight.reshape(()))
            else:
                if suffix == "weight_scale" and loaded_weight.dtype == torch.float8_e4m3fn:
                    loaded_weight = loaded_weight.view(torch.uint8)
                elif loaded_weight.dtype != torch.uint8:
                    raise ValueError(f"{weight_name} requires serialized byte/FP8 data")
                if prefix == "w13":
                    target = target.chunk(2, dim=0)[0 if shard_id == "w1" else 1]
                dim = 1 if prefix == "w2" else 0
                if loaded_weight.shape != target.shape:
                    if loaded_weight.shape[dim] != target.shape[dim] * layer.moe_config.tp_size:
                        raise ValueError(f"Incompatible NVFP4 TP shard for {weight_name}")
                    loaded_weight = loaded_weight.narrow(
                        dim, layer.moe_config.tp_rank * target.shape[dim], target.shape[dim]
                    )
                if loaded_weight.shape != target.shape:
                    raise ValueError(f"Incompatible NVFP4 tensor shape for {weight_name}")
                target.copy_(loaded_weight)
            param.nvfp4_loaded_shards.add((local_id, shard_id))
            return True if return_success else None

        load.supports_moe_loading = True
        return load

    def process_weights_after_loading(self, layer):
        if get_current_vllm_config().load_config.load_format == "dummy":
            for prefix in ("w13", "w2"):
                getattr(layer, f"{prefix}_weight").data.random_(0, 256)
                getattr(layer, f"{prefix}_weight_scale").data.fill_(40)  # E4M3 0.25
                getattr(layer, f"{prefix}_weight_global_scale").data.fill_(128)
                getattr(layer, f"{prefix}_input_global_scale").data.fill_(1)
        else:
            for prefix, shards in (("w13", ("w1", "w3")), ("w2", ("w2",))):
                for suffix in ("weight", "weight_scale", "weight_global_scale"):
                    param = getattr(layer, f"{prefix}_{suffix}")
                    expected = {(expert, shard) for expert in range(param.shape[0]) for shard in shards}
                    missing = expected - param.nvfp4_loaded_shards
                    if missing:
                        raise ValueError(
                            f"Incomplete NVFP4 checkpoint for {prefix}_{suffix}: missing {len(missing)} expert shards"
                        )

    @staticmethod
    def _weight(layer, prefix):
        global_scale = getattr(layer, f"{prefix}_weight_global_scale")
        if prefix == "w13":
            global_scale = global_scale.repeat_interleave(layer.w13_weight.shape[1] // 2, dim=1)
        return (
            dequantize_nvfp4(
                getattr(layer, f"{prefix}_weight"),
                getattr(layer, f"{prefix}_weight_scale"),
                global_scale.unsqueeze(-1),
            )
            .transpose(1, 2)
            .contiguous()
        )

    def apply_gmm1(self, mlp_compute_input):
        return torch_npu.npu_grouped_matmul(
            x=[mlp_compute_input.hidden_states],
            weight=[self._weight(mlp_compute_input.layer, "w13")],
            split_item=2,
            group_type=0,
            group_list_type=mlp_compute_input.group_list_type,
            group_list=mlp_compute_input.group_list,
        )[0]

    def apply_gmm2(self, mlp_compute_input, hidden_states, act_out_scale):
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[self._weight(mlp_compute_input.layer, "w2")],
            split_item=2,
            group_type=0,
            group_list_type=mlp_compute_input.group_list_type,
            group_list=mlp_compute_input.group_list,
        )[0]
