# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_ascend.ascend_config import validate_additional_config_bool
from vllm_ascend.quantization.utils import is_fused_moe_layer


@register_quantization_config("mxfp8")
@register_quantization_config("modelopt_mxfp8")
class AscendModelOptMxFp8Config(ModelOptMxFp8Config):
    """Load ModelOpt-compatible MXFP8 checkpoints with Ascend methods."""

    @classmethod
    def from_config(cls, config):
        result = super().from_config(config)
        metadata = {**config.get("quantization", {}), **config}
        result.moe_quant_algo = str(metadata.get("moe_quant_algo", "MXFP8")).upper()
        result.moe_quant_format = metadata.get("moe_quant_format")
        result.moe_group_size = metadata.get("group_size", 16)
        result.weight_block_size = metadata.get("weight_block_size", [1, 32])
        if result.moe_quant_algo not in ("MXFP8", "NVFP4"):
            raise ValueError(f"Unsupported MXFP8 checkpoint MoE algorithm: {result.moe_quant_algo}")
        if result.moe_quant_algo == "NVFP4" and (
            result.moe_quant_format != "nvfp4-pack-quantized" or result.moe_group_size != 16
        ):
            raise ValueError("NVFP4 experts require nvfp4-pack-quantized with group_size=16")
        return result

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware does not support "get_min_capability" feature.')

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        from ..method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )
        from ..methods.w8a8.w8a8_mxfp8 import (
            AscendW8A8MXFP8DynamicFusedMoEMethod,
            AscendW8A8MXFP8DynamicLinearMethod,
        )

        if self.is_layer_excluded(prefix):
            if isinstance(layer, VocabParallelEmbedding):
                return UnquantizedEmbeddingMethod()
            if isinstance(layer, LinearBase):
                return AscendUnquantizedLinearMethod()
            if is_fused_moe_layer(layer):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid)
            return None

        # Match the compatibility behavior retained by ModelOpt for older
        # checkpoints whose vision modules were not listed in exclude_modules.
        if any(module_name in prefix for module_name in ("vision_tower", "vision_model", "vit_large_projector")):
            if isinstance(layer, VocabParallelEmbedding):
                return UnquantizedEmbeddingMethod()
            if isinstance(layer, LinearBase):
                return AscendUnquantizedLinearMethod()
            return None

        vllm_config = get_current_vllm_config_or_none()
        bf16_compat = vllm_config is not None and validate_additional_config_bool(
            vllm_config.additional_config.get("modelopt_bf16_compat", False), "additional_config.modelopt_bf16_compat"
        )
        if bf16_compat:
            from ..methods.modelopt_bf16 import Mxfp8BF16LinearMethod, Nvfp4BF16MoEMethod

            if vllm_config.model_config.dtype != torch.bfloat16 or not vllm_config.model_config.enforce_eager:
                raise ValueError("modelopt_bf16_compat requires dtype=bfloat16 and enforce_eager=True")
            if vllm_config.lora_config is not None:
                raise ValueError("modelopt_bf16_compat does not support LoRA")
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                if self.weight_block_size != [1, 32]:
                    raise ValueError("BF16 MXFP8 compatibility requires weight_block_size=[1, 32]")
                return Mxfp8BF16LinearMethod()
            if is_fused_moe_layer(layer):
                if self.moe_quant_algo != "NVFP4":
                    raise ValueError("BF16 MoE compatibility currently requires NVFP4 experts")
                return Nvfp4BF16MoEMethod(layer.moe_config, tid2eid)
        elif self.moe_quant_algo == "NVFP4":
            raise ValueError(
                "NVFP4 experts cannot use the Ascend MXFP8 MoE method. "
                "For eager BF16 compatibility, set additional_config.modelopt_bf16_compat=true."
            )

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return AscendLinearMethod(AscendW8A8MXFP8DynamicLinearMethod())

        if is_fused_moe_layer(layer):
            return AscendFusedMoEMethod(
                AscendW8A8MXFP8DynamicFusedMoEMethod(),
                layer.moe_config,
                tid2eid,
            )

        # Embeddings and the BF16 KV cache keep their native methods.
        return None
