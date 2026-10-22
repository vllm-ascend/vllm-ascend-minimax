# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Opt-in registration for the MiniMax M3.1 Flash DSpark checkpoint."""


def register():
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    # The prepared draft is a standalone text model. vLLM registers the VL/MTP
    # wrappers itself, but delegates this model_type to Transformers AutoConfig.
    # Preserve any native config registration, including the A5 implementation.
    if "minimax_m3_text" not in CONFIG_MAPPING:
        AutoConfig.register("minimax_m3_text", MiniMaxM3TextConfig)

    # Only the separately prepared A3 view uses this alias. Leave the original
    # DSparkMiniMaxDraftModel architecture to its native implementation.
    ModelRegistry.register_model(
        "MiniMaxM31FlashA3DSparkModel",
        "minimax_dspark_adapter.model:MiniMaxFlashDSparkForCausalLM",
    )
