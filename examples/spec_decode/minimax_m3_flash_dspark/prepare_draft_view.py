# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Prepare a separate draft config view. Source weights are never changed."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from minimax_dspark_adapter.weights import validate_header

MAX_SAFETENSORS_HEADER_BYTES = 10 * 1024 * 1024


def normalized_config(source, hidden_norm, final_norm):
    if source.get("architectures") != ["DSparkMiniMaxDraftModel"]:
        raise ValueError("Unexpected source draft architecture")
    config = copy.deepcopy(source["text_config"])
    for key in (
        "dspark_block_size",
        "dspark_markov_rank",
        "dspark_markov_head_type",
        "dspark_noise_token_id",
        "dspark_ffn_hidden_size",
        "dspark_target_layer_ids",
        "dspark_hybrid_context_fusion",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "dflash_config",
        "dspark_config",
    ):
        if key in source:
            config[key] = copy.deepcopy(source[key])
    if config["num_hidden_layers"] != 5 or config["dspark_hybrid_context_fusion"]:
        raise ValueError("Only the five-layer non-hybrid Flash draft is handled")
    config.update(
        {
            "model_type": "minimax_m3_text",
            "architectures": ["MiniMaxM31FlashA3DSparkModel"],
            "dspark_minimax_flash_adapter": 1,
            "dspark_hidden_norm_type": hidden_norm,
            "dspark_final_norm_type": final_norm,
            "markov_rank": config["dspark_markov_rank"],
            "markov_head_type": config["dspark_markov_head_type"],
            "mask_token_id": config["dspark_noise_token_id"],
            "ptd_token_id": config["dspark_noise_token_id"],
            "intermediate_size": config["dspark_ffn_hidden_size"],
            "draft_vocab_size": config["vocab_size"],
            "n_predict": config["dspark_block_size"],
            "num_nextn_predict_layers": 0,
            "num_mtp_modules": 0,
            "use_aux_hidden_state": True,
            "sample_from_anchor": True,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": config["rope_theta"],
                "partial_rotary_factor": config["rotary_dim"] / config["head_dim"],
            },
            "dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
    )
    config.pop("quantization_config", None)
    return config


def read_header(path):
    with path.open("rb") as file:
        size = int.from_bytes(file.read(8), "little")
        if not 0 < size < MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError("Invalid safetensors header size")
        return json.loads(file.read(size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--hidden-norm", choices=("rms", "gemma"), required=True)
    parser.add_argument("--final-norm", choices=("rms", "gemma"), required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or destination in source.parents or source in destination.parents:
        raise ValueError("Destination must be separate from the source checkpoint")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to replace a nonempty destination")
    if (source / "adapter-preparation.json").exists():
        raise ValueError("Source contains reserved provenance file adapter-preparation.json")
    source_bytes = (source / "config.json").read_bytes()
    original = json.loads(source_bytes)
    config = normalized_config(original, args.hidden_norm, args.final_norm)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    all_header = {}
    for filename in sorted(set(index["weight_map"].values())):
        for name, metadata in read_header(source / filename).items():
            if name == "__metadata__":
                continue
            if name in all_header:
                raise ValueError(f"Duplicate tensor across shards: {name}")
            all_header[name] = metadata
    validate_header(all_header, config)
    if set(all_header) != set(index["weight_map"]):
        raise ValueError("Source index and actual shard header do not agree")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in source.iterdir():
        if path.is_file() and path.name != "config.json":
            os.symlink(str(path), destination / path.name)
    evidence = {
        "source": str(source),
        "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "validated_tensor_count": len(all_header),
        "draft_norm_choices": {"hidden_norm": args.hidden_norm, "final_norm": args.final_norm},
        "note": "Experimental adapter; auxiliary norm choices require acceptance verification",
    }
    (destination / "adapter-preparation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
