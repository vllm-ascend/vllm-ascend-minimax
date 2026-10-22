# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Strict, CPU-testable checkpoint schema; no vLLM imports required."""

import math

import regex as re

PREFIX = "language_model.model.dspark."
FP8_BLOCK_SIZE = 32
DEQUANTIZATION_ROWS_PER_CHUNK = 256
E8M0_EXPONENT_BIAS = 127
E8M0_NAN_ENCODING = 255


def map_weight_name(name):
    if not name.startswith(PREFIX):
        raise ValueError(f"Unexpected tensor outside standalone DSpark: {name}")
    name = name[len(PREFIX) :]
    name = re.sub(r"^(layers\.\d+)\.decoder_layer\.", r"\1.", name)
    if name.startswith("final_norm."):
        name = "norm." + name[len("final_norm.") :]
    return name


def expected_shapes(config):
    h = config["hidden_size"]
    hd = config["head_dim"]
    q = config["num_attention_heads"] * hd
    kv = config["num_key_value_heads"] * hd
    ffn = config["dspark_ffn_hidden_size"]
    n = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    rank = config["dspark_markov_rank"]
    shapes = {
        "fc.weight": [h, h * len(config["dspark_target_layer_ids"])],
        "hidden_norm.weight": [h],
        "norm.weight": [h],
        "markov_head.markov_w1.weight": [vocab, rank],
        "markov_head.markov_w2.weight": [vocab, rank],
        "confidence_head.proj.weight": [1, h + rank],
        "confidence_head.proj.bias": [1],
    }
    for i in range(n):
        p = f"layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[p + norm + ".weight"] = [h]
        for norm in ("q_norm", "k_norm"):
            shapes[p + "self_attn." + norm + ".weight"] = [hd]
        matrices = {
            "self_attn.q_proj": [q, h],
            "self_attn.k_proj": [kv, h],
            "self_attn.v_proj": [kv, h],
            "self_attn.o_proj": [h, q],
            "mlp.gate_proj": [ffn, h],
            "mlp.up_proj": [ffn, h],
            "mlp.down_proj": [h, ffn],
        }
        for name, shape in matrices.items():
            shapes[p + name + ".weight"] = shape
            shapes[p + name + ".weight_scale_inv"] = [shape[0], math.ceil(shape[1] / FP8_BLOCK_SIZE)]
    return shapes


def validate_header(header, config):
    expected = expected_shapes(config)
    mapped = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        mapped_name = map_weight_name(name)
        if mapped_name in mapped:
            raise ValueError(f"Duplicate mapped tensor: {mapped_name}")
        mapped[mapped_name] = info
    if set(mapped) != set(expected):
        raise ValueError(
            f"Tensor schema mismatch: missing={set(expected) - set(mapped)}, extra={set(mapped) - set(expected)}"
        )
    for name, shape in expected.items():
        if list(mapped[name]["shape"]) != shape:
            raise ValueError(f"Tensor shape mismatch: {name}, {mapped[name]['shape']} != {shape}")
        dtype = mapped[name]["dtype"]
        if name.endswith("weight_scale_inv"):
            if dtype != "U8":
                raise ValueError(f"Expected U8 E8M0 scale, got {name}: {dtype}")
        elif name.endswith(".weight") and name.replace(".weight", ".weight_scale_inv") in expected:
            if dtype != "F8_E4M3":
                raise ValueError(f"Expected E4M3 weight, got {name}: {dtype}")
        elif dtype != "BF16":
            raise ValueError(f"Expected BF16 auxiliary/norm tensor, got {name}: {dtype}")
    return mapped


def resolve_fp8(weight, scale, out_dtype):
    """Resolve FP8 * numeric scale per 32 input columns on the CPU.

    The checkpoint stores E8M0 scales as U8 exponent bytes, biased by 127.
    They multiply E4M3 values after decoding; raw bytes are not numeric scales.
    Keep staging bounded and compute the scale product in float32.
    """
    import torch

    if weight.device.type != "cpu" or scale.device.type != "cpu":
        raise ValueError("Draft FP8 resolution must run on CPU")
    if weight.ndim != 2 or scale.shape != (weight.shape[0], math.ceil(weight.shape[1] / FP8_BLOCK_SIZE)):
        raise ValueError("Draft FP8 matrix and 1x32 scales do not agree")
    if scale.dtype != torch.uint8:
        raise ValueError("Draft weight_scale_inv must be U8 E8M0")
    if (scale == E8M0_NAN_ENCODING).any():
        raise ValueError("E8M0 exponent 255 denotes NaN")
    out = torch.empty(weight.shape, dtype=out_dtype, device="cpu")
    for start in range(0, weight.shape[0], DEQUANTIZATION_ROWS_PER_CHUNK):
        stop = min(start + DEQUANTIZATION_ROWS_PER_CHUNK, weight.shape[0])
        decoded = torch.ldexp(
            torch.ones(scale[start:stop].shape, dtype=torch.float32), scale[start:stop].int() - E8M0_EXPONENT_BIAS
        )
        repeated = decoded.repeat_interleave(FP8_BLOCK_SIZE, dim=1)[:, : weight.shape[1]]
        out[start:stop] = weight[start:stop].float() * repeated
    return out


def iter_resolved_weights(weights, config, dtype):
    import torch

    tensors = {}
    for name, value in weights:
        if name in tensors:
            raise ValueError(f"Duplicate input tensor: {name}")
        tensors[name] = value
    header = {}
    dtype_names = {torch.bfloat16: "BF16", torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3"}
    for name, value in tensors.items():
        header[name] = {"shape": list(value.shape), "dtype": dtype_names.get(value.dtype, str(value.dtype))}
    validate_header(header, config)
    for name in list(tensors):
        if name.endswith(".weight_scale_inv"):
            continue
        value = tensors.pop(name)
        scale_name = name.replace(".weight", ".weight_scale_inv")
        if name.endswith(".weight") and scale_name in tensors:
            value = resolve_fp8(value, tensors.pop(scale_name), dtype)
        yield map_weight_name(name), value
    if tensors:
        raise ValueError(f"Unconsumed DSpark tensors: {tuple(tensors)}")
