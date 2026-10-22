# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Check decode scores and time the bundled MsaIndexScore with ACL Graphs.

Run baseline and candidate in separate processes with their respective editable
installations and custom OPP packages. This measures one operator, not serving.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

BLOCK_SIZE = 128
NUM_HEADS = 4
HEAD_DIM = 128
SCORE_ALIGNMENT = 16
FILL_THRESHOLD = -1.0e30


def _make_case(dtype, q_len, kv_lens):
    generator = torch.Generator().manual_seed(2026)
    pages = math.ceil(max(kv_lens) / BLOCK_SIZE)
    query_cpu = torch.randint(-4, 5, (len(kv_lens) * q_len, NUM_HEADS, HEAD_DIM), generator=generator).float() / 8
    key_cpu = torch.randint(-4, 5, (pages, BLOCK_SIZE, 1, HEAD_DIM), generator=generator).float() / 8
    table_cpu = torch.stack([torch.randperm(pages, generator=generator) for _ in kv_lens]).int()
    args = (
        query_cpu.to(dtype).npu(),
        key_cpu.to(dtype).npu(),
        table_cpu.npu(),
        torch.zeros(len(kv_lens), dtype=torch.int32).npu(),
    )
    kwargs = {
        "atten_mask": torch.zeros((2048, 2048), dtype=torch.int8, device="npu"),
        "actual_seq_qlen": torch.arange(len(kv_lens) + 1, dtype=torch.int32).mul(q_len).npu(),
        "actual_seq_klen": torch.tensor(kv_lens, dtype=torch.int32).npu(),
        "layout_key": "BBND",
        "sparse_mode": 3,
        "init_blocks": 0,
        "local_blocks": 0,
    }
    width = math.ceil(pages / SCORE_ALIGNMENT) * SCORE_ALIGNMENT
    expected = torch.full((NUM_HEADS, len(kv_lens) * q_len, width), torch.finfo(torch.float32).min)
    for batch, kv_len in enumerate(kv_lens):
        logical_key = key_cpu[table_cpu[batch].long(), :, 0, :].reshape(-1, HEAD_DIM)
        for token in range(q_len):
            visible = kv_len - q_len + token + 1
            blocks = math.ceil(visible / BLOCK_SIZE)
            scores = query_cpu[batch * q_len + token] @ logical_key[: blocks * BLOCK_SIZE].T
            scores[:, visible:] = -torch.inf
            expected[:, batch * q_len + token, :blocks] = scores.reshape(NUM_HEADS, blocks, BLOCK_SIZE).amax(dim=-1)
    return args, kwargs, expected


def _check_score(actual, expected):
    actual = actual.float().cpu()
    valid = expected > FILL_THRESHOLD
    torch.testing.assert_close(actual <= FILL_THRESHOLD, ~valid, rtol=0, atol=0)
    torch.testing.assert_close(actual[valid], expected[valid], rtol=1.0e-3, atol=1.0e-3)


def _measure_case(dtype, q_len, kv_lens, iterations, trials):
    args, kwargs, expected = _make_case(dtype, q_len, kv_lens)

    def call():
        return torch.ops._C_ascend.npu_msa_index_score(*args, **kwargs)

    for _ in range(10):
        output = call()
    torch.npu.synchronize()
    _check_score(output, expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(iterations):
            output = call()
    for _ in range(2):
        graph.replay()
    torch.npu.synchronize()
    _check_score(output, expected)
    timings = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / iterations)
    _check_score(output, expected)
    return {
        "dtype": str(dtype),
        "q_len": q_len,
        "kv_lens": kv_lens,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "iterations": iterations,
        "trials_us": timings,
        "median_us": statistics.median(timings),
        "correctness": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.trials <= 0:
        parser.error("iterations and trials must be positive")
    torch.set_num_threads(8)
    cases = (
        (1, (8192,)),
        (1, (32768, 16384, 4096, 129)),
        (1, tuple(2048 * (1 + i % 8) for i in range(32))),
        (4, (32768, 16384, 4096, 129)),
    )
    results = []
    for dtype in (torch.float16, torch.bfloat16):
        for q_len, kv_lens in cases:
            result = _measure_case(dtype, q_len, kv_lens, args.iterations, args.trials)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "isolated operator ACL Graph timing", "results": results}, indent=2))


if __name__ == "__main__":
    main()
