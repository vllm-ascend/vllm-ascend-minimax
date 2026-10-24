#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

# Run inside the installed Ascend container. Optional first argument: model path.
set -euo pipefail

model_path=/mnt/weight/minimax_m3dot1_w8a8_2
if (( $# > 0 )) && [[ "$1" != --* ]]; then
    model_path=$1
    shift
fi

cd /workspace
# MiniMax's sparse indexer and DSpark use the V2 runner cache/metadata path.
export VLLM_USE_V2_MODEL_RUNNER=1
exec vllm serve "$model_path" \
    --host 0.0.0.0 --port 8000 \
    --api-server-count 1 \
    --served-model-name minimax-m3-dspark \
    --trust-remote-code --dtype bfloat16 --quantization ascend \
    --tensor-parallel-size 4 --data-parallel-size 2 \
    --enable-expert-parallel \
    --disable-hybrid-kv-cache-manager \
    --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --reasoning-parser minimax_m3 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}' \
    "$@"
