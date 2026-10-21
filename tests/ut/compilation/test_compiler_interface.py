# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.compilation.compiler_interface import _configure_backend


@pytest.mark.parametrize("enable_static_kernel", [False, True])
def test_npugraph_static_kernel_blacklist(monkeypatch, enable_static_kernel):
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    blacklist_setter = MagicMock()
    static_kernel = ModuleType("npugraph_ex._acl_concrete_graph.static_kernel")
    static_kernel._set_static_kernel_blacklist = blacklist_setter
    monkeypatch.setitem(sys.modules, static_kernel.__name__, static_kernel)
    process_options = MagicMock()
    ascend_config = SimpleNamespace(enable_static_kernel=enable_static_kernel, enable_super_kernel=False)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(local_world_size=2, data_parallel_size_local=4),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8]),
    )

    _configure_backend(SimpleNamespace(), ascend_config, vllm_config, process_options)

    options = process_options.call_args.args[1]["options"]
    if not enable_static_kernel:
        blacklist_setter.assert_not_called()
        assert "static_kernel_compile" not in options
        return

    blacklist_setter.assert_called_once()
    blacklist = blacklist_setter.call_args.args[0]
    assert blacklist == [
        "TopKV2",
        "GenericBlockSparseAttention",
        "QuantBatchMatmulV3",
        "TransposeQuantBatchMatMul",
        "QuantLightningIndexer",
    ]
    # npugraph_ex excludes operators by substring matching the JSON filename.
    assert any(op in "static_kernel_GenericBlockSparseAttention_001.json" for op in blacklist)
    assert not any(op in "static_kernel_MatMulV2_001.json" for op in blacklist)
    assert options["static_kernel_compile"] is True
    assert options["_vllm_aclnn_static_kernel_sym_range"] == [1, 2, 4]


def test_torchair_backend_does_not_import_npugraph_blacklist(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setitem(sys.modules, "npugraph_ex._acl_concrete_graph.static_kernel", None)
    config = MagicMock()
    ascend_config = SimpleNamespace(enable_static_kernel=True)
    vllm_config = SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8]),
    )

    _configure_backend(config, ascend_config, vllm_config)

    assert config.experimental_config.aclgraph._aclnn_static_shape_kernel is True
    assert config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range == [1, 2, 4]
