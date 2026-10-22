# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from setuptools import find_packages, setup

setup(
    name="minimax-dspark-flash-adapter",
    version="0.0.1",
    packages=find_packages(include=["minimax_dspark_adapter", "minimax_dspark_adapter.*"]),
    entry_points={"vllm.general_plugins": ["minimax_dspark_flash=minimax_dspark_adapter:register"]},
)
