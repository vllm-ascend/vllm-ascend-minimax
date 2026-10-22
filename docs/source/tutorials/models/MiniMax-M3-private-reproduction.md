# MiniMax-M3.1-Flash private source reproduction

This page records the private A3 W8A8 integration snapshot. It distinguishes
repository patches from native build artifacts and deployment configuration.
The completed checks below do not establish full-model or GPQA accuracy.

## Audited A3 source and image identity

| Component | Pinned identity |
| --- | --- |
| Private vLLM Ascend source | `83a9f2d284660f284a9239a91d603398705e8d1f` |
| Paired vLLM source | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| A3 base image | `sha256:704ca8fa3c6b82047185ed40b0376dd0f321d520b7c50bfc5317a1d00093492f` |
| Reproduction image tag | `m3.1flash0day:b109-83a9f2d28-20261023` |
| Reproduction image ID | `sha256:d14f442f3435fb86bd15fa9dec4967b824f8bc33a2836f7d2e1da56341f00bca` |

These identities describe the audited A3 snapshot before the FP8 softmax fix
below. They are historical artifact identities, not aliases for the latest
private branch. Read the actual Git revision and image manifest for each run.

The source snapshot includes private PRs [#2](https://github.com/ycl-ai-infra/vllm-ascend-private/pull/2),
[#4](https://github.com/ycl-ai-infra/vllm-ascend-private/pull/4), and
[#6](https://github.com/ycl-ai-infra/vllm-ascend-private/pull/6).
PR #3's BF16 compatibility changes were reverted by the pinned commit.
Always resolve the paired framework revision from `.github/vllm-main-verified.commit`;
do not pair a private plugin commit with an arbitrary framework branch head.

The supplied image is an internal artifact, not a published registry image.
Its source manifest and file checksums are under `/opt/m31flash/`.
Weights, host addresses, interface selection, and credentials are deployment inputs.

## Native build requirements

PR #2 changes MsaIndexScore C++/AscendC implementations. PR #4 adds bundled GBSA,
GBSA metadata, AICPU code, and Torch dispatch bindings. Updating Python source or
committing a container with bind-mounted source does not include these native changes.

Build the paired vLLM for the empty backend and install the plugin from its pinned
source with native compilation enabled. Use `--no-build-isolation --no-deps` to
retain the image's tested Torch, torch_npu, and Triton versions. The recorded base
has Python 3.11.10, Torch 2.10.0, torch_npu 2.10.0.post7, and CANN 9.2.0.
Setuptools >=77 and setuptools-rust are build prerequisites; the optional Rust
frontend was not required for this Python serving integration.

Set `SOC_VERSION` to the actual A3 SoC and bound CPU build concurrency, for example
`MAX_JOBS=64 CMAKE_BUILD_PARALLEL_LEVEL=64`. Do not set `COMPILE_CUSTOM_KERNELS=0`
as a substitute for rebuilding the native package. A Python-only editable
registration may follow a separately verified native build.

An installed `fla_npu_opp_env.pth` can register competing OPP definitions during
the custom-operator build. In an isolated build container, temporarily disable
that hook and restore it after the build. Preserve the original file and ensure
restoration on build failure as well. This is a build environment adjustment,
not an additional framework source patch.

Install the complete generated `_cann_ops_custom` package, including
the Scatter ND update operator, both GBSA entry points, metadata AICPU libraries, and
their JSON definitions. Copying only a Torch `.so` or only GBSA leaves an
incomplete runtime. In the earlier deployment, restoring the complete generated
package fixed a missing Scatter entry point; no Scatter source patch was made.

Configure the new vendor before NPU initialization, then start fresh processes.
See [bundled GBSA initialization and build notes](https://github.com/ycl-ai-infra/vllm-ascend-private/blob/83a9f2d284660f284a9239a91d603398705e8d1f/csrc/attention/generic_block_sparse_attention/UPSTREAM.md)
for `bootstrap_custom_op_env(include_vendor_lib=True)` and standalone tests.

The reproduction image contains native binaries rebuilt inside the recorded A3
base. Its native source inputs are unchanged by the PR #3 revert; the manifest
records both the original native-build commit and the final source commit.

## Source patch audit

An earlier environment had one additional runtime source change plus its unit
test: hardware-dependent GBSA `softmax_precision` selection for A5. The changed
GBSA function matches the precision portion of existing private
[PR #5](https://github.com/ycl-ai-infra/vllm-ascend-private/pull/5).
That PR also contains A5 index-decode changes. The precision fix is adapted
separately here to the bundled operator's current contract:

- FP8 E4M3 KV caches select `quant_mode=5` and `softmax_precision=1` in both
  metadata generation and attention execution. The Ascend 950 tiler rejects
  FP8 full quantization with precision 0.
- Unquantized inputs retain `softmax_precision=0`, including A5 BF16. Unlike the
  earlier external operator, bundled GBSA supports this mode on A5; precision 0
  uses FP32 softmax and precision 1 uses FP16/BF16 softmax. Selecting 1 for every
  A5 input would unnecessarily change BF16 precision.
- A2/A3 BF16 continue to use precision 0. No index-decode routing, operator
  registration, native source, or PR #3 BF16 compatibility code is changed.

This fixes the Python call arguments; it does not establish that the full A5
bundled dispatcher or model deployment is validated. Keep the remaining A5
integration and its hardware validation separate.

The FP8 call-site regression was reproduced with four failing unit cases on
the audited source (decode and three prefill cache layouts). With the fix,
158 MiniMax/model-quantization/compiler unit cases passed. Those tests mock the
GBSA calls and verify both operators' arguments; they do not execute A5 kernels.
An A5 NPU validation of this adapted call site remains outstanding.

The audited A3 snapshot used no uncommitted framework source patches. Its A3
`softmax_precision` remains 0. The framework's 7,155 tracked files match the
paired upstream commit, and both serving containers' 13,103 code/native files
match the packaged image. Generated operator Python files in the old deployment
matched the base package; they were not handwritten fixes.

Process-group shutdown handling, host-role checks, and AISBench configuration
serialization were deployment-script changes. They are supplied in the image's
`/opt/m31flash/deployment/` directory. They are distinct from model or operator
source patches; recreate host-specific configuration when using those scripts.

## Recorded topology and validation scope

The intended topology is one P host with PP4 x TP4 and one D host containing four
TP4 decode replicas (DP4 x TP4). Each host uses 16 A3 chips. D uses
`FULL_DECODE_ONLY`; P uses eager execution. The configured limits are P/D
`max_num_seqs=32/128`, `max_num_batched_tokens=32768`, `max_model_len=263000`, and
memory utilization 0.92. Reuse the [PD configuration guidance](MiniMax-M3.md#53-prefill-decode-disaggregation)
with the actual model, endpoints, rank mapping, and HCCS transport configuration.

The audited A3 snapshot passed 154 unit tests and 57 operator/Meta/eager/graph tests;
four A5-only FP8 tests were skipped. Both final A3 containers also passed
relocated-library graph checks and BF16/FP16 Scatter eager/capture/replay checks.
A 2 MiB Mooncake Ascend Direct NPU transfer over HCCS passed with byte comparison.
These checks do not replace full-model PD startup or a proxy-routed request.

Record the source commit, image ID, native checksums, prompt, dataset hash,
answer extractor, and sampling settings with every accuracy run. Start a new
GPQA run after changing the source or native package. The earlier partial run's
154/165 (93.33%) result belongs to its earlier source snapshot; it is not the
accuracy of this pinned snapshot. Full-model startup and a new GPQA run remain
necessary before declaring this commit an accuracy baseline.
