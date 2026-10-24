# MiniMax M3.1 Flash BF16 DSpark

该适配直接读取主模型目录下 `dspark/*.safetensors` 中的原生 BF16 权重。
安装本仓库后，模型注册、配置补齐和架构兼容 patch 随 Ascend 插件初始化自动生效。
不需要单独安装 adapter、运行 `prepare_draft_view.py`、手动执行 patch 或转换权重。

## 权重与配置

已核对 `/mnt/weight/minimax_m3dot1_w8a8_2/dspark`：包含 62 个 BF16 tensor，
没有 FP8 tensor 或量化 scale。主模型仍按自己的 ModelSlim 量化描述加载，
草稿模型保持 BF16，不继承主模型量化方式。

服务优先读取 `dspark/config.json`。本次已将
[官方配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e/dspark/config.json)
原样复制到该路径，SHA256 为
`d3f2ef9a308be0c5689e98b348d8cff73b90487db22ba79e9269f9b1f25cfa3a`。
配置包括：
5 层草稿、FFN 12288、Markov rank 256、block size 7、window 4096、
target layers `[3, 17, 31, 45, 59]`、noise token 200058，以及 Gemma norm。
代码在内存中整理上述参数，并按实际权重关闭草稿量化；不改写配置文件内容。
配置缺失的同规格导出可回退到上述官方参数。该适配限定于 hidden size 6144、
60 层、vocab size 200064 的目标模型，并校验全部草稿权重名称、形状和 dtype。
原始权重 tensor 和 vLLM 源文件不被写入。

官方仓库的 DSpark 权重含 FP8 与 scale，其索引不能直接用于当前 BF16 shard。
本次只复制官方配置，使用当前目录的 62 个 BF16 tensor，无需重新下载或转换权重。

旧 FP8 + scale 草稿会被明确拒绝，不执行自动反量化。

## 安装与服务

使用配套 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，
在匹配硬件的 v0.30.0rc2 镜像中正常安装本仓库：

```bash
cd /vllm-workspace/vllm-ascend
SOC_VERSION=ascend950dt_9582 pip install --no-build-isolation --no-deps -e .
```

首次安装需编译私仓配套的 GBSA 等原生算子，保留镜像的 Torch、torch_npu 和
Transformers 版本。不要用关闭原生编译替代完整安装。

上述 SOC 对应本机 950DT 9582；其他机器按实际型号设置。
950DT 单机 TP4 × DP2 使用 V2 runner 启动：

可直接在容器内执行仓库提供的
[部署脚本](../../../../examples/spec_decode/minimax_m3_flash_dspark/serve.sh)：

```bash
bash /vllm-workspace/vllm-ascend/examples/spec_decode/minimax_m3_flash_dspark/serve.sh
```

脚本仅启动服务，不安装额外插件或转换权重。等价的直接启动命令如下：

```bash
cd /workspace
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /mnt/weight/minimax_m3dot1_w8a8_2 \
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
  --additional-config '{"enable_flashcomm1":true,"ascend_compilation_config":{"fuse_qknorm_rope":false,"fuse_norm_quant":false}}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}'
```

`enforce_eager` 在上述 speculative config 中仅作用于草稿。服务参数明确使用
DSpark、7 个推测 token，并关闭 `fuse_qknorm_rope`。
无需提供额外 draft 路径，代码自动选择主模型目录中的 `dspark/`。
关闭 hybrid KV cache manager，使草稿滑窗 cache 使用完整分配，保持 MiniMax
indexer 的 128-token block；代价是草稿 cache 会占用更多内存。

## 功能验证

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3-dspark","messages":[{"role":"user","content":"请简短介绍北京。"}],"max_tokens":128,"temperature":0}'
```

应同时确认 HTTP 200、非空生成内容及 DSpark draft/accepted token 指标。
仅配置构建成功或服务 readiness 不代表推理验证通过。
以上是启动示例，准确率、吞吐及长上下文能力需另行评测。

## 本次实权重验证

2026-10-24，在全新 `v0.30.0rc2-a5-openeuler` 容器、8 张 950DT 9582 上，
使用上述部署脚本和原始权重完成验证：

| 检查 | 结果 |
| --- | --- |
| `/v1/models` | HTTP 200，模型名 `minimax-m3-dspark` |
| 文本问答 | HTTP 200，返回非空中文回答 |
| 336 × 336 红色色块图片问答 | HTTP 200，回答“红色（纯红色块）” |
| DSpark | 两次请求共 68 轮 draft、476 个 draft tokens、98 个 accepted tokens |
| 目标模型 ACLGraph | 完成捕获，日志确认 V2 target graph replay active |
| EP / FlashComm1 | TP4 × DP2、EP8，日志确认 FlashComm1 启用 |

476 / 68 = 7，与配置的每轮 7 个推测 token 一致。上述指标只用于确认推测执行，
两个样例不能代表接受率、准确率或吞吐基准。此次使用 8192 上下文、最大 16 序列，
未验证 128k 上下文或并发容量上限；草稿使用 eager，目标模型使用 FULL_DECODE_ONLY。

相关单元测试 48 项通过。扩展运行三个 MiniMax 测试文件得到 170 passed / 16 failed；
相同的 16 个失败在独立 release 基线 `482d91c2` 中复现，未声明完整测试集通过。
原始验证响应、指标和服务日志保存在服务器的
`/home/x00939195/native-dspark-pr19-20261024/validation.json`、`metrics.txt` 和 `serve.log`。
