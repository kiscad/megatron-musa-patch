# megatron-musa-patch（中文速览）

项目首要目标是让 **Megatron-LM / Megatron-Core 在 MUSA 平台上无缝运行**：上游仓库的单元测试和训练脚本无需修改即可直接运行，**ms-swift 等上层框架调用 Megatron-Core 接口时无需区分 MUSA / CUDA**。

该契约覆盖的是 Megatron-Core **已有**的接口面，并不是「上层框架永远不许修改」的绝对规则：当模型架构过新、Megatron-LM 版本跟进较慢时，ms-swift 或 Megatron-Bridge 可能率先实现了对应支持——此时该特性的 MUSA 适配放在对应上层框架内才更合理。

本仓库承担 **Megatron-Core 既有契约面**的运行时适配。在该面上，受支持的 MUSA 环境安装本包后，调用方应继续使用原有代码，无需 MUSA 分支、额外 import、替代模型类或设备判断。这是项目的验收目标，不代表当前所有上游测试与训练配置都已通过。上层框架先行于 Megatron-Core 实现的能力（新架构、新 kernel），其设备适配归该框架所有，不在本仓库范围内。

## 兼容性契约与当前覆盖

| 使用场景 | 验收目标 |
|---|---|
| Megatron-LM 单元测试 | 原始测试及断言直接运行，包括其中的 CUDA API 写法和 `nccl` 配置。兼容问题在本包修复，不通过改写或跳过测试掩盖 MUSA 故障。 |
| Megatron-LM 训练 | 原始 Python 和 shell 训练脚本无需源码修改，也不依赖专用 MUSA 启动器；保持模型、优化器、分布式和 checkpoint 语义。 |
| ms-swift 等 Megatron-Core 调用方 | 保持 import 路径、公开签名、配置对象、输出和 state dict 契约。只安装 `megatron-core` wheel 时也必须具备 core 兼容能力，不能依赖 `megatron.training`。 |

在上述契约面内，要求调用方把 `cuda` 改成 `musa`、把 `nccl` 改成 `mccl`、添加 `import megatron_musa_patch` 或关闭原本要求的功能，属于兼容缺口；这些做法可用于定位问题，但不满足调用方无需修改的目标。上层框架为接入全新模型架构所做的接线代码则是另一回事：其设备适配按设计落在该框架内。

当前实现优先保证正确性，在必要处使用 PyTorch 归一化、同步 DP 规约、进程内 checkpoint bucket 写入等保守回退，并提供 TE/FP8、融合 RoPE 等专项检查。这些检查与示例只覆盖具体路径，不能据此宣称上游全量测试、原始训练脚本或 ms-swift 端到端已经全部兼容。尚存失败、跳过项及必需的参数覆盖都要明确记录。性能优化应在正确性验证后推进，并保持同一调用契约；回退代价及移除条件必须在 patch ledger 中可追溯。

开发机制见[贡献指南](CONTRIBUTING_zh.md)；后续 coding agent 从 [AGENTS.md](AGENTS.md) 开始，按其中的流程实施和验收。[English](README.md)。

## 快速上手

```bash
git clone https://github.com/kiscad/megatron-musa-patch
cd megatron-musa-patch && git checkout v0.16.1-dev

# MUSA 容器内的 torch / torch_musa 是互相绑定的厂商构建。直接
# `pip install .` 会解析 torchada 依赖并拉取最新上游 torch（如 2.11），
# 覆盖厂商锁定版本、破坏 MUSA 栈。请关闭依赖解析安装：
pip install --no-deps torchada==0.1.86
pip install --no-deps .

# 若希望保留依赖解析：用 constraints 锁定厂商栈（按容器内 torch 版本调整）：
#   printf 'torch==2.7.1\n' > constraints-musa.txt
#   pip install -c constraints-musa.txt .

cd /path/to/Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
    ...其余参数照旧...
```

`torch.backends` entry point 会在 `import torch` 结束时激活补丁集，`sys.meta_path` watcher 在目标模块一出现时使补丁生效。上面的命令表示沿用原有参数，实际配置仍需按下文的覆盖范围和取舍验证。激活必须发生在 Megatron 使用 CUDA API 之前；默认通道应在无需调用方改动的情况下满足这一时序。晚激活只能修复受支持的模块级别名，不能修复已经创建好的对象。

查看已注册的 patch（含根因、策略与移除条件）：

```bash
python -c "import megatron_musa_patch as m, json; print(json.dumps(m.report(), indent=2))"
```

新进程中的报告是惰性的：目标模块尚未导入时出现 `pending` 属正常现象。若要在诊断进程中主动解析目标，可先调用 `m.apply()` 再调用 `m.report()`；这不能替代原始调用方所使用的自动激活验收。

`examples/run_pretrain_smoke.sh` 是 2 卡 5 步的端到端自检：

```bash
MEGATRON_LM_PATH=/path/to/Megatron-LM NPUS=2 PYTHON=/path/to/venv/bin/python \
    bash examples/run_pretrain_smoke.sh
```

`examples/train_llama3_8b_musa.sh` 把上游 llama3-8b H100 FP8 示例适配到 MUSA
（默认经 MT-TransformerEngine 走 FP8 + RMSNorm；设 `DTYPE=bf16` 则用纯 PyTorch
local 实现；48GB 卡默认 TP=2；并缩减了默认 mock 训练计划）。完整的改动清单和
短程联调用的环境变量覆盖见脚本头部注释。

```bash
MEGATRON_LM_PATH=/path/to/Megatron-LM PYTHON=/path/to/venv/bin/python \
    bash examples/train_llama3_8b_musa.sh
```

这些示例使用了特定参数，用于诊断，不能证明所有原始上游启动器均可不修改运行。`run_pretrain_smoke.sh` 默认创建并打印唯一临时目录；显式 `OUTPUT_DIR` 必须不存在或为空，不会删除已有输出，相对路径以调用目录为准。上游测试、训练及框架调用的分步验收方法见 [AGENTS.md](AGENTS.md)。

## 需要哪种 Megatron？

任何能 import 到 `megatron` 包的安装方式都可以：

```bash
# a) Megatron-LM 源码 —— 推荐：还有 megatron.training 可用
git clone https://github.com/NVIDIA/Megatron-LM
git -C Megatron-LM checkout core_v0.16.1
export PYTHONPATH=/path/to/Megatron-LM

# b) 只装 pip wheel —— 只有 megatron.core，通常由上层训练框架带入
pip install megatron-core==0.16.1
```

方式 b) 是上层框架消费 [Megatron-Core](https://pypi.org/project/megatron-core/) 时的典型场景——例如 **ms-swift** 的 Megatron 训练（要求 `megatron-core>=0.16`）或 **Megatron Bridge 0.1–0.3**（对应 Megatron-Core 0.14–0.16）。此时只有 `megatron.core` / `torch` 侧的 patch 会生效；`megatron.training` / `megatron.legacy` 的 patch 会被记录为 `skipped`，属预期行为。**Megatron Bridge ≥ 0.4 需要 Megatron-Core 0.17+**，超出本包支持范围。

本包刻意不把 Megatron 声明成 pip 依赖：wheel 只含 `megatron/core/`，PyPI 上也没有 `megatron-lm`，而真实用法需要锁定到某个 tag 的源码 checkout。支持的上游范围：`>=0.14,<0.17`——超出会告警（设 `MEGATRON_MUSA_PATCH_STRICT=1` 则报错）。

## 保守调试参数

以下参数用于选择保守的 PyTorch 路径以缩小故障范围，是可选调试手段，不是兼容性契约，也不应成为下游必须编码的前提。若原始配置只有添加这些参数才能运行，仍需将原始失败记录为待修复缺口：

```
--transformer-impl local
--no-masked-softmax-fusion --no-gradient-accumulation-fusion
--no-bias-swiglu-fusion --no-bias-dropout-fusion --no-rope-fusion
--attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
--distributed-backend nccl        # torchada 会自动改写成 mccl
CUDA_DEVICE_MAX_CONNECTIONS=1
```

`--no-rope-fusion` 不再是启动的必要条件：MUSA 上融合 rotary embedding 由 apex 的 kernel 提供（见[补丁目录](docs/PATCHES_zh.md)中 rope 相关条目）；`tests/test_rope.py` 中需显式开启的硬件用例会将融合的 `sbhd`/`thd` 结果与 Megatron 的非融合参考实现比对。若仍想走纯 PyTorch 路径，可以继续保留该参数——`rotary_interleaved` 模型和 context parallel 下的 packed 序列会自动回退到该路径，并打印一次告警。

## 改了什么

MUSA 版 PyTorch 完全没有 `_cuda_*` 绑定，`torch.cuda` 是空壳，而 Megatron 到处引用它。[torchada](https://pypi.org/project/torchada/)（摩尔线程自己的 CUDA→MUSA 适配层）负责机械翻译；本包在其之上补 4 个 Megatron 专用覆盖，外加约四十个 Megatron patch：torch.cuda / torch.distributed 兼容层、LayerNorm/RMSNorm 与 TE norm-linear 回退、TE attention 能力分发与 FP8 模型初始化、MoE topk/permutation/grouped-GEMM 降级、MT apex fork 的融合 RoPE、GDN chunked gated delta rule 的 torch-kernels TileLang kernel、checkpoint 写入与屏障加固，以及训练参数兼容。

逐条补丁的目录（每个补丁的目标与根因、kernel 栈版本绑定、已退役补丁）见[补丁目录](docs/PATCHES_zh.md)。实时、机器可读的清单（`id`、`rationale`、`strategy`、`upstream`、`remove_when`）是 `megatron_musa_patch.report()`。

### 需要知道的取舍

* `torch.cuda.is_available()` 是对 MUSA 的实时探测，不是常量——但上游有几处把它当作「是不是 NVIDIA」的探针（例如 FP8 checkpoint 路径用它决定是否 import TransformerEngine）；要开 FP8 checkpoint 请先复核这几处。
* checkpoint bucket 写入在进程内串行（吞吐换稳定，不改格式）。如果你的 MUSA 版本可以 fork-after-init，设 `MEGATRON_MUSA_PATCH_CKPT_FORK=1` 恢复上游的 fork 式 writer。Megatron 外层的异步保存路径（`async_utils.DynamicAsyncCaller`）仍会独立 fork，不受本 patch 影响。
* DP-overlap 开关（`--overlap-grad-reduce` / `--overlap-param-gather`）被 fused-wgrad/DDP 集成策略关闭，可能降低吞吐。`MEGATRON_MUSA_PATCH_DP_OVERLAP=1` 可恢复上游行为用于测试（旧写法 `MEGATRON_MUSA_PATCH_TP_OVERLAP` 仍有效）。`tp_comm_overlap` 不受影响。
* 单独的 `--profile` 会自动启用 `--use-pytorch-profiler`，并在 rank 0 打印警告。用 `MEGATRON_MUSA_PATCH_DISABLE` 跳过 `megatron.training.profile.pytorch` 可恢复上游 profiler 选择。
* rope 融合是 MUSA 上的 kernel 选择，而非上游默认实现：融合 kernel 来自摩尔线程的 apex，且只融合 apex 实现了的组合。`rotary_interleaved` 模型以及 context parallel 下的 packed 序列会降级到 Megatron 的非融合 rotary embedding，并打印一次告警——结果正确、速度较慢，但不再直接报错。`MEGATRON_MUSA_PATCH_ROPE_FUSION=0` 可完全拒绝该回退，此时上游的 `apply_rope_fusion is not available` 报错依旧存在（需自行加 `--no-rope-fusion`）。
* chunked gated delta rule（Qwen3.5 / Qwen3-Next 这类混合模型的 GDN 层）同样是 kernel 选择：flash-linear-attention 的 Triton kernel 仍是正确性参照和回退路径，落在 torch-kernels 已审计包络内的调用改走其 TileLang kernel——单算子（fwd+bwd、4K 序列）快 4.8–6.0 倍，8 卡 ms-swift Qwen3.5 端到端步时长低 2.0–9.5%。TileLang kernel 首次使用会 JIT 编译，多 rank 运行前请先用 `examples/warm_gdn_tilelang.py` 预热；TileLang 栈与 torch_musa 构建及 musa_toolkits 版本绑定。实测数据、预热流程与版本矩阵见[补丁目录](docs/PATCHES_zh.md)。`MEGATRON_MUSA_PATCH_GDN_TILELANG=0` 可全部恢复 FLA。

### 补丁独立性

尽量独立选择补丁；local norm 的两个标志位需要
`megatron.fusions.fused-layer-norm.pure-torch`，checkpoint 的 `host-barrier-context`
需要 `host-barrier-proxy`。这些同模块协作关系通过 `report()[...]["requires"]` 显式报告；
前置补丁缺失、禁用或主动退出时，消费者记录带原因的 `skipped`，不会隐式启用前置补丁。
block norm 在 local class 补丁禁用时自行构造回退。RoPE dispatcher 在调用时检查实际 kernel，
使用私有配置副本；没有 apex kernel 时只透传。完整边界见[补丁独立性检查记录](docs/PATCH_INDEPENDENCE.md)。

## 手动生效

三个通道都幂等。自动激活和显式 import 注册 watcher，不主动导入 Megatron，但会处理已经加载的模块；`apply()` 则主动导入目标并立即应用补丁。显式激活用于诊断和扩展，调用方无需修改的验收必须覆盖自动通道：

| 通道 | 触发点 | 关闭开关 |
|---|---|---|
| `torch.backends` entry point（默认） | `import torch` 结尾 | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 或 `MEGATRON_MUSA_PATCH_AUTOLOAD=0` |
| `import megatron_musa_patch` | 显式 import | `MEGATRON_MUSA_PATCH=0` |
| `megatron_musa_patch.apply()` | 显式、立即 | `MEGATRON_MUSA_PATCH=0` |

### 从上游源码迁移的修改

启动时间同步、checkpoint 主机屏障和信号别名注册现位于 `patches/_control_collectives.py`，无需修改 Megatron 源码。torch 代理仅绑定到两个 Megatron 模块，其他通信调用直接透传。Gloo 组按 distributed world 身份缓存，由正常进程组销毁流程清理；禁用任意一个 checkpoint patch 都不会启用屏障重定向。safe-globals patch 扩展待注册列表；PyTorch 已注册的别名在卸载后仍保留，与上游注册生命周期一致。

原先修改过的 H100 FP8 示例迁移为启动器，保留 16 层及缩减后的训练样本计划：

```bash
MEGATRON_LM_PATH=/path/to/Megatron-LM PYTHON=/path/to/venv/bin/python \
    bash examples/train_llama3_8b_h100_fp8.sh
```

启动器执行未修改的上游脚本，通过命令行覆盖 `NUM_LAYERS`、`TRAIN_SAMPLES`、`LR_DECAY_SAMPLES`、`LR_WARMUP_SAMPLES`，这些只是示例默认值，不会更改其他训练任务的参数。

## 环境变量开关

| 变量 | 默认 | 含义 |
|---|---|---|
| `MEGATRON_MUSA_PATCH` | `1` | 总开关。`0` 同时关闭自动通道和 `import megatron_musa_patch`。 |
| `MEGATRON_MUSA_PATCH_AUTOLOAD` | `1` | 只关自动（entry point）通道。 |
| `MEGATRON_MUSA_PATCH_DISABLE` | 空 | 逗号分隔的 patch id，跳过这些 patch。 |
| `MEGATRON_MUSA_PATCH_ONLY` | 空 | 白名单，优先级高于 `..._DISABLE`；未知 id 会告警。 |
| `MEGATRON_MUSA_PATCH_STRICT` | `0` | 上游版本超出支持范围时报错而不是警告。 |
| `MEGATRON_MUSA_PATCH_DEBUG` | `0` | 以 INFO 级别打印每个已生效的 patch。 |
| `MEGATRON_MUSA_PATCH_ARCH` | `8.3` | 合成的 NVIDIA 架构号，例如 `9.0`。 |
| `MEGATRON_MUSA_PATCH_BLOCK_LAYERNORM` | `local` | 设为 `upstream` 则保留 `LayerNormImpl = TENorm`。 |
| `MEGATRON_MUSA_PATCH_TE_FUSED_LAYERNORM` | `0` | 设为 `1` 恢复上游 TE 融合 norm-linear，供升级验证。 |
| `MEGATRON_MUSA_PATCH_TE_NORM` | `0` | 设为 `1` 保留原生 TE 独立 LayerNorm/RMSNorm，用于升级验证。 |
| `MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES` | 空 | `1`/`true`/`*` 放行所有版本门控；包名列表只放行指定包。不会绕过能力探针和补丁选择。 |
| `MEGATRON_MUSA_PATCH_ROPE_FUSION` | `1` | 设为 `0` 拒绝 apex 融合 RoPE 回退，上游会继续报告 `apply_rope_fusion` 不可用。 |
| `MEGATRON_MUSA_PATCH_GDN_TILELANG` | `1` | 设为 `0` 拒绝 torch-kernels TileLang 的 chunked gated delta rule 分发，全部保留 flash-linear-attention。 |
| `MEGATRON_MUSA_PATCH_ATTN_BACKEND` | `auto` | MuDNN flash 反向不支持的输入形状的 kernel 选择顺序：`auto`（实测反向安全窗口内用原生 flash，其次 mate 的 TileLang flash，最后 TE 非融合后端）、`mudnn`（前向可跑即用原生 flash）、`mate`（优先 TileLang kernel）、`unfused`（参考后端）。 |
| `MEGATRON_MUSA_PATCH_JIT_WARMUP` | `0` | 设为 `1` 保留上游的 JIT 预热。 |
| `MEGATRON_MUSA_PATCH_CKPT_FORK` | `0` | 设为 `1` 保留上游的 fork 式 checkpoint writer。 |
| `MEGATRON_MUSA_PATCH_DP_OVERLAP` | `0` | 设为 `1` 恢复 DP-overlap 开关（旧写法 `MEGATRON_MUSA_PATCH_TP_OVERLAP` 在本变量未设置时生效）。 |
| `MEGATRON_MUSA_PATCH_TEARDOWN` | `1` | 设为 `0` 跳过进程组清理钩子。 |

独立 norm 和 block norm patch 覆盖不到 TE `LayerNormLinear` 内部的归一化。RMSNorm 直接构造上游 TE 融合模块，保留原生前向/反向路径，不增加运行时封装。norm-linear 回退仅对 LayerNorm 生效，覆盖使用 `TELayerNormColumnParallelLinear` 的 Megatron spec，包括 GPT 的 QKV、FC1，保留 `layer_norm_weight`/`layer_norm_bias` 及其副本式 checkpoint 分片。FP8 量化和 GEMM 仍由 TE 执行；取消融合会影响性能和数值舍入。直接使用 TE 模块或 TE operation fuser 不在此替换范围内。跨模块类型或 TE 版本的 FP8 extra-state 兼容性仍需 checkpoint 验证。

运行 norm-linear CPU 与单卡/双卡 MUSA 回归测试：

```bash
MEGATRON_MUSA_RUN_INTEGRATION=1 MEGATRON_LM_PATH=/path/to/Megatron-LM \
    python -m pytest tests/test_te_layer_norm.py -q
```

## 排障

* **`PatchTargetMissing`** —— 上游改名或删掉了符号；报错里有 patch id、符号和检测到的 Megatron 版本。用 `MEGATRON_MUSA_PATCH_DISABLE=<id>` 跳过，或更新目标（见[贡献指南](CONTRIBUTING_zh.md#6-如何新增一个-patch)）。
* **训练仍然找不到 CUDA** —— 先确认激活状态：`python -c "import megatron_musa_patch as m, json; print(json.dumps(m.report(), indent=2))"`。全部 `pending` 通常表示目标未导入；报告为空表示当前未注册补丁，例如设了 `MEGATRON_MUSA_PATCH=0`。另行检查 entry point 元数据：`python -c "from importlib.metadata import entry_points; print(entry_points(group='torch.backends'))"`。
* **某个 patch 帮了倒忙** —— 用 `MEGATRON_MUSA_PATCH_DISABLE=<id>` 逐个定位，或 `MEGATRON_MUSA_PATCH=0` 整体关掉。`MEGATRON_MUSA_PATCH_DEBUG=1` 会打印每个 patch 的落地过程。

## 版本约定

包版本跟随所 patch 的 Megatron-LM 版本：本分支（`v0.16.1-dev`）上是 `0.16.1.dev0`，切发布版为 `0.16.1`，其上的修复为 `0.16.1.post1`，下一次跟进上游在新分支 `v0.16.2-dev` 上为 `0.16.2.dev0`。

## 已验证组合

以下是已有文档记录的参考环境，不是完整测试矩阵。声明的上游范围（`>=0.14,<0.17`）只是版本守卫，不能证明所有版本、功能或框架均已通过。每次验证都应记录实际 revision、命令、通过/失败/跳过数量及未测路径。

Python 3.10 · PyTorch 2.7.1a0（MUSA 版）· torch_musa 2.7.1 · torchada 0.1.86 · Megatron-LM `core_v0.16.1`（megatron-core 0.16.1）· MT-TransformerEngine 2.0.0 · apex（MT fork，融合 RoPE）· MTT S5000 · tilelang_musa 0.1.12+musa.2.git1feb38c3 · torch-kernels 0.1.0 · musa_toolkits 4.3.7。

TileLang 栈（tilelang-musa 与 torch-kernels 的 tilelang 算子）与 torch_musa 构建及 musa_toolkits 版本互相绑定——已验证矩阵与升级规则见[补丁目录](docs/PATCHES_zh.md#tilelang-栈版本绑定)。

上述两个早期 TE Hook 是等待 Megatron 激活规则的限定例外；注册路径仍不导入加速器依赖。
设备层仅清理 Transformers 的 CUDA/BF16/FP16/TF32 与 FlashAttention 探测缓存，
并在 apply、失败回滚和 unapply 后执行，不清理其他包可用性缓存。
safe-utils 桥接覆盖普通导入；不支持在运行中 reload 厂商 TE 模块，
更换 TE 或补丁后应启动新进程。RoPE 分发还会识别 Core 原生 TE 包装的
TE < 2.3 interleaved 限制，告警后走 Megatron 的非融合实现。

### 声明式版本门控

报告包含每条补丁的 `version_gates` 及跳过原因；元数据缺失时继续目标/能力探测。
三个 TE norm 补丁声明 `transformer_engine >=2.0,<2.1`，这只表示适用范围，不证明
其他版本已修复。上表 override 开关只绕过版本门控。数字比较及源码/能力探针规则见
[贡献指南](CONTRIBUTING_zh.md#版本门控)。删除补丁前必须禁用它并重跑原始回归。
