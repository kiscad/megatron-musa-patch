# 补丁目录

本页描述本包可以安装的每一个补丁：目标是什么、为什么存在、运行上需要注意什么。
实时、机器可读的来源（`id`、`rationale`、`strategy`、`upstream`、`remove_when`、
`requires`、`version_gates`）是 `megatron_musa_patch.report()`；本页是它的可读版本，
允许比代码滞后一次改动。

## 目录

| id | 目标 | 根因（简述） |
|---|---|---|
| `python.typing.override.backport` | `typing.override`（Megatron hook） | Megatron 0.19 要求 Python ≥3.12 并在模块层 `from typing import override`；MUSA 轮子是 CPython 3.10 构建，`megatron.training` 直接 import 失败。改为发布 `typing_extensions` 的实现（PEP 698 只是标记，无运行时语义）|
| `torch.cuda.compat-layer` | `torch.cuda` | MUSA 版没有 CUDA 绑定；torchada + 4 个覆盖：`is_available()` 实时探测、`Tensor.type()` CUDA 名字、`CUDAGraph` 别名、tensor 子类（如 TE `Float8Tensor`）的 `.musa()` 迁移修复 |
| `torch.cuda.device-capability.nvidia-scale` | `torch.cuda.get_device_capability` | Megatron 拿 capability 和 NVIDIA 阈值比较（≥8 的 grouped-GEMM 门）；上报合成的 `8.3` |
| `megatron.training.get-device-arch-version.nvidia-scale` | `megatron.training.utils:get_device_arch_version` | 同类比较，走 `device_properties().major` |
| `torch.distributed.clean-teardown` | `atexit` | 带着存活的 MCCL 进程组退出的运行出现过 watchdog 报错；退出时尽力 `destroy_process_group()` |
| `megatron.fusions.fused-layer-norm.pure-torch` | `fused_layer_norm:FusedLayerNorm` | apex 能 import 但其 CUDA 扩展不可用；按 config 权威选择的纯 PyTorch LayerNorm/RMSNorm 回退 |
| `megatron.fusions.fused-layer-norm.have-apex-flag` | `fused_layer_norm:HAVE_FUSED_LAYER_NORM` | 消费方按该 flag 分支；类被替换后保持其语义为真 |
| `megatron.fusions.persist-layer-norm.disable` | `fused_layer_norm:HAVE_PERSIST_LAYER_NORM` | 回退实现不会执行 apex 的 persistent kernel |
| `megatron.transformer-block.layer-norm.impl-local` | `transformer_block:LayerNormImpl` | 受影响的 MUSA 栈上 TE 的 norm 算子在 `allocateSpace` 处 abort，即使是 `local` spec |
| `transformer_engine.layer-norm-linear.native-unfused` | `transformer_engine:LayerNormLinear` | 直接 TE 模型（megatron-FSDP 套件中的 te.pytorch.TransformerLayer）经 module/_common.py:apply_normalization 归一化，在 MUSA allocateSpace 断言中止。子类化 TE 模块，仅对符合条件的普通路径（无 FP8、无 UB overlap、无 offload、tp_size 1、不返回 layernorm 输出）改走函数式 norm + F.linear，保留参数、isinstance 与 checkpoint 契约。 |
| `transformer_engine.layer-norm-mlp.native-unfused` | `transformer_engine:LayerNormMLP` | MLP block 同一融合 norm 故障；符合条件的普通路径为函数式 norm -> fc1 -> gelu(tanh)/relu -> fc2，全部走普通 PyTorch 算子。门控激活与非普通配置保留原始 forward。 |
| `megatron.te.layer-norm-linear.unfused` | `transformer_engine:TELayerNormColumnParallelLinear` | 仅对 LayerNorm 用 PyTorch norm + TE Linear 绕过 QKV/FC1 内融合 norm 的 `allocateSpace` 错误，保留 FP8 Linear 和 norm checkpoint 参数名。norm 步骤遵循 TE 的 op 契约——以 autocast dtype 或 norm 权重 dtype 计算并把输入 cast 到该 dtype，和原生内核一样吸收混合 dtype 激活（如 bf16 模型中 fp32 学习位置嵌入之和） |
| `megatron.te.cpu-offload-context.signature-dispatch` | `transformer_engine:get_cpu_offload_context` | 每次构建 `TransformerBlock` 都会调用 TE 的 CPU-offload helper；当 fork 报告的版本号与真实签名不一致时会选中六参数（TE ≥ 2.5）调用而 fork 只接受五参数，模型还没建好就报错；改为按已安装函数自身的签名分发 |
| `megatron.fsdp.premul-sum.device-prescale` | `fsdp...param_and_grad_buffer:gradient_reduce_preprocessing` | torch_musa/MCCL 没有实现 PREMUL_SUM；FSDP 梯度平均在该分支设备端 `mul_` 预缩放后改用 SUM，其余分支透传 |
| `megatron.bridge-communicator.subgroups-backend` | `pipeline_parallel.bridge_communicator:dist` | `new_subgroups_by_enumeration` 在 c10d 内部调用 `new_group`，绕开 torchada 的 nccl→mccl 翻译；在模块局部代理中仅翻译确切的 `nccl` 请求 |
| `megatron.hyper-comm-grid.subgroups-backend` | `hyper_comm_grid:dist` | 与 bridge communicator 相同的绕行路径；同一代理，仅翻译 `nccl` |
| `megatron.te.grouped-linear.mem-monitor-compat` | `transformer_engine...grouped_linear.py` 的 `musa_patch` 导入 | MT-TE 硬依赖旧 `musa_patch.mem_utils.MemMonitor`；提供带 `max_token_num` 计数的最小 shim（拥有 sys.modules 条目，可撤销），不覆盖已有包 |
| `megatron.moe.topk.fp64-reference` | `transformer.moe.moe_utils:torch` | MuDNN TopK 不支持 float64；moe_utils 的 torch 全局换成转发代理，仅对 FP64 topk 用 CPU 同精度求索引 + 设备端 gather |
| `megatron.moe.permutation.unfused-musa` | `transformer.moe.moe_utils:permute` | TE 的 moe_permute 内核对 FP32/FP64 中止；仅在该组合降级到上游 `fused=False` 参考实现 |
| `megatron.moe.unpermutation.unfused-musa` | `transformer.moe.moe_utils:unpermute` | 与 permute 成对降级，保证两端索引格式一致（requires 声明） |
| `megatron.moe.grouped-gemm.torch-ops` | `transformer.moe.grouped_gemm_util:ops` | fanshiqing grouped_gemm 无 MUSA 构建；提供每专家 `torch.matmul` 的参考 `ops.gmm`（含 trans_b、空专家梯度），vendor 存在时拒绝 |
| `megatron.moe.grouped-gemm.available-flag` | `grouped_gemm_util:grouped_gemm_is_available` | 回退安装后如实报告可用（requires torch-ops） |
| `megatron.moe.grouped-gemm.assert-noop` | `grouped_gemm_util:assert_grouped_gemm_is_available` | GroupedMLP 构造断言改为查询被修补的可用性标志 |
| `megatron.softmax.kernel-availability.musa` | `fused_softmax:FusedScaleMaskSoftmax.is_kernel_available` | 探测内部 import `scaled_masked_softmax_cuda` 会直接 ModuleNotFoundError；扩展缺失时返回 False 走 torch 回退 |
| `megatron.te.norm.unfused-musa` | `transformer_engine:TENorm` | TE 独立 LayerNorm/RMSNorm 在 MUSA `allocateSpace` 中止；子类化 TE 模块仅替换 forward（functional norm，参数 cast 到输入 dtype），isinstance/分片契约保留 |
| `transformer_engine.dot-product-attention.capability-dispatch` | `transformer_engine:DotProductAttention.forward`（原地） | 直接 TE 模型（te.pytorch.TransformerLayer，56 条 mfsdp 用例）的 attention 走 TE 原生类，MUSA 移植将其固定为 flash：fp32 激活模型死于 'FlashAttention only supports FP16 and BF16'。forward 在原类上原地包裹（MT-TE 自身 __init__ 经模块全局解析 super()，替换类会破坏构造）；符合条件的 flash 不支持稠密调用改走 TE UnfusedDotProductAttention，mask 语义与 Megatron wrapper 分发一致。 |
| `megatron.te.attention.capability-dispatch` | `transformer_engine:TEDotProductAttention.forward` | 按内核能力分发、不实现 attention 数学。原生 MT-TE flash 承接 FP16/BF16、head_dim 64–192、无 dropout 的输入，且当调用可能进入反向（训练、checkpoint 重算、开启梯度且输入需要梯度）时仅限实测 MuDNN 反向安全窗口 {64,80,96,112,128,160} 内的相等 qk/v 维度；其反向内核对 144、168–192 以及实测过的混合 qk/v 维度（MLA 的 192/128）均失败。反向不支持的形状在安装了 mate 且落在其已验证包络内时走 mate 的 TileLang flash（`mate.flash_attn_varlen_func`），否则走 TE 自带的 UnfusedDotProductAttention 后端；非 padding 类 mask 丢弃 mask 张量，与被替换的 flash 路径行为一致。padding THD 切成逐序列厂商调用（原生 THD 丢弃 `cu_seqlens` 会得到 NaN）。`MEGATRON_MUSA_PATCH_ATTN_BACKEND=auto|mudnn|mate|unfused` 选择 kernel 顺序；mate 的 TileLang kernel 按 head_dim 类型首次使用时 JIT 编译（多 rank 运行前预热 ~/.tilelang）。TE 硬编码 flash 标记消失时整体放弃（源码探针）。CP>1/FP8 DPA/特殊 softmax/window/max-logit 仍交上游。 |
| `megatron.te.quantized-model-init.delayed-compat` | `transformer_engine.pytorch.quantized_model_init` | 等待 Megatron 激活的 Hook 管理 TE 缺失属性；仅将显式 DelayedScaling 委托 fp8_model_init，保留高精度初始化和嵌套上下文。其他启用的 recipe 明确拒绝，不覆盖原生属性。 |
| `megatron.te.factory-shim.torchscript-compat` | `transformer_engine:musa/patch_after_import_torch` | MT-TE 将 torch 工厂函数（tensor/zeros/ones/empty/rand/arange/empty_like）重绑为无类型设备翻译包装，任何覆盖工厂调用的 eager `torch.jit.script` 都会编译失败；现在每次脚本化编译时向 TorchScript 的 ATen builtin 表临时登记七个已知厂商包装，编译期间也不重绑 eager 工厂， `device='cuda'` 翻译保持不变。仅对 MUSA TE fork 启用，需在 Megatron 导入前生效。编译图保持 ATen 设备语义，不翻译图内 CUDA 字符串；TE 无工厂包装时自动跳过（源码探针）。撤销保留第三方后续替换。 |
| `megatron.te.utils-module.safe-seed` | `transformer_engine:musa/pytorch/utils` | 厂商模块在导入期裸迭代 sys.modules 并对 lazy 模块做 getattr，transformers 5.x 下会令整个 TE 导入链崩溃（'dictionary changed size during iteration'）；在其执行前种入去掉该危险循环的等价模块（与 musa_patch.mem_utils shim 同一模式）。厂商修复该循环后自动跳过（源码探针）。 |
| `megatron.embeddings.fused-rope.apex` | `rope_utils:fused_apply_rotary_pos_emb` | Megatron 的融合 `sbhd` kernel 依赖 MUSA TE 树中并不存在的 `…attention.rope` 导入，导致 argparse 默认的 `apply_rope_fusion` 在第一步之前就被拒绝；改为绑定 apex 的等价 kernel |
| `megatron.embeddings.fused-rope-thd.apex` | `rope_utils:fused_apply_rotary_pos_emb_thd` | 同一处缺失导入的 packed（`thd`）部分；使用 apex 的 padded 布局 kernel，`cp_size=1` |
| `megatron.embeddings.rope-fusion.unfused-fallback` | `rope_utils:apply_rotary_pos_emb` | apex 没有 interleaved 和 context parallel 变体；仅把这类调用降级到上游的非融合分支（告警一次），而不是在训练中途报错 |
| `megatron.ssm.gated-delta-rule.tilelang` | `ssm.gated_delta_net:chunk_gated_delta_rule` | GatedDeltaNet 通过 flash-linear-attention 的 Triton kernel 计算 chunked gated delta rule——这是 Qwen3.5 这类混合层模型在 MUSA 上最慢的部分；落在 torch-kernels 已审计包络内的调用改用其 TileLang 实现，其余调用原样回退 FLA |
| `mcore_bridge.ssm.gated-delta-rule.tilelang` | `mcore_bridge.model.modules.gated_delta_net:chunk_gated_delta_rule` | ms-swift 实际执行的 mcore-bridge forward 把 fla 的符号导入到了自己的命名空间，只 patch Megatron 的绑定触及不到它；同一分发器，独立可选 |
| `megatron.legacy.fused-kernels.load.noop` | `legacy.fused_kernels:load` | 该 loader 探测 `nvcc`/定义扩展构建入口——没有 MUSA 构建路径 |
| `megatron.training.set-jit-fusion-options.noop`（含 `initialize` 别名） | `set_jit_fusion_options` | 启动期 CUDA 预热/编译路径需先在 MUSA 上验证；验证前跳过 |
| `megatron.dist-ckpt.musa-cpu-staging` | DCP filesystem 设备选择器（Megatron hook） | CUDA 兼容层下按实际 MUSA stream 选择设备，避免跳过 CPU staging；保留上游异步拷贝、同步和 checkpoint 分片 |
| `megatron.training.overlap-flags.noop` | `training.arguments:validate_args` | 观测到的 TE fused-wgrad/DDP 组合会让 `param.grad` 为 None，破坏 Megatron overlap 反向 hook；DP-overlap 开关被强制关闭并告警 |
| `megatron.training.profile.pytorch` | `training.arguments:validate_args` | 裸 `--profile` 会走 `cudaProfilerStart/Stop`/NVTX，MUSA 运行时不提供；改为自动启用 `--use-pytorch-profiler` |
| `megatron.training.start-time.integer-microseconds` | `training:torch` | 仅在 MCCL 上将启动时间 MIN 归约转换为整数微秒 |
| `megatron.training.checkpoint.host-barrier-proxy` / `host-barrier-context` | `checkpointing:torch` / `save_checkpoint` | checkpoint I/O 前创建全局 Gloo 组，仅将保存屏障切换到该组 |
| `megatron.serialization.signal-member-globals` | `safe_globals:SAFE_GLOBALS` | 注册信号成员别名，支持 Python 3.10 weights-only 参数加载 |

每个 patch 记录 `rationale`（观测到的根因，限定于具体版本，不夸大未验证的结论）、`strategy`（替换实现做了什么、保留了什么）、对应的上游文件，以及绑定到具体测试的 `remove_when` 复审条件。`uninstall()`/`unapply()` 会还原本包拥有的改动；有外部副作用的 hook（如 torchada import 期的全局变更）会如实标注为不可逆。

**已退役：** `megatron.training.ckpt-format.no-torch-dist`（静默把 `--ckpt-format torch_dist` 改写为 `torch`）已移除。该改写改变了保存/加载语义——`torch` 是 Megatron 的 legacy 格式，不是 `torch.distributed.checkpoint` 的等价后端，改写还会破坏异步保存和 FSDP 配置。如需 legacy writer，请显式传 `--ckpt-format torch`。

## Kernel 栈补丁

这些补丁是为 MUSA 选择 kernel，而不是修复坏掉的调用。实测数据、预热要求和版本
绑定记录在本页，README 只保留概览。

### Chunked gated delta rule → torch-kernels TileLang

**补丁：** `megatron.ssm.gated-delta-rule.tilelang`、
`mcore_bridge.ssm.gated-delta-rule.tilelang` · **开关：**
`MEGATRON_MUSA_PATCH_GDN_TILELANG`（默认 `1`）

Qwen3.5 / Qwen3-Next 这类混合模型的 GDN 层计算 chunked gated delta rule。
flash-linear-attention 的 Triton kernel 仍是正确性参照和回退路径；落在
torch-kernels 已审计 TileLang 包络内的调用（MUSA 设备、bf16 q/k/v/beta、
fp32 衰减、算子自带的 `gdn_dense_supported` / `gdn_varlen_supported` shape
守卫、无 fla 专属参数）改走 torch-kernels 的 TileLang 实现。分发按调用逐次
依据张量元数据决定（packed 调用会读取一次 `cu_seqlens` 校验每段至少两个
chunk 的下限），不缓存判定。两个绑定都要 patch：ms-swift 执行的
mcore-bridge forward 把 fla 的符号导入到了自己的命名空间。

MTT S5000 实测（B=1、DK=DV=128、l2 归一化 q/k，预热 3 次后 20 次取中位数，
fwd+bwd）：

| shape | fla 0.5.2 | torch-kernels TileLang | 加速比 |
|---|---|---|---|
| S=4096, H=8 | 5.16 ms | 1.07 ms | 4.8x |
| S=4096, H=32 | 13.92 ms | 2.31 ms | 6.0x |

数值：fla 与 TileLang 各自对 Megatron fp32 参考实现的 relL2 约 4e-3
（fla 3.9e-3，TileLang 3.6e-3）；两者梯度（dq/dk/dv/dg/dβ）relL2 ≤ 6.6e-3。

8 卡 ms-swift Qwen3.5 端到端训练（30 次迭代、长样本、相同数据与种子，
第 3 次迭代之后的稳态均值）：

| 并行布局 | fla | TileLang | 步时长 |
|---|---|---|---|
| TP=4 / PP=1 | 4.93 s/it | 4.67 s/it | 低 5.3% |
| TP=1 / PP=4（DP=2） | 2.74 s/it | 2.48 s/it | 低 9.5% |
| TP=1 / PP=2（DP=4） | 1.89 s/it | 1.85 s/it | 低 2.0% |

运行注意事项：

* TileLang kernel 首次使用时按头数和 dense/unpadded 特化 JIT 编译（需数
  分钟，缓存在 `~/.tilelang`）。**多 rank 运行前请先用
  `examples/warm_gdn_tilelang.py --heads <单卡头数>` 单进程预热一次**——
  所有 rank 同时向共享缓存首次编译同样的 kernel 曾导致运行崩溃（设备错误
  / SIGABRT）。缓存预热后，完整的 8 卡训练全程正常。
* FLA 自己在新头数/shape 桶上的首次调用同样要编译（H=32 首步实测约 13
  分钟 Triton 编译）；TileLang 的预热把这笔开销移出了训练过程。
* `MEGATRON_MUSA_PATCH_GDN_TILELANG=0` 同时拒绝两个补丁，全部恢复 FLA。

#### TileLang 栈版本绑定

当前环境中的 TileLang 栈是一套整体匹配的组合：**tilelang-musa 与 tilelang
算子包（torch-kernels）是针对特定的 torch_musa 和 musa_toolkits 版本构建的，
其版本与 MUSA 栈互相绑定。**已验证组合：

| 组件 | 已验证版本 |
|---|---|
| PyTorch | 2.7.1a0（MUSA 版） |
| torch_musa | 2.7.1 |
| musa_toolkits | 4.3.7 |
| tilelang_musa | 0.1.12+musa.2.git1feb38c3 |
| torch-kernels（tilelang 算子包） | 0.1.0 |

* 升级 torch_musa 或 musa_toolkits 时，必须连同匹配的 tilelang-musa /
  torch-kernels 构建一起升级；不得混用针对其他 MUSA 栈编译的 tilelang-musa。
* `~/.tilelang` 下的 JIT kernel 缓存按 tilelang 构建 id 命名（如
  `0.1.12_musa_2_git1feb38c3-x86_64`）。栈发生任何变化后，请删除过期缓存
  并重新预热，让 kernel 按新栈重新编译。
