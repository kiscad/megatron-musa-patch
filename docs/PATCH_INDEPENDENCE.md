# 补丁独立性检查记录

独立性是指可明确选择、失败不留下未记账变更、卸载不越过所有权边界，以及尽量不依赖
其他补丁的安装顺序。它不表示 Megatron 在缺少 MUSA 设备适配或必需 kernel 的情况下仍能运行。

## 逐模块边界

| 模块 | 补丁职责 | 独立性与必要协作 |
|---|---|---|
| `_python_compat.py` | 解释器层 backport（`typing.override`、`typing.Concatenate`） | 只拥有自己发布的 `typing` 绑定，不依赖任何其他补丁；已存在该属性时主动放弃。必须先于 Megatron 模块执行，否则上游 import 直接失败。 |
| `_torch_backend.py` / `backends/torch_cuda.py` | CUDA 名称的 MUSA 设备契约 | 平台前置，先于使用 CUDA API 的 Megatron 代码。torchada 的全局副作用不可撤销，本包仅撤销自己拥有的覆盖。 |
| `_device_arch.py` | capability 与训练架构号 | 两条补丁分别可选，只共享同一环境配置的解析函数。capability hook 必须位于设备适配之后，才会修改最终 CUDA proxy。合成数值不代表硬件支持相应 NVIDIA kernel。 |
| `_distributed.py` | 退出清理、FSDP 预缩放、subgroup backend 翻译 | 各自可选；卸载 atexit 回调不销毁运行中的进程组，通信代理只修改对应模块。 |
| `_transformer_engine.py` | TE 签名适配、导入桥接、量化初始化 | 各自拥有绑定及 undo；保留上游版本谓词。两个早期 Hook 仅适用 MUSA TE；mem-monitor shim 不覆盖已安装或已导入的包。 |
| `_attention.py` | 按能力选择 TE attention 后端 | 独立源码/输入探测；复用 TE 数学实现，不依赖 norm 或 MoE 补丁。 |
| `_moe.py` | FP64 top-k、permute/unpermute 回退 | top-k 独立；unpermute 显式要求 permute。成对使用两项 permutation 补丁；单独启用 permute 不保证完整 dispatch/restore 链路。 |
| `_grouped_gemm.py` | 逐专家 matmul 回退及可用性接口 | flag/assert 要求 ops；完整调用方还需要 availability flag，不能由只安装 ops 的结果推断整个 GroupedMLP 可用。 |
| `_sort.py` | bool 键排序（四种拼写） | 单一 hook 原子持有四个绑定并整体 undo，缺的是同一个能力；只认 bool+MUSA 输入，其它 dtype/设备零介入，不依赖任何其它补丁。 |
| `_softmax.py` | 扩展缺失时的融合可用性判断 | 独立选择，数学实现留给上游回退。 |
| `_layer_norm.py` | standalone norm、flags、block norm、TE norm-linear | block norm 可自行构造 PyTorch 回退；TE norm-linear 不依赖 standalone/block 补丁；两个 flags 必须在 local class 补丁成功后生效。 |
| `_rope.py` | sbhd/thd kernel、dispatcher 回退 | 三条分别可选，dispatcher 在调用时识别当前绑定的 apex kernel，不依赖注册顺序。不修改共享模型配置；原生 TE kernel 不触发 apex 探测。单独 kernel 不提供 dispatcher 的 interleaved/CP 回退。 |
| `_training.py` | profile、DP overlap、legacy loader、两处 JIT helper | 每条可单独选择；profile/overlap 共用目标但处理不同字段，正反注册顺序均验证。JIT 定义处与已导入的别名分别拥有绑定；正常 Python 导入仍会继承定义处的当前值。 |
| `_checkpointing.py` | DCP staging 的设备选择 | 只拥有 DCP 的设备选择器，独立于控制面屏障；不改上游 writer、结果协议和 checkpoint 格式。 |
| `_control_collectives.py` | 启动时间同步、checkpoint host barrier、signal aliases | 启动时间和 signal 别名独立。host barrier 必须由 proxy/context 协作，context 显式依赖 proxy；只装 proxy 不重定向 barrier，也不创建组。 |

补丁模块不互相 import；共享机制放在 engine/backend 层。共同的平台前提、同目标 wrapper
链、下表显式前置关系不应被描述为完全没有依赖。

## 显式前置关系

| 消费者 id | `requires` |
|---|---|
| `megatron.fusions.fused-layer-norm.have-apex-flag` | `megatron.fusions.fused-layer-norm.pure-torch` |
| `megatron.fusions.persist-layer-norm.disable` | `megatron.fusions.fused-layer-norm.pure-torch` |
| `megatron.training.checkpoint.host-barrier-context` | `megatron.training.checkpoint.host-barrier-proxy` |
| `megatron.moe.unpermutation.unfused-musa` | `megatron.moe.permutation.unfused-musa` |
| `megatron.moe.grouped-gemm.available-flag` | `megatron.moe.grouped-gemm.torch-ops` |
| `megatron.moe.grouped-gemm.assert-noop` | `megatron.moe.grouped-gemm.torch-ops` |

`requires` 仅支持同一模块、不同属性的 `AttrPatch`。引擎按目标依赖排序，保持同目标工厂的
注册顺序。缺失、禁用或主动退出的前置不会自动启用；消费者的 `report()` 标为 `skipped`，
`detail` 给出缺少的 id。循环、同目标及跨模块前置在注册阶段拒绝，避免引入隐式导入图。
晚注册的前置可以使原先跳过的消费者重新评估；用户运行中改变过滤开关仍需卸载后重装。

## 生命周期保证

- reload/追加注册重建时，初次应用后才导入的别名也跟随当前绑定；缺少前置且无旧绑定时不解析目标符号。
- 引擎提交属性后若别名更新失败，现在回滚目标和本次已更新的旧别名，保留此前成功应用
  的版本。失败卸载的属性保留清理记录，可重试；其他补丁仍能继续清理。卸载别名扫描排除
  独立声明的目标，避免越过其生命周期。
- norm flags 的排序不再靠 tuple 中“恰好 class 在前”。block norm 在 class patch 关闭时
  不会退回已知不工作的 apex class。
- checkpoint context 不再在 proxy 缺失时虚报 `applied`。
- RoPE dispatcher 不再在安装时固化 kernel 状态，也不在调用期间临时修改共享
  `config.apply_rope_fusion`；packed interleaved 同样回退到上游非融合路径。
- 新增顺序排列、ONLY/DISABLE、动态 kernel 替换、失败提交/卸载重试及 ledger 依赖检查。

## 验证与限制

主要回归入口为 `tests/test_engine_dependencies.py`、`tests/test_engine_lifecycle.py`、
`tests/test_patch_independence.py`，以及每个模块已有契约测试。真实 MUSA 检查使用原有
norm-linear、RoPE、双 rank 控制面通信和训练 smoke；执行方式见 [AGENTS.md](../AGENTS.md)。

CPU stub 的独立性不等同于每种补丁子集都能跑完整训练。TE 版本/源码门控、同步 DP 回退、合成
capability、异步 checkpoint 外层 fork 仍有文档所述限制。只安装 wheel 的环境、完整上游
测试集合和 ms-swift 全训练组合需要各自的实测证据，不能由本地回归或 smoke 外推。
