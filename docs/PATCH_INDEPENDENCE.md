# 补丁独立性检查记录

独立性是指可明确选择、失败不留下未记账变更、卸载不越过所有权边界，以及尽量不依赖
其他补丁的安装顺序。它不表示 Megatron 在缺少 MUSA 设备适配或必需 kernel 的情况下仍能运行。

## 逐模块边界

| 模块 | 补丁职责 | 独立性与必要协作 |
|---|---|---|
| `_torch_backend.py` / `backends/torch_cuda.py` | CUDA 名称的 MUSA 设备契约 | 平台前置，先于使用 CUDA API 的 Megatron 代码。torchada 的全局副作用不可撤销，本包仅撤销自己拥有的覆盖。 |
| `_device_arch.py` | capability 与训练架构号 | 两条补丁分别可选，只共享同一环境配置的解析函数。capability hook 必须位于设备适配之后，才会修改最终 CUDA proxy。合成数值不代表硬件支持相应 NVIDIA kernel。 |
| `_distributed.py` | 正常退出时的进程组清理 | 可单独安装/卸载自己的 atexit 回调；不要求其他 patch，卸载不销毁运行中的进程组。 |
| `_transformer_engine.py` | TE 版本策略、CPU offload 签名 | 两条分别可选。签名适配读取真实函数，不要求版本补丁已生效；版本策略仍较宽泛，不能替代具体 TE 功能验证。 |
| `_layer_norm.py` | standalone norm、flags、block norm、TE norm-linear | block norm 可自行构造 PyTorch 回退；TE norm-linear 不依赖 standalone/block 补丁；两个 flags 必须在 local class 补丁成功后生效。 |
| `_rope.py` | sbhd/thd kernel、dispatcher 回退 | 三条分别可选，dispatcher 在调用时识别当前绑定的 apex kernel，不依赖注册顺序。不修改共享模型配置；原生 TE kernel 不触发 apex 探测。单独 kernel 不提供 dispatcher 的 interleaved/CP 回退。 |
| `_training.py` | profile、DP overlap、legacy loader、两处 JIT helper | 每条可单独选择；profile/overlap 共用目标但处理不同字段，正反注册顺序均验证。JIT 定义处与已导入的别名分别拥有绑定；正常 Python 导入仍会继承定义处的当前值。 |
| `_checkpointing.py` | checkpoint bucket 串行写入 | 独立于控制面屏障，保留上游结果协议和格式。外层 async caller 的 fork 不在该补丁覆盖内。 |
| `_control_collectives.py` | 启动时间同步、checkpoint host barrier、signal aliases | 启动时间和 signal 别名独立。host barrier 必须由 proxy/context 协作，context 显式依赖 proxy；只装 proxy 不重定向 barrier，也不创建组。 |

补丁模块不互相 import；共享机制放在 engine/backend 层。共同的平台前提、同目标 wrapper
链、上述三条显式前置关系不应被描述为完全没有依赖。

## 显式前置关系

| 消费者 id | `requires` |
|---|---|
| `megatron.fusions.fused-layer-norm.have-apex-flag` | `megatron.fusions.fused-layer-norm.pure-torch` |
| `megatron.fusions.persist-layer-norm.disable` | `megatron.fusions.fused-layer-norm.pure-torch` |
| `megatron.training.checkpoint.host-barrier-context` | `megatron.training.checkpoint.host-barrier-proxy` |

`requires` 仅支持同一模块、不同属性的 `AttrPatch`。引擎按目标依赖排序，保持同目标工厂的
注册顺序。缺失、禁用或主动退出的前置不会自动启用；消费者的 `report()` 标为 `skipped`，
`detail` 给出缺少的 id。循环、同目标及跨模块前置在注册阶段拒绝，避免引入隐式导入图。
晚注册的前置可以使原先跳过的消费者重新评估；用户运行中改变过滤开关仍需卸载后重装。

## 本次修复

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

CPU stub 的独立性不等同于每种补丁子集都能跑完整训练。TE 版本策略、同步 DP 回退、合成
capability、异步 checkpoint 外层 fork 仍有文档所述限制。只安装 wheel 的环境、完整上游
测试集合和 ms-swift 全训练组合需要各自的实测证据，不能由本地回归或 smoke 外推。
