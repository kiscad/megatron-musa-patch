# 2026-09-16 源码检查与验证

本次检查覆盖 `src/megatron_musa_patch/` 全部模块，重点为选择性启用、必要协作、注册顺序、
别名所有权、失败回滚和卸载重试。实现变更及仍存在的依赖见
[补丁独立性检查记录](PATCH_INDEPENDENCE.md)。

## 环境

- Python 3.10；PyTorch `2.7.1a0+gitunknown`；torch_musa `2.7.1`；torchada `0.1.86`。
- Megatron-Core wheel `0.16.1`；MT-TransformerEngine `2.0.0`；ms-swift `4.6.0.dev0`。
- Megatron-LM checkout revision：`55ac70825`。该 checkout 存在已有的 llama3 shell 示例修改，
  本次未使用该修改脚本作为原始脚本通过的证据，也未修改上游源码或测试。
- 本包任务前 HEAD：`a6050d3`，已保留为 `backup/pre-independence-review-20260916`。

下列命令在本包根目录执行，除非另有注明。`PYTHON` 指向同一 MUSA 虚拟环境解释器，
`MEGATRON_LM_PATH` 指向上述 checkout。

## 实际结果

| 检查 | 结果 | 证据范围 |
|---|---|---|
| 改动前默认本包测试 | 273 passed / 13 skipped | 未开启需显式选择的集成轮次；作为修复前基线。 |
| 修复后完整本包测试，开启 MUSA 集成 | **332 passed，无跳过** | 包含 norm-linear 单/双 rank、RoPE 前后向、双 rank 控制面通信及新增独立性/回滚测试。 |
| 原始 Megatron parallel-state 用例，双 rank | **每个 rank 2 passed** | 两种并行顺序；原始测试内容及断言未修改，通过自动激活运行。 |
| 双卡 5 步训练 smoke，启用 RoPE fusion | **退出码 0** | loss 有限，无 skipped/NaN iteration，完成 `torch_dist` checkpoint 保存，正常退出。 |
| wheel-only 自动激活/API 基础检查 | **core 路径、配置构造、CUDA 写法分配均成功** | core 来自 site-packages，`megatron.training` 不存在，分配结果为 `musa:0`。 |
| ms-swift 导入 | **失败，已复现外部栈问题** | TE 对 `torch.arange` 的变参包装不能被 TorchScript 编译，尚未进入实际训练。 |

完整测试：

```bash
MEGATRON_MUSA_RUN_INTEGRATION=1 MEGATRON_LM_PATH="$MEGATRON_LM_PATH" \
    "$PYTHON" -m pytest tests -q
```

原始上游测试（在 Megatron-LM 根目录）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=1 MEGATRON_MUSA_PATCH=1 \
    PYTHONPATH="$MEGATRON_LM_PATH" \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m pytest tests/unit_tests/test_parallel_state.py::test_tensor_model_parellel_world_size -q -ra
```

训练 smoke（`OUTPUT_DIR` 必须是可重建的专用临时目录）：

```bash
MEGATRON_LM_PATH="$MEGATRON_LM_PATH" PYTHON="$PYTHON" NPUS=2 ROPE_FUSION=1 \
    OUTPUT_DIR=/tmp/musa-independence-smoke-20260916 bash examples/run_pretrain_smoke.sh
```

本机日志位于 `/tmp/musa-independence-tests-final.log`、
`/tmp/musa-independence-upstream.log`、`/tmp/musa-independence-smoke.log`，未放入仓库。
本次 smoke 不保存 optimizer/RNG，不作为完整训练状态恢复或 async checkpoint 安全性的证据。

## 提交重组与中间状态

当前开发分支从根提交起重组为 12 个主题：引擎与激活、torchada 设备适配、架构与退出清理、
TE 接口、归一化、RoPE、训练策略、checkpoint writer、控制面通信、集成/独立性检查、示例、
文档。功能提交带对应测试，逐步更新 ledger；最终树保留原有项目内容及本次修复。

每个提交均在独立目录运行其当时已有的 CPU 测试（隐藏 MUSA 设备、关闭集成轮次），12 个
快照全部通过；完整快照结果为 **304 passed / 28 skipped**，跳过项为硬件或外部 checkout
相关检查。另行开启真实 MUSA 集成得到上表的 332 全通过结果。

重组过程的 commit/测试清单及逐次日志保存在本机
`/tmp/megatron-musa-repartition-n8pbg2hk/manifest.json` 和相邻 `tests-*.log`。
原始历史既有备份分支，也保存为同目录下经过 `git bundle verify` 的
`original-history.bundle`。远端引用及原有备份分支保留，未执行 push。

## ms-swift 阻塞的定位

不插入本包 import，直接 `import swift.megatron` 失败于厂商
`transformer_engine/musa/__init__.py` 中的 `patched_arange(*args, **kwargs)`。
ms-swift 的 `swift/sequence_parallel/zigzag_ring_attn.py` 编译 `get_half_lse` 时，TorchScript
不能编译该变参函数。只有 wheel 的 core 配置与设备通路在此前已成功。

在新进程关闭本包后，显式加载 torch、torch_musa、TE，再导入 ms-swift，可复现相同错误：

```bash
MEGATRON_MUSA_PATCH=0 "$PYTHON" -c \
    'import torch; import torch_musa; import transformer_engine.pytorch; import swift.megatron'
```

该对照证明此失败不需要本包补丁生效。原始错误及对照日志分别为
`/tmp/musa-independence-swift.log`、`/tmp/musa-independence-swift-baseline.log`。
本次没有修改厂商安装源码或关闭 TorchScript 来掩盖问题；后续需单独验证 TE 与 torchada
工厂函数适配的组合，再进行完整 ms-swift 训练验收。

## 未覆盖范围

未运行上游全量测试矩阵、长程模型收敛、多机训练、完整 optimizer/RNG 保存恢复或 ms-swift
端到端训练。原始单测、wheel 基础通路和短训练通过，只证明表中列出的路径。
