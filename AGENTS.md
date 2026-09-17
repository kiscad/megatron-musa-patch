# megatron-musa-patch：Agent 开发操作手册

本文是本目录及子目录的 coding agent 工作约定。开始任务前阅读本文、[中文 README](README_zh.md)
和[贡献指南](CONTRIBUTING_zh.md)；实现细节以当前源码为准。适用的上级约定和用户明确指令优先。
本文件统一维护项目开发约定、操作流程和验收要求。

## 1. 目标与完成标准

本项目让 **Megatron-LM / Megatron-Core 在 MUSA 上无缝运行**。适配责任在本包内部，验收面包括：

| 验收面 | 必须保持的行为 |
|---|---|
| 上游单元测试 | 直接运行原始测试、fixture 和断言，保留 CUDA API、`nccl` 等原有写法。 |
| 上游训练脚本 | 原始 Python/shell 入口无需修改，不依赖本仓库专用启动器才能运行。 |
| ms-swift 等上层框架 | 原有 `megatron.core` import、配置、模型构造和调用方式不变，无需分辨 MUSA/CUDA，也无需主动注册本包补丁。 |

环境需要匹配的 MUSA PyTorch、torch_musa、torchada 及任务实际使用的 TE/apex 等依赖。
允许通过原有参数入口指定数据、输出、卡数和适合硬件资源的规模。要求调用方改设备字符串、
关闭原有功能、插入 patch import 或改成 MUSA 专用模型类，仍是兼容缺口。

“先跑起来”是实施阶段，不是缩小目标。当前的 norm、RoPE、DP overlap、checkpoint 等回退
必须注明触发条件、性能代价、限制及移除证据。仅 `import` 成功、本仓库测试通过、改参数后的
smoke 成功，均不能单独证明上述三类验收完成。不要把未测功能写成已支持。

## 2. 不得破坏的约束

1. **改动收敛在本仓库。** 不以修改 Megatron-LM、ms-swift、torch、TE 或 apex 的安装源码
   作为交付方案；其他仓库可读作定位依据。不得改上游测试预期、放宽容差或添加 MUSA
   skip/xfail 来消除失败。环境资源缺失可以记录，但不能计为兼容通过。
2. **保留调用契约。** 检查位置/关键字参数、默认值、返回类型、shape/dtype/device、梯度、
   RNG、异常、分布式同步和 checkpoint key/sharding。数值回退需要参考实现和合理容差；
   禁止静默改 checkpoint 格式、模型结构、精度、优化器或训练计划。
3. **不复制整份上游文件。** 优先复用上游实现或包裹单个符号。必须改写单个函数时，记录
   原始位置、必要差异和升级检查点；不要建设与上游平行维护的模型实现。
4. **保持惰性激活。** 注册路径保持标准库可用；`patches/` 顶层不导入 torch、Megatron、
   TE 或 torchada。torchada import 有全局副作用，不用它探测是否安装。真实设备适配等待
   Megatron；显式 `apply()` 是诊断入口。自动 entry point 的异常边界不能破坏 `import torch`。
5. **使用现有引擎和 ledger。** 不另建 `.pth`、`sitecustomize`、源码重写或遍历全部模块的
   patch 系统。不要直接把 `sys.modules["torch.cuda"]` 指向 `torch.musa`。通用翻译交给
   torchada；本包增加有根因和测试支撑的 Megatron 契约补偿。
6. **生命周期可检查。** 重复 install/apply 不叠加 wrapper；reload 后行为稳定；unapply
   只撤销本包拥有的绑定，不覆盖第三方后来写入的对象。Hook 失败清理自己的部分变更，拥有
   状态时提供 `undo`。torchada 等依赖的不可逆副作用要明确，彻底隔离用新进程。
7. **不把 core 修复藏在训练入口。** wheel 用户没有 `megatron.training` / `megatron.legacy`。
   若问题影响 `megatron.core` 调用，补丁必须在 core 路径独立生效，不能依赖训练参数校验。
8. **独立选择，显式协作。** 不依赖兄弟 patch 模块的私有状态或默认注册顺序。必须协作的
   同模块不同目标使用 `AttrPatch.requires` 声明；前置未生效时跳过并报告原因，不能隐式
   开启用户禁用的补丁。覆盖单独选择、逆序注册、失败回滚和卸载。设备适配属于运行环境
   前提；同一符号上的 wrapper 链顺序仍是有意设计。完整边界见
   [补丁独立性检查记录](docs/PATCH_INDEPENDENCE.md)。

## 3. 从哪里读、在哪里改

| 位置 | 职责与修改时机 |
|---|---|
| `pyproject.toml` | 依赖、唯一包版本来源、`torch.backends` entry point、pytest 配置。 |
| `src/megatron_musa_patch/activation.py`、`__init__.py` | 自动/显式/立即激活及公开 API；导入时序问题先看这里。 |
| `_engine.py` | import watcher、属性链、别名、所有权、回滚、report；仅修改通用机制。 |
| `_compat.py`、`_errors.py` | 版本与目标解析、异常上下文；版本范围同时检查声明和判断函数。 |
| `_env.py` | 环境开关集中说明与惰性读取；不要把环境变量值缓存到模块常量。 |
| `backends/torch_cuda.py` | torchada 之上的设备契约补偿，不实现第二套通用适配层。 |
| `patches/_*.py`、`patches/__init__.py` | 具体补丁与 `MODULES` 注册；同目标的先后顺序有意义。 |
| `tests/test_*.py` | 引擎、补丁契约与集成回归；硬件 worker 为 `*_smoke.py`。 |
| `examples/` | 复现、诊断和演示入口；不承担原始调用方必需的适配逻辑。 |

设备 API → `backends/torch_cuda.py` / `_torch_backend.py`；TE 调用 → `_transformer_engine.py`；
归一化 → `_layer_norm.py`；RoPE → `_rope.py`；训练参数与 profiler → `_training.py`；
checkpoint writer → `_checkpointing.py`；控制面通信 → `_control_collectives.py`；
退出清理 → `_distributed.py`。优先在对应模块中实现最小修复。

## 4. 开工：确认边界并取得可复现基线

以下命令使用 Bash。先替换路径；使用真实 MUSA 环境的同一个解释器，不假定父目录名、
系统 `python` 或 PATH 上的 `torchrun` 正确。不要为运行测试随意升级厂商 torch/TE 栈。

```bash
cd /path/to/megatron-musa-patch
export MMP_ROOT="$PWD"
export MEGATRON_LM_PATH=/path/to/Megatron-LM
export MMP_ARTIFACTS="$(mktemp -d /tmp/megatron-musa-validation.XXXXXX)"

git status --short
git diff --stat
git diff --cached --stat
git -C "$MEGATRON_LM_PATH" status --short
git -C "$MEGATRON_LM_PATH" rev-parse HEAD
"$MMP_PYTHON" -m pip show megatron-musa-patch megatron-core torch torch_musa torchada transformer-engine
```

- 阅读已有 staged/unstaged/untracked 改动，保留用户工作，不做 reset/checkout 覆盖。
- 上游工作区不干净时记录基线，必要时使用独立 checkout/worktree 验证指定 revision。
  已被修改的上游脚本跑通，不能报告为“原始脚本通过”。
- 确认实际加载位置。`megatron` 是 namespace 包，安装元数据版本可能与 checkout 不一致；
  在诊断子进程记录 `megatron.core.__file__`、`megatron.__path__` 和上游 git SHA。
- 用原始失败命令收集 traceback、rank、环境开关、输入、dtype、并行规模和退出码。
  将日志放在 `$MMP_ARTIFACTS`。先判断环境/依赖、激活、API 漂移、算子、通信或序列化问题。
- 缺数据、硬件或明确的模型配置时，先完成可执行的本地定位与回归，准确记录依赖的缺口。

开发安装（该解释器中需已具备运行依赖及 `dev` 所需 pytest/pytest-cov）：

```bash
"$MMP_PYTHON" -m pip install --no-deps -e "$MMP_ROOT[dev]"
```

`--no-deps` 避免安装操作改变现有运行栈；缺依赖时按项目和厂商环境约束单独补齐。
当前声明的版本范围见 `_compat.py`（`>=0.14,<0.17`），分支目标见 README；两者都不是
完整兼容证据，报告必须记录实际版本和 revision。

## 5. 实施：从原始失败到最小补丁

1. 用 `rg` 定位上游符号和本仓库相关 patch/test，确认根因及最小复现。使用新进程比较启用/
   禁用单个 patch 的行为；禁用总设备层本就可能失败，不能据此推断具体算子有问题。
2. 选择保持原有配置的最小接入点。若需要上层框架添加 `if musa` 或训练脚本关闭功能，
   继续向本包内寻找适配点，并把临时 workaround 留在问题记录中。
3. 在对应模块添加 `AttrPatch` 或 `HookPatch`，填写完整 ledger：

   | 字段 | 必填内容 |
   |---|---|
   | `id` | 唯一且稳定，沿用现有按目标/行为命名的风格。 |
   | `target` / `trigger` | 精确模块/符号或触发模块，不按框架名称分支。 |
   | `rationale` | 观测到的故障、受影响环境、根因；把推测与验证结论区分开。 |
   | `strategy` | 改了什么、保留了什么、回退代价和副作用。 |
   | `upstream` | 上游文件和相关符号，改写较多时附 revision。 |
   | `remove_when` | 在哪些测试和环境中，关闭该补丁仍通过即可删除。 |

4. 新模块同时加入 `patches/__init__.py` 的 import 和 `MODULES`；只修当前模块则不改
   全局顺序。`replace(original)` 返回 `None` 表示主动放弃；Hook 返回 `False` 表示放弃。
   不用这两个分支吞掉真实失败或缺失的必需符号。
5. wrapper 使用 `functools.wraps` 并验证签名和参数透传；保留未受影响路径。按已安装 API
   的真实签名/能力做适配，不用伪造版本或 capability 代替功能验证。已有合成 capability
   和 TE 版本策略不代表相应 kernel 实际可用。
6. 需要升级验证/退出开关时，在 `_env.py` 记录默认值、优先级和用途，并在两份 README
   同步。开关用于隔离和恢复上游行为，正常调用不能依赖下游手动开启新增兼容补丁。
7. 增加能捕获原始错误的回归测试，再运行受影响的真实入口。涉及 engine 时覆盖幂等、
   reload、别名、同目标链、所有权和失败回滚；涉及计算时比较前向/反向和 checkpoint。
8. 同步双语文档和本手册中受影响的流程。已满足 `remove_when` 的补丁应在保留回归测试后
   删除，并复跑原始入口；不要长期保留已经不需要的 workaround。

## 6. 分层验证：记录执行了什么，而不是只记录退出码

按改动影响选择必要层级；纯文档改动检查链接、命令和源码一致性即可，不要求启动训练。
代码改动先跑相关回归，再跑本包测试；改动 core 接口或激活机制时，源码、wheel、真实调用方
都需验证。资源不足时明确写未执行项目，不得补造通过结果。

### 6.1 本包回归和真实栈

```bash
cd "$MMP_ROOT"
"$MMP_PYTHON" -m pytest tests/test_activation.py tests/test_engine.py \
    tests/test_engine_lifecycle.py tests/test_ledger.py -q
"$MMP_PYTHON" -m pytest tests -q

MEGATRON_MUSA_RUN_INTEGRATION=1 MEGATRON_LM_PATH="$MEGATRON_LM_PATH" \
    "$MMP_PYTHON" -m pytest tests/test_megatron_integration.py \
    tests/test_te_layer_norm.py tests/test_rope.py tests/test_control_collectives.py -q
```

首条适用于机制/激活修改；只改某个算子时先运行对应测试文件。`tests/conftest.py` 在本包
测试进程内关闭自动激活，真实集成通过子进程单独启用。因此本包 pytest 通过不代表上游
pytest 的自动激活已通过。部分设备契约在检测到 MUSA 时直接运行，其余硬件轮次需显式开启；
始终查看实际 skip 原因。TE norm-linear 和控制面通信的部分检查需要两张卡。

### 6.2 直接运行上游原始单元测试

在独立 shell/进程中从上游根目录启动；不要将本包 `conftest.py` 或 patch import 注入上游。
下例在当前源码中的原始 parallel-state 用例上验证双 rank 通路。不同 revision 先确认 node id
仍存在，并查看对应的资源要求；不能只因测试名字含 parallel 就假定两卡能跑整个文件。

```bash
(
    cd "$MEGATRON_LM_PATH"
    export PYTHONPATH="$MEGATRON_LM_PATH${PYTHONPATH:+:$PYTHONPATH}"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=1
    export MEGATRON_MUSA_PATCH=1
    export MEGATRON_MUSA_PATCH_AUTOLOAD=1
    unset MEGATRON_MUSA_PATCH_ONLY MEGATRON_MUSA_PATCH_DISABLE
    "$MMP_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
        -m pytest tests/unit_tests/test_parallel_state.py::test_tensor_model_parellel_world_size -q -ra
)
```

先复跑任务涉及的原始失败 node id，再扩展相关模块；完整目标仍是上游测试集合。具备相应
资源和依赖后，可从同一上游目录运行完整集合，或沿用该 revision 的 CI 分桶入口：

```bash
# 8 卡示例，先核对测试资源要求。
(
    cd "$MEGATRON_LM_PATH"
    export PYTHONPATH="$MEGATRON_LM_PATH${PYTHONPATH:+:$PYTHONPATH}"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=1
    export MEGATRON_MUSA_PATCH=1
    export MEGATRON_MUSA_PATCH_AUTOLOAD=1
    unset MEGATRON_MUSA_PATCH_ONLY MEGATRON_MUSA_PATCH_DISABLE
    "$MMP_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
        -m pytest tests/unit_tests -ra
)
```

查看上游 `tests/unit_tests/run_ci_test.sh`、`conftest.py` 和测试本身：部分用例固定 TP/PP
组合，数据 fixture 可能使用 `/opt/data`；CI 分桶可能带 NVIDIA 硬件过滤，不能原样套用过滤
后宣称 MUSA 全覆盖。记录每个 rank 的结果、收集数和 skip 原因。当前上游 conftest 会将
“无测试收集”的退出码改为 0，因此必须确认确有用例执行。资源不足、依赖缺失与兼容失败
分别记录；未经修复的失败不能被新增 skip/xfail 隐藏。

### 6.3 训练、保存与恢复

先用本包小规模脚本定位基础链路（脚本会删除并重建指定 `OUTPUT_DIR`，只给它专用目录）：

```bash
cd "$MMP_ROOT"
MEGATRON_LM_PATH="$MEGATRON_LM_PATH" PYTHON="$MMP_PYTHON" NPUS=2 \
    OUTPUT_DIR="$MMP_ARTIFACTS/smoke" bash examples/run_pretrain_smoke.sh
```

然后在已核实未修改的上游 checkout 中，运行任务对应的原始 `pretrain_gpt.py` 命令或 shell
训练脚本，保留失败复现时的功能参数。按原入口允许的方式设置路径和资源；如果必须关闭某
功能才能启动，先记录该缺口，不能把 workaround 的结果当作原命令通过。

验收应包括模型构建、前向/反向、参数更新、有限 loss、多 rank 正常退出；涉及保存时，检查
checkpoint 内容并在新进程恢复后继续至少一步。涉及 optimizer/RNG 时验证相应状态恢复。
本包 smoke 默认 `--no-save-optim`、`--no-save-rng`，不能替代这类恢复测试。
`train_llama3_8b_musa.sh` 是适配示例；`train_llama3_8b_h100_fp8.sh` 启动器会覆盖层数和
训练计划，两者成功都不能单独证明原始上游脚本及原始配置通过。

### 6.4 wheel-only 与 ms-swift

在只安装目标 `megatron-core` wheel 的独立环境中测试；确认不存在 Megatron-LM editable
安装，也没有通过 `.pth`、`PYTHONPATH` 或当前目录导入 checkout。使用该环境解释器
替换 `MMP_PYTHON`，安装本包后可从临时目录做以下检查：

```bash
(
    cd "$MMP_ARTIFACTS"
    unset PYTHONPATH MEGATRON_LM_PATH
    unset MEGATRON_MUSA_PATCH_ONLY MEGATRON_MUSA_PATCH_DISABLE
    TORCH_DEVICE_BACKEND_AUTOLOAD=1 MEGATRON_MUSA_PATCH=1 MEGATRON_MUSA_PATCH_AUTOLOAD=1 \
        "$MMP_PYTHON" - <<'PY'
import importlib.util
import torch
import megatron.core
from megatron.core.transformer.transformer_config import TransformerConfig

print("core path:", megatron.core.__file__)
assert importlib.util.find_spec("megatron.training") is None
config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
print("config:", type(config).__name__)
print("device:", torch.empty(1, device="cuda").device)
PY
)
```

这只是安装、配置和设备通路检查。继续用该 wheel 环境运行任务对应的真实模型构造、
forward/backward，以及 ms-swift 原有 Megatron 训练命令/示例。记录 ms-swift revision、
模型与数据、CLI、并行规模、精度、保存/恢复结果；不得新增 MUSA 分支、改 import 路径、
换专用模型类或依赖本包自定义 launcher。至少覆盖实际训练步；涉及 checkpoint 的改动还要
保存并恢复。仅配置对象构造成功或 ms-swift import 成功，不算框架验收。

wheel 环境缺少 `megatron.training` / `megatron.legacy` 时，对应补丁的缺失跳过正常；
所需 core 补丁未生效则不正常。CPU/CUDA 对照用于确认未受影响路径和总开关退出行为；
不要在没有证据时宣称本包在非 MUSA 环境自动无副作用。需要禁用对照时在新进程设置
`MEGATRON_MUSA_PATCH=0`，记录这是显式禁用结果。

## 7. 调试顺序与报告解释

先查同一解释器的安装和 entry point，再查开关、导入路径、目标符号，最后查 kernel/通信。
下面是**诊断进程**，不是自动激活验收：

```bash
"$MMP_PYTHON" -c 'from importlib.metadata import entry_points; print(entry_points(group="torch.backends"))'
MEGATRON_MUSA_PATCH_DEBUG=1 "$MMP_PYTHON" - <<'PY'
import json
import megatron_musa_patch as m
m.apply()
print(json.dumps(m.report(), indent=2))
PY
```

| 现象 | 处理方式 |
|---|---|
| 目标未导入前 `pending` | 惰性激活的正常状态；读取 `report()` 不会主动应用补丁。 |
| `apply()` 后仍 `pending` / `failed` | 检查异常、目标是否解析、record `detail`，不要报告“全部应用”。 |
| `skipped` | 区分环境禁用、模块不存在、工厂主动退出；判断该任务是否真的不需要它。 |
| `PatchTargetMissing` | 查实际上游文件和符号漂移；不能捕获后继续伪装成功。 |
| `PatchConflict` | 查绑定所有权和第三方改写，避免无条件覆盖或扩大别名扫描范围。 |
| CUDA 不可用或 backend 不匹配 | 查解释器、entry point、torch_musa、torchada 和激活时序；不要让调用方改设备名绕过。 |
| 通信挂起 / 退出异常 | 保留各 rank 日志，区分训练 collective、控制面 barrier 和 teardown；验证调用顺序与组生命周期。 |

隔离单个补丁可用 `MEGATRON_MUSA_PATCH_DISABLE=<id>`；`ONLY` 优先于 `DISABLE`，且只启用
一个高层补丁可能缺少设备层前置条件。开关比较使用新进程，不依赖 `uninstall()` 清除
torchada 等依赖已经产生的全局副作用。

## 8. 交付前检查与结果模板

- [ ] 改动只包含任务所需文件，保留已有工作；上游和框架入口没有被修改来迎合补丁。
- [ ] 新 patch 已注册、元数据完整；开关、失败路径、卸载/所有权行为有相应验证。
- [ ] 已重跑原始失败命令，并验证适用的 API、数值、梯度、分布式与 checkpoint 契约。
- [ ] 已执行适用的本包/原始上游/训练/wheel/ms-swift 验收；未执行项和原因有明确记录。
- [ ] README 中英版本、CONTRIBUTING 中英版本及本文的相关内容一致。
- [ ] `git diff --check` 通过；复核 `git diff`、`git diff --cached` 和 untracked 文件，
      确认没有误带日志、模型、checkpoint 或外部源码副本。

交接时使用以下信息结构，内容简洁但可复现：

```text
问题与根因：原始入口 / 失败现象 / 已证实原因
实现：修改位置 / patch id / 保留的调用契约 / 回退及副作用
环境：各仓库 revision / Python 与依赖版本 / 设备数量 / 关键开关
验证：完整命令或日志位置 / 实际收集、通过、失败、跳过数 / 退出码
未覆盖：未执行或仍失败的路径 / 原因 / 临时 workaround / 下一步复现命令
移除条件：依赖或上游修复后，关闭哪些补丁、运行哪些测试即可确认删除
```

单次任务完成应指该任务范围内的修复和适用验证完成；项目整体无缝兼容的目标必须持续以
上述三类原始调用方的实际运行结果衡量。
