# 上游测试全量扫描 → 根因分流 → 全局修复：方法论文档

本文把「用 coding agent 对 Megatron-LM v0.16.1 做 MUSA 适配」过程中沉淀的
**测试全量扫描 → 失败归纳 → 根因分流 → 全局修复方案 → 分批实施与验收 → 进度报告**
工作流整理成可复用的范例。所有数据、命令与产物均来自真实运行记录
（`../../musa-test-report/`，基线 Megatron-LM `55ac7082` / core_v0.16.1，
8 × MTT S5000，torch 2.7.1a0 / torch_musa 2.7.1 / torchada 0.1.86 / MT-TE 2.0.0）。

适用场景：接入**新的厂商硬件栈**、**升级 Megatron 版本**、或初次为一个大体量上游
仓库建立系统化适配时。核心思想只有一条：

> 先让证据完整落地（全量跑、全量留痕），再做归纳与分流，最后按分流批量实施；
> 每一步的产物都是下一步的输入，也是可复核的证据。

---

## 1. 流程总览

```text
阶段 1  全量测试扫描 sweep.py          → results/*.json + *.log（174/174 文件）
阶段 2  聚合与签名   aggregate.py      → failures.json + failure_signatures.md
阶段 3  根因归类     classify.py       → ROOT_CAUSE_REPORT.md（分类规则表 + 归属模块）
阶段 4  全局修复方案 build_remediation_inventory.py
                                       → REMEDIATION_PLAN_zh.md + CASES/TSV（1080 → 六路分流）
阶段 5  分批实施与验收 devcheck.py    → dev/<label>/（基线 diff + 新证据，不覆盖基线）
阶段 6  进度与状态报告 REMEDIATION_STATUS.md → 任务表：补丁 id / 验证文件 基线→本次 / 遗留去向
```

要点：

- **每个阶段是独立脚本 + 独立产物**，可重跑、可审计、可交接；
- 修复方案（阶段 4）是对阶段 3 报告的**源码核对后再分流**，不是直接照抄报告建议；
- 实施按方案的任务编号分批（P01…C05…），每批都有「基线 → 本次」的量化验收。

---

## 2. 阶段 1：全量测试扫描

### 2.1 设计要点（`sweep.py`）

| 设计决策 | 理由 |
|---|---|
| **每个测试文件一个独立的 8-rank torchrun 会话** | 单文件的 hang/crash 只损失一个文件，不会级联污染整轮 |
| 清空上游 `addopts`（含 `-x`） | 上游配置会在首个失败处停止，必须覆盖为持续执行 |
| `--tb=long -ra` | 保留完整 traceback 与 skip 原因，供后续分类 |
| 每文件超时（900 s）+ teardown 宽限（120 s） | 区分「测试失败」「超时」「teardown 挂起」三类现象 |
| 产物双写：`results/<slug>.json`（机器可读）+ `.log`（人读） | JSON 供聚合脚本，log 供人工核对与链接 |
| `--experimental` 独立轮次 | 上游 `mark.experimental` 的用例单独成 pass，不污染主轮次计数 |

### 2.2 覆盖声明与已知缺口

本轮 `tests/unit_tests/**/test_*.py` 共 **174/174** 文件全部执行（inventory 见
`unit_files.txt` / `trial_files.txt`）。必须同时声明**没有覆盖**的部分：

- `tests/functional_tests/**`：`nemo_run` 未安装，golden-values fixture 直接报错；
- 803 个用例被上游自己的 `skipif` 跳过（原因在各文件日志中）；
- 无 NVIDIA/CUDA 对照环境——分类来自 traceback 与源码核对，不确定的标 *triage*。

> 覆盖缺口与「失败」同等重要：它们决定了结论的表述强度（“已验证” vs “未验证”）。

### 2.3 结果形态

```text
files executed: 174; non-zero exit: 83
test cases: 2132 passed, 1035 failed, 28 collect/setup errors, 803 skipped
bounded timeouts: 2; teardown hangs: 1; startup anomalies: 6
```

---

## 3. 阶段 2：聚合与签名

`aggregate.py` 从 `results/*.json` + `.log` 提取每个失败用例的：

- 异常首行（traceback 的 `E ` 行）；
- 最深的 `file.py:line` 上游帧；
- 参数化用例的完整 node id。

产出 `failures.json`（逐条）与 `failure_signatures.md`（按异常类型汇总）：

| exception | count | 示例位置 |
|---|---:|---|
| `AssertionError` | 545 | optimizer/muon.py:188 |
| `RuntimeError` | 341 | fsdp/param_and_grad_buffer.py:3213 |
| `ImportError` / `ModuleNotFoundError` | 98 | mamba_mixer.py:174 等 |
| `triton.compiler.errors.CompilationError` | 12 | batch_invariant_kernels.py:213 |

> 这一步的价值：把「174 份日志」压缩成「可计数的签名」，
> 让下一步的分类规则可以写成 **regex → 根因 → 归属模块** 的表驱动形式。

---

## 4. 阶段 3：根因归类（`classify.py`）

### 4.1 机制

- **规则表 first-match-wins**：`(匹配 "exc_type: exc" 的 regex, 根因类别, 归属 patch 模块)`；
- 每条失败落到唯一类别，并标注**负责修复的模块**（`patches/_transformer_engine.py`、
  `_distributed.py`、`environment (not a patch defect)` 等）；
- 无法确定的标 **triage**，不强行归类。

### 4.2 归类结果（示例）

| 根因 | failing | 文件数 | 归属 |
|---|---:|---:|---|
| `emerging_optimizers` 未安装 | 279 | 1 | 环境 |
| MCCL 不识别 ReduceOp（PREMUL_SUM） | 119 | 2 | `_distributed.py` |
| MT-TE `allocateSpace` 断言 | 100 | 7 | vendor / `_layer_norm.py` 可绕 |
| MT-TE 对 FP32 误选 flash 后端 | 84 | 15 | `_transformer_engine.py` |
| `grouped_gemm` 未安装 | 72 | 3 | 环境 |
| MT-TE TopkOut / permutation / flash varlen 等 kernel | 94 | 8 | vendor / 可回退 |
| …（完整表见 ROOT_CAUSE_REPORT.md §2） | | | |

### 4.3 统计口径纪律（容易出错的地方）

1. `1035 failed + 28 errors = 1063` 是**文件摘要合计**；`failures.json` 含跨 rank
   补充记录，实际 **1081 条观察**（1046 FAILED + 35 ERROR），去重后 **1080 个唯一
   node id**——两种口径不能混用相加。
2. 收集错误的 node id 可能只有文件路径，**不能据此推算文件内未收集的用例数**。
3. 「6 个 hard crash」实际是 `startup_failures.json` 里 6 个异常收尾文件：
   2 个 timeout + 若干「rank 0 通过其他 rank 失败」的 log-once 文件；
   SIGSEGV/SIGABRT/timeout/teardown-hang 必须分开统计。
4. 实验轮次（`__exp`）单独成表：19 条失败拆成 12 device + 6 shape + 1 FP32，
   不能整体归因为 device wrapper。

---

## 5. 阶段 4：全局修复方案（对报告的“源码核对再分流”）

`build_remediation_inventory.py` + 人工源码核对，把 1080 个唯一 node id 分成
**互斥六路**（`remediation_counts.json` / `REMEDIATION_PLAN_zh.md` / `REMEDIATION_CASES.md`）：

| 分流 | node id | 含义 |
|---|---:|---|
| A1 PATCH | 323 | 本包内明确的局部适配路径 |
| A2 PATCH_CONDITIONAL | 195 | 设计可行，但须先补数值/并行/状态契约 |
| B TE | 101 | 属 TransformerEngine 修复责任（保留 TE 模型/量化语义的边界） |
| C1 BLOCKED | 114 | 当前栈/项目边界下无法兼容（Mamba、batch-invariant Triton、Graph/RNG…） |
| C2 ENVIRONMENT | 287 | 先补环境（如 279 条 `emerging_optimizers`），不应写成“不可修复” |
| C3 TRIAGE | 60 | 证据不足，暂不承诺修复位置 |

### 5.1 对初版报告的根因修正（经验最密集的部分）

初步报告的归因建议**不能直接采用**——逐类用源码与实测核对后修正了 9 处：

| 初版归因 | 源码核对后的结论 |
|---|---|
| Flash 误选是合成 capability 8.3 造成 | TE 两处**硬编码** `use_flash_attention = True`；改 capability/backend selector 均无效 |
| ReduceOp 未知 → 搬到 CPU collective | FSDP 预缩放默认生成 PREMUL_SUM，torch_musa `mcclOp` 无该映射（enum 8 == 日志 `\x08`）；正确做法是设备端 `mul_` + SUM |
| 45 条 FP8 recipe 是改名，可 alias | MT 只有固定 tile_size 的 MTFP8BlockScaling；current/block scaling **不等价**，不能 alias |
| 100 条 allocator 全部等 TE | 拆开：44 条独立 LayerNorm 路径可覆盖 TENorm；56 条直接 TE 模型用例不在 norm-linear 补丁射程 |
| 21 条 TE API drift 同因 | 拆成 15 max-logit + 2 quantized_model_init + 4 norm-linear 子类构造破坏，修法各不同 |
| torchada 没翻译 nccl | 已翻译 `init_process_group`/`new_group`；20 条来自 `new_subgroups_by_enumeration` 在 c10d 内部**绕开**外层翻译 |
| 53 条 TopkOut 关 fusion 即可 | traceback 全落在普通 `torch.topk`（FP64 输入）——关 fusion 无效 |
| “Invalid type for 16 bit” 表示输入是 16-bit | 错误文案与实测相反：FP32/FP64 失败、FP16/BF16 正常；**错误文案会说谎，要以实测 dtype 矩阵为准** |
| safe_globals 是白名单不全 | 实际是 `Unsupported operand 80`；不能拿 `weights_only=False` 掩盖 |

两条责任边界声明（写进方案，避免争议）：

- “必须在 TE 修复”指**保留 TE 模型、量化语义与原生接口**的工程边界，
  不是声称 Python 理论上无法重写算法；
- 根因属于 TE ≠ 每个用例只能等 TE：某 TE 原生故障可以同时存在 Megatron 层回退
  （按 A1/A2 实施并逐用例验收）。

---

## 6. 阶段 5：分批实施与验收

### 6.1 实施批次的量化验收（基线 → 本次，per 文件）

| 任务 / 补丁 id | 验证文件：基线 → 本次 | 遗留去向 |
|---|---|---|
| P01 `fsdp.premul-sum.device-prescale` | test_mfsdp_fully_shard 178→78 | 余量迁移 C05/T02 |
| P02 `bridge/hyper-comm-grid.subgroups-backend` | 11→7；bridge 16 条 nccl 错误消除 | 厂商 kernel timeout（非补丁引入） |
| P03 `te.grouped-linear.mem-monitor-compat` | test_moe_experts 58→52 | 余量迁移 C01 |
| P04 `moe.topk.fp64-reference` | test_moe_layer_discrepancy 32→**0** | aux_loss 数值差异归 R01 |
| P05 `moe.permutation/unpermutation.unfused-musa` | test_a2a_token_dispatcher 18→**0** | — |
| P06 `moe.grouped-gemm.*`（3 个） | test_moe_experts 52→**0（198 passed）** | — |
| P07 norm-linear 子类构造修正 | 4 个构造 ERROR 消除 | forward 迁移 C01 |
| P08 撤销无条件 TE 版本谓词 | test_bert_model 4→2（诚实降级） | 1 条断言的是上游旧缺陷，不伪造 |
| P09 `softmax.kernel-availability.musa` | ModuleNotFoundError 消除 | 余 7 条 head_dim（C01） |
| P10 Tensor.musa 重入修正 | 包内 6 个硬件用例固化 | — |
| C01 attention 回退 | test_model_swap 28→**0**；packed 6→**0**；mfsdp 相关大幅下降 | 见下「回归警示」 |
| C02 独立 norm 回退 | test_attention_variant_dsa 41→**6** | — |
| C04/C05 quantized-init / DCP CPU staging | 2 条 / 78→**0** | 下层障碍暴露 T02/T01 |

每批实施同时交付：包内回归（`pytest tests/`，392 → 后续 487 passed）+
**新补丁的包内契约测试**（如 `test_attention_fallback.py` 16 项、含 CPU/MUSA 对照与
“禁止 SDPA 再分发”的负向断言）。

### 6.2 验收纪律

1. **验收命令与基线扫描完全一致**（8 rank / 文件隔离 / addopts 清空 / 超时相同），
   只有输出目录不同——`devcheck.py --label <label> <files>` 专用于“改后重跑 + 基线 diff”；
2. 证据目录 `dev/<label>/` 按批次隔离，**永不覆盖基线** `results/`；
3. 除验收文件外，同时跑**包内回归**捕获补丁间相互作用；
4. 单补丁对照用 `MEGATRON_MUSA_PATCH_DISABLE=<patch-id>`（新进程）。

### 6.3 回归警示实例（必须写进状态报告）

`resharding/test_model_swap.py` 在窄版 C01 下 **0 失败**，扩展版后 **23 失败**
（异步 `MUSA error: unknown error`）。处置流程：单用例隔离通过 →
`MUSA_LAUNCH_BLOCKING=1` 下通过 → 判定为跨用例设备状态/同步问题，
**在状态报告中明确标注为回归并给出定位建议**，而不是混在改进数字里。

### 6.4 诚实降级实例

- P08 后 `rng_error` 1 例**不可修复**：上游测试断言的是 TE 1.7–1.10 的已知缺陷，
  MT-TE 接受该参数 → `DID NOT RAISE`。按约定**不伪造**，维持失败并注明原因；
- C03 max-logit：旧库无可验证的统计契约 → **未实施**，15 条维持失败并注明；
- 旧库参考（`musa_patch/dot_product_attention.py` 等）只取接口/scale/RNG 线索，
  不复制实现（旧 attention 强制 flash、忽略 mask，不能用来绕开同一 native 故障）。

### 6.5 “修一层、露一层”是常态

C05 修完 DCP CPU staging（78→0）后暴露 T02：56 条直接 TE 模型的
`allocateSpace`。这不是回归，而是上一层的报错**遮蔽**了下一层。状态报告必须把
“已消除”与“新暴露”分列（如 178→62 = 56 T02 + 6 FP8 链路）。

---

## 7. 阶段 6：进度与状态报告（`REMEDIATION_STATUS.md`）

表结构（每个任务一行，可复核、可续接）：

```text
| 任务 / 补丁 id（patches/ 模块） | 验证文件与结果（基线 → 本次） | 备注（遗留去向 / 厂商问题 / 诚实降级） |
```

外加：

- 包内回归数字（`pytest tests/`）；
- 「已知未覆盖」段（CP>1、C03、split-phase attention 等显式列出）；
- 交付物清单（新增/修改的代码与测试文件、验收工具、环境变更记录）；
- 复跑命令（`devcheck.py --label <label> <files>`、单补丁 DISABLE 对照）；
- 第二轮续修记录（新证据目录、3 个原始单 rank 用例回归）。

---

## 8. 经验教训清单

1. **文件级隔离**是大型上游套件扫描的第一原则：hang/crash 只损失一个文件。
2. **清空上游 addopts（-x）**，否则只得到第一个失败。
3. **双写产物**：机器可读 JSON + 人读 log；后续所有阶段只消费产物，不重跑推理。
4. **统计口径先行**：观察记录数 ≠ 唯一 node id 数；summary 合计 ≠ failures.json；
   crash/timeout/hang 分类统计。口径错误会传导进所有下游结论。
5. **先清环境缺口**（本例 421 条 optional deps），否则真实信号被淹没。
6. **错误文案会说谎**：“16 bit” 实为 FP32/FP64 失败、“Unsupported operand 80”
   实为 dtype 问题、白名单报错实为算子缺陷——以实测矩阵（dtype × 路径）为准。
7. **堆栈首帧 ≠ 根因归属**：初版报告 9 处归因被源码核对推翻
   （硬编码 flash、绕过翻译的 subgroup 路径、被 fused 标签掩盖的普通 topk…）。
8. **分流必须互斥且保留原始证据**：每条 node id 唯一路径 + 原始记录号 + 日志链接。
9. **诚实降级**：断言上游旧缺陷的用例不伪造通过；无法验证契约的方案标注
   “未实施”，不产出半成品 patch。
10. **修一层露一层**：上层报错会遮蔽下层缺陷，状态报告要分列“消除/新暴露”。
11. **基线永不覆盖**：验收产物写 `dev/<label>/`，与基线 `results/` 物理隔离。
12. **回归要单独定性**：改进数字里混入回归会掩盖跨用例设备状态问题
    （单用例隔离 + LAUNCH_BLOCKING 定位是有效手段）。

---

## 9. 产物清单与复现

| 文件（`../../musa-test-report/`） | 用途 | 生成者 |
|---|---|---|
| `sweep.py` | 全量扫描（文件级隔离 8-rank 会话） | 手写 |
| `unit_files.txt` / `trial_files.txt` | 扫描清单（主轮次 / experimental） | sweep 输入 |
| `results/<slug>.{json,log}` | 每文件 pytest 结果与完整日志 | sweep |
| `startup_failures.json` | 异常收尾文件（timeout / rank 分化） | sweep |
| `aggregate.py` → `failures.json`, `failure_signatures.md` | 逐用例异常签名 | 手写 |
| `classify.py` → `ROOT_CAUSE_REPORT.md` | 规则表根因归类 + 归属模块 | 手写 |
| `build_remediation_inventory.py` → `REMEDIATION_PLAN_zh.md`, `REMEDIATION_CASES.md`, `remediation_cases.tsv`, `remediation_counts.json` | 六路分流与逐用例清单（含日志链接） | 手写 |
| `devcheck.py` → `dev/<label>/` | 改后重跑 + 基线 diff（不覆盖基线） | 手写 |
| `REMEDIATION_STATUS.md` | 实施状态：补丁 id / 基线→本次 / 遗留 / 回归警示 | 手写 |

复现（新基线）：

```bash
cd musa-test-report
python sweep.py                       # 全量 / --experimental 实验轮次
python aggregate.py && python classify.py
python build_remediation_inventory.py
# 修复批次：
python devcheck.py --label p01 tests/unit_tests/distributed/megatron_fsdp/test_mfsdp_fully_shard.py
```

## 10. 新一轮适配的复用检查单

- [ ] 冻结环境表（Megatron SHA / patch 版本 / torch / torch_musa / torchada / TE / 卡数）
- [ ] 重新生成 `unit_files.txt`，确认上游新增/删除的测试文件
- [ ] 更新 `classify.py` 规则表（新增异常签名；删除已修复根因）
- [ ] 重跑 sweep → aggregate → classify，得到新基线报告
- [ ] 重新分流并生成 REMEDIATION_PLAN（沿用六路定义）
- [ ] 按任务编号分批实施；每批 `devcheck.py` 验收 + 包内回归
- [ ] 更新 REMEDIATION_STATUS（保留历史轮次，分列消除/新暴露/回归）
- [ ] 满足 `remove_when` 的补丁按手册 §5.8 退役，并复跑原始入口确认

> 相关文档：[补丁独立性检查记录](PATCH_INDEPENDENCE.md)、
> [ms-swift 适配总结报告](ms-swift-musa-adaptation-report.zh.md)、
> 原始素材目录 `../../musa-test-report/`。
