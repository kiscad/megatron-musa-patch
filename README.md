# megatron-musa-patch

The goal is to run **unmodified Megatron-LM / Megatron-Core seamlessly on MUSA**: upstream unit tests and training scripts should run as supplied, and frameworks such as **ms-swift should call the same Megatron-Core interfaces without distinguishing MUSA from CUDA**.

This repository owns the runtime adaptation. Installing it in a supported MUSA environment should be sufficient; callers should not need a MUSA fork, a special import, a replacement model class, or a device-specific branch. This is the acceptance target, not a claim that every upstream test and training configuration already passes.

## Compatibility contract and current coverage

| Surface | Acceptance target |
|---|---|
| Megatron-LM unit tests | Run the original tests and assertions, including their CUDA-spelled APIs and `nccl` configuration. Fix compatibility in this package; do not rewrite or skip tests to hide a MUSA failure. |
| Megatron-LM training | Run original Python and shell training scripts without source edits or required MUSA launchers. Preserve model, optimizer, distributed and checkpoint semantics. |
| Megatron-Core callers, including ms-swift | Preserve import paths, public signatures, configuration objects, outputs and state-dict contracts. Core compatibility must work with a `megatron-core` wheel, without depending on `megatron.training` being present. |

Installing MUSA dependencies and supplying existing launcher inputs (data paths, output directories, device count and a resource-appropriate model size) are environment setup. Requiring callers to replace `cuda` with `musa`, `nccl` with `mccl`, add `import megatron_musa_patch`, or disable a requested feature is a compatibility gap. Such workarounds may help diagnosis, but do not satisfy the unchanged-caller target.

The current implementation prioritizes correctness and uses conservative fallbacks where needed: PyTorch normalization, synchronous DP reduction and in-process checkpoint bucket writes. It also has targeted TE/FP8 and fused-RoPE checks. These checks and the examples cover specific paths; they do not establish full upstream-suite coverage, unchanged-script coverage, or end-to-end ms-swift compatibility. Record remaining failures, skips and required overrides explicitly. Performance work follows correctness and must retain the same caller contract; fallback trade-offs and removal conditions remain reviewable in the patch ledger.

For development, read [CONTRIBUTING.md](CONTRIBUTING.md). Coding agents should start with [AGENTS.md](AGENTS.md), which defines the working procedure and acceptance evidence. [中文文档](README_zh.md).

## Quick start

```bash
git clone https://github.com/kiscad/megatron-musa-patch
cd megatron-musa-patch && git checkout v0.16.1-dev
pip install .

cd /path/to/Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
    ...your usual arguments...
```

A `torch.backends` entry point activates the patch set at the end of `import torch`, and a `sys.meta_path` watcher applies each patch as soon as its target module exists. The command above illustrates retaining your existing arguments; validate the actual configuration against the coverage and trade-offs below. Activation must precede Megatron's use of CUDA APIs; the default channel is intended to meet this deadline without caller edits. Late activation repairs supported module-level aliases only, not already-created objects.

Inspect registered patches, including their root cause, strategy and removal condition:

```bash
python -c "import megatron_musa_patch as m, json; print(json.dumps(m.report(), indent=2))"
```

This fresh-process report is lazy: `pending` is expected before the target imports. To resolve targets in a diagnostic process, call `m.apply()` before `m.report()`; this explicit activation does not test the automatic channel used by unchanged callers.

`examples/run_pretrain_smoke.sh` is a 2-GPU, 5-step end-to-end self-check:

```bash
MEGATRON_LM_PATH=/path/to/Megatron-LM NPUS=2 PYTHON=/path/to/venv/bin/python \
    bash examples/run_pretrain_smoke.sh
```

`examples/train_llama3_8b_musa.sh` adapts upstream's llama3-8b H100 FP8 example for
MUSA (FP8 via MT-TransformerEngine with RMSNorm, or `DTYPE=bf16` for the pure-PyTorch
local impl; TP=2 for 48GB cards; a shrunk default mock schedule). See the script
header for the full list of changes and the env overrides for short bring-up runs.

```bash
MEGATRON_LM_PATH=/path/to/Megatron-LM PYTHON=/path/to/venv/bin/python \
    bash examples/train_llama3_8b_musa.sh
```

These are diagnostic examples with selected arguments, not proof that every original upstream launcher works unchanged. `run_pretrain_smoke.sh` recreates its `OUTPUT_DIR`; use a dedicated scratch directory. The step-by-step upstream-test, training and framework acceptance procedure is in [AGENTS.md](AGENTS.md).

## Which Megatron?

Anything that provides an importable `megatron` package:

```bash
# a) Megatron-LM source checkout -- recommended: also gives megatron.training
git clone https://github.com/NVIDIA/Megatron-LM
git -C Megatron-LM checkout core_v0.16.1
export PYTHONPATH=/path/to/Megatron-LM

# b) just the pip wheel -- megatron.core only, typically pulled in by an
#    upper-level training framework
pip install megatron-core==0.16.1
```

Option b) is the typical setup when [Megatron-Core](https://pypi.org/project/megatron-core/) is consumed through an upper-level framework — for example **ms-swift**'s Megatron training (`megatron-core>=0.16`) or **Megatron Bridge 0.1–0.3** (Megatron-Core 0.14–0.16). Only the `megatron.core` / `torch` patches apply there; the `megatron.training` / `megatron.legacy` patches are recorded as `skipped`, which is expected. **Megatron Bridge ≥ 0.4 needs Megatron-Core 0.17+**, outside this package's supported range.

Megatron is deliberately not a pip dependency here: the wheel ships only `megatron/core/`, there is no `megatron-lm` on PyPI, and the real workflow wants a checkout pinned to a tag. Supported upstream range: `>=0.14,<0.17` — outside it you get a warning (an error with `MEGATRON_MUSA_PATCH_STRICT=1`).

## Conservative bring-up flags

These select a conservative PyTorch path for diagnosis. They are optional workarounds for narrowing failures, not the compatibility contract or a prerequisite that downstream callers should have to encode. If a requested upstream configuration only works after adding them, record the original failure as an open gap:

```
--transformer-impl local
--no-masked-softmax-fusion --no-gradient-accumulation-fusion
--no-bias-swiglu-fusion --no-bias-dropout-fusion --no-rope-fusion
--attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
--distributed-backend nccl        # rewritten to mccl for you by torchada
CUDA_DEVICE_MAX_CONNECTIONS=1
```

`--no-rope-fusion` is no longer needed for a run to start: fused rotary embedding
is served by apex's kernels on MUSA (see the rope rows below), and the fused
`sbhd`/`thd` results are checked against Megatron's unfused reference by the
opt-in hardware test in `tests/test_rope.py`. Keep the flag for a PyTorch path —
`rotary_interleaved` models and packed sequences under context parallel fall back
to it automatically, with a warning.

## What gets changed

On a MUSA PyTorch build `torch.cuda` is a dead shell (no `_cuda_*` bindings) and Megatron references it everywhere. [torchada](https://pypi.org/project/torchada/) — Moore Threads' own CUDA→MUSA adapter — does the mechanical translation; this package adds four Megatron-specific overrides on top of it plus the Megatron patches below. The live, machine-readable list — `id`, `rationale`, `strategy`, `upstream`, `remove_when` — is `megatron_musa_patch.report()`.

| id | target | root cause (short) |
|---|---|---|
| `torch.cuda.compat-layer` | `torch.cuda` | no CUDA bindings on the MUSA build; torchada + four overrides: live `is_available()` probe, `Tensor.type()` CUDA names, `CUDAGraph` alias, `.musa()` transfer fix for tensor subclasses (e.g. TE `Float8Tensor`) |
| `torch.cuda.device-capability.nvidia-scale` | `torch.cuda.get_device_capability` | Megatron compares capability against NVIDIA thresholds (≥8 grouped-GEMM gate); report a synthetic `8.3` |
| `megatron.training.get-device-arch-version.nvidia-scale` | `megatron.training.utils:get_device_arch_version` | same comparison via `device_properties().major` |
| `torch.distributed.clean-teardown` | `atexit` | runs that exit with live MCCL groups showed watchdog errors; best-effort `destroy_process_group()` at exit |
| `megatron.fusions.fused-layer-norm.pure-torch` | `fused_layer_norm:FusedLayerNorm` | apex imports but its CUDA extension is unavailable; config-authoritative PyTorch LayerNorm/RMSNorm fallback |
| `megatron.fusions.fused-layer-norm.have-apex-flag` | `fused_layer_norm:HAVE_FUSED_LAYER_NORM` | consumers branch on the flag; keep it truthful once the class is replaced |
| `megatron.fusions.persist-layer-norm.disable` | `fused_layer_norm:HAVE_PERSIST_LAYER_NORM` | the fallback never runs apex's persistent kernel |
| `megatron.transformer-block.layer-norm.impl-local` | `transformer_block:LayerNormImpl` | TE's norm op aborted (`allocateSpace`) on the affected MUSA stack even for a `local` spec |
| `megatron.te.layer-norm-linear.unfused` | `transformer_engine:TELayerNormColumnParallelLinear` | For LayerNorm only, PyTorch norm + TE Linear bypasses the fused norm `allocateSpace` failure in QKV/FC1 while retaining FP8 Linear and norm checkpoint names |
| `megatron.te.cpu-offload-context.signature-dispatch` | `transformer_engine:get_cpu_offload_context` | every `TransformerBlock` build calls TE's CPU-offload helper; when the fork's reported version disagrees with its real signature the six-argument (TE ≥ 2.5) call is picked while the fork takes five, so the run died before the model existed; dispatch on the installed function's own signature |
| `megatron.fsdp.premul-sum.device-prescale` | `fsdp...param_and_grad_buffer:gradient_reduce_preprocessing` | torch_musa/MCCL has no PREMUL_SUM; the FSDP averaging branch prescales on device with `mul_` and reduces with SUM, other branches pass through |
| `megatron.bridge-communicator.subgroups-backend` | `pipeline_parallel.bridge_communicator:dist` | `new_subgroups_by_enumeration` calls `new_group` inside c10d, bypassing torchada's nccl→mccl translation; a module-local proxy translates exactly the `nccl` request |
| `megatron.hyper-comm-grid.subgroups-backend` | `hyper_comm_grid:dist` | same bypass as the bridge communicator; same proxy, `nccl` only |
| `megatron.te.grouped-linear.mem-monitor-compat` | `musa_patch` import in `transformer_engine...grouped_linear.py` | MT-TE hard-depends on the legacy `musa_patch.mem_utils.MemMonitor`; a minimal shim with the `max_token_num` counter is owned (sys.modules entries, undoable) and never shadows an existing package |
| `megatron.moe.topk.fp64-reference` | `transformer.moe.moe_utils:torch` | MuDNN TopK rejects float64; moe_utils's torch global becomes a forwarding proxy adapting only FP64 topk via same-precision CPU indices + device gather |
| `megatron.moe.permutation.unfused-musa` | `transformer.moe.moe_utils:permute` | TE's moe_permute kernel aborts for FP32/FP64; demote exactly that combination to upstream's `fused=False` reference |
| `megatron.moe.unpermutation.unfused-musa` | `transformer.moe.moe_utils:unpermute` | paired demotion so both ends share one index format (declared via requires) |
| `megatron.moe.grouped-gemm.torch-ops` | `transformer.moe.grouped_gemm_util:ops` | the fanshiqing grouped_gemm has no MUSA build; reference `ops.gmm` from per-expert `torch.matmul` (trans_b, empty-expert gradients), declined when the vendor package exists |
| `megatron.moe.grouped-gemm.available-flag` | `grouped_gemm_util:grouped_gemm_is_available` | report truthfully once the fallback is installed (requires torch-ops) |
| `megatron.moe.grouped-gemm.assert-noop` | `grouped_gemm_util:assert_grouped_gemm_is_available` | GroupedMLP's construction assert now consults the patched availability flag |
| `megatron.softmax.kernel-availability.musa` | `fused_softmax:FusedScaleMaskSoftmax.is_kernel_available` | the probe imports `scaled_masked_softmax_cuda` and dies with ModuleNotFoundError; return False when the extension is absent so the torch fallback runs |
| `megatron.te.norm.unfused-musa` | `transformer_engine:TENorm` | TE's standalone LayerNorm/RMSNorm aborts in MUSA `allocateSpace`; subclass the TE modules and replace only forward (functional norm, params cast to input dtype), keeping isinstance/sharding contracts |
| `megatron.te.attention.capability-dispatch` | `transformer_engine:TEDotProductAttention.forward` | Capability dispatch that implements no attention math: flash-capable inputs (FP16/BF16, head_dim 64–192, no dropout) run the native MT-TE flash path; everything else eligible runs TE's own UnfusedDotProductAttention backend; padded THD is sliced into per-sequence vendor calls because native THD drops `cu_seqlens` and yields NaN. Declines entirely when the TE hard-coded-flash marker is gone (source probe). CP>1/FP8 DPA/special softmax/windows/max-logit remain upstream. |
| `megatron.te.quantized-model-init.delayed-compat` | `transformer_engine.pytorch.quantized_model_init` | Megatron-triggered, ownership-tracked TE alias delegates only explicit DelayedScaling to native fp8_model_init; preserves high-precision initialization and nested contexts. Other enabled recipes are rejected; native attributes are never overwritten. |
| `megatron.te.factory-shim.torchscript-compat` | `transformer_engine:musa/patch_after_import_torch` | MT-TE rebinds torch factory functions (tensor/zeros/ones/empty/rand/arange/empty_like) to untyped device-translation wrappers, so any eager `torch.jit.script` over a factory call fails; each scripting call temporarily registers the seven known vendor wrappers in TorchScript’s ATen builtin table, leaving eager factory bindings unchanged during compilation and preserving `device='cuda'` translation intact. MUSA-fork-only; needed before Megatron import. Compiled graphs retain ATen device semantics; CUDA string literals are not translated inside them. Skips when the installed TE has no factory wrappers (source probe). Undo preserves subsequent third-party replacements. |
| `megatron.te.utils-module.safe-seed` | `transformer_engine:musa/pytorch/utils` | The vendor module iterates sys.modules with lazy-module getattrs at import time, which crashes the whole TE import with transformers 5.x ('dictionary changed size during iteration'); a replica without the fragile loop is seeded before the vendor body runs (same pattern as the musa_patch.mem_utils shim). Skips when the vendor loop is gone (source probe). |
| `megatron.embeddings.fused-rope.apex` | `rope_utils:fused_apply_rotary_pos_emb` | Megatron gated its fused `sbhd` kernel on a TE import (`…attention.rope`) the MUSA TE tree does not have, so `apply_rope_fusion` — argparse's default — was rejected before the first step; bind apex's equivalent kernel |
| `megatron.embeddings.fused-rope-thd.apex` | `rope_utils:fused_apply_rotary_pos_emb_thd` | the packed (`thd`) half of the same missing import; apex's padded-layout kernel, `cp_size=1` |
| `megatron.embeddings.rope-fusion.unfused-fallback` | `rope_utils:apply_rotary_pos_emb` | apex has no interleaved and no context-parallel variant; demote exactly those calls to upstream's unfused branch (warned once) rather than aborting mid-step |
| `megatron.legacy.fused-kernels.load.noop` | `legacy.fused_kernels:load` | the loader probes `nvcc`/defines an extension builder — no MUSA build route |
| `megatron.training.set-jit-fusion-options.noop` (+ `initialize` alias) | `set_jit_fusion_options` | startup CUDA warm-up/compile path needs validation on MUSA; skipped until proven |
| `megatron.dist-ckpt.musa-cpu-staging` | DCP filesystem device selector (Megatron hook) | use the actual MUSA stream device under CUDA API emulation so CPU staging is not skipped; preserve upstream asynchronous copies, synchronization and checkpoint sharding |
| `megatron.dist-ckpt.no-fork-writer` | `FileSystemWriterAsync.write_preloaded_data_multiproc` | forked bucket workers segfaulted in `torch.save` after MUSA init and hung the parent; write the same buckets sequentially in-process |
| `megatron.training.overlap-flags.noop` | `training.arguments:validate_args` | the observed TE fused-wgrad/DDP integration left `param.grad` None, breaking Megatron's overlap backward hook; DP-overlap flags are forced off with a warning |
| `megatron.training.profile.pytorch` | `training.arguments:validate_args` | bare `--profile` enters `cudaProfilerStart/Stop`/NVTX, which the MUSA runtime does not provide; `--use-pytorch-profiler` is enabled instead |
| `megatron.training.start-time.integer-microseconds` | `training:torch` | Module-local startup timestamp MIN uses integer microseconds on MCCL |
| `megatron.training.checkpoint.host-barrier-proxy` / `host-barrier-context` | `checkpointing:torch` / `save_checkpoint` | Create a world Gloo group before checkpoint I/O; only save barriers use it |
| `megatron.serialization.signal-member-globals` | `safe_globals:SAFE_GLOBALS` | Explicit signal member aliases allow Python 3.10 weights-only argument loading |

Every patch record carries a `rationale` (observed root cause, version-specific and honest about what is *not* proven), a `strategy` (what the replacement does and preserves), the upstream file it adapts, and a `remove_when` review condition tied to specific tests. `uninstall()`/`unapply()` restore owned changes; hooks with external side effects (e.g. torchada's import-time mutations) are documented as irreversible.

**Retired:** `megatron.training.ckpt-format.no-torch-dist` (which silently rewrote `--ckpt-format torch_dist` to `torch`) is gone. The rewrite changed save/load semantics — `torch` is Megatron's legacy format, not a drop-in backend for `torch.distributed.checkpoint`, and the rewrite broke async save and FSDP configurations. Pass `--ckpt-format torch` explicitly if you want the legacy writer.

### Trade-offs worth knowing

Patch selection is independent where possible. The two local norm flags require
`megatron.fusions.fused-layer-norm.pure-torch`, and checkpoint
`host-barrier-context` requires `host-barrier-proxy`. These same-module
requirements appear in `report()[...]["requires"]`; a missing, disabled or declined
companion produces a reasoned `skipped` status, never an implicit enable.
Block norm builds its own fallback if the local-class patch is disabled. RoPE
dispatch inspects live kernels and uses a private config copy; it remains a
pass-through wrapper when no apex kernel is installed. See the full
[patch independence review](docs/PATCH_INDEPENDENCE.md).

* `torch.cuda.is_available()` is a live MUSA probe, not a constant — but upstream also uses it as an *"am I on NVIDIA?"* probe in a few places (e.g. the FP8 checkpoint path gates its TransformerEngine import on it); review those if you enable FP8 checkpoints.
* Checkpoint bucket writes are serialised in-process (a throughput trade, not a format change). `MEGATRON_MUSA_PATCH_CKPT_FORK=1` restores upstream's forked writer if your MUSA build survives fork-after-init. Megatron's outer async-save path (`async_utils.DynamicAsyncCaller`) still forks independently of this patch.
* DP-overlap flags (`--overlap-grad-reduce` / `--overlap-param-gather`) are disabled by the fused-wgrad/DDP integration policy, which may reduce throughput. `MEGATRON_MUSA_PATCH_DP_OVERLAP=1` restores upstream behaviour for testing (legacy spelling `MEGATRON_MUSA_PATCH_TP_OVERLAP` still works). This does not touch `tp_comm_overlap`.
* Bare `--profile` automatically enables `--use-pytorch-profiler` and warns on rank 0. Skip `megatron.training.profile.pytorch` with `MEGATRON_MUSA_PATCH_DISABLE` to restore upstream profiler selection.
* Rope fusion is a MUSA kernel choice, not upstream's: the fused kernels come from Moore Threads' apex, and only the combinations apex implements are fused. `rotary_interleaved` models and packed sequences under context parallel are demoted to Megatron's unfused rotary embedding with a one-time warning — correct, slower, and no longer fatal. `MEGATRON_MUSA_PATCH_ROPE_FUSION=0` declines the fallback entirely, which leaves upstream's `apply_rope_fusion is not available` error in place (then pass `--no-rope-fusion`).

## Activating it yourself

All three channels are idempotent. Automatic activation and explicit import register the watcher without proactively importing Megatron; they can patch modules already loaded. `apply()` actively imports targets and applies patches immediately. Explicit activation is for diagnostics and extensions; unchanged-caller acceptance must exercise autoload:

| channel | when it fires | kill switch |
|---|---|---|
| `torch.backends` entry point (default) | end of `import torch` | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` or `MEGATRON_MUSA_PATCH_AUTOLOAD=0` |
| `import megatron_musa_patch` | explicit | `MEGATRON_MUSA_PATCH=0` |
| `megatron_musa_patch.apply()` | explicit, immediate | `MEGATRON_MUSA_PATCH=0` |

## Switches

| Variable | Default | Meaning |
|---|---|---|
| `MEGATRON_MUSA_PATCH` | `1` | Master switch. `0` disables auto-activation *and* makes `import megatron_musa_patch` a no-op. |
| `MEGATRON_MUSA_PATCH_AUTOLOAD` | `1` | Only the automatic (entry point) channel. |
| `MEGATRON_MUSA_PATCH_DISABLE` | *(empty)* | Comma-separated patch ids to skip. |
| `MEGATRON_MUSA_PATCH_ONLY` | *(empty)* | Whitelist; overrides `..._DISABLE`. Unknown ids are warned about. |
| `MEGATRON_MUSA_PATCH_STRICT` | `0` | Upstream version drift raises instead of warning. |
| `MEGATRON_MUSA_PATCH_DEBUG` | `0` | Log every applied patch at INFO. |
| `MEGATRON_MUSA_PATCH_ARCH` | `8.3` | Synthetic NVIDIA capability, e.g. `9.0`. |
| `MEGATRON_MUSA_PATCH_BLOCK_LAYERNORM` | `local` | `upstream` keeps `LayerNormImpl = TENorm`. |
| `MEGATRON_MUSA_PATCH_TE_FUSED_LAYERNORM` | `0` | `1` restores upstream TE fused norm-linear for upgrade testing. |
| `MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES` | *(empty)* | `1`/`true`/`*` bypasses all version gates; a comma-separated distribution list bypasses only those gates. Capability probes and patch selection still apply. |
| `MEGATRON_MUSA_PATCH_ROPE_FUSION` | `1` | `0` declines the apex fused-RoPE fallback, so upstream keeps reporting `apply_rope_fusion` as unavailable. |
| `MEGATRON_MUSA_PATCH_JIT_WARMUP` | `0` | `1` keeps upstream's JIT warm-up. |
| `MEGATRON_MUSA_PATCH_CKPT_FORK` | `0` | `1` keeps upstream's forked checkpoint writer. |
| `MEGATRON_MUSA_PATCH_DP_OVERLAP` | `0` | `1` honours the DP-overlap flags again (legacy spelling `MEGATRON_MUSA_PATCH_TP_OVERLAP` applies when this is unset). |
| `MEGATRON_MUSA_PATCH_TEARDOWN` | `1` | `0` skips the clean process-group shutdown handler. |

The local norm and block norm patches do not cover normalization inside TE `LayerNormLinear`. RMSNorm constructs the original TE fused module, preserving its native forward/backward path without an extra runtime wrapper. The norm-linear fallback applies only to LayerNorm in Megatron specs using `TELayerNormColumnParallelLinear`, including GPT QKV and FC1. It retains `layer_norm_weight`/`layer_norm_bias` and their replicated checkpoint sharding. FP8 quantization and GEMM remain in TE; fusion performance and rounding can differ. Direct TE modules and the TE operation fuser are outside this replacement. Existing TE FP8 extra-state compatibility across module types or TE versions still requires checkpoint validation.

Run the norm-linear CPU and single-/two-device MUSA regression tests:

```bash
MEGATRON_MUSA_RUN_INTEGRATION=1 MEGATRON_LM_PATH=/path/to/Megatron-LM \
    python -m pytest tests/test_te_layer_norm.py -q
```

### Migrated upstream source changes

Startup timestamp synchronization, checkpoint host barriers and signal aliases now live in `patches/_control_collectives.py`. No Megatron source edits are required. The torch namespaces are local to the two Megatron modules, and unrelated collectives pass through. Host groups are cached by distributed-world identity; normal distributed teardown owns their destruction. Disabling either checkpoint patch leaves its barrier redirection inactive. The safe-globals patch extends the registration list; aliases already registered by PyTorch remain registered after uninstall, matching upstream registration lifetime.

The previously edited H100 FP8 example is available as a launcher with 16 layers and the reduced sample schedule:

```bash
MEGATRON_LM_PATH=/path/to/Megatron-LM PYTHON=/path/to/venv/bin/python \
    bash examples/train_llama3_8b_h100_fp8.sh
```

It runs the unchanged upstream script, overriding `NUM_LAYERS`, `TRAIN_SAMPLES`, `LR_DECAY_SAMPLES`, and `LR_WARMUP_SAMPLES` via CLI arguments. These are example defaults, not global training policies.

## Troubleshooting

* **`PatchTargetMissing`** — upstream renamed or removed a symbol; the message names the patch, the symbol and the detected Megatron version. Skip it with `MEGATRON_MUSA_PATCH_DISABLE=<id>`, or update the target (see [the contributor guide](CONTRIBUTING.md#6-adding-a-patch)).
* **Training still cannot find CUDA** — check activation with `python -c "import megatron_musa_patch as m, json; print(json.dumps(m.report(), indent=2))"`. All `pending` → Megatron was never imported. Empty report → `MEGATRON_MUSA_PATCH=0` is set, or the entry point is not installed: `python -c "from importlib.metadata import entry_points; print(entry_points(group='torch.backends'))"`.
* **A patch made things worse** — bisect with `MEGATRON_MUSA_PATCH_DISABLE=<id>`, or switch everything off with `MEGATRON_MUSA_PATCH=0`. `MEGATRON_MUSA_PATCH_DEBUG=1` logs every patch as it lands.

## Versioning

The package version tracks the Megatron-LM release it patches: `0.16.1.dev0` on this branch (`v0.16.1-dev`), `0.16.1` for a release cut from it, `0.16.1.post1` for a fix, `0.16.2.dev0` on the next upstream's new branch.

## Verified against

The recorded reference environment below is not an exhaustive test matrix. The declared upstream range (`>=0.14,<0.17`) is a version guard, not evidence that every version, feature or framework has passed. Report actual revisions, commands, pass/fail/skip counts and untested paths for each validation run.

Python 3.10 · PyTorch 2.7.1a0 (MUSA build) · torch_musa 2.7.1 · torchada 0.1.86 · Megatron-LM `core_v0.16.1` (`megatron-core` 0.16.1) · MT-TransformerEngine 2.0.0 · apex (MT fork, fused RoPE) · MTT S5000.

The two early TE hooks are the only exceptions to Megatron-triggered activation;
registration remains free of accelerator imports. The device layer clears only
Transformers CUDA/BF16/FP16/TF32 and FlashAttention probe caches after activation,
rollback and unapply. Unrelated package caches are preserved. The safe-utils
bridge covers ordinary imports; live reload of vendor TE modules is unsupported.
Start a fresh process after changing TE or patches. The RoPE dispatcher also
selects Megatron's unfused branch for interleaved calls through native Core TE
wrappers when their TE < 2.3 version guard would reject the call.

### Declarative version gates

Both `AttrPatch` and `HookPatch` accept `version_gates=("transformer_engine >=2.0,<2.1",)`.
Comparisons within a string and gates within the tuple are AND-ed; an empty tuple imposes no restriction.
The three TE norm patches currently declare this range. A range describes patch applicability,
**not evidence that the vendor implementation outside the range is fixed**.

- Gates read distribution metadata without importing packages. Names are case-insensitive;
  hyphens, underscores and dots are equivalent.
- Supported operators: `>=`, `>`, `<=`, `<`, `==`, `!=`. Bounds must be dotted integers.
  Numeric release tuples are zero-padded, so `2.0 == 2.0.0`. Installed rc/dev/post/local
  suffixes do not affect ordering: `2.0rc1` and `2.0+vendor` count as `2.0`.
  This is not full PEP 440; epochs, wildcards and `~=` are unsupported.
- Missing metadata leaves target resolution/capability checks in charge; it is not a verified match.
  Unrecognized installed versions block. Reports retain the declaration in `version_gates` and
  explain blocked gates in the skipped record's `detail`. On first application, if every patch
  on a target is excluded, the engine does not resolve a potentially removed upstream symbol.
- `MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES=1` (also `true` or `*`) bypasses all version gates;
  `=transformer_engine,transformers` bypasses only those packages; `0/false/off` bypasses none.
  ONLY/DISABLE, companion requirements and source/capability probes still apply.
  Set switches before activation. Changing them does not undo existing wrappers or rerun hooks;
  use a fresh process, or unapply/install for this package's reversible changes.

Source markers are build fingerprints, not correctness proofs. Probing does not execute modules:
missing packages/files return False, unsupported layouts return None. Current callers keep their
workarounds on None and decline on False; even a formatting change may remove a marker, so upgrades
still require original import and numerical regression tests. The MoE topk probe uses deterministic
input and checks the result without consuming training RNG. To verify an upstream fix, disable the
patch and rerun its original regression; IGNORE_VERSION_GATES instead trials an out-of-range patch.
