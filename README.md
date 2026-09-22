# megatron-musa-patch

The primary goal is to run **unmodified Megatron-LM / Megatron-Core seamlessly on MUSA**: upstream unit tests and training scripts should run as supplied, and frameworks such as **ms-swift should call the same Megatron-Core interfaces without distinguishing MUSA from CUDA**.

This contract covers the interfaces Megatron-Core already provides. It is deliberately **not** an absolute "upper frameworks must never change" rule: when a model architecture is too new for Megatron-LM to track, ms-swift or Megatron-Bridge may ship the support first — and then the MUSA adaptation for that feature belongs in the corresponding upper framework, which is the more reasonable boundary.

This repository owns the runtime adaptation for **Megatron-Core's existing contract surface**. Installing it in a supported MUSA environment should be sufficient for that surface; callers should not need a MUSA fork, a special import, a replacement model class, or a device-specific branch. This is the acceptance target, not a claim that every upstream test and training configuration already passes. Adaptations for capabilities an upper framework implements ahead of Megatron-Core (new architectures, new kernels) belong to that framework and are out of scope here.

## Compatibility contract and current coverage

| Surface | Acceptance target |
|---|---|
| Megatron-LM unit tests | Run the original tests and assertions, including their CUDA-spelled APIs and `nccl` configuration. Fix compatibility in this package; do not rewrite or skip tests to hide a MUSA failure. |
| Megatron-LM training | Run original Python and shell training scripts without source edits or required MUSA launchers. Preserve model, optimizer, distributed and checkpoint semantics. |
| Megatron-Core callers, including ms-swift | Preserve import paths, public signatures, configuration objects, outputs and state-dict contracts. Core compatibility must work with a `megatron-core` wheel, without depending on `megatron.training` being present. |

Installing MUSA dependencies and supplying existing launcher inputs (data paths, output directories, device count and a resource-appropriate model size) are environment setup. Within the contract surface above, requiring callers to replace `cuda` with `musa`, `nccl` with `mccl`, add `import megatron_musa_patch`, or disable a requested feature is a compatibility gap. Such workarounds may help diagnosis, but do not satisfy the unchanged-caller target. Upper-framework code that wires Megatron to a brand-new model architecture is a different matter: its device adaptation lives in that framework by design.

The current implementation prioritizes correctness and uses conservative fallbacks where needed: PyTorch normalization, synchronous DP reduction and in-process checkpoint bucket writes. It also has targeted TE/FP8 and fused-RoPE checks. These checks and the examples cover specific paths; they do not establish full upstream-suite coverage, unchanged-script coverage, or end-to-end ms-swift compatibility. Record remaining failures, skips and required overrides explicitly. Performance work follows correctness and must retain the same caller contract; fallback trade-offs and removal conditions remain reviewable in the patch ledger.

For development, read [CONTRIBUTING.md](CONTRIBUTING.md). Coding agents should start with [AGENTS.md](AGENTS.md), which defines the working procedure and acceptance evidence. [中文文档](README_zh.md).

## Quick start

```bash
git clone https://github.com/kiscad/megatron-musa-patch
cd megatron-musa-patch && git checkout v0.16.1-dev

# The MUSA container already provides torch/torch_musa pinned to each other.
# A plain `pip install .` resolves the torchada dependency and would pull the
# latest upstream torch (e.g. 2.11), replacing the vendor-pinned build and
# breaking the MUSA stack. Install without dependency resolution:
pip install --no-deps torchada==0.1.86
pip install --no-deps .

# Alternative if you want the resolver to run: constrain the vendor stack so
# pip cannot upgrade it (adjust the pin to your container's torch build):
#   printf 'torch==2.7.1\n' > constraints-musa.txt
#   pip install -c constraints-musa.txt .

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

These are diagnostic examples with selected arguments, not proof that every original upstream launcher works unchanged. `run_pretrain_smoke.sh` defaults to a unique temporary directory and prints its path. An explicit `OUTPUT_DIR` must be new or empty; existing outputs are never deleted, and relative paths resolve from the calling directory. The step-by-step upstream-test, training and framework acceptance procedure is in [AGENTS.md](AGENTS.md).

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

Option b) is the typical setup when [Megatron-Core](https://pypi.org/project/megatron-core/) is consumed through an upper-level framework — for example **ms-swift**'s Megatron training (`megatron-core>=0.16`) or **Megatron Bridge 0.1–0.3** (Megatron-Core 0.14–0.16). Only the `megatron.core` / `torch` patches apply there; the `megatron.training` / `megatron.legacy` patches are recorded as `skipped`, which is expected. **Megatron Bridge ≥ 0.4 needs Megatron-Core 0.17+**; this branch targets 0.19, so the 0.17/0.18 line falls between the two supported ranges.

Megatron is deliberately not a pip dependency here: the wheel ships only `megatron/core/`, there is no `megatron-lm` on PyPI, and the real workflow wants a checkout pinned to a tag. Supported upstream range: `>=0.19,<0.20` — outside it you get a warning (an error with `MEGATRON_MUSA_PATCH_STRICT=1`). The 0.14–0.16 line is served by the `v0.16.1-dev` branch.

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
is served by apex's kernels on MUSA (see the rope entries in the [patch catalog](docs/PATCHES.md)), and the fused
`sbhd`/`thd` results are checked against Megatron's unfused reference by the
opt-in hardware test in `tests/test_rope.py`. Keep the flag for a PyTorch path —
`rotary_interleaved` models and packed sequences under context parallel fall back
to it automatically, with a warning.

## What gets changed

On a MUSA PyTorch build `torch.cuda` is a dead shell (no `_cuda_*` bindings) and Megatron references it everywhere. [torchada](https://pypi.org/project/torchada/) — Moore Threads' own CUDA→MUSA adapter — does the mechanical translation; this package adds five Megatron-specific overrides on top of it plus roughly forty Megatron patches: torch.cuda / torch.distributed shims, LayerNorm/RMSNorm and TE norm-linear fallbacks, TE attention capability dispatch and FP8 model init, MoE topk/permutation/grouped-GEMM demotions, fused RoPE via the MT apex fork, the GDN chunked gated delta rule on torch-kernels' TileLang kernels, checkpoint writer/barrier hardening, and training-argument compatibility.

The per-patch catalog — one row per patch with its target and root cause, plus kernel-stack version bindings and retired patches — lives in [docs/PATCHES.md](docs/PATCHES.md). The live, machine-readable list — `id`, `rationale`, `strategy`, `upstream`, `remove_when` — is `megatron_musa_patch.report()`.

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
* DP-overlap flags (`--overlap-grad-reduce` / `--overlap-param-gather`) are disabled by the fused-wgrad/DDP integration policy, which may reduce throughput. `MEGATRON_MUSA_PATCH_DP_OVERLAP=1` restores upstream behaviour for testing (legacy spelling `MEGATRON_MUSA_PATCH_TP_OVERLAP` still works). This does not touch `tp_comm_overlap`.
* Bare `--profile` automatically enables `--use-pytorch-profiler` and warns on rank 0. Skip `megatron.training.profile.pytorch` with `MEGATRON_MUSA_PATCH_DISABLE` to restore upstream profiler selection.
* Rope fusion is a MUSA kernel choice, not upstream's: the fused kernels come from Moore Threads' apex, and only the combinations apex implements are fused. `rotary_interleaved` models and packed sequences under context parallel are demoted to Megatron's unfused rotary embedding with a one-time warning — correct, slower, and no longer fatal. `MEGATRON_MUSA_PATCH_ROPE_FUSION=0` declines the fallback entirely, which leaves upstream's `apply_rope_fusion is not available` error in place (then pass `--no-rope-fusion`).
* The chunked gated delta rule (the GDN layers of Qwen3.5/Qwen3-Next-style hybrid models) is a kernel choice too: flash-linear-attention's Triton kernels stay the correctness reference and the fallback, while calls inside torch-kernels' audited envelope dispatch to its TileLang kernels — 4.8–6.0x per operator call at 4K sequence, 2.0–9.5% lower step time in 8-card ms-swift Qwen3.5 training. The kernels JIT-compile on first use, so pre-warm with `examples/warm_gdn_tilelang.py` before multi-rank runs; the TileLang stack is version-bound to the torch_musa build and musa_toolkits. Measured numbers, the pre-warm procedure and the version matrix are in [docs/PATCHES.md](docs/PATCHES.md). `MEGATRON_MUSA_PATCH_GDN_TILELANG=0` restores FLA everywhere.

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
| `MEGATRON_MUSA_PATCH_TE_NORM` | `0` | Set `1` to keep native TE standalone LayerNorm/RMSNorm for upgrade validation. |
| `MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES` | *(empty)* | `1`/`true`/`*` bypasses all version gates; a comma-separated distribution list bypasses only those gates. Capability probes and patch selection still apply. |
| `MEGATRON_MUSA_PATCH_ROPE_FUSION` | `1` | `0` declines the apex fused-RoPE fallback, so upstream keeps reporting `apply_rope_fusion` as unavailable. |
| `MEGATRON_MUSA_PATCH_GDN_TILELANG` | `1` | `0` declines the torch-kernels TileLang dispatch of the chunked gated delta rule, keeping flash-linear-attention's kernels everywhere. |
| `MEGATRON_MUSA_PATCH_ATTN_BACKEND` | `auto` | Attention kernel order for the shapes MuDNN's flash backward rejects: `auto` (native flash inside its measured backward window, then mate's TileLang flash, then TE's unfused backend), `mudnn` (native flash whenever the forward accepts it), `mate` (prefer the TileLang kernels), `unfused` (reference backend). |
| `MEGATRON_MUSA_PATCH_JIT_WARMUP` | `0` | `1` keeps upstream's JIT warm-up. |
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
* **Training still cannot find CUDA** — check activation with `python -c "import megatron_musa_patch as m, json; print(json.dumps(m.report(), indent=2))"`. All `pending` usually means the targets have not been imported. An empty report means no patches were registered, for example because `MEGATRON_MUSA_PATCH=0` is set. Check entry-point metadata separately: `python -c "from importlib.metadata import entry_points; print(entry_points(group='torch.backends'))"`.
* **A patch made things worse** — bisect with `MEGATRON_MUSA_PATCH_DISABLE=<id>`, or switch everything off with `MEGATRON_MUSA_PATCH=0`. `MEGATRON_MUSA_PATCH_DEBUG=1` logs every patch as it lands.

## Versioning

The package version tracks the Megatron-LM release it patches: `0.16.1.dev0` on this branch (`v0.16.1-dev`), `0.16.1` for a release cut from it, `0.16.1.post1` for a fix, `0.16.2.dev0` on the next upstream's new branch.

## Verified against

The recorded reference environment below is not an exhaustive test matrix. The declared upstream range (`>=0.19,<0.20`) is a version guard, not evidence that every version, feature or framework has passed. Report actual revisions, commands, pass/fail/skip counts and untested paths for each validation run.

Python 3.10 · PyTorch 2.7.1a0 (MUSA build) · torch_musa 2.7.1 · torchada 0.1.86 · Megatron-LM `core_v0.16.1` (`megatron-core` 0.16.1) · MT-TransformerEngine 2.0.0 · apex (MT fork, fused RoPE) · MTT S5000 · tilelang_musa 0.1.12+musa.2.git1feb38c3 · torch-kernels 0.1.0 · musa_toolkits 4.3.7.

The TileLang stack (tilelang-musa and torch-kernels' tilelang operators) is version-bound to the torch_musa build and musa_toolkits — the verified matrix and upgrade rules are in [docs/PATCHES.md](docs/PATCHES.md#tilelang-stack-version-binding).

The two early TE hooks are the only exceptions to Megatron-triggered activation;
registration remains free of accelerator imports. The device layer clears only
Transformers CUDA/BF16/FP16/TF32 and FlashAttention probe caches after activation,
rollback and unapply. Unrelated package caches are preserved. The safe-utils
bridge covers ordinary imports; live reload of vendor TE modules is unsupported.
Start a fresh process after changing TE or patches. The RoPE dispatcher also
selects Megatron's unfused branch for interleaved calls through native Core TE
wrappers when their TE < 2.3 version guard would reject the call.

### Declarative version gates

Reports include each patch's `version_gates` and skip reason. Missing metadata
keeps target/capability checks in charge. The three TE norm patches declare
`transformer_engine >=2.0,<2.1`; this is applicability, not proof that other
versions are fixed. The override switch above bypasses only version gates.
See the [contributor guide](CONTRIBUTING.md#version-gates) for numeric comparison
rules and source/capability probes. Disable the patch and rerun its original
regression before concluding a workaround can be removed.
