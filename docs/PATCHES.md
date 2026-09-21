# Patch catalog

One page describing every patch this package can install: what it targets,
why it exists, and what to know operationally. The live, machine-readable
source — `id`, `rationale`, `strategy`, `upstream`, `remove_when`,
`requires`, `version_gates` — is `megatron_musa_patch.report()`; this page
is its human-readable companion and may lag the code by one change.

## Catalog

| id | target | root cause (short) |
|---|---|---|
| `python.typing.override.backport` | `typing.override` (Megatron hook) | Megatron 0.19 requires Python ≥3.12 and imports `typing.override` at module scope; the MUSA wheels are CPython 3.10 builds, so `megatron.training` fails to import. Publish the `typing_extensions` implementation (PEP 698 marker only, no runtime effect) |
| `torch.cuda.compat-layer` | `torch.cuda` | no CUDA bindings on the MUSA build; torchada + four overrides: live `is_available()` probe, `Tensor.type()` CUDA names, `CUDAGraph` alias, `.musa()` transfer fix for tensor subclasses (e.g. TE `Float8Tensor`) |
| `torch.cuda.device-capability.nvidia-scale` | `torch.cuda.get_device_capability` | Megatron compares capability against NVIDIA thresholds (≥8 grouped-GEMM gate); report a synthetic `8.3` |
| `megatron.training.get-device-arch-version.nvidia-scale` | `megatron.training.utils:get_device_arch_version` | same comparison via `device_properties().major` |
| `torch.distributed.clean-teardown` | `atexit` | runs that exit with live MCCL groups showed watchdog errors; best-effort `destroy_process_group()` at exit |
| `megatron.fusions.fused-layer-norm.pure-torch` | `fused_layer_norm:FusedLayerNorm` | apex imports but its CUDA extension is unavailable; config-authoritative PyTorch LayerNorm/RMSNorm fallback |
| `megatron.fusions.fused-layer-norm.have-apex-flag` | `fused_layer_norm:HAVE_FUSED_LAYER_NORM` | consumers branch on the flag; keep it truthful once the class is replaced |
| `megatron.fusions.persist-layer-norm.disable` | `fused_layer_norm:HAVE_PERSIST_LAYER_NORM` | the fallback never runs apex's persistent kernel |
| `megatron.transformer-block.layer-norm.impl-local` | `transformer_block:LayerNormImpl` | TE's norm op aborted (`allocateSpace`) on the affected MUSA stack even for a `local` spec |
| `transformer_engine.layer-norm-linear.native-unfused` | `transformer_engine:LayerNormLinear` | Direct TE models (te.pytorch.TransformerLayer in the megatron-FSDP suite) normalize via module/_common.py:apply_normalization, which aborts in the MUSA allocateSpace assertion. Subclass the TE module and reroute only the eligible plain path (no FP8, no UB overlap, no offload, tp_size 1, no layernorm-output return) through functional norm + F.linear, keeping parameters, isinstance and checkpoint contracts. |
| `transformer_engine.layer-norm-mlp.native-unfused` | `transformer_engine:LayerNormMLP` | Same fused-norm failure in the MLP block; the eligible plain path is functional norm -> fc1 -> gelu(tanh)/relu -> fc2 on plain PyTorch ops. Gated activations and non-plain configurations keep the original forward. |
| `megatron.te.layer-norm-linear.unfused` | `transformer_engine:TELayerNormColumnParallelLinear` | For LayerNorm only, PyTorch norm + TE Linear bypasses the fused norm `allocateSpace` failure in QKV/FC1 while retaining FP8 Linear and norm checkpoint names. The norm step follows TE's op contract — it computes in the autocast dtype or the norm-weight dtype and casts the *input* to it, absorbing mixed-dtype activations (e.g. the fp32 learned position-embedding sum entering a bf16 model) exactly like the native kernel |
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
| `megatron.te.norm.unfused-musa` | `transformer_engine:TENorm` | TE's standalone LayerNorm/RMSNorm aborts in MUSA `allocateSpace`; subclass the TE modules and replace only forward (functional norm under TE's dtype contract: compute in the autocast dtype or the parameter dtype, casting the input — never the reverse), keeping isinstance/sharding contracts |
| `transformer_engine.dot-product-attention.capability-dispatch` | `transformer_engine:DotProductAttention.forward` (in place) | Direct TE models (te.pytorch.TransformerLayer, 56 mfsdp cases) run attention through TE's native class, which the MUSA port pins to flash: fp32-activation models die on 'FlashAttention only supports FP16 and BF16'. forward is wrapped in place on the original class (MT-TE's own __init__ resolves super() through its module global, so class replacement breaks construction); eligible flash-unsupported dense calls reroute to TE's UnfusedDotProductAttention with the same mask semantics as the Megatron-wrapper dispatch. |
| `megatron.te.attention.capability-dispatch` | `transformer_engine:TEDotProductAttention.forward` | Capability dispatch that implements no attention math. The native MT-TE flash path serves FP16/BF16 inputs with head_dim 64–192 and no dropout — and, when the call may reach backward (training, checkpoint recompute, grad-enabled with grad-requiring inputs), only equal qk/v head dims inside the measured MuDNN backward window {64,80,96,112,128,160}; its backward kernel fails for 144 and 168–192 and for every measured mixed qk/v pair (MLA's 192/128). The shapes the backward rejects run mate's TileLang flash (`mate.flash_attn_varlen_func`) when mate is installed and the call is inside its verified envelope, else TE's own UnfusedDotProductAttention backend; non-padding mask types drop the mask tensor exactly like the flash path. Padded THD is sliced into per-sequence vendor calls because native THD drops `cu_seqlens` and yields NaN. `MEGATRON_MUSA_PATCH_ATTN_BACKEND=auto|mudnn|mate|unfused` selects the kernel order; mate's TileLang kernels JIT-compile per head-dim class on first use (pre-warm ~/.tilelang before multi-rank runs). Declines entirely when the TE hard-coded-flash marker is gone (source probe). CP>1/FP8 DPA/special softmax/windows/max-logit remain upstream. |
| `megatron.te.quantized-model-init.delayed-compat` | `transformer_engine.pytorch.quantized_model_init` | Megatron-triggered, ownership-tracked TE alias delegates only explicit DelayedScaling to native fp8_model_init; preserves high-precision initialization and nested contexts. Other enabled recipes are rejected; native attributes are never overwritten. |
| `megatron.te.factory-shim.torchscript-compat` | `transformer_engine:musa/patch_after_import_torch` | MT-TE rebinds torch factory functions (tensor/zeros/ones/empty/rand/arange/empty_like) to untyped device-translation wrappers, so any eager `torch.jit.script` over a factory call fails; each scripting call temporarily registers the seven known vendor wrappers in TorchScript’s ATen builtin table, leaving eager factory bindings unchanged during compilation and preserving `device='cuda'` translation intact. MUSA-fork-only; needed before Megatron import. Compiled graphs retain ATen device semantics; CUDA string literals are not translated inside them. Skips when the installed TE has no factory wrappers (source probe). Undo preserves subsequent third-party replacements. |
| `megatron.te.utils-module.safe-seed` | `transformer_engine:musa/pytorch/utils` | The vendor module iterates sys.modules with lazy-module getattrs at import time, which crashes the whole TE import with transformers 5.x ('dictionary changed size during iteration'); a replica without the fragile loop is seeded before the vendor body runs (same pattern as the musa_patch.mem_utils shim). Skips when the vendor loop is gone (source probe). |
| `megatron.embeddings.fused-rope.apex` | `rope_utils:fused_apply_rotary_pos_emb` | Megatron gated its fused `sbhd` kernel on a TE import (`…attention.rope`) the MUSA TE tree does not have, so `apply_rope_fusion` — argparse's default — was rejected before the first step; bind apex's equivalent kernel |
| `megatron.embeddings.fused-rope-thd.apex` | `rope_utils:fused_apply_rotary_pos_emb_thd` | the packed (`thd`) half of the same missing import; apex's padded-layout kernel, `cp_size=1` |
| `megatron.embeddings.rope-fusion.unfused-fallback` | `rope_utils:apply_rotary_pos_emb` | apex has no interleaved and no context-parallel variant; demote exactly those calls to upstream's unfused branch (warned once) rather than aborting mid-step |
| `megatron.ssm.gated-delta-rule.tilelang` | `ssm.gated_delta_net:chunk_gated_delta_rule` | GatedDeltaNet runs flash-linear-attention's Triton chunked gated delta rule — the slowest kernels of Qwen3.5-style hybrid models on MUSA; calls inside torch-kernels' audited envelope dispatch to its TileLang implementation, everything else falls back to FLA unchanged |
| `mcore_bridge.ssm.gated-delta-rule.tilelang` | `mcore_bridge.model.modules.gated_delta_net:chunk_gated_delta_rule` | the mcore-bridge forward ms-swift executes re-imports fla's symbol into its own namespace, so Megatron's binding alone never reaches it; same dispatcher, independently selectable |
| `megatron.legacy.fused-kernels.load.noop` | `legacy.fused_kernels:load` | the loader probes `nvcc`/defines an extension builder — no MUSA build route |
| `megatron.training.set-jit-fusion-options.noop` (+ `initialize` alias) | `set_jit_fusion_options` | startup CUDA warm-up/compile path needs validation on MUSA; skipped until proven |
| `megatron.dist-ckpt.musa-cpu-staging` | DCP filesystem device selector (Megatron hook) | use the actual MUSA stream device under CUDA API emulation so CPU staging is not skipped; preserve upstream asynchronous copies, synchronization and checkpoint sharding |
| `megatron.training.overlap-flags.noop` | `training.arguments:validate_args` | the observed TE fused-wgrad/DDP integration left `param.grad` None, breaking Megatron's overlap backward hook; DP-overlap flags are forced off with a warning |
| `megatron.training.profile.pytorch` | `training.arguments:validate_args` | bare `--profile` enters `cudaProfilerStart/Stop`/NVTX, which the MUSA runtime does not provide; `--use-pytorch-profiler` is enabled instead |
| `megatron.training.start-time.integer-microseconds` | `training:torch` | Module-local startup timestamp MIN uses integer microseconds on MCCL |
| `megatron.training.checkpoint.host-barrier-proxy` / `host-barrier-context` | `checkpointing:torch` / `save_checkpoint` | Create a world Gloo group before checkpoint I/O; only save barriers use it |
| `megatron.serialization.signal-member-globals` | `safe_globals:SAFE_GLOBALS` | Explicit signal member aliases allow Python 3.10 weights-only argument loading |

Every patch record carries a `rationale` (observed root cause, version-specific and honest about what is *not* proven), a `strategy` (what the replacement does and preserves), the upstream file it adapts, and a `remove_when` review condition tied to specific tests. `uninstall()`/`unapply()` restore owned changes; hooks with external side effects (e.g. torchada's import-time mutations) are documented as irreversible.

**Retired:** `megatron.training.ckpt-format.no-torch-dist` (which silently rewrote `--ckpt-format torch_dist` to `torch`) is gone. The rewrite changed save/load semantics — `torch` is Megatron's legacy format, not a drop-in backend for `torch.distributed.checkpoint`, and the rewrite broke async save and FSDP configurations. Pass `--ckpt-format torch` explicitly if you want the legacy writer.

## Kernel-stack patches

These patches choose a MUSA kernel instead of repairing a broken call. Their
measured numbers, pre-warm requirements and version bindings live here so the
README stays a summary.

### Chunked gated delta rule → torch-kernels TileLang

**Patches:** `megatron.ssm.gated-delta-rule.tilelang`,
`mcore_bridge.ssm.gated-delta-rule.tilelang` · **Switch:**
`MEGATRON_MUSA_PATCH_GDN_TILELANG` (default `1`)

The GDN layers of Qwen3.5 / Qwen3-Next-style hybrid models compute the
chunked gated delta rule. flash-linear-attention's Triton kernels remain the
correctness reference and the fallback; calls inside torch-kernels' audited
TileLang envelope (MUSA device, bf16 q/k/v/beta, fp32 decay, the operator's
own `gdn_dense_supported` / `gdn_varlen_supported` shape guards, no
fla-only keywords) dispatch to torch-kernels' TileLang implementation.
Dispatch is decided per call from tensor metadata (packed calls read
`cu_seqlens` once to validate the per-sequence two-chunk floor) and never
caches a verdict. Both bindings must be patched: the mcore-bridge forward
ms-swift executes re-imports fla's symbol into its own namespace.

Measured on MTT S5000 (B=1, DK=DV=128, l2-normalized q/k, fwd+bwd median of
20 reps after 3 warm-ups):

| shape | fla 0.5.2 | torch-kernels TileLang | speedup |
|---|---|---|---|
| S=4096, H=8 | 5.16 ms | 1.07 ms | 4.8x |
| S=4096, H=32 | 13.92 ms | 2.31 ms | 6.0x |

Numerics: fla and TileLang agree with Megatron's fp32 reference within
relL2 ≈ 4e-3 each (fla 3.9e-3, TileLang 3.6e-3); gradients (dq/dk/dv/dg/dβ)
agree between the two within relL2 ≤ 6.6e-3.

End-to-end 8-card ms-swift Qwen3.5 training (30 iterations, long samples,
steady-state mean of iterations 4–30, identical data and seed):

| parallel layout | fla | TileLang | step time |
|---|---|---|---|
| TP=4 / PP=1 | 4.93 s/it | 4.67 s/it | 5.3% lower |
| TP=1 / PP=4 (DP=2) | 2.74 s/it | 2.48 s/it | 9.5% lower |
| TP=1 / PP=2 (DP=4) | 1.89 s/it | 1.85 s/it | 2.0% lower |

Operational notes:

* The TileLang kernels JIT-compile on first use per head count and
  dense/unpadded specialization (minutes, cached in `~/.tilelang`).
  **Pre-warm the cache once with
  `examples/warm_gdn_tilelang.py --heads <per-rank-heads>` before a
  multi-rank run** — every rank compiling the same kernels into the shared
  cache concurrently has crashed runs (device error / SIGABRT). With a warm
  cache the full 8-card training runs clean.
* FLA's own first encounter with new head-count/shape buckets also compiles
  (an H=32 first iteration measured ~13 min of Triton compilation); the
  TileLang pre-warm shifts that cost off the training run.
* `MEGATRON_MUSA_PATCH_GDN_TILELANG=0` declines both patches and restores
  FLA everywhere.

#### TileLang stack version binding

The TileLang stack in this environment is a matched set: **tilelang-musa and
the tilelang operator package (torch-kernels) are built against a specific
torch_musa and musa_toolkits release, and their versions are bound to those
of the MUSA stack.** The verified combination:

| component | verified version |
|---|---|
| PyTorch | 2.7.1a0 (MUSA build) |
| torch_musa | 2.7.1 |
| musa_toolkits | 4.3.7 |
| tilelang_musa | 0.1.12+musa.2.git1feb38c3 |
| torch-kernels (tilelang operators) | 0.1.0 |

* Upgrade torch_musa or musa_toolkits only together with a matching
  tilelang-musa / torch-kernels build; never mix a tilelang-musa compiled
  against a different MUSA stack.
* The JIT kernel cache under `~/.tilelang` is keyed by the tilelang build id
  (e.g. `0.1.12_musa_2_git1feb38c3-x86_64`). After any change to the stack,
  remove the stale cache and re-run the pre-warm so kernels are recompiled
  against the new build.
