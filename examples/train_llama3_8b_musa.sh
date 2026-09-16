#!/usr/bin/env bash
# MUSA adaptation of upstream Megatron-LM/examples/llama/train_llama3_8b_h100_fp8.sh.
#
# Changes vs upstream (each is required or strongly recommended on MUSA,
# see README.md "Trade-offs worth knowing"):
#   * DTYPE=fp8 (default) -- via MT-TransformerEngine; RMSNorm only (MT-TE has no
#                            working LayerNorm kernels: nvte_layernorm_fwd/bwd are
#                            commented out, LayerNorm asserts in allocateSpace).
#                            DTYPE=bf16 selects the pure-PyTorch local impl below.
#   * bf16: --transformer-impl local -- the pure-PyTorch local spec, the only
#                            non-TE path this patch set validates; do NOT pass
#                            "--attention-backend local" explicitly (upstream
#                            assertion: args.spec is None at validation on
#                            core_r0.16.x).
#   * no --attention-backend fused -- that is TE's FlashAttention path,
#                            unverified on MUSA; each branch keeps Megatron's
#                            default backend for its spec.
#   * no --cross-entropy-loss-fusion / --profile -- torch.jit fused CE and the
#                            profiler (MUPTI) are unverified on MUSA; drop them
#                            for the first bring-up, re-enable one at a time.
#   * TP_SIZE=2 (not 1)   -- 8B bf16 does not fit one 48GB S5000 (weights 16G +
#                            grads 16G + sharded optimizer ~12G + activations).
#                            TP=2 keeps DP=4 for optimizer sharding and is the
#                            lowest-memory point; it can still OOM -- add
#                            --recompute-activations (already on below) or
#                            shrink first if so.
#   * --ckpt-format torch    -- torch_dist is unvalidated on MUSA; see README
#                            "Trade-offs worth knowing".
#
# Mock data is the default (TOKENIZER_ARG=DATA_ARG=MOCK), exactly like upstream.
#
# Usage:
#   MEGATRON_LM_PATH=/path/to/Megatron-LM PYTHON=/path/to/venv/bin/python \
#       bash train_llama3_8b_musa.sh [CKPT_DIR] [TB_DIR] [TOKENIZER] [DATA_PREFIX]
#
# Short bring-up run (env overrides, no editing needed):
#   TRAIN_SAMPLES=20 LR_WARMUP_SAMPLES=2 EVAL_ITERS=0 EVAL_INTERVAL=1000 \
#       SAVE_INTERVAL=1000 GPUS_PER_NODE=2 NUM_LAYERS=4 GLOBAL_BATCH_SIZE=2 \
#       bash train_llama3_8b_musa.sh
#
# About --train-samples: upstream's 1953125000 is the 16T-token long-run
# benchmark schedule. Mock training never consumes it, but the dataset index
# arrays scale with it (~20+ GB RAM, minutes of CPU build) -- and while the
# building rank is busy, the other ranks wait in a torch.distributed.barrier
# whose MCCL kernel gets killed by the MUSA kernel watchdog after ~3 minutes
# ("MUSA error: the launch timed out and was terminated"). The default here is
# 1% of it (still 1.2M iterations at GBS=16); restore the upstream numbers with
#   TRAIN_SAMPLES=1953125000 LR_WARMUP_SAMPLES=3906252
# which also needs a cache prebuild (single-rank run with the same data args)
# plus --dataloader-fast-cache-load to skip the barrier.
set -euo pipefail

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:?set MEGATRON_LM_PATH to a Megatron-LM checkout (core_r0.16.x)}"
PYTHON="${PYTHON:?set PYTHON to the venv interpreter that has torch_musa + torchada}"
# Use the venv's torchrun: a bare `torchrun` from PATH may hit another env.
TORCHRUN="${TORCHRUN:-$PYTHON -m torch.distributed.run}"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_musa"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_musa"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}      # Data prefix, or "MOCK"

mkdir -p "$(dirname "$CHECKPOINT_PATH")" "$(dirname "$TENSORBOARD_LOGS_PATH")"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

TP_SIZE=${TP_SIZE:-2}
CP_SIZE=1
PP_SIZE=1
DTYPE=${DTYPE:-fp8}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-19531250}      # 1% of upstream's 16T-token benchmark schedule
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-$((TRAIN_SAMPLES / 1000))}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
NUM_LAYERS=${NUM_LAYERS:-32}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
MAX_POSITION_EMBEDDINGS=8192

DATA_CACHE_PATH="${PWD}/benchmark_cache_llama3_8b_musa"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank 0
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "$NUM_LAYERS"
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    # H100 script's FP8 recipe via MT-TransformerEngine (its default impl).
    # RMSNorm only: MT-TE has no working LayerNorm kernels (see header).
    DTYPE_ARGS+=(
        --normalization RMSNorm
        --fp8-format hybrid
        --fp8-amax-history-len 1024
        --fp8-amax-compute-algo max
        --fp8-param-gather
        --ckpt-format torch    # torch_dist chokes on TE Float8Tensor shards:
                               # _mcore_to_dcp_compatible_tensor views the local
                               # fp8 payload with the global bf16 shape
    )
else
    DTYPE_ARGS+=(
        --transformer-impl local        # MUSA: local spec (pure PyTorch path)
        --apply-layernorm-1p
        --recompute-activations         # MUSA: 48GB cards, trade some speed for memory
        --ckpt-format torch             # legacy sharded writer: the torch_dist
                                        # backend (torch.distributed.checkpoint)
                                        # is unvalidated on MUSA; this package no
                                        # longer rewrites formats silently
    )
fi

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --bf16
    --grad-reduce-in-bf16
    --train-samples "$TRAIN_SAMPLES"
    --lr-decay-samples "$TRAIN_SAMPLES"
    --lr-warmup-samples "$LR_WARMUP_SAMPLES"
    --lr 0.00015
    --min-lr 0.00001
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --manual-gc
    --empty-unused-memory-level 1
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --sequence-parallel
)

# NOTE: no --overlap-grad-reduce / --overlap-param-gather here. MUSA/MCCL
# cannot hide collectives behind compute yet; the patch package would ignore
# the flags anyway, so we keep the argument list honest.
DDP_ARGS=(
    --use-distributed-optimizer
)

DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 128256
        --data-cache-path "$DATA_CACHE_PATH"
        --tiktoken-pattern v2
        --split "99,1,0"
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
else
    DATA_ARGS_LIST+=(
        --data-path "$DATA_ARG"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_ARG"
        --data-cache-path "$DATA_CACHE_PATH"
        --split "99,1,0"
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
        --vocab-size 128256
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters "${EVAL_ITERS:-32}"
    --eval-interval "${EVAL_INTERVAL:-100}"
    --save-interval "${SAVE_INTERVAL:-1000}"
    --log-throughput
    "${EXIT_DURATION:+--exit-duration-in-mins ${EXIT_DURATION}}"
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

cd "$MEGATRON_LM_PATH"
$TORCHRUN ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DDP_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
