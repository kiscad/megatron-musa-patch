#!/usr/bin/env bash
# End-to-end bring-up check: a tiny GPT pretraining run on MUSA.
#
# It exercises the parts of Megatron that the patch set targets: argument
# validation, `_compile_dependencies` (dataset index builder + fused kernels),
# model construction, the local (non-TransformerEngine) layer spec, the
# checkpoint save path and the training loop itself.
#
# ROPE_FUSION=1 drops `--no-rope-fusion` and exercises the apex fused kernels
# instead (Megatron has no positive flag: fusion is the argparse default, and the
# integration tests already compare the kernels against the unfused reference).
#
# Usage:
#   MEGATRON_LM_PATH=/path/to/Megatron-LM NPUS=2 bash examples/run_pretrain_smoke.sh
#   MEGATRON_LM_PATH=/path/to/Megatron-LM NPUS=2 ROPE_FUSION=1 bash examples/run_pretrain_smoke.sh
set -euo pipefail

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/ipfs/jingyuan/Megatron-LM-core_v0.19.0}"
# Use the interpreter that has torch + torch_musa installed; a bare `torchrun`
# from PATH may resolve to a different environment.
PYTHON="${PYTHON:-python}"
NPUS="${NPUS:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/megatron-musa-smoke.XXXXXX")}"

ROPE_ARGS=(--no-rope-fusion)
if [ "${ROPE_FUSION:-0}" = "1" ]; then
    ROPE_ARGS=()
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# Never delete an existing run. Resolve relative paths before changing into
# the upstream checkout, so checkpoints land where the caller requested.
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: OUTPUT_DIR must be new or empty: $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
echo "Smoke output: $OUTPUT_DIR"

cd "${MEGATRON_LM_PATH}"

"${PYTHON}" -m torch.distributed.run --nproc_per_node="${NPUS}" --standalone \
    pretrain_gpt.py \
    --mock-data --tokenizer-type NullTokenizer --vocab-size 1024 \
    --seq-length 128 --max-position-embeddings 128 \
    --num-layers 2 --hidden-size 128 --ffn-hidden-size 256 --num-attention-heads 4 \
    --micro-batch-size 2 --global-batch-size "$((NPUS * 2))" \
    --tensor-model-parallel-size "${NPUS}" --pipeline-model-parallel-size 1 \
    --train-iters "${TRAIN_ITERS}" \
    --lr 1e-4 --min-lr 1e-5 --lr-decay-style constant \
    --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 \
    --bf16 --transformer-impl local \
    --no-masked-softmax-fusion --no-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion --no-bias-dropout-fusion "${ROPE_ARGS[@]}" \
    --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32 \
    --no-persist-layer-norm \
    --distributed-backend nccl \
    --log-interval 1 --eval-interval 1000 --eval-iters 0 \
    --save-interval 1000 \
    --save "${OUTPUT_DIR}" \
    --no-load-rng --no-save-rng \
    --no-save-optim \
    "$@"
