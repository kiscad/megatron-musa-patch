#!/usr/bin/env bash
# Run the upstream example with the previously local 16-layer / 1% schedule.
# CLI overrides live here; upstream model/training code remains untouched.
set -euo pipefail
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/ipfs/jingyuan/Megatron-LM-core_v0.19.0}"
export MUSA_TRAIN_PYTHON="${PYTHON:-python}"
export MUSA_NUM_LAYERS="${NUM_LAYERS:-16}"
export MUSA_TRAIN_SAMPLES="${TRAIN_SAMPLES:-19531250}"
export MUSA_LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-19492187}"
export MUSA_LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-39062}"
launcher_dir=$(mktemp -d)
trap 'rm -rf "$launcher_dir"' EXIT
cat > "$launcher_dir/torchrun" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
exec "$MUSA_TRAIN_PYTHON" -m torch.distributed.run "$@" \
    --num-layers "$MUSA_NUM_LAYERS" \
    --train-samples "$MUSA_TRAIN_SAMPLES" \
    --lr-decay-samples "$MUSA_LR_DECAY_SAMPLES" \
    --lr-warmup-samples "$MUSA_LR_WARMUP_SAMPLES"
LAUNCHER
chmod +x "$launcher_dir/torchrun"
cd "$MEGATRON_LM_PATH"
PATH="$launcher_dir:$PATH" bash examples/llama/train_llama3_8b_h100_fp8.sh "$@"
