#!/usr/bin/env bash
# Training on the bot 8gpu class. Overrides: RUN_NAME, MODEL, NTOK, LEVELS, STEPS, NPROC, BATCH, EXTRA.
source "$(dirname "$0")/common.sh"
NPROC="${NPROC:-8}"
RUN_NAME="${RUN_NAME:-hier-vq-n10-v2048}"
MODEL="${MODEL:-hier_vq}"
STEPS="${STEPS:-300000}"
export OMP_NUM_THREADS=4
OVERRIDES=(run_name="$RUN_NAME" model="$MODEL" train.steps="$STEPS" train.batch_size="${BATCH:-256}"
           train.num_workers="${WORKERS:-10}" train.eval_every="${EVAL_EVERY:-10000}" train.ckpt_every="${CKPT_EVERY:-5000}")
[ -n "$NTOK" ] && OVERRIDES+=(model.num_tokens="$NTOK")
[ -n "$LEVELS" ] && OVERRIDES+=(model.quantizer.levels="$LEVELS")
[ -n "$EXTRA" ] && OVERRIDES+=(${EXTRA//;/ })
echo "[launch] ${OVERRIDES[*]}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
LAUNCH=(--num_machines 1 --num_processes "$NPROC" --mixed_precision bf16 --dynamo_backend no)
[ "$NPROC" -gt 1 ] && LAUNCH+=(--multi_gpu)     # accelerate refuses --multi_gpu with a single process
accelerate launch "${LAUNCH[@]}" \
  train.py "${OVERRIDES[@]}"
echo "[launch] done $(date -Is)"
