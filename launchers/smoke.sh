#!/usr/bin/env bash
# Short end-to-end check on the IB node (tmux): pytest, then 200 training steps on 2 GPUs with Comet enabled.
source "$(dirname "$0")/common.sh"
echo "[smoke] pytest"
python -m pytest -q
echo "[smoke] training 200 steps on GPUs ${SMOKE_GPUS:-1,2}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPUS:-1,2}" NPROC=2 STEPS=200 EVAL_EVERY=100 CKPT_EVERY=100 WORKERS=8 \
  RUN_NAME="${RUN_NAME:-hier-smoke}" bash --noprofile --norc "$(dirname "$0")/train.sh"
echo "[smoke] done $(date -Is)"
