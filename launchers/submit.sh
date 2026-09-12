#!/usr/bin/env bash
# Submit a training job to the bot queue (run on an EXP node from the directory holding CLOUD_USER_TOKEN).
# Usage: TAG=r09 RUN_NAME=hier-fsq-n10-v2048 NTOK=10 LEVELS='[8,8,8,4]' submit.sh
set -e -o pipefail
BASE=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization
SRC=$BASE/vqvae_lattent_actions/launchers
TAG="${TAG:?set TAG, e.g. r09}"
RUN_NAME="${RUN_NAME:-hier-fsq-n10-v2048}"
DIR=$BASE/runs/${TAG}_hier
cd /mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov
mkdir -p "$DIR"
cp "$SRC/common.sh" "$SRC/train.sh" "$DIR/"
rm -f "$DIR/exit_code"
ENVS="RUN_NAME=$RUN_NAME MODEL=${MODEL:-hier_fsq} NPROC=8 STEPS=${STEPS:-300000} WORKERS=${WORKERS:-10}"
[ -n "$NTOK" ] && ENVS="$ENVS NTOK=$NTOK"
[ -n "$LEVELS" ] && ENVS="$ENVS LEVELS=$LEVELS"
[ -n "$EXTRA" ] && ENVS="$ENVS EXTRA=$EXTRA"      # hydra overrides, ';'-separated; train.sh splits them"
CMD="cd $DIR && env $ENVS bash --noprofile --norc $DIR/train.sh"
echo "[submit] lerobot-research-${TAG}-hier: $CMD"
# the team 8gpu quota is often fully used, so wait for a team slot instead of being refused outright
bot submit -t 8gpu -H 48 -n "lerobot-research-${TAG}-hier" -c "$CMD" ${QUEUE_FLAGS:---team-wait} --json | tee "$DIR/submit.json"
