#!/usr/bin/env bash
# Submit a training job to the bot queue (run on an EXP node from the directory holding CLOUD_USER_TOKEN).
# Usage: TAG=r09 RUN_NAME=hier-fsq-n10-v2048 NTOK=10 LEVELS='[8,8,8,4]' submit.sh
set -e -o pipefail
BASE=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization
SRC=$BASE/vqvae_lattent_actions/launchers
TAG="${TAG:?set TAG, e.g. r09}"
RUN_NAME="${RUN_NAME:-hier-vq-n10-v2048}"
DIR=$BASE/runs/${TAG}_hier
cd /mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov
mkdir -p "$DIR"
cp "$SRC/common.sh" "$SRC/train.sh" "$DIR/"
rm -f "$DIR/exit_code"
# Pin the code. A low-priority job is stopped and rerun by the queue, possibly days later, and would otherwise import
# whatever the shared checkout holds at that moment. The copy is the committed HEAD, so refuse a dirty checkout, and
# refuse to swap the code under a run directory that already holds training output.
CODE="$DIR/code"
for repo in vqvae_lattent_actions action_chunks; do
  if [ -n "$(git -C "$BASE/$repo" status --porcelain --untracked-files=no)" ] && [ -z "$ALLOW_DIRTY" ]; then
    echo "[submit] $BASE/$repo has uncommitted changes the pinned copy would miss (ALLOW_DIRTY=1 overrides)"; exit 2
  fi
done
if [ -f "$CODE/REVISION" ] && [ -n "$(ls -A "$DIR/out" 2>/dev/null)" ] && [ -z "$REPIN" ]; then
  echo "[submit] $DIR has output from code $(tr '\n' ' ' < "$CODE/REVISION")- REPIN=1 replaces that code"; exit 2
fi
rm -rf "$CODE"
for repo in vqvae_lattent_actions action_chunks; do
  mkdir -p "$CODE/$repo"
  git -C "$BASE/$repo" archive HEAD | tar -x -C "$CODE/$repo"
  echo "$repo $(git -C "$BASE/$repo" rev-parse --short HEAD)" >> "$CODE/REVISION"
done
CLASS="${CLASS:-8gpu}"                            # 1gpu for probes, 8gpu for production
ENVS="RUN_NAME=$RUN_NAME MODEL=${MODEL:-hier_vq} NPROC=${NPROC:-8} STEPS=${STEPS:-300000} WORKERS=${WORKERS:-10}"
for var in EVAL_EVERY CKPT_EVERY BATCH; do
  eval "value=\${$var:-}"
  [ -n "$value" ] && ENVS="$ENVS $var=$value"
done
[ -n "$NTOK" ] && ENVS="$ENVS NTOK=$NTOK"
[ -n "$LEVELS" ] && ENVS="$ENVS LEVELS='$LEVELS'"
# Hydra overrides are ';'-separated. The value reaches the queue as part of a command string that a shell parses,
# so an unquoted ';' ends the command there and the overrides never reach training.
[ -n "$EXTRA" ] && ENVS="$ENVS EXTRA='$EXTRA'"
ENVS="$ENVS REPO=$CODE/vqvae_lattent_actions ACTION_CHUNKS=$CODE/action_chunks"
CMD="cd $DIR && env $ENVS bash --noprofile --norc $DIR/train.sh"
if [ -n "$LOW_PRIORITY" ]; then
  # Preemptible: the queue may stop the job and rerun this same command. The trainer then resumes from
  # out/<run>/checkpoints/latest.pt, so keep CKPT_EVERY small. The queue forbids combining this with --team-wait.
  QUEUE_FLAGS="--low-priority"
fi
case "${QUEUE_FLAGS:-}" in
  *--low-priority*--team-wait*|*--team-wait*--low-priority*) echo "[submit] --low-priority and --team-wait exclude each other"; exit 2;;
esac
echo "[submit] lerobot-research-${TAG}-hier (${QUEUE_FLAGS:---team-wait}): $CMD"
# normal jobs wait for a team slot instead of being refused when the team quota is full
bot submit -t "$CLASS" -H "${HOURS:-48}" -n "lerobot-research-${TAG}-hier" -c "$CMD" ${QUEUE_FLAGS:---team-wait} --json | tee "$DIR/submit.json"
