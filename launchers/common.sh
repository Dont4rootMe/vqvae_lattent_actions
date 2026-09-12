# Shared environment for runs on the Sber cluster (sourced by every launcher).
BASE=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization
REPO=${REPO:-$BASE/vqvae_lattent_actions}
ACTION_CHUNKS=$BASE/action_chunks
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
trap 'rc=$?; echo "[launch] exit=$rc at $(date -Is)"; echo "$rc" > "$RUN_DIR/exit_code"; exit $rc' EXIT
echo "[launch] start $(date -Is) host=$(hostname) run_dir=$RUN_DIR"
export MAMBA_EXE=/mnt/virtual_ai0001071-01239_SR006-nfs2/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/mnt/virtual_ai0001071-01239_SR006-nfs2/micromamba
test -x "$MAMBA_EXE" || { echo "[launch] micromamba missing"; exit 1; }
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate ai_lerobot_qwen3vl_develop || { echo "[launch] env activation failed"; exit 1; }
set -e -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$ACTION_CHUNKS:$BASE/pylib:$BASE/pylib_comet"
export COMET_CONFIG=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/.comet.config   # key lives here, never in git
export VQLA_OUT="${VQLA_OUT:-$RUN_DIR/out}"
echo "[launch] python=$(python -c 'import sys;print(sys.executable)') repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a) action_chunks=$(git -C "$ACTION_CHUNKS" rev-parse --short HEAD 2>/dev/null || echo n/a)"
cd "$REPO"
