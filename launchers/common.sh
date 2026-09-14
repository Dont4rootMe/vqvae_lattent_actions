# Shared environment for runs on the Sber cluster (sourced by every launcher).
BASE=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization
# submit.sh pins both repositories into the run directory and passes them in; a bare run uses the shared checkouts
REPO=${REPO:-$BASE/vqvae_lattent_actions}
ACTION_CHUNKS=${ACTION_CHUNKS:-$BASE/action_chunks}
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
# A low-priority job that the queue stopped and reran starts here again; a stale exit_code would read as finished.
rm -f "$RUN_DIR/exit_code"
trap 'rc=$?; echo "[launch] exit=$rc at $(date -Is)"; echo "$rc" > "$RUN_DIR/exit_code"; exit $rc' EXIT
# A stopped job must not read as a clean finish: on a signal the EXIT trap would otherwise record the status of the
# last completed command, usually 0.
trap 'exit 143' TERM
trap 'exit 130' INT
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
revision () { git -C "$1" rev-parse --short HEAD 2>/dev/null || grep -s "^$(basename "$1") " "$1/../REVISION" | cut -d' ' -f2 || echo n/a; }
echo "[launch] python=$(python -c 'import sys;print(sys.executable)') repo=$(revision "$REPO") action_chunks=$(revision "$ACTION_CHUNKS") code=$REPO"
cd "$REPO"
