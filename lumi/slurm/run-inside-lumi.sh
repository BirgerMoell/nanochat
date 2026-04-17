#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run-inside-lumi.sh setup|first|pilot}"

LUMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$LUMI_DIR/.." && pwd)"
WORKDIR="${NANOCHAT_LUMI_WORKDIR:-$REPO_ROOT/lumi-work}"
BASE_DIR="$WORKDIR/cache"

load_lumi_ai_container() {
  if ! type module >/dev/null 2>&1; then
    # LUMI normally provides module through the login shell, but batch shells can
    # be sparse depending on user dotfiles.
    # shellcheck disable=SC1091
    source /etc/profile
  fi

  module purge
  module use /appl/local/laifs/modules
  module load lumi-aif-singularity-bindings

  if [[ -z "${SIF:-}" ]]; then
    SIF="$(ls -1d /appl/local/laifs/containers/lumi-multitorch-*/lumi-multitorch-full-*.sif | sort | tail -n 1)"
    export SIF
  fi

  if [[ ! -f "$SIF" ]]; then
    echo "Could not find LUMI AI Factory container SIF." >&2
    exit 1
  fi
}

prepare_environment() {
  mkdir -p "$WORKDIR" "$BASE_DIR"
  load_lumi_ai_container

  singularity run "$SIF" bash -lc "
    set -euo pipefail
    cd '$REPO_ROOT'
    python -m venv .venv-lumi --system-site-packages
    source .venv-lumi/bin/activate
    python -m pip install -U pip
    python -m pip install -e . --no-deps
    python -m pip install \
      datasets fastapi ipykernel kernels matplotlib psutil python-dotenv regex \
      rustbpe scipy setuptools tabulate tiktoken tokenizers transformers \
      uvicorn wandb zstandard
    python - <<'PY'
import torch
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('cuda api available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
PY
  "
}

container_python() {
  singularity run "$SIF" bash -lc "
    set -euo pipefail
    cd '$REPO_ROOT'
    source .venv-lumi/bin/activate
    export NANOCHAT_BASE_DIR='$BASE_DIR'
    export HF_HOME='$BASE_DIR/hf'
    export XDG_CACHE_HOME='$BASE_DIR/xdg'
    export WANDB_RUN=dummy
    export OMP_NUM_THREADS=1
    export NANOCHAT_DTYPE=bfloat16
    mkdir -p \"\$NANOCHAT_BASE_DIR\" \"\$HF_HOME\" \"\$XDG_CACHE_HOME\"
    $*
  "
}

configure_rocm_runtime() {
  export MIOPEN_USER_DB_PATH="/tmp/${USER:-user}-miopen-cache-${SLURM_JOB_ID:-manual}"
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
  mkdir -p "$MIOPEN_USER_DB_PATH"

  # RCCL uses NCCL-compatible variable names.
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export NCCL_NET_GDR_LEVEL=3
}

run_first() {
  prepare_environment
  configure_rocm_runtime

  container_python "
    python -m nanochat.report reset

    python -m nanochat.dataset -n 1
    python -m scripts.tok_train --max-chars=10000000
    python -m scripts.tok_eval

    python -m scripts.base_train \
      --depth=4 \
      --model-tag=lumi-first \
      --max-seq-len=256 \
      --device-batch-size=4 \
      --total-batch-size=4096 \
      --eval-every=10 \
      --eval-tokens=8192 \
      --core-metric-every=-1 \
      --sample-every=-1 \
      --save-every=-1 \
      --num-iterations=20 \
      --run=dummy

    python -m scripts.base_eval \
      --model-tag=lumi-first \
      --device-batch-size=1 \
      --split-tokens=4096 \
      --max-per-task=4 || true

    curl -L -o \"\$NANOCHAT_BASE_DIR/identity_conversations.jsonl\" \
      https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    python -m scripts.chat_sft \
      --model-tag=lumi-first \
      --max-seq-len=256 \
      --device-batch-size=4 \
      --total-batch-size=4096 \
      --eval-every=-1 \
      --eval-tokens=4096 \
      --chatcore-every=-1 \
      --num-iterations=20 \
      --run=dummy

    python -m scripts.chat_cli -p 'Say hello from nanochat on LUMI in one short sentence.' || true
    python -m nanochat.report generate || true
  "
}

run_pilot() {
  prepare_environment
  configure_rocm_runtime

  srun --ntasks=1 --gpus-per-task=8 singularity run "$SIF" bash -lc "
    set -euo pipefail
    cd '$REPO_ROOT'
    source .venv-lumi/bin/activate
    export NANOCHAT_BASE_DIR='$BASE_DIR'
    export HF_HOME='$BASE_DIR/hf'
    export XDG_CACHE_HOME='$BASE_DIR/xdg'
    export WANDB_RUN=dummy
    export OMP_NUM_THREADS=1
    export NANOCHAT_DTYPE=bfloat16
    mkdir -p \"\$NANOCHAT_BASE_DIR\" \"\$HF_HOME\" \"\$XDG_CACHE_HOME\"

    python -m nanochat.report reset
    python -m nanochat.dataset -n 8

    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval

    torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
      --depth=8 \
      --model-tag=lumi-pilot \
      --max-seq-len=512 \
      --device-batch-size=8 \
      --total-batch-size=65536 \
      --eval-every=25 \
      --eval-tokens=524288 \
      --core-metric-every=-1 \
      --sample-every=-1 \
      --save-every=-1 \
      --num-iterations=100 \
      --run=dummy

    torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
      --model-tag=lumi-pilot \
      --device-batch-size=1 \
      --split-tokens=16384 \
      --max-per-task=16 || true

    curl -L -o \"\$NANOCHAT_BASE_DIR/identity_conversations.jsonl\" \
      https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
      --model-tag=lumi-pilot \
      --max-seq-len=512 \
      --device-batch-size=8 \
      --total-batch-size=65536 \
      --eval-every=-1 \
      --eval-tokens=524288 \
      --chatcore-every=-1 \
      --num-iterations=100 \
      --run=dummy

    python -m scripts.chat_cli -p 'Say hello from nanochat on LUMI in one short sentence.' || true
    python -m nanochat.report generate || true
  "
}

case "$MODE" in
  setup)
    prepare_environment
    ;;
  first)
    run_first
    ;;
  pilot)
    run_pilot
    ;;
  *)
    echo "Unknown internal mode: $MODE" >&2
    exit 2
    ;;
esac
