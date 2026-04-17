#!/usr/bin/env bash
set -euo pipefail

LUMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LUMI_DIR/.." && pwd)"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

PROJECT="${NANOCHAT_LUMI_PROJECT:-}"
WORKDIR="${NANOCHAT_LUMI_WORKDIR:-$REPO_ROOT/lumi-work}"
JOB_ID_FILE="$LUMI_DIR/.last-job-id"

usage() {
  cat <<'EOF'
nanochat LUMI runner

Usage:
  bash lumi/run-lumi.sh first --project project_462000963
  bash lumi/run-lumi.sh pilot --project project_462000963
  bash lumi/run-lumi.sh setup --project project_462000963
  bash lumi/run-lumi.sh shell --project project_462000963
  bash lumi/run-lumi.sh status
  bash lumi/run-lumi.sh logs

Options:
  --project ID       Slurm account, e.g. project_462000963
  --workdir PATH     Cache/artifact directory (default: ./lumi-work)

Environment overrides:
  NANOCHAT_LUMI_PROJECT
  NANOCHAT_LUMI_WORKDIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:?missing project id after --project}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:?missing path after --workdir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

infer_project() {
  local path="${PWD}"
  if [[ "$path" =~ /(scratch|project|flash)/(project_[0-9]+)(/|$) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  fi
}

need_project() {
  if [[ -z "$PROJECT" ]]; then
    PROJECT="$(infer_project || true)"
  fi
  if [[ -z "$PROJECT" ]]; then
    echo "Could not infer Slurm project. Pass --project project_XXXXXXXXX." >&2
    exit 2
  fi
}

ensure_dirs() {
  mkdir -p "$WORKDIR" "$REPO_ROOT/logs"
}

submit_job() {
  local name="$1"
  local sbatch_file="$2"
  need_project
  ensure_dirs
  local output
  output="$(sbatch \
    -A "$PROJECT" \
    --export=ALL,NANOCHAT_LUMI_PROJECT="$PROJECT",NANOCHAT_LUMI_WORKDIR="$WORKDIR",NANOCHAT_LUMI_SCRIPT="$LUMI_DIR/run-lumi.sh" \
    "$sbatch_file")"
  local job_id
  job_id="$(awk '{print $4}' <<<"$output")"
  printf '%s\n' "$job_id" > "$JOB_ID_FILE"
  echo "$name submitted as job $job_id"
  echo "Logs: $REPO_ROOT/logs"
}

status() {
  if [[ -f "$JOB_ID_FILE" ]]; then
    local job_id
    job_id="$(cat "$JOB_ID_FILE")"
    squeue -j "$job_id" || true
  else
    squeue -u "$USER" || true
  fi
}

logs() {
  local latest
  latest="$(ls -t "$REPO_ROOT"/logs/*.out 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "No log files found in $REPO_ROOT/logs yet." >&2
    exit 1
  fi
  echo "Tailing $latest"
  tail -n 120 -f "$latest"
}

case "$COMMAND" in
  first)
    submit_job "First nanochat LUMI run" "$LUMI_DIR/slurm/first.sbatch"
    ;;
  pilot)
    submit_job "8-GPU nanochat LUMI pilot" "$LUMI_DIR/slurm/pilot.sbatch"
    ;;
  setup)
    ensure_dirs
    "$LUMI_DIR/slurm/run-inside-lumi.sh" setup
    ;;
  shell)
    need_project
    ensure_dirs
    exec srun -A "$PROJECT" -p small-g -t 00:30:00 \
      --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem=64G \
      --export=ALL,NANOCHAT_LUMI_PROJECT="$PROJECT",NANOCHAT_LUMI_WORKDIR="$WORKDIR" \
      --pty bash -lc "cd '$REPO_ROOT' && exec bash"
    ;;
  status)
    status
    ;;
  logs)
    logs
    ;;
  _inside)
    "$LUMI_DIR/slurm/run-inside-lumi.sh" "${1:?missing internal mode}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 2
    ;;
esac
