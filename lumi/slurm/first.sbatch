#!/bin/bash
#SBATCH -p small-g
#SBATCH -t 00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH -J nanochat-first
#SBATCH -o logs/lumi-first-%j.out
#SBATCH -e logs/lumi-first-%j.err

set -euo pipefail

cd "$(dirname "$NANOCHAT_LUMI_SCRIPT")/.."
"$NANOCHAT_LUMI_SCRIPT" _inside first
