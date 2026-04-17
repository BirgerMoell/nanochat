#!/bin/bash
#SBATCH -p standard-g
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=8
#SBATCH -J nanochat-pilot
#SBATCH -o logs/lumi-pilot-%j.out
#SBATCH -e logs/lumi-pilot-%j.err

set -euo pipefail

cd "$(dirname "$NANOCHAT_LUMI_SCRIPT")/.."
"$NANOCHAT_LUMI_SCRIPT" _inside pilot
