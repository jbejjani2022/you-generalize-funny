#!/usr/bin/env bash
# Generic entry: pass resources on the sbatch command line, then the ygf subcommand and args.
#   sbatch --account=aip-dkd --gres=gpu:l40s:1 -c 8 --mem=64G --time=1:00:00 \
#     --output=logs/%x-%j.out --job-name=<name> scripts/job.sbatch smoke --model talkie-tulu
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME=/home/jbejjani/projects/aip-dkd/jbejjani/.cache/huggingface
export TRAIT_ARTIFACT_ROOT=/scratch/jbejjani/talkie/trait
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run --frozen --extra gpu ygf "$@"
