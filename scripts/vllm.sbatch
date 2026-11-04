#!/usr/bin/env bash
# vLLM sampling: same CLI shape as job.sbatch, e.g.
#   sbatch --account=aip-dkd --gres=gpu:l40s:1 -c 8 --mem=96G --time=2:00:00 --output=logs/%x-%j.out \
#     --job-name=<name> scripts/vllm.sbatch sample --spec specs/x0.json --model talkie-it
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.2
export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"
export HF_HOME=/home/jbejjani/projects/aip-dkd/jbejjani/.cache/huggingface
export VLLM_CACHE_ROOT=/scratch/jbejjani/talkie/ygf/vllm-cache
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run --frozen --project vllm-env python -m ygf.vllm_sample "$@"
