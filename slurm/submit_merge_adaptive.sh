#!/bin/bash
#SBATCH --job-name=merge_adapt
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-32
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_adaptive_%a.out
#SBATCH --error=logs/merge_adaptive_%a.err

source setup_motionbert.sh

DATA="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/FineVideo-VLA"
AGENT_DIR="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/agent_tokens_adaptive"
OUT_DIR="${DATA}/final_dataset_adaptive"

mkdir -p logs "${OUT_DIR}"

echo "[Worker ${SLURM_ARRAY_TASK_ID}] Starting merge (adaptive PCHIP)..."

python -u pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "${DATA}/training_ready_rank_*.jsonl" \
  --agent-tokens-dir "${AGENT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --output-prefix "final_vla_adaptive" \
  --skip-existing

echo "[Worker ${SLURM_ARRAY_TASK_ID}] Done."
