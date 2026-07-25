#!/bin/bash
#SBATCH --job-name=phase7_w24_dc50
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase7_w24_dc50_%j.out
#SBATCH --error=logs/phase7_w24_dc50_%j.err

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3

DATA="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/FineVideo-VLA"

mkdir -p logs "${DATA}/megatron_dataset_adaptive_w24_dropcosmos50"

echo "=== Phase 7 v5-ablation: drop_cosmos 0.85 -> 0.5, everything else same as w24 production ==="
echo "Input:   ${DATA}/final_dataset_adaptive_w24/final_vla_adaptive_rank_*.jsonl"
echo "Output:  ${DATA}/megatron_dataset_adaptive_w24_dropcosmos50/"
echo "--skip-existing: reuses files already written by the earlier login-node run before it was killed and moved to sbatch."

python -u pipeline_pose/phase7_flatten.py \
    --input-glob "${DATA}/final_dataset_adaptive_w24/final_vla_adaptive_rank_*.jsonl" \
    --output-dir "${DATA}/megatron_dataset_adaptive_w24_dropcosmos50" \
    --drop_cosmos 0.5 \
    --workers 32 \
    --skip-existing

echo "=== Phase 7 v5-ablation done ==="
