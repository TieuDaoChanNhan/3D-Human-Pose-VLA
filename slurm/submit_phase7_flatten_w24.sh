#!/bin/bash
#SBATCH --job-name=phase7_w24
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase7_w24_%j.out
#SBATCH --error=logs/phase7_w24_%j.err

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3
# same reasoning as submit_phase7_flatten_v6.sh: phase7_flatten.py needs `wn`
# (text augmentation), only available via ~/.local under this module set,
# not env_motion_final -- don't source setup_motionbert.sh here.

DATA="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/FineVideo-VLA"

mkdir -p logs "${DATA}/megatron_dataset_adaptive_w24"

echo "=== Phase 7 w24 Flatten (window=24, aspect-preserving cosmos, drop_cosmos=0.85) ==="
echo "Input:   ${DATA}/final_dataset_adaptive_w24/final_vla_adaptive_rank_*.jsonl"
echo "Output:  ${DATA}/megatron_dataset_adaptive_w24/"
echo "drop_cosmos=0.85 -- decided 2026-07-23: real pre-dropout count showed cosmos"
echo "at 92.5% of FineVideo's own token budget (11,572.5M/12,516.2M), far higher than"
echo "the old v5/window=8 figure (73.9%) that motivated the previous ~0.65 estimate."
echo "drop_seed/drop_snac stay at their 0.0 defaults (always keep) -- unchanged."

python -u pipeline_pose/phase7_flatten.py \
    --input-glob "${DATA}/final_dataset_adaptive_w24/final_vla_adaptive_rank_*.jsonl" \
    --output-dir "${DATA}/megatron_dataset_adaptive_w24" \
    --drop_cosmos 0.85 \
    --workers 32 \
    --skip-existing

echo "=== Phase 7 w24 done ==="
