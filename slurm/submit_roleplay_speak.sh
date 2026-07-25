#!/bin/bash
#SBATCH --job-name=roleplay_speak
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/roleplay_speak_%j.log

# 2026-07-23: full-scale re-tokenize of laion/emotional-roleplay in
# --format speak (full L0+L1+L2, Leo-matched offsets/order -- see
# data_prep/laion_emotional_roleplay/tokenize_snac.py's OFFSET_L2
# docstring). Never run at full scale before (L2 wasn't decided until this
# session). Source data copied from /p to /e first (compute nodes don't
# mount /p -- see feedback_data_storage_location memory) -- see
# /e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/laion_emotional_roleplay_data/.
# Output uses the "speak" filename prefix (roleplay_snac_speak_flat_*.jsonl),
# separate from the existing listen-format shards -- doesn't collide, no
# risk of the resume-skip trap that hit Step A.

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export HF_HOME=/e/project1/reformo/nguyen38/jupiter_cache/huggingface

cd /e/project1/reformo/nguyen38/3d-human-pose

python -u data_prep/laion_emotional_roleplay/tokenize_snac.py \
    --format speak \
    --input-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/laion_emotional_roleplay_data \
    --output-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/laion_emotional_roleplay_flattened_speak

echo "DONE"
