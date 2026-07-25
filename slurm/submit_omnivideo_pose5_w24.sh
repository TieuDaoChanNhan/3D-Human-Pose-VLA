#!/bin/bash
#SBATCH --job-name=omni_p5_w24
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/omni_p5_w24_%j.log

# 2026-07-23: OmniVideo-100K Phase 5 (adaptive PCHIP agent tokens),
# window=24 to match FineVideo-VLA's pivot. Reuses the shared
# pipeline_pose/phase5_adaptive_pchip.py directly (same script FineVideo
# uses) -- sports subset only (1,126 videos), small enough for 1 task.

source /e/project1/reformo/nguyen38/3d-human-pose/setup_motionbert.sh
cd /e/project1/reformo/nguyen38/3d-human-pose

DATA_ROOT=/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k

python -u pipeline_pose/phase5_adaptive_pchip.py \
    --input-dir  "$DATA_ROOT/pose_yolo_cleaned_30fps_w24" \
    --output-dir "$DATA_ROOT/pose_agent_tokens_adaptive_w24" \
    --window-frames 24 \
    --stride 24

echo "OmniVideo-100K Phase 5 (window=24) done."
