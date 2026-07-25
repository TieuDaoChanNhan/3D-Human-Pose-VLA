#!/bin/bash
#SBATCH --job-name=yolo_w24_full
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=256
#SBATCH --time=08:00:00
#SBATCH --output=logs/yolo_w24_full_%j.log

# 2026-07-23: full-scale Phase 4 rerun at window=24, adapted from
# submit_yolo.sh (window=8 production). Reads Phase 3's states_jsonl_w24
# output (job kin_w24_full). New output dir (yolo_cleaned_w24).

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

mkdir -p logs/yolo_w24_workers

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
echo "NVIDIA MPS activated for all 4 GPUs"

NUM_WORKERS=128
echo "Launching $NUM_WORKERS workers distributed across 4 GH200 GPUs (window=24)..."

for i in $(seq 1 $NUM_WORKERS); do
    GPU_ID=$(( (i - 1) % 4 ))
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    SLURM_ARRAY_TASK_ID=$i \
    SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u pipeline_pose/phase4_yolo_cleaner.py \
        --videos-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/shared_window_independent/videos_staging" \
        --input-dir  "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/states_jsonl_w24" \
        --resampled-npy-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/3d_npy_30fps" \
        --output-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/yolo_cleaned_w24" \
        --model      "/e/project1/reformo/nguyen38/3d-human-pose/yolo26n.pt" \
        --window-size 24 \
        --batch-size 128 > logs/yolo_w24_workers/worker_${i}.log 2>&1 &
done

wait
echo quit | nvidia-cuda-mps-control
echo "Phase 4 (window=24) full-scale done."
