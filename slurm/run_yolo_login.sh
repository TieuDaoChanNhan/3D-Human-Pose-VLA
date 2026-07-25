#!/bin/bash
# Phase 4 (YOLO cleaner) for FineVideo, run directly on the JUPITER login node
# instead of via SLURM -- use only while the booster partition is under
# maintenance and submit_yolo.sh's job would stay PENDING.
#
# Same script/args/output dirs as slurm/submit_yolo.sh, but scaled down for a
# single shared GPU (login node has 1x GH200, not 4x) instead of 128
# workers/4 GPUs with MPS.

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

mkdir -p logs/yolo_workers_login

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-login
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-login
nvidia-cuda-mps-control -d
echo "NVIDIA MPS activated for login-node GPU 0."

NUM_WORKERS=4
echo "Launching $NUM_WORKERS workers on the single login-node GPU (nice -n 15)..."

for i in $(seq 1 $NUM_WORKERS); do
    CUDA_VISIBLE_DEVICES=0 \
    SLURM_ARRAY_TASK_ID=$i \
    SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    nice -n 15 \
    python -u pipeline_pose/phase4_yolo_cleaner.py \
        --videos-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/shared_window_independent/videos_staging" \
        --input-dir  "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/states_jsonl_30fps" \
        --resampled-npy-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/3d_npy_30fps" \
        --output-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/yolo_cleaned_30fps" \
        --model      "/e/project1/reformo/nguyen38/3d-human-pose/yolo26n.pt" \
        --batch-size 128 > "logs/yolo_workers_login/worker_${i}.log" 2>&1 &
done

wait
echo quit | nvidia-cuda-mps-control
echo "All $NUM_WORKERS login-node workers finished."
