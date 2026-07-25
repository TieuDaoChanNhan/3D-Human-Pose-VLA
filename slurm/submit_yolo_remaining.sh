#!/bin/bash
#SBATCH --job-name=yolo_cleanup_remaining
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4                # Request all 4 GH200 GPUs on one node
#SBATCH --cpus-per-task=256     # Maximise CPU for OpenCV decoding
#SBATCH --time=08:00:00         # Expected to finish within 8 hours
#SBATCH --output=logs/yolo_master_%j.log

# Same as submit_yolo.sh, but --input-dir points at a symlink farm containing
# only the videos NOT yet done (states_jsonl_30fps_remaining_phase4/) instead
# of the full 40,305-file states_jsonl_30fps/. Contiguous split over the full
# list clustered already-done videos (from an earlier 4-worker login-node
# run) into ~37.5% of the 128 SLURM workers, leaving the other ~80 with
# almost 2x the average real work -- splitting over the remaining-only list
# instead gives every worker an even ~171-video share.

# Activate environment
module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

mkdir -p logs/yolo_workers

# ==========================================
# START NVIDIA MPS FOR ALL 4 GPUs
# ==========================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
echo "🚀 NVIDIA MPS activated for all 4 GPUs!"

# Total workers (128 workers / 4 GPUs = 32 workers per GPU)
NUM_WORKERS=128
echo "🚀 Launching $NUM_WORKERS workers distributed across 4 GH200 GPUs..."

# Worker dispatch loop
for i in $(seq 1 $NUM_WORKERS); do
    # Round-robin: distribute evenly across 4 GPUs (0, 1, 2, 3)
    GPU_ID=$(( (i - 1) % 4 ))

    echo "Starting worker $i -> assigned to GPU $GPU_ID"

    # Restrict worker to its assigned GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    SLURM_ARRAY_TASK_ID=$i \
    SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u pipeline_pose/phase4_yolo_cleaner.py \
        --videos-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/shared_window_independent/videos_staging" \
        --input-dir  "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/states_jsonl_30fps_remaining_phase4" \
        --resampled-npy-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/3d_npy_30fps" \
        --output-dir "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/yolo_cleaned_30fps" \
        --model      "/e/project1/reformo/nguyen38/3d-human-pose/yolo26n.pt" \
        --batch-size 128 > logs/yolo_workers/worker_${i}.log 2>&1 &
done

wait
echo quit | nvidia-cuda-mps-control
echo "🎉 ALL DONE!"
