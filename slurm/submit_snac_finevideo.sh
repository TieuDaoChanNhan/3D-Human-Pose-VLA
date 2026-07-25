#!/bin/bash
#SBATCH --job-name=snac_fv_full
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=256
#SBATCH --time=06:00:00
#SBATCH --output=logs/snac_finevideo_full_%j.log

# 2026-07-23: full-scale SNAC (listen-format) tokenization of FineVideo-VLA
# activity audio, adapted from submit_yolo_w24.sh's worker-pool pattern.
# Old version of this script targeted JUWELS (laionize account, /p paths,
# ppc64le env) -- rewritten for JUPITER (reformo/booster, /e/data1 paths,
# env_stable_vla), matching pipeline_pose/snac_finevideo.py's fixed
# defaults. Prereq: task list already built
#   (/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/FineVideo-VLA/snac_task_list.json,
#    40,798 videos / 372,385 activities, built 2026-07-23 12:13) --
#   rerun `python pipeline_pose/snac_finevideo.py --build-tasks` first if
#   final_dataset_adaptive changes.
# Smoke-tested on 1 video (P0Ol42Fz3ic, 4/4 activities ok) before this submit
# -- see samples/finevideo-vla/snac_listen_smoketest/.

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

export HF_HOME=/e/project1/reformo/nguyen38/jupiter_cache/huggingface
export HF_HUB_OFFLINE=1

mkdir -p logs/snac_finevideo_workers

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
echo "NVIDIA MPS activated for all 4 GPUs"

# 2026-07-23: 128 workers (32/GPU, copied from submit_yolo_w24.sh) OOM-crashed
# almost every worker. Dropped to 32 (8/GPU, matching
# submit_snac_omnivideo_w24.sh's ratio) -- STILL OOM'd (observed workers
# holding 10-16 GiB each; FineVideo activities can run much longer than
# omnivideo's whole-video audio, so SNAC's per-request memory peak is
# higher here even at the same worker count). Dropped further to 16 (4/GPU).
NUM_WORKERS=16
echo "Launching $NUM_WORKERS workers distributed across 4 GH200 GPUs (SNAC listen-format)..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    GPU_ID=$(( i % 4 ))
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    SLURM_ARRAY_TASK_ID=$i \
    SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u pipeline_pose/snac_finevideo.py \
        --format listen > logs/snac_finevideo_workers/worker_${i}.log 2>&1 &
done

wait
echo quit | nvidia-cuda-mps-control
echo "SNAC FineVideo-VLA (listen-format) full-scale done."
