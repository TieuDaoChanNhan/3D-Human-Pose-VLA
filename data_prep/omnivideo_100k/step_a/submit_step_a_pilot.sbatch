#!/bin/bash
#SBATCH --job-name=omni100k_stepA_pilot
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs per node
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_stepA_pilot_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_stepA_pilot_err.log

module --force purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load PyTorch/2.5.1
module load torchvision/0.20.1

source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export FFMPEG_PATH=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pilot: 8 rank * 6 video = 48 video, đo throughput thật trước khi tính quy mô full-scale.
srun python -u /e/project1/reformo/nguyen38/3d-human-pose/data_prep/omnivideo_100k/step_a/step_a_tokenize_video.py \
    --output-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k/step_a_output_pilot \
    --limit 48
