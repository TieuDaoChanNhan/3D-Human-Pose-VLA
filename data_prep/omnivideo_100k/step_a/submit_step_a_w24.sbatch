#!/bin/bash
#SBATCH --job-name=omni100k_stepA_w24
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs per node
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=05:00:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_stepA_w24_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_stepA_w24_err.log

# 2026-07-23: v3/w24 = window=24 pivot to match FineVideo-VLA (CHUNK_SIZE
# 8->24, COSMOS_STRIDE=3 added) on top of v2's cosmos resolution/aspect-ratio
# fix + seed2 import/tie_weights fix + scratch-on-data1 fix (all still in
# effect, step_a_tokenize_video.py itself was edited in place, not forked).
# New --output-dir: step_a_output/ and step_a_output_v2/ (both window=8)
# were deleted after this pivot -- see project_window24_new_standard memory.

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

srun python -u /e/project1/reformo/nguyen38/3d-human-pose/data_prep/omnivideo_100k/step_a/step_a_tokenize_video.py \
    --output-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k/step_a_output_w24
