#!/bin/bash
#SBATCH --job-name=omni100k_phase1_pilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase1_pilot_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase1_pilot_err.log

cd /e/project1/reformo/nguyen38/3d-human-pose

source setup_hrnet_gpu.sh   # env_hrnet_datasets_v1 -- the env for Phase 1,
                             # DIFFERENT from env_stable_vla (Step A) and env_motion_final (Phase 2)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pilot: 4 ranks * 6 videos = first 24 videos of sports_subset_video_ids.txt,
# to measure real throughput + person-detection rate before running the full 1,256 videos.
srun python -u data_prep/omnivideo_100k/pose/phase1_hrnet_omnivideo.py \
    --output-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k/pose_2d_json_pilot \
    --limit 24
