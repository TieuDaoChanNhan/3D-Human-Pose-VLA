#!/bin/bash
#SBATCH --job-name=phase7_v6
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase7_v6_%j.out
#SBATCH --error=logs/phase7_v6_%j.err

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3
# (not setup_motionbert.sh -- phase7_flatten.py hard-imports `wn` for text
# augmentation, which only exists in ~/.local site-packages under Python
# 3.12.3, not in env_motion_final. Installed 21/07/2026: `python3 -m pip
# install --user wn` + `python3 -m wn download oewn:2024` from a login node.
# NOT sourcing activate_env_tools.sh here: it tries to source a venv under
# /p, which compute nodes can't see (confirmed via a real job's stderr:
# "No such file or directory") -- harmless since ~/.local is used either
# way, but noisy/misleading in the logs, so just doing the module load
# directly instead.)

DATA="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/FineVideo-VLA"

mkdir -p logs "${DATA}/megatron_dataset_v6"

echo "=== Phase 7 v6 Flatten (wrapper-token fix + fps-mismatch-fixed agent tokens) ==="
echo "Input:   ${DATA}/final_dataset_adaptive_v5/final_vla_adaptive_rank_*.jsonl"
echo "Output:  ${DATA}/megatron_dataset_v6/"
echo "(named v6, not v5 -- megatron_dataset_v5 is the currently-published"
echo " EmpathicRobotics/FineVideo-Phase7-Flattened built from pre-fix data;"
echo " keeping this separate avoids overwriting/confusing it until we're"
echo " ready to re-upload.)"

python -u pipeline_pose/phase7_flatten.py \
    --input-glob "${DATA}/final_dataset_adaptive_v5/final_vla_adaptive_rank_*.jsonl" \
    --output-dir "${DATA}/megatron_dataset_v6" \
    --workers 32 \
    --skip-existing

echo "=== Phase 7 v6 done ==="
