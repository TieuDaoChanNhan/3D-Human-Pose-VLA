#!/bin/bash
#SBATCH --job-name=h4d_coco2h36m
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1               # Single node -- 22 zip files, CPU-only numpy work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=logs/h4d_coco2h36m_%j.log

# Converts Harmony4D's poses3d/*.npy (COCO-17 order) to this project's
# H36M-17 order, native 20fps (not resampled here).
# See data_prep/harmony4d/convert_coco_to_h36m.py for the full mapping,
# verification notes, and output layout.

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

cd /e/project1/reformo/nguyen38/3d-human-pose

HARMONY4D_ROOT="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d"
OUTPUT_DIR="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_native20fps"
mkdir -p "$OUTPUT_DIR"

NUM_WORKERS=22   # 15 train zips + 7 test zips -- one worker per zip file
MAX_ID=$((NUM_WORKERS - 1))

echo "🚀 Launching $NUM_WORKERS workers (1 per zip file, 15 train + 7 test)..."

for i in $(seq 0 $MAX_ID); do
    export SLURM_ARRAY_TASK_ID=$((i + 1))
    export SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS

    python -u data_prep/harmony4d/convert_coco_to_h36m.py \
        --harmony4d-root "$HARMONY4D_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --splits train test \
        > "logs/h4d_convert_worker_${i}.log" 2>&1 &
done

wait

echo "🎉 All $NUM_WORKERS workers finished."
echo "Manifest entries per worker:"
wc -l "$OUTPUT_DIR"/manifest_task*.jsonl 2>/dev/null
echo "Total sequences converted:"
find "$OUTPUT_DIR" -name ".done" | wc -l
