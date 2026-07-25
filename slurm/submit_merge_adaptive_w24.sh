#!/bin/bash
#SBATCH --job-name=merge_w24
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-32
#SBATCH --time=03:00:00
#SBATCH --output=logs/merge_w24_%a.out
#SBATCH --error=logs/merge_w24_%a.err

source setup_motionbert.sh

DATA="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/FineVideo-VLA"
AGENT_DIR="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/agent_tokens_adaptive_w24"
SNAC_DIR="${DATA}/snac_tokens_w24"
CAPTIONS_DIR="${DATA}/captions_dict"
SPEECH_DIR="${DATA}/speech_segments"
OUT_DIR="${DATA}/final_dataset_adaptive_w24"

mkdir -p logs "${OUT_DIR}"

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Merge w24: training_ready_w24 (Step A job 1022987, aspect-preserving cosmos, window=24) + agent_tokens_adaptive_w24 (Phase 5, window=24) + snac_tokens_w24 (SNAC FineVideo, window=24) + captions_dict/speech_segments (window-independent, reused unchanged). --chunk-size 24 is mandatory here -- script default is still 8."

python -u pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "${DATA}/training_ready_w24_rank_*.jsonl" \
  --agent-tokens-dir    "${AGENT_DIR}" \
  --snac-tokens-dir     "${SNAC_DIR}" \
  --captions-dir        "${CAPTIONS_DIR}" \
  --speech-segments-dir "${SPEECH_DIR}" \
  --output-dir          "${OUT_DIR}" \
  --output-prefix       "final_vla_adaptive" \
  --chunk-size 24 \
  --skip-existing

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Done."
