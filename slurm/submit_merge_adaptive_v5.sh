#!/bin/bash
#SBATCH --job-name=merge_v5
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-32
#SBATCH --time=03:00:00
#SBATCH --output=logs/merge_v5_%a.out
#SBATCH --error=logs/merge_v5_%a.err

source setup_motionbert.sh

DATA="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/FineVideo-VLA"
AGENT_DIR="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/agent_tokens_adaptive"
SNAC_DIR="${DATA}/snac_tokens"
CAPTIONS_DIR="${DATA}/captions_dict"
SPEECH_DIR="${DATA}/speech_segments"
OUT_DIR="${DATA}/final_dataset_adaptive_v5"

mkdir -p logs "${OUT_DIR}"

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Starting merge v5: full fresh injection (agent from re-run Phase 4/5 fps-mismatch fix, Jul 21) + snac + caption + speech, single pass on top of raw training_ready (not v3 -- v3's agent/snac predate the fps-mismatch fix). Runs on JUPITER/booster -- /p (JUWELS storage) is not mounted on JUPITER compute nodes, only its login node, so all inputs were staged to /e first."

python -u pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "${DATA}/training_ready_rank_*.jsonl" \
  --agent-tokens-dir    "${AGENT_DIR}" \
  --snac-tokens-dir     "${SNAC_DIR}" \
  --captions-dir        "${CAPTIONS_DIR}" \
  --speech-segments-dir "${SPEECH_DIR}" \
  --output-dir          "${OUT_DIR}" \
  --output-prefix       "final_vla_adaptive" \
  --skip-existing

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Done."
