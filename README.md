# FineVideo-VLA Dataset Pipeline

**⚠️ If you're picking this up on JUWELS: see `TOKENIZE_TODO.md` first** — it lists exactly which
datasets on `/p` still need (re-)tokenizing into Megatron `.bin/.idx` before the next training run,
and which existing tokenized outputs are stale.

**⚠️ Scope note (2026-07-20):** this repo's pipeline (below) covers the **video + 3D-pose branch** of a broader **omni-modal** project — not the whole scope anymore. Per Huu (project lead), the target model binds *any* modality pair (image, video, sound, action, IMU, ...), as long as the source is permissively licensed and the mix is balanced across modalities. Non-video/pose sources (e.g. `synth_llava` image+caption pairs, `laion/emotional-roleplay-finetuning-dataset` speech+text) are being folded in under `data_prep/`, tokenized via whichever existing modality (`seed2` for standalone images, `snac` for standalone audio) fits, rather than requiring every source to produce video+pose+agent tokens. See `../CLAUDE.md`'s Project Overview and `datasets.md` for the full picture; everything below describes this repo's original, still-primary video+pose branch.

This repository contains the **complete pipeline** for building the FineVideo-VLA pretraining dataset from HuggingFace's [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) dataset (~40K YouTube videos). The output is a Megatron-LM-ready flat JSONL dataset.

There are two parallel branches that produce different token types and are merged at the end:

| Branch | What it produces | Entry point |
|--------|-----------------|-------------|
| **Prototype pipeline** (Step A) | Seed2 + Cosmos + AVC-LM video tokens | `pipeline_video/pipeline.py` |
| **3D pose pipeline** (Steps B–G) | Adaptive PCHIP per-joint tokens (17 joints, variable CPs) | `pipeline_pose/phase1_hrnet_gpu.py` … `pipeline_pose/phase5_adaptive_pchip.py` |

Each video activity in the final dataset produces an interleaved token sequence:
```
USER: <activity_description> [Speech: ...]  ASSISTANT:
  <seed2> <seed2_N> ... </seed2>       # 1 FPS semantic keyframe     (vocab: 8192)
  <cosmos> <cosmos_N> ... </cosmos>    # every 8 frames, spatial     (vocab: 64000)
  <avc_lm> <avclm_N> ... </avc_lm>    # every 8 frames, H.264 BPE   (vocab: 8192)
  <agent> <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_N> ... </pelvis> ... </agent>
```

After flattening, `<tag> N </tag>` becomes `<tag_N>` for seed2/cosmos/avc_lm.
Agent tokens are already self-describing (`<pelvis_x_128>` etc) and pass through unchanged.

---

## HuggingFace Datasets & Tokenizer

| Resource | Description | Size |
|----------|-------------|------|
| [FineVideo-Prototype-Tokenized](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Prototype-Tokenized) | Base video tokens (Seed2/Cosmos/AVC-LM) from prototype pipeline | ~660 GB |
| [FineVideo-Phase2-3DPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase2-3DPose) | 3D pose NPY from MotionBERT (after Phase 2) | ~259 GB |
| [FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) | YOLO-cleaned 3D poses (raw floats, after Phase 3+4) | ~107 GB |
| [FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) | Full hierarchical merged dataset with agent tokens (after Phase 5+6) | ~657 GB |
| [FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) | Flat Megatron-LM JSONL (final output, ready for pretraining) | ~19 GB |
| [tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | HuggingFace tokenizer (GPT-NeoX-20b + 93,938 VLA tokens, 144,215 total) | — |

All datasets under `EmpathicRobotics/`, split 152 train / 8 test shards (95/5, seed 42), gzip compressed.

---

## Repository Layout

```
├── pipeline_video/         # Step A: Seed2, Cosmos, AVC-LM tokenization
│   ├── pipeline.py                 ← main entry point (read this first)
│   ├── pipeline_1gpu.py            single-GPU debug version
│   ├── submit_official.sbatch      SLURM job: 40 nodes × 4 GPU
│   ├── cosmos_tokenizer/           Cosmos tokenizer source
│   ├── seed2/                      Seed2 tokenizer source + vocab
│   ├── avc_lm_v2/                  AVC-LM BPE vocab (used by pipeline.py)
│   ├── pretrained_ckpts/           Cosmos model configs (weights gitignored)
│   └── README.md                   documents which files are production vs. experimental
│
├── pipeline_pose/               # Steps B–H: 3D pose → adaptive tokens → merge → flatten
│   ├── phase1_hrnet_gpu.py         2D pose estimation (HRNet)
│   ├── phase2_motionbert_gpu.py    3D pose lifting (MotionBERT)
│   ├── phase2_5_resample_30fps.py  Resample native-fps poses to 30fps
│   ├── phase3_kinematics_processor.py  Signal filter + kinematics
│   ├── phase4_yolo_cleaner.py      YOLO person-presence filter
│   ├── phase5_adaptive_pchip.py    Adaptive PCHIP per-joint tokeniser (2/4/8 CPs)
│   ├── phase6_merge_adaptive.py    Inject adaptive <agent> blocks into training_ready
│   ├── phase7_flatten.py           Flatten merged → Megatron flat JSONL (current: v4)
│   └── snac_finevideo.py           SNAC audio tokenization
│
├── tools/                      # Standalone utilities, grouped by purpose (see tools/README.md)
│   ├── upload/                     HF upload scripts + dataset cards
│   ├── tokenizer/                  Vocab expansion, tokenizer build, verification
│   ├── inventory/                  Token/dataset counting, overlap checks, data validation
│   ├── eval/                       Model sanity checks, agent-token decoding
│   ├── visualize/                  Skeleton/pose rendering for visual QA
│   ├── analysis/                   One-off compression/tradeoff analyses
│   ├── extract/                    Small per-video data extraction helpers
│   └── README.md
│
├── investigations/             # Probing/converting EXTERNAL datasets (not FineVideo) —
│   │                             separate from the core pipeline above (see investigations/README.md)
│   ├── mixturevitae_multimodal/    Paused probe of an external HF dataset for SNAC/caption content
│   └── mv_omni_seed_conversion/    MixtureVitae-Omni <seed_N> → this project's <seed2_N> vocab
│
├── manual_checks/               # Interactive debug/inference scripts — NOT an automated test suite
│                                  (see manual_checks/README.md)
│
├── archive/                     # Deprecated code, kept for reference only, nothing here runs
│   ├── pipeline_pose_deprecated/   Legacy XYZT / opaque-256 tokenizer formats (superseded)
│   ├── slurm_deprecated/           SLURM scripts for the legacy formats above
│   ├── tools_deprecated/           One-off scripts that already ran to completion
│   ├── dev_deprecated/             Old single-video dev/demo scripts (stale references)
│   └── root_notes_deprecated/      Stray leftover files, kept just in case
│
├── slurm/                      # SLURM submit scripts (all invoke pipeline_pose/*.py)
│   ├── submit_hrnet.sh             Phase 1
│   ├── submit_motionbert.sh        Phase 2
│   ├── submit_phase2_5.sh          Phase 2.5
│   ├── submit_kinematics.sh        Phase 3
│   ├── submit_yolo.sh              Phase 4
│   ├── submit_phase5_adaptive.sh   Phase 5 (adaptive PCHIP)
│   ├── submit_merge_adaptive.sh    Phase 6 (merge)
│   ├── submit_merge_adaptive_v2.sh Phase 6 v2 (+ SNAC injection)
│   ├── submit_phase7_flatten.sh    Phase 7
│   ├── submit_phase7_v3.sh         Phase 7 v3
│   ├── submit_phase7_v4.sh         Phase 7 v4 (current)
│   └── submit_snac_finevideo.sh    SNAC audio tokenization
│
├── envs/                       # Conda environment YAML specs
├── vocab/                      # gpt-neox-20b-vocab.json (base) + vocab_expanded.json
├── samples/                    # Sample outputs for inspection
├── documents/                  # Reference PDFs / Discord export chat logs
├── setup_motionbert.sh         # Activate env for Steps B–H
└── setup_hrnet_gpu.sh          # Activate env for Step B (HRNet only)
```

---

## End-to-End Pipeline

**Note (2026-07-25): this diagram describes the original window=8 pipeline
(8-frame Cosmos chunks). The current/live pipeline uses window=24 — see
`window24_current/` under `$DATA` and CLAUDE.md. Data below moved to
`$DATA/window8_legacy/` during the 2026-07-25 data1 reorg.**

```
$DATA = /e/data1/datasets/playground/mmlaion/shared/nguyen38

──────────────── BRANCH A: Video Tokens ────────────────────────────────────

Step A   pipeline_video/pipeline.py   (submit_official.sbatch, 40 nodes × 4 GPU)
         Extracts frames at 30fps; tokenizes every activity segment with
         Seed2 (1fps), Cosmos (8-frame), and AVC-LM (8-frame).
         → $DATA/window8_legacy/FineVideo-VLA/training_ready_rank_*.jsonl

──────────────── BRANCH B: 3D Pose / Adaptive PCHIP Tokens ───────────────

Step B   pipeline_pose/phase1_hrnet_gpu.py          (slurm/submit_hrnet.sh)
         HRNet + Faster R-CNN 2D pose estimation
         → outputs/2d_json/{video_id}_2d.json

Step C   pipeline_pose/phase2_motionbert_gpu.py     (slurm/submit_motionbert.sh)
         MotionBERT 3D lifting (native video fps)
         → outputs/3d_npy/{video_id}.npy

Step D   pipeline_pose/phase2_5_resample_30fps.py   (slurm/submit_phase2_5.sh)
         Resample native-fps 3D poses to 30fps via linear interpolation.
         Required so Steps E–G share the same time grid as Branch A.
         → outputs/3d_npy_30fps/{video_id}.npy

Step E   pipeline_pose/phase3_kinematics_processor.py  (slurm/submit_kinematics.sh)
         Signal filter, bone normalisation, kinematics (pos/vel/acc)
         → outputs/states_jsonl/{video_id}_states.jsonl   (windows × 8 × 153)

Step F   pipeline_pose/phase4_yolo_cleaner.py       (slurm/submit_yolo.sh)
         Drop 8-frame windows where ≥ 4 frames have no detected person.
         → outputs/yolo_cleaned_30fps/{video_id}_cleaned.jsonl

Step G   pipeline_pose/phase5_adaptive_pchip.py     (slurm/submit_phase5_adaptive.sh)
         Adaptive PCHIP per-joint tokenisation: 2/4/8 CPs based on curvature
         → outputs/agent_tokens_adaptive/{video_id}_tokens.jsonl

──────────────── MERGE + FLATTEN ───────────────────────────────────────────

Step H   pipeline_pose/phase6_merge_adaptive.py     (slurm/submit_merge_adaptive.sh)
         Injects <agent> blocks (with per-joint named tokens) after each
         <avc_lm> block in training_ready files. Adds chunk_timing + timing_meta.
         → $DATA/window8_legacy/FineVideo-VLA/final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl

Step I   pipeline_pose/phase7_flatten.py            (run on login node or SLURM)
         Hierarchical JSON → Megatron flat JSONL.
         Agent blocks pass through unchanged (already self-describing).
         → $DATA/window8_legacy/FineVideo-VLA/megatron_dataset_adaptive/flat_*.jsonl

Step J   tokenize_vla_adaptive.sbatch          (4 nodes, Ray-distributed)
         Megatron-LM tokenization using EmpathicRobotics/tokenizer-vla-adaptive.
         All VLA tokens are atomic (added via add_tokens, not manual JSON).
         → /p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/

Step K   tools/upload/upload_flattened_hf.py
         Compress + upload to EmpathicRobotics/FineVideo-Phase7-Flattened
```

Steps B–G run from `3d-human-pose/` as working directory. `outputs/` is a symlink to `$DATA/outputs/`.

---

## Adaptive PCHIP Token Format (current)

Each 8-frame chunk produces **variable-length** self-describing tokens (171–579, typical ~250–300):

```
<fps_30>
<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
         <pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
<r_hip>  <r_hip_t_0>  <r_hip_x_140> <r_hip_y_130> <r_hip_z_126>
         <r_hip_t_7>  <r_hip_x_141> <r_hip_y_128> <r_hip_z_124> </r_hip>
...17 joints total...
```

| Component | Description |
|-----------|-------------|
| `<fps_N>` | Frame rate (always 30) |
| `<joint>` / `</joint>` | Per-joint wrapper tags |
| `<joint_t_N>` | Frame index 0–7 within the window (control point time) |
| `<joint_x_N>`, `<joint_y_N>`, `<joint_z_N>` | Quantized position, `N = clip(round((v + 2.0) / 4.0 * 255), 0, 255)` |

- **Joint order** (H36M 17-joint): pelvis, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, spine, thorax, nose, head_top, l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist
- **CP tiers**: curvature < tau_low → 2 CPs | tau_low–tau_high → 4 CPs | >= tau_high → 8 CPs
- **Coordinate range**: [-2.0 m, +2.0 m], precision ~15.7 mm
- **Reconstruction**: parse CPs per joint, apply PCHIP interpolation to recover all 8 frames

### Legacy formats

| Format | Script | Tokens/chunk | Status |
|--------|--------|-------------|--------|
| XYZT (fixed 409) | `phase5b_xyzt_tokenizer.py` | 409 | Superseded |
| Opaque uint8 (256) | `phase5_interpolation_tokenizer.py` | 256 | Superseded |

---

## Token Alignment

All four token types share the same **30fps frame grid**:

| Token type | Fires at | Covers frames | Timestamp formula |
|------------|----------|---------------|-------------------|
| Seed2 | every 30 frames | single frame | `activity_start + k × 1.0 s` |
| Cosmos | every 8 frames | frames `[8k, 8k+7]` | `activity_start + k × 8/30 s` |
| AVC-LM | every 8 frames | frames `[8k, 8k+7]` | `activity_start + k × 8/30 s` |
| Agent (adaptive) | every 8 frames | frames `[8k, 8k+7]` | `activity_start + window_id × 8/30 s` |

**Agent tokens may be non-contiguous** — Phase F (YOLO) drops windows with no detected person. Each surviving record stores `window_id` (original frame offset), so timestamps are always recoverable.

---

## Flattened Dataset Statistics

The final flattened output (`megatron_dataset_adaptive/`, agent-only with modality dropout) contains:

| Metric | Value |
|--------|-------|
| Total files | 160 |
| Total records | 69,844 |
| Total size | ~19.2 GB |
| Avg file size | ~120 MB (range: 85.8–176.7 MB) |
| Malformed JSON | 0 |
| Records with agent | **100%** (agent-only filter) |

**Modality coverage** (after dropout):

| Modality | Coverage | Avg tokens/record |
|----------|----------|-------------------|
| seed2 | 100% | ~1,320 |
| cosmos | ~88% | ~3,091 |
| avclm | ~49% | ~7,260 |
| agent (3D pose) | 100% | ~9,712 |

---

## Environment Setup

**Step A — prototype pipeline** (Seed2 + Cosmos + AVC-LM):
```bash
module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export FFMPEG_PATH=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
```
Model weights are **not committed** — download once with `pipeline_video/download.py` (requires `HF_TOKEN`).

**Steps B–H — 3D pose pipeline** (HRNet, MotionBERT, YOLO, tokenizer):
```bash
source setup_motionbert.sh    # activates env_motion_final/
```

**Step B only — HRNet** (different env):
```bash
source setup_hrnet_gpu.sh
```

**Flatten + upload** (tools):
```bash
source setup_motionbert.sh    # or any env with huggingface_hub installed
```

Environment YAML specs are in `envs/`.

---

## Cluster (JUPITER — `booster` partition)

- **Hardware:** GH200 nodes, 4 GPUs per node, 288 CPU cores per node
- **Account:** `reformo`
- GPU assignment: `SLURM_LOCALID` → `cuda:{local_id}`
- File partitioning: `SLURM_ARRAY_TASK_ID` + `SLURM_ARRAY_TASK_COUNT`
- All phases are **safe to re-run** — each script skips already-completed output files

---

## Key Data Paths

All data lives under `$DATA = /e/data1/datasets/playground/mmlaion/shared/nguyen38`.

| What | Path |
|------|------|
| FineVideo HF dataset | `/e/scratch/reformo/nguyen38/finevideo_disk` |
| Intermediate pose outputs | `$DATA/outputs/` |
| `training_ready` JSONL (Step A output) | `$DATA/window8_legacy/FineVideo-VLA/training_ready_rank_*.jsonl` |
| Adaptive merged JSONL (Step H output) | `$DATA/window8_legacy/FineVideo-VLA/final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl` |
| Flat Megatron JSONL (Step I output) | `$DATA/window8_legacy/FineVideo-VLA/megatron_dataset_adaptive/flat_*.jsonl` |
| Phase 5 adaptive tokens | `$DATA/outputs/agent_tokens_adaptive/{video_id}_tokens.jsonl` |
| HF upload staging (merged) | `$DATA/window8_legacy/FineVideo-VLA/hf_upload_adaptive/` |
| HF upload staging (flattened) | `$DATA/window8_legacy/FineVideo-VLA/hf_upload_flattened_adaptive/` |

---

## Vocabulary & Tokenizer

`tools/tokenizer/expand_vocab.py` extends the GPT-NeoX-20b base (`vocab/gpt-neox-20b-vocab.json`) with:

| Token range | Count |
|-------------|-------|
| `<agent_0>` … `<agent_255>` (legacy) | 256 |
| `<avclm_0>` … `<avclm_8191>` | 8,192 |
| `<seed2_0>` … `<seed2_8191>` | 8,192 |
| `<cosmos_0>` … `<cosmos_63999>` | 64,000 |
| `<fps_1>` … `<fps_60>` | 60 |
| `<{joint}>` / `</{joint}>` wrappers | 34 (17 × 2) |
| `<{joint}_x_N>`, `_y_N`, `_z_N` (0–255) | 13,056 |
| `<{joint}_t_N>` (0–7) | 136 |
| Wrapper tags (`<agent>`, `</agent>`, …) | 8 |

Output: `vocab/vocab_expanded.json` (JSON lookup only).

**Important:** The vocab JSON alone does not make BPE tokenizers treat these as atomic tokens. A separate HuggingFace tokenizer was created using `tokenizer.add_tokens(special_tokens=True)` and published at [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive). This tokenizer must be used for Megatron-LM tokenization — the base GPT-NeoX-20b tokenizer will incorrectly split tokens like `<seed2_1137>` into sub-pieces.

```bash
python tools/tokenizer/check_vocab.py       # verify vocab_size (rounds to nearest 128 for Megatron)
python tools/upload/upload_tokenizer.py     # create + upload HF tokenizer
```

---

## Useful Commands

```bash
# Flatten the adaptive merged dataset (Step I)
python pipeline_pose/phase7_flatten.py

# Upload flattened dataset to HuggingFace
export HF_TOKEN='hf_...'
python tools/upload/upload_flattened_hf.py

# Upload the VLA tokenizer to HuggingFace
python tools/upload/upload_tokenizer.py

# Upload merged agent dataset to HuggingFace
python tools/upload/upload_vla_agent_hf.py

# Decode a random agent token block from a final_vla file
python tools/eval/decode_agent_tokens.py --seed 42

# Sanity-check a flat Megatron dataset
python tools/inventory/check_flattened_data.py

# Count tokens (per modality) for a new external HF dataset before deciding to integrate it
python tools/inventory/peek_multimodal.py --only some_file.jsonl.gz
python tools/inventory/count_multimodal_tokens.py --sample-mb 75

# Render a skeleton-only video from a states JSONL
python tools/visualize/render_filtered_skeleton.py \
    --video-real videos/sample.mp4 \
    --jsonl outputs/states_jsonl/sample_states.jsonl \
    --output outputs/skeleton.mp4
```

---

## Third-Party Dependencies

- **[MotionBERT](https://github.com/Walter0807/MotionBERT)** — 3D pose lifting (Step C). Checkpoint: `third_party/MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin`
- **HRNet** — model weights in `hrnet_storage/` (gitignored)
- **YOLO** — `yolo26n.pt` expected in working directory for Step F
- **Cosmos tokenizer weights** — download via `pipeline_video/download.py`; stored in `pipeline_video/pretrained_ckpts/` (gitignored)
- **Seed2 model weights** — `pipeline_video/seed2/model.safetensors` and `ae.safetensors` (gitignored)
