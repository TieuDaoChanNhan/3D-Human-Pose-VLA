# PAB-Spline VLA / Omni-Modal Project — Project Overview

*Last updated: 2026-07-26. Written for the internal team and leadership (Huu, Van Khue, technical collaborators) — technical terms are used where needed, with explanations for readers who don't touch the code day-to-day.*

---

# PART 1 — QUICK OVERVIEW

## 1. What this project is

This project builds a **large language model (LLM) trained to be omni-modal**: a single neural network (Qwen3 backbone) that can read/understand and generate multiple "modalities" — video, still images, audio, 3D human body motion (pose/action), and text — by converting ALL of these data types into **tokens** (exactly how an LLM already treats words) and interleaving them in the same training sequence.

The name "PAB-Spline VLA" (Pose-Action-Behavior Spline, Vision-Language-Action) comes from the project's original narrow framing: "train a humanoid robot from human videos." That name is still used for the video+pose branch (the largest, most mature branch), but **the project's actual scope has since broadened** — see section 2.

## 2. Motivation & the scope pivot

Originally, the goal was: use ~40,000 public YouTube videos (the FineVideo dataset) to extract real 3D human motion, tokenize that motion into "agent tokens," then train a single LLM to generate plausible actions from a text description — a step toward teaching humanoid robots to imitate human behavior.

On 2026-07-20, Huu (project lead) directly confirmed the actual scope is much broader:

> *"omni means all modes: image, video, sound, action, imu, etc. as long as we balance the dataset and create cross-modal bindings."*

In other words, the acceptance criteria for a new data source is not "does it contain video/pose/action," but:
1. **Permissive license** (reusable, no copyright entanglement) — a hard rule, not up for re-litigation.
2. **Balanced modality mix** — no single data type should dominate and dilute the others.
3. **Real cross-modal binding** — the data must help the model learn a genuine relationship between modalities (e.g., audio that truly matches the depicted image/action), not just a grab-bag of unrelated data types stacked together.

The video+pose branch (FineVideo/OmniVideo) remains the largest and most developed, but it is no longer the entire scope — the project has also added synthetic image+caption data (`synth_llava`), synthetic speech+text (`laion/emotional-roleplay`), and a large audio+text corpus (`MV-Omni`).

## 3. Architecture at a glance — "everything is a token"

The core idea (similar to AnyGPT — a comparable published system — but extended further on the "action" side):

- Each modality has its own tokenizer that turns raw data into a sequence of integer token ids:
  - **Seed2** — still images/keyframes → 8,192 possible tokens
  - **Cosmos** — short spatio-temporal video clips → 64,000 possible tokens
  - **Agent** — 3D human motion (17 joints, H36M skeleton standard) → position/time tokens, encoded via adaptive PCHIP (see the technical section in Part 2)
  - **SNAC / Listen / Speak** — audio → discrete tokens from the SNAC (Orpheus) audio codec
  - **Text** — plain text (captions, questions, dialogue)
- All these token families are merged into **one shared vocabulary** for a single tokenizer (currently Qwen3-based, ~274,561 real tokens) — every VLA token (e.g. `<seed2_1137>`, `<pelvis_x_128>`) is registered as **one atomic token**, unlike ordinary BPE, which would otherwise shred an unfamiliar string into sub-pieces.
- A training record is one text sequence interleaving multiple modality blocks in temporal order, each wrapped in its own open/close tag (`<seed2>...</seed2>`, `<cosmos>...</cosmos>`, `<agent>...</agent>`, `<listen>...</listen>`, `<caption>...</caption>`...) — giving the model an explicit "this block ended, a new modality begins" signal.
- Current backbone: **Qwen3 1.7B** (actually ~1.97B parameters once the expanded-vocab embedding is added), trained on the JUPITER cluster (GH200 GPUs) via Megatron-LM.

## 4. Data sources (as of 2026-07-26)

| Source | Primary modality | Scale (real tokens, latest mix) | Role |
|---|---|---|---|
| **FineVideo-VLA** | video + pose + audio + text | ~10.9B | Flagship branch — real YouTube video + extracted 3D pose |
| **OmniVideo-100K** | video + pose (subset) + audio + QA | ~1.98B | Ready-made video + multimodal question-answer pairs |
| **Harmony4D** | pose (close two-person interaction) | ~0.32B (20x oversampled) | Fills the occlusion/multi-person gap that single-camera video (FineVideo) can't cover |
| **MV-Omni** (MixtureVitae-Omni) | audio + image + text | ~20.4B | The largest source — the main "language backbone" contributor, no pose/video |
| **synth_llava / synth_llava2** | still images + caption | ~0.1B | Synthetic image+caption pairs, the project lead's own data |
| **laion/emotional-roleplay** | audio (synthetic speech) + text | ~0.11B | Audio↔text, teaches the model both a "listen" (`<listen>`) and "speak" (`<speak>`) role |

All six sources were just rebuilt on a **window=8** architecture (each pose/video chunk spans 8 frames) with the new audio format (`<listen>`/`<speak>` instead of bare `<snac>`) — a combined **33.83 billion real tokens**, all published publicly on HuggingFace (the `EmpathicRobotics` org).

## 5. Training history highlights (v1 → v6)

| Version | Architecture | Test PPL (lower is better) | Notes |
|---|---|---|---|
| v1/v2 | window=8, GPT-NeoX then Qwen3 | **v2: 5.77** | v2 remains the strongest baseline the project has ever achieved |
| v3 | window=24, too many variables changed at once | 27.58 | Clear regression |
| v4 | window=24, fixed 2 bugs (drop_cosmos, doc-packing) | 15.78 | Better than v3 but still far behind v2 |
| v5 | window=24, reduced drop_cosmos | 16.75 | No PPL improvement, only changed cosmos-continuity behavior |
| **v6** (seq4096/8192/16384) | **window=8 rebuild** (`w8_new` mix, v2's architecture + Harmony4D/MV-Omni/roleplay added) | **5.98 – 6.36** | Closes back in on v2, recovering nearly everything lost across v3-v5 |

**Biggest lesson**: v3/v4/v5's problem was never "window=24 is inherently worse" — it was that **too many variables changed simultaneously** (window size, source mix, dropout, token cost), making the root cause impossible to isolate. v6 returned to the recipe already proven to work in v2 (window=8) and only added new data — and PPL recovered almost completely.

## 6. The most important finding: the instruction-following gap

Across **every version tested (v2 → v6, no exception)**: when given a completely novel description (never seen in training, e.g. "A person is running in a park"), the caption the model generates **always misses the topic** (inventing things like "riding a mountain bike," "riding a slide"...) — the model is good at REPRODUCING exactly what it has seen in training, but has not learned to reason/condition on genuinely new text.

This is the **one gap no version has fixed**, despite trying many different pretraining variables (window size, dropout, sequence length, source mix). The diagnosed root cause, via comparison with the AnyGPT paper: the project is entirely missing one type of data — **instruction-tuning data** (diverse synthetic dialogue that teaches the model to condition correctly on a new request), analogous to the AnyInstruct-108k dataset AnyGPT used.

## 7. Current status

- **Only one publicly released model**: `EmpathicRobotics/vla-1.7b-qwen3-v2` (still the best-performing one; v3-v6 remain internal checkpoints, not yet published).
- **6 datasets published/being published to HF** (latest version, window=8 + listen/speak): `FineVideo-Phase7-Flattened`, `harmony4d-flattened`, `omnivideo-100k-final`, `MV-Omni`, `synth-llava`, `emotional-roleplay-finetuning-dataset-flattened`.
- **Tokenizer**: `EmpathicRobotics/tokenizer-vla-qwen3-v2` (274,561 real vocab) is the current recommendation — supports every modality token plus `<listen>`/`<speak>`.
- A reusable eval suite now exists (sanity/atomicity, temporal-continuity, full-chain text-to-media) to compare versions objectively.

## 8. What's next (short roadmap)

1. **Highest priority**: build "VLA-Instruct" — a dedicated instruction-tuning SFT stage (retrieval-based: an LLM generates diverse dialogue scripts → placeholders get replaced with real, semantically-matched tokens pulled from already-tokenized data), aimed squarely at the instruction-following gap.
2. Pick one seq_length for v6 (seq8192 currently leads) once multi-seed verification is done.
3. Consider integrating DROID (real robot action data, Apache 2.0) — requires designing a dedicated 7-DoF action vocabulary; lower priority than #1.
4. Keep license/eval discipline as new data sources are added.

---

# PART 2 — FULL DETAIL

## 2.1. Full background & motivation

### Phase 1 — "VLA for humanoid" (before 2026-07-20)

The original goal: use public YouTube video (the **FineVideo** dataset, ~40,000 videos, released by HuggingFaceFV) to:
1. Extract 3D human motion (pose) from ordinary 2D video (no special camera rig needed).
2. Tokenize that motion into "agent tokens" — a kind of "action language" an LLM can generate the same way it generates text.
3. Combine this with image tokens (Seed2), video tokens (Cosmos), audio tokens, and caption/text, and train one LLM that can both understand and generate video and action from a text description.

The underlying bet: if an LLM can "speak" plausible human motion learned from tens of thousands of hours of real video (far cheaper than collecting real robot data), that is a viable path toward training humanoid robots to mimic human behavior, using closed-loop simulation as the ultimate yardstick.

### Phase 2 — Pivot to omni-modal (2026-07-20)

Huu directly confirmed via Discord that the project's real scope is **omni-modal** — binding any combination of modalities (image, video, audio, action, IMU...), not necessarily tied to robots/human action at all. The acceptance criteria for a new data source: (1) permissive license, (2) balanced modality proportion within the total dataset, (3) real cross-modal binding (not just stacking unrelated data types).

Two sources were added exactly under this new criteria (containing no video/pose/action at all):
- `synth_llava`/`synth_llava2` (`mixture-vitae-backup/MixtureVitae-Backup`) — ~604K synthetic image+caption pairs, the project lead's own data, used to feed `<seed2_N>` tokens (Seed2 is the only tokenizer that accepts a standalone image, no video needed).
- `laion/emotional-roleplay-finetuning-dataset` — 67,491 synthetic TTS speech clips + text, feeding SNAC/audio tokens.

The FineVideo/OmniVideo branch (video+pose+action) is still the largest and carries the original "VLA for humanoid" framing — but it is no longer the whole scope. One internal concern has been flagged: as the data scope expands quickly, eval-protocol/research discipline needs to keep pace, to avoid accumulating "eval debt" (adding new sources without a matching way to measure quality).

## 2.2. Full system architecture

### 2.2.1. Tokenizing each modality

| Modality | Tokenizer | Vocab size | Notes |
|---|---|---|---|
| Seed2 (image/keyframe) | Seed2Tokenizer (based on Stable Diffusion 2.1-unclip) | 8,192 | 1 frame → 32 tokens, kept at 100% (no dropout) |
| Cosmos (short video) | NVIDIA Cosmos tokenizer | 64,000 | Each 8-frame chunk → 200 tokens (window=8) or 896 tokens (window=24, now retired); kept at 50% (random per-chunk dropout to balance modality proportions) |
| Agent (3D pose) | Custom — Adaptive PCHIP | variable, across 17 joints × (t, x, y, z) | See details in 2.2.3 |
| SNAC/Listen/Speak (audio) | SNAC (Orpheus, `hubertsiuzdak/snac_24khz`) | 3 bands: L0 (128266-132361), L1a (132362-136457), L1b (144650-148745), 12,290 tokens total | `<listen>` = listening (input), `<speak>` = speaking (model's own generated turn) — a role distinction, not a format distinction |
| Text (caption/speech/dialogue) | Standard BPE (Qwen3 base) | remaining vocab | No dedicated token family needed — natural text shares the base BPE vocabulary |

Every modality token is registered via `add_tokens(special_tokens=True)` to guarantee **atomicity** (never split into sub-pieces). This fixes an important early bug: without proper registration, `<seed2_1137>` gets split into 7 separate BPE pieces — the project's very first model (`vla-1.7b-pab-spline-25b-test`) shipped with exactly this bug.

### 2.2.2. Tokenizer & vocab history

| Tokenizer | Base | Vocab | Used for |
|---|---|---|---|
| `tokenizer-vla-adaptive` | GPT-NeoX-20b | 144,215 | v1 — agent tokens only, no SNAC yet |
| `tokenizer-vla-adaptive-v2` | GPT-NeoX-20b | 156,509 | v3-v6 (GPT-NeoX branch) — adds SNAC (bare `<snac>`) + caption/speech |
| `tokenizer-vla-qwen3` | Qwen3 | 257,901 | First Qwen3-based tokenizer, used by the v2 model |
| **`tokenizer-vla-qwen3-v2`** | Qwen3 | **274,561** (padded to 274,688) | **Current recommendation** — adds the `<listen>`/`<speak>` wrapper (replacing bare `<snac>`), 2 newly-discovered SNAC bands found by streaming real data |

### 2.2.3. Agent (3D pose) token format — Adaptive PCHIP

Each time window (8 or 24 frames) of one person in a video is encoded as a token sequence across 17 H36M-standard joints (pelvis, hips, knees, ankles, spine, shoulders, elbows, wrists, head...):

```
<fps_30>
<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
         <pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
<r_hip>  <r_hip_t_0>  <r_hip_x_140> ...  </r_hip>
...17 joints...
```

- Each joint is represented with **2, 4, or 8 control points**, depending on how curved the motion is (a nearly-static joint → 2 points; complex motion → 8 points) — this is the "adaptive" idea, saving tokens on simple motion.
- x/y/z coordinates are quantized to uint8 integers (0-255), mapping the range [-2.0m, +2.0m] around the root (pelvis-centered).
- To reconstruct all 8 (or 24) original frames from the sparse control points, the pipeline uses **PCHIP** interpolation (Piecewise Cubic Hermite Interpolating Polynomial) — smoother than linear interpolation and free of the overshoot ordinary cubic splines can produce.

### 2.2.4. Sequence format & the window=8 vs window=24 history

A training record is one text sequence starting with a header (`### Title:`, `### Context:`, `### Keywords:`, `### Speech:`), followed by interleaved modality blocks in temporal order, one group per 8 (or 24) frames:

```
chunk 0: [caption?] [seed2?] [cosmos] [agent?] [listen?] [speech?]
chunk 1:            [cosmos] [agent?] [listen?]
...
```

The project once moved from window=8 to window=24 (mid-2026), expecting "a longer window = more context per step for the model." However, experimental results (see 2.4) showed **window=8 (the v2 checkpoint) outperforming every window=24 successor** on every measured axis — not because window=24 is inherently worse, but because the move to window=24 bundled in far too many simultaneous changes (cosmos costing 4.5x more tokens/chunk → forcing heavier dropout, a smaller corpus, a new wrapper format...). As a result, **v6 (2026-07-26) reverted to window=8** as the standard architecture, only adding new data on top of the recipe already proven to work.

## 2.3. Full data pipeline, source by source

### FineVideo-VLA (flagship branch, video+pose)
- **Origin**: ~40,000 YouTube videos from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) (HuggingFaceFV), Apache 2.0, content is self-contained.
- **Pipeline**: Step A (tokenize video → seed2/cosmos) → Phase 1 (HRNet 2D joint detection) → Phase 2 (MotionBERT 2D→3D lifting) → Phase 3 (kinematic normalization/noise filtering) → Phase 4 (YOLO person-detection filtering) → Phase 5 (adaptive PCHIP agent tokenization) → Phase 6 (merge agent+audio into the video stream) → Phase 7 (flatten into Megatron-ready JSONL).
- **Current scale (v7, window=8, listen/speak)**: 371,892 records, 10,926,767,551 real tokens.
- **HF repo**: `EmpathicRobotics/FineVideo-Phase7-Flattened` (final dataset), plus intermediate repos (`FineVideo-Prototype-Tokenized`, `FineVideo-Phase2-3DPose`, `FineVideo-Phase4-YOLOPose`, `FineVideo-Phase5-AgentTokens`).

### OmniVideo-100K
- **Origin**: [MiG-NJU/OmniVideo-100K](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K), Apache 2.0, real 52.9GB.
- **Content**: video + ready-made multimodal QA (99,983 QA pairs: 70,017 open-ended + 29,966 multiple-choice); pose (agent tokens) only ran on a sports subset (~15% of videos have agent tokens).
- **Current scale (window=8)**: 5,214 records (one per video), 1,979,126,756 real tokens.
- **HF repo**: `EmpathicRobotics/omnivideo-100k-final`.

### Harmony4D
- **Origin**: [Harmony4D](https://jyuntins.github.io/harmony4d/) — multi-camera motion capture of close two-person interactions (hugging, martial arts, sword fighting, ballroom dancing).
- **Role**: fills a gap FineVideo's single-camera pipeline can't cover — occlusion and multi-person interaction. FineVideo has to discard ~56% of windows to occlusion/hallucination filters; Harmony4D is ground-truth multi-camera data, with 416/416 tracks passing clean (no need for the filters designed for monocular-estimation error).
- **Oversampling**: 20x (416 physical tracks is tiny relative to the rest of the mix, so it's replicated 20x per Van Khue's decision) — a real bug was found and fixed here: Megatron automatically internally repeats any source whose weighted sample target exceeds its physical content (`_get_num_epochs`), which had silently pushed the effective oversampling to ~30.7x instead of the intended 20x; fixed by giving Harmony4D its own dedicated weight (equal to its exact physical share) instead of a bucket-proportional split.
- **Current scale (window=8)**: 8,320 records (416 tracks × 20x), 315,545,360 real tokens.
- **HF repo**: `EmpathicRobotics/harmony4d-flattened`.

### MV-Omni (MixtureVitae-Omni)
- **Origin**: HF `mixture-vitae/MixtureVitae-Omni` (`valid_snac` split) — 1,593,301 real records (verified by fully decompressing all shards on 2026-07-26, correcting an old, never-actually-counted "~1.78M" estimate).
- **Content**: audio (SNAC) + image (Seed2, converted from the original `<seed_N>` tags) + text formatted as questions ("Q: Listen to this and tell me what you heard..."). **No pose or cosmos content**.
- **Role**: the largest single source in the whole mix (39.58% of the `w8_new` mix's total weight) — the main contributor to the "language backbone" and audio↔text binding.
- **Scale (real tokens, used in v6)**: 20,389,561,883 tokens.
- **License note**: no documented license confirmation was found in this project's records (unlike `synth_llava`, which has an explicit "confirmed permissive by Huu directly" note) — both the `mixture-vitae` and `mixture-vitae-backup` orgs are controlled by the project lead, so this is presumed to be his own data, but tagged `license: other` (no specific SPDX id) per Van Khue's 2026-07-26 decision, pending a more explicit confirmation if ever needed.
- **HF repo**: `EmpathicRobotics/MV-Omni` (newly uploaded).

### synth_llava / synth_llava2
- **Origin**: `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` — synthetic image+caption data (llava_pretrain-style) created by Huu himself, directly confirmed permissive (2026-07-21).
- **Scale**: 603,999 records (56 `synth_llava` shards + 95 `synth_llava2` shards), 19,327,968 seed2 tokens (32 tokens/image).
- **Role**: adds pure `<seed2_N>` token volume (still images), no video/action.
- **HF repo**: `EmpathicRobotics/synth-llava`.

### laion/emotional-roleplay-finetuning-dataset
- **Origin**: [laion/emotional-roleplay-finetuning-dataset](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset), CC-BY-4.0.
- **Content**: 67,459/67,491 records (32 rows dropped for out-of-range `adherence_score`) — synthetic TTS speech (MOSS-TTS-Local v1.5), multilingual (German-majority), ~184 hours of audio, encoded in "speak" SNAC format (7 tokens/12.5Hz frame, all 3 codebook levels — unlike "listen" format, which drops the finest level).
- **Role**: teaches the model the `<speak>` role (the model's own generated speech) — complementing `<listen>` (already present from FineVideo/OmniVideo).
- **Scale**: 54,578,440 SNAC tokens.
- **HF repo**: `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened`.

### Sources investigated but NOT used (license reasons)

| Source | Reason for exclusion |
|---|---|
| stera-10m | Not permissive (Huu + Van Khue consensus, 07/18) |
| AgiBot World | CC BY-NC-SA 4.0 (NonCommercial) |
| Apple EgoDex | CC-BY-NC-ND (NonCommercial + No-Derivatives) — a shame, since it matches the use case closely |
| Meta ego-1k / EgoBrain | FAIR Noncommercial / CC-BY-NC; EgoBrain is also off-topic (EEG, not robotics) |
| JRDB-Pose3D | Non-commercial license |
| SenseNova-SI-8M | Original image license could NOT be verified (despite an Apache-2.0 HF tag) — shelved |
| Open X-Embodiment | A registry of 55-60 sub-datasets with inconsistent licenses — needs a per-sub-dataset audit |
| MINT-1T-HTML (image portion) | URL-only, no way to trace the underlying image license — image portion dropped, text kept |

### Candidates under consideration (not yet integrated)
- **DROID** (`nvidia/Cosmos3-DROID`) — real robot action data, OpenMDW-1.1 license (commercial use allowed), 707GB — needs its own 7-DoF action vocabulary (unlike the 17-joint human agent-token scheme).
- **NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim** — CC-BY-4.0, 345K+ simulated humanoid/robot-arm trajectories — the strongest robot-action candidate found so far.
- **IPEC-COMMUNITY/EO-Data1.5M** — Apache 2.0, 1.5M samples, exactly the interleaved vision-language-action format needed; can't be merged directly (robot action-space differs from the human agent-token scheme) but is the best available template for VLA-Instruct.

## 2.4. Full v1 → v6 training history

| Version | Date | Architecture | Train iters | Test PPL | Key lesson |
|---|---|---|---|---|---|
| v1 | ~Mar 2026 | window=8, GPT-NeoX, agent only | — | — | Tokenizer bug (BPE splitting VLA tokens) — the first model shipped with this bug |
| v2 | Jun 2026 | window=8, Qwen3, 5 sources | 7,632 | **5.77** | The project's strongest baseline to date — no later version has beaten it |
| v3 | 2026-07-02 | window=24, window+wrapper+corpus+drop_cosmos all changed at once | 881 | 27.58 | A severe regression — too many simultaneous variables, root cause not isolated |
| v4 | 2026-07-24 | window=24, same as v3 + more training steps | ~2032+ | 15.78 | Better than v3 thanks to longer training, not a fix of the actual root cause |
| v5 | 2026-07-25 | window=24, drop_cosmos reduced 0.85→0.5 | comparable to v4 | 16.75 | No PPL change (rejects the "cosmos density" hypothesis), but clearly improved cosmos-continuity over time |
| **v6-seq4096** | 2026-07-26 | **window=8 rebuild** (`w8_new`, 6 sources, v2's architecture) | 8,065 | 6.36 | Closest match to v2's own recipe, best at returning to `<agent>` (4 consecutive blocks, never seen in any other version) |
| **v6-seq8192** | 2026-07-26 | window=8 rebuild, seq_length=8192 | 4,033 | **5.98** | Most balanced — best PPL among the 3 v6 variants, all 5/5 modalities present in the full-chain test |
| **v6-seq16384** | 2026-07-26 | window=8 rebuild, seq_length=16384 | 2,016 | 6.12 | Weakest of the 3 v6 variants on every axis — likely due to having the fewest training iterations (same token budget, longer sequence → fewer steps) |

**A notable operational incident during v6 training**: an accidental double-submit briefly created 5 SLURM jobs instead of 3 (caught and cancelled in time, no checkpoint affected); 2 of the 3 jobs' `current.log` symlinks pointed to the wrong (cancelled) job — reading only the symlinked log would have looked like a training failure, but the real per-job-id logs (via `sacct`) confirmed all 3 real jobs COMPLETED cleanly.

## 2.5. Full evaluation methodology

The project uses three complementary eval types — none substitutes for the others:

### (a) Sanity / token atomicity
Checks two things: (1) whether every VLA token tokenizes to exactly one token id (atomicity), and (2) whether the model correctly completes structure when primed with a real partial record (e.g., primed with 3/17 real joints of an agent block, expecting the model to complete the remaining 14 joints with plausible values). Script: `eval_vla_v2/v3/v6_sanity.py` — each window-size era needs its own matching real priming record (window=8 differs from window=24); reusing the wrong one tests something the model was never trained on.

### (b) Temporal continuity
The question: when primed with a single real window/chunk and left to sample-generate a long continuation, does the model **actually return to that same modality** over time, and does the newly generated content genuinely progress (rather than freezing/repeating)? Script: `eval_temporal_continuity.py`, measured via displacement between consecutive pose windows (agent) and token-position overlap (cosmos).

**Key finding**: v2 (window=8) is markedly more durable than v3/v4/v5 (window=24) at returning to `<agent>`/`<cosmos>`. v6-seq4096 achieved, for the first time, **4 consecutive non-frozen agent blocks** in a single generation — the best result to date, though so far verified on only 1 seed (more seeds needed to rule out chance).

### (c) Full-chain text-to-media (the real "text → multimedia" test)
Given a COMPLETELY NEW prompt (absent from any training record), let the model sample-generate a modality chain, then **actually decode** every generated modality into a viewable file: seed2→PNG image, cosmos→MP4 video, agent→pose JSON + skeleton image, listen/speak→WAV audio. Script: `gen_full_chain_v3.py`.

**Key finding (and the single most important finding of the whole project)**: tested with the prompt "A person is running in a park" on v2 and all 3 v6 variants — **no version generated an on-topic caption** (all invented "riding a bike/slide/raft/airplane"). This is the direct evidence behind the instruction-following gap described in section 6 of Part 1.

## 2.6. Infrastructure & tooling

- **Compute cluster**: JUPITER, `booster` partition — GH200 nodes, 4 GPUs/node, 288 CPU cores/node, account `reformo`.
- **Training framework**: Megatron-LM (via the `oellm-autoexp` wrapper), Apptainer container (binds only `/e`, cannot see `/p`).
- **Two separate Python environments** (never mixed): the prototype tokenization pipeline (Seed2/Cosmos/AVC-LM) and the 3D pose pipeline (HRNet/MotionBERT).
- **Standardized data locations** (after the 2026-07-25 reorg): `window8_legacy/` (old version), `window24_current/` (the w24 experiment, no longer "current" in practice despite the name), `w8_new/` (the newest mix used to train v6) — all under `/e/data1/datasets/playground/mmlaion/shared/nguyen38/`.
- **Main repo**: `3d-human-pose/` (video+pose pipeline, holds all of `data_prep/`, `tools/eval/`, `tools/upload/`), kept separate from the `oellm-autoexp/` training repo.

## 2.7. Known limitations & open problems

1. **Instruction-following (priority #1)** — as described above, the largest gap, present in every version, unfixed by any pretraining change tried so far.
2. **Modality persistence (improved but not yet stable)** — v6 shows clear improvement but most of the evidence is from a single seed; more seeds are needed to confirm it isn't chance.
3. **Window=8's own contribution hasn't been isolated from everything that changed alongside it** — v2 and v6 differ in MANY variables at once (source mix, audio wrapper, Harmony4D oversampling...); no single-variable ablation isolating window size alone has been run yet.
4. **No real robot-action data yet** — all current agent tokens come from estimating human pose from video (not real robot data); DROID/GR00T-Sim are candidates but not yet integrated.
5. **The 1.7B model may be hitting a capacity ceiling** — packing this many modalities into a relatively small model has a real cost (AnyGPT reports the same phenomenon) — no clear evidence yet distinguishing "needs more data" from "model is too small."
6. **Eval discipline hasn't kept pace with data-source growth** — every new source should ship with a minimum eval alongside it, to avoid repeating the situation where a new eval script had to be written only after training had already finished (as happened with w8_new).

## 2.8. Detailed roadmap

1. **VLA-Instruct** (highest priority) — design: (a) use an LLM to generate diverse dialogue scripts as text placeholders (following the AnyInstruct recipe: ~100 meta-topics → tens of thousands of concrete topics); (b) **retrieval, not generation** — use existing caption embeddings to find real clips/pose-windows/audio that semantically match, in already-tokenized data, replacing the placeholders with real tokens; (c) train this as a **dedicated SFT stage after pretraining** (matching AnyGPT's own recipe: pretrain then instruction-tune), not mixed into the current 64-node pretraining blend.
2. Pick one seq_length for v6 after multi-seed verification (seq8192 currently has the best PPL and full 5-modality coverage in the full-chain test).
3. Consider integrating DROID — requires a dedicated 7-DoF action vocabulary, lower priority than VLA-Instruct.
4. Publish the chosen v6 checkpoint to HF as `vla-1.7b-qwen3-v6` (not yet done — currently only an internal checkpoint).
5. Maintain discipline going forward: every new data source needs (a) license verification first, (b) a minimum eval alongside it, (c) never deleting an older data version when a newer one arrives (keep it for comparison/rollback).

## 2.9. Glossary

| Term | Explanation |
|---|---|
| **VLA** | Vision-Language-Action — a model trained jointly on vision, language, and action |
| **Token** | The smallest unit an LLM processes — usually a word/word-piece; here extended to also represent images/video/audio/action |
| **Tokenizer** | The rule set that converts raw data (text, image, audio...) into a token sequence |
| **Atomic token** | A token that is never split further during tokenization (important so the model learns 1 token = 1 VLA unit) |
| **PPL (Perplexity)** | A measure of how "surprised" the model is by real data — lower is better, meaning the model predicts real data more accurately |
| **Window** | The number of frames grouped into one time "window" when tokenizing video/pose (8 or 24 frames) |
| **Modality drift** | The model "drifting" into a different modality and never returning to the original one, even when the real ground truth does return |
| **Instruction-following** | The model's ability to correctly follow a NEW request/description (never seen before), rather than just reproducing memorized training data |
| **Oversampling** | Replicating a small data source N times to raise its share of the overall dataset |
| **Sampled vs Greedy decoding** | Greedy = always pick the highest-probability token (prone to repetition loops); Sampled = pick randomly, weighted by probability (more natural, less repetitive) |

## 2.10. Appendix — Resources & paths

**HuggingFace models** (`EmpathicRobotics` org):
- `vla-1.7b-qwen3-v2` — the current best publicly released model
- `vla-1.7b-pab-spline-adaptive`, `vla-1.7b-pab-spline-25b-test` — first-generation models (tokenizer bug)
- `tokenizer-vla-qwen3-v2` — the current recommended tokenizer

**HuggingFace datasets** (`EmpathicRobotics` org):
- `FineVideo-Phase7-Flattened`, `FineVideo-Phase5-AgentTokens`, `FineVideo-Phase4-YOLOPose`, `FineVideo-Phase2-3DPose`, `FineVideo-Prototype-Tokenized`
- `harmony4d-flattened`, `omnivideo-100k-final`, `MV-Omni`, `synth-llava`, `emotional-roleplay-finetuning-dataset-flattened`

**Code repo**: `3d-human-pose/` (main pipeline) — see `CLAUDE.md` (technical guide), `datasets.md` (detailed data inventory), `PROGRESS_VI.md`/`REPORT.md` (full chronological development log).
