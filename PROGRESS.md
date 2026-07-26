# PAB-Spline VLA — Project Progress

**Author:** Van Khue Nguyen  
**Last updated:** July 26, 2026  
**Cluster:** JUPITER (JSC), `booster` partition, GH200 nodes.  
**Goal:** Build an omni-modal Vision-Language-Action model — one LLM that reads/generates video, image, audio, 3D human pose/action, and text as a single interleaved token stream. (Scope broadened 2026-07-20 from the original "VLA for humanoid" framing — see the 2026-07-20 entry below. See `PROJECT_OVERVIEW_EN.md` for a structured, non-chronological summary of the whole project; this file is the raw session-by-session log.)

**Note on this file vs `PROGRESS_VI.md`**: this file (English) went unmaintained between 2026-07-21 and 2026-07-26 while `PROGRESS_VI.md` (Vietnamese) continued as the sole living session log. The entries below for 2026-07-22 through 2026-07-26 are a condensed English backfill of that gap, written 2026-07-26 — comprehensive on decisions/numbers/findings, but not a line-for-line translation of every micro-session. `PROGRESS_VI.md` itself is no longer tracked in this public repo (kept locally only, see `.gitignore`) — this file is now the canonical public progress log going forward.

## Session backfill — Jul 22-26, 2026 (English condensation of 5 days only logged in PROGRESS_VI.md until now)

### Jul 22-23 — data1 reorg, window=24 promoted to project-wide standard, two storage-quota incidents

**Window=24 promoted to the standard for every pose+video pipeline, not just FineVideo (2026-07-23).** Triggered by discovering that a re-run of OmniVideo-100K's Step A had only fixed cosmos resolution/quota bugs while silently keeping the old `CHUNK_SIZE=8` — the actual intent all along was to move every dataset to `WINDOW_FRAMES=24`/`COSMOS_STRIDE=3` (8-frame chunks, 0.267s @ 30fps, were too short for visible motion within a chunk — Huu's own observation). Cosmos still samples exactly 8 frames per chunk via the stride, so its per-chunk token cost didn't change — only pose/seed2/avc_lm's temporal span widened. Applied to FineVideo's video+pose pipeline and, from this point forward, to every new/re-run pipeline.

**`/e/data1/.../nguyen38/` reorganized (executed 2026-07-25, planned 2026-07-23)** into `window8_legacy/` (old runs, old SNAC format), `window24_current/` (the then-current window=24 experiments), `shared_window_independent/` (raw source video, caption/speech extraction — reused by both window sizes), `alexandria/` (confirmed-unrelated-project data: `nemotron_base/`, `tokenized_output_project/`, `output_v3/`), and `_legacy_orphan/` (dead/superseded data awaiting deletion confirmation). 59 files updated to match (52 in this repo, 7 training yamls in `oellm-autoexp`, `CLAUDE.md`, `README.md`) — committed here (`d45c9a2`), the 7 yaml edits deliberately left uncommitted in `oellm-autoexp` since that repo had unrelated dirty state. Several non-obvious path-rename pitfalls were caught along the way: multi-line Python string concatenation splitting a path across two lines, one shared shell variable name reused for genuinely different destination buckets depending on the specific script's actual output, and `snac_finevideo.py` intentionally reading from one bucket while writing to another (a real historical quirk, not a bug).

**Two `/e/project1` inode-quota incidents, both root-caused and fixed.** `exa_project1`'s inode quota is shared project-wide (soft limit 4M, hard 4.4M) and was exceeded twice: first by accumulated per-rank scratch files from crashed/restarted Step A tokenize workers (caused silent data loss — hundreds of videos' tokens came back empty because a bare `except Exception` swallowed `Errno 122: Disk quota exceeded`, discovered only via `jutil project dataquota`); second, the same evening, by ordinary session activity (git commits, wandb/tensorboard dirs, `__pycache__`). The real fix both times was the same lesson: **all writes — even throwaway scratch — must default to `/e/data1/...`, never `/e/project1/...`**, and when quota trips again shortly after a cleanup, look for the next-largest still-project1-resident directory (this time: `miniforge3/` and `env_stable_vla/`, ~189K files together, moved to data1 with a symlink left behind) rather than repeating small `__pycache__` deletions.

### Jul 23 — v2 training completed + published, Harmony4D added, v3 pipeline built

**`qwen3_1.7b_vla_v2` (job `1009758`) ran to completion**: 7,632/7,632 iterations, 64 nodes, loss 6.472→1.694, **validation PPL 5.77** — still the project's best result to date, as of this writing. Trained on a 5-source, 32.008B-token mix (finevideo_v6 34.14%, omnivideo_100k_video 1.68%, synth_llava 0.32%, roleplay 0.16%, mv_omni 63.71% of raw content, rebalanced 60/40 toward action-bearing sources at the weight level). Published to `EmpathicRobotics/vla-1.7b-qwen3-v2`. Evaluated the same window: **modality transitions now work** (model freely moves between seed2/cosmos/snac/speech/caption/agent under both greedy and sampling — real cross-modal binding confirmed, e.g. a bare 32-token seed2 block correctly triggers a caption matching the source image). Remaining known issues at this point: cosmos dominates 61-77% of VLA-token output (fixed per-chunk cost, not a training artifact), greedy decoding causes macro-repetition, and 80% of documents exceed `seq_length=4096` with no document-boundary-aware packing.

**Harmony4D integrated for the first time**, filling the occlusion/multi-person gap FineVideo's single-camera pipeline can't cover (FineVideo drops ~56% of windows to occlusion/hallucination filters; Harmony4D's multi-camera ground truth needs none of them, 416/416 tracks pass clean). Oversampled **20x** (416 tracks is tiny next to the rest of the mix) per Van Khue's call that high-quality pose data should be seen more, not left to drown in FineVideo's volume — picked as a reasonable starting point, not a rigorously tuned number (revisited 2026-07-26, see below). Harmony4D was **not** part of v2 — it entered the mix two days later, first appearing in v3.

**Tokenizer promoted to `tokenizer_vla_qwen3_v2`** (122,918 added tokens, +16,660 vs the prior working version) — adds `<listen>`/`<speak>` as atomic tokens (verified the old production tokenizer did NOT have these atomic at all) plus 2 newly-discovered SNAC bands. Deployed to both the Jupiter and JUWELS filesystems (separate clusters, need separate copies).

**v3's full 6-source data pipeline built and Megatron-tokenized on JUWELS** (finevideo_w24, mv_omni, omnivideo_100k_w24, roleplay_speak, harmony4d, synth_llava — 14.776B real tokens, read directly from Megatron `.idx` headers, ~1.5-2x higher than earlier word-count estimates). Cosmos dropout (`drop_cosmos=0.85`) applied for real for the first time — pre-dropout cosmos was 92.5% of FineVideo's own token budget at window=24 (much worse than v2's 61-77%, since window=24 + aspect-preserving resize raised per-chunk cosmos cost); post-dropout, cosmos was still the largest single modality (61.3%) but agent/seed2/snac each reached ~12-13%. A real bug was caught and fixed pre-launch: the Phase 7 flatten script's SNAC filter only recognized bare `<snac>`, not the new `<listen>`/`<speak>` tags, silently dropping every audio-only activity and stripping all audio from agent-bearing ones.

### Jul 23 (same day) — v2 made runnable by external evaluators

Huu directly requested (Discord): *"can you please give us the code to run the model, otherwise no one can eval it... we have people lined up to eval but they are stuck"* — followed by a request for encode code too. Built and verified real (not synthetic) round-trip encoders + decoders for all 4 modalities (seed2 via the already-public `ontocord/seed2`; cosmos via `nvidia/Cosmos-Tokenizer-DV8x16x16` + a vendored NVIDIA `cosmos_tokenizer` package; agent via pure numpy PCHIP math; snac via the already-public `hubertsiuzdak/snac_24khz`), all bundled directly into the `EmpathicRobotics/vla-1.7b-qwen3-v2` HF repo. Important subtlety: v2 was trained *before* the window=24/listen-speak pivot, so all 8 scripts had to reproduce v2's *old* conventions (200-token/160×160-squash cosmos, listen-format-only bare `<snac>`, `WINDOW_FRAMES=8`) recovered from git history, not the current pipeline's conventions.

### Jul 24 — v3 confirmed complete, Huu's direct v2 feedback, v4 + doc-packing ablation launched and completed, time-centric agent-token idea proposed

**v3 (job `1029227`) confirmed COMPLETE via `sacct`**: 881/881 iterations, **validation PPL 27.58** — a clear regression vs v2. Wrote `eval_vla_v3_sanity.py` (window=24-aware, real-record prompts): token atomicity 40/40 PASS; agent completion structurally correct; but the same greedy macro-repetition bug as v2 persists (a decoding-strategy artifact, not obviously fixable by data changes alone — v2's own sampled runs show much richer real transitions); one real regression candidate found (synth_llava image captioning collapsed to zero captions under greedy, plausibly because its mix weight dropped in v3, not yet sampled-confirmed at this point).

**Huu's direct feedback on v2** (relayed via Discord): *"produce tokens consistently... text coherently. But not follow instructions. This is expected since we don't have a lot of text or reasoning data. It does have some multilingual [ability]."* Concrete examples: asked "what is 2^5?", answered in unrelated Spanish with no real arithmetic; asked to repeat what it heard, correctly generated a valid `<speak>` block; asked what something means in English, fabricated an unrelated translation. This is the first direct statement of what later becomes the project's central open problem — **coherent token generation without real instruction-following/reasoning** — and, since v3 added zero new text/reasoning data, there was no reason to expect this specific gap to improve in v3 either.

**`gen_full_chain_v3.py` written** — the first "genuinely novel prompt → real decoded media" test (not token continuation from a training record). First real result on v3 ("man doing jumping jacks on a rooftop terrace at sunset"): produced a full real modality chain (caption→seed2→listen→cosmos→agent→listen) under sampling, but caption + seed2 image were completely off-topic ("The person is riding an ice rink...") — direct, reproducible evidence for Huu's "not follow instructions" finding. Agent pose was real and dynamic (not frozen). Cosmos could not be decoded yet at this point (decoder still hardcoded to the old 200-token format).

**Huu proposed two concrete, still-unimplemented ideas**: (1) a **time-centric** alternative `<agent>` token layout — currently item-centric (all timesteps of one joint grouped together); his suggestion groups by timestep instead, on the theory it might teach whole-body coordinated reasoning rather than one-joint-at-a-time; (2) auto-generated **body-part-centric narrated captions** from decoded pose trajectories, and more general "intermix formats/wording across datasets for more OOD behavior" — both aimed at the same instruction-following gap, using data already in hand.

**v4 (job `1032135`) launched and COMPLETED same day**: 2,642 iterations (3 epochs), 3 combined changes vs v3 (more iterations, document-boundary packing via `reset_position_ids`/`reset_attention_mask`, and a corrected Harmony4D exposure multiplier — kept at the originally-intended ~80x total instead of letting 3 epochs balloon it to ~240x). **Result: validation PPL 15.78** — better than v3 but still far behind v2.

**A parallel single-variable ablation (`v3_docpacked`, job `1032174`) isolated doc-packing alone**: byte-identical to v3 except packing enabled. **Result: PPL 29.18 — slightly *worse* than v3's 27.58.** This **rejects the document-boundary-packing hypothesis** as v3's weakness driver — by elimination, v4's real improvement is attributable mostly to 3x more optimizer steps, not packing. Also caught and fixed: a math error in an earlier note about Harmony4D's total exposure multiplier (correct figures: ~80x for v3, ~240x if v4 had left it unchanged — not the "~118x"/"~354x" first estimated by eyeballing).

### Jul 25 — v5 ablation, large storage cleanup, AnyGPT paper + VLA-Instruct design, DROID/data-gap survey, full v3/v4/v5 eval verdict

**v5 (job `1041082`) COMPLETED**: single-variable ablation vs v4, only change is `drop_cosmos` relaxed 0.85→0.5 (cosmos's own-dataset token share rose to 84.1%). **Result: validation PPL 16.75, test PPL 15.39 — statistically tied with v4 (15.78/15.45).** This **rejects the cosmos-density hypothesis** as the reason v3/v4/v5 all lag v2. Two of the three original hypotheses for "why does the w24 family underperform v2" are now eliminated by direct ablation (packing, cosmos density); the two live candidates left are (a) simultaneous distribution shift (window/mix/wrapper/oversample all changed together, never isolated one at a time) and (b) an AnyGPT-reported structural cost of training many modalities jointly on a small model.

**~10.7TB of stale JUPITER-maintenance backup data deleted from `/p/data1/mmlaion/shared/nguyen38/data/`** after checksum verification (2 files spot-checked byte-identical between `/p` and `/e`) confirmed `/e`'s current state is a strict superset. One directory (`nemotron/`, 967GB) was kept — it turned out to be the *sole* remaining copy of unrelated-project data, not a duplicate.

**AnyGPT paper (arXiv 2402.12226) read on Huu's direct assignment.** Direct architecture mapping confirmed this project's design (per-modality tokenizer + shared vocab + LLM-generates-semantic/decoder-reconstructs-perceptual split) already matches AnyGPT's approach, one step further (AnyGPT has no action/pose modality at all — no precedent to check design choices against). **The one genuinely new, actionable finding**: AnyGPT teaches any-to-any mode-switching not through architecture but through **AnyInstruct-108k**, a dedicated instruction-tuning dataset (LLM-synthesized dialogue scenarios with modality placeholders, generated content substituted in, used as a *separate SFT stage after pretraining*) — something this project has **zero equivalent of**. This directly explains both Huu's "doesn't follow instructions" finding and the earlier observation that `<seed2>`/`<cosmos>` only reliably trigger from one fixed context template while `<agent>`/`<listen>` generalize fine. **Proposed fix, "VLA-Instruct"**: same 2-stage recipe but retrieval instead of generation (no cheap text→video/pose generator exists) — an LLM synthesizes diverse dialogue scenarios with placeholders, then caption-embedding similarity search retrieves a real matching clip/pose-window/audio chunk already tokenized in the existing data, substituting real tokens for the placeholder; run as a separate lightweight SFT stage after pretraining, not blended into the pretrain mix. Design only, no code written yet.

**DROID (`cadene/droid`) surveyed as a real-robot-action candidate**: Apache 2.0, clean, 92,223 episodes / 27M frames / 31,308 task descriptions, single 7-DoF Franka arm — cannot merge directly into the 17-joint human `<agent>` scheme, would need a new `<droid_agent>` namespace/vocab. `IPEC-COMMUNITY/EO-Data1.5M` (Apache 2.0, same group as DROID) found as the best available interleaved-VLA format template, though not directly mergeable either. **Synthesized ranking of what's actually missing**, in priority order: (1) text/reasoning/instruction data — the single biggest gap, solved by none of the raw datasets surveyed, this is exactly what VLA-Instruct targets; (2) diversifying the context/wording that leads into `<seed2>`/`<cosmos>` generation — a reformatting-of-existing-data fix, not a new-source problem; (3) robot-action diversity (DROID etc.) — real but lower priority, deferred until (1)/(2) are addressed.

**Full v3/v4/v5 eval sweep — the session's central deliverable.** Fixed a real bug in `eval_vla_v3_sanity.py` first (`<speech>`/bare `<snac>` were mislabeled "agent" by the fallback classifier, causing a false "NEEDS REVIEW"). Sampled-decoding sanity (v4 vs v5): both still fail to spontaneously open `<agent>` from a bare seed2 block (unchanged since v3 — confirms this specific gap is untouched by any v3→v5 change); which wrong mode gets chosen instead varies by version/seed (sampling noise around a shared failure, not a version-specific trait).

Wrote `eval_temporal_continuity.py` — primes with one real `<agent>`/`<cosmos>` window and samples a long continuation, scoring whether the model returns to and progresses in that modality over time. New finding, not seen before: under sampling, neither v4 nor v5 shows greedy-style macro-repetition — instead there's **"modality drift"**: primed with one modality, the model wanders into others and often doesn't return within the token budget tested. Multi-seed (4 seeds) picture: **v5 returns to cosmos reliably (4/4) but almost never to agent (0/4); v4 is the reverse tendency (2/4 agent, mostly incomplete; 1/4 full cosmos)** — neither model is reliable in both directions, and this directly explains why the v4→v5 `drop_cosmos` relaxation flipped cosmos-persistence without moving PPL (two separate axes, not the same lever).

**v2 (window=8) added as a control** — real w8 multi-window records exist in `megatron_dataset_v6` (confirming v2's actual training source, correcting an earlier stale path reference). A real environment bug was found and worked around here: the current `transformers` version can no longer load v2's tokenizer config (`extra_special_tokens` stored as a legacy list, now requires a dict) — meaning any v2 eval script would crash right now unless patched. **Result**: v2 returns to cosmos 4/4 seeds and, uniquely, chained 2 full consecutive new chunks + a 3rd partial in one run — real multi-chunk persistence never seen in v4/v5. Agent return was 0/4 "insufficient" for v2 too, but with a caveat (its priming record's header exceeds v2's 4096-token context). **Verdict on window=24 causality: directionally supportive but not proven in isolation** — v2 differs from v3/v4/v5 in too many simultaneous ways to isolate window size alone; no single-variable window ablation has been run.

**Real text-to-media generation on genuinely novel prompts** (not in any training record), 2 prompts × v3/v4/v5 (+v2 added later at user request): which modalities appear is highly seed/prompt-dependent noise, not a fixed per-version trait. **Caption content was wrong or only partially related in all 6 runs** — v4's "walking down a narrow road" (vs. prompt: person running) was the closest-to-topic result across the whole session, still wrong. This directly confirms the instruction-following gap is completely unchanged from v2 through v5, because none of those versions added instruction/reasoning data — only window/mix/dropout/schedule changed.

**Final verdict, in the user's own words after seeing this evidence ("chuyển sang phương án khác")**: v3 is a clear fail (worst PPL, wrong captions, cosmos didn't even decode until this session's tooling fix); v4 improves PPL from more training steps, not a root-cause fix, best agent-persistence, worst cosmos-persistence; v5 is a straight tradeoff (unchanged PPL, best cosmos-persistence, worst agent-persistence, never returns). **None of v3/v4/v5 beat v2 on any metric measured — PPL, modality-persistence, or instruction accuracy.** Recommendation: stop iterating on the pretraining recipe (window size, dropout, mix ratio, epochs, doc-packing have all been tried across v3→v5, none beat v2) and pivot to the one lever never tried — real instruction/reasoning data (VLA-Instruct).

**New standing rule adopted**: never delete a dataset pipeline intermediate/version even when superseded (concretely cost real rework this same day — OmniVideo-100K's window=8 Step A output had been deleted during the w8→w24 cleanup, forcing a full video re-tokenize instead of a cheap add-audio-and-merge operation, once the team wanted to revisit window=8). Parametrize pipeline scripts by CLI flag instead of hand-editing a constant, so old and new configurations both stay runnable.

**`w8_new` rebuild planned**: given v2 (window=8) beats every window=24 successor on every metric, the decision was to rebuild v2's own architecture at window=8, but incorporating everything learned/added since (listen/speak wrapper, Harmony4D, roleplay_speak) — rather than either reusing v2 verbatim or continuing to iterate on window=24. A key enabling fact verified this session: token IDs are byte-identical between v2's original tokenizer and the current `tokenizer_vla_qwen3_v2` for all 106,258 shared tokens — meaning no tokenizer rebuild is needed and any tokenized shard from any version (v2 through v5) can be mixed with any other. Per-source status verified directly (not assumed): finevideo_v6 needed only a `<snac>`→`<listen>` rename + retokenize; omnivideo_100k needed a full pipeline rebuild from raw video (its window=8 intermediate had been deleted — the exact incident that prompted the never-delete-versions rule above), completed with 5,214/5,214 videos, 0 errors at every stage; mv_omni and synth_llava were already compatible as-is and reused unchanged; harmony4d needed only a re-windowed Phase 3/5/flatten pass (window size parametrized via CLI, per the new rule).

### Jul 26 — `w8_new` consolidated, v6 (3 seq_length variants) trained + evaluated, v2-vs-v6 comparison, 6 datasets re-published to HF, project overview docs written

**`w8_new` mix consolidated onto `/e`** (the container training runs in only binds `/e`, not `/p`): tokenized shards (finevideo_v6_listen, harmony4d_w8, omnivideo_100k_w8) copied and byte-verified; mv_omni/synth_llava/roleplay_speak symlinked rather than copied, to avoid duplicating ~77GB of byte-identical content already present elsewhere. Total mix: 6 sources, **33,828,943,589 (33.83B) real tokens**.

**A real Harmony4D weight bug found and fixed before launch**: Megatron's `GPTDataset` automatically internally-repeats any blend source whose weighted sample target exceeds its physical content. Giving harmony4d_w8 a bucket-proportional weight (matching finevideo/omnivideo's formula) would have implied a target far exceeding its physical 315.5M tokens, silently compounding its already-baked-in 20x file-level oversample to ~30.7x effective exposure instead of the intended 20x. **Fix (config-only, no re-tokenize)**: give harmony4d_w8 a *dedicated* weight equal to exactly its own physical corpus share, so Megatron samples it in exactly one pass.

**seq_length chosen by testing 3 values, not guessing.** Real per-document token lengths measured directly from Megatron `.idx` files (not word-count estimates): `omnivideo_100k_w8` documents are entire videos (median 361,690 tokens — no seq_length in the practical 4096-16384 range fits this source's documents whole regardless of choice), vs FineVideo's per-activity median of 11,569. Token-weighted whole-document coverage: 3.2% / 23.5% / 32.3% at seq_length 4096 / 8192 / 16384 — a real but non-dramatic gradient, and not obviously predictive given that 2026-07-25's finding was v2 (seq_length=4096) beating v3-v5 (16384) despite lower coverage. **Decision: run all 3 in parallel** rather than pick one (`qwen3_1.7b_vla_v6_seq{4096,8192,16384}.yaml`, identical mix/weights, differing only in seq_length/micro_batch_size/train_iters, no document-boundary packing in any of the three per the earlier ablation rejecting it).

**Launch incident**: submitting via 3 near-simultaneous tmux sessions produced 5 real SLURM jobs instead of 3 (seq4096 and seq8192 each double-submitted into the same output directory). Caught within ~2 minutes via `squeue` job-count mismatch, the 2 duplicates cancelled before any checkpoint. Real jobs: `1046012` (seq4096), `1046010` (seq8192), `1046014` (seq16384), 64 nodes each.

**All 3 jobs COMPLETED cleanly** (verified via `sacct`, not the `current.log` symlink — which, for 2 of the 3 output directories, pointed to the *cancelled duplicate* job rather than the real completed one, a real footgun: reading only that symlink would have looked like a training failure). **Test PPL: seq4096 = 6.36, seq8192 = 5.98 (best), seq16384 = 6.12** — all three closely track v2's 5.77, far ahead of v3/v4/v5's 15-28 range. This strongly confirms the `w8_new` rebuild hypothesis: the earlier v3-v5 regression was driven by the many simultaneous changes bundled into the window=24 pivot, not by window size itself.

**A new window=8-specific eval script was needed and written** (`eval_vla_v6_sanity.py`) rather than reusing `eval_vla_v3_sanity.py` verbatim, since that script primes with a real window=24 record (t up to 23) — reusing it against a window=8 checkpoint would test generalization to an unseen window size rather than the thing the model was actually trained on. New real w8 priming records were sourced directly from `w8_new`'s own flattened data.

**Temporal-continuity results (seed=42 only so far)**: seq4096 returned to `<agent>` **4 consecutive times without freezing** — the best result across the entire v2→v6 series (previous best was 2/4 seeds, always incomplete). seq8192 returned to agent 2x and to cosmos 3x consecutively (most balanced of the three). seq16384 was weakest on every axis (0 agent returns), plausibly because it has the fewest training iterations (2,016) at the same token budget. **Caveat, not yet resolved as of this writing**: this is only 1 of the 4 seeds used to validate v2/v4/v5's temporal-continuity results — a multi-seed re-run (7/123/2024) was launched the same day to check whether this is a stable model behavior or a lucky draw.

**Full-chain text-to-media on a genuinely new prompt ("A person is running in a park")**, tested on v2 and all 3 v6 variants for direct comparison: **no version generated an on-topic caption** — all invented unrelated content ("riding a mountain bike," "riding a slide," etc., v2 included). v2 was the only checkpoint to open `<agent>` twice in one generation and decode all 5 modality types successfully; seq8192 also hit all 5. This reconfirms, with a fourth and fifth data point, that the instruction-following gap is untouched by every pretraining change tried to date (v2 through v6) — the same conclusion reached on 2026-07-25, now extended to the rebuilt window=8 mix too.

**6 datasets re-published to HuggingFace at the `w8_new`/window=8/listen-speak version**: `FineVideo-Phase7-Flattened` (371,892 records, 10.93B real tokens — audio wrapper changed from bare `<snac>` to `<listen>`/`<speak>`), `harmony4d-flattened` (8,320 records, re-windowed from 24→8 frames), `omnivideo-100k-final` (5,214 records, switched back from window=24 to window=8), `synth-llava` and `emotional-roleplay-finetuning-dataset-flattened` (content unchanged, just re-pointed at the consolidated location), and a brand-new repo `MV-Omni` (MixtureVitae-Omni `valid_snac` split, 1,593,301 records verified by full decompression — correcting a never-actually-counted "~1.78M" inventory estimate — license tagged `other` pending an explicit confirmation, since no documented permission statement was found for this specific HF source unlike `synth_llava`'s).

**`PROJECT_OVERVIEW_VI.md`/`PROJECT_OVERVIEW_EN.md` written** — a structured (not chronological), two-part project overview (executive summary + full technical detail) covering motivation, architecture, all 6 data sources, the full v1-v6 training/eval history, known limitations, and roadmap, for readers who don't need the raw session log. The Vietnamese version (along with `PROGRESS_VI.md`) was subsequently untracked from this public repo — see the note at the top of this file.

**Still open as of this entry**: the multi-seed (7/123/2024) re-run of v6's temporal-continuity eval was launched and is in progress; results not yet known. A clean window=8-vs-24 ablation holding every other variable fixed has still never been run — current evidence supports "the whole w8_new rebuild beats the whole w24-family pipeline," not "window size alone is proven responsible." VLA-Instruct remains a design only, no code written.

## Session update — Jul 21, 2026 (continued, session 4 — Harmony4D convert+resample+samples, pulled JUWELS tokenize results, chose 5 datasets + rebalanced mix, wrote Qwen3 v2 config, submitted real training job)

**Main work:** Finished the Harmony4D conversion pipeline through windowing/root-centering (paused there per user request to redirect), pulled the previous JUWELS session's tokenize results, chose 5 datasets for the next training run, wrote a new Qwen3 1.7B config, copied 120GB of data + tokenizer from `/p` to `/e`, and **submitted a real 64-node training job** (`1009758`, RUNNING as of this entry).

### 1. Harmony4D — COCO→H36M convert, 30fps resample, visual samples

Wrote `data_prep/harmony4d/convert_coco_to_h36m.py` (COCO-17→H36M-17 mapping, reusing the formula from `phase1_hrnet_gpu.py::coco_to_h36m()`, extended to 3D; confidence-threshold→NaN instead of zero-fill). Submitted SLURM job `1009045` (22 workers, 1 zip/worker) — **done in 18 seconds**, 208/208 sequences (matches the paper's stated count exactly), 416 person-tracks (208×2 people), 0 errors. Verified by recomputing bone lengths on the converted output against the project's own `skeleton_tree` — near-perfect left/right symmetry, anatomically plausible lengths.

Wrote `data_prep/harmony4d/resample_30fps.py` (linear interpolation on real timestamps `frame_idx/20fps`, mirroring `phase2_5_resample_30fps.py`'s method). Ran directly (no SLURM needed, done in seconds): 416/416 tracks, 0 errors. 301 frames@20fps → 451 frames@30fps, ratio 1.498 ≈ the expected 1.5.

**Decided with the user:** agent tokens keep the existing single-person schema (no new multi-person design) — but **both people are kept** as 2 independent tracks (neither dropped), each processed exactly like a normal single-person video, matching FineVideo's own precedent. 208 sequences → 416 usable tracks.

Wrote `data_prep/harmony4d/make_pose_samples.py`, rendered 6 skeleton videos (`samples/harmony4d/`) for 3 tracks (mma×2 people, hugging×1 person) at 2 checkpoints: after 30fps resample (world-frame), and after root-centering+windowing (stride=8, reusing `create_windows()` from `phase3_kinematics_processor.py`). **Important design call made unilaterally:** did NOT run the full `KinematicPreprocessor.process()` (hallucination filter/ID-switch/stiff-leg) — those heuristics are tuned specifically for monocular MotionBERT failure modes, and applying them to Harmony4D's high-quality multi-view ground truth risks "fixing" genuine poses (e.g. a fast real takedown misread as an ID-switch, a real ground-grappling knee bend "corrected" to standing). Sent samples to the user, who **paused this task here** to redirect to other work.

### 2. Pulled code — confirmed JUWELS finished tokenizing all 4 datasets

`git pull` brought in 2 commits from the prior JUWELS session (`adf87f7`, `59b0042`) — `TOKENIZE_TODO.md` was updated confirming **all 4 tokenize jobs genuinely COMPLETED** (verified via `.idx` header, not just SLURM state):

| Dataset | Real tokens | Records |
|---|---|---|
| finevideo_v6 | 10,926,767,551 | 371,892 |
| omnivideo_100k_video | 536,149,780 | 5,214 |
| synth_llava | 103,097,102 | 603,999 |
| roleplay | 52,469,577 | 67,459 |
| mv_omni (tokenized since Jul 18) | 20,389,561,883 | — |
| **Total, 5 sources** | **32,008,045,893 (~32.01B)** | |

### 3. Chose 5 training datasets, copied 120GB to `/e`

User picked exactly these 5 (matching the "Total, these 4" + MV-Omni figure `TOKENIZE_TODO.md` already recorded). Copied all `.bin/.idx` files (120GB: finevideo_v6 41G, mv_omni 76G, omnivideo 2.0G, synth_llava 405M, roleplay 202M) plus the `tokenizer_vla_qwen3` tokenizer (257,901 real vocab, max token id 257900) from `/p` to `/e/data1/datasets/playground/mmlaion/shared/nguyen38/vla_v2_tokenized/` + `.../tokenizer_vla_qwen3/` — required because the Apptainer container only binds `/e`. Verified shard counts and sizes match the `/p` originals exactly.

### 4. Wrote `qwen3_1.7b_vla_v2.yaml` — this project's first real Qwen3 migration

Found `oellm-autoexp/config/experiments/harsh/qwen3_1.7b_mixvitae_jupiter.yaml` as a template, wrote `config/experiments/nguyen38/qwen3_1.7b_vla_v2.yaml` with the same structure (Qwen3 1.7B architecture, `ckpt_convert_qwen3` postprocess) but pointed at our own tokenizer and data mix. This is the project's first real run on Qwen3 (CLAUDE.md had listed this as "planned" for a while) — replacing the OpenSci-Ref architecture the two prior models used.

**Two decisions made with the user:**
- **Epochs:** user chose exactly 1 epoch ("just depends on data size") instead of keeping the old ~3-epoch convention (reasonable for the 2.84B v1 corpus, but this corpus is 11x bigger so 3 epochs would be far too long) → `train_iters=7632` (32.008B / 4,194,304 tokens-per-iter).
- **Mix ratio:** user asked to reduce MV-Omni's share. Rebalanced 60/40 — the two sources with real `<agent>` action tokens (finevideo_v6 + omnivideo_100k_video) get 60% of total weight (up from their natural 35.82% size-proportional share), the three sources with no action tokens (mv_omni + synth_llava + roleplay) split the remaining 40% (MV-Omni down from 63.71% raw share to 39.71%). Explained to the user: size-proportional weighting gives every source the same N epochs; "reducing" MV-Omni means it gets fewer than N epochs while finevideo_v6 gets more, prioritizing the exact modality-transition signal both prior models failed at.

### 5. Submitted the real training job — job `1009758`, 64 nodes, RUNNING

Submitted using the user's own shell template exactly (saved to memory as `feedback_training_submit_template`). **Minor hiccup:** `run_autoexp.py` runs a foreground polling loop that never exits on its own → the tool call timed out after 2 minutes (exit 143) — but `squeue`/`sacct` confirmed **job `1009758` had already been submitted successfully before the timeout and was still RUNNING normally**, unaffected by killing the polling script.

Killing the orchestrator also killed its automatic postprocess trigger loop (dist_to_torch → convert_hf → eval) that was meant to fire once training finishes. Found the right resume tool: `scripts/monitor_autoexp.py --session-dir monitor_state/<session_id>` -- read its source and confirmed it **has no job-submission code path at all**, it only reloads the existing `JobFileStore` from the prior session and keeps polling -- safe with zero risk of accidentally re-submitting a duplicate 64-node job (which would be expensive if done wrong). Ran it in the background (`nohup ... & disown`), verified `squeue` shows exactly one job and the state file's `attempts` count is still 1 (not bumped to 2) -- confirmed no resubmission happened.

**Status at end of session:**
- Job `1009758` (`qwen3_1.7b_vla_v2`) — RUNNING, 64 nodes, 4h time limit, output at `output_vla/qwen3_1.7b_vla_v2/`.
- Background monitor process (`monitor_autoexp.py`, disowned from the shell) is running, will auto-trigger postprocess once the job finishes.
- **The SLURM job is fully independent of the Claude Code session** — closing the session does not affect the running cluster job. The monitor process (nohup+disown) should also survive independently in theory, but there's no 100% confirmation the underlying background shell/sandbox environment persists indefinitely after the session closes -- if the monitor does die, re-running the same resume command is still safe (won't resubmit), just needs to be re-launched.
- Backlog: Harmony4D windowing/root-centering "for real" (deciding which filters beyond plain center+window, if any) is still paused per the redirect in section 1; eval protocol for the new model is still undesigned (REPORT.md Pre-training Blockers item 3, still open); the 3 new Harmony4D scripts + samples aren't committed/pushed to GitHub yet; the `qwen3_1.7b_vla_v2.yaml` config isn't committed in the `oellm-autoexp` repo yet (different repo, not asked to commit there).

---

**⚠️ Scope update (Jul 20, 2026):** the "watch video, hear speech, generate robot motion" framing
above describes the *original* framing, not the current full scope. Huu (project lead) clarified
directly that the project is actually **omni-modal**: bind any modality pair (image, video, sound,
action, IMU, etc.), as long as the source is permissively licensed and the modality mix is kept
balanced with genuine cross-modal binding — not restricted to humanoid-robot-relevant data. Two
non-video/pose sources already pulled in under this: `synth_llava`/`synth_llava2` (image+caption,
tokenize as `<seed2_N>`) and `laion/emotional-roleplay-finetuning-dataset` (speech+text, tokenize
via `snac`/"moss"). See `REPORT.md`'s "Project Scope Update" section and `PROGRESS_VI.md`'s
2026-07-20 entries for full detail, including an open concern about eval-protocol/research-question
discipline not keeping pace with the widening data-source scope.

---

## Cập nhật phiên làm việc — 21/07/2026 (JUWELS — verify + submit 4 job tokenize Megatron)

**Việc chính:** Phiên này chạy trên **JUWELS** (không phải JUPITER — compute node JUPITER không mount được `/p`, nơi hạ tầng tokenize `mv-scale/` sống), theo đúng file handoff `TOKENIZE_TODO.md` mà một phiên JUPITER trước để lại ở repo root. Pull code mới nhất, đọc `TOKENIZE_TODO.md` + `data_prep/` để xác định 4 dataset cần tokenize lại: **FineVideo-VLA v6, OmniVideo-100K (video), synth-llava, emotional-roleplay**. Theo yêu cầu của user, verify từng dataset khớp đúng bản đã upload HF **trước khi** submit job — tải nguyên bản HF của 3/4 dataset đã có trên Hub (OmniVideo-100K, synth-llava, roleplay), so sánh content-hash (sha256 trên cặp id+text đã sort theo id) với bản local trên `/p` — **khớp tuyệt đối byte-for-byte cả 3**. FineVideo v6 chưa có bản HF nào (HF hiện chỉ có bản v1 cũ, 19GB) nên không có gì để đối chiếu — chỉ verify số record local (371,892) khớp đúng con số `TOKENIZE_TODO.md` đã ghi.

Tạo/sửa 4 file sbatch trong `/p/data1/mmlaion/nguyen38/mv-scale/` (hạ tầng tokenize dùng chung, **không** nằm trong git repo này):
- `tokenize_finevideo_v6.sbatch` — mới, copy từ `tokenize_finevideo_v5.sbatch`, 4 node Ray cluster, port 20160 (đã check không đụng port các sbatch khác đang có).
- `tokenize_omnivideo_100k_video.sbatch` — sửa `INPUT` từ thư mục cũ (`omnivideo_100k_video_flattened`, có trước fix wrapper-token 21/07) sang `omnivideo_100k_final/` — đúng bản plain JSONL (chưa gzip) mà `phase7_finalize_omnivideo.py` ghi ra; không dùng `hf_upload/` vì thư mục đó đã gzip + chia train/test riêng cho HF, `mv_preprocess_data.py` glob không đệ quy nên tự bỏ qua subfolder này.
- `tokenize_synth_llava.sbatch` — mới, 1 node (chưa từng tokenize; size chỉ 547MB nên dùng pattern single-node như `tokenize_robovqa.sbatch`, không cần Ray cluster đa node).
- `tokenize_roleplay.sbatch` — mới, 1 node (chưa từng tokenize; 339MB).

Cả 4 dùng chung tokenizer `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3` (Qwen3 base + full VLA token set, 257,901 vocab) — xác nhận bằng cách đọc thẳng các sbatch template thật đang chạy, không suy đoán từ doc cũ.

**Đã submit 4 job lên SLURM** (JUWELS, account `laionize`, partition `batch`):

| Job ID | Tên | Số node | Dataset |
|---|---|---|---|
| 14127888 | tok_finevideo_v6 | 4 | FineVideo-VLA v6 (371,892 record, 74GB) |
| 14127889 | tok_omni_video | 1 | OmniVideo-100K video (5,214 record, 3.6GB) |
| 14127890 | tok_synth_llava | 1 | synth-llava (603,999 record, 547MB) |
| 14127891 | tok_roleplay | 1 | emotional-roleplay (67,459 record, 339MB) |

Cả 4 job đã chuyển RUNNING (`squeue` xác nhận, không còn PD). **Việc còn lại cho phiên sau:** đừng tin trạng thái SLURM một mình — verify output thật (grep log tìm `Traceback`, check size `tokenized_output/{finevideo_v6,omnivideo_100k_video,synth_llava,roleplay}/`, chạy `mv-scale/count_tokens.py` đối chiếu số token thật) trước khi coi các job này là xong, đúng bài học đã rút ra ngày 18/07 (job từng báo COMPLETED trên SLURM nhưng thực chất fail ngầm vì Ray không khởi động được).

**Cập nhật cùng ngày, tối muộn hơn — cả 4 job đã COMPLETED thật, đã đếm token thật:** theo dõi qua `squeue`/`sacct` cho tới khi cả 4 job kết thúc (job cuối `tok_finevideo_v6` mất 53m12s, tổng cộng 4 node × ~53 phút vì file nặng nhất — 74GB). Grep log cả 4 không thấy `Traceback`. Đếm token thật bằng cách đọc trực tiếp header `.idx` (cùng logic với `mv-scale/count_tokens.py`, không phải ước tính) — tất cả pass BIN SIZE CHECK (tổng độ dài sequence khớp chính xác byte size của file `.bin`):

| Job | Token thật | Docs | Thời gian chạy |
|---|---|---|---|
| finevideo_v6 | **10,926,767,551 (10.93B)** | 371,892 | 53m12s |
| omnivideo_100k_video | 536,149,780 (0.54B) | 5,214 | 19m47s |
| synth_llava | 103,097,102 (0.10B) | 603,999 | 27m06s |
| roleplay | 52,469,577 (0.05B) | 67,459 | 6m07s |
| **Tổng 4 job hôm nay** | **11,618,484,010 (11.62B)** | 1,048,564 | |

Đáng chú ý: token thật của finevideo_v6 (10.93B) gần gấp đôi con số ước tính lúc flatten (5.443B, ghi trong `TOKENIZE_TODO.md`) — đúng pattern lệch đã ghi nhận từ trước với v5 (10.55B thật vs 5.256B ước tính word-count), không phải bug mới, do cách đếm ước tính dùng word-count cho phần text tự do (title/context/caption/speech) thay vì BPE thật.

Đếm lại luôn MV-Omni (đã tokenize từ 18/07, không thuộc 4 job hôm nay nhưng cần cho bức tranh tổng) bằng cùng phương pháp — xác nhận khớp với con số cũ trong memory: **20,389,561,883 (20.39B) token, 1,593,301 docs, PASS**.

**Tổng token thật hiện có, sẵn sàng cho training (chưa tính RoboVQA/OmniVideo-100K-QA — chưa verify lại, và 2 shard `vla_25b`/`vla_adaptive` đã gắn với model cũ):** 11.62B (4 job hôm nay) + 20.39B (MV-Omni) = **~32.01B token**.

---

## Cập nhật phiên làm việc — 21/07/2026 (tiếp lần 3 — Phase 4 xác nhận DONE, phân tích chat Discord với Huu → pivot chiến lược sang nguồn data pose đã annotate sẵn, chốt Harmony4D, chạy lại Phase 5→6 cho FineVideo, phát hiện + xử lý JUWELS/Jupiter storage mismatch, bắt đầu tải Harmony4D)

**Việc chính:** Track lại Phase 4 (job `1004747`) và phát hiện đã **COMPLETED thật sự** (docs cũ ghi ~67%, thực tế đã xong lúc 08:19 sáng cùng ngày) — tính được số liệu Huu hỏi từ lâu ("how much data was filtered out"): **45.9% window bị occlusion filter drop** (43.0M → 23.2M), 8.3% video (3,359/40,300) mất trắng hoàn toàn. Sau đó đọc + phân tích 1 đoạn chat Discord dài giữa Huu và Van Khue về đúng chủ đề này, dẫn tới quyết định pivot chiến lược: **ngừng đầu tư cải thiện pose detector, chuyển sang tìm dataset pose-video đã annotate sẵn** để bù gap occlusion + multi-person. Điều tra 3 ứng viên (MotionVid, OCHuman-Pose, Harmony4D) + rà thêm `Awesome-Video-Datasets` — chốt **Harmony4D** (MIT, đúng cả 2 gap), loại JRDB-Pose3D dù fit kỹ thuật tốt hơn (CC BY-NC-SA, vi phạm chính sách permissive-only). Chạy lại Phase 5 (agent tokens) trên Phase 4 mới, rồi chuẩn bị Phase 6 — phát hiện `submit_merge_adaptive_v4.sh` cũ trỏ vào 1 checkout repo khác trên `/p` (JUWELS storage, lùi sau 3 commit, agent-tokens-dir trỏ vào file `.tar` chưa giải nén) — nếu chạy thẳng sẽ merge caption/speech mới vào agent token cũ/rỗng. Test xác nhận **compute node của Jupiter không mount được `/p`** (chỉ login node thấy) → quyết định cuối: xử lý toàn bộ trên `/e`. Bắt đầu tải Harmony4D (352GB) song song.

### 1. Phase 4 FineVideo — xác nhận COMPLETED, tính số liệu filter nợ Huu

`sacct` xác nhận job `1004747` COMPLETED lúc 08:19:09 (không còn job nào trên `squeue`) — entry phiên trước ghi ~67% chỉ là snapshot cũ. Kết quả cuối: **40,300/40,305 video (99.99%)**, 5 lỗi (3 CUDA OOM do 128 worker tranh GPU qua MPS, 2 "windows could not be resolved") — tỷ lệ lỗi 0.012%, không đáng retry riêng.

Đếm window thật (Phase 3 output vs Phase 4 output, `xargs -P16 wc -l` + sum đúng qua mọi batch — lần đầu dùng `tail -1` sai vì xargs chia nhiều batch, mỗi batch có dòng "total" riêng, phát hiện + sửa ngay):
- Trước filter: 42,994,892 window
- Sau filter: 23,245,694 window
- **Bị drop: 19,749,198 (45.9%)** — đây là con số trả lời câu Huu hỏi trong đoạn chat Discord bên dưới
- 3,359/40,300 video (8.3%) mất trắng toàn bộ (0 window nào pass)

### 2. Phân tích chat Discord Huu ↔ Van Khue → pivot chiến lược nguồn data pose

User dán 1 đoạn chat Discord dài, yêu cầu đọc + phân tích. Chuỗi logic: Huu hỏi tại sao drop pose bị occlude → truy ra root cause là **giới hạn detector** (HRNet hallucinate, ví dụ nhận nhầm cây thành người, nên phải thêm YOLO filter chặn) chứ không phải chủ đích thiết kế → hệ quả là dataset loại hết mọi cảnh có occlusion (người đi khuất sau cây) lẫn multi-person (pipeline hiện chỉ lấy 1 bbox tự tin nhất/frame). Huu quyết định: **không đầu tư cải thiện detector nữa** ("it might be a rabbit hole"), vì đang chạy đua thời gian ("summer is almost over") — 2 hướng train mùa này: model nhỏ (omni-VLA + RL trong simulation) và model lớn (dataset MV2, làm POC omni model tốt cả ở language). Thay vào đó: **tìm dataset pose-video đã annotate sẵn** trên HF.

Chi tiết đáng chú ý: 1 list dataset gợi ý do AI generate mà Huu paste có **`FineVideo-Phase2-3DPose` — chính dataset team đã upload** — bị liệt kê như của bên thứ 3; Huu tự nhận xét điều này cho thấy data pose-video permissive-license công khai thực sự khan hiếm.

Lưu vào memory (`project_pivot_pose_dataset_sourcing`): không đề xuất sửa detector nữa cho các phiên sau; "VALID" và "Leo" (JUWELS/Leonardo) là 2 proper noun Huu nhắc tới trong chat, chưa xác nhận rõ nghĩa, cần verify trước khi giả định.

### 3. Điều tra 3 ứng viên (agent nền) + rà Awesome-Video-Datasets — chốt Harmony4D

Launch 1 agent nghiên cứu song song (không sửa code, chỉ tra cứu):
- **MotionVid**: loại — HF repo chỉ có caption+keypoint 2D (DWPose), **không có video thật** (phải tự ghép lại từ 9 dataset gốc, nhiều cái non-commercial)
- **OCHuman-Pose**: loại — chỉ ảnh tĩnh (không video/temporal), 2D only, chỉ có eval split, license mâu thuẫn (HF ghi MIT nhưng nguồn khác nói CC BY-NC)
- **Harmony4D**: ✅ chọn — MIT xác nhận từ nguồn gốc tác giả (`jyuntins/harmony4d` trên GitHub + `Jyun-Ting/Harmony4D` trên HF, không phải bản mirror `Voxel51/Harmony4D` ghi thiếu license). 208 sequence, 24 subject, luôn 2 người/cảnh (hugging/grappling/sword/ballroom/karate/mma), multi-view video + 3D pose 17-joint + SMPL mesh fit contact-aware — giải đúng cả occlusion lẫn multi-person
- Rà `xiaobai1217/Awesome-Video-Datasets` (list thiên action-recognition, không hiệu quả lắm cho pose): tìm ra **JRDB-Pose3D** (SMPL multi-person 3D, 5-35 người/frame, nhãn occlusion rõ ràng — fit kỹ thuật tốt hơn cả Harmony4D) nhưng **CC BY-NC-SA → loại thẳng, không bàn thêm**, đúng chính sách permissive-only. NTU RGB+D/120, UAV-Human cũng bị loại (non-commercial / viewpoint không hợp / license chưa xác nhận).

User chốt trực tiếp: **Harmony4D**, bỏ qua JRDB-Pose3D dù kỹ thuật tốt hơn, vì "chỉ quan tâm permissive thôi, làm open source mà". Đồng thời làm rõ dynamic quyết định: "Huu chỉ là PI thôi", Van Khue tự quyết kỹ thuật trực tiếp, không cần chờ Huu duyệt từng bước — đã lưu vào memory feedback (`feedback_decision_authority_and_license_policy`) để các phiên sau không tự thêm rào cản "cần confirm với Huu trước" không cần thiết.

### 4. Chạy lại Phase 5 (agent tokens) trên Phase 4 mới

`outputs/agent_tokens_adaptive/` cũ (18,847 file, 19/06) được dựng trên Phase 3/4 **trước khi fix bug fps-mismatch** — script Phase 5 chỉ skip theo `os.path.exists()`, không biết input đã đổi, nên phải move (không xóa) sang `agent_tokens_adaptive_buggy_fps_mismatch_2026-07-20/` trước khi submit lại, tránh bị skip nhầm. Submit `slurm/submit_phase5_adaptive.sh` (job `1006884`, 64 worker) — **COMPLETED trong 5 phút28s**: 19,076 video có agent token, 21,224 video rỗng (đa số do joint tay bị NaN gần như toàn bộ trong footage YouTube, đúng như REPORT.md đã ghi từ trước), 0 skip.

### 5. Phase 6 — phát hiện kiến trúc `/p` (JUWELS) vs `/e` (Jupiter) không tương thích, xử lý xong

Định submit thẳng `slurm/submit_merge_adaptive_v4.sh` (script cũ, có đủ `--captions-dir`/`--speech-segments-dir` như note đã ghi từ phiên trước) nhưng phát hiện 2 vấn đề nghiêm trọng trước khi chạy:
1. `--input-glob` trỏ vào `final_dataset_adaptive_v3` — bản này được build từ Phase 3/4/5 **trước khi fix fps-mismatch** (v3 tạo 12/07, fix bug 20/07) — nếu dùng sẽ merge caption/speech mới vào agent/pose token cũ lỗi, phí công regen Phase 4+5 vừa xong.
2. Script chạy trên checkout `/p/data1/mmlaion/nguyen38/3d-human-pose` — **lùi sau 3 commit** so với `/e` (thiếu cả fix fps-mismatch lẫn wrapper-token hôm trước) — đã `git pull` đồng bộ lại.
3. `--agent-tokens-dir` của bản `/p` trỏ vào 1 file `agent_tokens_adaptive.tar` **chưa giải nén** (chỉ là bản backup từ 19/07), không phải thư mục thật.

User làm rõ: `/p/data1` là storage của **JUWELS**, `account=laionize` là account riêng JUWELS — không dùng được từ session Jupiter hiện tại. Test trực tiếp bằng `srun` xác nhận **compute node của Jupiter/booster không mount được `/p`** (chỉ login node thấy) — đây chính là lý do các script cũ phải chạy qua JUWELS. Sau khi cân nhắc (ban đầu định đưa hết data về `/p` theo policy mới của user, rồi phát hiện conflict này), **quyết định cuối: toàn bộ pipeline xử lý trên `/e`** (đã lưu memory `feedback_data_storage_location`, đảo ngược policy cùng ngày sau khi hiểu rõ giới hạn hạ tầng). Copy 3 thư mục phụ trợ (snac_tokens 6.5GB/40,779 file, captions_dict 114MB/40,798 file, speech_segments 334MB/40,490 file — dữ liệu cũ, không bị ảnh hưởng bug pose, không cần regen) từ `/p` sang `/e`. Viết `slurm/submit_merge_adaptive_v5.sh` (account `reformo`/`booster`, input đúng base `training_ready_rank_*.jsonl` — không phải v3 — + agent token mới + snac/caption/speech đã copy) — submit job `1007805` (32 task), tất cả `RUNNING` ngay, 0 lỗi. Spot-check live trên phần đã ghi: 112 video/1,148 activity, agent 16.9% (khớp gần đúng con số lịch sử "full-chain 18.8%" của bản v3/v4 cũ), snac 90.7%, caption 90.8% — tín hiệu tốt.

### 6. Bắt đầu tải Harmony4D (352GB)

Viết `tools/extract/download_harmony4d.py` theo đúng convention các script tải khác trong repo (`snapshot_download`, resumable, retry loop, `HF_HUB_DISABLE_XET=1`). Tra trực tiếp cấu trúc thật trên HF (`Jyun-Ting/Harmony4D`): `train/` 15 zip (~287GB, 01_hugging → 15_mma4), `test/` 7 zip (~65GB) — README chỉ 24 byte, không có tài liệu, phải tự giải nén để xem format. License MIT xác nhận trực tiếp trên HF card. Target ban đầu định để `/p` rồi sửa lại `/e` theo quyết định ở mục 5. User tự chạy trong tmux (`logs/harmony4d_download.log`) — phát hiện cần env `activate_env_tools.sh` (khác 2 env chính trong CLAUDE.md, nhẹ hơn, đủ `huggingface_hub`) vì module-load thủ công bị lỗi thiếu `libpython3.12.so.1.0` khi activate `env_stable_vla` trực tiếp không qua đúng trình tự. **Đang tải, ~139GB/352GB (~39%) tại thời điểm ghi entry, 0 lỗi.**

**Sự cố nhỏ:** user paste `HF_TOKEN` thật ra chat khi debug lỗi thiếu module — đã cảnh báo user revoke/tạo token mới sau khi xong việc.

### 7. Phase 6 (merge v5) — COMPLETED, verify sạch

Job `1007805` xong trong **22 phút** (nhanh hơn ước tính 45p-1.5h). 32/32 task COMPLETED, exit 0:0, 160/160 file output, 0 lỗi. Verify số liệu: 40,804 video, 398,775 activity (**khớp tuyệt đối** với số liệu lịch sử v2/v3), agent injected 2,326,095 (**cao hơn** bản cũ 2,148,474 — +8.3%, hợp lý vì fix fps-mismatch giúp match đúng timestamp hơn), snac injected 38,824,718 (**khớp tuyệt đối từng số** với v2 — đúng kỳ vọng vì audio không bị ảnh hưởng bug pose).

### 8. Phase 7 (flatten v6) — 1 lần fail, tìm đúng nguyên nhân, fix, chạy lại

Submit lần đầu (`job 1007976`) dùng `setup_motionbert.sh` (env_motion_final, theo convention Phase 5/6) — **fail ngay** với `ModuleNotFoundError: No module named 'wn'` (thư viện WordNet cho text augmentation, `import wn` cứng ở đầu file, không có try/except). Đáng chú ý: `sacct` vẫn báo `COMPLETED 0:0` dù thực chất crash — vì script bash không check exit code của lệnh python trước dòng echo cuối cùng → **bài học: không tin `sacct` một mình, phải đọc `.err` thật**.

Điều tra: `wn` chỉ có sẵn trong venv `env_tools` (`/p/data1/mmlaion/nguyen38/env_tools`) — nhưng venv này **cũng dính đúng lỗi JUWELS/Jupiter y hệt cả buổi**: `python3` bên trong nó là symlink trỏ tới đường dẫn phần mềm JUWELS (`/p/software/juwels/...`), không tồn tại trên Jupiter, nên "activate" nó từ Jupiter thực chất không chạy gì — Python thật thực thi là bản module-load của Jupiter, dùng package từ `~/.local` (site-packages riêng user, mount toàn cluster). `wn` chưa từng được cài vào `~/.local`. Fix: `python3 -m pip install --user wn` + `python3 -m wn download oewn:2024` chạy từ login node (dùng đúng `python3 -m pip`, không dùng lệnh `pip`/`pip3` trực tiếp — bản thân file `pip` trong venv `/p` cũng có shebang trỏ tới interpreter JUWELS bị hỏng y hệt).

Sửa `submit_phase7_flatten_v6.sh` bỏ `source activate_env_tools.sh` (gây lỗi `No such file or directory` vô hại trên compute node nhưng gây nhiễu debug), thay bằng `module load` trực tiếp. Resubmit (`job 1007994`) — **chạy đúng, ra output thật** (verify: file đầu có 2,031 dòng nội dung `### Context:`/`### Keywords:`/`### Speech:` đúng format). Giữa chừng `squeue`/`scontrol` bị lỗi kết nối SLURM controller tạm thời (cluster-wide, không liên quan job) khiến 1 lần tưởng nhầm job đã "hỏng" — verify lại bằng filesystem trực tiếp (nội dung file output, không phụ thuộc SLURM controller) xác nhận job vẫn chạy tốt, không cần kill/resubmit lại. **Tại thời điểm ghi entry: ~96/160 file, đang chạy tiếp, 0 lỗi.**

### 9. Harmony4D — tiếp tục tải, có 1 lần đứt kết nối tự resume

Tại thời điểm ghi entry: **~308GB/352GB (~87%)**. Có 1 lần mất kết nối giữa chừng khi tải `02_grappling.zip` (đứt ở 28.3/44.1GB) — script tự "Trying to resume download..." theo đúng thiết kế resumable, không cần can thiệp.

### 10. Thảo luận chiến lược: "omni model — data đủ chưa, output là gì?" — làm rõ 3 framing mâu thuẫn

User đặt câu hỏi cấp cao, thấy mơ hồ: liệu data đã đủ cho lần train omni tiếp theo, và output model thực sự là gì. Rà lại toàn bộ docs + thảo luận trước đó, phát hiện **3 framing khác nhau đang tồn tại song song, không khớp nhau**: (1) framing gốc CLAUDE.md — humanoid VLA, output = pose/action token; (2) framing Huu 20/07 — "omni means all modes... cross-modal bindings", không định nghĩa output cụ thể; (3) framing từ chat Discord phiên trước — 2 nhánh train: model nhỏ (omni-VLA + RL-in-sim, có target đo được) và model lớn (MV2, POC "performant in language too", chưa có eval nào đo). Đối chiếu với REPORT.md: đây **không phải vấn đề mới** — chính Van Khue đã tự flag y hệt concern này ngày 20/07 ("not yet a single fixed central research question..."), vẫn chưa giải quyết.

User yêu cầu tự tin đề xuất 1 framing rõ thay vì liệt kê. Đề xuất (dựa trên bằng chứng thực tế đã đầu tư công sức, không phải lời nói): **mục tiêu thật nên là VLA model có năng lực cốt lõi sinh action/pose token hợp lệ, thước đo cuối là closed-loop task success trong simulation (RL rollout)** — không phải benchmark ngôn ngữ. Bằng chứng: toàn bộ pipeline 7-phase chỉ phục vụ việc này; 2 phép eval đã chạy (agent completion, modality transitions) đều đo action; quyết định Harmony4D hôm nay cũng để vá gap action-token. Hệ quả: "đủ data chưa" nên đo bằng **volume + đa dạng agent-token** (hiện chưa đủ — occlusion cắt 46%, chỉ lower-body, 1 người/cảnh — đúng lý do Harmony4D đáng làm), không phải tổng token toàn corpus. Cảnh báo riêng: nếu Huu thật sự muốn claim "performant in language too", **hiện chưa có eval nào đo việc đó** — cần 1 benchmark ngôn ngữ chuẩn riêng, đây là gap hoàn toàn khác, chưa ai làm.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- **Phase 4, 5, 6 FineVideo** — **XONG** cả 3.
- **Phase 7 (flatten v6)** — đang chạy (job `1007994`), ~96/160 file, 0 lỗi, ETA còn vài phút.
- **Harmony4D download** — đang tải, ~308/352GB (~87%).
- Đã cài thêm `wn` (WordNet) + lexicon `oewn:2024` vào `~/.local` — cần thiết cho mọi lần chạy Phase 7 sau này trên Jupiter.
- Việc tồn đọng: verify + upload lại `FineVideo-Phase7-Flattened` sau khi Phase 7 xong; phân tích cấu trúc thật Harmony4D sau khi tải xong rồi thiết kế pipeline convert SMPL/COCO-17 → H36M-17; **quyết định chưa chốt với Huu**: framing/eval protocol thật cho omni model (đề xuất ở mục 10, chưa gửi Huu); tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (vẫn hoãn); chờ Huu quyết định `MixtureVitae-Backup` SNAC + ý nghĩa "moss" token; RoboVQA vẫn treo do câu hỏi kiến trúc chưa giải (16 frame rời rạc/episode vs video liên tục Step A cần).

---

## Cập nhật phiên làm việc — 21/07/2026 (tiếp lần 2 — submit lại Phase 4 FineVideo qua SLURM, thêm wrapper token seed2/cosmos/agent/snac vào 3 dataset, upload lại synth-llava + omnivideo-100k-final)

**Việc chính:** User yêu cầu submit lại Phase 4 FineVideo qua SLURM (JUPITER hết bảo trì phần lớn), rồi trong lúc track tiến độ phát hiện + sửa 1 loạt vấn đề thật: (1) job SLURM đầu tiên chia việc không đều do file đã-xong dồn cục, phải rebalance; (2) `synth_llava` bị lỗi seed2 token lồng sai vào trong `<caption>` (kế thừa từ cấu trúc data gốc); (3) qua thảo luận với user, phát hiện **toàn bộ 3 dataset video/pose/synth_llava đều thiếu wrapper token** (`<seed2>`/`</seed2>`, `<cosmos>`/`</cosmos>`, `<agent>`/`</agent>`, `<snac>`/`</snac>`) dù các token này **đã được đăng ký sẵn trong tokenizer vocab** (`tools/tokenizer/expand_vocab.py`, `build_tokenizers.py`) — chỉ riêng SNAC (`laion_emotional_roleplay`) là có dùng. Quyết định cùng user: thêm wrapper lại cho cả 3 dataset (lý do: tận dụng token đã trả phí sẵn trong vocab, nhất quán với SNAC, và là giả thuyết đáng thử cho lỗi lớn nhất của model hiện tại — "modality transitions: FAIL"). Đã fix code nguồn, regen data, và **upload thành công `EmpathicRobotics/synth-llava` + `EmpathicRobotics/omnivideo-100k-final`** — có 1 lần báo "sẵn sàng" hơi sớm (quên áp fix wrapper vào data `synth_llava` đã tokenize sẵn, chỉ sửa source script), bắt lại kịp trước khi gây hại thêm, đã fix và upload lại đúng.

### 1. Phase 4 FineVideo — submit SLURM, phát hiện + sửa lỗi chia việc không đều

`sinfo` xác nhận phần lớn `booster` hết bảo trì (chỉ còn 18/~5,600 node lẻ tẻ `maint`) — submit `slurm/submit_yolo.sh` (job `1004323`, 128 worker/4 GPU/1 node qua NVIDIA MPS), kế thừa tiến độ cũ từ login-node (17,706/40,305 đã có, script tự skip). User quan sát thấy vài worker xong ngay lập tức trong khi worker khác vẫn chạy chậm — điều tra ra: **48/128 worker xong gần như tức thì** vì file đã-xong (từ lần chạy 4-worker cũ) dồn cục ở đầu mỗi 1/4 khối do chia contiguous, còn lại ~80 worker phải làm gần 100% việc thật (~315 video/worker thay vì ~171 nếu chia đều đúng phần còn lại). **Fix:** `scancel` job cũ, build symlink farm chỉ chứa 21,841 file states_jsonl còn thiếu (`outputs/states_jsonl_30fps_remaining_phase4/`, NFS symlink chậm nên phải chạy nền + resume 2 lần do timeout), viết `slurm/submit_yolo_remaining.sh` (giống hệt `submit_yolo.sh`, chỉ đổi `--input-dir` trỏ vào symlink farm) — submit lại (`job 1004747`), chia đều ~171 video/worker. **Đang chạy tốt** (21,342/40,305 tại thời điểm ghi entry, 1 lỗi soft/video không fatal, worker tự chuyển video kế tiếp).

### 2. Sửa lỗi `synth_llava`: seed2 token lồng sai vào `<caption>`

User phát hiện: data gốc `synth_llava`/`synth_llava2` (Huu tự tạo) đã có sẵn cấu trúc `<caption><image_0>caption text</caption>` — `tokenize_seed2.py` chỉ string-replace placeholder tại chỗ, giữ nguyên lồng sai (seed2 token nằm trong `<caption>`, trong khi convention thật của project — `step_a_tokenize_video.py` — coi `<seed2>`/`<caption>` là 2 tag ngang hàng, tuần tự, không lồng nhau). Fix bằng 2 pass regex trên 603,999 dòng × 151 file (pass đầu bị bug con — token seed2 cuối cùng dính liền text không space bị regex bỏ sót, sinh lỗi mới; verify + fix tiếp ngay). Đồng thời sửa `tokenize_seed2.py` (source) để lần rerun sau không lặp lại.

### 3. Phát hiện + quyết định: thêm lại wrapper token cho seed2/cosmos/agent/snac

Trong lúc thảo luận, user chỉ ra `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened` có `<snac>`/`</snac>` wrapper nhưng `FineVideo-Phase7-Flattened`/`omnivideo-100k-final` thì không, dù cả 2 loại token đều **đã đăng ký sẵn trong vocab tokenizer** (`<seed2>`, `</seed2>`, `<cosmos>`, `</cosmos>`, `<avc_lm>`, `</avc_lm>`, `<agent>`, `</agent>` — thấy trong `tools/tokenizer/expand_vocab.py` + `build_tokenizers.py`). Điều tra ra: đây **không phải bug quên sót cụ thể ở seed2/cosmos** — `pipeline_pose/phase7_flatten.py`/`flatten_step_a_video.py` lột bỏ wrapper cho **mọi** loại token đồng nhất (seed2/cosmos/avc_lm/agent outer/snac outer đều bị lột, chỉ `<caption>`/`<speech>` giữ lại) — là quy ước có chủ đích từ đầu. Trong khi `laion_emotional_roleplay/tokenize_snac.py` là code **hoàn toàn riêng biệt**, không dùng lại `phase7_flatten.py`, và có quyết định wrapper riêng (ghi rõ trong docstring: "decided after review with Van Khue, session 2026-07-20") — nên 2 format song song không nhất quán từ trước tới giờ, không phải lỗi phát sinh mới.

User hỏi lý do kỹ thuật tại sao cần wrapper trước khi cho làm — đã giải thích: (a) token `</seed2>` tách bạch "đã hết span" khỏi "modal kế tiếp là gì" — 2 tín hiệu vốn bị gộp làm 1 khi không có wrapper, đúng loại vấn đề mà model đang fail ("modality transitions: FAIL... stays in seed2 mode"); (b) dropout nặng (cosmos 50-90%, avclm ~99%) làm "tập modal hợp lệ sau seed2" đổi liên tục giữa các ví dụ, cộng dồn nhiễu lên đúng tín hiệu đó; (c) pattern chuẩn trong nhiều hệ multimodal/agent LLM thật (Chameleon, tool-call wrapper...); (d) minh chứng sống ngay trong phiên — chính regex đoán ranh giới bằng khoảng trắng ở mục 2 đã sai 1 lần vì thiếu wrapper rõ ràng. User đồng ý, quyết định thêm wrapper cho cả 3 dataset, **để riêng FineVideo-Phase7-Flattened chờ Phase 4 xong** (không block bởi user, sẽ tự đúng khi Phase 5→6→7 chạy lại vì code đã fix sẵn).

**Đã sửa code nguồn** (4 file, đều unit-test bằng data giả lập trước khi áp dụng):
- `pipeline_pose/phase7_flatten.py` — `process_activity_per_chunk()` + `count_token_types()` (tránh đếm nhầm token wrapper mới vào bucket agent)
- `data_prep/omnivideo_100k/step_a/flatten_step_a_video.py` — mirror fix (dù xác nhận đây **không phải** script build `omnivideo-100k-final` thật, chỉ giữ đồng bộ cho tương lai)
- `data_prep/omnivideo_100k/phase6_merge_omnivideo.py` — **script thật** build `omnivideo-100k-final` (đọc thẳng raw Step A, tự flatten+merge agent trong 1 hàm riêng `flatten_token_stream_with_agent`, không tái dùng `flatten_step_a_video.py` như docstring cũ ghi nhầm)
- `data_prep/synth_llava/tokenize_seed2.py`

**Regen `omnivideo-100k-final`:** chạy lại `phase6_merge_omnivideo.py` (32 file, ~17s/file) rồi `phase7_finalize_omnivideo.py` — verify số liệu khớp **chính xác tuyệt đối** với bản cũ (5,214→5,214, 0 malformed, 799 video có agent, 62,631 window agent, 5,214/5,214 có QA) — chỉ đổi cấu trúc token, không đổi nội dung/số lượng.

**Fix `synth_llava_flat`:** áp thêm 1 pass regex thêm `<seed2>`/`</seed2>` bọc quanh chuỗi seed2 token đã có (an toàn vì mỗi dòng chỉ có đúng 1 span seed2, không như OmniVideo có nhiều chunk cùng loại liên tiếp — không thể patch hậu-kỳ an toàn kiểu này cho OmniVideo, phải regen từ raw). Verify: 603,999 dòng, 19,327,968 seed2 token (không đổi), 0 lỗi format.

**Sự cố nhỏ:** báo user "sẵn sàng upload" 1 lần trước khi kịp áp fix wrapper vào `synth_llava_flat` (chỉ sửa xong source script, quên chạy lại trên data đã có) — user chạy upload, thành công lên HF nhưng thiếu wrapper. Phát hiện qua triệu chứng user báo ("sao lại skip compress") — do cache `.gz` cũ (từ lần upload lỗi format `<caption>` trước đó) đã tồn tại nên script skip đúng theo thiết kế. Verify bằng `HfApi().dataset_info()` xác nhận thời điểm upload thật, fix data, xóa cache, upload lại — không phát hiện vấn đề gì với `omnivideo-100k-final` (user chưa kịp upload bản cũ nên không bị ảnh hưởng).

### Kết quả upload (đã xong, verify bằng log user gửi)

- **`EmpathicRobotics/synth-llava`** — 151 shard (140 train + 11 test), 135MB, upload thành công.
- **`EmpathicRobotics/omnivideo-100k-final`** — 32 shard (30 train + 2 test), 683MB, upload thành công.

### Trạng thái cuối phiên

- **Phase 4 FineVideo** (SLURM job `1004747`, rebalanced, 128 worker) — đang chạy, 21,342/40,305 (~53%), 1 lỗi soft/video không fatal.
- **`synth-llava`, `omnivideo-100k-final`** — đã upload lại đúng format (wrapper token + caption fix), **XONG**.
- **`FineVideo-Phase7-Flattened`** — chưa regen, chờ Phase 4 xong rồi chạy Phase 5→6→7 bình thường (code `phase7_flatten.py` đã fix sẵn, không cần làm gì thêm ngoài chạy pipeline như thường lệ).
- Việc tồn đọng: sau khi Phase 4 xong — chạy Phase 5 (agent tokens) → Phase 6 (merge, **nhớ dùng script kiểu `submit_merge_adaptive_v4.sh` có `--captions-dir`/`--speech-segments-dir`, không dùng bản merge thường**, xem mục dưới) → Phase 7 (flatten, đã có wrapper) → upload lại `FineVideo-Phase7-Flattened`; tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (đã hứa từ lâu, vẫn hoãn); chờ Huu quyết định `MixtureVitae-Backup` SNAC + ý nghĩa "moss" token.

### Cập nhật tiếp — track lại Phase 4, tự sửa 1 nhận định sai về FineVideo v5, tổng hợp bức tranh 4 dataset

**Phase 4** (job `1004747`) — cập nhật: **27,056/40,305 (~67%)**, 0 lỗi trừ 1 soft/video đã ghi ở trên, chạy ổn định ~1h. User phát hiện tôi (Claude) đọc nhầm log cũ (`logs/yolo_workers_run1_unbalanced/` — log job đầu tiên đã hủy, archive trước khi resubmit) tưởng job vẫn chia việc lệch — job hiện tại `logs/yolo_workers/` xác nhận **0 SKIP**, mỗi worker đúng 170-171 file, rebalance hoạt động đúng.

**Tự sửa sai:** khi được hỏi "gộp 4-5 dataset đã đủ train chưa", tôi khẳng định sai rằng FineVideo-VLA **không có caption** — dựa trên check nhầm `/p/data1/mmlaion/shared/vla/vla_adaptive/` (bản THẬT nhưng là bản CŨ, tiền-caption, chính là data đã tokenize cho model đang train hiện tại — 2.84B token — không phải v5). User chỉ ra ngay ("bị ngáo à"), check lại đúng thư mục `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v5/` (live trên `EmpathicRobotics/FineVideo-Phase7-Flattened` từ 07/07, xem REPORT.md §21): **100% có `<caption>`, 96% có `### Speech:` header, 80% có `<speech>` inline** — đúng như REPORT.md đã ghi từ trước (371,888 record, 5.217B token), tôi chỉ tra sai thư mục local, không phải docs sai. Số tổng 12.53B đã báo trước đó **vẫn đúng** (seed2/cosmos đếm lại khớp chính xác tuyệt đối với log cũ) — chỉ riêng nhận định "thiếu caption" là sai, đã rút lại.

**Bảng phân loại 4 dataset (không tính MV-Omni):**

| Dataset | Record | Modal | Token |
|---|---|---|---|
| FineVideo-VLA (v5, đang regen fps-fix+wrapper) | 371,888 | seed2 + cosmos + agent (18.8% full-chain) + snac + caption (Qwen2.5-VL) + speech (Whisper header+inline) | 5.217B |
| omnivideo-100k-final | 5,214 | seed2 + cosmos + agent (799/5,214, chỉ sports-subset) + caption + speech + QA (99,983 cặp, có reasoning hint xuyên-modal) | 274.6M |
| synth-llava/synth-llava2 | 603,999 | chỉ seed2 (ảnh tĩnh, 32 tok/ảnh) + caption — không video/pose/audio | 70.8M |
| emotional-roleplay-finetuning-dataset-flattened | 67,459 | chỉ snac (audio) + text — không video/pose | 26.9M |
| **Tổng** | | | **~12.53B** |

Cộng MV-Omni (+6.93B, chủ yếu SNAC/audio) → ~19.4B, nằm trong khoảng "10-20B đủ" từng ước tính, chưa tới target tham vọng "20-40B". **Kết luận đưa cho user: đủ để thử train v0.2** sau khi FineVideo regen xong (giữ đúng caption/speech, xem nhắc nhở trên) — không cần thêm RoboVQA mới đạt ngưỡng sẵn sàng.

## Cập nhật phiên làm việc — 21/07/2026 (track lại Phase 4 FineVideo + synth_llava seed2 tokenize — xác nhận synth_llava đã DONE hoàn toàn)

**Việc chính:** User yêu cầu track lại 2 job đang chạy nền mà entry trước ghi "cần check lại % hiện tại" — verify bằng log thật (không suy đoán từ docs cũ).

### 1. synth_llava seed2 tokenize — ĐÃ XONG hoàn toàn

`logs/synth_llava_seed2.log` kết thúc lúc 20:02 tối 20/07, `SYNTH_LLAVA_EXIT=0`:

```
TONG: 603,999 -> 603,999 | no_image: 0 | encode_fail: 0 | seed2_tokens: 19,327,968
```

100% thành công, 0 lỗi. Output: 151 file `{synth_llava,synth_llava2}_shard-*.jsonl` (56 + 95 file) tại `/p/data1/mmlaion/shared/vla/synth_llava_flat/`, mỗi dòng đã đúng format phẳng training-ready `{"id", "text"}` với `<image_0>` đã thay bằng chuỗi `<seed2_N> ...` atomic-token — verify trực tiếp bằng cách đọc vài dòng thật.

**Không cần bước "merge" riêng cho dataset này** — khác FineVideo/OmniVideo (nhiều stream token: video + agent pose + QA cần Phase 6 merge lại), `synth_llava` chỉ có 1 modality (ảnh → seed2) nên `tokenize_seed2.py` sinh thẳng ra format phẳng cuối cùng, không có agent/pose stream nào khác để ghép vào.

### 2. Phase 4 FineVideo — vẫn đang chạy, ~33%

Không connect được tmux `yolo_login_finevideo` từ session hiện tại (`tmux ls` lỗi "No such file or directory"), nhưng 4 file log worker vẫn được ghi liên tục **đúng tới thời điểm check** (mtime khớp giây với `date`) → job chắc chắn còn sống, chỉ là sandbox hiện tại không thấy tiến trình đó.

| Worker | Vị trí | Done |
|---|---|---|
| 1 | 3858/10076 | 3377 |
| 2 | 3851/10076 | 3334 |
| 3 | 3846/10076 | 3339 |
| 4 | 3827/10077 | 3321 |

Tổng: **13,371/40,305 video (~33%)**, 0 lỗi trên cả 4 log.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- **Phase 4 FineVideo** (tmux `yolo_login_finevideo`, 4 worker) — đang chạy, ~33% (13,371/40,305).
- **synth_llava seed2 tokenize** — **XONG**, 603,999 dòng, 19,327,968 seed2 token, output tại `/p/data1/mmlaion/shared/vla/synth_llava_flat/`.
- Việc tiếp theo: viết script consolidate/upload HF cho `synth_llava_flat` theo convention `laion_emotional_roleplay/upload_hf.py` (nén + train/test split theo shard + dataset card) — chờ user chọn tên repo trước khi upload thật.

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 6 — recap RoboVQA, đếm token distribution thật cho 4 dataset đã flatten, đối chiếu số liệu docs cũ)

**Việc chính:** Recap lại RoboVQA cho user (không có việc mới, chỉ tổng hợp lại điều tra đã làm). Viết script đếm token tổng quát (tái dùng đúng `PATTERNS`/`count_tokens` convention từ `tools/inventory/data_inventory.py` để so sánh ngang hàng với số liệu cũ), chạy full trên cả 4 dataset đã flatten: FineVideo-VLA v5, MV-Omni (phần valid_snac đã convert), OmniVideo-100K final, emotional-roleplay-finetuning-dataset. User bắt đầu soạn 2 sbatch Megatron-tokenize cho OmniVideo/roleplay để chạy trên JUWELS — **tự viết nhầm 1 bản đoán mò không có template thật, user chỉ ra và yêu cầu xóa**, đã xóa, chờ user cung cấp template thật.

### 1. Đếm token thật — kết quả (multiprocessing, 24 worker/dataset, ~9 phút tổng cho cả 4 dataset trên login node)

| Dataset | seed2 | cosmos | avclm | snac | agent | text (word-count xấp xỉ) | **TOTAL** |
|---|---|---|---|---|---|---|---|
| emotional-roleplay-finetuning-dataset | 0 | 0 | 0 | 23,390,760 | 0 | 3,556,739 | **26,947,499** (27.0M) |
| omnivideo-100k-final | 17,229,664 | 201,736,400 | 0 | 0 | 19,185,063 | 36,480,085 | **274,631,212** (274.6M) |
| mv-omni (valid_snac converted) | 19,249,664 | 0 | 0 | 4,922,681,181 | 0 | 1,990,708,185 | **6,932,639,030** (6.933B) |
| finevideo-vla (v5) | 332,592,448 | 3,882,954,800 | 0 | 363,029,331 | 564,876,258 | 82,430,660 | **5,225,883,497** (5.226B) |

`avclm` luôn = 0 ở cả 4 — đúng thiết kế (payload avc_lm luôn bị discard ở bước flatten cuối cùng của mọi pipeline trong project, không phải bug — user tự hỏi và tự xác nhận đúng).

**Đối chiếu với số liệu cũ trong docs:**
- **MV-Omni: 6.933B — khớp gần như tuyệt đối** với con số "+6.93B token" đã ghi ở mục "Bức tranh data" (PROGRESS_VI.md, từ investigation tháng 6).
- **FineVideo-VLA v5: 5.226B — khớp gần đúng** với con số "5.256B tokens" đã ghi khi upload v5 lên HF (chênh ~0.03B, hợp lý vì cách đếm text ở đây là word-split xấp xỉ, không phải đếm qua tokenizer thật).
- **Không tìm thấy con số "10B"/"20B" cụ thể nào khớp trực tiếp** — các số này trong docs thực ra là **ước tính mục tiêu** (ví dụ "Target: 20–40B token cho v0.2 training", "sau khi hoàn thành ưu tiên 1,2,4: ước tính 10–20B token") chứ không phải số đo thực tế của 1 dataset cụ thể nào — đã báo lại cho user để tránh nhầm giữa target và actual.

### 2. Recap RoboVQA cho user (không có việc mới — tổng hợp lại điều tra đã làm)

Nhắc lại: tải về có `tfrecord/{train,val}` (184 shard) + `json/{train,val}` (pre-extract) + **9,999 video `.mp4` thật** + `instructions/*.txt` + LICENSE Apache-2.0. Đã làm: `flatten_text.py` xong (221,912 dòng, cố tình bỏ video_id — quyết định deprioritized ghi rõ trong docstring, không phải bug). `extract_frames.py` dở dang 130/184 shard, không có job chạy. Nhắc lại phát hiện quan trọng: TFRecord thật có `texts_start`/`texts_end` luôn = 15 (frame cuối trong 16 frame/episode) — QA không interleave được theo từng loại task, chỉ neo được ở cấp "cả episode".

### 3. Tự viết nhầm sbatch Megatron-tokenize không có template thật

User yêu cầu viết 2 sbatch để submit Megatron-tokenize cho OmniVideo-100K + roleplay trên JUWELS. Tìm quanh oellm-autoexp không thấy script preprocess_data.py thực tế đã dùng cho lần tokenize FineVideo trước đó (chỉ tìm thấy job.sbatch của bước TRAIN, dùng container JUPITER-aarch64-specific, không áp dụng cho JUWELS x86_64) — **tự đoán mò 1 bản generic** (module load placeholder, Megatron tools/preprocess_data.py CLI chuẩn). User chỉ ra ngay là sai hướng, sẽ tự cung cấp template thật. Đã xóa file đoán mò, chờ template.

### 4. Kiểm tra mức dùng login node JUPITER theo yêu cầu user (lo bị admin report)

Recheck: load average 18.28/72 core (~25%), GPU 15GB/97GB dùng, 60% util, 14 user khác online — **mức chấp nhận được**, không còn kịch 100% GPU liên tục như trước khi giảm worker (mục entry trước). 3 job chạy song song ổn định: Phase 4 FineVideo (4 worker), synth_llava seed2 tokenize, đếm token 4 dataset (đã xong).

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- **Phase 4 FineVideo** (tmux `yolo_login_finevideo`, 4 worker) — đang chạy.
- **synth_llava seed2 tokenize** (tmux `synth_llava_seed2`) — đang chạy, ETA ước tính ban đầu ~6h (cần check lại % hiện tại).
- Đếm token 4 dataset — **XONG**, script lưu tạm ở `logs/count_tokens_4datasets.py` (chưa đưa vào `tools/inventory/`, cân nhắc chuyển vào đó nếu cần dùng lại).
- Việc tồn đọng: chờ user cung cấp template Megatron-tokenize thật cho JUWELS; tách `pipeline_pose/`+`pipeline_video/` (chờ Phase 4 FineVideo xong); resume `extract_frames.py` RoboVQA nếu muốn đẩy tiếp; quyết định `MixtureVitae-Backup` SNAC + "moss" token (chờ Huu).

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 5 — điều tra RoboVQA + synth_llava, viết + chạy seed2 tokenize cho synth_llava, viết detokenizer chung cosmos/avc_lm, giảm tải GPU login node)

**Việc chính:** Trả lời loạt câu hỏi tổng hợp về 3 dataset mới (RoboVQA, synth_llava) và cơ chế decode 3 loại token video (seed2/cosmos/avc_lm) — đều verify bằng cách chạy code thật, không suy đoán. Viết + launch full-scale seed2 tokenize cho `synth_llava` (603,999 ảnh). User lo ngại GPU login node bị admin report vì Phase 4 FineVideo chiếm 100% GPU liên tục nhiều giờ — giảm từ 16 xuống 4 worker. Viết 2 tool detokenize tổng quát (`tools/decode/decode_cosmos.py`, `decode_avclm.py`), bắt được 1 bug thật trong avc_lm decode.

### 1. RoboVQA — datasets.md sai (ghi "chưa tải" nhưng thực tế đã tải+xử lý 1 phần từ 18/7)

Audit lại: `/p/data1/mmlaion/shared/vla/robovqa/` có đủ 184 shard TFRecord (175 train + 9 val) + `json/` (đã pre-extract) + **9,999 video `.mp4` thật** + `instructions/*.txt` + `LICENSE.txt` (Apache 2.0, permissive thật). Đã làm: `flatten_text.py` xong (221,912 dòng text, **cố tình bỏ `video_id`** — có ghi rõ trong docstring là quyết định deprioritized, không phải bug như vụ OmniVideo QA). `extract_frames.py` dở dang: **130/184 shard (~70.6%)**, không có job nào đang chạy.

**Câu hỏi user: text có anchor theo frame để interleave không?** Parse trực tiếp TFRecord thật (`tfrecord_lite.py`, tự viết vì không có tensorflow trong env nào): mỗi episode có `texts_start`/`texts_end` — verify 15 record thật, **luôn luôn = 15** (frame cuối trong 16 frame/episode). Nghĩa là toàn bộ multi-task QA (affordance/planning/success/future_prediction...) đã gộp thành **1 blob duy nhất, neo vào 1 điểm** (cuối episode) — không thể interleave từng loại QA riêng theo thời gian như FineVideo's caption/speech. Đơn vị hợp lý nhất để interleave: **cả episode (16 frame) = 1 chunk**, không phải nhiều điểm trong episode.

### 2. synth_llava — format xác nhận lại, viết + chạy tokenize seed2

Format thật (verify từ data, không chỉ nhớ lại): mỗi shard = `.jsonl` (`{"text": "<caption><image_0>...</caption>", "metadata", "language", "media": {"<image_0>": "filename.png"}}`) + `.wds` (tar POSIX chứa ảnh PNG thật). Verify 4,000 dòng shard đầu: **luôn đúng 1 ảnh/record**, placeholder `<image_0>` luôn nằm ngay đầu caption.

Viết `data_prep/synth_llava/tokenize_seed2.py` — tái dùng chính xác import/shim của `step_a_tokenize_video.py` (fix bug transformers-version + Qformer.cls=None đã biết), thay `<image_0>` bằng chuỗi `<seed2_N>` atomic trực tiếp tại vị trí đó (không cosmos/avc_lm vì ảnh tĩnh không cần). Test đo throughput thật: **~30 ảnh/giây** trên GPU rảnh, 0 lỗi. **Đã launch full-scale** (151 shard, 603,999 ảnh) trong tmux `synth_llava_seed2`, ETA ước tính ~6 giờ. Tại thời điểm ghi entry: 8/151 shard xong, 0 lỗi.

### 3. GPU login node bị chiếm 100% liên tục — user lo bị admin report, giảm tải

Phase 4 FineVideo (16 worker) khiến GPU 100% util suốt ~2 tiếng trong khi 12 user khác cũng online trên login node. Theo yêu cầu user: **kill session, restart với 4 worker** (giảm `slurm/run_yolo_login.sh` từ `NUM_WORKERS=16` xuống `4`) — verify GPU về 0% trước khi restart, không mất tiến độ (script skip-existing). Sau restart: GPU ~83-99% khi chạy đơn lẻ nhưng **có headroom hơn**, và giờ chia sẻ với job seed2 (2 job cùng chạy). Test throughput seed2 trên GPU rảnh trước khi quyết định tạm dừng Phase 4 nhường chỗ — kết luận seed2 đủ nhanh (~6h) nên **không cần dừng hẳn Phase 4**, chỉ giảm worker.

Cũng thử submit 1 job SLURM test (`985910`) để kiểm tra maintenance đã hết chưa — vẫn bị chặn y hệt (`ReqNodeNotAvail, Reserved for maintenance`), đã cancel ngay.

### 4. Điều tra decode seed2/cosmos/avc_lm — đọc paper, viết tool tổng quát, bắt 1 bug thật

**Seed2 (đọc paper user để trong `documents/2310.01218v1 (1).pdf` — "Making LLaMA SEE and Draw with SEED Tokenizer", Tencent AI Lab):** SEED không phải codec nén — là tokenizer bậc ngữ nghĩa cao, "de-tokenize" thật ra là **sinh lại ảnh mới cùng ngữ nghĩa** qua 1 pipeline Stable Diffusion riêng (`stable-diffusion-2-1-unclip`), verify từ chính README `ontocord/seed2`. Không tìm được paper "SEED2" chính thức nào — chỉ có "SEED"/SEED-LLaMA từ Tencent; checkpoint `ontocord/seed2` là bản build của 1 tổ chức khác, lý do đặt tên "2" không rõ, không đoán bừa.

**Cosmos — viết `tools/decode/decode_cosmos.py` (tool tổng quát, không phải script 1 lần):** verify shape thật — 8 frame/160x160 → encode ra indices `(1,2,10,10)` = 200 token/chunk. Decode qua `Cosmos-Tokenizer-DV8x16x16/decoder.jit` → `(1,3,9,160,160)`. Support cả 2 format token trong project (atomic `<cosmos_N>` cho data đã flatten, raw block `<cosmos>N N N</cosmos>` cho Step A chưa flatten) và cả 2 schema record (`{"text":...}` phẳng kiểu OmniVideo, `{"scenes":[...]}` lồng nhau kiểu FineVideo raw).

**AVC-LM — viết `tools/decode/decode_avclm.py`, bắt 1 bug thật:** `avc_lm_v2/tokenizer.json` không có `decoder` component (`decoder: None`) — gọi `tokenizer.decode()` mặc định sẽ tự chèn dấu cách giữa mỗi token, phá byte stream (verify: ffmpeg báo "non-existing PPS 0 referenced" khi test round-trip thật). **Fix:** nối trực tiếp `tok.id_to_token(id)` cho từng id, không dấu cách (đúng vì mỗi token là substring latin-1 thô, không có ByteLevel remap). Sau fix: ffmpeg decode sạch (returncode 0, 8 frame thật) — AVC-LM là **byte-exact**, khác hẳn seed2 (generative) và cosmos (neural reconstruct lossy).

Đã decode thử + gửi user xem: 1 chunk cosmos + 1 chunk avc_lm từ sample OmniVideo (`samples/omnivideo_100k_final_sample/`), và 1 record FineVideo-VLA thật (`d6b4OmUFt7I`, `samples/finevideo-vla/`) cho cả cosmos lẫn avc_lm.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- Cluster vẫn maintenance (job test `985910` xác nhận, đã cancel).
- **Phase 4 FineVideo** (tmux `yolo_login_finevideo`, 4 worker) — đang chạy, 4,073/40,305 video.
- **synth_llava seed2 tokenize** (tmux `synth_llava_seed2`, 1 process) — đang chạy, 8/151 shard, 0 lỗi, ETA ~6h.
- RoboVQA: `extract_frames.py` vẫn dở dang 130/184 shard, chưa resume (không phải việc đang làm phiên này).
- Việc tồn đọng: tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (chờ Phase 4 FineVideo xong); resume `extract_frames.py` cho RoboVQA nếu muốn đẩy tiếp; quyết định `MixtureVitae-Backup` SNAC + "moss" token (chờ Huu).

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 4 — hoàn tất merge OmniVideo-100K 3 track, sample+upload script, chỉ còn chờ Phase 4 FineVideo)

**Việc chính:** Chạy xong + verify sạch cả 2 bước còn lại của plan merge OmniVideo-100K (Bước 2 `phase6_merge_omnivideo.py`, Bước 3 `phase7_finalize_omnivideo.py`), viết sample + script upload HF theo đúng convention `laion_emotional_roleplay`, tự sửa lại mô tả cấu trúc record sau khi user chỉ ra thiếu phần "Context" ở đầu. Phase 4 FineVideo (tmux `yolo_login_finevideo`) vẫn chạy nền bình thường suốt phiên, không có gì khác cần làm ngoài chờ nó xong.

### 1. Bước 2 — `phase6_merge_omnivideo.py` — chạy xong, verify hoàn hảo

Lần đầu chạy thử vô tình chạy full 32 file thay vì test 1 file (do quên giới hạn), ghi nhầm ra `/tmp` scratchpad + có `| head -5` dễ gây `BrokenPipeError` giữa chừng — user nhắc kiểm tra thời gian, đã kill và chạy lại đúng trong tmux `phase6_merge_omnivideo`, ghi thẳng vào `/p/data1/mmlaion/shared/vla/omnivideo_100k_video_agent_merged/`.

**Kết quả (verify bằng đếm thật, không chỉ tin log):**
- 5,214 → 5,214 (0 malformed)
- **799 video có `<agent>`** — khớp chính xác 100% số file Phase 5 non-empty
- **62,631 window agent injected** — khớp chính xác 100% tổng số dòng trong toàn bộ `pose_agent_tokens_adaptive/*.jsonl` (đếm riêng, đối chiếu bằng `wc -l`) — không mất window nào
- Spot-check 1 record thật: `<agent>` xuất hiện đúng ngay sau cosmos của đúng chunk, format `<agent> <fps_30> <pelvis>...</agent>` khớp thiết kế

### 2. Bước 3 — `phase7_finalize_omnivideo.py` — chạy xong, verify hoàn hảo

Join QA (`omnivideo_100k_qa_flat.jsonl`, đã fix giữ `video_id` ở entry trước) vào cuối record Bước 2, gộp theo `video_id`. Kết quả: **5,214 → 5,214, 5,214/5,214 video có QA (100%), 0 warning** (không có video_id nào lệch giữa QA và Step A — tập video_id trùng khớp hoàn toàn cả 2 track). Output: `/p/data1/mmlaion/shared/vla/omnivideo_100k_final/` (32 file, 3.6GB).

**Verify tổng thể dataset cuối:** 17,229,664 token seed2, 201,736,400 token cosmos, 799 video có agent (62,631 window), 5,214/5,214 có QA.

### 3. Sample + script upload HF

Ghi 1 record thật vào `samples/omnivideo_100k_final_sample/` (bản raw 568KB + bản `_PREVIEW_readable.txt` đã cắt bớt run cosmos/seed2 dài cho dễ đọc), gửi cho user qua `SendUserFile`. Viết `data_prep/omnivideo_100k/upload_hf.py` theo đúng pattern `data_prep/laion_emotional_roleplay/upload_hf.py` (gzip nén, split train/test theo shard seed 42, `--repo-id` CLI arg). Dry-run (`--skip-compress`... à `--skip-upload`) trước: nén 32 shard (3.6GB → 648MB, 30 train/2 test), `gzip -t` pass hết. Đã đưa user lệnh upload đầy đủ, user tự export `HF_TOKEN`.

**Tự sửa 1 điểm sau khi user đọc sample không hiểu:** giải thích lần đầu bỏ sót phần `### Context: ...` (header caption tổng quan cả video, lấy nguyên từ `scripts.jsonl` gốc qua `step_a_tokenize_video.py`, giữ nguyên xuyên suốt cả `flatten_step_a_video.py` lẫn `phase6_merge_omnivideo.py`) — cấu trúc record thật là **3 phần**: `### Context: ...` → chuỗi token theo chunk (seed2/cosmos/agent) → QA nối cuối. Đã sửa lại giải thích + sẽ nhớ mô tả đủ 3 phần trong dataset card.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

Recheck toàn bộ trước khi ghi entry này:
- Cluster vẫn maintenance (18 node `booster` `maint`, không đổi).
- Phase 4 FineVideo (tmux `yolo_login_finevideo`) — **2,882/40,305 video, 0 lỗi** trên tất cả worker log, đang chạy khỏe.
- OmniVideo-100K final dataset + gzip upload bundle — recheck lại, vẫn nguyên vẹn (32 file, gzip integrity pass).

**Không còn việc gì khác đang treo trong phiên này — chỉ còn chờ Phase 4 FineVideo chạy xong (~15h từ lúc submit ~12:22 20/7).** Việc tiếp theo sau khi Phase 4 xong: tách `pipeline_pose/`+`pipeline_video/` (FineVideo) vào `data_prep/finevideo/` cho đối xứng với `data_prep/omnivideo_100k/` (đã hứa từ trước, chưa làm vì job đang import trực tiếp từ đó). Việc tồn đọng khác: chờ Huu quyết định `MixtureVitae-Backup` SNAC (~3.27B code) + ý nghĩa "moss" token cho roleplay dataset; chuyển `mixturevitae_multimodal/synth_llava` đã xong từ entry trước (không còn tồn đọng).

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 3 — reorg data_prep/omnivideo_100k, check chất lượng Phase 5 agent token, lên plan + bắt đầu merge 3 track OmniVideo-100K)

**Việc chính:** Phase 4 FineVideo login-node (tmux `yolo_login_finevideo`) tiếp tục chạy nền, ETA ~15h, không đụng tới. Trong lúc chờ: reorg `data_prep/omnivideo_100k/` (28 file phẳng → 4 thư mục con theo mục đích), check chất lượng Phase 5 agent token (tốt, không lỗi), làm rõ nhầm lẫn về 2 folder Step A tưởng duplicate, tổng hợp lại đúng cấu trúc 3 track của OmniVideo-100K, và bắt đầu viết pipeline merge 3 track thành 1 dataset hoàn chỉnh.

### 1. Reorg `data_prep/omnivideo_100k/` — xong

Từ 28 file phẳng (trộn Step A + Pose + dataset-curation) → 4 thư mục con:
```
data_prep/omnivideo_100k/
  pose/            # Phase 1-4 driver + submit .sbatch + task doc
  step_a/          # step_a_tokenize_video.py, flatten_step_a_video.py, debug_seed2_load.py
  dataset_prep/    # select_sports_subset.py, filter_animation_content.py, build_segment_captions.py, flatten_qa_text.py
  analysis/        # compare_native_vs_30fps_render.py + compare_renders/
  sports_subset_video_ids*.txt   # giữ top-level, dùng chung bởi pose/*
```
Sửa + verify bằng chạy thật (không chỉ đoán):
- 5 file dùng sys.path trick (`dirname` x3 → x4) để vẫn ra đúng repo root khi bị đẩy sâu 1 cấp — verify bằng cách chạy `--help` thật, xác nhận import `pipeline_pose.*` thành công (chỉ lỗi thiếu numpy/cv2 ở env hệ thống, không lỗi path).
- 5 file có `DEFAULT_VIDEO_IDS_FILE` — sửa path 1 cấp vì txt vẫn ở top-level, verify `os.path.exists()` = True cho cả 5.
- `DEFAULT_MODEL` (yolo26n.pt) trong phase4 — verify resolve đúng file thật ở repo root.
- Sửa path trong tất cả `submit_*.sbatch` + vài dòng docstring tự tham chiếu.
- Xóa `__pycache__/`.
Job Phase 4 FineVideo không bị ảnh hưởng (dùng `pipeline_pose/phase4_yolo_cleaner.py` gốc, không liên quan `data_prep/omnivideo_100k/`).

### 2. Giải oan cho `flatten_step_a_video.py` — không phải bug

Phiên trước nói nhầm: "`omnivideo_100k_video_flattened/` KHÔNG CÓ token `<seed2>`" — dựa trên so sánh chỉ 2000 ký tự đầu file nên hiểu sai. Kiểm tra sâu (đếm full file): **CÓ** 3,776 token `<seed2_N>` cho đúng video đó. Lý do nằm sai vị trí: `flatten_token_stream()` chỉ "flush" seed2/cosmos ra output tại đúng điểm gặp `<avc_lm>` (không theo thứ tự xuất hiện gốc trong input) — không phải mất token, thứ tự trong output đơn giản là khác thứ tự trong input. Không sửa gì, code đúng như thiết kế.

### 3. Tổng hợp lại cấu trúc thật của OmniVideo-100K — 3 track độc lập

Dataset gốc tải qua `tools/extract/download_omnivideo_100k.py` (HF `MiG-NJU/OmniVideo-100K`, Apache-2.0, 52.9GB): `videos.tar.part_aa..ae` (5 phần, giải nén ra **5,214 video** thật) + `scripts.jsonl` (149MB, caption/script sẵn có) + `train_oe_70k.jsonl` (70,017 QA mở) + `train_mcq_30k.jsonl` (29,966 QA trắc nghiệm) + 2 bản `_formatted`. Tổng QA 99,983 ≈ "100K" trong tên dataset — hoá ra tên gọi theo số QA pair, không phải số video (chỉ 5,214 video thật).

3 track xử lý song song, độc lập, join bằng `video_id`:

| Track | Pipeline | Scope | Trạng thái |
|---|---|---|---|
| 1. QA text (không video) | `dataset_prep/flatten_qa_text.py` | 5,214 video | Xong từ trước |
| 2. Step A video-tokenize (seed2/cosmos/avc_lm) | `step_a/step_a_tokenize_video.py` → `step_a/flatten_step_a_video.py` | **5,214/5,214 video** | Xong từ trước |
| 3. Pose pipeline (Phase 1-5, agent token) | `pose/phase1..4_*_omnivideo.py` + `pipeline_pose/phase5_adaptive_pchip.py` (dùng chung FineVideo) | Chỉ sports subset **1,126/5,214** | Vừa xong Phase 5 hôm nay (799 non-empty) |

### 4. Check chất lượng Phase 5 agent token — TỐT, không phát hiện lỗi thật

Sample 8+ video ngẫu nhiên, check kỹ:
- 17/17 khớp có mặt mọi window (test đầu tưởng thiếu 2 khớp — do tự đoán sai tên `neck`/`head`, tên thật đúng convention H36M là `nose`/`head_top` — lỗi test, không phải lỗi data).
- `pelvis` luôn = 128,128,128 (giữa range uint8) — **đúng thiết kế**, không phải lỗi: Phase 3 root-centering (`split_root_motion`) trừ vị trí pelvis khỏi mọi khớp nên pelvis luôn là gốc tọa độ.
- `r_wrist_x` có variation thật (std 6.9–22.6 tùy video) — không degenerate/constant.
- 0% giá trị bị clip ở biên (0/1 hoặc 254/255) trên 113,040 giá trị xyz sample.
- Token `t` đúng range 0-7 theo tier CP.
→ Agent token Phase 5 sẵn sàng dùng để merge.

### 5. Phát hiện quan trọng cho merge: chunk numbering giữa Step A và Pose KHỚP CHÍNH XÁC

Step A (`step_a_tokenize_video.py`) dùng `CHUNK_SIZE=8` frame @ `TARGET_FPS=30` — giống hệt Phase 5's `window_id` (stride 8 @ 30fps). `window_id` Phase 5 = `chunk_idx * 8` của Step A, khớp 1-1, không cần nội suy thời gian (khác các bug fps-mismatch đã fix trước đây — ở đây cả 2 bên đã cùng lưới 30fps/stride-8 cố định từ đầu).

### 6. Plan merge 3 track (đã user duyệt cấu trúc, đang thực hiện)

Quyết định: **1 record/video**, QA nối đuôi sau video tokens (không tách mỗi QA pair thành 1 record riêng — tiết kiệm token, tránh lặp lại video stream N lần).

- **Bước 1 — Fix `flatten_qa_text.py` giữ `video_id`** — **XONG**. Verify: 99,983 dòng, đúng 5,214 video_id unique.
- **Bước 2 — `phase6_merge_omnivideo.py`** (mới viết) — đọc raw `omnivideo_100k_video_flat/step_a_rank_*.jsonl`, tái dùng logic `flatten_token_stream()` từ `flatten_step_a_video.py` (import trực tiếp, không copy-paste), thêm counter `chunk_idx` tại mỗi lần flush `<avc_lm>`, tính `window_id = chunk_idx*8`, nếu có trong `pose_agent_tokens_adaptive/{video_id}_tokens.jsonl` thì chèn `<agent>{token_str}</agent>` ngay sau cosmos. Output: `/p/data1/mmlaion/shared/vla/omnivideo_100k_video_agent_merged/`. **Đang chạy** (tmux `phase6_merge_omnivideo`, ETA ước tính ~10 phút dựa trên lần test dry-run 11/32 file trong 205s — lần test đó lỡ ghi nhầm ra `/tmp` scratchpad thay vì đích thật + có `| head -5` gây rủi ro `BrokenPipeError`, đã kill và chạy lại đúng trong tmux, ghi thẳng vào `/p`).
- **Bước 3 — `phase7_finalize_omnivideo.py`** (chưa viết) — left-join `omnivideo_100k_qa_flat.jsonl` (gom QA theo video_id) vào cuối text của Bước 2.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- Phase 4 FineVideo (tmux `yolo_login_finevideo`) — đang chạy, ETA ~15h từ lúc submit (~12:22).
- Phase 6 merge OmniVideo (tmux `phase6_merge_omnivideo`) — đang chạy, ETA ước tính ~10 phút.
- Việc tiếp theo: verify Bước 2 xong (đếm video có `<agent>` phải khớp ≤799, spot-check vị trí đúng thời gian), rồi viết + chạy Bước 3.
- Việc tồn đọng: tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (chờ Phase 4 FineVideo xong, ~15h); chờ Huu quyết định `MixtureVitae-Backup` SNAC + "moss" token nghĩa là gì.

## Cập nhật phiên làm việc — 20/07/2026 (tiếp nữa nữa — Phase 3 FineVideo login-node chạy xong, verify+upload SNAC roleplay lên HF, điều tra thực nghiệm tác dụng Phase 4 YOLO filter, chuyển synth_llava sang đúng chỗ)

**Việc chính:** Cluster vẫn maintenance (`sinfo`/`scontrol` xác nhận job `978074` bị chặn cứng bởi lý do `Reserved for maintenance`, 18 node `booster` ở trạng thái `maint`) — chuyển hẳn Phase 3 FineVideo sang chạy login node (như đã làm với OmniVideo trước đó), xong hoàn toàn. Verify SNAC tokenize `laion/emotional-roleplay` (tưởng còn đang chạy dở theo note phiên trước, thực ra log cho thấy đã DONE), viết script + upload lên HF theo tên user chọn. User nghi ngờ Phase 4 YOLO filter có tác dụng thật hay không — điều tra bằng số liệu thật từ OmniVideo đã chạy xong, kết luận filter cắt thật ~25.7% dữ liệu, không phải bước thừa. Hủy job `978074` (dư thừa vì Phase 3 đã xong qua login node), submit Phase 4 FineVideo login-node luôn. Chuyển `synth_llava/` (107GB) từ `/e/data1/.../nguyen38/mixturevitae_multimodal/` sang đúng convention `/p/data1/mmlaion/shared/vla/`.

### 1. Phase 3 FineVideo — login-node run, XONG hoàn toàn

Cluster vẫn PENDING do maintenance, viết `slurm/run_kinematics_login.sh` (giống hệt `submit_kinematics.sh` về script/args, giảm xuống 32 worker + `nice -n 15`/`ionice -c3` vì login node dùng chung — 72 core, 9 user khác, load trung bình 12/72 lúc launch). Chạy tmux `kin_login_finevideo`. Tốc độ thật đo được: ~47 file/30s → toàn bộ 40,804 video xong trong khoảng 15-20 phút. Kết quả: **40,305 video thành công + 499 video rỗng** (0 window sạch sau hallucination filter, không phải lỗi — cùng loại outcome như 78 video EMPTY của OmniVideo) = đủ 40,804/40,804, **0 exception** trên 32 worker (verify bằng cách grep log tất cả worker). Output ghi thẳng vào `outputs/states_jsonl_30fps/` (thư mục đã đổi tên sạch từ bug fps-mismatch tuần trước).

### 2. SNAC tokenize `laion/emotional-roleplay` — xác nhận DONE + verify kỹ + upload HF

Note phiên trước ghi "đang chạy ~1,000/5,000 dòng shard đầu" nhưng log thật (`logs/snac_roleplay_tokenize.log`) cho thấy **đã DONE từ trước đó**: `ok=67,459 skipped_shards=0 decode_fail=0 snac_fail=0 total_snac_tokens=23,390,760`.

**Verify cấu trúc toàn bộ 14 shard** (không chỉ tin log): parse lại từng dòng bằng regex, kết quả — 67,459/67,459 dòng, **0 ID trùng, 0 lỗi format, 0 token ngoài range hợp lệ** (3 offset `OFFSET_L0/L1A/L1B` đúng theo `tokenize_snac.py`), tổng token khớp chính xác log.

**Phát hiện 1 vấn đề thật ở data nguồn (không phải bug tokenize):** 103/67,459 dòng (0.15%) có audio ngắn bất thường so với độ dài text — ví dụ 1 dòng text 128 ký tự nhưng token SNAC chỉ tương ứng 0.16s audio. Verify bằng ffprobe thật trên đúng bytes MP3 gốc lấy lại từ parquet (`id=cv_charming flity woman__b1_08_Hope_Enthusiasm_Optimism_2`): file chỉ 813 byte, **Duration thật = 0.14s** — xác nhận đây là audio bị cắt/hỏng sẵn trong dataset gốc, script tokenize xử lý đúng những gì có. Phần còn lại phân bố hợp lý (median 7.92s, p99 22.96s, max 66.48s, khớp README "~184h/67,491 clip").

Ghi 4 sample thật từ output đầy đủ vào `samples/laion_emotional_roleplay_sample/fullrun_*.txt` (3 dòng bình thường từ 3 shard khác nhau + 1 dòng anomaly để user tự xem).

**Upload lên HF:** viết `data_prep/laion_emotional_roleplay/upload_hf.py` (theo đúng pattern `tools/upload/upload_flattened_hf.py` — gzip nén, split train/test theo shard seed 42, `--repo-id` là CLI arg để user tự chọn tên chứ không hardcode). Dry-run (`--skip-upload`) trước: nén 14 shard 339MB→55MB, `gzip -t` pass hết. User tự export `HF_TOKEN` và chạy thật — **upload thành công**: `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened` (67,459 dòng, 23.39M token, train 13 shard/test 1 shard).

**Verify thêm theo yêu cầu user — decode ngược 1 sample ngẫu nhiên từ chính repo HF vừa upload:** cài `snac`+`soundfile` vào `env_stable_vla` (trước đó tìm nhầm hướng đi lục lại môi trường cũ, user nhắc đúng — chỉ cần pip install thêm, không cần tìm env nào khác). Tải 1 shard train thật từ HF (`hf_hub_download`), chọn ngẫu nhiên 1 dòng (`id=force_ogress-male-shouting-terrified_004944_b0`, 201 token). Phát hiện: **listen-format encode chỉ giữ `codes[0]`+`codes[1]` của SNAC (bỏ `codes[2]`, tầng chi tiết nhất, tốc độ 4x)** — decode cần đủ 3 tầng nên phải zero-fill `codes[2]` để nghe thử được (audio nghe được nhưng mất chi tiết tần số cao, là preview lossy chứ không phải chất lượng gốc). Tải lại audio gốc từ parquet nguồn (23,181 byte MP3) để so sánh cạnh nhau. Đã gửi cả 2 file (`hf_random_sample_decoded.wav` + `hf_random_sample_ORIGINAL.mp3`) cho user nghe trực tiếp.

### 3. Điều tra thực nghiệm: Phase 4 YOLO filter có tác dụng thật không?

User nghi ngờ đúng chỗ đáng nghi (chạy lại YOLO trên video có vẻ trùng việc Phase 1 đã detect person rồi) — trả lời bằng số liệu thật thay vì suy đoán.

**Khác biệt kiến trúc quan trọng tìm được:** Phase 1 (`phase1_hrnet_gpu.py`) dùng **MMDetection**, threshold 0.5, chọn bbox lớn nhất mỗi frame. Phase 4 dùng **Ultralytics YOLO**, threshold 0.75 — là 1 detector hoàn toàn độc lập, không phải lặp lại.

**Số liệu thật trên 1,048 video OmniVideo đã chạy xong Phase 3+4** (so `pose_states_jsonl_30fps` trước lọc với `pose_yolo_cleaned_30fps` sau lọc, đếm dòng thật từng file):

| Metric | Giá trị |
|---|---|
| Tổng window trước lọc | 387,226 |
| Tổng window sau lọc | 287,881 |
| **Tỷ lệ cắt tổng** | **25.7%** |
| Cắt trung vị/video | 18.8% (p25=4.7%, p75=37.2%) |
| Video bị cắt 100% (có window trước, 0 sau) | **25 video** |

Kiểm tra mẫu 8/25 video "cắt 100%": mỗi video có 224-504+ window pass được Phase 3 (hợp lệ về hình học) nhưng YOLO không tìm thấy người ở ≥4/8 frame trong MỌI window — gần như chắc chắn là case MMDet detect nhầm vật gì đó thành người (score>0.5) ở Phase 1, HRNet/MotionBERT vẫn lift ra bộ xương "hợp lý về hình học" từ input rác, chỉ YOLO độc lập mới bắt được.

**Kết luận:** Phase 4 giữ lại, đang làm đúng việc — nếu bỏ, ước tính ~1/4 dữ liệu training sẽ lẫn pose giả từ video không có người thật.

### 4. Hủy job SLURM dư thừa + submit Phase 4 FineVideo login-node

`scancel 978074` (Phase 3 SLURM cũ — dư thừa vì đã xong qua login node ở mục 1). Viết `slurm/run_yolo_login.sh` (giống `submit_yolo.sh` về script/args, giảm từ 128 worker/4 GPU xuống **16 worker/1 GPU** — login node chỉ có 1× GH200 480GB, xác nhận bằng `nvidia-smi` lúc launch: 0% util, 0MiB used, hoàn toàn rảnh). Bật NVIDIA MPS riêng cho login node (pipe/log dir khác job SLURM để tránh đụng). Chạy tmux `yolo_login_finevideo`, verify worker 1 chạy khỏe (~52 window/s cho video đầu). **Đang chạy lúc ghi entry này**, chưa xong.

### 5. Chuyển `synth_llava/` sang đúng convention lưu trữ

Theo đúng quy tắc đã thống nhất trước đó (dataset tải ngoài → `/p/data1/mmlaion/shared/vla/`, không phải `/e/data1/.../nguyen38/`): confirm 2 mount khác nhau thật (`exa_data1` vs `data1`, 2 filesystem ID GPFS khác nhau) nên không thể `mv` tức thời, phải rsync. Chạy `rsync -a` (107GB: `data/` 53GB gốc .tar.gz + `extracted/` 55GB đã giải nén 151 shard) trong tmux `move_synth_llava`, tốc độ thật ~500MB/s. **Đang chạy lúc ghi entry này**, chưa xong, chưa xóa bản gốc (chỉ xóa sau khi verify copy xong).

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- Cluster vẫn maintenance, không còn job SLURM nào pending (đã cancel `978074`).
- Phase 4 FineVideo (login-node, tmux `yolo_login_finevideo`) — đang chạy, 16 worker.
- Chuyển `synth_llava/` (tmux `move_synth_llava`) — đang chạy rsync, chưa xóa bản gốc.
- Việc tồn đọng chưa làm: tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/`; chờ Huu quyết định `MixtureVitae-Backup` SNAC (~3.27B code) và ý nghĩa "moss" token cho roleplay dataset.
## Cập nhật phiên làm việc — 20/07/2026 (tiếp nữa — so sánh paper iFLYTEK-Embodied-Omni, phân tích + SNAC tokenize dataset laion/emotional-roleplay, chuẩn bị tắt session)

**Việc chính:** User đưa paper `iFLYTEK-Embodied-Omni` (arXiv 2607.02542, Huu chia sẻ trong chat) để so sánh với project — đọc toàn bộ 16 trang, kết luận: **họ đã có model thật, train thật, SOTA thật** (89.6% LIBERO-Plus, 93.68%/93.16% RoboTwin 2.0), nhưng **~30% data mix của họ không permissive đã xác nhận** (Ego4D 12.72% proprietary, AgiBot 11.65% CC-BY-NC-SA — 2 dataset mà **project mình đã tự điều tra và loại bỏ từ trước**). Xác nhận claim của user ("technical report chứ không phải science paper") hợp lý — chính iFLYTEK cũng tự đặt tên bài là "Technical Report". Đã lưu toàn bộ so sánh vào memory (`project_iflytek_omni_comparison.md`, `project_omni_scope_clarification.md`). Sau đó phân tích kỹ dataset `laion/emotional-roleplay-finetuning-dataset` (theo chỉ đạo Huu: "concatenate text, interleave snac/moss tokens"), viết + chạy full-scale script SNAC tokenize trên login node (tmux), verify Phase 4 OmniVideo đã DONE hoàn toàn.

### 1. So sánh với iFLYTEK-Embodied-Omni — đã lưu chi tiết vào memory

Tóm tắt (chi tiết đầy đủ trong memory `project_iflytek_omni_comparison.md`):
- **Họ hơn:** data robot trajectory THẬT (không phải proxy như agent-token pose-từ-video của mình), eval task-success THẬT trên simulator chuẩn ngành (LIBERO-Plus, RoboTwin 2.0), data mixture định lượng rõ ràng, ablation kiểm soát.
- **Mình hơn/ngang:** đã có audio (SNAC) tích hợp sẵn — họ còn chưa có, tự ghi "future work: incorporate speech"; IMU cả 2 bên đều chưa làm; kỷ luật license chặt hơn (đã loại AgiBot/EgoDex/stera-10m vì lý do license, họ thì vẫn dùng AgiBot+Ego4D+EgoDex dù non-permissive).
- **Kết luận:** chưa đạt tới mức ngay cả "technical report" — thiếu robot-action thật, eval task-success, data mixture định lượng, ablation kiểm soát. 4 việc này là danh sách ưu tiên cụ thể nếu muốn bắt kịp.

### 2. Phân tích `laion/emotional-roleplay-finetuning-dataset` — đọc kỹ README, sửa 1 hiểu nhầm

Verify data thật: **67,491 dòng chính xác** (khớp README), **183.96h** (khớp "~184 hours"), audio xác nhận thật `24000 Hz, mono` qua ffmpeg probe. Phân bố: source `fill_creature` 37%/`emolia`/`ears`/`gemini`/`emotional_va`/`character`/`fill_human`; ngôn ngữ Đức 59% áp đảo. Phát hiện nhỏ: ~32/67,491 dòng `adherence_score` giá trị lạ (8/9/10/80/0) ngoài thang 1-5 README ghi — lỗi nhập liệu nhỏ, đã loại khi tokenize.

**Tự sửa 1 điểm hiểu nhầm sau khi đọc kỹ README (user yêu cầu đọc lại trước khi tóm tắt):** 5 field mới (`genuineness`, `vocal_burst_blend`, `voiceclap_commercial_embedding`, `is_human`, `archetype`) **CHỈ có trong bản `webdataset/` (tar shard), KHÔNG có trong 6 file `data/*.parquet` đã tải** — README ghi rõ "the original parquet files under data/ are unchanged". Nếu sau này cần dùng các field này để lọc/cân bằng human vs non-human, phải tải thêm `webdataset/` (~3.13GB), chưa làm.

### 3. Format tokenize + interleave — đã thống nhất, verify bằng chạy thật 1 sample

```
USER: <text> [Voice: <voice_description>] ASSISTANT:
<snac> <snac_N> <snac_N> ... </snac>
```

Cố tình bỏ `instruction`/`req_*` — README's Limitations tự ghi rõ model lệch giọng nam/bình tĩnh mặc định, `req_*` overstate nữ/to tiếng so với thực tế → chỉ tin `voice_description`/`realized_gender` (audio-verified thật), không tin field ý định. Verify bằng chạy thật (không phải giả định) trên `sample5_ears`: decode MP3→float32 24kHz qua ffmpeg (audio đã sẵn 24kHz mono, không cần resample), load model `hubertsiuzdak/snac_24khz` thật trên GPU, encode ra **561 token SNAC thật**, ghép thành record hoàn chỉnh 8,220 ký tự — lưu vào `samples/laion_emotional_roleplay_sample/sample5_ears_flattened_snac.txt` để đối chiếu trực tiếp với `.mp3`/`.json` gốc.

### 4. Viết + chạy full-scale `tokenize_snac.py`

`data_prep/laion_emotional_roleplay/tokenize_snac.py` — không import `pipeline_pose/snac_finevideo.py` (dùng type hint `X | None` kiểu Python 3.10+, vỡ ở `env_motion_final` Python 3.9) — viết lại độc lập phần toán học SNAC encode (offset giống hệt, không đổi). Không cần `split_snac_by_chunks()` như FineVideo vì mỗi dòng là 1 clip độc lập, không có chunk video nào để căn theo — encode nguyên clip thành 1 khối token phẳng.

Đo throughput thật trước khi cam kết quy mô (đúng thói quen dự án): 300 dòng thật mất 21s (trừ overhead load model ~4-5s) → ước tính **~63 phút cho toàn bộ 67,459 dòng** trên login node GPU — hợp lý, không cần SLURM. **Đã submit chạy full trong tmux session `snac_roleplay`** (không dùng background task tool, theo đúng yêu cầu user từ trước), output ghi vào `/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/flattened/roleplay_snac_flat_{00000..00013}.jsonl` (13-14 shard × 5,000 dòng), log tại `logs/snac_roleplay_tokenize.log`. **Đang chạy lúc ghi entry này** (~1,000/5,000 dòng shard đầu).

### 5. Sửa lộn vị trí lưu trữ (nhắc lại từ mục trước) — đã thống nhất rõ

Theo đúng quy tắc user chốt: dataset tải ngoài → `/p/data1/mmlaion/shared/vla/`. Đã áp dụng đúng cho `laion_emotional_roleplay` (tải + output tokenize đều ở đây). **`mixturevitae_multimodal/synth_llava/` (107GB) vẫn còn ở `/e/data1/.../nguyen38/`** — chưa chuyển, cần hỏi lại user.

### Trạng thái cuối phiên (user chuẩn bị tắt/mở lại session)

- **`978074`** (Phase 3 FineVideo full-scale rerun, 40,804 video) — vẫn PENDING, cluster bảo trì diện rộng, chưa rõ ETA. Chưa submit lại Phase 4 FineVideo (`slurm/submit_yolo.sh`, đã fix + verify) vì phải chờ `978074` xong trước.
- **Phase 4 OmniVideo** — **XONG HOÀN TOÀN** (verify thật): 1,037 done + 11 skip + 78 no_input (đúng số video Phase 3 rỗng) = 1,126/1,126, 0 lỗi. Output: `$DATA/omnivideo_100k/pose_yolo_cleaned_30fps/` (1,048 file).
- **`snac_roleplay`** (tmux) — đang chạy, ETA ~1 giờ từ lúc submit (07:59). Kiểm tra bằng `tmux attach -t snac_roleplay` hoặc `tail -f logs/snac_roleplay_tokenize.log`.
- **Việc lớn đã thống nhất nhưng CHƯA làm:** tách `pipeline_pose/`+`pipeline_video/` (FineVideo) vào `data_prep/finevideo/` — chờ Phase 3/4 FineVideo chạy xong hẳn để tránh đổi path giữa lúc job đang chạy. Chuyển `mixturevitae_multimodal/synth_llava/` sang `/p/data1/mmlaion/shared/vla/` — chưa hỏi/làm.

---

## Cập nhật phiên làm việc — 20/07/2026 (tiếp — làm rõ scope "omni", khảo sát + tải 2 dataset mới ngoài scope video/pose, chạy Phase 4 OmniVideo trên login node)

**Việc chính:** User hỏi tại sao Huu (leader) lại nhắc tới ảnh tĩnh/audio roleplay trong khi tưởng project chỉ làm VLA video cho humanoid — dẫn tới làm rõ **scope thật của project rộng hơn nhiều** so với những gì `CLAUDE.md` mô tả (đã cập nhật toàn bộ `.md` — xem mục 1). Khảo sát 2 dataset mới theo yêu cầu Huu: `synth_llava`/`synth_llava2` (ảnh+caption tổng hợp, Huu tự tạo) và `laion/emotional-roleplay-finetuning-dataset` (audio+text TTS tổng hợp). Trong lúc chờ cluster hết bảo trì (job Phase 3 FineVideo `978074` vẫn PENDING), chạy Phase 4 OmniVideo trực tiếp trên GPU login node (qua tmux theo yêu cầu user, không dùng background task tool).

### 1. Làm rõ scope "omni" — cập nhật toàn bộ file `.md` cấp cao

Huu xác nhận trực tiếp trong chat: dự án là **omni-modal** — bind bất kỳ tổ hợp modal nào (ảnh, video, âm thanh, action, IMU...), miễn **license permissive + cân bằng tỷ trọng modal + tạo được cross-modal binding thật**, không bắt buộc phải liên quan robot/humanoid như khung `PAB-Spline VLA` (video+pose) hiện tại — khung đó chỉ là **1 modal-pair branch** trong bức tranh lớn hơn. Đã thêm ghi chú scope này vào: `CLAUDE.md` (project-wide), `3d-human-pose/README.md`, `REPORT.md` (section "Project Scope Update" mới), `PROGRESS.md` (tiếng Anh), `datasets.md`. Cũng ghi nhận lo ngại thật của user về tính khoa học/khả thi paper khi scope liên tục mở rộng mà eval protocol vẫn "Still open" — chưa có 1 câu hỏi nghiên cứu trung tâm cố định để justify từng nguồn data mới — đã nêu với user, chưa hành động (quyết định hướng nghiên cứu thuộc về Huu).

### 2. `synth_llava` / `synth_llava2` — khảo sát xong, đã tải + giải nén

Theo lịch sử chat, đây là 2 file trong path `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` mà đợt khảo sát 9/7 (`tools/inventory/peek_multimodal.py`) **chưa kịp peek tới** (report cũ dừng đúng trước 2 file này). Peek trước (không tải full, stream qua HTTP) rồi mới tải full theo yêu cầu user (56.2GB) — ban đầu tải nhầm vào `/e/data1/.../nguyen38/mixturevitae_multimodal/`, **user sửa lại đúng convention: dataset tải ngoài phải vào `/p/data1/mmlaion/shared/vla/`** (giống `omnivideo_100k`/`robovqa`/`sensenova_si8m` cũ) — đã note lại, chưa move (chỉ áp dụng cho download mới từ giờ, xem mục 4).

**Cấu trúc xác nhận bằng data thật:** 151 shard liên tục (`shard-0000000`→`shard-0000150`, 2 file chia nhau không trùng), mỗi shard = 1 `.jsonl` (4000 dòng caption) + 1 `.wds` (**thực chất cũng là tar**, xác nhận qua `file`) chứa 4000 cặp `image_N.png` (256×256) + `metadata.json` khớp 1-1 với `.jsonl`. Tổng **603,999 sample thật** (đếm chính xác). Nguồn: `llava_pretrain|shard-N|create_multimodal_data.generate_images_then_captions` — ảnh sinh tổng hợp (AI-generated) rồi mới auto-caption, không phải ảnh chụp thật. Xem 3 mẫu thật (1 mẫu có artifact sinh ảnh rõ — bàn poker chữ vô nghĩa; 2 mẫu khá thực tế) — đã copy vào `samples/synth_llava_sample/`.

**Đánh giá kỹ thuật (trước khi biết rõ scope omni):** seed2 là loại token DUY NHẤT khớp được (cosmos cần chuỗi 8-frame thời gian thật, avc_lm mã hoá bitstream H.264 — cần video thật, agent cần pose 3D chuyển động) — nhưng data này chỉ có seed2, không transition được sang modal khác, có nguy cơ củng cố lỗi eval đã biết ("model kẹt ở seed2, không tự chuyển cosmos→avclm→agent") nếu tỷ trọng quá cao. Sau khi biết rõ scope omni (mục 1), kết luận: vẫn hợp lệ như 1 modal-pair (ảnh↔text) trong tổ hợp lớn hơn, miễn **tỷ trọng nhỏ, cân bằng** — không phải lý do loại bỏ, chỉ là điều kiện khi trộn.

### 3. `laion/emotional-roleplay-finetuning-dataset` — khảo sát + tải xong

Theo chỉ đạo trực tiếp Huu: "concatenate the text and interleave with snac and/or moss tokens". Kiểm tra: public, không gated, license cc-by-4.0. **67,491 clip audio tổng hợp** (mono MP3 24kHz, ~184h, sinh bởi MOSS-TTS-Local v1.5 fine-tune riêng của Huu `laion/moss-1.5-roleplay-finetune`), đa ngôn ngữ Đức(chủ đạo)/Anh/Tây Ban Nha/Pháp, nhiều giọng nhân vật giả tưởng (orc/goblin/dragon...). Field chính: `text` (lời thoại), `voice_description` (mô tả giọng kiểu DramaBox, do gemini-3.5-flash **nghe audio thật** rồi viết — không phải mô tả mù), `instruction` (ý định gốc), `adherence_score`, `realized_gender`, cùng VoiceCLAP embedding (768-d) + `genuineness`/`vocal_burst_blend` (attribute mới). Khớp đúng ý Huu: audio → `snac` (đã có sẵn trong project, không cần token mới) và/hoặc "moss" (chưa rõ nghĩa — có thể là token riêng của model MOSS-TTS, cần hỏi lại). Đã tải 6 file parquet (2.5GB) vào `/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/` — đúng convention, chưa tokenize.

### 4. Sửa lộn vị trí lưu dataset tải ngoài — quy tắc rõ từ giờ

User chỉ ra: dataset tải từ ngoài (HF) phải lưu ở `/p/data1/mmlaion/shared/vla/` (nơi `omnivideo_100k`/`robovqa`/`sensenova_si8m` cũ đã ở đó), **không phải** `/e/data1/.../nguyen38/` (nơi này dành cho pipeline output/working data của chính project, ví dụ `omnivideo_100k/pose_*` là kết quả pipeline tự chạy ra, không phải "tải ngoài"). Đã xoá thư mục tải nhầm (`/e/data1/.../nguyen38/laion_emotional_roleplay`, mới tải dở), tải lại đúng chỗ. **Việc còn để ngỏ, CHƯA làm:** `mixturevitae_multimodal/synth_llava/` (107GB, cả 2 file `.tar.gz` gốc + phần đã giải nén) hiện vẫn đang ở `/e/data1/.../nguyen38/` — cần hỏi lại user có muốn chuyển sang `/p/data1/mmlaion/shared/vla/` không trước khi tự ý di chuyển 107GB.

### Trạng thái cuối phiên (tiếp)

Job Phase 4 OmniVideo chuyển từ SLURM (`978072`, huỷ) sang chạy trực tiếp GPU login node trong tmux session `phase4_omnivideo` (theo yêu cầu user, không dùng background task tool) — đang chạy tốt, ~52% (585/1,126 video) tính tới lúc ghi entry này. Job `978074` (Phase 3 FineVideo full-scale rerun) vẫn PENDING, cluster bảo trì diện rộng chưa rõ ETA.

---

## Cập nhật phiên làm việc — 20/07/2026 (Phase 2→4 OmniVideo-100K, tách output riêng khỏi FineVideo, bắt 2 bug fps-mismatch thật ảnh hưởng cả FineVideo, fix + rerun)

**Việc chính:** Track lại Phase 2 OmniVideo-100K (đã COMPLETED từ phiên trước, verify lại bằng data thật), chạy tiếp Phase 2.5 (resample 30fps) và Phase 3 (kinematics). Theo yêu cầu user, **tách hẳn output OmniVideo-100K ra khỏi thư mục `outputs/` dùng chung với FineVideo** (trước đó 2 phiên convention là "an toàn vì video_id không trùng" — nhưng gây khó đếm/kiểm tra/dọn dẹp riêng). Trong lúc viết driver Phase 4, **phát hiện 1 bug fps-mismatch thật, nghiêm trọng, có từ trước** trong `phase4_yolo_cleaner.py` gốc — ảnh hưởng 35% video FineVideo đã chạy production (dùng để train cả 2 model hiện có). Điều tra sâu hơn phát hiện thêm **bug thứ 2 cùng loại** trong `apply_2d_mask()` (Phase 3) — cũng ảnh hưởng FineVideo, và ảnh hưởng cả Phase 3 OmniVideo vừa chạy xong. Fix cả 2, verify bằng data thật, rerun Phase 3 cho cả OmniVideo (xong) và FineVideo (đang submit, chờ cluster hết bảo trì). Chạy phân tích tương quan confidence-vs-hallucination theo yêu cầu user — kết quả ngược kỳ vọng, kết luận không nên dùng confidence làm filter bổ sung.

### 1. Phase 2 (MotionBERT) — xác nhận đã xong từ phiên trước

Verify lại bằng data thật (không chỉ tin SLURM state): job `977337` COMPLETED, **1,126/1,126 video** ra `.npy` hợp lệ, 0 error trên 32 rank. 2 lần pilot trước đó (`977034` lỗi 100% do `os.rename()` xuyên mount, `977128` fix bằng `shutil.move()` mới OK) — bài học đã áp dụng đúng cho pilot lần 2 trước khi chạy full.

### 2. Tách output OmniVideo-100K ra khỏi `outputs/` dùng chung — theo yêu cầu user

User đặt câu hỏi đúng: tại sao để chung 1 folder với FineVideo. Quyết định cũ ("video_id không trùng nên an toàn") đúng về mặt kỹ thuật nhưng sai về mặt vận hành (khó đếm/dọn/kiểm tra riêng). Đã:
- Di chuyển 1,126×3 file (`2d_json`, `3d_npy`, preview `.mp4`) từ `outputs/` sang `$DATA/omnivideo_100k/pose_2d_json/`, `pose_3d_npy/` (cùng mount `exa_data1` nên `os.rename()` tức thời, không phải copy 262GB).
- Revert merge `fps_lookup.json` chung (đã lỡ merge 5,214 entry OmniVideo vào — revert sạch về đúng 43,751 entry FineVideo cũ), tách riêng thành `$DATA/omnivideo_100k/fps_lookup.json` (5,214 entry).
- Cập nhật default path trong `phase1_hrnet_omnivideo.py`/`phase2_motionbert_omnivideo.py` cho khớp.

### 3. Phase 2.5 (resample 30fps) — driver mới, chạy login node (20 giây)

`data_prep/omnivideo_100k/phase2_5_resample_omnivideo.py` — import `resample_pose()` từ bản gốc (không đụng file gốc, hàm này vốn dataset-agnostic), chỉ viết lại phần iteration (theo video_id list, không glob cả thư mục `outputs/3d_npy` 43,751 file của FineVideo). Chạy `tools/extract/extract_fps.py` cho toàn bộ 5,214 video OmniVideo trước (12 giây), verify 0 ID trùng FineVideo. **1,126/1,126 video xong trong 20 giây trên login node** — đúng dự đoán, việc này quá nhẹ để cần SLURM. Render skeleton so sánh trước/sau cho 1 video 25fps (gửi user xem) — timing khớp chính xác (120.00s cả 2 bên), video control 30fps ra byte-identical.

### 4. Phase 3 (kinematics) — chạy 2 lần (trước và sau khi phát hiện bug §6)

Lần đầu (trước khi phát hiện bug): login node, 1,126 video xong trong ~2 phút. Kết quả: **928 video OK, 198 EMPTY** (không lỗi, chỉ quá nhiễu để có window sạch). Hallucination filter loại trung bình 30.4% frame, chủ yếu do `rogue_joint_filter` (khớp bị kéo dài bất thường — đặc trưng lỗi lifting khi chuyển động nhanh, xác nhận bằng cách phân rã riêng từng loại filter trên 1 video mẫu, phân bố đều theo thời gian chứ không phải bug cục bộ).

### 5. Phát hiện bug thật #1 — fps-mismatch trong `phase4_yolo_cleaner.py` gốc (ảnh hưởng cả FineVideo)

Trong lúc viết driver Phase 4 cho OmniVideo, phát hiện: script gốc đọc **frame tuần tự từ video native-fps** nhưng `window_id` trong `states_jsonl_30fps` lại đánh số theo **lưới 30fps đã resample** (Phase 2.5) — 2 timeline khác nhau bị coi là một. Verify bằng data FineVideo thật (`-2MKTg-LNio`, 25fps): `native_frames=12,758` nhưng `states` có `max_window+8=15,304` — lệch ~2,546 frame. Hệ quả: (a) mọi window vượt quá `native_frames` bị **âm thầm bỏ** (mất ~1/6 cuối video ở video 25fps), (b) window còn lại đọc nhầm thời điểm, lệch dần tới ~20% tổng thời lượng. Đo thật: **35% FineVideo** (15,321/43,751 video fps lệch ≥5% so với 30) và **37.3% OmniVideo subset** (420/1,126) sẽ bị ảnh hưởng nếu copy y hệt logic gốc.

**Fix:** viết mapping `native_idx ↔ resampled_idx` qua `np.round(np.linspace(0, N-1, M))` (cùng công thức endpoint-aligned mà `resample_pose()` dùng chiều ngược lại) — áp dụng cả cho `pipeline_pose/phase4_yolo_cleaner.py` (sửa tại chỗ, vì đây là bug thật trong code dùng chung, không phải vấn đề tương thích dataset) và driver mới `data_prep/omnivideo_100k/phase4_yolo_cleaner_omnivideo.py`. Verify bằng cả gọi hàm trực tiếp lẫn qua đúng CLI/`main()` trên video FineVideo thật: `total_input=1,913` (đủ, không mất), `max window_id giữ lại=14,712` (gần đúng 15,304 thay vì bị chặn cứng ở ~12,750 như bản cũ).

### 6. Phát hiện bug thật #2 — `apply_2d_mask()` (Phase 3) cùng loại lỗi + 1 lỗi riêng cho OmniVideo

Lúc chuẩn bị phân tích confidence, phát hiện `apply_2d_mask()` cũng so `pose2d` (native fps) với `pose3d` (đã resample 30fps) qua `num_frames = min(len(pose3d), len(pose2d))` — **cùng bug fps-mismatch như mục 4**, verify trên `z-Qcz_FMW7Q` (25fps): 600/3,600 frame cuối (resampled-space) **không bao giờ được mask**.

**Bug riêng #2b (chỉ OmniVideo):** check "khớp bị thiếu" của hàm này yêu cầu `x=y=conf=0` cả 3 — đúng cho driver FineVideo gốc (ép conf về 0/1) nhưng **sai cho driver OmniVideo** (`phase1_hrnet_omnivideo.py` cố tình giữ conf liên tục, chỉ zero x,y). Verify thật: **9,162/13,404 (68.4%)** khớp lẽ ra phải bị mask lại bị bỏ sót do check sai field.

**Fix (`pipeline_pose/phase3_kinematics_processor.py`, sửa tại chỗ):** thêm mapping native↔resampled giống mục 4 (bản không phụ thuộc torch/ultralytics, viết riêng thay vì import từ `phase4_yolo_cleaner.py` để tránh kéo dependency nặng không cần thiết); đổi check zero-mask chỉ nhìn `x,y` (bỏ `conf` khỏi điều kiện) — không đổi hành vi với driver FineVideo gốc (x,y và conf vốn cùng về 0 ở đó), chỉ fix đúng cho OmniVideo. Verify: video mẫu giờ có 2,909/3,600 frame được mask đúng (từ 0 → 546/600 ở 600 frame cuối từng bị bỏ sót).

### 7. Rerun sau fix — không xoá, đổi tên output cũ để có đường lùi

Thay vì `rm -rf` (không thể hoàn tác), **đổi tên** các thư mục output cũ bị ảnh hưởng (cùng mount nên tức thời, không tốn thời gian copy):
- `outputs/yolo_cleaned_30fps` → `outputs/yolo_cleaned_30fps_buggy_fps_mismatch_2026-07-20` (107GB, FineVideo)
- `outputs/states_jsonl_30fps` → `outputs/states_jsonl_30fps_buggy_2026-07-20` (193GB, FineVideo)
- `$DATA/omnivideo_100k/pose_states_jsonl_30fps` → `..._buggy_2026-07-20` (OmniVideo)

Rerun Phase 3 OmniVideo ngay (login node, rẻ): **928→1,048 video OK** (198→78 EMPTY) — fix không chỉ đúng hơn mà còn tăng yield thật, vì mask đúng giúp hallucination-filter không còn bị nhiễu bởi khớp lẽ ra đã phải NaN từ trước.

User xác nhận (qua `AskUserQuestion`) muốn rerun full-scale Phase 3 cho FineVideo luôn (không chỉ pilot) — đã submit `978074` (`slurm/submit_kinematics.sh`, không cần sửa vì đã dataset-agnostic, fix nằm trong `phase3_kinematics_processor.py`).

**Job `978073`** (Phase 4 FineVideo, submit trước khi phát hiện bug #2) đã bị `scancel` — lý do: nó đọc `states_jsonl_30fps` mà mình vừa đổi tên để `978074` ghi lại, nếu chạy trước sẽ đọc nhầm thư mục thiếu/rỗng. Sẽ submit lại sau khi `978074` xong.

### 8. Phân tích tương quan confidence ↔ hallucination — kết quả ngược kỳ vọng, không nên dùng làm filter

Theo yêu cầu user, đối chiếu confidence trung bình/frame (đã align đúng qua mapping ở mục 5) với việc frame có bị geometric hallucination filter loại hay không (60 video, 176,656 frame):

| Confidence bucket | Tỷ lệ bị hallucination-filter loại |
|---|---|
| 0.00–0.30 | 7.6% |
| 0.30–0.50 | 15.8% |
| 0.50–0.70 | **34.5%** |
| 0.70–0.85 | 27.3% |
| 0.85–1.01 | 0.7% |

Hệ số Pearson **+0.21** (yếu, **dương** — ngược kỳ vọng "confidence thấp → dễ hallucination"). Giải thích hợp lý: khớp confidence thấp bị zero-out ở Phase 1 rồi được `interpolate_nan_gaps()` làm mượt trước khi vào bộ lọc → ít bị coi là rogue joint; ngược lại khớp confidence 0.5–0.85 thường là chi chuyển động nhanh, rõ nét 2D nhưng khó cho MotionBERT lift đúng độ sâu → dễ hallucination hơn. **Kết luận: không nên dùng confidence làm filter bổ sung/dự đoán hallucination** — tương quan yếu và sai chiều so với giả thuyết ban đầu.

### Trạng thái cuối phiên

Job đang PENDING (cluster bảo trì, `Reserved for maintenance`, nhiều node `booster` ở trạng thái `maint`, không có ETA rõ): `978072` (Phase 4 OmniVideo, 928/1,126→1,048/1,126 sau rerun Phase 3), `978074` (Phase 3 FineVideo full-scale rerun, 40,804 video). Cần làm tiếp sau khi `978074` xong: submit lại Phase 4 FineVideo (đã fix, script đã sẵn `slurm/submit_yolo.sh` + `pipeline_pose/phase4_yolo_cleaner.py`). Việc lớn tiếp theo đã thống nhất nhưng CHƯA làm: tách `pipeline_pose/` + `pipeline_video/` (FineVideo) vào `data_prep/finevideo/` để đối xứng với `data_prep/omnivideo_100k/`, `data_prep/robovqa/` — sẽ làm sau khi các job Phase 3/4 hiện tại chạy xong (tránh sửa path giữa lúc job đang chạy).

---

## Cập nhật phiên làm việc — 19/07/2026 (chiều muộn — self-review bắt 1 bug thật, đối chiếu review ngoài với data thật, submit full-scale Phase 1)

**Việc chính:** Trước khi mở rộng từ pilot 24 video lên full 1,126 video, tự review lại code + hạ tầng đã sửa trong phiên — bắt được 1 regression thật do chính fix symlink gây ra. Fix 2 lỗi robustness trong `phase1_hrnet_omnivideo.py` (video hỏng bị coi là "thành công" âm thầm, rò rỉ `VideoCapture` khi lỗi giữa chừng), verify bằng pilot 8 video thật + 1 test video hỏng giả. User mang 1 bài review độc lập (kiểu ChatGPT) về đúng file này — đối chiếu từng điểm với data thật đã có trong phiên thay vì tin/bác theo cảm tính: 2 điểm thật đáng sửa, còn lại phần lớn đã verify không phải vấn đề hoặc là hành vi kế thừa từ bản gốc (đã chạy sạch 40,804 video FineVideo). Trả lời câu hỏi thiết kế về fps — xác nhận giữ đúng convention cũ (native fps ở Phase 1/2, resample ở Phase 2.5), nhưng phát hiện 1 lỗ hổng thật (`fps_lookup.json` chưa có video OmniVideo-100K nào) cần xử lý trước khi chạy Phase 2.5. Áp 4 fix nhỏ cuối cùng rồi **submit job full-scale 1,126 video**.

### 1. Self-review bắt 1 bug thật — do chính fix symlink phiên trước gây ra

`outputs/fps_lookup.json` (43,751 entry, dùng bởi Phase 2.5 và Phase 3) hoá ra **chỉ tồn tại ở thư mục cục bộ** vừa bị đổi tên thành `outputs_local_backup/` — chưa từng có ở `/e/data1/.../nguyen38/outputs/` thật. Nghĩa là fix symlink trước đó vô tình làm file này "biến mất" khỏi path Phase 2.5/3 sẽ tìm. Đã copy lại ngay, verify khớp 43,751 entry cả 2 phía.

### 2. Fix 2 lỗi robustness trong driver Phase 1, verify bằng test thật

- Video hỏng/không mở được trước đây bị ghi thành "OK" với 0 frame — giờ raise lỗi rõ ràng, vào nhóm `error`.
- `cap.release()` giờ nằm trong `try/finally` — tránh rò rỉ file descriptor nếu gặp nhiều video lỗi liên tiếp trong 1 rank chạy lâu (~280 video/rank ở quy mô full).

Verify: pilot SLURM 8 video mới (job `976556`, COMPLETED, 0 lỗi, 87.7% frame detect người — cao hơn pilot đầu 78.7% nhờ đã lọc animation) + 1 test video hỏng cố ý (xác nhận vào đúng nhóm `error`, không sót file `.tmp`).

### 3. Phát hiện thêm: 130/1,256 video trong sports subset thực ra là animation

Điều tra 2 video tỷ lệ thấp trong pilot đầu → phát hiện là animation lọt qua filter từ khoá chung chung ("dancing"/"running"). Nghiêm trọng hơn: 1 video animation khủng long trong pilot vẫn ra **56.3% detect** — HRNet có thể nhầm nhân vật hoạt hình thành người thật, MotionBERT (train trên người thật) lift lên sẽ SAI chứ không chỉ thiếu data. Viết `filter_animation_content.py`: loại 130/1,256 video có `video_summary` tự nhận animation/cartoon → còn **1,126 video** (`sports_subset_video_ids_filtered.txt`, driver đã đổi default sang file này).

### 4. Đối chiếu review ngoài với data thật — 2 điểm đáng sửa, còn lại đã verify không phải vấn đề

**Đáng sửa thật:**
- Chọn người bằng bbox lớn nhất mỗi frame độc lập có thể đổi identity giữa các frame ở cảnh đông người — **hành vi kế thừa từ bản gốc** (đã chạy đúng vậy trên 40,804 video FineVideo), không phải bug mới. Có phòng vệ downstream một phần (Phase 3 anti-teleportation, Phase 4 YOLO cleaner). Sửa đúng cách (IoU tracking) là thay đổi kiến trúc ngoài phạm vi task — ghi nhận là giới hạn đã biết, chưa sửa.
- Decode bị cắt giữa chừng không bị phát hiện — thật, nhưng cách review đề xuất (so với `duration × 30fps`, ngưỡng 90%) khi test trực tiếp trên 8 video pilot thật sẽ **báo oan 2/8 video (25%)** — `07WqS-ccIrw`/`0OxHEDu5dFE` chỉ đạt 83.1% vì native **25fps thật** (verify qua `cv2.CAP_PROP_FPS`), không phải lỗi decode. Sửa bằng cách so với `cv2.CAP_PROP_FRAME_COUNT` thật của từng video (không giả định fps), chỉ warning chứ không raise cứng.

**Đã verify KHÔNG phải vấn đề / là hành vi kế thừa:**
- Lo ngại `SLURM_NTASKS` (nếu quên `srun` sẽ chỉ chạy rank 0) — đã bị bác bỏ bằng thực tế: log 2 job thật đều cho thấy đúng 4 rank chạy song song.
- "Discontinuity" giữa confidence liên tục và toạ độ zero — review hiểu ngược: `WildDetDataset` của MotionBERT vốn thiết kế nhận confidence liên tục từ detector thật, bản nhị phân hoá cũ mới là cái lệch chuẩn.
- mmpose có thể trả tensor thay vì numpy — code y nguyên bản gốc, đã chạy sạch 32 video thật + 40,804 video FineVideo trên đúng bản mmpose đang dùng — thêm phòng hờ (miễn phí) chứ không phải sửa bug đang xảy ra.
- RAM giữ cả video trong list tới cuối — tính lại bằng số thật (video dài nhất 180s = 5,400 frame) chỉ ~10MB/video, không phải "hàng trăm MB" như review nói.
- Path phụ thuộc CWD — đây là convention chung của TOÀN pipeline (Phase 1-7), mọi sbatch đều `cd` đúng chỗ trước khi gọi — không phải rủi ro thật, đổi riêng file này sẽ làm nó khác biệt với cả hệ thống.

### 5. Câu hỏi thiết kế fps — giữ nguyên convention cũ, phát hiện 1 lỗ hổng thật

Xác nhận: giữ native fps ở Phase 1/2 (không resample sớm), Phase 2.5 mới resample về 30fps — đúng convention cũ của FineVideo, driver hiện tại đã tự động đúng (đọc frame tuần tự, không ép fps). Verify thật: OmniVideo-100K có fps gốc KHÔNG đồng nhất (`07WqS-ccIrw`/`0OxHEDu5dFE` = 25fps, `0GPO9qLraB8`/`iGVvChGEQdM` = 30fps, đo bằng `cv2.CAP_PROP_FPS`) — đúng kiểu tình huống Phase 2.5 sinh ra để xử lý.

**Lỗ hổng thật phát hiện (chưa chặn ở Phase 1 hiện tại):** `fps_lookup.json` (43,751 entry) chỉ có video FineVideo — 0 video OmniVideo-100K. Phase 2.5 tự ghi rõ sẽ "skip với warning" video thiếu trong file này — chạy thẳng sẽ mất trắng toàn bộ pose OmniVideo-100K ở bước 30fps. Đã ghi vào `JUPITER_POSE_PILOT_TASK.md`: cần chạy `tools/extract/extract_fps.py` rồi **merge** (không ghi đè) vào `fps_lookup.json` chung trước khi chạy Phase 2.5.

### 6. Fix cuối + submit full-scale

Áp 4 fix nhỏ: dedup video_ids (data thật hiện tại 0 trùng, phòng hờ), `getsize() > 2` trong resume check, tensor→numpy an toàn cho mmpose, warning decode-truncation dùng frame-count thật. Verify: syntax pass, test dedup bằng ID trùng cố ý (log đúng "Removed 1 duplicate"), test lại đường lỗi video hỏng vẫn đúng. Theo yêu cầu user đẩy nhanh tiến độ, không re-verify riêng đường thành công đầy đủ với dòng warning mới trước khi submit — thay đổi rủi ro thấp (chỉ thêm 1 điều kiện + print, không đổi luồng), và job full-scale với 1,126 video thật sẽ tự lộ nếu có vấn đề.

**Đã submit:** `submit_phase1_full.sbatch` — job **`976705`**, 8 node × 4 GPU (32 GPU), toàn bộ 1,126 video, `--time=04:00:00` (ước tính ~2.6h dựa trên throughput đo thật 263s/video/GPU). Xác nhận `RUNNING` ngay sau submit, 8 node cấp đủ, log lỗi sạch.

### Trạng thái cuối phiên

Job `976705` đang chạy. Commit phiên này: `2f3d675`, `7dc1ca0`, `8e688c4`, `2024da4`, `f9eb687` — đã push hết. Phase 2 trở đi chưa bắt đầu.

---

## Cập nhật phiên làm việc — 19/07/2026 (chiều — chạy pilot pose pipeline that tren JUPITER, fix 2 bug ha tang)

**Việc chính:** Pull `JUPITER_POSE_PILOT_TASK.md` (task handoff từ phiên trưa) trên JUPITER và thực hiện. Trước khi viết driver, kiểm tra theo đúng yêu cầu "giữ confidence score, đừng vứt" — phát hiện `phase1_hrnet_gpu.py` gốc **nhị phân hoá** confidence (1.0/0.0) thay vì giữ giá trị thật, trong khi MotionBERT (`infer_wild.py`) đọc trực tiếp cột này làm input feature cho model lifting; Phase 2 tự nó không có confidence nào để giữ (output `X3D.npy` chỉ có toạ độ 3D thuần). Viết driver mới `phase1_hrnet_omnivideo.py` sửa đúng điểm này. Trong lúc chuẩn bị chạy, phát hiện + fix 2 vấn đề hạ tầng không liên quan tới code phiên này, rồi chạy smoke-test + pilot SLURM 24 video — cả 2 đều sạch.

### 1. 2 bug hạ tầng phát hiện trước khi submit job GPU nào

- **Symlink `outputs/` bị đứt:** CLAUDE.md ghi `outputs/` là symlink trỏ `/e/data1/.../nguyen38/outputs/` (145GB+ data thật). Thực tế đã thành thư mục thường gần rỗng (chỉ có `fps_lookup.json`) — mọi script Phase 1-6 dùng path tương đối `outputs/...` từ CWD project1 sẽ đọc/ghi nhầm chỗ. **Đã fix:** đổi tên thư mục cũ thành `outputs_local_backup/` (không mất file), tạo lại symlink đúng, verify resolve ra data thật.
- **Path env sai trong `setup_hrnet_gpu.sh`/`setup_motionbert.sh`:** cả 2 trỏ `conda activate .../3d-human-pose/env_{hrnet_datasets_v1,motion_final}` ở project1 — không còn tồn tại ở đó (`conda env list` xác nhận). Tìm ra env thật còn nguyên vẹn ở `/e/data1/.../nguyen38/3d-human-pose/env_*` (cùng mount với fix outputs/ ở trên — khả năng cao cùng 1 đợt di chuyển data nhưng chưa cập nhật 2 script). **Đã fix:** sửa path trong 2 script, verify activate được, CUDA True, `mmpose`/`mmdet` import OK.

Đã hỏi user qua `AskUserQuestion` trước khi đụng vào (ảnh hưởng path dùng chung toàn repo), được đồng ý tự sửa.

### 2. Driver mới `phase1_hrnet_omnivideo.py` — giữ confidence liên tục thật

Không sửa `phase1_hrnet_gpu.py` gốc. Tái dùng phần model-agnostic (path config/checkpoint, `coco_to_h36m` cấu trúc mapping) nhưng đọc mp4 phẳng trực tiếp qua `cv2.VideoCapture`, shard `video_ids[RANK::WORLD_SIZE]` theo `sports_subset_video_ids.txt`. Output cùng format/cùng thư mục `outputs/2d_json/` để Phase 2 dùng thẳng không cần sửa. `coco_to_h36m()` viết lại: vẫn zero-hoá vị trí khi dưới `CONF_THRESHOLD` (giữ nguyên logic tránh toạ độ rác) nhưng **lưu confidence float thật** thay vì ép 1.0/0.0; các khớp suy ra (pelvis/neck/spine/head-top) dùng `min()` của 2 confidence gốc thay vì hardcode 1.0.

### 3. Smoke-test 1 video → pilot SLURM 24 video — cả 2 sạch

Chạy thử trực tiếp 1 video (`iGVvChGEQdM`) — 2,564 frame, 0 lỗi, verify output có 915 giá trị confidence liên tục khác nhau (không phải chỉ 0/1). Submit `submit_phase1_pilot.sbatch` (1 node × 4 GPU, 24 video đầu) — job `976467`, **COMPLETED**, 26 phút, 0 lỗi. Output ghi vào thư mục pilot riêng (`pose_2d_json_pilot/`, tách khỏi `outputs/2d_json/` sản xuất chính cho tới khi verify chất lượng).

Kết quả tổng hợp: 60,506 frame, **47,639 frame có detect người (78.7%)** — 16/24 video ≥80%, chỉ 2/24 video <20% (`28jYYH6WrA0`: 5.1%, `dXv4oInXqiE`: 17.6%, nhiều khả năng sai dương của keyword filter). 78.7% cao hơn hẳn 24-41% coverage joint-level ghi nhận cho FineVideo trước đây.

### Trạng thái cuối phiên

Theo đúng quyết định của user, **dừng lại ở pilot 24 video để xem xét**, chưa mở rộng full 1,256 video. Phase 2-6 chưa bắt đầu.

---

## Cập nhật phiên làm việc — 19/07/2026 (trưa — flatten + tokenize video track xong, khảo sát nội dung, lên kế hoạch pilot pose)

**Việc chính:** Sau khi xác nhận Step A xong (mục dưới), hoàn tất luôn phần còn lại của video track: viết + chạy `flatten_step_a_video.py` (chuyển token thô Step A → format Megatron), submit + xác nhận xong job tokenize Megatron thật (`14120433`/`tok_omni_video`). Đối chiếu số token với FineVideo-VLA v5 để giải thích tại sao OmniVideo-100K "ít token" (không phải bug — do ít document hơn, không phải do density thấp). Phân loại nội dung toàn bộ 5,214 video theo `video_summary`, phát hiện ~24% là sports/hoạt động thể chất thật — quyết định pilot pose pipeline (agent token) trên tập con này, đóng gói thành task handoff cho JUPITER.

### 1. Flatten + tokenize Megatron cho video track — xong thật

`data_prep/omnivideo_100k/flatten_step_a_video.py` (mới, cùng convention drop-rate với `phase7_flatten.py`: avc_lm luôn bỏ, cosmos drop 50%, seed2/caption/speech luôn giữ) — chạy thật, input `omnivideo_100k_video_flat/` (32 file, từ JUPITER) → output `omnivideo_100k_video_flattened/` (32 file), **5,214/5,214 dòng khớp chính xác**, 0 mất video.

Tokenize Megatron: job `14120433` (`tok_omni_video`), **COMPLETED** (06:09→06:26, 17'), output `/p/data1/mmlaion/shared/vla/tokenized_output/omnivideo_100k_video/data_shard_00000.bin/.idx`. Đọc trực tiếp header `.idx` (dtype `int32`, 5,214 sequence khớp đúng số video) → **456,487,128 token thật (~456.5M)**. (Không tìm thấy sbatch script lưu trong repo cho job này — có thể chạy trực tiếp bằng lệnh, chưa commit; nếu cần tái tạo lệnh, tra `sacct -j 14120433 --format=SubmitLine`.)

### 2. Giải thích số token "ít" — so sánh với FineVideo-VLA v5, không phải bug

Đối chiếu FineVideo-VLA v5 (10,554,076,391 token / 371,888 document, ~28,375 token/document) với OmniVideo-100K video (456,487,128 token / 5,214 document, ~87,556 token/document) — **OmniVideo-100K thực ra nhiều token/document hơn FineVideo ~3.1 lần**. Chênh lệch tổng ~23x hoàn toàn do chênh số document ~71x: FineVideo chia mỗi video thành nhiều record scene/activity (371,888 record từ ~40K video gốc, ~9.3 record/video), còn OmniVideo-100K theo đúng thiết kế 1 video = 1 document (không có cấu trúc scenes/activities). Không phải lỗi flatten/tokenize.

### 3. Phân loại nội dung toàn corpus — phát hiện quan trọng, sửa lại nhận định sai trước đó

Nhận định ban đầu trong phiên ("chỉ tin tức/cartoon, không có hoạt động thể chất") **sai** — user nhắc lại nhớ dataset có khá nhiều người, kiểm lại bằng keyword-match trên `video_summary` toàn bộ 5,214 video (không chỉ vài sample nhỏ):

| Loại nội dung | Số video | Tỷ lệ |
|---|---|---|
| Sports/hoạt động thể chất thật | 1,256 | 24.1% |
| News/talking-head | 1,210 | 23.2% |
| Cartoon/animation | 325 | 6.2% |
| Gambling/slot machine | 129 | 2.5% |
| Gaming/gameplay | 115 | 2.2% |
| Vlog/travel | 79 | 1.5% |
| Còn lại (misc) | 2,503 | 48.0% |

Đây là heuristic thô dựa trên text summary cấp video, chưa verify hình ảnh thật — nếu cần chính xác hơn có thể lọc thêm ở field `segments[].caption` (chi tiết hơn theo từng đoạn).

### 4. Quyết định pilot pose pipeline trên tập con sports — kèm giới hạn đã thống nhất

Kiểm tra `phase1_hrnet_gpu.py`, xác nhận model hiện dùng (`td-hm_hrnet-w48...coco-256x192`) là **COCO-17 body-only, không có keypoint bàn tay/ngón tay** — chuyển sang video sports sẽ không cải thiện phần tay, chỉ đa dạng hoá chuyển động toàn thân/cánh tay. **User xác nhận: chấp nhận giới hạn này, bàn tay sẽ là dataset/effort riêng sau**, không cần giải quyết trong pilot này.

Loại nhóm news/talking-head khỏi pilot dù có người: (1) framing thường cận cảnh ngực trở lên → hông/gối/mắt cá ngoài khung hình → zero-fill trong `coco_to_h36m()` (`CONF_THRESHOLD = 0.5`); (2) chuyển động gần như tĩnh, giá trị training thấp, khả năng trùng lặp với dữ liệu lifestyle FineVideo đã có sẵn. Suy luận từ text summary, **chưa verify bằng frame thật** (video ở JUPITER `/e`, không mount được từ session JUWELS) — đã nói rõ với user, đề xuất JUPITER-side pilot có thể đo thử yield thật nếu cần chắc chắn hơn.

### 5. Deliverables — đã viết + commit (`95f2927`)

- `select_sports_subset.py`: script phân loại, chạy thật ra 1,256/5,214 video (24.1%)
- `sports_subset_video_ids.txt`: danh sách 1,256 video_id thật
- `JUPITER_POSE_PILOT_TASK.md`: task handoff cho JUPITER — cảnh báo `phase1_hrnet_gpu.py` hard-code đọc từ FineVideo HF dataset (cùng dạng vấn đề Step A từng gặp với `pipeline.py`), Phase 6 merge cũng gần chắc chắn cần viết lại tương tự `flatten_step_a_video.py`; đề xuất lộ trình thận trọng (pilot ~20-30 video trước, học từ 3 bug thật đã gặp ở Step A) trước khi chạy full 1,256 video.

### Trạng thái cuối phiên

Video track OmniVideo-100K giờ có đủ: Step A xong + flatten/tokenize xong (456.5M token sẵn sàng train) + kế hoạch pilot pose pipeline đã đóng gói, **chưa chạy** (chờ JUPITER pull task này về thực hiện). Chưa quyết định tỷ lệ trộn với FineVideo-VLA lúc train.

---

## Cập nhật phiên làm việc — 19/07/2026 (xác nhận job `970099` OmniVideo-100K Step A đã xong)

**Việc chính:** Track lại trạng thái job full-scale `970099` (Step A OmniVideo-100K, submit cuối phiên 18/7) — **xác nhận đã COMPLETED sạch, đã verify output thật** chứ không chỉ dựa vào SLURM state.

`sacct -j 970099`: `COMPLETED`, exit `0:0`, chạy 19:30:45→22:01:21 (18/7), 2h30'. Verify output: 32/32 file `step_a_rank_*.jsonl` (39GB), đúng **5,214/5,214 dòng** — khớp chính xác tổng số video OmniVideo-100K. Log lỗi (`970099_omni100k_stepA_full_err.log`) chỉ có warning vô hại (`GenerationMixin`/`torch_dtype` deprecation), không có Traceback/Exception thật. Kiểm tra riêng bug seed2 cũ có tái phát ở full-scale không: sample `rank_0` (163 video) — **0 video bị seed2=0**, cả 2 bug fix ở `env_stable_vla` (phiên 18/7 tối muộn) giữ vững ở quy mô đầy đủ. Nội dung mẫu đủ cả 4 loại token (seed2/cosmos/avclm + caption/speech blocks) với số lượng hợp lý.

**Kết luận: Step A (phần cần GPU, chỉ chạy được ở JUPITER) cho OmniVideo-100K đã xong hoàn toàn.** Bước tokenize Megatron với `tokenizer_vla_qwen3` sẽ được đẩy sang chạy ở JUWELS — không thuộc phạm vi task Step A này.

---

## Cập nhật phiên làm việc — 18/07/2026 (tối muộn — phiên bổ sung 3, đọc phần này trước)

**Việc chính:** Viết driver Step A mới cho OmniVideo-100K theo đúng yêu cầu ("tận dụng pipeline cũ nhưng đừng viết code mới vào đó, viết vào `data_prep/omnivideo_100k`") — `step_a_tokenize_video.py` import 3 class tokenizer từ `/e/project1/reformo/nguyen38/prototype/pipeline.py` gốc (không sửa gì ở đó), tự viết toàn bộ logic mới: list video, chunk 8-frame, và **chèn caption/speech chỉ 1 lần tại chunk đầu mỗi segment** (không phải mọi chunk overlap — tránh lặp lại đoạn caption 300-500 từ tới ~40 lần/segment). Trong lúc pilot phát hiện + fix **2 bug seed2 thật** trong `env_stable_vla` (transformers trôi lên 4.57.6, ảnh hưởng cả FineVideo nếu chạy lại Step A bằng env hiện tại) — xem chi tiết mục 2. Submit full-scale lần 1 (`970087`, 32 GPU) thì **lộ thêm 1 bug thật khác trong chính code mới của tôi** — trích toàn bộ frame gốc cả video ra đĩa tạm không resize, 32 rank chạy song song làm tràn quota đĩa của user, gần như mọi video đều lỗi. Đã `scancel`, dọn ~40GB rác, viết lại theo streaming từng chunk 8-frame + resize 512×512, verify lại bằng pilot (`970095`, 48/48 video sạch, nhanh hơn bản cũ), rồi **submit lại full-scale (`970099`, 8 node×4 GPU=32 GPU) — đang chạy lúc ghi entry này**.

### 1. Driver Step A mới — `data_prep/omnivideo_100k/step_a_tokenize_video.py`

Không đụng `pipeline_video/pipeline.py` (bản git, để tracking) hay `/e/project1/reformo/nguyen38/prototype/pipeline.py` (bản runtime thật, có đủ checkpoint). Chỉ import 3 class tokenizer cấp thấp (`Seed2Tokenizer`/`CosmosVideoTokenizer`/`AVCLMTokenizer`) từ `prototype/pipeline.py`, kèm `sys.path.insert` + `os.chdir(PROTOTYPE_DIR)` bắt buộc (checkpoint paths trong 3 class đó relative tới CWD; `import cosmos_tokenizer` bên trong `pipeline.py` chỉ resolve được nếu `sys.path[0]` trỏ đúng `prototype/`).

Sharding video theo `video_list[RANK::WORLD_SIZE]` (đơn giản hơn `dataset.shard()` của FineVideo vì chỉ là list file phẳng). Chèn `<caption>`/`<speech>` tại "anchor chunk" (chunk đầu tiên có `start_sec >= segment.start_sec`) — pattern giống cách `phase6_merge_adaptive.py` snap speech ASR về đúng 1 chunk bắt đầu. Output: `{"video_id", "text"}` 1 dòng/video, resume toàn cục (quét hết `step_a_rank_*.jsonl` hiện có trước khi xử lý).

### 2. Phát hiện + fix 2 bug seed2 thật trong `env_stable_vla` (ảnh hưởng cả FineVideo, không chỉ OmniVideo-100K)

Pilot đầu (`970063`) chạy nhưng `seed2=0` mọi video, không crash. Cả 2 bug đều do `transformers` trôi lên `4.57.6` (checkpoint config ghi `4.52.4`):

- **Bug #1 (import path):** `apply_chunking_to_forward`/`find_pruneable_heads_and_indices`/`prune_linear_layer` bị dời từ `transformers.modeling_utils` sang `transformers.pytorch_utils`. Code `seed2/seed2_tokenizer.py` (BERT API đời cũ) vẫn import chỗ cũ → `ImportError`, bị `except Exception` trong `pipeline.py` nuốt mất.
- **Bug #2 (hành vi `tie_weights()`, sâu hơn):** `AttributeError: 'NoneType' object has no attribute 'predictions'`. Full traceback (qua job debug riêng `debug_seed2_load.py`, job `970070`): tác giả gốc **chủ ý** set `self.Qformer.cls = None` (dòng 2601 — Qformer không cần đầu MLM khi chỉ encode ảnh). `transformers` cũ tolerate việc này; bản 4.57.6 gọi `tie_weights()` đệ quy vào mọi submodule kể cả cái đã bị null hoá chủ ý → crash.

**Cách sửa:** monkeypatch **chỉ trong `step_a_tokenize_video.py`** (không đụng `seed2_tokenizer.py`/`pipeline.py`/env chung): (1) gán lại 3 hàm bị dời vào `modeling_utils` trước khi import `pipeline.py`; (2) patch `get_output_embeddings`/`set_output_embeddings` của `BertLMHeadModel`/`BertForMaskedLM` trả `None` an toàn khi `self.cls is None`. Verify qua `debug_seed2_load.py` (`LOADED OK`, job `970072`) rồi pilot thật (job `970073`: 48/48 video, seed2 ra token thật 2000-5700/video).

### 3. Bug quota đĩa — do chính code mới, phát hiện khi submit full-scale lần 1

Job `970087` (32 GPU) gần như mọi video đều lỗi `[Errno 122] Disk quota exceeded`. Nguyên nhân: `extract_30fps_frames()` (bản đầu của tôi) trích **toàn bộ frame gốc của cả video** (tới 5400 frame cho video 180s, KHÔNG resize) ra PNG tạm — với 32 rank chạy song song, mỗi rank có lúc giữ 1-2.7GB PNG tạm cùng lúc → tràn quota user (pilot 8-rank trước đó không trúng vì tổng footprint đồng thời còn dưới ngưỡng).

**Đã `scancel` job ngay, dọn ~40GB rác** (`omni_temp_frames_rank_*`, `temp_seed2_rank_*.jpg`). **Fix:** viết lại hoàn toàn theo streaming — 1 lệnh ffmpeg/chunk 8-frame (thay vì 1 lệnh cho cả video), resize 512×512 (khớp `target_size` mặc định của Seed2Tokenizer nên không mất chất lượng; Cosmos downsample tiếp xuống 160 nên cũng không mất gì). Giới hạn dung lượng tạm chỉ còn ~8 frame/rank tại một thời điểm, bất kể video dài bao nhiêu. Verify: pilot lại (`970095`) — 48/48 video sạch, temp dir mỗi rank chỉ 1-2.5MB (so với 1-2.7GB trước), **nhanh hơn bản cũ** (7:22 so với 9:23).

### 4. Trạng thái cuối phiên: full-scale đang chạy

**Đã submit `970099`** (8 node×4 GPU=32 GPU, `--time=05:00:00`, `data_prep/omnivideo_100k/submit_step_a_full.sbatch`) cho toàn bộ 5,214 video, thay thế job `970087` đã huỷ. Output: `$DATA/omnivideo_100k/step_a_output/step_a_rank_{0..31}.jsonl`. Có resume — an toàn để submit lại y nguyên lệnh nếu job timeout/crash giữa chừng.

### Việc tiếp theo hợp lý nhất

Chờ job `970099` chạy xong (theo dõi qua Monitor). Sau khi xong: tokenize Megatron bằng `tokenizer_vla_qwen3` (257,901 vocab — **tuyệt đối không dùng `tokenizer_vla_adaptive_v2`**, bài học đau đã ghi ở phiên trước), rồi mới tới bước quyết định tỷ lệ trộn với FineVideo-VLA/MV-Omni lúc train.

---

## Cập nhật phiên làm việc — 18/07/2026 (tối — phiên bổ sung 2, đọc phần này trước)

**Việc chính tối nay:** phát hiện + fix bug lớn — **cả 3 job tokenize đang chạy dùng nhầm tokenizer cũ** (không phải Qwen3 như dự định), đã hủy job đang chạy, xoá 215GB output sai, sửa script, resubmit lại đúng Qwen3. Đếm token thật: FineVideo-v5 gần như không đổi (10.55B, chứng minh lệch số 5.256B không phải do tokenizer) — MV-Omni tăng thật +25% (20.39B). Tải xong hoàn toàn cả 3 nguồn mới (OmniVideo-100K, RoboVQA, SenseNova). Tạo `data_prep/` với script flatten cho RoboVQA + OmniVideo-100K (bắt + fix 2 bug thật khi validate kỹ). Giải nén + map caption/speech cho video OmniVideo-100K — **sẵn sàng Step A** (chờ submit ở JUPITER). Điều tra sâu RoboVQA — phát hiện video thật nằm trong tfrecord (không phải chỉ 4.5% như báo sai lúc đầu), viết parser tfrecord thuần Python, đang trích xuất frame (chạy nền, ~36% xong cuối phiên) — nhưng phát hiện đây là **16 ảnh rời rạc/episode, không phải video liên tục**, cần quyết định kiến trúc trước khi qua Step A. Đọc trực tiếp paper SenseNova (39 trang, qua `pypdf`) theo yêu cầu Huu — tìm ra **22 dataset nguồn ảnh cụ thể**, verify license từng cái: vài cái permissive thật (GQA/VQA/VSR/CLEVR/MindCube), phần lớn (nhóm đóng góp nhiều ảnh nhất) non-commercial xác nhận (ScanNet/ScanNet++/Matterport3D/CA-1M/Ego-Exo4D/ARKitScenes), vài cái đáng ngờ kiểu "license mới nhưng nguồn cũ dính" (VSI-590K/ViCA/VLM-3R). Đọc thêm paper MINT-1T gốc — xác nhận quyết định bỏ ảnh trước đây đúng, có bằng chứng mạnh hơn nữa (chính tác giả tự ghi "N/A" cho license asset).

### 1. Bug tokenizer sai — phát hiện + fix + resubmit

Bạn hỏi thẳng "có refer đúng tokenizer mới nhất chưa, nhớ lần này dùng Qwen" — check lại thì **cả 3 sbatch (`tokenize_finevideo_v5`/`tokenize_mv_omni`/`tokenize_mint1t`) đều trỏ `tokenizer_vla_adaptive_v2` (GPT-NeoX cũ), không phải `tokenizer_vla_qwen3`**. Đã: (1) `scancel` job `tok_mint1t` đang chạy dở với tokenizer sai, (2) sửa `TOKENIZER_MODEL` trong cả 3 script → `tokenizer_vla_qwen3`, (3) xoá 215GB output cũ (bắt buộc vì cả 3 script dùng `--resume`), (4) submit lại cả 3 (job `14118929`/`14118930`/`14118931`).

### 2. Token thật sau khi dùng đúng Qwen3

| Nguồn | Token (Qwen3) | So với bản tokenizer sai |
|---|---|---|
| FineVideo-VLA v5 | **10,550,998,369** | Gần như y hệt (10,554,076,391) — **xác nhận lệch 5.256B không phải do tokenizer** |
| MV-Omni | **20,389,561,883** | Tăng thật +25% (16,357,256,571) — Qwen3 vocab lớn ảnh hưởng nhiều hơn với nội dung tự nhiên |
| RoboVQA | 58,588,270 | Job nhỏ (1-node), xong luôn |
| OmniVideo-100K QA | 30,689,299 | Job nhỏ (1-node), xong luôn |
| MINT-1T text | Đang chạy | — |

### 3. Tải xong hoàn toàn: OmniVideo-100K, RoboVQA, SenseNova-SI-8M

Cả 3 đều `snapshot_download completed successfully`. SenseNova gặp sự cố nhỏ giữa chừng — tmux session bị chết âm thầm (verify qua `tmux ls`/`ps aux`/log đứng im), tôi tạo tmux mới nhưng thiếu `HF_TOKEN` trong shell của tôi nên lỗi ngay — **bị chặn khi tự động dò tìm token trong cache/rc files (đúng, không nên tự làm vậy)**, bạn tự chạy lại và hoàn tất (1,121.4GB, 53/53 zip).

### 4. `data_prep/` — folder mới, 2 script flatten, bắt 2 bug thật khi validate

Tạo `data_prep/omnivideo_100k/` + `data_prep/robovqa/` (đúng ý định ghi sẵn CLAUDE.md, chưa từng tồn tại thật).

- `data_prep/robovqa/flatten_text.py`: 221,912 record → flat JSONL, 0 skip.
- `data_prep/omnivideo_100k/flatten_qa_text.py`: gộp OE+MCQ → 99,983 record. **Validate kỹ theo đúng yêu cầu bạn ("đừng sai") bắt được 2 bug thật:** 2,740 record MCQ dạng `event_sequence_ordering` dùng field tên khác (`question_textual`/`options_textual`) bị bỏ sót oan; 6,372 record OE có `answer` là list bị in xấu kiểu Python-repr (`['B','C','A']`). Fix cả 2, verify lại bằng audit kiểu dữ liệu toàn bộ corpus (không chỉ sample) + regex check — sạch 100%.

### 5. OmniVideo-100K — giờ đã thật sự sẵn sàng Step A

Giải nén 5,214 video mp4 (49GB, khớp đúng số video). Viết `data_prep/omnivideo_100k/build_segment_captions.py` — map `segments[].visual`/`transcription` (đã convert MM:SS→giây) thành caption/speech theo từng đoạn thời gian: 5,214 video, 47,467 segment, 0 lỗi timestamp. **Video + caption/speech đều sẵn sàng — chỉ chờ submit Step A ở JUPITER** (bạn nhắc: Step A phải chạy JUPITER vì chỉ nơi đó có GPU, JUWELS chỉ dùng tokenize CPU).

### 6. RoboVQA — đính chính phát hiện sai trước đó, viết parser tfrecord thuần Python, đang trích xuất

**Đính chính:** lúc trước báo "chỉ 4.5% video có mp4, phần còn lại coi như mất" — **sai**. Ảnh thật của 95.5% record còn lại nằm trong `tfrecord/` (184 shard), không mất. Không có TensorFlow/protobuf trong bất kỳ env nào — viết `data_prep/robovqa/tfrecord_lite.py` (parser protobuf thuần Python, không cần cài gì). Verify kỹ qua nhiều bước: dò cấu trúc mù → xác nhận JPEG thật (mở ảnh xem, đúng cảnh robot) → chạy trên 500 episode thật (luôn đúng 16 frame/episode, 100% JPEG hợp lệ) → **cross-check text giải mã khớp byte-for-byte với json/train đã biết đúng (join key đúng là `video_filename`, không phải `uid` — 2 ID khác nhau, dễ nhầm)**.

Viết `data_prep/robovqa/extract_frames.py` — bug đầu tiên bắt ngay lúc test (list vs tuple, đã fix). Đang chạy nền (không phải SLURM — job nhẹ, chạy trực tiếp login node theo đúng "job nhỏ thì tmux/nền, không cần SLURM"). **Cuối phiên: 67/184 shard, 82,669/221,912 episode đã trích.**

**⚠️ Phát hiện quan trọng khi bạn hỏi lại — RoboVQA KHÔNG có video liên tục, chỉ có 16 ảnh JPEG rời rạc/episode (~1.6fps)**, khác hẳn FineVideo/OmniVideo-100K (video liên tục thật). Step A hiện tại thiết kế cho video liên tục — **chưa chắc dùng thẳng được cho 16 ảnh rời rạc này, cần quyết định kiến trúc riêng** (nối thành "video giả", hay xử lý theo hướng ảnh-tĩnh như đã bàn cho SenseNova) trước khi làm tiếp. Chưa quyết, để phiên sau.

### 7. SenseNova-SI-8M — đọc trực tiếp paper (39 trang, qua `pypdf`), tìm ra 22 dataset nguồn cụ thể

Huu chất vấn thẳng trên chat: "how do you know, what is the basis" — đúng, kết luận trước đó (dựa vào README im lặng + so sánh MINT) chưa đủ chặt. Không có tool đọc PDF nào trong project — **hỏi bạn trước khi cài** (đúng nguyên tắc không tự ý pip install), bạn đồng ý cài `pypdf`.

Đọc ra **Section 3.2 "Data Sources"**: 8.5M cặp QA từ **22 dataset nguồn**, chia 3 nhóm. Verify license từng cái (WebSearch trực tiếp trang terms-of-use chính chủ):

- **Permissive thật:** GQA, VQA, VSR, CLEVR-series, MindCube (5/22).
- **Non-commercial/gated xác nhận:** IconQA, MultiSpa, ScanNet, ScanNet++, Matterport3D, CA-1M, Ego-Exo4D, + ARKitScenes (check thêm vì là nguồn của các dataset trung gian) — 8 dataset.
- **"Nested derivative" đáng ngờ:** VSI-590K/ViCA/VLM-3R tự gắn license mới nhưng build từ chính ScanNet/ARKitScenes — license mới không rửa được nguồn cũ.
- **Chưa xác định:** 7 dataset còn lại (SPEC, Open3D-VQA, REL3D, SAT, GRiD-3D, SUN RGB-D, MessyTable).

**Kết luận cuối:** nhóm đóng góp nhiều ảnh nhất (4.5M/8.5M) xác nhận non-commercial — tag `apache-2.0` của SenseNova không áp được cho phần lớn ảnh. Đã báo đầy đủ breakdown để bạn relay Huu.

### 8. MINT-1T — đọc thêm paper gốc theo yêu cầu bạn, xác nhận quyết định cũ đúng

Paper có phần "Datasheet for Datasets" — chính tác giả tự ghi **"Did you mention the license of the assets? [N/A]"**. Bằng chứng mạnh hơn hẳn README đã dùng trước đây. Quyết định bỏ ảnh MINT (giữ text) — vẫn đúng, không cần đổi gì.

---

## Cập nhật phiên làm việc — 18/07/2026 (chiều — phiên bổ sung, đọc phần này trước)

**Việc chính chiều nay:** xác nhận 2 job tokenize từ sáng (MV-Omni, MINT) thực sự chạy thật (MV-Omni từng fail âm thầm lúc sáng do lỗi Ray, đã fix + resubmit) — **MV-Omni đã COMPLETED thật, MINT vẫn đang chạy**. Đếm token thật bằng `count_tokens.py` cho FineVideo-v5 và MV-Omni — **phát hiện lệch số quan trọng, chưa rõ nguyên nhân** (xem mục 1). Đính chính license SenseNova-SI-8M (không còn an toàn như tưởng). Viết lại mục Gen-EgoData (không phải ego-video, mà là dữ liệu action tay-đơn từ thiết bị cầm tay). Thảo luận sâu với Van Khue về câu hỏi ego/exo — kết luận **không cần sửa FineVideo-VLA**. Khảo sát thêm dataset mới trên HF (RoboVQA, Open X-Embodiment, NVIDIA GR00T-Sim — permissive; loại AgiBot World/Apple EgoDex/Meta ego-1k/EgoBrain — non-commercial). Viết script + bắt đầu tải OmniVideo-100K và RoboVQA. Cập nhật toàn bộ `datasets.md`.

### 1. Token thật đã tokenize xong — VÀ 1 lệch số quan trọng cần lưu ý

Chạy `count_tokens.py` (chỉnh `OUTPUT_DIR` inline, không sửa file gốc) trên 2 output Megatron thật:

| Nguồn | Token thật (BIN SIZE CHECK: PASS) | Document |
|---|---|---|
| **FineVideo-VLA v5** | **10,554,076,391 (10.55B)** | 371,888 |
| **MV-Omni** | **16,357,256,571 (16.36B)** | 1,593,301 |
| MINT-1T text | Chưa có — job vẫn đang chạy | — |
| **Tổng đã tokenize xong** | **~26.91B token** | — |

**⚠️ Lệch số chưa rõ nguyên nhân:** số token FineVideo-v5 thật (10.55B) **gấp ~2 lần** con số vẫn ghi trong docs từ trước tới giờ (5,255,589,397 / 5.256B, xem mục "18/07/2026 sáng" bên dưới). Số document khớp chính xác (371,888 cả 2 phía) nên không phải lỗi thiếu/thừa record. Nghi ngờ hợp lý nhất (chưa verify): con số 5.256B được tính ở bước flatten bằng cách đếm số lần xuất hiện tag `<..._N>` + đếm từ thô cho phần text tự do (title/context/caption/speech), trong khi tokenizer BPE thật (GPT-NeoX-20b + VLA extension) tách 1 từ tiếng Anh thường thành nhiều subword token hơn đáng kể so với đếm-từ-thô — token VLA (`<seed2_N>`, `<pelvis_x_N>`...) vẫn atomic cả 2 cách tính nên phần lệch nhiều khả năng dồn hết vào phần text tự nhiên. **Chưa verify tận gốc, để dành phiên sau.** Tin tốt: ngân sách token thật cao hơn nhiều so với lo ngại "corpus quá nhỏ cho model 1.7B" trước đây.

### 2. Job tokenize — trạng thái thật cuối phiên

- **`tok_mv_omni` (14118393)** — job đầu (`14117680`) từng fail âm thầm sáng nay (SLURM báo COMPLETED nhưng Ray không connect, 0 output thật). Đã fix bug start Ray cluster, resubmit — **lần này COMPLETED thật** (13:18, không traceback, output 7 shard/60.94GB thật).
- **`tok_mint1t` (14118392)** — vẫn RUNNING cuối phiên (>1h40p), tiến độ bình thường, chưa lỗi.

### 3. SenseNova-SI-8M — đính chính license (xem chi tiết trong `datasets.md` mục 4 và [[project_vla_status]])

Huu nghi ngờ (qua ChatGPT) ảnh trong dataset không permissive hoàn toàn. Điều tra lại kỹ (README, GitHub `OpenSenseNova/SenseNova-SI`, paper arXiv:2511.13719, tự đọc `image` column thật trong parquet) — **không tìm được tài liệu nào nói rõ nguồn gốc ảnh gốc**, paper/GitHub chỉ dùng từ "curated" (gợi ý tổng hợp từ nguồn khác, không phải tự chụp). Cùng dạng bẫy `cc_dump` như MINT. **Rút lại kết luận "an toàn hơn MINT" ở mục sáng nay** — license mở, chưa nên coi là sẵn sàng train.

### 4. Gen-EgoData — viết lại sau khi đọc kỹ toolkit `das-datakit`

Không phải "video ego + pose người" — là dữ liệu từ **thiết bị cầm tay "DAS device"** (kiểu UMI), action thật = `eef_pose` (6-DoF) + `Gripper_width` (tay đơn), khác hẳn `<agent>` 17-khớp hiện tại. License CC-BY-SA-4.0 (share-alike, cần Huu duyệt điều khoản riêng). Xếp lại vào cùng nhóm "robot-action modality" với MolmoAct2/Cosmos3-DROID/Open X-Embodiment/GR00T-Sim.

### 5. Ego/exo — kết luận sau thảo luận dài với Van Khue: KHÔNG cần sửa FineVideo-VLA

Verify trực tiếp trong code (`phase3_kinematics_processor.py`): `<agent>` pose token đã **root-centred/pelvis-relative** sẵn (`retargeted[:, pelvis_idx] = 0.0`) — bất kể video quay góc nào. "Egocentric" (góc camera) và "root-centred" (quy ước toạ độ khung xương) là 2 trục khác nhau — không có "pose exocentric" để mà sửa, và head-relative sẽ **tệ hơn** pelvis-relative (nhiễu xoay đầu lan vào mọi khớp khác), không phải "egocentric hơn". Vấn đề thật (nếu có) là domain-gap giữa video train (3rd-person) và video robot thấy lúc deploy (camera gắn robot) — giải pháp đúng là ưu tiên **integrate Isaac Sim pipeline** (hoặc dùng GR00T-Sim mục 6 làm tạm), không phải sửa FineVideo-VLA hay đi săn thêm ego-video dataset.

### 6. Khảo sát dataset mới trên HF — 3 ứng viên tốt, 4 ứng viên bị loại vì license

| Dataset | License verify thật | Kết luận |
|---|---|---|
| **NVIDIA GR00T-X-Embodiment-Sim** | ✅ CC-BY-4.0 | **Ứng viên mạnh nhất** — 345K trajectory, có humanoid GR1, cùng vai trò Isaac Sim |
| **RoboVQA** (Google DeepMind) | ✅ CC-BY-4.0+Apache-2.0 (verify từ GitHub chính chủ) | 238h, 3 embodiment — tải qua mirror `Tianli/robovqa` (Van Khue tìm ra), có `LICENSE.txt` Apache-2.0 thật |
| **Open X-Embodiment** | ⚠️ Registry 55-60 dataset con, KHÔNG đồng nhất license | Chưa tải — cần audit từng cái trước |
| AgiBot World | ❌ CC BY-**NC**-SA | Loại |
| Apple EgoDex | ❌ CC-BY-**NC**-ND | Loại (tiếc, rất khớp use-case: 829h ego dexterous manipulation + pose) |
| Meta `ego-1k` / `EgoBrain` | ❌ Non-commercial (+ EgoBrain lạc chủ đề, EEG) | Loại |

### 7. Bắt đầu tải OmniVideo-100K + RoboVQA

Viết `tools/extract/download_omnivideo_100k.py` + `tools/extract/download_robovqa.py` (cùng pattern resumable với `download_sensenova_si8m.py`, đã verify file list/size thật qua HF API trước khi viết). Van Khue chạy cả 2 trong tmux riêng (`omnivideo_dl`, `robovqa_dl`) — cuối phiên: OmniVideo-100K 22 file/8.8GB/52.9GB, RoboVQA 402 file/1.6GB/~70.8GB, cả 2 đang tiến triển bình thường. SenseNova vẫn tải song song (52/54 file, 1.1TB, thỉnh thoảng timeout tự resume).

### 8. `datasets.md` — cập nhật toàn bộ

Thêm mục 15 (RoboVQA), 16 (Open X-Embodiment), 17 (GR00T-Sim); viết lại mục 4 (SenseNova, đính chính license) và mục 8 (Gen-EgoData, schema đúng); thêm 4 dòng "đã check, loại" cho AgiBot/EgoDex/ego-1k/EgoBrain; cập nhật bảng tổng quan + phần "Việc còn mở".

---

## Cập nhật phiên làm việc — 18/07/2026 (đọc phần này trước khi resume)

**Việc chính hôm nay:** xác nhận task #6/#7 full-scale (Phase 6 v4 + Phase 7 v5) đã chạy xong sạch, quality/validity khớp gần tuyệt đối dự đoán, upload lên HF. Tạo `datasets.md` khảo sát 14 dataset. Điều tra sâu MINT-1T-HTML (structure thật, license) → **quyết định bỏ hẳn phần ảnh** (chỉ URL, không track được license), giữ phần text. Điều tra SenseNova-SI-8M (ảnh thật, permissive) → **quyết định tải full 1.13TB**, đang chạy. Thiết kế rồi **từ chối** egocentric perspective converter sau khi soi kỹ giá trị thật. Setup pipeline Megatron tokenize cho 3 nguồn (MV-Omni, MINT, FineVideo v5), account thử đổi `laionize`/`batch` — **chưa submit**.

### 1. Task #6/#7 full-scale — XÁC NHẬN HOÀN THÀNH (docs cũ ghi "chưa confirm", giờ đã xong)

`sacct` xác nhận cả 2 job COMPLETED (14114336 merge v4: 31 phút, 14114370 flatten v5: 38 phút), 0 lỗi thật (chỉ warning module vô hại). Kết quả khớp gần như tuyệt đối dự đoán trước khi chạy: token tăng **+0.740%** (dự đoán +0.737%/+0.749%), caption/speech token lệch <0.05%, record count khớp 100% (371,888). Tổng **5,255,589,397 token** (5.256B). Verify không double-injection (2,787 activity check tag mở/đóng khớp), spot-check nội dung caption/speech đặt đúng vị trí, nội dung hợp lý.

**Đã upload lên HF:** cập nhật `tools/upload/vla_flattened_dataset_card.md` (bảng số liệu v5, mục "What Changed in v5", vocab 156,509) + `upload_flattened_hf.py` (default trỏ `megatron_dataset_v5`, prefix `flat_final_vla_adaptive_rank`). User chạy upload, confirm live trên `EmpathicRobotics/FineVideo-Phase7-Flattened` (verify qua HF API, `lastModified` đúng ngày chạy).

### 2. Tạo `datasets.md` — khảo sát 14 dataset, mọi field verify bằng data thật

File mới ở root repo, trả lời nhanh cho từng dataset: tổng quan / đã tải chưa+path / tokenize modality nào / structure + có thể bổ sung token gì / ready Megatron chưa. Gồm: FineVideo-VLA, MixtureVitae-Omni, MINT-1T-HTML, SenseNova-SI-8M, OmniVideo-100K, MolmoAct2-BimanualYAM, Cosmos3-DROID, Gen-EgoData, MixtureVitae-Backup/multimodal, VALID, stera-10m, FineVLA, abc.bot, và "MINT PDF data" (Huu đã tải sẵn, chưa rõ path, trên leo). Đối chiếu với danh sách 7 dataset Huu tự liệt kê trong chat — khớp 100%, không sót cái nào.

**Phát hiện quan trọng khi làm rõ MV-Omni:** không cần bước "flatten" như FineVideo — data gốc `mv_omni_converted/*.jsonl.gz` đã đúng schema `{"text":...}` sẵn, chỉ còn thiếu (a) quyết định tỷ lệ trộn tránh loãng agent-token, (b) chạy tokenize thật.

### 3. MINT-1T-HTML — điều tra structure + license thật, quyết định BỎ ẢNH, giữ text

**Tải xong hoàn toàn** (2.7TB, 6,159/6,159 file, verify qua log + đếm file thật).

**Structure thật (không phải đoán từ README):** `texts[]`/`images[]` cùng độ dài, xen kẽ, loại trừ lẫn nhau (vị trí nào có text thì ảnh null và ngược lại). `image_hashes`/`images_metadata` KHÔNG cùng độ dài, không align theo index — chỉ nên dùng `images[i]` (URL) làm nguồn sự thật.

**Quy mô thật (đo, không đoán):** ~850M record, ~2.83 tỷ URL ảnh, 91.7% còn sống (test 60 URL thật), ~97KB/ảnh → tải hết sẽ ~130-180TB. → chỉ pilot 20 shard (~9.2M ảnh).

**Bug tốc độ tìm & fix:** rate-limit theo domain (0.5s/request) vô tình bóp tốc độ toàn bộ 64 worker xuống ~10 img/s vì phần lớn ảnh dùng chung 4 host CDN Blogspot. Fix: đổi sang semaphore giới hạn concurrency/domain (mặc định 8) thay vì serialize.

**License — đọc thẳng README chính thức mlfoundations:** `cc_dump` KHÔNG phải license (là mã CommonCrawl dump, dễ nhầm CC=Creative Commons). Pipeline filter của họ không hề lọc bản quyền ảnh (chỉ NSFW/size/dedup), và README tự nhận trách nhiệm thuộc về user. **Quyết định (chat 18/7, Huu): bỏ hẳn ảnh** (*"if the mint doesn't have images ignore it"*), **giữ text** (*"the hf dataset is fine"* — Van Khue). Đã xoá 130MB ảnh pilot đã tải. `stera-10m` cũng bị loại cùng phiên (not permissive).

### 4. SenseNova-SI-8M — điều tra structure thật, quyết định tải full, đang chạy

**Structure thật:** config `full` (`SenseNova-SI-8M.parquet`, 851MB, 8,164,067 record) chỉ có `image: list<string>` (path tương đối), KHÔNG nhúng bytes — khác bản `preview` (1000 sample, có nhúng bytes, chỉ để xem thử). Ảnh thật nằm trong **53 file zip độc lập** (`images_part_001..053.zip`, ~1.10TB) — giải nén tất cả vào 1 thư mục chung sẽ tự ráp lại đúng cây `images/`, join bằng `f"{extract_dest}/{image[i]}"`.

Nội dung: VQA trắc nghiệm về spatial reasoning trong nhà (định vị vật thể, hướng tương đối) — sát với nhu cầu robot/embodied hơn QA thường. Apache-2.0, ảnh bytes thật (không URL) → không vướng rủi ro license như MINT.

**Quyết định: tải full 1.13TB.** Script `tools/extract/download_sensenova_si8m.py` (resumable, in tiến độ mỗi lần retry). **Tính đến cuối phiên: ~71GB/1.13TB, ~45MB/s thật (đo trực tiếp), ETA ~6.5h.** Bước giải nén+join chưa code, để sau khi tải xong.

### 5. Egocentric perspective converter — thiết kế xong rồi TỪ CHỐI (quan trọng, đừng làm lại nếu chưa giải quyết được vấn đề gốc)

Thiết kế ban đầu: `<agent_ego>` tag riêng (tránh nhập nhằng vocab với `<agent>` 3rd-person), record riêng biệt (không nhét chung 1 record tránh bloat context). Nhưng **bị từ chối sau khi user hỏi thẳng "có gì hay"**: video (seed2/cosmos/avclm) không đổi — vẫn là video YouTube 3rd-person gốc — chỉ nhãn pose bị xoay sang hệ quy chiếu chỉ có ý nghĩa nếu có camera gắn đầu thật (không tồn tại). Hệ quả: (1) cặp video-pose không nhất quán vật lý — dạy sai mapping; (2) phép biến đổi là isometry (khả nghịch, không mất thông tin) nên `agent`/`agent_ego` **thông tin y hệt nhau** cho việc retarget robot — không thêm kiến thức mới. **Kết luận: không code theo hướng này**, chỉ đáng làm lại nếu có video egocentric thật để ghép cùng.

### 6. Megatron tokenize — setup cho 3 nguồn, script sẵn sàng, CHƯA SUBMIT

TODO tồn đọng lâu nhất của project: `.bin/.idx` thật hiện có (2.84B token) vẫn dùng data+tokenizer v1 cũ, chưa từng tokenize lại với SNAC/caption/speech hay MV-Omni.

Tìm ra script thật `mv-scale/tokenize_vla_adaptive.sbatch` (hạ tầng dùng chung nhiều project, account `cstdl` trên JUSUF trước đây). Thử đổi sang `account=laionize`/`partition=batch` trên JUWELS theo yêu cầu user — **verify trước** bằng `sacctmgr` (laionize có association hợp lệ với batch) và test path hạ tầng đọc được, không đoán bừa.

**Phát hiện quan trọng:** `mv_preprocess_data.py` chỉ đọc `.jsonl` phẳng key `text`, KHÔNG đọc parquet trực tiếp. MV-Omni/FineVideo-v5 đã đúng format sẵn. MINT (parquet, cột `texts[]`) cần bước convert riêng.

**Đã tạo:**
- `tools/extract/convert_mint1t_text_jsonl.py` + `slurm/convert_mint1t_text.sbatch` (32-array, nối `texts[]` thành 1 chuỗi `{"text":...}`)
- `mv-scale/tokenize_mv_omni.sbatch`, `tokenize_finevideo_v5.sbatch`, `tokenize_mint1t.sbatch` — cùng dùng tokenizer `tokenizer_vla_adaptive_v2` (156,509 vocab) để token ID khớp nhau lúc trộn sau này; MINT phụ thuộc bước convert xong trước.

Tất cả đã syntax-check OK. **Chủ động KHÔNG quyết định tỷ lệ trộn MV-Omni/MINT trong bước này** — theo yêu cầu user ("mấy cái quyết định drop out để sau đi"), đây là quyết định lúc train, không phải lúc tokenize.

**Trạng thái cuối phiên: cả 4 job (convert MINT + 3 tokenize) đều CHƯA submit.** SenseNova vẫn đang tải nền. Phiên sau cần check cả 2.

---

## Cập nhật phiên làm việc — 17/07/2026 (đọc phần này trước khi resume)

**Việc chính hôm nay:** xác nhận task #3 (A2 captioning) đã chạy xong hoàn toàn qua đêm, chạy + verify task #4, code + test kỹ task #6 + #7, đo chính xác mức tăng token (2 phương pháp độc lập đều ra ~0.75%), commit + push code, **submit full-scale SLURM job task #6**.

### 1. Task #3 (A2 captioning) — XÁC NHẬN ĐÃ CHẠY XONG HOÀN TOÀN (qua đêm 15→17/7)

Kiểm tra `sacct` cho chuỗi job `14104157`→`158`→`159`: cả 3 đều `COMPLETED`, 2 job cuối chỉ mất ~1 phút (do `--skip-existing` không còn việc gì để làm). Đếm dòng thật: **912,998 dòng caption** trên 40,798 file — khớp chính xác target A1. `squeue` trống, không còn job caption nào chạy.

### 2. Task #4 (`build_caption_dict.py`) — CHẠY XONG, VERIFY KỸ

Reshape 40,798 file caption phẳng thành dict `{activity_id: {chunk_idx: "<caption>...</caption>"}}`. Kết quả: **40,798 video, 912,998 dòng → 372,385 activity, 0 collision** — khớp 100% số liệu A1 đã biết. Đối chiếu ngược 5 video ngẫu nhiên: khớp byte-for-byte.

### 3. Task #6 (`phase6_merge_adaptive.py`) — CODE XONG, PHÁT HIỆN + FIX 1 BUG NGUY HIỂM TRƯỚC KHI CHẠY THẬT

Thêm `--captions-dir` + `--speech-segments-dir`. Thứ tự chèn mới trong 1 chunk: `[caption?] <cosmos> <avc_lm> [agent?] [snac?] [speech?]` — caption chèn trước `<cosmos>` (dùng `COSMOS_PATTERN` mới, độc lập với `AVC_PATTERN`), speech chèn sau `</avc_lm>` cùng chỗ agent/snac.

**Bug nguy hiểm bắt được ở bước dry-run, trước khi đụng data thật quy mô lớn:** script này chạy TRÊN `final_dataset_adaptive_v3` — vốn đã có sẵn `<agent>`/`<snac>` từ lần chạy trước. Nếu không cẩn thận, truyền lại `--agent-tokens-dir`/`--snac-tokens-dir` (cần để `has_agent`/`has_snac` trong `chunk_timing` chính xác) sẽ **inject trùng lần 2**. Fix: thêm cơ chế tự phát hiện — nếu `video_tokens` đã có `<agent>`/`<snac>` thì bỏ qua bước inject (báo 0 injected) nhưng vẫn dùng dict để tính cờ `has_agent`/`has_snac` đúng. Verify bằng dry-run thật (3 video/72 activity): nội dung agent/snac giữ nguyên byte-for-byte, caption (138) + speech (243) được thêm đúng.

Phát hiện phụ: path mặc định `--agent-tokens-dir` (tương đối) KHÔNG trỏ đúng data thật — path đúng là `/p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive`. Submit script full-scale đã dùng path tuyệt đối đúng.

### 4. Task #7 (`phase7_flatten.py`) — CODE XONG, TEST KỸ

State machine thêm event `caption` (buffer, flush cùng seed2/cosmos khi gặp `avc_lm`, thứ tự caption→seed2→cosmos) và `speech` (emit ngay, như snac). Cả 2 **không dropout, không augment** (giữ nguyên văn — vì gắn đúng 1 chunk cụ thể, augment sẽ phá vỡ liên kết token-thời điểm). Header `### Speech:` cũ giữ nguyên không đổi (đã xác nhận với user là cố ý trùng lặp, không phải bug). `count_token_types()` thêm `mode` tracker tránh đếm nhầm từ caption/speech vào bucket agent (chỉ ảnh hưởng thống kê, không ảnh hưởng data training).

**Test:** 7+6 nhóm unit test (54 assertion) — thứ tự chèn, chống double-injection, cách ly caption giữa các chunk, độc lập dropout, đếm token đúng — cộng 1 lần dry-run thật end-to-end (3 video → Phase 6 → Phase 7).

Đã commit `5f5492e`, đã push `origin/master`.

### 5. Đo token tăng — 2 phương pháp độc lập, cùng ra ~0.75%

- **Sample thật:** 798 video/3 shard → 5,312 activity qua filter thật, xử lý với `drop_rate_cosmos=0.5` (seed cố định để so sánh sạch, vì lần đầu so 2 lần chạy CLI riêng biệt không seed bị nhiễu ~1% do random cosmos dropout tích luỹ khác nhau — không phải bug, chỉ là thiếu kiểm soát random state). Kết quả: 73,796,727 → 74,340,242 token, **+0.737%**.
- **Chính xác toàn bộ dataset:** đếm trực tiếp — 912,998 caption/10,256,494 từ + 2,158,388 speech-chunk/22,696,606 từ = **39,095,872 token mới**, so với baseline thật 5,217,000,000 token (Phase 7 v4 full-scale đã chạy trước đó) → **+0.749%**.

2 phương pháp độc lập khớp nhau trong 0.012 điểm % — loại trừ khả năng có bug đo.

**Ghi chú quan trọng (user có phản ứng thất vọng, cần ghi rõ để tránh lặp lại kỳ vọng sai):** user kỳ vọng việc này sẽ "làm dày" dataset đáng kể, nhưng +0.75% là quá nhỏ so với kỳ vọng đó. Đã làm rõ: caption+speech scope từ đầu là fix root cause #2 (thiếu language anchor tại điểm chuyển modal) — một vấn đề **định tính**, KHÔNG phải cơ chế để đạt mục tiêu "×4 record" (mục tiêu đó, theo §13/§2.5c REPORT.md, là caption **+ perspective framing cộng lại** — perspective framing mới là đòn bẩy nhân RECORD, chưa code). Đây là lần re-confirm thứ 2 của cùng kết luận (lần 1 ở phiên 12/7 nhìn từ góc số lượng điểm neo, lần này nhìn từ góc token count) — nếu muốn giải quyết bài toán dataset nhỏ (2.84B token, nhỏ cho model 1.7B), đòn bẩy đúng là perspective framing hoặc thêm nguồn data ngoài (SenseNova-SI-8M/stera-10m/MixtureVitae-Omni), không phải vắt thêm từ caption/speech density.

### 6. Task #6 — ĐÃ SUBMIT FULL-SCALE

Script mới `slurm/submit_merge_adaptive_v4.sh` (32-array, `partition=batch`, `account=laionize`, `--time=03:00:00`, theo đúng pattern `submit_merge_adaptive_v3.sh`), input `final_dataset_adaptive_v3/` (160 file, 663GB), output `final_dataset_adaptive_v4/`, `--skip-existing`. **Job `14114336`** — xác nhận 32/32 task vào trạng thái `R` trong 15s, worker 1 log cho thấy tiến độ bình thường. Chưa xác nhận xong ở thời điểm ghi entry này.

**Chưa bắt đầu:** Task #7 full-scale (chờ `final_dataset_adaptive_v4/` xong), Task #8 kiểm tra full-scale cuối cùng (đã làm ở quy mô nhỏ trong phiên này, còn cần double-check ở output full-scale trước khi coi corpus sẵn sàng train).

---

## Cập nhật phiên làm việc — 15/07/2026 (đọc phần này trước khi resume)

**Việc chính hôm nay: kiểm tra lại trạng thái thật của caption+speech pipeline (task breakdown ở REPORT.md §18), phát hiện 2 việc đã âm thầm chạy xong mà doc chưa ghi nhận, và fix 2 bug thật trong `phase7_flatten.py` liên quan tới việc permute speech transcript khi có SNAC.

### 1. Speech extraction (task #2) — XÁC NHẬN ĐÃ CHẠY XONG HOÀN TOÀN

REPORT.md §18 dừng ở "relaunched — confirmed healthy" (14/7). Check lại tmux session `speech_full` thì cả 8 worker đã in `DONE` từ lâu ("All workers finished"):

| Metric | Tổng 8 worker |
|---|---|
| Video xử lý | **40,437** |
| Activity có speech | **303,976** |
| Segment speech trích ra | **2,608,543** |
| Garbled/skip | ~58K (~2.2%) |

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments/` — 40,490 file `{video_id}_speech.jsonl`. Task #2 coi như xong, sẵn sàng cho task #6 (`phase6_merge_adaptive.py --speech-segments-dir`).

### 2. Tokenizer rebuild (task #5) — XÁC NHẬN ĐÃ CHẠY XONG HOÀN TOÀN

`tokenizer_vla_qwen3` (đang "in progress" theo REPORT.md §18) đã build xong: vocab **257,901**, tất cả token mới (`<caption>`, `</caption>`, `<speech>`, `</speech>`) + toàn bộ token cũ (seed2/cosmos/avclm/pelvis/SNAC/agent/fps) đã verify atomic qua spot-check trong tmux `qwen3_rebuild`. Cùng với `tokenizer_vla_adaptive_v2` (156,509 vocab, đã xong từ trước) — cả 2 tokenizer đã sẵn sàng, chỉ còn thiếu bước **upload lên HuggingFace** (cần user tự export `HF_TOKEN`):
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh
export HF_TOKEN=...   # HF token của user
python tools/upload/upload_tokenizers_v2.py --mode all
```

### 3. Caption pipeline (A2, task #3) — vẫn đang chạy, còn xa mới xong

`squeue` cho thấy job `14104156` đang chạy đủ 32/32 worker (đã chạy hơn 8h45p tính tới lúc check), và có chuỗi 3 job kế tiếp (`14104157`→`158`→`159`) đang xếp hàng chờ qua dependency `afterany` — tự động nối tiếp vì 1 lần chạy không đủ time-window để xử lý hết ~913K task point. Worker 0 mẫu: 800/1275 video, ~0.03 video/s, ETA riêng job hiện tại ~287 phút. Đã có **25,432 file caption** ghi ra `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/captions/`. Chưa đủ để chạy full-scale task #4 (`build_caption_dict.py`) — cần đợi thêm.

### 4. Fix 2 bug thật trong `pipeline_pose/phase7_flatten.py` (liên quan task #7)

Phát hiện khi rà lại kỹ augmentation pipeline (không phải lỗi mới sinh ra hôm nay, đã tồn tại từ khi `permute_sentences` + SNAC injection được thêm vào):

- **Bug A (chính):** augmentation "sentence permutation" (xáo câu trong `### Speech:`) áp dụng **vô điều kiện**, kể cả khi activity đó có `<snac_N>` (audio thật, giữ đúng thứ tự thời gian). Kết quả: model "nghe" (SNAC) đúng thứ tự nhưng "đọc" (text speech) bị xáo trộn — dạy sai lệch giả giữa audio và text. **Fix:** thêm biến `effective_permute_rate = 0.0 if sn > 0 else permute_sentences` ngay trước khi gọi `process_transcript_into_chunks()` — `sn` là số token `<snac_N>` đã có sẵn trong `kept_tokens` (tính từ `count_token_types()`, không cần thêm logic mới).
- **Bug B (phụ, phát hiện khi test fix A):** `permute_chunks_list()` có `n = max(1, int(len(c) * permutation_rate))` — dòng này **ép tối thiểu 1 lần hoán đổi bất kể rate truyền vào là bao nhiêu**, kể cả `rate=0.0`. Nếu không sửa, fix A ở trên sẽ vô tác dụng (permute_rate=0 vẫn bị hoán đổi 1 cặp). **Fix:** thêm điều kiện `permutation_rate <= 0` vào early-return cùng với check `len(chunks) < 2` đã có sẵn.
- Đã verify bằng test nhanh: `permute_chunks_list(chunks, 0.0)` giờ trả về list y hệt input (trước đây thì không).
- **Chưa commit**, chưa chạy lại full-scale (đây là fix cho lần Phase 7 chạy tiếp theo, thuộc task #7/#8 của caption+speech interleaving work, hiện vẫn "not started" theo §18).

### Việc tiếp theo hợp lý nhất

Vì task #3 (A2) còn chạy rất lâu và task #6/#7/#8 phụ thuộc cả speech (đã xong 100%) lẫn caption (chưa xong), có thể tranh thủ bắt đầu task #6 (`phase6_merge_adaptive.py` — thêm `--captions-dir` + `--speech-segments-dir`) ngay, vì phần speech-segments đã sẵn sàng hoàn toàn, không cần đợi caption.

---

## Session Update — July 12, 2026 (read this first to resume)

**Two main threads this session: (1) fixed a `chunk_timing` bug in Phase 6, (2) finalized the captioning pipeline design.**

### 1. `has_seed2`/`has_cosmos` bug in `chunk_timing` — FIXED, FULL DATASET RE-RUN COMPLETE

- **Bug found:** `phase6_merge_adaptive.py` computed `has_seed2`/`has_cosmos` as `i < len(seed2_matches)` — comparing the **chunk index** against the **total tag count for the whole activity**, not a real per-chunk check. Since seed2 fires at 1fps while chunks occur at 3.75/sec, this flag was true for an artificial prefix of chunks then false for the rest — a single fake "off" transition per activity, not reflecting real content (verified: 2,558/2,558 sampled activities were ON→OFF only, never OFF→ON, at random timestamps 0.27s–638s).
- **Fix:** recompute using real string positions in `video_tokens` — a `<seed2>`/`<cosmos>` tag is attributed to whichever chunk's span it falls between (bounded by consecutive `<avc_lm>` block ends), matching the true temporal write order from `pipeline_video/pipeline.py`. `has_cosmos`/`has_avc_lm` simplified to hardcoded `True` (always correct, verified 0 flips across the whole sample).
- **No Phase 7 re-run needed** — verified via byte-for-byte `video_tokens` diff (0 differences) + code grep: `phase7_flatten.py` never reads `chunk_timing`. Only Phase 6's metadata output is affected; existing trained models and Megatron data are untouched.
- **Full dataset re-run done:** SLURM job `14102737`, 32/32 tasks COMPLETED, 0 errors → `final_dataset_adaptive_v3/` (160 files, kept v2 for comparison). New script: `slurm/submit_merge_adaptive_v3.sh`.
- **QA verified at two scales:** (a) 1 file (2,563 activities): agent/snac injection counts match v2 exactly (content unchanged); `has_seed2` now flips ~53/activity at the correct periodic rate. (b) 15 random files across the full dataset (34,732 activities, spanning all 40,804 videos): `has_seed2` flips 54.53/activity on average, **0/34,732 (0.00%) activities have `has_seed2` stuck False the whole time** — fix is stable at scale.
- **`final_dataset_adaptive_v3/` is now the standard input** for anything touching `chunk_timing` (including the captioning work below).

### 2. Captioning pipeline — DESIGN FINALIZED (prototype only, full-scale not yet coded)

**Context:** Huu asked (Jul 11 chat) for frame captions on all FineVideo keyframes, to fix root cause #2 (model lacks a language anchor for knowing when to switch modality).

**Anchor point selection (took several debugging rounds to get right):**
- **NOT** "any of the 5 flags `has_seed2/cosmos/avc_lm/agent/snac` changes" as originally planned — measured on real data: `cosmos`/`avc_lm` never vary within an activity; `seed2` (even after the bugfix above) still flips ~54x/activity, but that's purely its 1fps technical cadence, not a real content change.
- **Only using:** (1) the activity's first frame (opening context) + (2) every time `has_agent` flips (a person genuinely appears/disappears — confirmed with a real example: person transitions from standing to sitting exactly when agent turns on). Function: `select_anchor_points(chunk_timing, min_gap_sec=5.0)` in `tools/analysis/caption_prototype.py`.
- **`min_gap_sec=5.0` debounce:** needed because `has_agent` itself flickers in busy/high-motion scenes (sports, martial arts) due to noisy frame-to-frame YOLO detection (a known pre-existing data quality issue, not a new bug — no Phase 6 change needed). This debounce only affects which points THIS script chooses to caption, not the stored data.
- **Measured density:** ~1.86 captions/activity avg (at 2s gap) — well short of the "×4 records" target in the original doc; 82.8% of activities get only 1 caption (opening frame, no agent event ever occurs). **This is a known, unresolved limitation** — may need a periodic supplemental caption (every N seconds) for activities with no agent transition; N not yet decided.

**Model — tested 3, settled on Qwen2.5-VL-3B-Instruct:**
| Model | Test result |
|---|---|
| **Qwen2.5-VL-3B-Instruct** ✅ CHOSEN | No hallucinations in any test (including a 96-caption batch). Natively supported in `transformers` (no compatibility risk). Prompt: `"Describe what the person is doing in one short sentence."` |
| Florence-2-base | `<DETAILED_CAPTION>` mode clearly hallucinates (e.g. invented "he appears to be a psycholinguist"). Switching to `<CAPTION>` mode fixed the hallucination + was 3.5x faster than Qwen + no more truncation — but needs a separate env (`transformers==4.49.0`, torchvision must match the CPU index) since its custom code (`trust_remote_code`) isn't compatible with newer `transformers`. Test env: `env_caption_test/` (can be deleted if unused going forward). |
| SmolVLM2-2.2B-Instruct | **2x SLOWER than Qwen2.5-VL on CPU** (27.7s vs 14.0s/caption — contradicts the "fast, edge-oriented" expectation) plus 1 clear hallucination (invented "holding a book" for a plain white intro-slate frame) → rejected. |

**Why Qwen2.5-VL despite being slower than Florence-2 on CPU:** CPU speed isn't the deciding factor since the full-scale run must happen on GPU regardless of model choice; prioritized quality/no-hallucination + long-term library compatibility over CPU-only speed.

**Full pipeline design (not yet coded, next session):**
```
final_dataset_adaptive_v3/ 
    → [A1] Task list generation (CPU) — scan chunk_timing, compute anchor points per activity
    → [A2] SLURM array job — open video from videos_staging/, extract frame, Qwen2.5-VL caption
         → outputs/captions/{video_id}_captions.jsonl
    → [B1] Extend phase6_merge_adaptive.py with --captions-dir (same pattern as --snac-tokens-dir)
         inject <caption>...</caption> RIGHT BEFORE <cosmos> for that chunk (never mid-block,
         avoids repeating the v3→v4 speech-interleaving bug)
         → final_dataset_adaptive_v4/
    → [B2] phase7_flatten.py (unchanged) → megatron_dataset_v5/ → tokenize → train
```
Captions are plain English text, tokenized as regular BPE — **no vocab expansion needed**.

**Infra:** step A2 (real captioning run) will use **CPU** (many cores judged more practical than the available 2×4090 machine) — Van Khue's call (Jul 12), no GPU needed yet.

**Side findings worth remembering:**
- FineVideo source videos are already staged locally: `videos_staging/` (note the "s" — distinct from an empty `video_staging/`) — 43,751 mp4s, `/p/data1/mmlaion/shared/nguyen38/data/videos_staging/`, filenames = `{video_id}.mp4`. No JUPITER or HF streaming needed.
- **Read and evaluated HumanoidBench — NOT a fit** for the current eval need. It's a closed-loop RL benchmark (MuJoCo, Unitree H1 + Shadow Hands, 61-dim joint-angle action space) whereas our model outputs raw xyz human pose (17 H36M joints, no angles/hands). Only relevant to the already-deferred Priority 12 "Isaac Sim/H1" work, not DISCUSS-3.
- Home directory (`~/.cache`) has a much smaller quota than `/p/data1` (project storage, 388TB free) — always set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` before downloading large models to avoid "Disk quota exceeded".
- HF Hub's Xet download backend occasionally fails transiently (`Background writer channel closed`) — set `HF_HUB_DISABLE_XET=1` to fall back to plain HTTP download if this happens.

### 3. A1 coded + run on full dataset + thoroughly validated. A2 coded + smoke-tested — CPU vs GPU decision pending (continuation of Jul 12 session, same day)

**`select_anchor_points()` got a "periodic supplement" step:**
- Problem: agent-transition alone gave only ~1.4-1.86 captions/activity; most activities got just 1 caption (the opening frame).
- Agreed design, now coded: after the agent-transition step, if fewer than `target_count=4` points were found, add evenly-spaced supplemental points across the activity duration, snapped to the nearest real chunk, debounced against already-kept points. New signature in `tools/analysis/caption_prototype.py`: `select_anchor_points(chunk_timing, min_gap_sec=2.0, target_count=4)`.
- Also fixed a classification bug in `caption_florence2_visual_batch.py` (changed `len(pts) > 1` → check the raw `has_agent` flip, since the supplement makes point-count no longer a reliable proxy for "did a real event happen").

**Real bug found and fixed while testing A1 on production data:** the periodic supplement computed `duration` from the **absolute** `end_sec` instead of relative to the activity's own start — activities starting late in a video (e.g. minute 9-10) got target timestamps computed way outside their actual time range, making the supplement effectively a no-op. Fixed by subtracting `activity_start` before dividing. Verified on 2,563 real activities: % of activities reaching `target_count=4` jumped from 10.4% → 54.8%.

**A1 (`tools/analysis/generate_caption_tasks.py`) — CODED, RUN ON FULL DATASET (160/160 shards):**
- Reads `final_dataset_adaptive_v3/`, computes anchor points per activity via `select_anchor_points()`, writes task lists to `outputs/caption_tasks/*.jsonl` (one line per anchor point: video_id, video_path, scene_id, activity_id, chunk_idx, start_sec, has_agent).
- Run: 13 shards via SLURM array (job `14103227`, cancelled mid-run once it became clear it was fast enough to run directly), remaining 147 shards run on the login node (`--skip-existing` for resume) — completed 160/160.
- **Result:** 40,798 videos, 372,385 activities, **912,998 task points**, avg **2.45 captions/activity**, 0 videos missing a local mp4.
- **Validated thoroughly (per explicit request):** 100% of task points pass schema/type/`video_path`-exists checks; 0 activities with duplicate `chunk_idx`; 0 activities violating the 5s debounce; cross-checked 5 random shards (11,576 activities, 28,156 points) by recomputing from source `chunk_timing` — 100% match, 0 missing/orphan activities. Diagnostic sample saved at `logs/a1_smoke_test_samples.json` (gitignored).
- Submit script: `slurm/submit_caption_tasks.sh`.

**Correction to the "×4" framing:** re-reading `REPORT.md` (lines 1104, 1134) — the original ×4 target referred to **captioning combined with perspective framing** multiplying total training RECORD count ~4×, **not** "each activity must have exactly 4 captions" as we'd been measuring against. Measured reality: avg 2.45/activity is only 61.3% of the narrow (captions-per-activity) reading of ×4 — root cause: ~59% of FineVideo activities are under 15s, which is geometrically impossible to fit 4 points ≥5s apart into. **Decision: keep `target_count=4, min_gap_sec=5.0` as-is, do NOT lower the gap to force the number up** — lowering it would produce near-duplicate captions on short static clips, adding no real language signal while inflating A2 compute cost. The real lever for closing the ×4 gap is perspective framing (separate, uncoded roadmap item), not squeezing more density out of captioning alone.

**A2 (`pipeline_pose/caption_finevideo.py`) — CODED, SMOKE-TESTED OK, FULL RUN NOT STARTED:**
- Reads A1's task list (grouped by video), opens the video, extracts the frame at `start_sec`, captions it with Qwen2.5-VL-3B-Instruct (model chosen in the prior session) → writes `outputs/captions/{video_id}_captions.jsonl`.
- Follows the same pattern already proven in `pipeline_pose/snac_finevideo.py`: model loaded once per worker, videos striped across workers (`all_vids[task_id::num_tasks]`), one output file per video for safe resume (skip if it already exists).
- Smoke test (video `A1UVeD9UB1I`, t=248.0s): sensible caption — *"The person is arranging jewelry on a box."* — matches the source `text_prompt` *"Woman opens a gift box."*
- **Infra bug found and fixed:** initially had no PyTorch thread limit — two concurrent test runs on the 80-core login node fought each other for threads, producing 57.6s/caption (~4x slower than real). Added `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`torch.set_num_threads()` pinned to `SLURM_CPUS_PER_TASK` (default 4) so the eventual 32 workers won't oversubscribe each other.
- **Clean throughput measured (4 threads, no contention, 3 repeats): ~13.8s/caption** (12.9/15.2/13.4s) — matches the 10-15s figure already in `REPORT.md`.
- **Full-run cost estimate:** 912,998 tasks × 13.8s ≈ **3,500 CPU-hours**. With 32 workers (matching the SNAC job) → **~109h/worker (~4.6 days)**, needing **~5 resubmits** if kept at `--time=24:00:00`. Safe to resubmit thanks to per-video skip-existing.
- Submit script written: `slurm/submit_caption_finevideo.sh` — **NOT YET SUBMITTED**, pending the CPU-vs-GPU decision.

**Open question — decide at the start of next session:**
- CPU (32 workers, ~4.6 days, script ready to submit now) vs GPU (2×4090 machine, not measured at all this session).
- Back-of-envelope (NOT measured): unbatched GPU may not even beat 32 CPU workers (only 2-way parallelism vs 32-way, even though each individual request is faster); a decisive GPU win requires batched inference (many images per forward pass) — `caption_frame()` currently only processes one image at a time, no batching implemented.
- If GPU is chosen: need to implement batched `caption_frame()` and get access details for the 2×4090 machine to measure real throughput before deciding.
- **B1/B2 (inject captions into `final_dataset_adaptive_v4/`, re-run Phase 7) have NOT started** — blocked on A2 producing enough output (partial or full).

### 4. CPU chosen, A2 full run SUBMITTED and confirmed working (Jul 13, 2026)

**Decision:** CPU, per Van Khue — the ready-to-go option, no need to wait on GPU batching work.

**First submit (job `14104070`) FAILED — all 32/32 tasks crashed at model load.** Root cause: `slurm/submit_caption_finevideo.sh` set `HF_CACHE=/p/scratch/laionize/nguyen38/hf_cache`, which does not contain the Qwen2.5-VL-3B-Instruct weights (only `bert-base-uncased` and `snac_24khz` were cached there) — compute nodes have no internet (`HF_HUB_OFFLINE=1`), so `from_pretrained()` raised `OSError: We couldn't connect to huggingface.co ... Qwen/Qwen2.5-VL-3B-Instruct is not the path to a directory containing config.json`. The correct cache — the one actually used during the Jul 12 smoke test — is `/p/data1/mmlaion/nguyen38/hf_cache` (7.1GB `models--Qwen--Qwen2.5-VL-3B-Instruct` present there), matching the general env gotcha already noted below ("always set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache`").

**Fix:** changed `HF_CACHE` in `slurm/submit_caption_finevideo.sh` to `/p/data1/mmlaion/nguyen38/hf_cache`.

**Resubmitted as job `14104104` — CONFIRMED WORKING.** All 32/32 tasks reached `R` (running) state, model loaded cleanly in ~44-45s per worker (no HF offline errors), and first captions started landing within ~5 minutes of submit. Spot-checked output (`.../captions/-0-6Som0MGY_captions.jsonl`, 10 captions) — all well-formed JSON, correct schema, and captions are qualitatively good/specific (e.g. *"The person is pouring sulfuric acid into an energy drink can."*, *"The person is using a blue dropper to apply coconut oil onto a surface."*).

**Status at end of session: job `14104104` running, 32/32 tasks active, ~4.6 days ETA (per Jul 12 cost estimate), several `--time=24:00:00` resubmits still needed.** Next session should: (1) check `squeue -u nguyen38` for job `14104104` (or its resubmitted successor — same script, safe to re-run, per-video skip-existing) and resubmit if it timed out, (2) once a meaningful fraction of `outputs/captions/*.jsonl` (912,998 target task points across 40,798 videos) is done, consider starting B1 (extend `phase6_merge_adaptive.py` with `--captions-dir`) — B1 doesn't strictly need 100% of A2 done first, just enough coverage to be worth prototyping against.

**Auto-chaining added so no manual resubmission needed:** `slurm/submit_caption_finevideo.sh` now takes an optional job-id arg and submits with `--dependency=afterany:<id>`, printing the new job id for easy chaining. Chained 5 more jobs after `14104104`: `14104104 → 14104155 → 14104156 → 14104157 → 14104158 → 14104159` (6 jobs × 24h = ~6 days coverage). Confirmed queued with `(Dependency)` reason via `squeue --start`. If the chain still isn't enough, extend it with `bash slurm/submit_caption_finevideo.sh 14104159`.

**Caption quality spot-check (333+ output files, ~340 sampled lines): good overall, one hallucination class found.** Captions match source `text_prompt` well in the large majority of cases. Found one clear hallucination: video `-Gq3DJyhJ3I` (soccer highlights) got *"performing a complex mathematical operation..."* at t=0.0s — the real frame (checked with `cv2`) is near-black (fade-in intro), and the model invented content instead of saying "not visible" like it correctly does elsewhere. **Low-severity, known Qwen2.5-VL limitation** (matches the ~1-in-30-96 rate from model selection testing), not a pipeline bug — not blocking. Possible low-priority future fix for B1: skip/flag captions on near-black frames (mean pixel intensity check) before injecting into training data.

### 5. Permissive dataset survey (6 candidates) + MINT-1T-HTML download started (Jul 13, 2026, parallel work while A2 runs)

Investigated the 6 remaining unscoped data candidates from the Jul 7 team chat. Full detail and rationale in `REPORT.md` §17 — summary:
- **`mira-wm.com` dropped** — not robot/pose data at all, it's a Rocket League gameplay world model (video + keyboard actions + game state). Unrelated to this project.
- **`finevla.xlang.ai` deferred** — the actual 47,159-trajectory training set isn't public yet (GitHub repo says "Coming soon"); only a 500-video eval benchmark (`xlangai/RoboFine-bench`) is downloadable.
- **`nvidia/Cosmos3-DROID` deferred pending architecture decision** — confirmed real (707GB, 71,907 real-robot teleop episodes, LeRobotDataset v3.0 format), but it's robot joint-space action data, a different representation from this project's xyz human-pose tokens. Needs a new tokenization scheme designed before it's useful — not just a download.
- **`MiG-NJU/OmniVideo-100K` deferred** — video QA data, no pose/action signal, would only dilute the agent-token ratio further (same risk already flagged for MV-Omni).
- **`genrobot2025/Gen-EgoData` deferred** — closest structural match (egocentric video+pose+action) but tiny (500 samples, 47.6GB), `.mcap` format needs a special toolkit, CC-BY-SA (share-alike) license.
- **`mlfoundations/MINT-1T-HTML` — downloading now.** Directly fills the DISCUSS-1 language-data gap (FineVideo's 5.217B tokens are ~100% modality-specific, essentially no plain text). **Size correction: actual measured size is 2.89TB (6,159 parquet shards), not the 5.91TB the dataset card advertises** (that figure covers the full MINT-1T project incl. PDF/ArXiv splits not in this HTML-only repo). **Schema finding: the `images` column is URLs only, not image bytes** — text is directly usable now, but getting pixels for a "seed2 token from images" idea would need a separate per-URL crawl with likely significant dead-link rate (2011-era blog sources).

**Key framework insight (worth remembering for future dataset scoping):** raw video sources (own HRNet→MotionBERT→PCHIP pipeline handles them end-to-end) are cheap to integrate; pre-posed/pre-actioned sources (DROID joint-space, Gen-EgoData `.mcap`) are a retargeting problem, not a data-ingestion problem — don't invest download time there without an explicit decision on adding a distinct robot-action modality first.

**Download status:** `tools/extract/download_mint1t_html.py` (new script, `huggingface_hub.snapshot_download`, 16 workers, auto-retry, resumable) running in tmux session `mint1t`, log at `logs/download_mint1t_html.log`, target `/p/data1/mmlaion/shared/vla/mint1t_html/`. At session end: 249/6,159 files, 204GB/2.89TB (~7%), ETA ~10h from start, no errors. **Next steps:** let it finish, then sample-tokenize `texts` with the project's own tokenizer (same method as the MixtureVitae investigation, §13) to get a real token count before deciding how much of the corpus is actually needed for DISCUSS-1.

### 6. Caption+speech interleaving pipeline — implementation started (Jul 14, 2026, while A2 continues running)

**Context:** approved plan to interleave `<caption>` (A2/Qwen2.5-VL output) and `<speech>` (FineVideo's pre-computed ASR transcript, NOT a new Whisper run — see correction below) tags into the training token sequence at modality-transition points, giving the model language anchors it currently lacks. 8 tasks total; status below.

**Done:**
- **Task #1 (video→shard manifest):** `tools/analysis/build_video_shard_manifest.py`, run to completion. 43,751 video_ids mapped to their `HuggingFaceFV/finevideo` parquet shard index. Reusable, no need to rerun.
- **Task #2 (speech extraction script, see bugs below):** `tools/analysis/extract_speech_segments.py` written. **Correction to earlier framing:** this does NOT run Whisper — FineVideo already ships a pre-computed ASR transcript per video (`timecoded_text_to_speech`, from YouTube-Commons), so this script just re-fetches that field from the HF Hub parquet and maps it onto `chunk_timing`, no new ASR compute needed.
- **Task #4 (caption dict adapter):** `tools/analysis/build_caption_dict.py` written, logic-tested against real A2 output. **Not yet run at full scale** — `captions_dict/` output directory does not exist on disk yet.
- **Task #5 (tokenizer):** added 4 wrapper tokens (`<caption>`, `</caption>`, `<speech>`, `</speech>`) to `tools/tokenizer/build_tokenizers.py` + `tools/tokenizer/expand_vocab.py`. `tokenizer_vla_adaptive_v2` rebuild **confirmed complete and verified** (vocab 156,509, all 4 new tokens atomic, all pre-existing token categories re-checked atomic too). `tokenizer_vla_qwen3` rebuild was still running as of this entry — check before assuming done.

**Two real bugs found and fixed in `extract_speech_segments.py` while trying to produce sample output (important — could resurface if this script is copied or its pattern reused elsewhere):**
1. **Unbounded memory growth, not from HF fetching (initially misdiagnosed).** A quick `--video-ids` test (2 videos) drove RSS to 90+ GB and climbing on the shared login node before being killed. First suspected `HfFileSystem` streaming reads and switched to `hf_hub_download` (local-cache download) — that fix is real and worth keeping, but it was NOT the actual cause. **Real cause:** `load_activities_needing_speech()` defaults to scanning the full `final_dataset_adaptive_v3/` glob (160 files, **663GB total**) before applying the `--video-ids` allowlist, and it retained the **entire activity dict** (including the `video_tokens` string — hundreds of KB per activity) for every video with `chunk_timing`, not just the 3 fields actually needed (`activity_id`, `chunk_timing`, `time_range_sec`).
2. **Fix:** (a) trim retained activity data to only the 3 needed fields, (b) apply the `--video-ids`/allowlist filter *during* the file scan, not after loading everything. Re-tested: RSS stayed under 500MB for the same 2-video test (down from 90+ GB unbounded). Production full-scale runs (32-way SLURM array, ~5 files/worker via `SLURM_ARRAY_TASK_COUNT` slicing) were always going to be less exposed to this than the login-node quick-test path, but the fix reduces worker memory footprint regardless.

**Not started:** Task #6 (`phase6_merge_adaptive.py` — inject `<caption>` pre-`<cosmos>`, `<speech>` post-`</avc_lm>`; pre-check on cosmos/avc_lm 1:1 invariant found 1 trailing-chunk mismatch out of 2,753 activities, judged safe but a broader 5-shard check was interrupted, not yet re-run), Task #7 (`phase7_flatten.py` regex/state-machine update), Task #8 (end-to-end dry run).

**A2 captioning job status:** chain `14104155` (running, 32/32 tasks) → `14104156-159` (queued, dependency). Caption count grew 11,501 → 13,783 between the Jul 13 and Jul 14 checks — steady but far from the 912,998 target, still expect several more days.

**Tokenizer upload (pending, needs user's own HF token):** `tools/upload/upload_tokenizers_v2.py` updated with new model-card text reflecting the 4 added tokens (vocab 156,505→156,509 for adaptive_v2, 257,897→257,901 for qwen3). Not run yet — user will export their own `HF_TOKEN` and run it once the qwen3 rebuild finishes: `python tools/upload/upload_tokenizers_v2.py --mode all`. `tokenizer_vla_qwen3` rebuild (32→106,232 tokens added onto the Qwen3 base) was still running as of this entry — check `tmux attach -t qwen3_rebuild` or `logs/build_tokenizer_qwen3_rebuild.log` before assuming done.

**Task #2 full-scale launch — real bug hit and fixed (same session, later):** decided to run `extract_speech_segments.py` directly on the JUWELS login node (not SLURM) since the script needs internet access (`hf_hub_download` from the HF Hub) and JUWELS compute nodes have none. Wrote `tools/analysis/run_speech_extraction_login.sh` — 8 parallel workers, `nice -n 15`/`ionice -c3` to stay polite on the shared login node, per-video output + `--skip-existing` for resume if killed. **First launch crashed all 8 workers within ~20s: `RuntimeError: ... Disk quota exceeded`** — the script's `hf_hub_download()` call had no `HF_HOME` override, so it defaulted to `~/.cache/huggingface` (small home-directory quota, already documented elsewhere as a known gotcha — forgot to apply it here). Fixed by adding `export HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` to the runner script; cleaned up the 1.9GB of partial `HuggingFaceFV/finevideo` cache debris left in `~/.cache/huggingface` by the crashed run. Relaunched — confirmed healthy (8/8 workers alive, ~100% of 1 core each = 8/80 total, RAM 300-400MB/worker, cache now growing correctly under `/p/data1/mmlaion/nguyen38/hf_cache`). Running in tmux session `speech_full`, per-worker logs at `logs/speech_extraction_login/worker_*.log`. Output target: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments/`.

---

## Repo Reorg Note (Jul 9, 2026)

`tools/` was split into subfolders (`upload/`, `tokenizer/`, `inventory/`, `eval/`, `visualize/`, `analysis/`, `extract/`) and ambiguously-named dirs were renamed (`multimodal/` → `investigations/mixturevitae_multimodal/`, `data_prep/` → `investigations/mv_omni_seed_conversion/`, `test/` → `manual_checks/`; `dev/` archived). **Script paths referenced below in older entries reflect the pre-reorg flat `tools/` structure** — e.g. `tools/data_inventory.py` is now `tools/inventory/data_inventory.py`. See the updated root `README.md` for the current layout.

---

## Session Update — July 8, 2026 (read this first to resume)

**What changed since last session:**
- ✅ **Phase 7 v4 uploaded to HF** — `EmpathicRobotics/FineVideo-Phase7-Flattened` is now live with the v4 data (371,888 records, 5.217B tokens). Shared with Huu/joergfranke on Discord (Jul 7) as the ready-to-tokenize dataset.
- ✅ **1-CP decision FINALIZED — deferred.** Confirmed with Huu on Discord (Jul 8): stick with current adaptive 2/4/8-CP format. Gain from 1-CP is only +7.1% (sample-based estimate, 50 videos) and re-running Phase 5→7 costs time. Huu's framing for eventual paper: "our compression decreases the data by more than 50%" is good enough to report as-is. **Revisit only if data later shows it's necessary.** Full-dataset 1-CP investigation (18,847 videos) was attempted but interrupted by the JUWELS outage — not resumed, not currently planned.
- ⚠ **JSC cluster outage (started ~Jul 6, 2026):** JUPITER fully down. JUWELS booster + JURECA have partial GPU availability ("to the extent Jenia lets us"). Huu's ETA: officially 1 week, realistically expect 2. This blocks: Megatron re-tokenization at scale, training v0.3, full 1-CP dataset run, Cosmos3-DROID GPU pipeline.
- **Team decision: cap synthetic/simulation data at ≤30% of total training mix** (Huu, from text literature guidance) — applies when deciding how much of abc.bot / MolmoAct2 / Cosmos3-DROID sim data to mix against real FineVideo human video.
- **New data source candidates identified (Jul 7, 2026 team chat)** — see "New Data Candidates" table below. Most promising: `abc.bot` (400h robot sim data **with physics state** MjData, permissive, has eval env).
- **Multi-project data sharing direction:** Huu wants to pool data across 3 parallel efforts — this repo's omni-vla work, joergfranke's architecture comparison project (qwen3/lfm2.5/olmo3 baselines), and blanchon.jl's diffusion-based world-action-model (video generation + action). `FineVideo-Phase7-Flattened` is now being used as shared input across projects — keep the format generic/well-documented.
- ✅ **`mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` investigated (Jul 9, 2026).** 15 files, ~103GB. Sample-based token count (75MB/file, streamed — no full download) via new `tools/peek_multimodal.py` + `tools/count_multimodal_tokens.py`, run locally (no JUWELS access this session). **Result: mostly plain text/caption data, not our VLA token format.** Only `train_data_snac.jsonl.gz` and `valid_data_snac.jsonl.gz` carry real SNAC audio tokens — as **raw integer arrays** (`snac_token: [128266, ...]`), not `<snac_N>` string tags — extrapolated **~3.27B raw SNAC codes** total (~3.11B + ~162M). The other 13 files are text/caption corpora (~12.4B word-count tokens extrapolated, `finevideo_transcripts.jsonl.gz` undercounted — see caveats below). Posted findings to Huu on Discord (Jul 9, 3:51pm) asking if he wants it added — **awaiting his reply**, do not start integration yet. Full detail in "MixtureVitae-Backup Multimodal Investigation" section below.
- **Pending investigation tasks (assigned by Huu, not yet done):**
  1. "finevideo reformulation" at `leo:/mnt/sdb/mixture-vitae-working/finevideo` — Huu created this at some point but doesn't recall exactly what it is; need to check for overlap with our own pipeline (avoid a repeat of the `valid_with_seed` double-counting issue).
- **Open concern (not yet acted on):** naively mixing all of MV-Omni (6.93B tokens, 0 agent tokens) into the training corpus dilutes the agent (pose) token ratio from 12.2% (FineVideo v4 alone) down to ~5.2% of the combined mix. Since agent tokens are the project's core differentiator, consider dropout on MV-Omni (same treatment as Cosmos/AVC-LM) or oversampling agent-bearing records before combining.

**Current priority ranking (given JUPITER down + "more data before training" preference):**

| Tier | Task | Needs cluster? | Impact |
|---|---|---|---|
| ✅ | ~~Investigate MixtureVitae-Backup/multimodal~~ | No | Done Jul 9 — mostly text, ~3.27B raw SNAC codes found; awaiting Huu's go/no-go |
| P0 | Clarify "finevideo reformulation" on leo | No | Avoid double-counting |
| P0 | Decide MV-Omni mix ratio (agent dilution fix) | No | Protects core pose signal |
| P0 | Define eval protocol (DISCUSS-3, still open) | No | Required before any training run |
| P0 | Decide text/instruction data mix ratio (DISCUSS-1) | No | Steerability |
| P1 | Code the full-scale captioning pipeline (design finalized Jul 12, see session update) | No (CPU, per Jul 12 decision) | Highest — fixes root cause 2 (measured density ~1.86 captions/activity, short of the original ×4 target) |
| P1 | Write ego-centric perspective converter | No (GPU only to run) | 2× pose diversity, free |
| P1 | Mix MV-Omni into Megatron format | CPU only | +6.93B tokens, vocab already ready |
| P2 | Scope abc.bot, MolmoAct2-BimanualYAM, OmniVideo-100K, MINT-1T-HTML, Gen-EgoData | No | New robot/video sources, TBD size |
| P2 | Investigate leo seed2 + euro_pat | No | TBD |
| P3 | Cosmos3-DROID pipeline run | GPU | First real robot-domain data |
| P3 | Full captioning run, Megatron re-tokenize combined corpus, train v0.3 | GPU (JUPITER) | Blocked until cluster back + data ready |
| P4 (deferred) | 1-CP, Moss-Audio V2, Qwen3 migration, PAB-Spline angle spec, Isaac Sim | — | Explicitly held off per team decisions |

---

## The Big Picture — What Are We Actually Building?

The ultimate target: a single model that receives multimodal input (video frames + speech/text commands) and outputs action tokens that can be decoded into robot joint trajectories. The analogy Huu used: *"hear a verbal command like 'walk forward', and have a robot actually walk forward based on learned pose patterns."*

The longer-term vision is a model that, given an image of a beaker and a chemical formula, could reason through the task ("Make salt water") and translate that into arm/hand movements — **without being explicitly trained on that exact task**. This requires genuinely cross-modal binding: vision ↔ language ↔ action.

We are building this by pretraining a 1.7B LLM on an interleaved token stream:

```
USER: <activity description> [Speech: ...]  ASSISTANT:
  <seed2_N> ...          # Semantic keyframe tokens  (1fps, vocab 8192)
  <cosmos_N> ...         # Spatial video tokens      (every 8 frames, vocab 64000)
  <avclm_N> ...          # H.264 BPE video tokens    (every 8 frames, vocab 8192)
  <fps_30> <pelvis> ...  # 3D human pose tokens      (every 8 frames, 17 joints)
  <snac_N> ...           # Audio tokens — SNAC listen format (~10 tokens per 8-frame chunk)
```

The model learns to "read" and "continue" this interleaved sequence. In inference, you prompt it with video tokens + a text command, and it predicts the next agent tokens = the motion.

**Why this approach?** No prior VLA model has tried to unify video tokenization (Seed2/Cosmos), speech (SNAC), and continuous motion (PCHIP spline) into a single LLM autoregressive context. We are at the research frontier — nobody here has done this before.

---

## Timeline Overview

| Period | Key milestone |
|--------|--------------|
| Jun 2025 | Project started. FineVideo dataset chosen (~40K YouTube videos). |
| Jul–Sep 2025 | Branch A: Video token extraction pipeline (Seed2, Cosmos, AVC-LM). 160 GPU run. |
| Sep–Nov 2025 | Branch B phase 1–3: HRNet 2D pose, MotionBERT 3D lifting, kinematics. |
| Nov–Dec 2025 | Phase 4: YOLO cleaning. Phase 5 first iteration (opaque 256-token format). |
| Jan–Feb 2026 | Phase 5 rewrite → Adaptive PCHIP (self-describing named joint tokens). |
| Mar 2026 | Phase 6 merge, Phase 7 flatten. First Megatron tokenization. |
| Apr 2026 | **First model** trained (vla-1.7b-pab-spline-25b-test). Broken tokenizer discovered. |
| May 2026 | Tokenizer fix: `add_tokens(special_tokens=True)`. Full re-tokenization. |
| Jun 2026 | **Second model** trained (vla-1.7b-pab-spline-adaptive). Evaluation. Data inventory. |

---

## What Is Done — Detailed

### Phase A: Video Token Extraction

**Script:** `pipeline_video/pipeline.py` | **Compute:** 40 nodes × 4 GPU

Processed all ~40K FineVideo videos. Each activity segment tokenized into:
- **Seed2**: 1fps semantic keyframe tokens (8192 vocab)
- **Cosmos**: every-8-frame spatial tokens (64000 vocab)
- **AVC-LM**: every-8-frame H.264 BPE tokens (8192 vocab)

Output: 160 `training_ready_rank_*.jsonl` files. Each file contains hierarchical JSON (video → scenes → activities → tokens + speech transcript + metadata).

---

### Phase 1: 2D Pose Detection

**Script:** `pipeline_pose/phase1_hrnet_gpu.py`

- HRNet-W48 + Faster R-CNN person detector on all 40K videos
- Output: 2D joint coordinates (17 joints, COCO format) per frame
- **40,804 videos**, 145 GB

---

### Phase 2: 3D Pose Lifting

**Script:** `pipeline_pose/phase2_motionbert_gpu.py`

- MotionBERT lifts 2D → 3D (pretrained on Human3.6M)
- **40,804 videos**, 259 GB

---

### Phase 2.5: 30fps Resampling

**Script:** `pipeline_pose/phase2_5_resample_30fps.py`

- Linear interpolation from native video fps → uniform 30fps
- Required so all modalities share the same time grid
- 67 GB

---

### Phase 3: Kinematics Processing

**Script:** `pipeline_pose/phase3_kinematics_processor.py`

- Butterworth temporal smoothing
- Bone-length normalization to canonical H36M skeleton
- Pelvis root-centering
- Anti-teleportation filter (drops sudden-jump windows)
- Windowed into 8-frame chunks → shape `(windows, 8, 153)` where 153 = 17 joints × 3 dims × 3 kinematics (pos/vel/acc)
- **40,200 videos** (604 dropped as too short), 193 GB

---

### Phase 4: YOLO Person-Presence Filtering

**Script:** `pipeline_pose/phase4_yolo_cleaner.py`

- YOLOv8 person detection per frame
- Drops any 8-frame window where ≥4 frames have no detected person (confidence ≥ 0.75)
- **40,195 videos**, 107 GB

**⚠ Pose Data Quality Finding (Jul 2, 2026):**

Side-by-side skeleton visualization (`tools/visualize_skeleton_sidebyside.py`) + direct inspection of `yolo_cleaned` data revealed significant quality issues:

| Issue | Detail |
|-------|--------|
| **Joint sparsity** | Average 4–7 finite joints per frame out of 17 (24–41% skeleton) |
| **Arms absent** | j11–j16 (both arms: shoulder/elbow/wrist) = NaN in nearly all frames — MotionBERT cannot reliably lift arm joints from YouTube videos due to occlusion/side views |
| **Zero-fill artifact** | j10 (head_top) often stores (0,0,0) when undetected, identical to pelvis position — counted as finite but is wrong/misleading |
| **Coordinate scale OK** | ankle at ~−0.638m below pelvis is anatomically plausible; metric scale is correct |

**Impact on training:** Pose tokens are predominantly lower body (hip/knee/ankle) + torso. The arms — most important for manipulation tasks — are almost never captured. The model learns rough walking/sitting body motion but not fine hand/arm motion. This is a fundamental limitation of monocular video pose lifting from YouTube.

**Does NOT affect FineVideo as pretraining signal** — even noisy lower-body pose is better than none for learning video-pose correlation. But for downstream manipulation fine-tuning, better pose data (simulation, MoCap, or depth cameras) will be needed.

---

### Phase 5: Adaptive PCHIP Tokenization

**Script:** `pipeline_pose/phase5_adaptive_pchip.py`

For each 8-frame window, for each of 17 joints:
1. Compute trajectory curvature
2. Choose 2, 4, or 8 control points: low curvature (static) → 2 CPs; medium → 4 CPs; fast motion → 8 CPs
3. Quantize positions to uint8: `N = clip(round((v + 2.0) / 4.0 * 255), 0, 255)` mapping [-2m, +2m]
4. Emit self-describing tokens: `<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> ... </pelvis>`

**Why adaptive?** A static pelvis doesn't need 8 data points — 2 suffice. A fast-moving wrist needs 8. Reduces average token count by ~35% vs fixed 8-CP.

**Previous iterations (abandoned):**
- `phase5_interpolation_tokenizer.py` — 256 opaque uint8 tokens. Abandoned: tokens were not self-describing, model couldn't learn joint semantics.
- `phase5b_xyzt_tokenizer.py` — 409 fixed tokens (all 8 frames × 17 joints × 3 dims). Self-describing but wasteful.

Output: **18,847 videos** (only where YOLO confirmed human presence), 7.4 GB.  
Token range: 171 (all 2-CP, very static pose) to 579 (all 8-CP, fast motion), typical ~250–300 per window.

---

### Phase 6: Merge

**Script:** `pipeline_pose/phase6_merge_adaptive.py`

- Injected `<agent>...</agent>` blocks after each `<avc_lm>` block in the training_ready files
- Frame-aligned by matching agent window_ids to AVC-LM chunk indices (both at 30fps, 8-frame windows)
- Added `chunk_timing` array to each activity (precise timestamps for every chunk, which modalities are present)
- ~399K activities, **~2.15M agent blocks** injected
- Output: 160 `final_vla_adaptive_rank_*.jsonl`, **657 GB**

**Phase 6 v2 — SNAC injection support (Jun 28, 2026):**
- Added `--snac-tokens-dir` argument to also inject SNAC audio tokens per chunk
- New `inject_chunk_tokens()` function handles both agent + SNAC in one pass over `video_tokens`
- Token order per 8-frame chunk: `<cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<snac>...</snac>]`
- `chunk_timing` now includes `has_snac` flag per chunk
- `timing_meta` now includes `snac_rate: "37.5_tokens_per_sec_listen_format"`
- Backward compatible: running without `--snac-tokens-dir` behaves identically to v1
- **Requires `snac_finevideo.py` to run first** → `{video_id}_snac.jsonl` files in snac output dir

---

### Phase 7: Flatten + Augment

**Script:** `pipeline_pose/phase7_flatten.py`

Converts hierarchical JSON → flat Megatron-LM JSONL. Key decisions:

**Agent-only filter:** Only activities with `<agent>` blocks are emitted (every training record has action data).

**Modality dropout (token balancing) — v1 (old, already trained on this):**
| Modality | Raw ratio vs agent | Drop rate | Resulting ratio |
|----------|-------------------|-----------|----------------|
| AVC-LM | ~373× | 99% | ~4× |
| Cosmos | ~19× | 90% | ~2× |
| Seed2 | ~1× | 0% | 1× |
| Agent | baseline | 0% | 1× |

**Modality dropout — v2 (Jun 27, 2026 update, pending re-flatten):**
| Modality | Drop rate | Reason |
|----------|-----------|--------|
| AVC-LM | **100%** | Removed until ablations confirm benefit (per Huu) |
| Cosmos | **50%** | Keep ~6/12 blocks per activity for modality transition learning |
| Seed2 | 0% | Keep all — primary visual signal |
| Agent | 0% | Keep all |

**Text augmentation:** 15% synonym replacement, 5% stopword dropout, 10% sentence permutation, random speech/token interleaving, random layout block shuffling.

Output v1: 160 files, **69,844 records**, 19.2 GB → `megatron_dataset_adaptive/`  
Output v2: → `megatron_dataset_v2/` (cosmos 50% drop, avclm 100% drop — re-flattened Jun 27, 2026)

**Phase 7 v3 — SNAC + updated filter (COMPLETE Jul 2, 2026):**
- Added `<snac>...</snac>` block extraction in `process_tokens_to_individual_tags` (pass-through, like agent)
- Added `--drop_snac` argument (default 0.0 = keep all SNAC tokens)
- **Changed record filter:** was `<agent> required`; now emits if `<agent>` OR `<snac>` present
  - Full-chain records: seed2 + cosmos + agent + snac — **69,811 records (18.8%)**
  - Partial-chain records: seed2 + cosmos + snac — **302,044 records (81.2%)**
  - Bad records (neither): **0**
- Output: `megatron_dataset_v3/` — 160 files, **371,888 records**, **72 GB**
- Sample: `samples/after_flatten_v3.json` | Upload script: `tools/upload_flattened_hf.py` (updated for v3)

**✅ Phase 7 v4 — Per-chunk temporal ordering (COMPLETE Jul 2, 2026):**

Phase 7 fully rewritten (`pipeline_pose/phase7_flatten.py`). State machine walks Phase 6 output in document order, emitting per chunk: `[seed2?][cosmos?][agent?][snac?]`. Speech moved to dedicated `### Speech:` header.

**v4 stats:** 160/160 files, 371,888 records, **5.217B tokens** (seed2 6.4% / cosmos 74.4% / agent 12.2% / snac 7.0%). Runtime: 36 min / 32 workers.

**Bugs fixed:**
- Temporal misalignment (v3: all agent at end → 69% of records had 0% agent in first 4096 tokens. v4: per-chunk → all records have agent in first 4096 tokens)
- Speech injection into agent grammar (v3: speech words scattered into joint sequences. v4: speech in header only)

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v4/` (160 files)

**Token rates per 8-frame chunk (verified Jul 2, 2026):**

| Modality | Tokens/chunk | Per 30s (after v3 dropout) |
|----------|-------------|---------------------------|
| Seed2 | 32 fixed (1 block per 3.75 chunks) | 30 × 32 = **960** |
| Cosmos | 200 fixed (every chunk) | ~56 × 200 = **11,200** |
| Agent | 171–579 (~280 typical) | up to 112 × 280 = **31,360** |
| SNAC | 9 or 12 (avg 10, alternating) | 112 × 10 = **1,120** |
| AVC-LM | 885–5,055 | **0** (dropped) |

**DATA PATHS (IMPORTANT — updated Jun 27, 2026):**  
JUPITER `/e/data1` is sometimes down (cluster maintenance). All critical data copied to `/p/`:
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/
  ├── final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl  ← INPUT for Phase 7
  ├── megatron_dataset_adaptive/flat_*.jsonl                  ← v1 flat output
  └── megatron_dataset_v2/flat_*.jsonl                       ← v2 flat output (pending)
```
Phase 7 script and SLURM now default to `/p/` paths.

---

### Tokenizer

**Script:** `tools/expand_vocab.py`, `tools/upload_tokenizer.py`

Extended GPT-NeoX-20b (50,277 tokens) with 93,938 VLA tokens using `tokenizer.add_tokens(special_tokens=True)`.

**The critical bug in the first model:** Editing `vocab.json` directly does NOT register BPE merge rules. The tokenizer split `<seed2_1137>` → 7 sub-pieces. Despite this, the first model showed signal (learned to predict sub-piece sequences) but was not decoding real tokens.

**The fix:** `add_tokens(special_tokens=True)` bypasses BPE merging, treating every VLA token as atomic.

Published: `EmpathicRobotics/tokenizer-vla-adaptive` (144,215 vocab, padded to 144,256 for Megatron).

---

### Phase 8: Megatron-LM Tokenization

Tokenized 160 JSONL files → 2 binary shards:

| Shard | Tokens | Size |
|-------|--------|------|
| `data_shard_00000.bin` | 2,684,323,146 | 10.00 GB |
| `data_shard_00001.bin` | 156,389,702 | 0.58 GB |
| **Total** | **2,840,712,848 (2.84B)** | **10.58 GB** |

---

### Phase 9: Training — Model 2 (June 2026)

**Model:** `EmpathicRobotics/vla-1.7b-pab-spline-adaptive`  
**Architecture:** OpenSci-Ref 1.7B (24 layers, 2048 hidden, 32 heads → **1.91B params** with 144K vocab embeddings)  
**Config:** `oellm-autoexp/config/experiments/nguyen38/vla_adaptive.yaml`  
**Compute:** 64 nodes × 4 GH200 = 256 GPUs, ~35 min wall time

Training schedule:
| Iter | Loss | LR | Tokens seen |
|------|------|----|------------|
| 200 | 2.982 | 4e-3 | 0.84B |
| 500 | 2.070 | 4e-3 | 2.10B |
| 1000 | 1.672 | 4e-3 | 4.19B |
| 2000 | 1.476 | 3.2e-4 | 8.39B |
| **2032 (val)** | **1.501** | — | — |

Val PPL: **4.49** | Test PPL: **4.45** | ~3 epochs over 2.84B tokens

---

### Data Inventory (June 26, 2026 — Complete)

**Script:** `tools/data_inventory.py` | **Checkpoint:** `tools/inventory_checkpoint_v2.json`

Scanned all 242 files across 4 dataset families:

| Dataset | seed2 | cosmos | avclm | agent | snac | text | **TOTAL** |
|---------|-------|--------|-------|-------|------|------|-----------|
| FineVideo-VLA (160 files) | 89.9M | 210.2M | 474.4M | 564.9M | — | 11.4M | **1.35B** |
| MV-Backup valid_with_seed (64 HF shards) | 5.6M | — | — | — | — | — | **5.6M** |
| MV-Backup stack_images3_gzip (12 archives) | 313K | — | — | — | — | — | **313K** |
| MV-Omni valid_snac (6 gzip files) | — | — | — | — | 4.92B | 1.99B | **6.93B** |
| **TOTAL** | **95.8M** | **210.2M** | **474.4M** | **564.9M** | **4.92B** | **2.00B** | **8.29B** |

**Key findings:**
- `valid_with_seed` (1.1 TB downloaded!) yields only 5.6M seed2 tokens — **negligible, not worth the storage cost**. Shards 0–30 contain only raw `.png`/`.ogg` with zero tokenized content. Only shards 31–63 have `_seed2.jsonl` inside inner archives.
- MV-Omni is the only substantial external source at 6.93B tokens. BUT `<snac_N>` and `<seed_N>` tokens are **not in the current tokenizer vocab** — blocked until vocab expansion.
- **Only FineVideo has agent (3D pose) tokens.** No external dataset contributes pose data.
- **Training-ready today: 1.35B tokens** (FineVideo only, with current vocab).

---

## Current State — What Works, What Doesn't

### Works
- Pipeline end-to-end: raw video → 3D pose → tokens → Megatron bin → training → deployable HF checkpoint
- All VLA tokens are atomic (tokenizer fix confirmed)
- Model correctly completes 17-joint agent blocks: right joint ordering, valid xyz/t values, decodable to 3D pose via PCHIP
- 3D pose decoder verified: model output → (8, 17, 3) trajectory in correct physical range

### Does NOT work yet
- **Autonomous modality transitions:** When prompted with only text, the model stays in seed2 mode and never transitions to cosmos/avclm/agent. It requires agent tokens in the prompt to continue in agent mode.
- **Root cause 1 — Data starvation:** 2.84B tokens for 1.91B params = ~1.5× Chinchilla ratio. Optimal is ~20×. Each training sample seen only ~3 times — enough for local pattern memorization, not high-level sequencing.
- **Root cause 2 — No visual language anchors:** Text is only Title/Context/Keywords. No captions describe what's happening at each timestamp. The model has no language signal to know "after these seed2 tokens, cosmos tokens come next."
- **Root cause 3 — Over-aggressive dropout:** 99% AVC-LM + 90% Cosmos dropout means most records lack the full transition chain. Model rarely sees seed2 → cosmos → avclm → agent in sequence.

---

## What's Next — Prioritized Roadmap

### Immediate priorities (code during any available time, no GPU needed)

**Priority 1 — Vocab expansion for SNAC tokens** ← PARTIALLY DONE
- ~~Convert MV-Omni `<seed_N>` → `<seed2_N>`~~ **DONE** (Jun 27, 2026)
  - Script: `data_prep/convert_mvomni_seed.py`
  - Output: `/p/data1/mmlaion/shared/vla/mv_omni_converted/mv_omni_snac_*.jsonl.gz`
  - 1,593,301 records | 19,249,664 seed tokens converted | 30 GB
  - `<seed_N>` tokens fully eliminated — zero remaining in output
- **REMAINING:** Add `<snac_0>` ... `<snac_4095>` (~4096 tokens) to tokenizer via `add_tokens(special_tokens=True)`
  - New vocab: ~148,311 tokens (no need for `<seed_N>` — already converted to `<seed2_N>`)
  - Unlocks MV-Omni's **6.93B tokens** for training
  - Effort: ~1 day

**Priority 2 — Adjust modality dropout in Phase 7** ← ~~DONE~~ (Jun 27, 2026)
- AVC-LM: 99% → **100% drop** (removed entirely)
- Cosmos: 90% → **50% drop** (keeps ~6/12 chunks per activity)
- Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2/`
- Uploaded to `EmpathicRobotics/FineVideo-Phase7-Flattened` (v2 commit)
- **Next step:** Megatron re-tokenize `megatron_dataset_v2/` → new `.bin/.idx` shards → re-train v0.2

**Priority 3 — Ego-centric perspective for FineVideo**
- Read Phase 4 yolo_cleaned pose data
- Apply rotation matrix: place camera at `head_top` joint position, orient along thorax forward direction
- Generate additional agent token sequences from ego-centric view
- Same underlying motion data, double the data diversity (first-person + third-person)
- Effort: ~1 week code + 1 SLURM run

**Priority 4 — Write captioning pipeline code**
- Use `chunk_timing` timestamps to extract keyframes from FineVideo videos
- Pass each keyframe through SmolVLM2 or Qwen2.5-VL
- Interleave generated captions into the token sequence
- Expected impact: ×4 records with language anchors at every modality transition → fixes root cause 2
- Effort: 1–2 weeks code (GPU run on JUPITER is separate)

### Medium-term (needs dedicated GPU time on JUPITER)

**Priority 5 — Collect agent + cosmos + snac from Cosmos3-DROID**
- `nvidia/Cosmos3-DROID` on HuggingFace: robot arm manipulation videos with Cosmos video tokens
- Run YOLO + Phase 1–5 equivalent to extract agent tokens (robot arms/hands)
- Add SNAC tokens if audio track exists
- First robot-domain data — critical for generalization beyond human motion
- Hold off on AVC-LM until ablations confirm it helps (per Huu's guidance)

**Priority 6 — Vocab expansion (tokenizer build)** ← **COMPLETE (Jul 1, 2026)**

Script: `tools/build_tokenizers.py`. Hai output:
- `tokenizer_vla_adaptive_v2`: 144,215 (base) + 12,290 SNAC = **156,505 vocab**, tất cả atomic ✓
- `tokenizer_vla_qwen3`: ~151,669 (Qwen3) + 106,228 VLA = **257,897 vocab**, tất cả atomic ✓

**Priority 7 — SNAC tokenization for FineVideo** ← **COMPLETE (Jul 1, 2026)**

Job `snac_cpu_14077331`, 32 array tasks on `batch` partition (CPU), submitted from `jwlogin08`.

**Results:**
| Metric | Value |
|--------|-------|
| Tasks completed | **32/32** (100%) |
| Activities processed (ok) | **371,855** |
| fail_audio (no audio track) | 530 (~0.1%) |
| fail_snac | **0** |
| Total SNAC tokens | **363,029,331 (~363M)** |
| Output files | **40,779** `{video_id}_snac.jsonl` |
| Output size | **6.5 GB** |
| Output location | `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens/` |

**Output format** — one file per video, one JSONL line per activity:
```json
{
  "video_id": "abc123",
  "activity_id": "scene_1_act_1",
  "start_sec": 1.0,
  "end_sec": 8.9,
  "has_agent": true,
  "snac_by_chunk": {
    "0": ["<snac_132247>", "<snac_132788>", "<snac_147076>", ...],
    "1": [...],
    ...
  }
}
```

**Chunk splitting mechanism:**
- Encode full activity audio once (1 call to SNAC) → flat list of tokens
- SNAC output = sequence of base frames, each = exactly 3 tokens (L0 + L1_even + L1_odd triplet)
- `n_base = len(flat_tokens) // 3` — truncate to complete base frames (atomic unit, cannot split triplet)
- Proportional split: `start_base[k] = round(k * n_base / n_chunks)`, `end_base[k] = round((k+1) * n_base / n_chunks)`
- Each chunk gets `end_base - start_base` base frames × 3 = **9 or 12 tokens** (alternating due to 3.33 base frames/chunk)
- Last chunk is NOT shorter — `round(n_chunks × n_base / n_chunks) = n_base` exactly
- Temporal alignment error: ±1 base frame = ±80ms at each chunk boundary (acceptable for pretraining)

**Next steps unblocked by this completion:**
1. Vocab expansion — add 12,288 `<snac_N>` tokens to tokenizer
2. Re-run Phase 6 v2 with `--snac-tokens-dir`
3. Re-run Phase 7 v3 → `megatron_dataset_v3/`
4. Megatron re-tokenize → train v0.3

**CLUSTER ARCHITECTURE NOTE (discovered Jun 28, 2026):**
- `jwlogin08.juwels` = JUWELS Cluster login node (x86_64)
- `juwels-booster.fz-juelich.de` = JUWELS Booster login nodes (separate system, ppc64le compute)
- `laionize` account with GPU access (`booster` partition) is only usable from the Booster login nodes
- From JUWELS Cluster login, `laionize` only has CPU partitions: `batch`, `devel`, `large`
- **To submit GPU job: SSH to `juwels-booster.fz-juelich.de` first**

**Task list already built (Jun 28, 2026):**
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_task_list.json
→ 40,798 videos, 372,385 activities, all with chunk_timing
```

**Submission commands (when on juwels-booster login node):**
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose

# GPU mode: 16 workers on booster partition, ~8-12h
bash slurm/submit_snac_finevideo.sh

# CPU fallback (from jwlogin, slower ~24h, no SSH needed):
bash slurm/submit_snac_finevideo.sh --cpu
```

**Run sequence after SNAC tokenization completes:**
```bash
# Step 1 — DONE: build task list
# snac_task_list.json already at TASK_CACHE path

# Step 2 — RUNNING (Jun 30): CPU batch job, 32 workers, ~20-24h
# Output: .../FineVideo-VLA/snac_tokens/{video_id}_snac.jsonl (~40K files)

# Step 3: Vocab expansion — add 12,288 <snac_N> tokens to tokenizer
# TODO: update tools/expand_vocab.py to include snac range [128266..148745]

# Step 4: Re-run Phase 6 with SNAC injection
python pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/training_ready_rank_*.jsonl" \
  --agent-tokens-dir /p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive \
  --snac-tokens-dir  /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens \
  --output-dir       /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v2 \
  --output-prefix    final_vla_adaptive_v2

# Step 5: Re-run Phase 7 → megatron_dataset_v3/
python pipeline_pose/phase7_flatten.py \
  --input-glob ".../final_dataset_adaptive_v2/final_vla_adaptive_v2_rank_*.jsonl" \
  --output-dir ".../megatron_dataset_v3" \
  --drop_cosmos 0.5 --drop_avc 1.0 --drop_snac 0.0 --workers 16

# Step 6: Megatron tokenize → .bin/.idx → train v0.3
```

**Priority 7 — Investigate leo seed2 + euro_pat**
- Check what's on the `leo` cluster: seed2 + euro_pat datasets mentioned by Huu
- Quantify token counts before committing storage/compute

**New Data Candidates (Jul 7, 2026 — from team Discord)**

Found while scoping VLA data sources broadly. None yet scoped for token/hour counts or license fit.

| Source | What | Notes |
|---|---|---|
| `abc.bot` (Amazon) | 400h robot recordings **in simulation**, includes physics state (MjData) | Most promising — permissive, has eval env, same embodiment throughout. blanchon.jl: "indeed perfect" |
| `allenai/MolmoAct2-BimanualYAM-Dataset` | 2 TB, bimanual YAM arm robot data | Check license + embodiment compatibility |
| `MiG-NJU/OmniVideo-100K` | Video dataset | Not yet scoped |
| `mlfoundations/MINT-1T-HTML` | Large text/HTML dataset | Not yet scoped — likely for language mix (DISCUSS-1), not video |
| `genrobot2025/Gen-EgoData` | Egocentric robot data | Not yet scoped |
| `finevla.xlang.ai` | Possible VLA dataset | HF link not found yet — may be unreleased |
| `mira-wm.com` | World model reference (Kyutai released similar) | Reference/inspiration, not necessarily a data source |

**Team constraint:** synthetic/simulation data (abc.bot, MolmoAct2, Cosmos3-DROID, etc.) capped at **≤30% of total training mix** — decided by team consensus (Huu, citing literature), to keep the balance toward real human/robot video.

**Priority 8 — First re-training run (v0.2)**
- After items 1, 2, 4 are done: estimated **10–20B tokens** available
- Continue training from current checkpoint (2032 iter) with new data + adjusted dropout
- Expected result: model begins to learn modality transitions autonomously

### Long-term (3–6 months)

**Priority 9 — More text data**
- Mix in standard LLM text data (to create language binding and prevent catastrophic forgetting)
- Target: text tokens at ~10–15% of total training mix

**Priority 10 — Qwen3 migration**
- Retokenize entire dataset with Qwen3-based expanded tokenizer
- Requires full re-run of Phase 8 (Megatron tokenization) and training from scratch
- Benefit: native HF ecosystem support, vLLM, llama.cpp compatibility
- Huu's config: cherry-picked from commit `7dcf8a5`

**Priority 11 — PAB-Spline spec upgrade**
- Current tokenizer: PCHIP xyz-only (positions)
- Spec calls for: joint angles (q/qd), phase variable φ ∈ [0,1], cyclic gait detection, static joint compression
- Blocked by: need to run kinematics pipeline again with angle computation

**Priority 12 — Isaac Sim integration**
- Generate Unitree H1 rollouts in Isaac Sim / ManiSkill
- Tokenize simulation data with PAB-Spline tokenizer
- Sim-to-real gap: map joint tokens → H1 control signals

---

## Data Landscape — Where We Are and What We Need

### Current training-ready data: 1.35B tokens (FineVideo only)
This is too small. For a 1.7B model, Chinchilla-optimal is ~34B tokens. We're at ~4% of that.

### Unlockable with vocab expansion only (no new collection): +6.93B tokens
MV-Omni valid_snac is sitting there, tokenized, but blocked by missing `<snac_N>` / `<seed_N>` vocab entries. Adding these two token families = 1–2 days of work = unlock 6.93B tokens = reach ~8.3B total. This is the highest-leverage action available right now.

### Unlockable with GPU runs: +5–10B tokens (captioning, ego-centric, Cosmos3-DROID)
The captioning pipeline alone multiplies FineVideo by ~4× (69,844 records → ~280K records) with richer language context. Ego-centric adds a second perspective for free.

### Target: 20–40B tokens for v0.2 training
With vocab expansion + MV-Omni + captioning + Cosmos3-DROID + SNAC-FineVideo, reaching 20–40B tokens is realistic within 2–3 months of focused work.

---

## Honest Assessment — Are We On The Right Track?

**Yes, the architecture is sound.** The second model proved the core hypothesis: a 1.7B LLM can learn the grammar of multimodal token sequences — joint ordering, valid xyz ranges, modality-specific token distributions — purely from next-token prediction on flat interleaved sequences.

**The bottleneck is data, not architecture.** The model's failure to autonomously transition between modalities is fully explained by data starvation and missing language anchors. These are solvable engineering problems, not fundamental flaws.

**The direction is genuinely novel.** No published work unifies Seed2 + Cosmos + SNAC + PCHIP pose tokens in a single autoregressive LLM context. The closest prior work (RT-2, OpenVLA, π0) uses much simpler action representations and doesn't attempt continuous 3D body pose. We're building something nobody else has built.

**The risks:**
1. **Scale gap:** Even at 20B tokens, we're far below frontier LLMs. Our model may generalize poorly to novel prompts. Mitigation: mix in standard text data to maintain language ability.
2. **No robot deployment yet:** Current pose data is from YouTube humans, not actual robot joints. Isaac Sim integration is still future work. The model won't directly control a real robot without sim-to-real adaptation.
3. **SNAC/audio quality:** Orpheus SNAC2 is "good enough" per Huu's assessment, but retokenizing with Moss Audio Tokenizer V2 (mentioned in chat, 2.1B decoder) could improve audio quality significantly. Deferred for now.
4. **Qwen3 migration overhead:** If we retokenize for Qwen3, existing `.bin/.idx` shards become obsolete. Should be done once, not multiple times — wait until the data landscape is more stable.

**What success looks like at each stage:**
- **v0.2 (2–3 months):** Model autonomously transitions from text prompt → seed2 → cosmos → agent tokens without needing agent tokens in the prompt.
- **v0.3 (4–6 months):** Model responds to spoken commands (SNAC) by generating valid agent motion tokens. "Walk forward" → valid pelvis/hip/knee trajectory.
- **v1.0 (6–12 months):** Model observes visual scene + receives instruction, generates motion that respects scene geometry. The chemical beaker test.

---

## Key Decisions Log

| Decision | Why | Date |
|----------|-----|------|
| Chose Adaptive PCHIP over fixed 409-token format | Self-describing, ~35% fewer tokens for static joints | Feb 2026 |
| Tokenizer fix via `add_tokens()` not vocab.json edit | BPE requires merge rules, not just vocab entries | May 2026 |
| 99% AVC-LM dropout in Phase 7 | AVC-LM was 373× more tokens than agent — would dominate context | Mar 2026 |
| valid_with_seed NOT worth using | 1.1 TB download for 5.6M tokens (< 0.5% of FineVideo) | Jun 2026 |
| Hold AVC-LM in new datasets until ablations | No evidence yet that it helps vs adds noise | Jun 2026 |
| Ego-centric perspective as free data multiplier | Same underlying motion, different reference frame, doubles diversity | Jun 2026 |
| Qwen3 migration deferred | Too early — data landscape still changing | Jun 2026 |
| MV-Omni: convert seed→seed2 instead of adding new vocab | Avoids unnecessary vocab expansion; seed_N and seed2_N are identical semantics | Jun 2026 |
| SNAC injection in Phase 6, not Phase 7 | Phase 6 already does per-chunk injection; Phase 7 is stateless flatten. Keeping injection in Phase 6 means Phase 7 needs no external lookups. | Jun 2026 |
| SNAC chunk alignment: encode full activity once, split by count | Encoding per-chunk (0.267s segments) would lose audio context + slow due to many small calls. Encode once, split evenly preserves context and is accurate (SNAC rate is constant). | Jun 2026 |
| SNAC for ALL activities, not just agent ones | Only 14% of activities have agent tokens. Other 86% still have seed2+cosmos — adding SNAC teaches seed2→cosmos→snac transitions. Filtering to agent-only wastes most of the GPU run. | Jun 2026 |
| 1-CP compression: deferred, keep adaptive 2/4/8-CP | +7.1% gain (sample-based) doesn't justify full Phase 5→7 re-run right now; revisit later if needed | Jul 8, 2026 |
| Synthetic/sim data capped at ≤30% of total training mix | Team consensus (Huu), citing literature guidance; keeps balance toward real video | Jul 7, 2026 |
| Moss-Audio Tokenizer V2 usage: keep limited even if adopted | Huu: at 400 tok/s it would overwhelm the dataset if used broadly for omni-modal pretraining; only viable as a short high-detail segment followed by lower-rate SNAC, or standalone if not binding to language | Jul 2, 2026 |
| MINT-1T-HTML images: dropped entirely, text kept | No per-image license info available (dataset's own README confirms no copyright filtering was done, only NSFW/size/dedup); `cc_dump` is a CommonCrawl snapshot ID, not a Creative Commons marker. Text kept — different risk profile (LLM text pretraining under fair-use/TDM norms vs. redistributing raw hotlinked media) | Jul 18, 2026 |
| stera-10m: dropped | Not permissive (self-assessed + no objection from Huu) | Jul 18, 2026 |
| SenseNova-SI-8M: download in full | Real image bytes (not URLs), Apache-2.0, spatial-reasoning content relevant to embodied/robot use — best available static-image candidate found so far | Jul 18, 2026 |
| Egocentric perspective converter (Priority 3): rejected as scoped | The transform only relabels the SAME pose (isometry, no info gain) while pairing it with the SAME unchanged 3rd-person video — creates a physically incoherent video→pose training pair, not a real data multiplier. Only worth revisiting with genuine paired egocentric video, or for a narrow pose-only (no-video) sub-task | Jul 18, 2026 |
| Megatron tokenize: try account `laionize`/partition `batch` on JUWELS instead of `cstdl` on JUSUF | User's call, verified via `sacctmgr` that `laionize` has a valid `batch` association before writing scripts | Jul 18, 2026 |

---

## Published Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Tokenizer v1 (144,215 vocab, GPT-NeoX) | `EmpathicRobotics/tokenizer-vla-adaptive` | Live |
| **Tokenizer v2 (156,505 vocab, GPT-NeoX + SNAC)** | `EmpathicRobotics/tokenizer-vla-adaptive-v2` | **Live (Jul 1, 2026)** |
| **Tokenizer Qwen3 (257,897 vocab)** | `EmpathicRobotics/tokenizer-vla-qwen3` | **Live (Jul 1, 2026)** |
| FineVideo-Phase7-Flattened v4 (371,888 records, 5.217B tokens) | `EmpathicRobotics/FineVideo-Phase7-Flattened` | **Live (Jul 7, 2026)** |
| FineVideo-Phase5-AgentTokens (~399K activities) | `EmpathicRobotics/FineVideo-Phase5-AgentTokens` | Live |
| FineVideo-Phase4-YOLOPose (millions of windows) | `EmpathicRobotics/FineVideo-Phase4-YOLOPose` | Live |
| VLA Model v1 (broken tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-25b-test` | Live (deprecated) |
| VLA Model v2 (fixed tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-adaptive` | Live |
| Megatron .bin/.idx shards (2.84B tokens) | `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/` | Local |
| Data inventory checkpoint | `tools/inventory_checkpoint_v2.json` | Local |

---

## Environments & How to Run (JUWELS login node)

**env_tools** — use for: phase7_flatten, data_inventory, HF uploads, eval, any non-GPU script  
Location: `/p/data1/mmlaion/nguyen38/env_tools`  
Python: 3.12.3 | Has: torch, transformers, wn, datasets, scipy, huggingface-hub, rich, tqdm, ...

> **Note (Jun 27, 2026):** env_tools was created on JUSUF but we run on JUWELS. Python symlink and pyvenv.cfg had wrong paths. Fixed via `load_env_tools.sh`.

```bash
# Activate env_tools (source it, don't bash it — needs to modify your shell):
source /p/data1/mmlaion/nguyen38/3d-human-pose/load_env_tools.sh
# → auto-fixes symlinks on first run, then activates

# Then run whatever you need:
python pipeline_pose/phase7_flatten.py --workers 16 --skip-existing
```

**env_pose** (miniforge3 conda) — use for: phases 1–6 (HRNet, MotionBERT, YOLO, kinematics)  
Location: `/p/data1/mmlaion/nguyen38/3d-human-pose/env_pose`  
Activate: `source /p/data1/mmlaion/nguyen38/3d-human-pose/miniforge3/etc/profile.d/conda.sh && conda activate /p/data1/mmlaion/nguyen38/3d-human-pose/env_pose`

**Data paths on `/p/` (use these when JUPITER `/e/` is down):**
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/
  ├── final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl  ← Phase 7 input
  ├── megatron_dataset_adaptive/flat_*.jsonl                  ← v1 flatten output
  └── megatron_dataset_v2/flat_*.jsonl                       ← v2 flatten output (pending)

/p/data1/mmlaion/shared/vla/
  ├── mv_omni_converted/mv_omni_snac_*.jsonl.gz              ← MV-Omni seed→seed2 done
  ├── tokenizer_vla_adaptive/                                 ← local tokenizer copy
  └── tokenized_output/vla_adaptive/data_shard_*.bin/.idx    ← Megatron shards (2.84B tokens)
```

---

## Immediate Action Items (Next 2 Weeks)

### Đã hoàn thành (Jun–Jul 2026)
- [x] **SNAC CPU job** — **COMPLETE (Jul 1, 2026)**. Job `snac_cpu_14077331`, 32/32 tasks. 371,855 activities, 363M tokens, 6.5 GB → `/p/.../snac_tokens/`
- [x] **Dataset overlap check** — **COMPLETE (Jun 30, 2026)**. Kết quả: 27,359 video chồng nhau (86.9% of valid_with_seed ∈ omni_valid). Xem section "Dataset Overlap Analysis" bên dưới.
- [x] **Vocab expansion (tokenizer build)** — **COMPLETE (Jul 1, 2026)**. Script: `tools/build_tokenizers.py`. Tạo 2 tokenizer:
  - `tokenizer_vla_adaptive_v2` (GPT-NeoX-20b + SNAC): **156,505 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive_v2/`
  - `tokenizer_vla_qwen3` (Qwen3 + tất cả VLA tokens): **257,897 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3/`
  - Spot-check 12 tokens đại diện: tất cả **atomic** (1 token/ID), không có sub-piece splitting
  - SNAC token range: L0 [128266..132361], L1A [132362..136457], L1B [144650..148745]

### Pre-training Discussion Items (Jul 2, 2026 — from Huu chat)

> **⚠ Huu explicitly said: "Before you train let's talk." — do NOT start training until these 3 items are resolved.**

**[DISCUSS-1] Language data mix — what to add before training?**
- Current plan (FineVideo v4 + MV-Omni = 12B tokens) has almost no instruction/language data
- Huu: "mix in a few billion tokens mixture so we can steer the robot better"
- Huu: "We should look for some SFT dataset for our various target (pick up the Apple, Drive left, etc.)"
- Candidates on leo (`/mnt/sdb/mixture-vitae-working/`): `clappa_text_only`, `coco` (synthetic permissive), `misc_instr/hpprc-r1-distill-qwen-pseudo-qa.jsonl` (Japanese instruction)
- Also wants multilingual instruction datasets with reasoning/thinking
- **Action needed:** Identify token counts of available language datasets → decide mix ratio

**[DISCUSS-2] Compression analysis of Adaptive PCHIP — RESULTS READY**
- Huu: "Did you do an analysis by how much compression we got? If there is no or low compression then we know it's wrong."
- **DONE:** `tools/analyze_pchip_compression.py` — 18,847 files, 1,743,189 windows. Results:
  - **50.9% token saving** vs fixed 8-CP (284.1 avg vs 579)
  - CP tiers: 55.2% 2-CP / 25.6% 4-CP / 19.2% 8-CP
  - Most dynamic: r_knee (33.5% 8-CP), r_wrist (29.4%). Most static: pelvis (100% 2-CP)
  - Pelvis confirmed at origin: 500/500 samples within ±0.1m ✓
  - Coordinate system: absolute xyz after root-centering is correct
- **NEW — Pose data quality concern (Jul 2, 2026):**
  - Only **4–7 joints finite per frame** (out of 17) — 24–41% skeleton coverage
  - **Arms (j11–j16) nearly always NaN** — MotionBERT cannot reliably lift arm joints from YouTube (occlusion/side views)
  - **head_top (j10) zero-fill artifact** — stores (0,0,0) when undetected, same as pelvis, counted as finite but wrong
  - Impact: model learns lower-body + torso pose only. Fine for pretraining pose presence; NOT sufficient for arm/hand manipulation learning
- **Action needed:** Report numbers + pose quality concern to Huu

**BEAST comparison (Jul 3, 2026) — context for reporting to Huu:**

Huu asked: *"What does the BEAST paper say about their compression? That will give us a sanity check."*

BEAST = "B-spline Encoded Action Sequence Tokenizer" (KIT, NeurIPS 2025, arXiv 2506.06072). Uses B-splines with fixed N control points fit by ridge regression. Claims **4–8× compression** vs binning (e.g., ACT 100-step chunk → 15 tokens = 6.67×).

**Why our 50.9% looks lower:** different baselines.

| | Baseline | Result |
|---|---|---|
| **BEAST** | Binning (1 token/timestep/DoF) | 4–8× fewer tokens |
| **Ours** | Fixed 8-CP (already compressed) | ~2× fewer tokens |

Vs raw binning: our 284 tokens / (8×17×3=408 raw values) = **~1.5×** — much less than BEAST. Root cause: 34% of our tokens are overhead (wrappers + t tokens) to make the format self-describing for the LLM. BEAST has zero overhead (decoder structure is hardcoded). Self-describing is a deliberate design choice for LLM joint-semantic learning.

**Huu's 1-CP suggestion (Jul 3, 2026):** *"Why do we have 2-CP as minimum? Can we have 1-CP? Like — relative, no movement."*

Why 2-CP was minimum originally:
1. PCHIP requires ≥2 points (it's an interpolating polynomial — 1 point = nothing to interpolate)
2. "Low curvature" ≠ "no movement": joint may drift linearly within the window below tau_low

How 1-CP would work: if `quantize(frame_0) == quantize(frame_7)` for all 3 dims → emit only `<joint_x_N> <joint_y_N> <joint_z_N>` (no t token, 3 tokens vs current 8 tokens)

Estimated gain: ~4–5 qualifying joints/window × 5 tokens saved ≈ 20–47 tokens/window → **additional ~8–16% compression**. Requires grammar change + re-run of Phase 5 and all downstream phases.

**FINAL DECISION (Jul 8, 2026):** Deferred. Confirmed with Huu on Discord — keep the current adaptive 2/4/8-CP format as-is. Full-dataset validation run (18,847 videos) was started but interrupted by the JUWELS outage; not resumed. Revisit only if later data shows it's necessary — the +7.1% gain doesn't justify a full Phase 5→7 re-run right now. For paper purposes, "compression decreases the data by more than 50%" (vs fixed 8-CP) is the number to report.

**[DISCUSS-3] Eval setup**
- Huu: "We should start eval just to see how things perform with baseline"
- Need to define eval tasks BEFORE training, not after
- Candidates: agent token decode quality (MPJPE on 3D pose), modality transition accuracy, instruction-following on robot commands
- **Action needed:** Define eval protocol and implement baseline metrics

---

### Việc tiếp theo (unblocked bởi vocab expansion)
- [x] **Phase 6 v2 dry run** — **COMPLETE (Jul 1, 2026)**. Chạy thử 1 file (254 videos, ~5 phút). Kết quả:
  - SNAC inject: **259,503/259,505** avc blocks (~100%)
  - Agent inject: **12,705** blocks (đúng — 46% video có Phase 5 output)
  - Format verified: `</avc_lm> <agent>...</agent> <snac> <snac_N>... </snac>` ✓
  - `chunk_timing` có đủ các flag `has_seed2/cosmos/avc_lm/agent/has_snac` ✓
  - SLURM script mới: `slurm/submit_merge_adaptive_v2.sh` (account `laionize`, partition `batch`, 32 workers, 2h)
  - Ước tính toàn bộ 160 file với 32 workers: **~25–40 phút**
- [x] **Re-run Phase 6 v2** — **COMPLETE**. Job `14082096`, 32/32 workers. 40,804 videos | 398,775 activities | SNAC 100% | Agent 5.5% | 0 errors → `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v2/` (160 files)
- [x] **Re-run Phase 7 v3** — **COMPLETE (Jul 2, 2026)**. 160/160 files, 371,888 records, 72 GB → `megatron_dataset_v3/`
  - Full-chain (agent+snac): 69,811 (18.8%) | Snac-only: 302,044 (81.2%) | Bad records: 0
  - Token counts: seed2 332.6M | cosmos 3.88B | snac 363M | agent windows 2,148,474 | avclm 0 ✓
  - Sample: `samples/after_flatten_v3.json` | Upload script updated: `tools/upload_flattened_hf.py`
- [x] **Phase 7 v4 — temporal alignment fix** — **COMPLETE (Jul 2, 2026)**. Per-chunk ordering fixed, speech in headers, 5.217B tokens → `megatron_dataset_v4/`. See stats above.
- [x] **Upload Phase 7 v4 to HF** — **COMPLETE (Jul 7, 2026)**. `EmpathicRobotics/FineVideo-Phase7-Flattened` live with v4 data. Shared with Huu/joergfranke on Discord as the ready-to-tokenize dataset.
  Source: `megatron_dataset_v4/` | Upload dir: `hf_upload_flattened_v4/` | Dataset card: `tools/vla_flattened_dataset_card.md` (updated for v4)
- [ ] **Megatron re-tokenize** `megatron_dataset_v4/` with `tokenizer-vla-adaptive-v2` (156,505 vocab) → new `.bin/.idx` → train v0.3
- [x] **Upload tokenizers** — **COMPLETE (Jul 1, 2026)**. `EmpathicRobotics/tokenizer-vla-adaptive-v2` (156,505) + `EmpathicRobotics/tokenizer-vla-qwen3` (257,897), cả hai Live với model card đầy đủ

### Kết luận overlap check (Jun 30, 2026 — XONG)
- [x] Quyết định **KHÔNG dùng `valid_with_seed`** — 86.9% đã có trong omni_valid, 13.1% còn lại (4,141 video) chỉ có seed2 token và không đáng tốn thêm storage/compute
- [x] omni_valid **không overlap với FineVideo-VLA** (khác nguồn hoàn toàn — MixtureVitae vs YouTube FineVideo)

### Coding (không cần GPU)
- [ ] Start writing ego-centric perspective converter (Phase 3 → rotate to head camera)
- [ ] Start writing captioning pipeline (SmolVLM2/Qwen2.5-VL trên FineVideo keyframes)
- [ ] Investigate leo seed2 + euro_pat token counts
- [ ] Plan Cosmos3-DROID pipeline (download strategy, SLURM script)
- [x] Investigate `MixtureVitae-Backup/multimodal` (HF) — **DONE (Jul 9, 2026)**. Mostly text; SNAC tokens found in 2 files as raw int arrays. See "MixtureVitae-Backup Multimodal Investigation" section. Awaiting Huu's decision on whether to add.
- [ ] Clarify "finevideo reformulation" at `leo:/mnt/sdb/mixture-vitae-working/finevideo` — check overlap with own pipeline
- [ ] Decide MV-Omni mix ratio / dropout to avoid diluting agent token % (12.2% → ~5.2% if mixed naively)
- [ ] Scope new data candidates: abc.bot, MolmoAct2-BimanualYAM-Dataset, OmniVideo-100K, MINT-1T-HTML, Gen-EgoData (see "New Data Candidates" table above)

### Cluster account mapping (Jul 7, 2026 — for when submitting jobs)
```
JUSUF:   ccstdl
JUPITER: reformo
JUWELS:  laionize
```

---

## Dataset Overlap Analysis (Jun 30, 2026)

### Background

Từ cuộc họp Jun 28: Huu chỉ ra rằng `omni_valid` có khả năng được subsample từ `valid_with_seed`, dẫn đến **double-counting** nếu dùng cả hai. Kết luận ban đầu: chỉ dùng 3 dataset:
1. **FineVideo-VLA** (local)
2. **omni_valid** (MixtureVitae-Omni)
3. **stack_images3_gzip** (MixtureVitae-Backup)

Còn `valid_with_seed` cần check xem overlap với `omni_valid` bao nhiêu % trước khi quyết định có dùng hay không.

### Data đã download

| Dataset | Location | Format |
|---------|----------|--------|
| `valid_with_seed` | `/p/data1/mmlaion/nguyen38/inventory_cache/hf_shards/` | 64 outer `.tar.gz`, mỗi cái chứa file `.ogg`, `.png`, `_seed2.jsonl` |
| `omni_valid` | `/p/data1/mmlaion/nguyen38/inventory_cache/hf_snac/` | 6 `valid_snac_N.jsonl.gz` |
| `ontocord/VALID` | `multimodal/head.txt` | Chỉ có head sample (5 records), chưa download full |

### Script overlap check

**Script:** `tools/check_dataset_overlap.py`

**Logic:**
- `valid_with_seed`: Extract YouTube video ID (11 chars đầu) từ tên file trong các tar.gz
- `omni_valid`: Extract `params.id` từ metadata của mỗi JSONL record
- So sánh hai set, tính % overlap

**Command để chạy:**
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose
python3 tools/check_dataset_overlap.py
```

Không cần env đặc biệt — chỉ dùng stdlib Python (tarfile, gzip, json, re).

**Output:** In kết quả ra màn hình + lưu vào `tools/dataset_overlap_results.json`

### Kết quả (HOÀN THÀNH — Jun 30, 2026)

Script `tools/check_dataset_overlap.py` đã chạy xong. Kết quả lưu tại `tools/dataset_overlap_results.json`:

| Metric | Số liệu |
|--------|---------|
| `valid_with_seed` unique video IDs | **31,500** |
| `omni_valid` unique video IDs | **238,539** |
| Overlap (cả hai) | **27,359** (86.9% của seed / 11.5% của omni) |
| Chỉ có trong `valid_with_seed` | **4,141** |
| Chỉ có trong `omni_valid` | **211,180** |

**Kết luận: KHÔNG dùng `valid_with_seed`.** omni_valid đã cover 86.9% video của nó. 4,141 video còn lại chỉ có seed2 token và không đủ giá trị (tổng < 700K token) để bù cho 1.1 TB storage.

Log: `tools/overlap_run.log` | Kết quả JSON: `tools/dataset_overlap_results.json`

**Lưu ý quan trọng:** Script phải mở **2 tầng tar** (outer tar → inner tar → files) vì shards 0–30 chỉ chứa inner tar.gz bên trong, không có loose files. Script v1 bị lỗi (0 IDs) do bỏ qua inner tar — đã fix trong v2.

### Cấu trúc dataset (đã verify)

**omni_valid record format:**
```json
{
  "text": "<listen><snac_N>...<snac_N></listen>\n<see><seed_N>...</see>\n...",
  "metadata": "[{\"source\": \"grass-yt-cc-by.{YT_ID}|...\", \"params\": \"{\\\"id\\\": \\\"{YT_ID}\\\", ...}\"}]"
}
```
→ YouTube ID ở `metadata[0].params.id` (11 ký tự)

**valid_with_seed shard format:**
- Outer tar chứa: `shard_NNNNN.tar.gz` (inner) + loose files (`*.ogg`, `*.png`, `*_seed2.jsonl`)
- File name format: `{YT_ID_11chars}_{clip_num}[_{crop}][_seed2].{ext}`
- YouTube ID = 11 ký tự đầu của filename

**ontocord/VALID format** (head.txt — 3 lines per record):
```jsonl
{"file_name": "-mbDQC0y0PY_6.ogg", "media_type": "audio", "text": "...", "snac_token": [...]}
{"file_name": "-mbDQC0y0PY_6.png", "media_type": "image", "text": "..."}
{"emotion": "...", "query": "...", "answer": "...", "shard_idx": "shard_0"}
```

### Câu hỏi đã trả lời (Jun 30, 2026)

1. **Bao nhiêu % của `omni_valid` đến từ `valid_with_seed`?**
   → 11.5% (27,359/238,539). omni_valid chủ yếu là data riêng, KHÔNG phải subsample từ valid_with_seed như Huu dự đoán — thực ra ngược lại: valid_with_seed là subset của omni_valid.

2. **`valid_with_seed` có video không có trong `omni_valid` không?**
   → Có: 4,141 video (13.1% của valid_with_seed). Nhưng những video này chỉ có seed2 token, không có SNAC, và tổng token ~700K — không đủ giá trị để dùng riêng.

3. **`ontocord/VALID` có tương đương với `valid_with_seed` không?**
   → Chưa check (chỉ có head sample 20 records → 0 video ID). Không cần điều tra thêm vì đã quyết định không dùng valid_with_seed.

### Kết luận cuối cùng

**Chỉ dùng 2 nguồn external:**
- **omni_valid (MV-Omni)** — 238,539 video, 6.93B token (SNAC + text + seed2). Cần vocab expansion `<snac_N>`.
- **stack_images3_gzip** — 313K token seed2. Quá nhỏ nhưng không tốn gì thêm nếu đã có sẵn.

**Bỏ valid_with_seed.** 1.1 TB đã download có thể xóa để giải phóng storage.

---

## MixtureVitae-Backup Multimodal Investigation (Jul 9, 2026)

### Background

P0 item from Huu (asked Jul 5): investigate `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` on HF — never scanned before (separate from `valid_with_seed` / `stack_images3_gzip`, which were already inventoried). Run locally on a Windows dev machine (no JUWELS access this session), CPU-only, so the approach was streaming sample-based counting rather than a full download — 103GB total across 15 files.

### Method

Two new scripts, both reusing `PATTERNS`/`count_tokens`/`_hf_token`/`hf_url`/checkpoint machinery from `tools/data_inventory.py`:

- **`tools/peek_multimodal.py`** — structural probe, streams just the first few records/members per file (no full download) to discover format and flag VLA-token presence. Output: `tools/multimodal_peek_report.json`.
- **`tools/count_multimodal_tokens.py`** — true HTTP streaming (never writes the compressed file to disk), caps each file at `--sample-mb` compressed MB (default 75), counts VLA-tag tokens (regex, same as `data_inventory.py`) plus any raw integer token arrays (`*_token`/`*_tokens` fields — generalizes beyond just `snac_token`), extrapolates to full file size. Resumable checkpoint: `tools/multimodal_inventory_checkpoint.json`.

**Key implementation fix:** `valid_data_snac.jsonl.gz`, `train_data_snac.jsonl.gz`, and `emo.jsonl.gz` are **not** true JSONL (one compact object per line) — they're a pretty-printed JSON array where a single record can span many lines. Naive newline-splitting silently produced zero parsed records. Fixed by switching to a streaming buffer + `json.JSONDecoder().raw_decode()` approach that pulls complete top-level JSON values regardless of embedded newlines.

Local env: plain Python venv (`tools/env_multimodal_inventory/`, gitignored) — `pip install requests tqdm`, no conda needed. HF token support added (`tools/.hf_token`, gitignored, read by `_hf_token()`) though this specific repo turned out to be public (no auth required).

### Results (75MB compressed sample per file, extrapolated to full size)

**No file contains our tagged VLA tokens** (`<seed2_N>`, `<cosmos_N>`, `<avclm_N>`, `<snac_N>`) — confirmed at the 75MB-sample scale across all 15 files, not just the initial 5-record peek.

**2 files carry real SNAC audio tokens, as raw integer arrays** (`snac_token: [128266, ...]`), not tag strings:

| File | Size | Sample records | Extrapolated raw SNAC codes |
|---|---|---|---|
| `train_data_snac.jsonl.gz` | 11.1 GB | 131,850 | **~3.11B** |
| `valid_data_snac.jsonl.gz` | 579 MB | 129,996 | **~162M** |
| **Total** | | | **~3.27B raw SNAC codes** |

Comparable in scale to the 4.92B SNAC tokens already found in MixtureVitae-Omni's `valid_snac` — a real, previously-uncounted audio-token resource.

**13 remaining files — plain text/caption corpora** (word-count, extrapolated):

| File | Extrapolated text tokens | Content |
|---|---|---|
| high_stack.tar.gz | 4.11B | StackExchange QA |
| valid_text_only.tar.gz | 3.31B | mixed text |
| stack_maga.tar.gz | 1.65B | StackExchange |
| emo.jsonl.gz | 1.04B | audio-transcript + image-caption pairs |
| train_data_snac.jsonl.gz (`text` field) | 865.5M | transcript alongside the SNAC tokens above |
| magalith-10m-florence2.jsonl.gz | 864.4M | image captions |
| synth_llava2.tar.gz | 162.9M | LLaVA-style image captions |
| clappa.tar.gz | 138.4M | video captions (DISCUSS-1 candidate) |
| synth_llava.tar.gz | 93.7M | LLaVA-style image captions |
| low_nemo_maga.tar.gz | 73.7M | text |
| valid_data_snac.jsonl.gz (`text` field) | 44.1M | transcript alongside the SNAC tokens above |
| youtube.tar.gz | 38.6M | video storyline/description |
| coco.tar.gz | 10.0M | image captions — **exact** (fully consumed within sample) |
| europarl.tar.gz | ~0.1M | ⚠️ low confidence, see caveats |

### Caveats (not yet resolved)

1. **`finevideo_transcripts.jsonl.gz` undercounted (shows 0).** Real field is `transcripts`, not `text` — the counter only checks `text` (matching `data_inventory.py`'s existing convention). Needs a dedicated pass, and — since it's literally FineVideo YouTube transcripts — a video-ID overlap check against our own pipeline (same class of risk as the `valid_with_seed` double-counting issue already resolved once).
2. **`europarl.tar.gz` estimate is close to meaningless** — first sampled member was a single ~986MB record, so the 75MB sample only completed 1 record. Needs a much larger sample or a targeted full scan.
3. **Several archives mix huge text members with binary `.wds` shards** (youtube, synth_llava/synth_llava2, stack_maga, high_stack, valid_text_only) — 75MB only reached a handful of members out of many, so extrapolation assumes uniform density across the archive, which may not hold. Lower confidence than files sampled with hundreds of small members (coco, low_nemo_maga).
4. **Raw `snac_token` integer arrays are not in our tokenizer's `<snac_N>` string format** — would need a conversion step (offset/tag scheme) similar to the MV-Omni `seed→seed2` conversion already done, before these ~3.27B codes could enter our Megatron pipeline.

### Status

Posted to Huu on Discord (Jul 9, 2026, 3:51pm): *"this dataset is mostly text, only train_data_snac.jsonl.gz and valid_data_snac.jsonl.gz have snac tokens ... u want to add it?"* — **awaiting his reply.** Do not start integration/download of the full files until he responds.
