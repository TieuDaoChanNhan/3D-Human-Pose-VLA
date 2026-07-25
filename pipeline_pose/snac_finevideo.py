#!/usr/bin/env python3
"""
SNAC tokenization for FineVideo-VLA activities.

Reads final_dataset_adaptive JSONL files, extracts audio from .mp4 videos,
and tokenizes each activity segment with SNAC_24kHz in "listen" format.

Listen format — 3 tokens per SNAC base frame (base rate = 12.5 Hz → 37.5 tokens/sec):
    token_1 = codes[0][i]       + 128266        →  <snac_128266> .. <snac_132361>
    token_2 = codes[1][2*i]     + 128266 + 4096 →  <snac_132362> .. <snac_136457>
    token_3 = codes[1][2*i+1]   + 128266 +16384 →  <snac_144650> .. <snac_148745>

Same Orpheus offset scheme as MixtureVitae-Omni → tokens are directly compatible.
Total unique SNAC token strings: 3 × 4096 = 12,288.

Output: {OUTPUT_DIR}/{video_id}_snac.jsonl
    One line per activity:
    {
      "video_id":"...", "activity_id":"...", "start_sec":1.0, "end_sec":8.9,
      "has_agent": true,
      "snac_by_chunk": {
        "0": ["<snac_130055>", "<snac_133001>", "<snac_145000>", ...],  // ~9-10 tokens
        "1": [...],
        ...
      }
    }

    snac_by_chunk keys are chunk_idx (integer as string), aligned to the same 8-frame
    grid as cosmos/avclm/agent. Phase7 reads chunk_idx → snac tokens directly.

Two modes:
  --build-tasks   Scan all final_dataset_adaptive files, write snac_task_list.json.
                  Run once on login node (or task 0) before the array job.
  (default)       Load task list, process this SLURM task's slice of videos.

SLURM usage:
    SLURM_ARRAY_TASK_ID   = task index (0-based)
    SLURM_ARRAY_TASK_COUNT = total number of tasks in the array

Local test:
    python pipeline_pose/snac_finevideo.py --build-tasks
    python pipeline_pose/snac_finevideo.py  # task_id=0, num_tasks=1
"""

import argparse
import glob
import json
import logging
import math
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────

VIDEO_DIR    = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/shared_window_independent/videos_staging"
# Still scanned for activity time_range_sec + has_agent -- those don't depend
# on window size. chunk_timing (per-window breakdown) is IGNORED (see
# _scan_one_rank_file docstring) because this file is window=8-based (the
# pre-pivot merge) and Phase 6 hasn't rerun at window=24 yet; n_chunks is
# instead recomputed independently from CHUNK_SIZE below, same pattern as
# data_prep/omnivideo_100k/snac_omnivideo.py.
INPUT_GLOB   = ("/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/FineVideo-VLA/"
                "final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl")
OUTPUT_DIR   = ("/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/FineVideo-VLA/snac_tokens_w24")
TASK_CACHE   = ("/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/FineVideo-VLA/"
                "snac_task_list_w24.json")
HF_CACHE     = "/e/project1/reformo/nguyen38/jupiter_cache/huggingface"
SNAC_MODEL   = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE  = 24000
TARGET_FPS   = 30
CHUNK_SIZE   = 24  # 2026-07-23 window=24 pivot -- must match step_a_tokenize_video.py's CHUNK_SIZE

# ── SNAC listen-format offsets (matches MixtureVitae-Omni) ───────────────────

OFFSET_L0  = 128266            # codes[0] base
OFFSET_L1A = 128266 + 4096     # codes[1] even frames   → 132362
OFFSET_L1B = 128266 + 4 * 4096 # codes[1] odd frames   → 144650

# 2026-07-23: speak-format offsets for codes[2] (fine, 50Hz, 4 sub-positions
# per base frame) -- corrected to match the REAL scheme in Huu/Chien's
# production snac_gpu.py on Leonardo (pipeline_video/snac_gpu.py), the
# Orpheus-standard SNAC packing layout. Sub-codes 0/1 sit between L1A and
# L1B (in what was wrongly assumed to be an unused gap); sub-codes 2/3 sit
# after L1B. See data_prep/laion_emotional_roleplay/tokenize_snac.py's
# OFFSET_L2 docstring for the full correction history. Must match exactly.
OFFSET_L2 = [136458, 140554, 148746, 152842]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Audio extraction
# ─────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    """Return path to ffmpeg binary, trying imageio_ffmpeg as fallback."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError("ffmpeg not found. Install ffmpeg or imageio_ffmpeg.")

_FFMPEG = None

def get_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = _find_ffmpeg()
    return _FFMPEG


def extract_full_audio(video_path: str) -> np.ndarray | None:
    """
    Extract full mono 24 kHz PCM audio from a video file.
    Pipes raw float32 PCM from ffmpeg directly to a numpy array — no temp files.
    Returns float32 array or None on any failure.
    """
    cmd = [
        get_ffmpeg(), "-y",
        "-i", video_path,
        "-vn",                     # strip video
        "-ac", "1",                # mono
        "-ar", str(SAMPLE_RATE),   # 24 kHz
        "-f",  "f32le",            # raw float32 PCM
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0 or not result.stdout:
            return None
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        return audio if len(audio) > 0 else None
    except Exception as e:
        log.debug(f"ffmpeg failed for {video_path}: {e}")
        return None


def slice_audio(
    audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Slice a float32 audio array to [start_sec, end_sec]."""
    s = max(0, int(start_sec * sr))
    e = min(len(audio), int(end_sec * sr))
    return audio[s:e]


# ─────────────────────────────────────────────────────────────────────────────
# SNAC tokenization
# ─────────────────────────────────────────────────────────────────────────────

def encode_listen(audio: np.ndarray, model, device: str) -> list[str]:
    """
    Encode a float32 audio array with SNAC_24kHz, return listen-format tokens.

    Listen format (3 tokens per base frame):
        <snac_{codes[0][i] + 128266}>
        <snac_{codes[1][2i]   + 132362}>
        <snac_{codes[1][2i+1] + 144650}>

    SNAC_24kHz hierarchy:
        codes[0] — base   codebook, 12.5 Hz
        codes[1] — mid    codebook, 25.0 Hz  (2× codes[0])
        codes[2] — fine   codebook, 50.0 Hz  (4× codes[0], not used in listen)

    Listen format ignores codes[2] (fine detail) to keep token count low
    (~37.5 tokens/sec vs 87.5 for full speak format). Matches MV-Omni.
    """
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
    with torch.inference_mode():
        codes = model.encode(tensor)  # list: [codes[0], codes[1], codes[2]]

    c0 = codes[0]  # (1, N0)
    c1 = codes[1]  # (1, N1), N1 == 2*N0

    n0 = c0.shape[1]
    tokens: list[str] = []
    for i in range(n0):
        i1a = 2 * i
        i1b = 2 * i + 1
        if i1b >= c1.shape[1]:
            break  # boundary guard: shouldn't happen for valid audio
        tokens.append(f"<snac_{c0[0, i].item() + OFFSET_L0}>")
        tokens.append(f"<snac_{c1[0, i1a].item() + OFFSET_L1A}>")
        tokens.append(f"<snac_{c1[0, i1b].item() + OFFSET_L1B}>")
    return tokens


def encode_speak(audio: np.ndarray, model, device: str) -> list[str]:
    """
    Encode a float32 audio array with SNAC_24kHz, return full speak-format
    tokens (7 tokens per base frame, +133% vs encode_listen()) -- 2026-07-22
    (REPORT.md #37), decided after a real audio A/B
    (tools/snac_l2_experiment.py) showed audibly better reconstruction.

    Speak format (2026-07-23, Leo-matched order -- L2 interleaved between
    L1a and L1b, not appended after):
        <snac_{codes[0][i]     + OFFSET_L0}>
        <snac_{codes[1][2i]    + OFFSET_L1A}>
        <snac_{codes[2][4i]    + OFFSET_L2[0]}>
        <snac_{codes[2][4i+1]  + OFFSET_L2[1]}>
        <snac_{codes[1][2i+1]  + OFFSET_L1B}>
        <snac_{codes[2][4i+2]  + OFFSET_L2[2]}>
        <snac_{codes[2][4i+3]  + OFFSET_L2[3]}>
    """
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = model.encode(tensor)

    c0, c1, c2 = codes[0], codes[1], codes[2]
    n0 = c0.shape[1]
    tokens: list[str] = []
    for i in range(n0):
        i1a, i1b = 2 * i, 2 * i + 1
        i2 = [4 * i + k for k in range(4)]
        if i1b >= c1.shape[1] or i2[-1] >= c2.shape[1]:
            break
        tokens.append(f"<snac_{c0[0, i].item() + OFFSET_L0}>")
        tokens.append(f"<snac_{c1[0, i1a].item() + OFFSET_L1A}>")
        tokens.append(f"<snac_{c2[0, i2[0]].item() + OFFSET_L2[0]}>")
        tokens.append(f"<snac_{c2[0, i2[1]].item() + OFFSET_L2[1]}>")
        tokens.append(f"<snac_{c1[0, i1b].item() + OFFSET_L1B}>")
        tokens.append(f"<snac_{c2[0, i2[2]].item() + OFFSET_L2[2]}>")
        tokens.append(f"<snac_{c2[0, i2[3]].item() + OFFSET_L2[3]}>")
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Task list building (pre-processing step)
# ─────────────────────────────────────────────────────────────────────────────

def _scan_one_rank_file(fpath: str) -> dict:
    """
    Scan one final_dataset_adaptive rank file for activity time boundaries.
    Returns {video_id: [activity_dict, ...]} for ALL activities with a valid
    time_range_sec.

    Each activity dict: activity_id, start_sec, end_sec, has_agent.

    2026-07-23: no longer reads chunk_timing (this file is the pre-window=24-
    pivot merge; its chunk_timing reflects the OLD 8-frame grid). n_chunks is
    instead recomputed in process_video() from (end_sec-start_sec) and the
    current CHUNK_SIZE=24 -- same independent-recompute pattern
    data_prep/omnivideo_100k/snac_omnivideo.py uses, so this script no longer
    needs to wait on Phase 6 merge to re-run at window=24 before it can align
    correctly.

    We tokenize ALL activities (not just agent) because:
    - Non-agent activities have seed2+cosmos → seed2+cosmos+snac trains modality transitions
    - Agent-only activities = only 14% of total; skipping the rest wastes 86% of this GPU run
    """
    tasks: dict = {}
    try:
        with open(fpath, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = rec.get("video_id", "")
                if not vid:
                    continue
                for scene in rec.get("scenes", []):
                    for act in scene.get("activities", []):
                        tr = act.get("time_range_sec")
                        if not tr or len(tr) < 2:
                            continue
                        has_agent = "<agent>" in act.get("video_tokens", "")
                        tasks.setdefault(vid, []).append({
                            "activity_id": act.get("activity_id", ""),
                            "start_sec":   float(tr[0]),
                            "end_sec":     float(tr[1]),
                            "has_agent":   has_agent,
                        })
    except Exception as e:
        log.warning(f"Error scanning {fpath}: {e}")
    return tasks


def build_task_list(input_glob: str, cache_path: str, workers: int = 8) -> dict:
    """
    Scan all final_dataset_adaptive rank files in parallel to build a task list.
    Saves result to cache_path as JSON.
    Returns {video_id: [activity_dicts]}.

    This is I/O-heavy (~657 GB total) — use multiprocessing to parallelize.
    Estimated wall time: 5–15 min with 8 workers on shared filesystem.
    """
    rank_files = sorted(glob.glob(input_glob))
    if not rank_files:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    log.info(f"Scanning {len(rank_files)} rank files with {workers} workers...")
    t0 = time.time()

    with multiprocessing.Pool(workers) as pool:
        partial_results = pool.map(_scan_one_rank_file, rank_files)

    # merge
    all_tasks: dict = {}
    for partial in partial_results:
        for vid, acts in partial.items():
            all_tasks.setdefault(vid, []).extend(acts)

    # deduplicate activities by activity_id (in case of overlap across ranks)
    for vid in all_tasks:
        seen = set()
        deduped = []
        for act in all_tasks[vid]:
            key = act["activity_id"]
            if key not in seen:
                seen.add(key)
                deduped.append(act)
        all_tasks[vid] = deduped

    log.info(
        f"Task list built: {len(all_tasks)} videos, "
        f"{sum(len(v) for v in all_tasks.values())} activities  "
        f"({time.time()-t0:.0f}s)"
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_tasks, f)
    log.info(f"Saved task list → {cache_path}")
    return all_tasks


# ─────────────────────────────────────────────────────────────────────────────
# Chunk alignment
# ─────────────────────────────────────────────────────────────────────────────

def split_snac_by_chunks(tokens: list[str], n_chunks: int) -> dict[int, list[str]]:
    """
    Split a flat SNAC listen token list evenly across n_chunks video chunks.

    Why: SNAC rate (12.5 Hz base × 3 tokens = 37.5 tok/s) does not divide evenly
    by the video chunk rate (30fps / 8 = 3.75 Hz). Encoding the full activity once
    preserves audio context; then we split by chunk count snapping to 3-token
    boundaries (one SNAC base frame = 3 listen tokens).

    Per 8-frame chunk at 30fps (0.267s): ~3.33 SNAC base frames → 9–10 listen tokens.
    """
    n_tokens = len(tokens)
    n_base   = n_tokens // 3  # truncate to complete base frames
    tokens   = tokens[:n_base * 3]

    result: dict[int, list[str]] = {}
    for k in range(n_chunks):
        start_base = round(k       * n_base / n_chunks)
        end_base   = round((k + 1) * n_base / n_chunks)
        result[k]  = tokens[start_base * 3 : end_base * 3]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    video_id: str,
    activities: list[dict],
    model,
    device: str,
    video_dir: str,
    output_dir: str,
    skip_existing: bool,
    encode_fn=encode_listen,
) -> dict:
    """
    Tokenize all activities for one video.

    Steps:
      1. Check skip — if output file exists and skip_existing, return immediately.
      2. Extract full audio from .mp4 once (1 ffmpeg call per video).
      3. For each activity: slice audio by time_range_sec, run SNAC encode once,
         then split the flat token list across chunks (preserving audio context).
      4. Write all results to {output_dir}/{video_id}_snac.jsonl.

    Output per activity: snac_by_chunk {chunk_idx → [tokens]}
    Phase6 merge uses this directly to inject SNAC tokens per 24-frame chunk,
    aligned with the cosmos/avclm/agent tokens that fire at the same chunk
    boundaries (n_chunks recomputed from CHUNK_SIZE, not read from a file).

    Returns stats: {ok, skipped_vid, failed_audio, failed_snac, tokens}
    """
    out_path = os.path.join(output_dir, f"{video_id}_snac.jsonl")
    if skip_existing and os.path.exists(out_path):
        return {"ok": 0, "skipped_vid": len(activities), "failed_audio": 0,
                "failed_snac": 0, "tokens": 0}

    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return {"ok": 0, "skipped_vid": 0, "failed_audio": len(activities),
                "failed_snac": 0, "tokens": 0}

    # Extract full audio once
    full_audio = extract_full_audio(video_path)
    if full_audio is None:
        log.warning(f"No audio: {video_path}")
        return {"ok": 0, "skipped_vid": 0, "failed_audio": len(activities),
                "failed_snac": 0, "tokens": 0}

    stats = {"ok": 0, "skipped_vid": 0, "failed_audio": 0, "failed_snac": 0, "tokens": 0}
    rows = []

    for act in activities:
        segment = slice_audio(full_audio, act["start_sec"], act["end_sec"])
        if len(segment) < int(SAMPLE_RATE * 0.1):  # skip segments < 100 ms
            stats["failed_audio"] += 1
            continue
        try:
            flat_tokens = encode_fn(segment, model, device)
        except Exception as e:
            log.warning(f"SNAC failed {video_id}/{act['activity_id']}: {e}")
            stats["failed_snac"] += 1
            continue
        if not flat_tokens:
            stats["failed_snac"] += 1
            continue

        # Split flat token list into per-chunk dicts, aligned to the same
        # 24-frame grid as cosmos/avclm/agent. n_chunks recomputed
        # independently from CHUNK_SIZE (not from a merged file's
        # chunk_timing -- see _scan_one_rank_file docstring).
        total_frames = max(1, round((act["end_sec"] - act["start_sec"]) * TARGET_FPS))
        n_chunks = math.ceil(total_frames / CHUNK_SIZE)
        by_chunk = split_snac_by_chunks(flat_tokens, n_chunks)
        snac_by_chunk = {str(k): v for k, v in by_chunk.items()}

        rows.append({
            "video_id":      video_id,
            "activity_id":   act["activity_id"],
            "start_sec":     round(act["start_sec"], 4),
            "end_sec":       round(act["end_sec"],   4),
            "has_agent":     act.get("has_agent", False),
            "snac_by_chunk": snac_by_chunk,
        })
        stats["ok"]     += 1
        stats["tokens"] += len(flat_tokens)

    if rows:
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SNAC tokenization for FineVideo-VLA")
    p.add_argument("--build-tasks",   action="store_true",
                   help="Scan final_dataset_adaptive and write snac_task_list.json, then exit.")
    p.add_argument("--input-glob",    default=INPUT_GLOB)
    p.add_argument("--output-dir",    default=OUTPUT_DIR)
    p.add_argument("--video-dir",     default=VIDEO_DIR)
    p.add_argument("--task-cache",    default=TASK_CACHE)
    p.add_argument("--hf-cache",      default=HF_CACHE)
    p.add_argument("--scan-workers",  type=int, default=8,
                   help="CPU workers for --build-tasks scan (default 8)")
    p.add_argument("--no-skip",       action="store_true",
                   help="Re-process videos even if output file exists")
    p.add_argument("--format", choices=["listen", "speak"], default="listen",
                   help="listen = L0+L1 only (current production, 3 tok/base-frame); "
                        "speak = full L0+L1+L2 (2026-07-22, 7 tok/base-frame, +133%% tokens). "
                        "speak requires L2 tokens added to the tokenizer vocab first.")
    return p.parse_args()


def main():
    args = parse_args()
    skip_existing = not args.no_skip

    # ── Set HF cache ─────────────────────────────────────────────────────────
    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.makedirs(args.hf_cache,    exist_ok=True)
    os.makedirs(args.output_dir,  exist_ok=True)

    # ── Mode: build task list ─────────────────────────────────────────────────
    if args.build_tasks:
        build_task_list(args.input_glob, args.task_cache, workers=args.scan_workers)
        return

    # ── Mode: tokenize ────────────────────────────────────────────────────────

    # SLURM array vars
    task_id   = int(os.environ.get("SLURM_ARRAY_TASK_ID",    "0"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT",  "1"))

    # Load task list (must exist — run --build-tasks first)
    if not os.path.exists(args.task_cache):
        log.error(
            f"Task list not found: {args.task_cache}\n"
            f"Run first:  python pipeline_pose/snac_finevideo.py --build-tasks"
        )
        sys.exit(1)
    with open(args.task_cache) as f:
        all_tasks = json.load(f)

    all_vids = sorted(all_tasks.keys())
    my_vids  = all_vids[task_id::num_tasks]
    log.info(
        f"Task {task_id}/{num_tasks}: {len(my_vids)}/{len(all_vids)} videos  "
        f"skip_existing={skip_existing}"
    )

    # ── Load SNAC model ───────────────────────────────────────────────────────
    from snac import SNAC  # imported here to avoid load cost during --build-tasks
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading SNAC model ({SNAC_MODEL}) on {device}...")
    t_load = time.time()
    os.environ["HF_HOME"] = args.hf_cache  # ensure hub/ subdir is found
    model  = SNAC.from_pretrained(SNAC_MODEL,
                                  local_files_only=True).eval().to(device)
    log.info(f"SNAC loaded ({time.time()-t_load:.1f}s)")

    # ── Process videos ────────────────────────────────────────────────────────
    cumul  = {"ok": 0, "skipped_vid": 0, "failed_audio": 0, "failed_snac": 0, "tokens": 0}
    t_start = time.time()

    encode_fn = encode_speak if args.format == "speak" else encode_listen

    for idx, vid in enumerate(my_vids, 1):
        s = process_video(
            vid, all_tasks[vid], model, device,
            args.video_dir, args.output_dir, skip_existing,
            encode_fn=encode_fn,
        )
        for k in cumul:
            cumul[k] += s[k]

        if idx % 100 == 0 or idx == len(my_vids):
            elapsed = time.time() - t_start
            rate    = idx / elapsed
            eta     = (len(my_vids) - idx) / rate if rate > 0 else 0
            log.info(
                f"[{idx:5d}/{len(my_vids)}]  vid={vid}  "
                f"ok={s['ok']} skip={s['skipped_vid']} "
                f"fail_audio={s['failed_audio']} fail_snac={s['failed_snac']}  "
                f"rate={rate:.1f}vid/s  ETA={eta/60:.0f}m  "
                f"total_tokens={cumul['tokens']:,}"
            )

    elapsed = time.time() - t_start
    log.info(
        f"DONE task {task_id}:  "
        f"ok={cumul['ok']}  skipped={cumul['skipped_vid']}  "
        f"fail_audio={cumul['failed_audio']}  fail_snac={cumul['failed_snac']}  "
        f"tokens={cumul['tokens']:,}  wall={elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
