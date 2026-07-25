#!/usr/bin/env python3
"""
SNAC (listen-format) tokenization for OmniVideo-100K.

Simpler than pipeline_pose/snac_finevideo.py: OmniVideo-100K has no
scenes/activities nesting (flat one-record-per-video schema, see
step_a/flatten_step_a_video.py's docstring) -- every video is one
continuous stream chunked at CHUNK_SIZE=24 frames (window=24 pivot,
2026-07-23, see step_a/step_a_tokenize_video.py). So instead of scanning a
merged dataset for per-activity chunk_timing, this script recomputes
n_chunks directly from each video's `duration` field in
omnivideo_100k_segment_captions.jsonl -- the exact same source and formula
(`total_frames = round(duration*30); n_chunks = ceil(total_frames/CHUNK_SIZE)`)
step_a_tokenize_video.py uses, so chunk_idx stays aligned between the two
pipelines without needing to scan Step A's own output.

`<listen>` (not `<speak>`) is correct here -- this is the video's own
ambient/background audio the model perceives, same role as FineVideo's
audio (see PROGRESS_VI.md 2026-07-23 "listen vs speak" design decision).
Reuses pipeline_pose/snac_finevideo.py's audio-extraction/SNAC-encoding
functions directly (import, not copy-paste) to avoid drift between the
two pipelines.

Output: {OUTPUT_DIR}/{video_id}_snac.jsonl (one line per video)
    {"video_id": "...", "snac_by_chunk": {"0": ["<snac_N>", ...], "1": [...], ...}}

Usage:
    python data_prep/omnivideo_100k/snac_omnivideo.py
    (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT shard the video list, same
    convention as snac_finevideo.py and every other driver in this repo.)
"""
import argparse
import json
import logging
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline_pose.snac_finevideo import (  # noqa: E402
    extract_full_audio, slice_audio, encode_listen, split_snac_by_chunks,
    SAMPLE_RATE, SNAC_MODEL,
)

TARGET_FPS = 30
CHUNK_SIZE = 24  # must match step_a/step_a_tokenize_video.py's CHUNK_SIZE

DATA_ROOT     = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
CAPTIONS_FILE = os.path.join(DATA_ROOT, "omnivideo_100k_segment_captions.jsonl")
VIDEO_DIR     = os.path.join(DATA_ROOT, "videos")
OUTPUT_DIR    = os.path.join(DATA_ROOT, "snac_tokens_w24")
HF_CACHE      = "/e/project1/reformo/nguyen38/jupiter_cache/huggingface"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                     datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(__name__)


def load_durations(path):
    """video_id -> duration (sec), same source step_a_tokenize_video.py reads."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vid = d.get("video_id")
            dur = d.get("duration", 0)
            if vid and dur:
                out[vid] = float(dur)
    return out


def process_video(video_id, duration_sec, model, device, video_dir, output_dir, skip_existing):
    out_path = os.path.join(output_dir, f"{video_id}_snac.jsonl")
    if skip_existing and os.path.exists(out_path):
        return {"ok": 0, "skipped_vid": 1, "failed_audio": 0, "failed_snac": 0, "tokens": 0}

    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return {"ok": 0, "skipped_vid": 0, "failed_audio": 1, "failed_snac": 0, "tokens": 0}

    full_audio = extract_full_audio(video_path)
    if full_audio is None:
        return {"ok": 0, "skipped_vid": 0, "failed_audio": 1, "failed_snac": 0, "tokens": 0}

    segment = slice_audio(full_audio, 0.0, duration_sec)
    if len(segment) < int(SAMPLE_RATE * 0.1):
        return {"ok": 0, "skipped_vid": 0, "failed_audio": 1, "failed_snac": 0, "tokens": 0}

    try:
        flat_tokens = encode_listen(segment, model, device)
    except Exception as e:
        log.warning(f"SNAC failed {video_id}: {e}")
        return {"ok": 0, "skipped_vid": 0, "failed_audio": 0, "failed_snac": 1, "tokens": 0}
    if not flat_tokens:
        return {"ok": 0, "skipped_vid": 0, "failed_audio": 0, "failed_snac": 1, "tokens": 0}

    total_frames = max(1, round(duration_sec * TARGET_FPS))
    n_chunks = math.ceil(total_frames / CHUNK_SIZE)
    by_chunk = split_snac_by_chunks(flat_tokens, n_chunks)
    snac_by_chunk = {str(k): v for k, v in by_chunk.items()}

    with open(out_path, "w") as f:
        f.write(json.dumps({"video_id": video_id, "snac_by_chunk": snac_by_chunk}) + "\n")

    return {"ok": 1, "skipped_vid": 0, "failed_audio": 0, "failed_snac": 0, "tokens": len(flat_tokens)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-file", default=CAPTIONS_FILE)
    ap.add_argument("--video-dir",     default=VIDEO_DIR)
    ap.add_argument("--output-dir",    default=OUTPUT_DIR)
    ap.add_argument("--hf-cache",      default=HF_CACHE)
    ap.add_argument("--no-skip",       action="store_true")
    args = ap.parse_args()
    skip_existing = not args.no_skip

    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.makedirs(args.hf_cache,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    durations = load_durations(args.captions_file)
    all_vids = sorted(durations.keys())

    task_id   = int(os.environ.get("SLURM_ARRAY_TASK_ID",   "0"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    my_vids   = all_vids[task_id::num_tasks]
    log.info(f"Task {task_id}/{num_tasks}: {len(my_vids)}/{len(all_vids)} videos  skip_existing={skip_existing}")

    from snac import SNAC
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading SNAC model ({SNAC_MODEL}) on {device}...")
    t_load = time.time()
    os.environ["HF_HOME"] = args.hf_cache
    model = SNAC.from_pretrained(SNAC_MODEL, local_files_only=True).eval().to(device)
    log.info(f"SNAC loaded ({time.time()-t_load:.1f}s)")

    cumul = {"ok": 0, "skipped_vid": 0, "failed_audio": 0, "failed_snac": 0, "tokens": 0}
    t_start = time.time()
    for idx, vid in enumerate(my_vids, 1):
        s = process_video(vid, durations[vid], model, device, args.video_dir, args.output_dir, skip_existing)
        for k in cumul:
            cumul[k] += s[k]
        if idx % 200 == 0 or idx == len(my_vids):
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(my_vids) - idx) / rate if rate > 0 else 0
            log.info(f"[{idx:5d}/{len(my_vids)}] ok={cumul['ok']} skip={cumul['skipped_vid']} "
                      f"fail_audio={cumul['failed_audio']} fail_snac={cumul['failed_snac']} "
                      f"rate={rate:.1f}vid/s ETA={eta/60:.0f}m total_tokens={cumul['tokens']:,}")

    log.info(f"DONE task {task_id}: {cumul}")


if __name__ == "__main__":
    main()
