"""Phase 4 (YOLO person-presence cleaning) for the sports subset of
OmniVideo-100K, run on JUPITER. Depends on Phase 3
(phase3_kinematics_omnivideo.py) having written
$DATA/omnivideo_100k/pose_states_jsonl_30fps_w24/{video_id}_states.jsonl.

2026-07-23: window=24 pivot to match FineVideo-VLA (was WINDOW_SIZE=8,
imported from pipeline_pose.phase4_yolo_cleaner's default; input/output
dirs were pose_states_jsonl_30fps / pose_yolo_cleaned_30fps) -- see
step_a/step_a_tokenize_video.py's CHUNK_SIZE comment for the full rationale.

Does not modify pipeline_pose/phase4_yolo_cleaner.py -- reuses its
dataset-agnostic building blocks (WindowRecord/RunMetrics dataclasses,
model loading/warmup, batched YOLO inference, frame-cache cleanup) via
import, but process_video()/load_windows_and_needed_frames() are
rewritten here to fix a real frame-index bug found while checking the
original against real data (see investigation notes, session of
2026-07-20):

    The original decodes video frames sequentially from the *native-fps*
    source file and uses the raw decode-order index directly as the
    dict key into frame_cache -- but window_id in states_jsonl_30fps is
    indexed on the *resampled-30fps* grid produced by Phase 2.5. For any
    video whose native fps != 30 (measured: 35% of FineVideo, 37.3% of
    this OmniVideo-100K subset), those two index spaces are different
    timelines. Concretely, on a 25fps video verified against real
    production output (-2MKTg-LNio): native_frames=12,758 but the
    30fps-space window_ids run up to 15,304 -- so (a) every window past
    native_frames is silently dropped (loses the last ~1/6 of the video
    for a 25fps source), and (b) every window that IS kept reads YOLO's
    person-presence result from the wrong point in time, drifting by up
    to ~20% of the video's duration by the end. This is a pre-existing,
    previously-unflagged issue in the already-completed FineVideo Phase 4
    run (out of scope to fix retroactively here -- that data is already
    used for training) -- it is not repeated in this driver.

    Fix: build an explicit native_idx <-> resampled_idx mapping via
    np.round(np.linspace(0, N-1, M)) (N = native frame count, M =
    resampled frame count read directly from the corresponding
    pose_3d_npy_30fps/{video_id}.npy shape -- the exact M Phase 2.5
    produced, not recomputed/guessed) -- the same endpoint-aligned
    linspace mapping resample_pose() (phase2_5_resample_30fps.py) used
    going the other direction. Frames are still decoded sequentially
    (no random seeks, same performance design as the original), but each
    decoded native frame's YOLO result is written into frame_cache under
    every resampled_idx that maps back to it, so windows are always keyed
    and read in the same (resampled/window_id) timeline they were defined
    in. For 30fps-native videos the mapping is the identity and behaviour
    matches the original exactly.

Also differs from the original in iteration/output location, same
reasoning as every other driver in this directory: iterates the known
1,126-video subset list (not a directory glob) and writes to its own
directory under the OmniVideo-100K data root, not the shared outputs/ tree.

Output: $DATA/omnivideo_100k/pose_yolo_cleaned_30fps_w24/{video_id}_cleaned.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Set, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pipeline_pose.phase4_yolo_cleaner import (  # noqa: E402
    WindowRecord, RunMetrics, PERSON_CLASS_ID, EMPTY_FRAME_CUTOFF,
    DEFAULT_THRESHOLD, DEFAULT_MODEL, DEFAULT_BATCH_SIZE, DEFAULT_IMGSZ,
    choose_device, load_model, warmup_model, infer_frame_batch, get_video_frame_count,
)

# 2026-07-23: window=24 to match FineVideo-VLA's pivot (was 8, imported from
# pipeline_pose.phase4_yolo_cleaner's default) -- see
# step_a/step_a_tokenize_video.py's CHUNK_SIZE comment for the full rationale.
WINDOW_SIZE = 24

DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
DEFAULT_VIDEO_IDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sports_subset_video_ids_filtered.txt")
DEFAULT_VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
DEFAULT_INPUT_DIR = os.path.join(DATA_ROOT, "pose_states_jsonl_30fps_w24")
DEFAULT_RESAMPLED_NPY_DIR = os.path.join(DATA_ROOT, "pose_3d_npy_30fps")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_ROOT, "pose_yolo_cleaned_30fps_w24")

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))


def build_native_resampled_maps(native_frames: int, resampled_frames: int):
    """Endpoint-aligned linspace mapping, inverse of resample_pose()'s own
    linspace(0,1,N) <-> linspace(0,1,M). Returns (native_for_resampled,
    resampled_for_native): the first is an array of length M giving the
    matching native frame index for each resampled index; the second is
    the reverse lookup (native_idx -> list of resampled indices)."""
    if resampled_frames < 1:
        return np.zeros(0, dtype=np.int64), defaultdict(list)
    native_for_resampled = np.round(
        np.linspace(0, max(native_frames - 1, 0), resampled_frames)
    ).astype(np.int64)
    resampled_for_native: Dict[int, List[int]] = defaultdict(list)
    for j, native_idx in enumerate(native_for_resampled):
        resampled_for_native[int(native_idx)].append(j)
    return native_for_resampled, resampled_for_native


def load_windows_and_needed_resampled_frames(
    jsonl_path: Path, resampled_frames: int,
) -> Tuple[List[WindowRecord], Set[int]]:
    """Same contract as the original load_windows_and_needed_frames(), but
    the out-of-range bound is checked against the resampled-space frame
    count (what window_id is actually indexed in), not the native one."""
    grouped: Dict[int, List[WindowRecord]] = defaultdict(list)
    skipped = 0
    total = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            payload = json.loads(line)
            if "window_id" not in payload:
                raise KeyError(f"Missing 'window_id' at line {line_no}")
            start_frame = payload["window_id"]
            end_frame = start_frame + WINDOW_SIZE - 1
            if start_frame < 0 or end_frame >= resampled_frames:
                skipped += 1
                continue
            grouped[start_frame].append(WindowRecord(start_frame=start_frame, raw_line=line))

    if total == 0:
        raise RuntimeError(f"Input JSONL is empty: {jsonl_path}")
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} invalid/out-of-range JSONL lines.")

    all_records: List[WindowRecord] = []
    needed_frames: Set[int] = set()
    for start_frame in sorted(grouped.keys()):
        records = grouped[start_frame]
        all_records.extend(records)
        for frame_idx in range(start_frame, start_frame + WINDOW_SIZE):
            needed_frames.add(frame_idx)

    return all_records, needed_frames


def try_resolve_ready_windows(pending_windows: Deque[WindowRecord],
                               frame_cache: Dict[int, bool], fout) -> Tuple[int, int]:
    cut = 0
    keep = 0
    while pending_windows:
        record = pending_windows[0]
        start = record.start_frame
        end = start + WINDOW_SIZE - 1
        values: List[bool] = []
        ready = True
        for frame_idx in range(start, end + 1):
            value = frame_cache.get(frame_idx)
            if value is None:
                ready = False
                break
            values.append(value)
        if not ready:
            break
        pending_windows.popleft()
        empty_count = sum(1 for v in values if not v)
        if empty_count >= EMPTY_FRAME_CUTOFF:
            cut += 1
        else:
            fout.write(record.raw_line + "\n")
            keep += 1
    return cut, keep


def cleanup_frame_cache(frame_cache: Dict[int, bool], pending_windows: Deque[WindowRecord]) -> None:
    if not pending_windows:
        frame_cache.clear()
        return
    earliest_needed_frame = pending_windows[0].start_frame
    for k in [k for k in frame_cache if k < earliest_needed_frame]:
        del frame_cache[k]


def process_video(video_path: Path, jsonl_in: Path, resampled_npy_path: Path,
                   jsonl_out: Path, model, threshold: float, batch_size: int,
                   imgsz: int) -> Tuple[int, int, int, RunMetrics]:
    native_frames = get_video_frame_count(video_path)
    resampled_frames = int(np.load(resampled_npy_path, mmap_mode="r").shape[0])

    all_records, needed_resampled_frames = load_windows_and_needed_resampled_frames(
        jsonl_in, resampled_frames=resampled_frames,
    )

    total_input = len(all_records)
    if total_input == 0:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w", encoding="utf-8"):
            pass
        return 0, 0, 0, RunMetrics(0.0, 0, 0, 0.0)

    native_for_resampled, resampled_for_native = build_native_resampled_maps(
        native_frames, resampled_frames,
    )
    # Only native frames that map to at least one *needed* resampled index are worth decoding-into-batch.
    needed_native_frames = {
        native_idx for native_idx, resampled_idxs in resampled_for_native.items()
        if any(j in needed_resampled_frames for j in resampled_idxs)
    }
    last_needed_native_frame = max(needed_native_frames) if needed_native_frames else -1

    pending_windows: Deque[WindowRecord] = deque(all_records)
    frame_cache: Dict[int, bool] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    device = choose_device()
    batch_frames: List[np.ndarray] = []
    batch_native_indices: List[int] = []
    total_cut = 0
    total_keep = 0
    video_read_frames = 0
    frames_sent_to_yolo = 0
    yolo_infer_time_sec = 0.0

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    def flush_batch(fout) -> Tuple[int, int]:
        nonlocal batch_frames, batch_native_indices, frames_sent_to_yolo, yolo_infer_time_sec
        if not batch_frames:
            return 0, 0
        infer_start = time.perf_counter()
        person_flags = infer_frame_batch(model, batch_frames, threshold, device, imgsz)
        yolo_infer_time_sec += time.perf_counter() - infer_start
        frames_sent_to_yolo += len(batch_frames)

        for native_idx, has_person in zip(batch_native_indices, person_flags):
            for resampled_idx in resampled_for_native.get(native_idx, []):
                frame_cache[resampled_idx] = has_person

        cut, keep = try_resolve_ready_windows(pending_windows, frame_cache, fout)
        cleanup_frame_cache(frame_cache, pending_windows)
        batch_frames = []
        batch_native_indices = []
        return cut, keep

    try:
        with jsonl_out.open("w", encoding="utf-8") as fout:
            native_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                video_read_frames += 1
                if native_idx > last_needed_native_frame:
                    break
                if native_idx not in needed_native_frames:
                    native_idx += 1
                    continue

                batch_frames.append(frame)
                batch_native_indices.append(native_idx)
                if len(batch_frames) >= batch_size:
                    cut, keep = flush_batch(fout)
                    total_cut += cut
                    total_keep += keep
                native_idx += 1

            if batch_frames:
                cut, keep = flush_batch(fout)
                total_cut += cut
                total_keep += keep
    finally:
        cap.release()

    total_time_sec = time.perf_counter() - start_time

    if pending_windows:
        raise RuntimeError(
            f"{len(pending_windows)} windows could not be resolved for {video_path} "
            "-- required native frames were not decoded/inferred."
        )

    metrics = RunMetrics(total_time_sec, video_read_frames, frames_sent_to_yolo, yolo_infer_time_sec)
    return total_input, total_cut, total_keep, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids-file", default=DEFAULT_VIDEO_IDS_FILE)
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--resampled-npy-dir", default=DEFAULT_RESAMPLED_NPY_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), DEFAULT_MODEL))
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = ap.parse_args()

    with open(args.video_ids_file) as f:
        video_ids = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    my_ids = video_ids[RANK::WORLD_SIZE]

    device = f"cuda:{LOCAL_RANK}" if os.environ.get("SLURM_LOCALID") is not None else choose_device()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(LOCAL_RANK))
    print(f"[Rank {RANK}/{WORLD_SIZE}] {len(my_ids)}/{len(video_ids)} videos assigned, device={device}")

    model = load_model(args.model, choose_device())
    warmup_model(model, choose_device(), args.imgsz)

    n_done = n_skip = n_no_input = n_error = 0
    for i, video_id in enumerate(my_ids):
        output_path = Path(args.output_dir) / f"{video_id}_cleaned.jsonl"
        if output_path.exists():
            n_skip += 1
            continue

        video_path = Path(args.videos_dir) / f"{video_id}.mp4"
        jsonl_in = Path(args.input_dir) / f"{video_id}_states.jsonl"
        resampled_npy = Path(args.resampled_npy_dir) / f"{video_id}.npy"
        if not (video_path.exists() and jsonl_in.exists() and resampled_npy.exists()):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR: missing input for {video_id}")
            n_no_input += 1
            continue

        temp_path = Path(f"{output_path}.tmp_rank{RANK}")
        try:
            total_input, total_cut, total_keep, metrics = process_video(
                video_path, jsonl_in, resampled_npy, temp_path,
                model, args.threshold, args.batch_size, args.imgsz,
            )
            os.replace(temp_path, output_path)  # same dir -> atomic, safe for resume
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) OK: {video_id} "
                  f"windows={total_input} kept={total_keep} cut={total_cut}")
            n_done += 1
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR {video_id}: {e}")
            n_error += 1

    print(f"[Rank {RANK}] DONE. done={n_done} skip={n_skip} no_input={n_no_input} error={n_error}")


if __name__ == "__main__":
    main()
