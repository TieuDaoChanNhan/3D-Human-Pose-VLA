#!/usr/bin/env python3
"""
Phase 2.5-equivalent for Harmony4D: resample each (sequence, person) H36M-17
pose track from native 20fps to this project's standard 30fps grid.

Mirrors pipeline_pose/phase2_5_resample_30fps.py's method (linear
interpolation along the time axis -- chosen there, and here, because linear
passes cleanly through NaN-adjacent segments without the overshoot cubic/PCHIP
would introduce; PCHIP is the right tool downstream in Phase 5, after NaN
frames are already handled). Two differences from that script, both because
Harmony4D's native frame indices are not guaranteed contiguous (some frames
may be missing upstream), unlike this project's own MotionBERT output:
  - resamples on real timestamps (`frame_idx / 20.0` seconds) read from each
    sequence's `<person_id>_frame_idx.npy`, not an assumed 0..N-1 grid
  - operates on (T, 17, 4) arrays (xyz + confidence, see
    convert_coco_to_h36m.py) instead of (T, 17, 3) -- confidence is
    interpolated the same as xyz; a NaN xyz value already marks a
    missing/low-confidence joint at that frame and NaN propagates through
    linear interpolation into any output frame whose window touches it,
    exactly like the zero-frame handling in the original script's docstring
    intends (except NaN here vs zero-fill there -- see
    convert_coco_to_h36m.py's confidence-handling note for why NaN).

Per Van Khue's decision (21/07/2026): agent tokens stay single-person, same
schema as the existing FineVideo pipeline (no multi-person tag). Harmony4D's
2 people per sequence are NOT merged into one record -- each (sequence,
person) pair is resampled independently and stays independent through every
later step, i.e. one Harmony4D sequence yields 2 separate single-person
tracks, doubling usable count to 416. This matches FineVideo's own existing
precedent of tracking only one person's pose per video even when others
appear on screen -- not a new inconsistency.

Input:  harmony4d_h36m_native20fps/<category>/<seq_id>/<person_id>.npy            (T,17,4)
        harmony4d_h36m_native20fps/<category>/<seq_id>/<person_id>_frame_idx.npy  (T,)
Output: harmony4d_h36m_30fps/<category>/<seq_id>/<person_id>.npy                  (M,17,4)

Resumable: skips a (sequence, person) if its output .npy already exists.

Usage:
    python3 data_prep/harmony4d/resample_30fps.py \
        --input-dir  /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_native20fps \
        --output-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_30fps
"""
import argparse
import glob
import os

import numpy as np
from scipy.interpolate import interp1d

NATIVE_FPS = 20.0
TARGET_FPS = 30.0


def resample_track(arr: np.ndarray, frame_idx: np.ndarray, native_fps: float = NATIVE_FPS,
                    dst_fps: float = TARGET_FPS) -> np.ndarray:
    """arr: (T, 17, 4) [xyz, confidence]. frame_idx: (T,) native frame indices (may have gaps)."""
    T = arr.shape[0]
    if T < 2:
        return arr

    t_src = frame_idx.astype(np.float64) / native_fps  # real seconds
    duration = t_src[-1] - t_src[0]
    M = max(2, round(duration * dst_fps) + 1)
    t_dst = np.linspace(t_src[0], t_src[-1], M)

    flat = arr.reshape(T, -1).astype(np.float64)
    f = interp1d(t_src, flat, axis=0, kind="linear", assume_sorted=True)
    return f(t_dst).astype(np.float32).reshape(M, 17, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",
                    default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_native20fps")
    ap.add_argument("--output-dir",
                    default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_30fps")
    args = ap.parse_args()

    pose_files = sorted(glob.glob(os.path.join(args.input_dir, "*", "*", "*.npy")))
    pose_files = [f for f in pose_files if not f.endswith("_frame_idx.npy")]

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    my_files = pose_files[task_id - 1::num_tasks]

    print(f"[task {task_id}/{num_tasks}] {len(my_files)}/{len(pose_files)} person-tracks assigned", flush=True)

    done = skipped = 0
    for pose_path in my_files:
        seq_dir = os.path.dirname(pose_path)
        category, seq_id = seq_dir.split(os.sep)[-2:]
        person_id = os.path.basename(pose_path).replace(".npy", "")
        idx_path = os.path.join(seq_dir, f"{person_id}_frame_idx.npy")

        out_dir = os.path.join(args.output_dir, category, seq_id)
        out_path = os.path.join(out_dir, f"{person_id}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue

        arr = np.load(pose_path)
        frame_idx = np.load(idx_path)
        resampled = resample_track(arr, frame_idx)

        os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, resampled)
        done += 1

    print(f"[task {task_id}/{num_tasks}] done: {done} resampled, {skipped} already existed", flush=True)


if __name__ == "__main__":
    main()
