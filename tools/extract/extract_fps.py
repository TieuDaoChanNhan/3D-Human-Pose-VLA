"""
Extract native FPS for every video in videos_staging/ and write
outputs/fps_lookup.json  →  { video_id: fps_float, ... }

Run from repo root:
    python tools/extract/extract_fps.py \
        --video-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/shared_window_independent/videos_staging \
        --output    outputs/fps_lookup.json \
        --workers   32
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2


def get_fps(video_path: str):
    video_id = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = cap.get(cv2.CAP_PROP_POS_MSEC)  # not reliable; use frame count fallback
    cap.release()

    if fps <= 0 or fps > 240:
        # Corrupted header — mark as unknown; pipeline will skip or default to 30
        return video_id, None, n_frames

    return video_id, round(float(fps), 6), n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    video_paths = sorted(
        str(p) for p in Path(args.video_dir).iterdir()
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi"}
    )
    print(f"Found {len(video_paths)} videos in {args.video_dir}")

    results = {}
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(get_fps, p): p for p in video_paths}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                vid_id, fps, n_frames = fut.result()
                results[vid_id] = fps
                if fps is None:
                    failed.append(vid_id)
            except Exception as e:
                path = futures[fut]
                vid_id = Path(path).stem
                results[vid_id] = None
                failed.append(vid_id)
            if done % 1000 == 0:
                print(f"  {done}/{len(video_paths)} processed ...", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    valid = sum(1 for v in results.values() if v is not None)
    print(f"\nDone. {valid}/{len(results)} videos with valid FPS.")
    if failed:
        print(f"  {len(failed)} videos had no readable FPS (stored as null): {failed[:5]} ...")
    print(f"Written to {args.output}")

    # Print FPS distribution
    from collections import Counter
    fps_counts = Counter(round(v) for v in results.values() if v is not None)
    print("\nFPS distribution (rounded):")
    for fps_val, count in sorted(fps_counts.items()):
        print(f"  {fps_val} fps: {count} videos")


if __name__ == "__main__":
    main()
