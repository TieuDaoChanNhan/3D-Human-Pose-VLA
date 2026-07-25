"""Phase 2.5 (native-fps -> 30fps resample) for the sports subset of OmniVideo-100K.

Depends on Phase 2 (phase2_motionbert_omnivideo.py) having written
$DATA/omnivideo_100k/pose_3d_npy/{video_id}.npy, and on
$DATA/omnivideo_100k/fps_lookup.json (all 5,214 OmniVideo-100K videos, from
tools/extract/extract_fps.py -- JUPITER_POSE_PILOT_TASK.md flagged the
absence of this as a real gap before this script could run at all).

Kept as its own file rather than merged into the shared outputs/fps_lookup.json
(FineVideo's, 43,751 entries) -- same reasoning as the pose_2d_json/pose_3d_npy
split below: keeps this corpus independently inspectable without touching
FineVideo's lookup table.

Does not modify pipeline_pose/phase2_5_resample_30fps.py -- that script is
already dataset-agnostic (takes --input-dir/--output-dir/--fps-json, no
FineVideo-specific paths), so the interpolation math (resample_pose) is
imported directly rather than copy-pasted. What differs here is iteration:
the original globs *all* of outputs/3d_npy/ (43,751+ files, almost all
FineVideo) to find the ~1,126 OmniVideo-100K ones that need resampling --
wasteful for a 1,126-video subset. This driver iterates the known
video_id list instead.

Output: $DATA/omnivideo_100k/pose_3d_npy_30fps/{video_id}.npy.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pipeline_pose.phase2_5_resample_30fps import resample_pose, TARGET_FPS, FPS_TOLERANCE  # noqa: E402

DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
DEFAULT_VIDEO_IDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sports_subset_video_ids_filtered.txt")
DEFAULT_INPUT_DIR = os.path.join(DATA_ROOT, "pose_3d_npy")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_ROOT, "pose_3d_npy_30fps")
DEFAULT_FPS_JSON = os.path.join(DATA_ROOT, "fps_lookup.json")

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids-file", default=DEFAULT_VIDEO_IDS_FILE)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--fps-json", default=DEFAULT_FPS_JSON)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.video_ids_file) as f:
        video_ids = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    my_ids = video_ids[RANK::WORLD_SIZE]

    with open(args.fps_json) as f:
        fps_lookup = json.load(f)

    print(f"[Rank {RANK}/{WORLD_SIZE}] {len(my_ids)}/{len(video_ids)} videos assigned")

    n_done = n_skip = n_no_fps = n_no_input = n_bad_shape = 0
    for i, video_id in enumerate(my_ids):
        out_path = os.path.join(args.output_dir, f"{video_id}.npy")
        if os.path.exists(out_path):
            n_skip += 1
            continue

        in_path = os.path.join(args.input_dir, f"{video_id}.npy")
        if not os.path.exists(in_path):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR: input npy not found: {video_id}")
            n_no_input += 1
            continue

        native_fps = fps_lookup.get(video_id)
        if not native_fps:
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR: missing/invalid fps: {video_id}")
            n_no_fps += 1
            continue

        arr = np.load(in_path)
        if arr.ndim != 3 or arr.shape[1:] != (17, 3):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR: bad shape {arr.shape}: {video_id}")
            n_bad_shape += 1
            continue

        if abs(native_fps / TARGET_FPS - 1.0) < FPS_TOLERANCE:
            resampled = arr
        else:
            resampled = resample_pose(arr, native_fps)

        tmp_path = out_path + f".tmp_rank{RANK}"
        # np.save(str_path, ...) auto-appends ".npy" if the given path doesn't already
        # end in it -- tmp_path ends in ".tmp_rankN", so save via an open file handle
        # instead (numpy leaves an explicit file object's name alone).
        with open(tmp_path, "wb") as f:
            np.save(f, resampled)
        os.rename(tmp_path, out_path)  # same directory -> atomic, safe for resume
        print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) OK: {video_id} "
              f"({arr.shape[0]}@{native_fps:.2f}fps -> {resampled.shape[0]}@30fps)")
        n_done += 1

    print(f"[Rank {RANK}] DONE. done={n_done} skip={n_skip} "
          f"no_input={n_no_input} no_fps={n_no_fps} bad_shape={n_bad_shape}")


if __name__ == "__main__":
    main()
