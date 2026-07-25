"""Phase 3 (kinematics processing -> 24-frame state windows) for the sports
subset of OmniVideo-100K, run on JUPITER. Depends on Phase 2.5
(phase2_5_resample_omnivideo.py) having written
$DATA/omnivideo_100k/pose_3d_npy_30fps/{video_id}.npy.

Does not modify pipeline_pose/phase3_kinematics_processor.py -- unlike
Phase 1/2's originals, that script is already dataset-agnostic (takes
--input-dir/--output-dir/--json-2d-dir as CLI args, no FineVideo-specific
paths), so KinematicPreprocessor and process_file are imported directly
rather than copy-pasted (~300 lines of numerically sensitive smoothing/
hallucination-filter code -- importing avoids drift between two copies).

What differs here is iteration and output location, same reasoning as
Phase 2.5: the original globs *all* of an --input-dir (would be 40,804+
files if pointed at the shared FineVideo directory) and shards via
SLURM_ARRAY_TASK_ID/COUNT; this driver iterates the known 1,126-video
subset list and shards via RANK::WORLD_SIZE (same pattern as every other
OmniVideo-100K driver in this directory), and writes to its own directory
under the OmniVideo-100K data root rather than the shared outputs/ tree
(see phase1_hrnet_omnivideo.py / phase2_motionbert_omnivideo.py docstrings
for why: keeps this corpus independently inspectable/cleanable).

No --fps-json is passed to KinematicPreprocessor (fps=30.0 default) --
input is already resampled to a uniform 30fps grid by Phase 2.5, matching
slurm/submit_kinematics.sh's own invocation for FineVideo (which also omits
--fps-json for the same reason).

2026-07-23: window=24 pivot to match FineVideo-VLA (was window_size=8,
stride=8; output dir was pose_states_jsonl_30fps) -- see
step_a/step_a_tokenize_video.py's CHUNK_SIZE comment for the full rationale.

Output: $DATA/omnivideo_100k/pose_states_jsonl_30fps_w24/{video_id}_states.jsonl
  shape (windows, 24, 17, 3), stride=24 (matches Phase 5 downstream, avoids
  storing redundant overlapping windows).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pipeline_pose.phase3_kinematics_processor import KinematicPreprocessor, process_file  # noqa: E402

DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
DEFAULT_VIDEO_IDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sports_subset_video_ids_filtered.txt")
DEFAULT_INPUT_DIR = os.path.join(DATA_ROOT, "pose_3d_npy_30fps")
DEFAULT_JSON_2D_DIR = os.path.join(DATA_ROOT, "pose_2d_json")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_ROOT, "pose_states_jsonl_30fps_w24")
# 2026-07-23: window=24 to match FineVideo-VLA's pivot (was 8) -- see
# step_a/step_a_tokenize_video.py's CHUNK_SIZE comment for the full rationale.
WINDOW_SIZE = 24
STRIDE = 24

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids-file", default=DEFAULT_VIDEO_IDS_FILE)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--json-2d-dir", default=DEFAULT_JSON_2D_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.video_ids_file) as f:
        video_ids = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    my_ids = video_ids[RANK::WORLD_SIZE]

    print(f"[Rank {RANK}/{WORLD_SIZE}] {len(my_ids)}/{len(video_ids)} videos assigned")

    processor = KinematicPreprocessor(fps=30.0)
    n_done = n_skip = n_no_input = n_empty = n_error = 0

    for i, video_id in enumerate(my_ids):
        final_jsonl = os.path.join(args.output_dir, f"{video_id}_states.jsonl")
        if os.path.exists(final_jsonl):
            n_skip += 1
            continue

        npy_path = os.path.join(args.input_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR: input npy not found: {video_id}")
            n_no_input += 1
            continue

        temp_jsonl = f"{final_jsonl}.tmp_rank{RANK}"
        try:
            success = process_file(
                npy_path, temp_jsonl, processor, video_id,
                json_2d_dir=args.json_2d_dir, stride=STRIDE, window_size=WINDOW_SIZE,
            )
            if success:
                os.replace(temp_jsonl, final_jsonl)  # same dir -> atomic, safe for resume
                print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) OK: {video_id}")
                n_done += 1
            else:
                if os.path.exists(temp_jsonl):
                    os.remove(temp_jsonl)
                print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) EMPTY (no valid windows): {video_id}")
                n_empty += 1
        except Exception as e:
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR {video_id}: {e}")
            n_error += 1

    print(f"[Rank {RANK}] DONE. done={n_done} skip={n_skip} "
          f"no_input={n_no_input} empty={n_empty} error={n_error}")


if __name__ == "__main__":
    main()
