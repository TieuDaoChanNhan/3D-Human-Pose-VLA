"""Phase 2 (3D lifting: MotionBERT) for the sports subset of OmniVideo-100K,
run on JUPITER. See data_prep/omnivideo_100k/JUPITER_POSE_PILOT_TASK.md for
full context. Depends on Phase 1 (phase1_hrnet_omnivideo.py) having already
written outputs/2d_json/{video_id}_2d.json for each video in the subset.

Does not modify pipeline_pose/phase2_motionbert_gpu.py -- the original hard-codes
reading video from the FineVideo HF arrow dataset (load_from_disk(DATASET_PATH)
+ cached_video_ids.json for sharding), and extracts the mp4 bytes embedded in
that dataset to a per-worker temp file before calling MotionBERT. None of that
exists for OmniVideo-100K (flat mp4 files on disk, no arrow dataset) -- this
script reads video_ids from the same subset list Phase 1 used and points
MotionBERT directly at the real mp4 on disk (no temp copy needed, since unlike
FineVideo's embedded bytes, the file already exists as a real path).

Path note: the original references the MotionBERT binary/configs via a bare
"MotionBERT/..." relative path, which only resolves if a symlink
MotionBERT -> third_party/MotionBERT exists at the repo root (.gitignore has
a "MotionBERT/" entry consistent with that convention). That symlink is
currently missing on JUPITER (same class of infra gap as the outputs/ symlink
fixed for Phase 1, commit 2f3d675) -- rather than depend on it, this driver
references third_party/MotionBERT/ directly.

Sharding follows the same RANK::WORLD_SIZE pattern as phase1_hrnet_omnivideo.py
and step_a_tokenize_video.py (simpler than the original's --offset/--total_workers
scheme, which was sized for FineVideo's 200-worker/40K-video run -- unnecessary
at this dataset's 1,126-video scale).

Output: $DATA/omnivideo_100k/pose_3d_npy/{video_id}.npy (+ .mp4 preview if
MotionBERT produces one).

Note: this used to write into the shared outputs/3d_npy/ (same directory
FineVideo uses). Moved to its own directory under the OmniVideo-100K data
root instead, same reasoning as phase1_hrnet_omnivideo.py's DEFAULT_OUTPUT_DIR
-- keeps this 1,126-video corpus independently inspectable/cleanable without
touching FineVideo's 262GB tree.
"""
import argparse
import os
import shutil
import subprocess

MOTIONBERT_ROOT = "third_party/MotionBERT"
CONFIG = os.path.join(MOTIONBERT_ROOT, "configs/pose3d/MB_ft_h36m.yaml")
CHECKPOINT = os.path.join(MOTIONBERT_ROOT, "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin")
INFER_SCRIPT = os.path.join(MOTIONBERT_ROOT, "infer_wild.py")

DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
DEFAULT_VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
DEFAULT_VIDEO_IDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sports_subset_video_ids_filtered.txt")
DEFAULT_INPUT_2D_DIR = os.path.join(DATA_ROOT, "pose_2d_json")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_ROOT, "pose_3d_npy")
WORKSPACE = "workspace_temp"

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))


def lift_video(video_id, videos_dir, json_2d_dir, worker_tmp_dir):
    vid_path = os.path.join(videos_dir, f"{video_id}.mp4")
    json_2d = os.path.join(json_2d_dir, f"{video_id}_2d.json")

    for f in os.listdir(worker_tmp_dir):
        os.remove(os.path.join(worker_tmp_dir, f))

    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)

    cmd = [
        "python", INFER_SCRIPT,
        "--config", CONFIG,
        "--evaluate", CHECKPOINT,
        "--json_path", json_2d,
        "--vid_path", vid_path,
        "--out_path", worker_tmp_dir,
        "--pixel",
    ]
    subprocess.run(cmd, env=my_env, check=True, stdout=subprocess.DEVNULL)

    x3d_npy = os.path.join(worker_tmp_dir, "X3D.npy")
    if not os.path.exists(x3d_npy):
        raise RuntimeError("infer_wild.py exited 0 but X3D.npy is missing")
    return x3d_npy, os.path.join(worker_tmp_dir, "X3D.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--video-ids-file", default=DEFAULT_VIDEO_IDS_FILE)
    ap.add_argument("--input-2d-dir", default=DEFAULT_INPUT_2D_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N videos (pilot run). 0 = all.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(WORKSPACE, exist_ok=True)
    worker_tmp_dir = os.path.join(WORKSPACE, f"omni_worker_{RANK}_mb_tmp")
    os.makedirs(worker_tmp_dir, exist_ok=True)

    with open(args.video_ids_file) as f:
        raw_ids = [line.strip() for line in f if line.strip()]
    video_ids = list(dict.fromkeys(raw_ids))  # dedup, order-preserving
    if args.limit > 0:
        video_ids = video_ids[: args.limit]
    my_ids = video_ids[RANK::WORLD_SIZE]

    print(f"[Rank {RANK}/{WORLD_SIZE}] {len(my_ids)}/{len(video_ids)} videos assigned, "
          f"CUDA_VISIBLE_DEVICES={LOCAL_RANK}")

    n_done = n_skip = n_wait = n_error = 0
    for i, video_id in enumerate(my_ids):
        final_npy = os.path.join(args.output_dir, f"{video_id}.npy")
        final_mp4 = os.path.join(args.output_dir, f"{video_id}.mp4")
        json_2d = os.path.join(args.input_2d_dir, f"{video_id}_2d.json")

        if os.path.exists(final_npy):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) Skip (already exists): {video_id}")
            n_skip += 1
            continue
        if not os.path.exists(json_2d):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) WAIT (2D not ready): {video_id}")
            n_wait += 1
            continue
        if not os.path.exists(os.path.join(args.videos_dir, f"{video_id}.mp4")):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR: source mp4 not found: {video_id}")
            n_error += 1
            continue

        try:
            x3d_npy, x3d_mp4 = lift_video(video_id, args.videos_dir, args.input_2d_dir, worker_tmp_dir)
            tmp_final_npy = final_npy + f".tmp_rank{RANK}"
            # workspace_temp/ (local scratch) and outputs/3d_npy/ (symlink onto a
            # different mount) are different filesystems -- os.rename() raises
            # EXDEV across mounts, so this hop needs shutil.move (copy+delete
            # fallback). The second hop below stays a same-directory os.rename,
            # which IS atomic and safe against concurrent resume runs.
            shutil.move(x3d_npy, tmp_final_npy)
            os.rename(tmp_final_npy, final_npy)
            if os.path.exists(x3d_mp4):
                shutil.move(x3d_mp4, final_mp4)
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) OK: {video_id}")
            n_done += 1
        except Exception as e:
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR {video_id}: {e}")
            n_error += 1

    print(f"[Rank {RANK}] DONE. done={n_done} skip={n_skip} wait={n_wait} error={n_error}")


if __name__ == "__main__":
    main()
