"""Phase 1 (2D pose: HRNet + Faster R-CNN) for the sports subset of
OmniVideo-100K, run on JUPITER. See
data_prep/omnivideo_100k/JUPITER_POSE_PILOT_TASK.md for full context (only
1,256/5,214 videos are selected via select_sports_subset.py).

Does not modify pipeline_pose/phase1_hrnet_gpu.py -- the original hard-codes
reading video from the FineVideo HF arrow dataset
(load_from_disk("/e/scratch/.../finevideo_disk") + cached_video_ids.json),
which doesn't work for OmniVideo-100K (flat mp4 files on disk, no arrow
dataset). This script reuses the model-agnostic part of the original (model
init, coco_to_h36m) but reads video directly from
$DATA/omnivideo_100k/videos/{video_id}.mp4 and shards over the sports-subset
list, following the same RANK::WORLD_SIZE pattern as step_a_tokenize_video.py.

One important difference from the original (per the requirement to keep
confidence scores, not discard them): coco_to_h36m() here KEEPS the continuous
confidence score from HRNet/the detector, instead of binarizing it to 1.0/0.0
like the original (phase1_hrnet_gpu.py lines 37-38:
`if c < CONF_THRESHOLD: return ..., 0.0` / `return ..., 1.0`). Downstream,
MotionBERT (third_party/MotionBERT/infer_wild.py via WildDetDataset) reads
that confidence column directly as an input feature for the lifting model --
a continuous value lets the model distinguish "somewhat uncertain" from
"almost certainly not visible", instead of only 2 levels like the original.
The (x, y) position is still zeroed below threshold (unchanged from the
original -- avoids feeding MotionBERT garbage coordinates); only the
confidence value is kept at its real, continuous value.

Output keeps the original's exact format ({"frame_id", "keypoints": [[x,y,conf]x17]}),
written to $DATA/omnivideo_100k/pose_2d_json/{video_id}_2d.json.

Note: this used to write into the shared outputs/2d_json/ (same directory
FineVideo uses, on the reasoning that video_ids from the two corpora never
collide, so it was "safe"). Moved to its own directory under the
OmniVideo-100K data root instead -- mixing a 1,126-file subset into a
40,804-file FineVideo directory made it needlessly hard to inspect, count,
or clean up this corpus independently, even though it was never incorrect.
"""
import argparse
import json
import os

import cv2
import numpy as np

# ================= MODEL CONFIGURATION (relative paths kept identical to the
# original -- this script must run with CWD = 3d-human-pose/) =================
POSE_CONFIG = "hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
POSE_CHECKPOINT = "hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
DET_CONFIG = "hrnet_storage/faster-rcnn_r50_fpn_1x_coco.py"
DET_CHECKPOINT = "hrnet_storage/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

CONF_THRESHOLD = 0.5

DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
DEFAULT_VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
DEFAULT_VIDEO_IDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sports_subset_video_ids_filtered.txt")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_ROOT, "pose_2d_json")

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def coco_to_h36m(coco_kpts):
    """Same as coco_to_h36m() in phase1_hrnet_gpu.py, but does NOT binarize
    confidence -- see module docstring."""
    h36m = np.zeros((17, 3), dtype=np.float32)

    def get_pt(idx):
        x, y, c = coco_kpts[idx]
        c = float(c)
        if c < CONF_THRESHOLD:
            return np.array([0.0, 0.0]), c
        return np.array([x, y]), c

    nose, c_nose = get_pt(0)
    lsho, c_lsho = get_pt(5)
    rsho, c_rsho = get_pt(6)
    lhip, c_lhip = get_pt(11)
    rhip, c_rhip = get_pt(12)

    h36m[1, :2], h36m[1, 2] = get_pt(12)
    h36m[2, :2], h36m[2, 2] = get_pt(14)
    h36m[3, :2], h36m[3, 2] = get_pt(16)
    h36m[4, :2], h36m[4, 2] = get_pt(11)
    h36m[5, :2], h36m[5, 2] = get_pt(13)
    h36m[6, :2], h36m[6, 2] = get_pt(15)
    h36m[11, :2], h36m[11, 2] = get_pt(5)
    h36m[12, :2], h36m[12, 2] = get_pt(7)
    h36m[13, :2], h36m[13, 2] = get_pt(9)
    h36m[14, :2], h36m[14, 2] = get_pt(6)
    h36m[15, :2], h36m[15, 2] = get_pt(8)
    h36m[16, :2], h36m[16, 2] = get_pt(10)
    h36m[9, :2], h36m[9, 2] = nose, c_nose

    # Gating kept as ">= threshold" (equivalent to the original's "> 0", since
    # the original had already binarized by this point) -- only the stored
    # confidence changes to the real value (min of the 2 contributing points,
    # instead of a hardcoded 1.0 like the original).
    if c_lhip >= CONF_THRESHOLD and c_rhip >= CONF_THRESHOLD:
        h36m[0, :2], h36m[0, 2] = (lhip + rhip) / 2.0, min(c_lhip, c_rhip)
    if c_lsho >= CONF_THRESHOLD and c_rsho >= CONF_THRESHOLD:
        h36m[8, :2], h36m[8, 2] = (lsho + rsho) / 2.0, min(c_lsho, c_rsho)
    if h36m[0, 2] >= CONF_THRESHOLD and h36m[8, 2] >= CONF_THRESHOLD:
        h36m[7, :2], h36m[7, 2] = (h36m[0, :2] + h36m[8, :2]) / 2.0, min(h36m[0, 2], h36m[8, 2])
    if c_nose >= CONF_THRESHOLD and h36m[8, 2] >= CONF_THRESHOLD:
        h36m[10, :2], h36m[10, 2] = nose + 0.8 * (nose - h36m[8, :2]), min(c_nose, h36m[8, 2])

    return h36m


def process_video_to_json(video_path, output_json_path, det_model, pose_model):
    from mmpose.apis import inference_topdown
    from mmdet.apis import inference_detector
    from mmengine.registry import init_default_scope

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture could not open {video_path}")

        # Container-reported frame count, used only as a post-hoc sanity check
        # below -- NOT used to derive an expected frame count from duration
        # (OmniVideo-100K has mixed native fps, e.g. 25fps vs 30fps; a
        # duration*30fps estimate would falsely flag legit 25fps videos as
        # "truncated" at ~83% -- verified empirically on 2 real pilot videos).
        container_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_frames_data = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            init_default_scope("mmdet")
            det_result = inference_detector(det_model, frame)
            pred_instances = det_result.pred_instances
            person_mask = (pred_instances.labels == 0) & (pred_instances.scores > 0.5)
            bboxes = pred_instances.bboxes[person_mask].cpu().numpy()

            frame_data = {"frame_id": frame_idx, "keypoints": []}
            if len(bboxes) > 0:
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                best_bbox = bboxes[np.argmax(areas)]

                init_default_scope("mmpose")
                pose_results = inference_topdown(pose_model, frame, bboxes=[best_bbox])

                if len(pose_results) > 0:
                    kpts = pose_results[0].pred_instances.keypoints[0]
                    scores = pose_results[0].pred_instances.keypoint_scores[0]
                    # Defensive: some mmpose versions/configs can return torch
                    # tensors here instead of numpy arrays; this codebase's
                    # installed mmpose==1.3.2 returns numpy (verified on 32
                    # real videos this session), but converting explicitly
                    # costs nothing and removes the version dependency.
                    if hasattr(kpts, "detach"):
                        kpts = kpts.detach().cpu().numpy()
                    if hasattr(scores, "detach"):
                        scores = scores.detach().cpu().numpy()
                    coco_final = np.concatenate([np.asarray(kpts), np.asarray(scores)[:, None]], axis=1)
                    h36m_keypoints = coco_to_h36m(coco_final)
                    frame_data["keypoints"] = h36m_keypoints.tolist()

            if not frame_data["keypoints"]:
                frame_data["keypoints"] = [[0.0, 0.0, 0.0]] * 17

            all_frames_data.append(frame_data)
            frame_idx += 1
            if frame_idx % 1000 == 0:
                print(f"   -> [Rank {RANK}] Frame: {frame_idx}", end="\r")
    finally:
        # Always release the VideoCapture even on an error mid-loop (e.g. an
        # OOM on one frame) -- otherwise cap leaks until GC collects it, which
        # could exhaust file descriptors across several bad videos in a row
        # in one long-running rank process (~280 videos/rank at the full
        # 1,126-video scale).
        cap.release()

    if frame_idx == 0:
        # cap.isOpened() can be True but the stream is empty/broken from the
        # first frame -- don't record this as "success" with 0 frames, which
        # would make the resume check treat the video as permanently done
        # with no error trail.
        raise RuntimeError(f"Read 0 frames from {video_path} (empty/corrupt video?)")

    if container_frame_count > 0 and frame_idx < 0.9 * container_frame_count:
        # Soft warning, not a hard failure: CAP_PROP_FRAME_COUNT is itself
        # sometimes an unreliable estimate for web-sourced/VFR video, so a
        # mismatch here isn't proof of a truncated decode -- but it's worth
        # a visible trail to spot-check later rather than staying silent.
        print(
            f"[Rank {RANK}] WARNING: {video_path} decoded {frame_idx}/{container_frame_count} "
            f"container-reported frames ({100 * frame_idx / container_frame_count:.1f}%) "
            "-- possible truncated decode, worth a manual check"
        )

    with open(output_json_path, "w") as f:
        json.dump(all_frames_data, f)
    return frame_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--video-ids-file", default=DEFAULT_VIDEO_IDS_FILE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N videos (pilot run). 0 = all.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.video_ids_file) as f:
        raw_ids = [line.strip() for line in f if line.strip()]
    video_ids = list(dict.fromkeys(raw_ids))  # dedup, order-preserving
    if len(video_ids) != len(raw_ids):
        print(f"[Rank {RANK}] Removed {len(raw_ids) - len(video_ids)} duplicate video IDs from {args.video_ids_file}")
    if args.limit > 0:
        video_ids = video_ids[: args.limit]
    my_ids = video_ids[RANK::WORLD_SIZE]

    print(f"[Rank {RANK}/{WORLD_SIZE}] {len(my_ids)}/{len(video_ids)} videos assigned, device={DEVICE}")

    print(f"[Rank {RANK}] Initializing HRNet & Faster R-CNN on {DEVICE}...")
    from mmpose.apis import init_model as init_pose_model
    from mmdet.apis import init_detector

    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)
    det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)

    n_done = n_skip = n_error = 0
    for i, video_id in enumerate(my_ids):
        video_path = os.path.join(args.videos_dir, f"{video_id}.mp4")
        final_json = os.path.join(args.output_dir, f"{video_id}_2d.json")
        tmp_json = final_json + f".tmp_rank{RANK}"

        # getsize() > 2 rules out a bare "[]" (or empty file) left over from an
        # older/interrupted run -- such a file would otherwise be treated as
        # permanently done and never reprocessed.
        if os.path.exists(final_json) and os.path.getsize(final_json) > 2:
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) Skip (already exists): {video_id}")
            n_skip += 1
            continue
        if not os.path.exists(video_path):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR: {video_path} not found")
            n_error += 1
            continue

        try:
            n_frames = process_video_to_json(video_path, tmp_json, det_model, pose_model)
            os.rename(tmp_json, final_json)
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) OK: {video_id} ({n_frames} frames)")
            n_done += 1
        except Exception as e:
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) ERROR {video_id}: {e}")
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
            n_error += 1

    print(f"[Rank {RANK}] DONE. done={n_done} skip={n_skip} error={n_error}")


if __name__ == "__main__":
    main()
