#!/usr/bin/env python3
"""
Convert Harmony4D's per-frame poses3d/*.npy (COCO-17 keypoint order) into this
project's H36M-17 joint order, so it can flow through the existing FineVideo
pose pipeline (pipeline_pose/phase3_kinematics_processor.py onward) unchanged.

Verified before writing this (21/07/2026):
  - Each Harmony4D sequence zip contains
    <category>/<seq_id>/processed_data/poses3d/<frame:05d>.npy, one file per
    frame, each a dict{person_id: (17,4) float32} = xyz (meters, world frame)
    + confidence.
  - That 17-joint order is COCO-17 (nose, l_eye, r_eye, l_ear, r_ear,
    l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip,
    l_knee, r_knee, l_ankle, r_ankle) -- NOT this project's H36M-17.
    Confirmed via bone-length coefficient-of-variation across 741 frames of a
    high-motion ("mma") sequence: every low-CV (rigid) joint pair matched an
    anatomical COCO-17 bone exactly, with left/right symmetric pairs coming
    out near-identical (e.g. (11,13) and (12,14) both ~0.426m = hip-knee on
    each side) -- never matched an H36M-17 bone under that indexing.
  - Native camera fps is 20 (ffprobe on a sequence's cam01/images/rgb.mp4:
    301 frames / 15.05s), not this project's usual 30fps. This script does
    NOT resample -- it preserves native 20fps frame indices and records the
    fps in the manifest; a later, separate step (mirroring
    pipeline_pose/phase2_5_resample_30fps.py) should do the 20->30fps
    resample before this feeds phase3_kinematics_processor.py, which assumes
    30fps windowing.

COCO->H36M mapping: identical convention to this project's own
pipeline_pose/phase1_hrnet_gpu.py::coco_to_h36m() (used for HRNet's 2D COCO
output) -- pelvis/thorax/spine are hip-midpoint/shoulder-midpoint/their
midpoint, head_top is nose extrapolated away from thorax by the same 0.8
ratio. This is a 3D adaptation of that exact mapping, not a new convention.

Confidence handling: a joint with confidence < --conf-threshold is written as
NaN in xyz (matching how missing joints are represented everywhere else in
this project's pose pipeline -- see phase3_kinematics_processor.py's
NaN-window skip logic), not zero-filled. Zero-fill for a derived joint like
head_top was flagged as an artifact in REPORT.md's Phase 4 section; NaN avoids
repeating it.

Output layout (one file per sequence per person, native 20fps, NOT resampled):
    <output-dir>/<category>/<seq_id>/<person_id>.npy
        (T, 17, 4) float32, H36M-17 order, xyz + confidence,
        NaN xyz for missing/low-confidence joints
    <output-dir>/<category>/<seq_id>/<person_id>_frame_idx.npy
        (T,) int32 -- native frame index of each row (frames can be
        non-contiguous if upstream Harmony4D processing dropped some)
    <output-dir>/<category>/<seq_id>/.done
        marker file, written only after all persons in that sequence are
        converted successfully
    <output-dir>/manifest_task<N>.jsonl
        one line per (category, seq_id, person_id):
        {"category", "seq_id", "person_id", "num_frames", "native_fps": 20,
         "source_zip", "output_path"}

Resumable: a sequence is skipped entirely if its `.done` marker already
exists -- safe to re-run / kill and restart.

Worker distribution: splits the *list of zip files* across
SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT, same pattern as every other
Phase 3-6 script in this project (see CLAUDE.md's "Worker Distribution
Pattern"). Reads are member-by-member via zipfile -- a zip is never fully
extracted to disk.

Usage:
    python3 data_prep/harmony4d/convert_coco_to_h36m.py \
        --harmony4d-root /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d \
        --output-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_native20fps
"""
import argparse
import glob
import io
import json
import os
import re
import zipfile
from collections import defaultdict

import numpy as np

# ============================================================
# COCO-17 -> H36M-17 mapping (3D adaptation of
# pipeline_pose/phase1_hrnet_gpu.py::coco_to_h36m())
# ============================================================
H36M_JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "nose", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

# COCO-17 keypoint indices
COCO_NOSE = 0
COCO_LSHO, COCO_RSHO = 5, 6
COCO_LELB, COCO_RELB = 7, 8
COCO_LWRI, COCO_RWRI = 9, 10
COCO_LHIP, COCO_RHIP = 11, 12
COCO_LKNEE, COCO_RKNEE = 13, 14
COCO_LANK, COCO_RANK = 15, 16

# H36M index -> source COCO index, for joints that map 1:1
DIRECT_MAP = {
    1: COCO_RHIP, 2: COCO_RKNEE, 3: COCO_RANK,
    4: COCO_LHIP, 5: COCO_LKNEE, 6: COCO_LANK,
    9: COCO_NOSE,
    11: COCO_LSHO, 12: COCO_LELB, 13: COCO_LWRI,
    14: COCO_RSHO, 15: COCO_RELB, 16: COCO_RWRI,
}

HEAD_TOP_EXTRAPOLATION_RATIO = 0.8  # matches phase1_hrnet_gpu.py exactly


def coco_to_h36m_3d(coco_kpts, conf_threshold):
    """coco_kpts: (17, 4) array [x, y, z, confidence] in COCO-17 order.
    Returns (17, 4) array in H36M-17 order; NaN xyz for missing/low-confidence joints."""

    def get_pt(idx):
        x, y, z, c = coco_kpts[idx]
        if c < conf_threshold:
            return np.array([np.nan, np.nan, np.nan]), 0.0
        return np.array([x, y, z], dtype=np.float32), float(c)

    h36m = np.full((17, 4), np.nan, dtype=np.float32)

    for h36m_idx, coco_idx in DIRECT_MAP.items():
        pt, c = get_pt(coco_idx)
        h36m[h36m_idx, :3] = pt
        h36m[h36m_idx, 3] = c

    lhip_pt, lhip_c = get_pt(COCO_LHIP)
    rhip_pt, rhip_c = get_pt(COCO_RHIP)
    if lhip_c > 0 and rhip_c > 0:
        h36m[0, :3] = (lhip_pt + rhip_pt) / 2.0
        h36m[0, 3] = min(lhip_c, rhip_c)

    lsho_pt, lsho_c = get_pt(COCO_LSHO)
    rsho_pt, rsho_c = get_pt(COCO_RSHO)
    if lsho_c > 0 and rsho_c > 0:
        h36m[8, :3] = (lsho_pt + rsho_pt) / 2.0
        h36m[8, 3] = min(lsho_c, rsho_c)

    if not np.isnan(h36m[0, 0]) and not np.isnan(h36m[8, 0]):
        h36m[7, :3] = (h36m[0, :3] + h36m[8, :3]) / 2.0
        h36m[7, 3] = min(h36m[0, 3], h36m[8, 3])

    nose_pt, nose_c = get_pt(COCO_NOSE)
    if nose_c > 0 and not np.isnan(h36m[8, 0]):
        h36m[10, :3] = nose_pt + HEAD_TOP_EXTRAPOLATION_RATIO * (nose_pt - h36m[8, :3])
        h36m[10, 3] = nose_c

    return h36m


# ============================================================
# Sequence discovery + conversion
# ============================================================
FRAME_RE = re.compile(r"processed_data/poses3d/(\d+)\.npy$")


def list_sequences_in_zip(zf):
    """Returns {(category, seq_id): [zip member names for poses3d frames]}."""
    seqs = defaultdict(list)
    for name in zf.namelist():
        m = FRAME_RE.search(name)
        if not m:
            continue
        category, seq_id = name.split("/")[0], name.split("/")[1]
        seqs[(category, seq_id)].append(name)
    return seqs


def convert_sequence(zf, frame_members, conf_threshold):
    """Returns {person_id: ((T,17,4) h36m array, (T,) frame_idx array)}."""
    frame_members = sorted(frame_members, key=lambda n: int(FRAME_RE.search(n).group(1)))
    per_person_frames = defaultdict(dict)  # person_id -> {frame_idx: (17,4) h36m array}

    for member in frame_members:
        frame_idx = int(FRAME_RE.search(member).group(1))
        raw = np.load(io.BytesIO(zf.read(member)), allow_pickle=True).item()
        for person_id, coco_kpts in raw.items():
            per_person_frames[person_id][frame_idx] = coco_to_h36m_3d(coco_kpts, conf_threshold)

    result = {}
    for person_id, frames_by_idx in per_person_frames.items():
        sorted_idx = sorted(frames_by_idx.keys())
        arr = np.stack([frames_by_idx[i] for i in sorted_idx])
        result[person_id] = (arr, np.array(sorted_idx, dtype=np.int32))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmony4d-root",
                    default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d")
    ap.add_argument("--output-dir",
                    default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_native20fps")
    ap.add_argument("--conf-threshold", type=float, default=0.3)
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    zip_paths = []
    for split in args.splits:
        zip_paths.extend(sorted(glob.glob(os.path.join(args.harmony4d_root, split, "*.zip"))))

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    my_zips = zip_paths[task_id - 1::num_tasks]

    print(f"[task {task_id}/{num_tasks}] {len(my_zips)}/{len(zip_paths)} zip files assigned", flush=True)

    manifest_path = os.path.join(args.output_dir, f"manifest_task{task_id}.jsonl")
    with open(manifest_path, "a") as mf:
        for zip_path in my_zips:
            print(f"[task {task_id}] opening {zip_path}", flush=True)
            with zipfile.ZipFile(zip_path) as zf:
                for (category, seq_id), frame_members in list_sequences_in_zip(zf).items():
                    seq_out_dir = os.path.join(args.output_dir, category, seq_id)
                    done_marker = os.path.join(seq_out_dir, ".done")
                    if os.path.exists(done_marker):
                        print(f"  [skip] {category}/{seq_id} already converted", flush=True)
                        continue

                    print(f"  converting {category}/{seq_id} ({len(frame_members)} frames)...", flush=True)
                    per_person = convert_sequence(zf, frame_members, args.conf_threshold)

                    os.makedirs(seq_out_dir, exist_ok=True)
                    for person_id, (arr, idx_arr) in per_person.items():
                        np.save(os.path.join(seq_out_dir, f"{person_id}.npy"), arr)
                        np.save(os.path.join(seq_out_dir, f"{person_id}_frame_idx.npy"), idx_arr)
                        mf.write(json.dumps({
                            "category": category, "seq_id": seq_id, "person_id": person_id,
                            "num_frames": int(len(idx_arr)), "native_fps": 20,
                            "source_zip": zip_path,
                            "output_path": os.path.join(seq_out_dir, f"{person_id}.npy"),
                        }) + "\n")
                    mf.flush()

                    with open(done_marker, "w"):
                        pass
                    print(f"    done, {len(per_person)} person-tracks written", flush=True)

    print(f"[task {task_id}/{num_tasks}] complete.", flush=True)


if __name__ == "__main__":
    main()
