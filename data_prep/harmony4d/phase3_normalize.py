#!/usr/bin/env python3
"""
Phase-3-equivalent for Harmony4D, normalize-only (no monocular noise filters).

2026-07-23, per Van Khue's explicit call: Harmony4D is multi-camera + SMPL-fit
ground truth, not a monocular estimate, so none of
phase3_kinematics_processor.py's noise-correction heuristics apply --
detect_hallucinations() (geometric-implausibility filter), the pelvis
teleport/ID-switch filter, temporal_smooth() (blurs real fast motion, e.g.
grappling), the per-joint velocity/acceleration anomaly filter (same problem
for MMA), and the stiff-leg spatial heuristic (fixes monocular depth-ambiguous
knee bends that don't happen in multi-view+SMPL data) are all skipped, per
make_pose_samples.py's docstring's original flag on this exact question.

What IS reused, because it's genuine format normalization rather than noise
correction: KinematicPreprocessor.split_root_motion() (pelvis-centering) and
.normalize_bone_lengths() (canonical skeleton scale -- keeps Harmony4D's
token-space scale consistent with every other data source's tokenization,
same target_bone_lengths dict), plus create_windows() (windowing +
small-gap forward-fill, generic).

Output goes straight to *_cleaned.jsonl naming (Phase 4's normal output
name), skipping Phase 4 entirely -- Phase 4 is YOLO-based occlusion
detection on the source video, which would be actively wrong here (2
overlapping bounding boxes during a real hug/grapple would look like
"occlusion" to YOLO but isn't -- Harmony4D's ground truth already tells us
exactly when each person exists). This lets Phase 5
(adaptive_pchip) consume the output unmodified.

Usage:
    python3 data_prep/harmony4d/phase3_normalize.py
"""
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline_pose"))

from phase3_kinematics_processor import (  # noqa: E402
    KinematicPreprocessor, create_windows, to_safe_json_list, interpolate_nan_gaps,
)

RESAMPLED_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_30fps"
OUTPUT_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/harmony4d_cleaned"
WINDOW_SIZE = 24
STRIDE = 24
CONF_THRESHOLD = 0.5


def process_track(xyz_conf: np.ndarray, processor: KinematicPreprocessor):
    """xyz_conf: (N, 17, 4) xyz + confidence. Returns (windows, valid_indices)."""
    xyz = xyz_conf[..., :3].copy()
    conf = xyz_conf[..., 3]
    xyz[conf < CONF_THRESHOLD] = np.nan

    # Generic small-gap fill (real occlusion/low-confidence gaps), same
    # utility Phase 3 uses -- not a noise-correction heuristic.
    xyz = interpolate_nan_gaps(xyz, max_gap=5)

    centered = processor.split_root_motion(xyz)
    valid = ~np.isnan(centered).any(axis=(1, 2))
    norm_pose = np.full_like(centered, np.nan)
    if np.any(valid):
        norm_pose[valid] = processor.normalize_bone_lengths(centered[valid])

    return create_windows(norm_pose, window_size=WINDOW_SIZE, stride=STRIDE)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor = KinematicPreprocessor(fps=30.0)

    tracks = []
    for category in sorted(os.listdir(RESAMPLED_DIR)):
        cat_dir = os.path.join(RESAMPLED_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for seq_id in sorted(os.listdir(cat_dir)):
            seq_dir = os.path.join(cat_dir, seq_id)
            if not os.path.isdir(seq_dir):
                continue
            for fname in sorted(os.listdir(seq_dir)):
                if fname.endswith(".npy"):
                    person_id = fname[:-4]
                    tracks.append((category, seq_id, person_id, os.path.join(seq_dir, fname)))

    print(f"{len(tracks)} tracks found under {RESAMPLED_DIR}")

    n_done = n_empty = 0
    for i, (category, seq_id, person_id, path) in enumerate(tracks, 1):
        video_id = f"{category}_{seq_id}_{person_id}"
        out_path = os.path.join(OUTPUT_DIR, f"{video_id}_cleaned.jsonl")
        if os.path.exists(out_path):
            continue

        arr = np.load(path)  # (N, 17, 4)
        windows, valid_indices = process_track(arr, processor)

        if len(windows) == 0:
            n_empty += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            for w, true_frame_id in zip(windows, valid_indices):
                record = {"window_id": int(true_frame_id), "states": to_safe_json_list(w)}
                f.write(json.dumps(record, allow_nan=False) + "\n")
        n_done += 1
        if i % 50 == 0 or i == len(tracks):
            print(f"[{i}/{len(tracks)}] done={n_done} empty={n_empty}")

    print(f"\nFinal: {n_done} tracks with windows, {n_empty} empty, {len(tracks) - n_done - n_empty} already existed.")


if __name__ == "__main__":
    main()
