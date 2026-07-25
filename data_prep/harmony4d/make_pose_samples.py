#!/usr/bin/env python3
"""
Generate visual samples of Harmony4D pose data at two checkpoints, for
sanity-checking convert_coco_to_h36m.py + resample_30fps.py before deciding
on the windowing/root-centering step for real:

  1. "resampled"  -- world-frame xyz straight out of resample_30fps.py
                     (already COCO->H36M converted, already 30fps)
  2. "windowed"   -- root-centered (pelvis subtracted per frame) and cut into
                     8-frame windows (stride=8), i.e. what phase3-equivalent
                     processing would produce for Harmony4D -- NOTE: this
                     intentionally does NOT run the full
                     KinematicPreprocessor.process() from
                     phase3_kinematics_processor.py. That method's
                     hallucination filter / ID-switch jump detection / stiff-leg
                     heuristic are tuned to correct monocular MotionBERT
                     failure modes (depth-ambiguous knee bends, single-camera
                     occlusion ID swaps) that don't apply to Harmony4D's
                     multi-view + SMPL-fit ground truth -- running them
                     unmodified risks "fixing" genuine fast/unusual poses
                     (e.g. real grappling ground positions) as if they were
                     sensor noise. Only `create_windows` (windowing) and a
                     plain pelvis-centering are reused here; the rest is an
                     open decision, not made by this script.

Rendering reuses tools/visualize/render_filtered_skeleton.py's projection/
draw functions unchanged (same blank-canvas orthographic skeleton style
already used for FineVideo samples).

Usage:
    python3 data_prep/harmony4d/make_pose_samples.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "visualize"))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline_pose"))

from render_filtered_skeleton import compute_global_xy_projection, draw_skeleton  # noqa: E402
from phase3_kinematics_processor import create_windows  # noqa: E402

import cv2  # noqa: E402

RESAMPLED_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d_h36m_30fps"
OUT_DIR = os.path.join(REPO_ROOT, "samples", "harmony4d")
FPS = 30
WIDTH, HEIGHT = 640, 640

# A few representative (category, seq_id, person_id) picks:
#   - high-motion (mma) x both people, to also show the "2 independent
#     single-person tracks per sequence" decision
#   - low-motion (hugging) for contrast
SAMPLES = [
    ("15_mma4", "016_mma4", "aria01"),
    ("15_mma4", "016_mma4", "aria02"),
    ("01_hugging", "002_hugging", "aria01"),
]


def render_xyz_sequence(xyz, out_path, fps=FPS, width=WIDTH, height=HEIGHT):
    """xyz: (N, 17, 3), may contain NaN."""
    pose_2d = compute_global_xy_projection(xyz, width, height)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")
    try:
        for i in range(xyz.shape[0]):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_skeleton(frame, pose_2d[i], color=(0, 255, 0), thickness=2)
            writer.write(frame)
    finally:
        writer.release()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for category, seq_id, person_id in SAMPLES:
        tag = f"{category}_{seq_id}_{person_id}"
        pose_path = os.path.join(RESAMPLED_DIR, category, seq_id, f"{person_id}.npy")
        if not os.path.exists(pose_path):
            print(f"[skip] missing {pose_path}")
            continue

        arr = np.load(pose_path)          # (N, 17, 4) xyz + confidence, 30fps, world-frame
        xyz = arr[..., :3]
        print(f"{tag}: {xyz.shape[0]} frames @ {FPS}fps")

        # --- Checkpoint 1: resampled (world-frame, pre-centering) ---
        out1 = os.path.join(OUT_DIR, f"{tag}_1_resampled_30fps.mp4")
        render_xyz_sequence(xyz, out1)
        print(f"  wrote {out1}")

        # --- Checkpoint 2: root-centered + windowed (stride=8) ---
        pelvis = xyz[:, 0:1, :]
        centered = xyz - pelvis  # pelvis becomes (0,0,0) every frame

        windows, valid_indices = create_windows(centered, window_size=8, stride=8)
        if len(windows) == 0:
            print(f"  [skip windowed render] no valid 8-frame windows for {tag}")
            continue

        # Windows are non-overlapping (stride=8) -> concatenate back into one
        # continuous clip for rendering, in original temporal order.
        windowed_concat = np.concatenate(list(windows), axis=0)  # (n_windows*8, 17, 3)
        out2 = os.path.join(OUT_DIR, f"{tag}_2_windowed_centered.mp4")
        render_xyz_sequence(windowed_concat, out2)
        print(f"  {len(windows)} windows ({len(windows) * 8} frames) -> wrote {out2}")

        # Also dump the windowed states as JSONL, same schema
        # render_filtered_skeleton.py's load_jsonl_states expects
        # ({"window_id", "states": (8,17,3)}), for downstream reuse/inspection.
        import json

        def to_safe(w):
            obj = w.astype(object)
            obj[np.isnan(w)] = None
            return obj.tolist()

        jsonl_path = os.path.join(OUT_DIR, f"{tag}_windowed_states.jsonl")
        with open(jsonl_path, "w") as f:
            for w, idx in zip(windows, valid_indices):
                f.write(json.dumps({"window_id": int(idx), "states": to_safe(w)}) + "\n")
        print(f"  wrote {jsonl_path}")


if __name__ == "__main__":
    main()
