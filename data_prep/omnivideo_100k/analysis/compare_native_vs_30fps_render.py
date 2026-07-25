"""One-off visual sanity check: render skeleton-only videos from the raw
native-fps Phase 2 output and from the Phase 2.5 30fps-resampled output, for
the same video_id, so a human can eyeball whether resampling changed the
motion (it shouldn't -- interpolation only adds/removes frames along the same
timeline, positions in space are untouched).

Reuses the projection/draw code from tools/visualize/render_filtered_skeleton.py
(generic numpy+cv2, no FineVideo coupling) instead of re-implementing it.

Usage:
    python data_prep/omnivideo_100k/analysis/compare_native_vs_30fps_render.py --video-id z-Qcz_FMW7Q
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from tools.visualize.render_filtered_skeleton import (  # noqa: E402
    compute_global_xy_projection, render_video,
)

DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
CANVAS_W, CANVAS_H = 640, 480


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "compare_renders"))
    args = ap.parse_args()

    vid = args.video_id
    native_npy = os.path.join(DATA_ROOT, "pose_3d_npy", f"{vid}.npy")
    resampled_npy = os.path.join(DATA_ROOT, "pose_3d_npy_30fps", f"{vid}.npy")
    fps_lookup = json.load(open(os.path.join(DATA_ROOT, "fps_lookup.json")))
    native_fps = fps_lookup[vid]

    os.makedirs(args.output_dir, exist_ok=True)

    pose_native = np.load(native_npy).astype(np.float64)
    pose_30fps = np.load(resampled_npy).astype(np.float64)

    print(f"{vid}: native {pose_native.shape[0]} frames @ {native_fps:.2f}fps "
          f"({pose_native.shape[0] / native_fps:.2f}s)  |  "
          f"30fps {pose_30fps.shape[0]} frames @ 30fps "
          f"({pose_30fps.shape[0] / 30.0:.2f}s)")

    out_native = os.path.join(args.output_dir, f"{vid}_before_native{native_fps:.0f}fps.mp4")
    out_30fps = os.path.join(args.output_dir, f"{vid}_after_30fps.mp4")

    proj_native = compute_global_xy_projection(pose_native, CANVAS_W, CANVAS_H)
    proj_30fps = compute_global_xy_projection(pose_30fps, CANVAS_W, CANVAS_H)

    render_video(proj_native, out_native, native_fps, CANVAS_W, CANVAS_H)
    render_video(proj_30fps, out_30fps, 30.0, CANVAS_W, CANVAS_H)

    print(f"Saved: {out_native}")
    print(f"Saved: {out_30fps}")


if __name__ == "__main__":
    main()
