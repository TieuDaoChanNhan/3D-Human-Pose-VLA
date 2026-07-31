#!/usr/bin/env python3
"""
Decode adaptive PCHIP agent tokens into 3D pose trajectories.

Takes the raw token string that the VLA model produces and reconstructs
the full (n_frames, 17, 3) skeleton trajectory via PCHIP interpolation.

Usage:
    # Decode tokens from stdin
    echo "<fps_30> <pelvis> <pelvis_t_0> ..." | python decode_agent_tokens.py

    # Decode from a text file
    python decode_agent_tokens.py --input generated_tokens.txt

    # Decode and save JSON
    python decode_agent_tokens.py --input tokens.txt --output poses.json

Token format (one 8-frame window):
    <fps_30>
    <pelvis> <pelvis_t_0> <pelvis_x_N> <pelvis_y_N> <pelvis_z_N>
             <pelvis_t_7> <pelvis_x_N> <pelvis_y_N> <pelvis_z_N> </pelvis>
    <r_hip>  <r_hip_t_0>  <r_hip_x_N> <r_hip_y_N> <r_hip_z_N> ... </r_hip>
    ...17 joints...

Dequantization: coord = N / 255.0 * 4.0 - 2.0   (metres, range [-2, 2])
Time tokens:    frame index 0-7 within the 8-frame window
Reconstruction: PCHIP interpolation over control points -> 8 frames

H36M joint order (17 joints):
  0  pelvis       4  l_hip        8  thorax      12 l_elbow     16 r_wrist
  1  r_hip        5  l_knee       9  nose        13 l_wrist
  2  r_knee       6  l_ankle     10  head_top    14 r_shoulder
  3  r_ankle      7  spine       11  l_shoulder  15 r_elbow
"""

import argparse
import json
import re
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

WINDOW_FRAMES = 8
COORD_RANGE = 2.0

JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "nose", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
N_JOINTS = len(JOINT_NAMES)


def dequantize(n: int) -> float:
    return n / 255.0 * (2.0 * COORD_RANGE) - COORD_RANGE


def parse_window(tokens: list[str]) -> dict:
    """Parse tokens for a single 8-frame window into per-joint control points.

    Args:
        tokens: list of token strings like ['<fps_30>', '<pelvis>', '<pelvis_t_0>', ...]

    Returns:
        dict with fps (int) and joints (dict mapping joint name to
        t_indices ndarray and cp_coords ndarray of shape (n_cp, 3)).
    """
    fps = 30
    if tokens and tokens[0].startswith("<fps_"):
        fps = int(re.match(r"<fps_(\d+)>", tokens[0]).group(1))
        tokens = tokens[1:]

    joints = {}
    i = 0
    while i < len(tokens):
        m = re.match(r"^<([a-z_]+)>$", tokens[i])
        if not m or m.group(1) not in JOINT_INDEX:
            i += 1
            continue

        name = m.group(1)
        close = f"</{name}>"
        i += 1
        t_indices = []
        coords = []

        while i < len(tokens) and tokens[i] != close:
            tm = re.match(rf"<{name}_t_(\d+)>$", tokens[i])
            if tm and i + 3 < len(tokens):
                xm = re.match(rf"<{name}_x_(\d+)>$", tokens[i + 1])
                ym = re.match(rf"<{name}_y_(\d+)>$", tokens[i + 2])
                zm = re.match(rf"<{name}_z_(\d+)>$", tokens[i + 3])
                if xm and ym and zm:
                    # t_indices/coords must grow together -- appending t before
                    # validating x/y/z (as this used to do) desyncs the two
                    # lists by 1 whenever the model emits a malformed quad
                    # (e.g. a token from a different joint/axis in this slot),
                    # which crashes PchipInterpolator downstream with a
                    # length(x) != length(y) error instead of just skipping
                    # the bad control point.
                    t_indices.append(int(tm.group(1)))
                    coords.append([
                        dequantize(int(xm.group(1))),
                        dequantize(int(ym.group(1))),
                        dequantize(int(zm.group(1))),
                    ])
                    i += 4
                    continue
            i += 1

        if i < len(tokens) and tokens[i] == close:
            i += 1

        if t_indices:
            joints[name] = {
                "t_indices": np.array(t_indices, dtype=int),
                "cp_coords": np.array(coords, dtype=np.float32),
            }

    return {"fps": fps, "joints": joints}


def reconstruct(parsed: dict) -> np.ndarray:
    """PCHIP-interpolate sparse control points into a full per-frame trajectory.

    Window length is inferred from the data itself (max t_index + 1 seen
    across all joints) rather than a fixed constant -- 2026-07-22 (REPORT.md
    #38): windows are now 24 frames (t up to 23), but this decoder is also
    used on older 8-frame data (t up to 7), so it must handle both without
    being told which convention a given token string uses.

    Returns ndarray of shape (window_frames, 17, 3) in metres, root-centred.
    """
    max_t = WINDOW_FRAMES - 1
    for jdata in parsed["joints"].values():
        if len(jdata["t_indices"]):
            max_t = max(max_t, int(jdata["t_indices"].max()))
    window_frames = max_t + 1

    t_out = np.arange(window_frames, dtype=np.float64)
    traj = np.zeros((window_frames, N_JOINTS, 3), dtype=np.float32)

    for name, jdata in parsed["joints"].items():
        j = JOINT_INDEX[name]
        t_cp = jdata["t_indices"].astype(np.float64)
        cp = jdata["cp_coords"]

        if len(t_cp) < 2:
            traj[:, j, :] = cp[0]
            continue

        for d in range(3):
            traj[:, j, d] = PchipInterpolator(t_cp, cp[:, d])(t_out)

    return traj


def decode(token_str: str) -> list[np.ndarray]:
    """Decode a token string into a list of (8, 17, 3) trajectories.

    Handles both single windows and multiple consecutive windows.
    """
    all_tokens = re.findall(r"<[^>]+>", token_str)
    if not all_tokens:
        return []

    # Split on <fps_N> boundaries — each is one window
    window_starts = [i for i, t in enumerate(all_tokens) if t.startswith("<fps_")]

    if not window_starts:
        parsed = parse_window(all_tokens)
        return [reconstruct(parsed)]

    trajectories = []
    for wi, start in enumerate(window_starts):
        end = window_starts[wi + 1] if wi + 1 < len(window_starts) else len(all_tokens)
        parsed = parse_window(all_tokens[start:end])
        trajectories.append(reconstruct(parsed))

    return trajectories


def to_json(trajectories: list[np.ndarray], fps: int = 30) -> dict:
    """Convert decoded trajectories to a JSON-serialisable dict.

    Per-window frame count is read from each trajectory's own shape (not a
    fixed constant) -- see reconstruct()'s docstring, 2026-07-22."""
    windows = []
    cum_frames = 0
    for i, traj in enumerate(trajectories):
        n_frames_this = traj.shape[0]
        motion = np.linalg.norm(traj[-1] - traj[0], axis=-1)
        top_movers = sorted(
            [(JOINT_NAMES[j], round(float(motion[j]), 4)) for j in range(N_JOINTS)],
            key=lambda x: x[1], reverse=True,
        )
        n_missing = sum(1 for name in JOINT_NAMES if name not in
                        {JOINT_NAMES[j] for j in range(N_JOINTS) if np.any(traj[:, j, :] != 0)})

        windows.append({
            "window": i,
            "time_sec": round(cum_frames / fps, 4),
            "trajectory": traj.tolist(),
            "value_range_m": [round(float(traj.min()), 4), round(float(traj.max()), 4)],
            "top_movers": top_movers[:5],
            "joints_all_zero": n_missing,
        })
        cum_frames += n_frames_this

    total_frames = sum(traj.shape[0] for traj in trajectories)
    return {
        "n_windows": len(trajectories),
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 4),
        "shape": [len(trajectories)] + list(trajectories[0].shape) if trajectories else [],
        "value_range_m": [round(float(min(t.min() for t in trajectories)), 4),
                           round(float(max(t.max() for t in trajectories)), 4)] if trajectories else [0, 0],
        "joint_names": JOINT_NAMES,
        "windows": windows,
    }


def main():
    p = argparse.ArgumentParser(description="Decode agent tokens to 3D poses.")
    p.add_argument("--input", "-i", default=None,
                   help="File containing agent tokens (default: read stdin)")
    p.add_argument("--output", "-o", default=None,
                   help="Save decoded poses to JSON file")
    args = p.parse_args()

    if args.input:
        with open(args.input, "r") as f:
            token_str = f.read()
    else:
        token_str = sys.stdin.read()

    token_str = token_str.strip()
    if not token_str:
        print("No tokens provided.", file=sys.stderr)
        sys.exit(1)

    trajectories = decode(token_str)
    if not trajectories:
        print("Could not parse any agent windows from input.", file=sys.stderr)
        sys.exit(1)

    result = to_json(trajectories)

    print(f"Decoded {result['n_windows']} windows "
          f"({result['total_frames']} frames, {result['duration_sec']}s)")
    print(f"Shape: {result['shape']}  (windows, frames, joints, xyz)")
    print(f"Value range: {result['value_range_m']} m")

    for w in result["windows"][:3]:
        print(f"\n  Window {w['window']} (t={w['time_sec']}s):")
        if w["joints_all_zero"] > 0:
            print(f"    WARNING: {w['joints_all_zero']} joints are all-zero (missing)")
        print(f"    Top movers: ", end="")
        print(", ".join(f"{name} {d:.3f}m" for name, d in w["top_movers"]))

    if result["n_windows"] > 3:
        print(f"\n  ... {result['n_windows'] - 3} more windows")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
