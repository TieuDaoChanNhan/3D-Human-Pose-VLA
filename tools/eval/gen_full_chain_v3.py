#!/usr/bin/env python3
"""
Full-chain generation + real media decode for vla-1.7b-qwen3-v3, on a prompt
that is NOT from any training record (unlike eval_vla_v3_sanity.py, which
uses real training prompts for ground-truth comparison).

Generates with sampling (greedy collapses into repetition loops -- see
PROGRESS_VI.md 2026-07-24), then actually decodes whatever modalities the
model produced into real files: seed2 -> PNG, agent -> pose JSON + a quick
matplotlib skeleton snapshot, listen -> WAV. Cosmos is decoded only if a
full 200-token chunk appears -- decode_cosmos.py is still pinned to the OLD
window=8/160x160-square format (CHUNK_TOKENS=200), not the window=24/
aspect-preserving/896-token-per-chunk format v3 was actually trained on, so
a v3-generated cosmos block (if any) will very likely NOT decode cleanly --
this is a known, unfixed gap, not a bug in this script.

Usage:
    python tools/eval/gen_full_chain_v3.py --output-dir /path/to/outdir
"""
import argparse
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decode"))
from decode_agent_tokens import decode as decode_agent, to_json, JOINT_NAMES

MODEL_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/qwen3_1.7b_vla_v3/hf/iter_0000881"
TOKENIZER_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/tokenizer_vla_qwen3_v2"

# Genuinely novel prompt -- not copied from any training record.
NOVEL_PROMPT = (
    "### Title: Rooftop workout at sunset\n"
    "### Context: A man performs a series of jumping jacks on a rooftop "
    "terrace as the sun sets behind the city skyline.\n"
)

BLOCK_RE = re.compile(
    r"<(caption|seed2|cosmos|agent|listen|speak)>(.*?)</\1>", re.DOTALL
)


def extract_blocks(text):
    """Return list of (tag, inner_text) in order of appearance."""
    return [(m.group(1), m.group(2)) for m in BLOCK_RE.finditer(text)]


def ids_from_block(inner_text, prefix):
    return [int(x) for x in re.findall(rf"<{prefix}_(\d+)>", inner_text)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt", default=NOVEL_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-new-tokens", type=int, default=1200,
                   help="Force generation past the first <|im_end|> so a short "
                        "early stop (e.g. right after one seed2 block) doesn't "
                        "cut off before cosmos/agent/listen get a chance to appear.")
    args = p.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Prompt:\n{args.prompt}\n")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    torch.manual_seed(args.seed)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)

    txt_path = os.path.join(args.output_dir, "generated_full_text.txt")
    with open(txt_path, "w") as f:
        f.write(args.prompt + "\n\n=== GENERATED ===\n\n" + generated)
    print(f"Saved full generated text -> {txt_path}")

    blocks = extract_blocks(generated)
    print(f"\nFound {len(blocks)} wrapped blocks: {[b[0] for b in blocks]}")

    done = {"caption": False, "seed2": False, "cosmos": False, "agent": False, "snac": False}

    for tag, inner in blocks:
        if tag == "caption" and not done["caption"]:
            print(f"\n[caption] {inner.strip()}")
            done["caption"] = True

        elif tag == "seed2" and not done["seed2"]:
            ids = ids_from_block(inner, "seed2")
            print(f"\n[seed2] {len(ids)} tokens -> decoding to image...")
            try:
                from decode_seed2 import decode_seed2_tokens
                out_png = os.path.join(args.output_dir, "seed2_image.png")
                decode_seed2_tokens(ids, out_png)
                print(f"  Saved -> {out_png}")
            except Exception as e:
                print(f"  seed2 decode FAILED: {e}")
            done["seed2"] = True

        elif tag == "cosmos" and not done["cosmos"]:
            ids = ids_from_block(inner, "cosmos")
            print(f"\n[cosmos] {len(ids)} tokens found (decoder expects exactly 200 "
                  f"for the OLD window=8/square-160 format; v3 was trained w24/"
                  f"aspect-preserving/~896-tok chunks, so this will likely fail --"
                  f" that's a known tooling gap, not a new bug)")
            if len(ids) == 200:
                try:
                    from decode_cosmos import decode_cosmos_chunk
                    out_mp4 = os.path.join(args.output_dir, "cosmos_chunk0.mp4")
                    decode_cosmos_chunk(ids, out_mp4)
                    print(f"  Saved -> {out_mp4}")
                except Exception as e:
                    print(f"  cosmos decode FAILED: {e}")
            else:
                print("  Skipped decode (wrong chunk size for current decode_cosmos.py).")
            done["cosmos"] = True

        elif tag == "agent" and not done["agent"]:
            print(f"\n[agent] decoding pose...")
            try:
                trajectories = decode_agent(f"<agent>{inner}</agent>")
                if trajectories:
                    result = to_json(trajectories)
                    import json
                    out_json = os.path.join(args.output_dir, "agent_pose.json")
                    with open(out_json, "w") as f:
                        json.dump(result, f, indent=1)
                    print(f"  Decoded {result['n_windows']} window(s), shape {result['shape']}, "
                          f"range {result['value_range_m']} m -> {out_json}")
                    for w in result["windows"]:
                        movers = ", ".join(f"{n} {d:.3f}m" for n, d in w["top_movers"][:3])
                        print(f"    window {w['window']}: {movers}")
                    render_skeleton_png(trajectories, os.path.join(args.output_dir, "agent_pose_skeleton.png"))
                else:
                    print("  Could not decode (no trajectories)")
            except Exception as e:
                print(f"  agent decode FAILED: {e}")
            done["agent"] = True

        elif tag in ("listen", "speak") and not done["snac"]:
            ids = ids_from_block(inner, "snac")
            print(f"\n[{tag}] {len(ids)} snac tokens -> decoding to audio...")
            try:
                from decode_snac import decode_snac_tokens, decode_speak_tokens
                out_wav = os.path.join(args.output_dir, "audio.wav")
                if tag == "listen":
                    decode_snac_tokens(ids, out_wav)
                else:
                    decode_speak_tokens(ids, out_wav)
                print(f"  Saved -> {out_wav}")
            except Exception as e:
                print(f"  snac decode FAILED: {e}")
            done["snac"] = True

    print("\nDone. Modalities decoded:", {k: v for k, v in done.items() if v})


def render_skeleton_png(trajectories, out_path):
    """Quick static plot: first + last frame of the first window, side by side."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
    edges = [
        ("pelvis", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
        ("pelvis", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
        ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "nose"), ("nose", "head_top"),
        ("thorax", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
        ("thorax", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ]

    traj = trajectories[0]
    fig = plt.figure(figsize=(10, 5))
    for i, frame_idx in enumerate([0, traj.shape[0] - 1]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        pts = traj[frame_idx]
        for a, b in edges:
            ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
            ax.plot([pts[ia, 0], pts[ib, 0]], [pts[ia, 2], pts[ib, 2]], [pts[ia, 1], pts[ib, 1]], "b-")
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c="r", s=15)
        ax.set_title(f"frame {frame_idx}")
        ax.invert_zaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved skeleton snapshot -> {out_path}")


if __name__ == "__main__":
    main()
