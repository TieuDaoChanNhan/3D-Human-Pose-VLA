#!/usr/bin/env python3
"""End-to-end verification of the edited VLA_Demo_Colab.ipynb notebook logic,
run for real (not just syntax-checked) against the live EmpathicRobotics/vla-1.7b-qwen3-v6
HF repo -- exactly what a Colab user would get from Runtime -> Run all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from huggingface_hub import snapshot_download
import vla_core as vc
import vla_app

print("=== 1. snapshot_download tools ===")
TOOLS = snapshot_download(
    vc.MODEL_ID,
    allow_patterns=["tools/*", "tools/**/*", "pipeline_pose/*"],
)
print("repo tools at:", TOOLS)

print("\n=== 2. vc.setup ===")
vc.setup(TOOLS)

print("\n=== 3. build_prompt (title+content -> separate header lines) ===")
prompt = vla_app.build_prompt(
    "Boy studying at his desk",
    "A young boy is sitting at a desk, studying and writing in a notebook.",
)
print(repr(prompt))
assert prompt.count("### Title:") == 1
assert prompt.count("### Context:") == 1
print("OK: two separate header lines")

print("\n=== 4. load_model (downloads the real 3.9GB v6 checkpoint) ===")
model, tok = vla_app.load_model()
print("context length (model.config.max_position_embeddings):",
      getattr(model.config, "max_position_embeddings", None))

print("\n=== 5. generate_once ===")
import torch
text = vla_app.generate_once(
    prompt, max_new_tokens=1200, do_sample=True, temperature=0.8,
    top_p=0.9, repetition_penalty=1.3, seed=42,
)
out_txt = os.path.join(os.path.dirname(__file__), "generated_full_text.txt")
with open(out_txt, "w") as f:
    f.write(prompt + "\n=== GENERATED ===\n" + text)
print(f"generated {len(text)} chars -> {out_txt}")

print("\n=== 6. summarize + _have ===")
counts = vc.summarize(text)
have = vla_app._have(counts)
print("counts:", counts)
print("have:", have)

OUT = os.path.dirname(__file__)

if have["pose"]:
    print("\n=== 7. decode_agent + render_pose_video (supersampled 30fps) ===")
    trajectories, summary = vc.decode_agent(text)
    pose_path = os.path.join(OUT, "pose.mp4")
    vc.render_pose_video(trajectories, pose_path)
    print(f"n_windows={summary['n_windows']} total_frames={summary['total_frames']} -> {pose_path}")
else:
    print("\n=== 7. no agent tokens produced this run (skipping pose) ===")

if have["video"]:
    print("\n=== 8. decode_cosmos (multi-chunk) ===")
    ids = [i for run in vc.token_runs(text, "cosmos") for i in run]
    chunks, leftover = vc.chunked(ids, vc.COSMOS_CHUNK)
    used = chunks[:15]
    video_path = os.path.join(OUT, "video.mp4")
    vc.decode_cosmos(used, video_path)
    print(f"{len(used)}/{len(chunks)} chunks decoded (leftover={leftover}) -> {video_path}")
else:
    print("\n=== 8. no complete cosmos chunk produced this run (skipping video) ===")

if have["audio"]:
    print("\n=== 9. decode_speak / decode_snac (wrapper-aware) ===")
    speak_ids = vc.wrapped_snac_runs(text, "speak")
    listen_ids = vc.wrapped_snac_runs(text, "listen")
    print(f"speak_snac ids: {len(speak_ids)}, listen_snac ids: {len(listen_ids)}")
    audio_path = os.path.join(OUT, "audio.wav")
    if len(speak_ids) >= vc.SPEAK_GROUP:
        n = len(speak_ids) - len(speak_ids) % vc.SPEAK_GROUP
        vc.decode_speak(speak_ids[:n], audio_path)
        print(f"decoded SPEAK format ({n} ids) -> {audio_path}")
    else:
        n = len(listen_ids) - len(listen_ids) % vc.SNAC_GROUP
        vc.decode_snac(listen_ids[:n], audio_path)
        print(f"decoded LISTEN format ({n} ids) -> {audio_path}")
else:
    print("\n=== 9. no snac tokens produced this run (skipping audio) ===")

if have["image"]:
    print("\n=== 10. decode_seed2 (generative, ~7.6GB first-use download) ===")
    run0 = next(r for r in vc.token_runs(text, "seed2") if len(r) >= vc.SEED2_QUERY_LEN)
    image_path = os.path.join(OUT, "image.png")
    vc.decode_seed2(run0[:vc.SEED2_QUERY_LEN], image_path)
    print(f"-> {image_path}")
else:
    print("\n=== 10. no seed2 tokens produced this run (skipping image) ===")

print("\n=== DONE ===")
