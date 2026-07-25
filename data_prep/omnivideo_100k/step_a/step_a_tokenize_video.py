"""Step A (video -> seed2/cosmos/avc_lm tokens) for OmniVideo-100K, run on JUPITER.

See data_prep/omnivideo_100k/JUPITER_STEP_A_TASK.md for full background.

Does not modify pipeline_video/pipeline.py -- only imports the 3 low-level
tokenizer classes (Seed2Tokenizer/CosmosVideoTokenizer/AVCLMTokenizer) from
the original, which has the real checkpoints, at
/e/project1/reformo/nguyen38/prototype/pipeline.py. All the new logic (video
listing, chunking, caption/speech injection, output writing) lives in this
file.

2026-07-23: window=24 pivot to match FineVideo-VLA (CHUNK_SIZE 8->24,
COSMOS_STRIDE=3 added -- see the CHUNK_SIZE comment below) -- was previously
CHUNK_SIZE=8, no striding.
"""
import argparse
import glob
import json
import math
import os
import sys

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"
DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k"
DEFAULT_VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
DEFAULT_CAPTIONS_JSONL = os.path.join(DATA_ROOT, "omnivideo_100k_segment_captions.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_ROOT, "step_a_output")
# 2026-07-23: see prototype/pipeline.py's identical fix -- per-rank scratch
# (temp frames, temp seed2 jpg) moved off exa_project1, which is over its
# project-wide inode soft limit (jutil project dataquota -p reformo), onto
# exa_data1 (this dataset's own area, massive inode headroom by comparison).
SCRATCH_DIR = os.path.join(DATA_ROOT, "step_a_scratch")
os.makedirs(SCRATCH_DIR, exist_ok=True)

# pipeline.py imports `cosmos_tokenizer` as a local (non-pip) package resolved
# via sys.path[0] == its own directory, and the 3 tokenizer classes load
# checkpoints via paths relative to CWD ("./seed2", "pretrained_ckpts/...").
# Both only resolve correctly if we run with prototype/ as CWD and on sys.path.
sys.path.insert(0, PROTOTYPE_DIR)
os.chdir(PROTOTYPE_DIR)

# env_stable_vla currently has transformers==4.57.6, which moved 3 helper
# functions from modeling_utils to pytorch_utils. The vendored
# seed2/seed2_tokenizer.py (old-style BERT code) still imports all 3 from the
# old location, so Seed2Tokenizer silently fails to load (encode_image() then
# always returns [], i.e. seed2=0 for every video with no error surfaced).
# Shimmed here only, so pipeline.py / seed2/ / the shared env stay untouched.
import transformers.modeling_utils as _modeling_utils  # noqa: E402
import transformers.pytorch_utils as _pytorch_utils  # noqa: E402
for _name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
    if not hasattr(_modeling_utils, _name):
        setattr(_modeling_utils, _name, getattr(_pytorch_utils, _name))

# seed2_tokenizer.py deliberately sets Qformer.cls = None (the encode-only
# path never needs the MLM head) — fine under the transformers version this
# was written for, but transformers==4.57.6's from_pretrained() now calls
# tie_weights() unconditionally on every submodule, and get_output_embeddings()
# crashes on self.cls being None instead of tolerating it. Import seed2_tokenizer
# ourselves first (pipeline.py's Seed2Tokenizer.load_tokenizer() re-imports the
# same cached module later) and patch both classes with this pattern to return
# None (== "no output embeddings") instead of crashing, matching the original
# intent. Verified via data_prep/omnivideo_100k/step_a/debug_seed2_load.py.
sys.path.insert(0, os.path.join(PROTOTYPE_DIR, "seed2"))  # noqa: E402
import seed2_tokenizer as _seed2_tokenizer  # noqa: E402


def _safe_get_output_embeddings(self):
    return None if self.cls is None else self.cls.predictions.decoder


def _safe_set_output_embeddings(self, new_embeddings):
    if self.cls is not None:
        self.cls.predictions.decoder = new_embeddings


for _cls in (_seed2_tokenizer.BertLMHeadModel, _seed2_tokenizer.BertForMaskedLM):
    _cls.get_output_embeddings = _safe_get_output_embeddings
    _cls.set_output_embeddings = _safe_set_output_embeddings

from pipeline import (  # noqa: E402
    Seed2Tokenizer,
    CosmosVideoTokenizer,
    AVCLMTokenizer,
    RANK,
    WORLD_SIZE,
    print_main,
)

FFMPEG_BIN = os.environ.get("FFMPEG_PATH")
TARGET_FPS = 30
# 2026-07-23: widened from 8 to 24 real frames (0.267s -> 0.8s @ 30fps) to
# match FineVideo-VLA's window=24 pivot (pipeline_video/pipeline.py,
# REPORT.md #38 "Option 2" -- 8 frames was too short for visible motion
# within a chunk). COSMOS_STRIDE samples every Nth real frame from the
# chunk down to exactly 8 images for the Cosmos encoder (unaffected token
# cost per chunk): CHUNK_SIZE / COSMOS_STRIDE must equal 8.
CHUNK_SIZE = 24
COSMOS_STRIDE = 3
assert CHUNK_SIZE % COSMOS_STRIDE == 0 and CHUNK_SIZE // COSMOS_STRIDE == 8, \
    "CHUNK_SIZE/COSMOS_STRIDE must give exactly 8 samples for the Cosmos encoder"


def load_captions(path):
    """video_id -> {"video_summary": str, "duration": float, "segments": [...]}"""
    by_video = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            by_video[d["video_id"]] = {
                "video_summary": d.get("video_summary", ""),
                "duration": d.get("duration", 0),
                "segments": d.get("segments", []),
            }
    return by_video


# Extracted at 512x512 (matches Seed2Tokenizer's own target_size, so no quality
# loss there). CosmosVideoTokenizer used to downsample further to a 160x160
# squash regardless of input size -- below Cosmos-Tokenizer-DV8x16x16's own
# documented 256px (shorter side) minimum, and non-aspect-preserving. Fixed
# 2026-07-22 to resize-shorter-side+center-crop at 256 (see REPORT.md #35);
# this file needed no change since it calls encode_video_chunk() with no
# target_size override. One ffmpeg call per CHUNK_SIZE-frame chunk
# (bounded to ~8 small PNGs on disk at a time) rather than dumping an entire
# video's frames upfront — full-video upfront extraction at native resolution
# (up to 5400 frames/video, unscaled) blew the per-user disk quota once 32
# ranks ran concurrently (job 970087), even though it never bit the 8-rank
# pilot. Bounding the on-disk working set to one chunk fixes this regardless
# of video length or how tight the quota margin is.
EXTRACT_SIZE = 512


def extract_chunk_frames(video_path, start_sec, num_frames, temp_dir):
    """Extract up to num_frames consecutive 30fps frames starting at start_sec."""
    from PIL import Image
    import subprocess

    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    command = [
        FFMPEG_BIN, "-y", "-ss", str(start_sec), "-i", video_path,
        "-t", str(num_frames / TARGET_FPS), "-r", str(TARGET_FPS),
        # 2026-07-23: aspect-preserving bound (was scale=EXTRACT_SIZE:EXTRACT_SIZE,
        # a hard square squash) -- that silently defeated the cosmos
        # aspect-preserving fix (prototype/pipeline.py's encode_video_chunk(),
        # REPORT.md #37) because it forced every frame to w0==h0 *before*
        # cosmos's own aspect-aware resize ever saw the real shape, so
        # omnivideo's cosmos output was still square (512 tok/chunk) instead
        # of matching FineVideo's non-square (~896 tok/chunk for 16:9).
        # force_original_aspect_ratio=decrease bounds both dimensions to
        # EXTRACT_SIZE (same disk-usage goal as before) without distorting
        # aspect; seed2's branch does its own explicit square resize
        # afterward (frame.resize((seed2.target_size,)*2)) so this doesn't
        # affect seed2.
        "-vf", f"scale={EXTRACT_SIZE}:{EXTRACT_SIZE}:force_original_aspect_ratio=decrease",
        "-f", "image2", os.path.join(temp_dir, "frame_%02d.png"),
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    frames = []
    for name in sorted(os.listdir(temp_dir)):
        if not name.endswith(".png"):
            continue
        img_path = os.path.join(temp_dir, name)
        img = Image.open(img_path).convert("RGB")
        frames.append(img.copy())
        img.close()
    return frames


def build_segment_anchors(segments, n_chunks):
    """Map each segment to ONE anchor chunk (the first chunk starting at/after
    segment start) and wrap its caption/speech in tags there.

    Deliberately NOT inserted at every overlapping chunk: a segment averages
    ~11s (~14 chunks at 24-frame/30fps) but captions run 300-500 words —
    inserting at every chunk would repeat the same paragraph ~14x. Mirrors
    the anchor-point
    approach already used for FineVideo captions/speech (see PROGRESS_VI.md).
    """
    anchors = {}
    for seg in segments:
        caption = (seg.get("caption") or "").strip()
        speech = (seg.get("speech") or "").strip()
        if not caption and not speech:
            continue
        chunk_idx = min(n_chunks - 1, max(0, math.floor(seg["start_sec"] * TARGET_FPS / CHUNK_SIZE)))
        entry = anchors.setdefault(chunk_idx, {"caption": "", "speech": ""})
        if caption:
            entry["caption"] = (entry["caption"] + " " if entry["caption"] else "") + caption
        if speech:
            entry["speech"] = (entry["speech"] + " " if entry["speech"] else "") + speech
    return anchors


def tokenize_video(video_path, duration_sec, seed2, cosmos, avc_lm, anchors, temp_dir):
    """Mirrors VLADatasetBuilder.tokenize_activity_frames() in pipeline.py, plus
    caption/speech injection: [<caption>?] <cosmos> <avc_lm> [<speech>?] per chunk.

    Streams frames one CHUNK_SIZE-frame chunk at a time (extract_chunk_frames)
    instead of loading the whole video upfront — see EXTRACT_SIZE comment for
    why. Cosmos only ever sees exactly 8 frames per chunk (padded[::COSMOS_STRIDE]),
    independent of CHUNK_SIZE.
    """
    parts = []
    counts = {"seed2": 0, "cosmos": 0, "avclm": 0, "caption": 0, "speech": 0}
    total_frames = max(1, round(duration_sec * TARGET_FPS))
    n_chunks = math.ceil(total_frames / CHUNK_SIZE)

    for chunk_idx in range(n_chunks):
        chunk_start_frame = chunk_idx * CHUNK_SIZE
        remaining = min(CHUNK_SIZE, total_frames - chunk_start_frame)
        chunk_start_sec = chunk_start_frame / TARGET_FPS

        chunk_frames = extract_chunk_frames(video_path, chunk_start_sec, remaining, temp_dir)
        if not chunk_frames:
            continue

        for local_idx, frame in enumerate(chunk_frames):
            global_idx = chunk_start_frame + local_idx
            if global_idx % TARGET_FPS == 0:
                temp_path = os.path.join(SCRATCH_DIR, f"temp_seed2_rank_{RANK}.jpg")
                frame.resize((seed2.target_size, seed2.target_size)).save(temp_path)
                seed2_ids = seed2.encode_image(temp_path)
                if seed2_ids:
                    parts.append(f"<seed2> {' '.join(map(str, seed2_ids))} </seed2>")
                    counts["seed2"] += len(seed2_ids)
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        anchor = anchors.get(chunk_idx)
        if anchor and anchor["caption"]:
            parts.append(f"<caption> {anchor['caption']} </caption>")
            counts["caption"] += 1

        padded = chunk_frames + [chunk_frames[-1]] * (CHUNK_SIZE - len(chunk_frames))
        cosmos_ids = cosmos.encode_video_chunk(padded[::COSMOS_STRIDE])
        if cosmos_ids:
            parts.append(f"<cosmos> {' '.join(map(str, cosmos_ids))} </cosmos>")
            counts["cosmos"] += len(cosmos_ids)

        avc_ids = avc_lm.encode_mp4_segment(video_path, chunk_start_sec, len(chunk_frames) / TARGET_FPS)
        if avc_ids:
            parts.append(f"<avc_lm> {' '.join(map(str, avc_ids))} </avc_lm>")
            counts["avclm"] += len(avc_ids)

        if anchor and anchor["speech"]:
            parts.append(f"<speech> {anchor['speech']} </speech>")
            counts["speech"] += 1

    return " ".join(parts), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--captions-jsonl", default=DEFAULT_CAPTIONS_JSONL)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="Chỉ xử lý N video đầu tiên (pilot run). 0 = tất cả.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, f"step_a_rank_{RANK}.jsonl")

    print_main(f"Loading captions from {args.captions_jsonl} ...")
    captions_by_video = load_captions(args.captions_jsonl)

    video_files = sorted(f for f in os.listdir(args.videos_dir) if f.endswith(".mp4"))
    if args.limit > 0:
        video_files = video_files[: args.limit]
    my_files = video_files[RANK::WORLD_SIZE]

    processed = set()
    for existing in glob.glob(os.path.join(args.output_dir, "step_a_rank_*.jsonl")):
        with open(existing, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["video_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f"[Rank {RANK}] {len(my_files)} video assigned, {len(processed)} already done globally.")

    seed2 = Seed2Tokenizer()
    cosmos = CosmosVideoTokenizer()
    avc_lm = AVCLMTokenizer()
    temp_frames_dir = os.path.join(SCRATCH_DIR, f"omni_temp_frames_rank_{RANK}")
    os.makedirs(temp_frames_dir, exist_ok=True)

    for i, fname in enumerate(my_files):
        video_id = fname[:-4]
        if video_id in processed:
            continue
        video_path = os.path.join(args.videos_dir, fname)
        meta = captions_by_video.get(video_id, {"video_summary": "", "duration": 0, "segments": []})
        duration_sec = meta.get("duration", 0)

        try:
            if duration_sec <= 0:
                print(f"[Rank {RANK}] {video_id}: no duration in captions metadata, skip")
                continue

            total_frames = max(1, round(duration_sec * TARGET_FPS))
            n_chunks = math.ceil(total_frames / CHUNK_SIZE)
            anchors = build_segment_anchors(meta["segments"], n_chunks)
            token_str, counts = tokenize_video(
                video_path, duration_sec, seed2, cosmos, avc_lm, anchors, temp_frames_dir
            )

            header = f"### Context: {meta['video_summary']}"
            record = {"video_id": video_id, "text": f"{header}\n{token_str}"}

            with open(output_jsonl, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[Rank {RANK}] ({i + 1}/{len(my_files)}) {video_id}: "
                  f"seed2={counts['seed2']} cosmos={counts['cosmos']} avclm={counts['avclm']} "
                  f"caption={counts['caption']} speech={counts['speech']}")
        except Exception as e:
            print(f"[Rank {RANK}] ERROR on {video_id}: {e}")
            continue

    print(f"[Rank {RANK}] Done. Output: {output_jsonl}")


if __name__ == "__main__":
    main()
