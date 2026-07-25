#!/usr/bin/env python3
"""
General-purpose Cosmos detokenizer -- turns `<cosmos_N>` tokens (from any of
our flattened datasets: FineVideo-VLA, OmniVideo-100K, ...) back into an
actual video.

Verified 20/07/2026 (see PROGRESS_VI.md entry) against a real 200-token
chunk from omnivideo_100k_final: for the DV8x16x16 checkpoint, encode() of
an 8-frame/160x160-square chunk produces indices of shape (1, 2, 10, 10) ==
200 flat tokens. This is the OLD preprocessing convention (square center-crop,
target_size=160) used by vla-1.7b-qwen3-v2.

2026-07-24: added support for the NEW convention (commit `7437967`, in use
since v3): aspect-preserving resize, no crop, shorter side = 256 -> for a
16:9 frame that's 256x448 -> indices shape (1, 2, 16, 28) == 896 flat tokens.
Both formats coexist in this project's history, so the chunk grid is now
auto-detected from the token count (200 -> old, 896 -> new) rather than a
single hardcoded constant. A different Cosmos-Tokenizer-* checkpoint or a
different target_size/aspect ratio would need its own grid added here.

This is a lossy, generative-but-deterministic *reconstruction* (the neural
decoder's output, not the original pixels) -- expect visible blur/artifacts,
this is the accepted tradeoff for a much lower token count than storing raw
pixels. Contrast with decode_avclm.py in this same directory, which is
byte-exact (BPE over raw H.264 bytes, not a neural codec).

Token values in `<cosmos_N>` are the RAW encoder codebook indices with no
vocab offset added (verified: flatten scripts across the project just wrap
the same digit string in `<cosmos_{n}>`, never add an offset) -- so N can be
fed straight into decode() after reshaping.

Usage:
    # From a raw list of ints:
    python tools/decode/decode_cosmos.py --tokens 18697,55801,44451,... --output out.mp4

    # From a flattened JSONL record (extracts the Nth 200-token chunk):
    python tools/decode/decode_cosmos.py --input-jsonl path.jsonl --record-id VIDEO_ID \
        --chunk-index 0 --output out.mp4
"""
import argparse
import json
import os
import re
import subprocess
import sys

# 2026-07-23: PROTOTYPE_DIR only exists on the internal cluster (not part of
# the public github.com/TieuDaoChanNhan/finevideo-vla repo -- 0 files under
# prototype/ are git-tracked). External users have no way to reach a
# hardcoded local path, so the checkpoint is now fetched from NVIDIA's public
# HF repo on first use instead (cached under the standard HF_HOME, same as
# any other from_pretrained() call) -- falls back to the local cluster copy
# first if it happens to be present, to avoid a redundant download here.
PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"
_LOCAL_CHECKPOINT_DEC = os.path.join(
    PROTOTYPE_DIR, "pretrained_ckpts/Cosmos-Tokenizer-DV8x16x16/decoder.jit"
)
COSMOS_HF_REPO = "nvidia/Cosmos-Tokenizer-DV8x16x16"

# (T', H', W') per 8-frame input chunk, this checkpoint -- two conventions
# coexist in this project's history (see module docstring). Keyed by total
# flat token count so callers can auto-detect which one applies.
CHUNK_GRID_OLD = (2, 10, 10)   # v2: square 160x160 center-crop -- 200 tokens
CHUNK_GRID_NEW = (2, 16, 28)   # v3+: aspect-preserving 256x448 (16:9) -- 896 tokens
GRIDS_BY_TOKEN_COUNT = {
    CHUNK_GRID_OLD[0] * CHUNK_GRID_OLD[1] * CHUNK_GRID_OLD[2]: CHUNK_GRID_OLD,
    CHUNK_GRID_NEW[0] * CHUNK_GRID_NEW[1] * CHUNK_GRID_NEW[2]: CHUNK_GRID_NEW,
}
# Backward-compat aliases (old code/callers importing these names directly).
CHUNK_GRID = CHUNK_GRID_OLD
CHUNK_TOKENS = CHUNK_GRID_OLD[0] * CHUNK_GRID_OLD[1] * CHUNK_GRID_OLD[2]  # 200


def _grid_for(n_tokens: int) -> tuple:
    grid = GRIDS_BY_TOKEN_COUNT.get(n_tokens)
    if grid is None:
        raise ValueError(
            f"{n_tokens} cosmos tokens doesn't match any known chunk grid "
            f"({sorted(GRIDS_BY_TOKEN_COUNT.keys())}). If this is a genuinely "
            f"new preprocessing convention, add its (T',H',W') grid to "
            f"GRIDS_BY_TOKEN_COUNT above rather than guessing."
        )
    return grid


def _resolve_checkpoint_dec() -> str:
    if os.path.exists(_LOCAL_CHECKPOINT_DEC):
        return _LOCAL_CHECKPOINT_DEC
    from huggingface_hub import hf_hub_download
    print(f"Local checkpoint not found -- downloading decoder.jit from {COSMOS_HF_REPO} "
          f"(~350MB, cached for future runs)...")
    return hf_hub_download(repo_id=COSMOS_HF_REPO, filename="decoder.jit")

_COSMOS_ATOMIC_RE = re.compile(r"<cosmos_(\d+)>")
_COSMOS_RAW_BLOCK_RE = re.compile(r"<cosmos>(.*?)</cosmos>", re.DOTALL)


def extract_chunk_tokens(text: str, chunk_index: int, chunk_tokens: int = None) -> list:
    """Pull out the Nth chunk-sized slice of cosmos ids from a record.
    Supports both formats found in this project:
      - flattened/atomic: `<cosmos_N> <cosmos_N> ...` (post-flatten output,
        e.g. omnivideo_100k_final, FineVideo's megatron_dataset_*) -- chunk
        size is ambiguous here (dropout leaves gaps, and old/new-format
        records both exist), so `chunk_tokens` (200 or 896) MUST be passed
        explicitly for this format.
      - raw pre-flatten block: `<cosmos>N N N...</cosmos>` (Step A's own
        output before any flatten script runs) -- chunk_index selects which
        `<cosmos>` block; its own length is used directly (no guessing
        needed, `chunk_tokens` is ignored/optional here).

    In the flattened/atomic format, chunks with <50% cosmos dropout keep-rate
    have gaps (some chunks entirely missing cosmos), so chunk_index there
    means "Nth cosmos chunk present in the stream", not "Nth temporal chunk
    of the video" -- the two only coincide if dropout happened to keep every
    chunk up to that point.
    """
    raw_blocks = _COSMOS_RAW_BLOCK_RE.findall(text)
    if raw_blocks:
        if chunk_index >= len(raw_blocks):
            raise ValueError(f"Requested chunk {chunk_index} but only {len(raw_blocks)} <cosmos> blocks in this record")
        chunk = [int(x) for x in raw_blocks[chunk_index].split() if x.isdigit()]
        _grid_for(len(chunk))  # raises a clear error if this isn't a recognized chunk size
        return chunk

    if chunk_tokens is None:
        raise ValueError(
            "Atomic <cosmos_N> stream has no block boundaries -- pass "
            "chunk_tokens=200 (old/v2 format) or 896 (new/v3+ format) explicitly."
        )
    all_ids = [int(x) for x in _COSMOS_ATOMIC_RE.findall(text)]
    start = chunk_index * chunk_tokens
    end = start + chunk_tokens
    chunk = all_ids[start:end]
    if len(chunk) != chunk_tokens:
        raise ValueError(
            f"Requested chunk {chunk_index} needs tokens [{start}:{end}) but only "
            f"{len(all_ids)} cosmos tokens total in this record."
        )
    return chunk


def _record_text(rec: dict) -> str:
    """Flat records (OmniVideo-100K, Megatron-flattened FineVideo): {"text": ...}.
    Raw FineVideo Step A records (training_ready_rank_*.jsonl) instead nest
    per-activity `video_tokens` under scenes[].activities[] -- concatenate
    all of them for the record so chunk_index can walk the whole video."""
    if "text" in rec:
        return rec["text"]
    if "scenes" in rec:
        return "".join(
            act.get("video_tokens", "")
            for scene in rec["scenes"]
            for act in scene.get("activities", [])
        )
    raise KeyError("Record has neither 'text' nor 'scenes' -- unrecognized schema")


def load_tokens_from_jsonl(path: str, record_id: str, chunk_index: int, chunk_tokens: int = None) -> list:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("video_id", rec.get("id"))
            if rid == record_id:
                return extract_chunk_tokens(_record_text(rec), chunk_index, chunk_tokens=chunk_tokens)
    raise KeyError(f"record_id={record_id!r} not found in {path}")


def decode_cosmos_chunk(token_ids: list, output_path: str, fps: int = 6) -> None:
    """token_ids: a full chunk's raw cosmos codebook indices -- 200 (old/v2,
    square 160x160) or 896 (new/v3+, aspect-preserving 256x448) both
    auto-detected from len(token_ids); see module docstring."""
    grid = _grid_for(len(token_ids))
    output_path = os.path.abspath(output_path)  # resolve before any cwd assumptions below

    # 2026-07-23: cosmos_tokenizer itself is also vendored (tools/decode/vendor/),
    # not pip-installable -- see vendor/cosmos_tokenizer/NOTICE.md. No os.chdir()
    # needed anymore (that was only for PROTOTYPE_DIR's relative lookups).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    import imageio_ffmpeg
    import torch
    import torchvision.transforms as T
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    checkpoint_dec = _resolve_checkpoint_dec()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = CausalVideoTokenizer(checkpoint_dec=checkpoint_dec).to(device)

    indices = torch.tensor(token_ids, dtype=torch.int64, device=device).view(1, *grid)
    with torch.no_grad():
        out = dec.decode(indices)  # (1, 3, T, H, W), range ~[-1, 1]

    out = ((out.float() + 1.0) / 2.0).clamp(0, 1).squeeze(0)  # (3, T, H, W)
    n_frames = out.shape[1]
    frame_h, frame_w = out.shape[2], out.shape[3]

    frame_dir = f"/tmp/cosmos_decode_{os.getpid()}"
    os.makedirs(frame_dir, exist_ok=True)
    to_pil = T.ToPILImage()
    for i in range(n_frames):
        to_pil(out[:, i, :, :].cpu()).save(f"{frame_dir}/frame_{i:02d}.png")

    # Upscale 2x with nearest-neighbor for visibility, preserving the decoded
    # frame's real aspect ratio (grid is no longer guaranteed square -- the
    # old 200-token/square format and the new 896-token/16:9 format have
    # different frame_h/frame_w, so a hardcoded "320:320" would squash the
    # new format's output).
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", f"{frame_dir}/frame_%02d.png",
         "-vf", f"scale={frame_w*2}:{frame_h*2}:flags=neighbor", "-pix_fmt", "yuv420p", output_path],
        check=True, capture_output=True,
    )
    for i in range(n_frames):
        os.remove(f"{frame_dir}/frame_{i:02d}.png")
    os.rmdir(frame_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", help="Comma-separated list of raw cosmos ids (200 = old/v2 format, 896 = new/v3+ format)")
    ap.add_argument("--input-jsonl", help="Flattened JSONL file to pull tokens from")
    ap.add_argument("--record-id", help="video_id/id field to select within --input-jsonl")
    ap.add_argument("--chunk-index", type=int, default=0,
                     help="Which cosmos chunk (in order of appearance) to decode, 0-indexed")
    ap.add_argument("--chunk-tokens", type=int, default=None, choices=[200, 896],
                     help="Chunk size for atomic <cosmos_N> streams with no block boundaries "
                          "(200=old/v2, 896=new/v3+). Not needed for raw <cosmos>...</cosmos> "
                          "block input, whose own length is used directly.")
    ap.add_argument("--fps", type=int, default=6, help="Output mp4 framerate (decoded frames are few, slow fps for visibility)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.tokens:
        token_ids = [int(x) for x in args.tokens.split(",")]
    elif args.input_jsonl and args.record_id:
        token_ids = load_tokens_from_jsonl(args.input_jsonl, args.record_id, args.chunk_index, chunk_tokens=args.chunk_tokens)
    else:
        ap.error("Provide either --tokens or (--input-jsonl and --record-id)")

    decode_cosmos_chunk(token_ids, args.output, fps=args.fps)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
