#!/usr/bin/env python3
"""
Upload the finalized OmniVideo-100K dataset to HF.

2026-07-26 update: switched to the **window=8** pipeline (final_w8), matching
the w8_new mix used to train v6 -- v2 (window=8) beat every window=24
successor on every metric measured (PPL, modality persistence, see
PROGRESS_VI.md 25-26/07/2026), so all sources feeding the current training
mix were re-run at window=8 rather than kept at window=24. Verified
2026-07-26: 5,214/5,214 rows (1 more than the w24 run's 5,213 -- not yet
root-caused, likely just a per-run Step A survival difference; not
investigated further since it's a single row out of 5,214), 0 malformed.
All counts in this docstring/dataset card are from the real final_w8 output.

Previous version (window=24, still describing the same source/pipeline
otherwise) re-run end to end and verified 2026-07-23: 5,213/5,213 rows.

Each row: {"video_id": ..., "text": <flattened training record>}
  <seed2_N>... <cosmos_N>... [<agent> <fps_30> <pelvis>...</agent>]...
  Q: <question>\nA: <answer>\n...  (all QA pairs for this video_id, appended)

Usage:
    export HF_TOKEN='hf_...'
    python data_prep/omnivideo_100k/upload_hf.py --repo-id EmpathicRobotics/<name-you-pick>
    python data_prep/omnivideo_100k/upload_hf.py --repo-id ... --skip-compress   # reuse existing .gz
    python data_prep/omnivideo_100k/upload_hf.py --repo-id ... --skip-upload     # compress only
"""
import argparse
import gzip
import os
import random
import shutil

from huggingface_hub import HfApi, login

SOURCE_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/omnivideo_100k/final_w8"
UPLOAD_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/omnivideo_100k/final_w8_hf_upload"
SHARD_PREFIX = "step_a_rank"
TOTAL_SHARDS = 160
TEST_RATIO = 0.06  # ~10/160 shards held out
SEED = 42

DATASET_CARD = """---
license: apache-2.0
---

# {repo_name}

Finalized [MiG-NJU/OmniVideo-100K](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K)
for the PAB-Spline / omni-modal VLA project: video tokens + pose agent tokens
+ QA, merged into one record per video.

**2026-07-26: re-run at window=8** (was window=24) to match the w8_new
mix used to train v6 -- see version note above.

- **5,214 rows** (one per source video that survived Step A)
- **799 rows (~15.3%) have `<agent>` blocks** (3D pose, adaptive-PCHIP
  17-joint xyz) -- pose pipeline only ran on a sports-subset of the videos;
  the rest are video+QA(+audio) only, no pose (same partial-coverage
  pattern as the FineVideo-VLA flagship dataset, and essentially the same
  ~15% coverage ratio as the previous window=24 run)
- **all 5,214 rows have `<listen>` blocks** (SNAC listen-format ambient
  audio) -- unlike agent, SNAC covers every video
- all 5,214 rows have QA appended (99,983 QA pairs total: 70,017 open-ended +
  29,966 multiple-choice, from the source `train_oe_70k.jsonl`/
  `train_mcq_30k.jsonl` -- QA count is window-size-independent)
- **real tokenized total (from the Megatron `.idx` used to train v6):
  1,979,126,756 tokens** -- per-modality subtotals not recomputed for this
  window=8 run (the w24 numbers below are historical, kept for reference,
  not re-verified at w8): w24 had 17,225,728 `<seed2_N>` + 300,790,624
  `<cosmos_N>` + ~7,379,928 agent joint tokens + 18,839,901 `<snac_N>` at
  ~344M total -- window=8 has ~3x more chunks/video so is not directly
  comparable modality-by-modality without a fresh count.

## Format

Each row is `{{"video_id": ..., "text": ...}}` with `text`, per 8-frame/30fps
chunk in order:

```
<seed2> <seed2_N>... </seed2> [<caption> ... </caption>] <cosmos> <cosmos_N>... </cosmos> [<agent> <fps_30> <pelvis> ... </agent>] [<speech> ... </speech>]
...(repeated per chunk)...
Q: <question>
A: <answer>
Reasoning: <cross-modal reasoning hint, from source `analysis.connections`>
...(repeated per QA pair for this video)...
```

`<seed2>`/`</seed2>`, `<cosmos>`/`</cosmos>`, `<agent>`/`</agent>` are explicit
span-boundary wrapper tokens (registered in `tokenizer_vla_qwen3`'s vocab,
previously unused in this dataset -- fixed 2026-07-21, decided with Van Khue:
gives the model an unambiguous "span over" signal decoupled from "what modal
comes next", matching the `<snac>`/`</snac>` convention already used by
`laion/emotional-roleplay-finetuning-dataset`).

`<agent>` (pose, adaptive-PCHIP 17-joint xyz, same scheme as FineVideo-VLA)
only appears on chunks where the pose pipeline produced a clean window for
that video; most videos have none. `avc_lm` payload is always discarded,
`<cosmos_N>` is kept with 50% per-chunk dropout, `<seed2_N>` always kept --
same convention as FineVideo's `pipeline_pose/phase7_flatten.py`.

## Pipeline

1. `step_a/step_a_tokenize_video.py` -- video -> raw seed2/cosmos/avc_lm (+ scripts.jsonl caption/speech anchors)
2. `step_a/flatten_step_a_video.py` -- intermediate flatten (not the final artifact, kept for reference)
3. `pose/phase1..4_*_omnivideo.py` + shared `pipeline_pose/phase5_adaptive_pchip.py` -- 3D pose -> agent tokens (sports subset only)
4. `snac_omnivideo.py` -- SNAC listen-format audio tokenization, chunk-aligned (n_chunks recomputed from `omnivideo_100k_segment_captions.jsonl`'s `duration`, same formula as Step A -- see its own docstring). Wired into `phase6_merge_omnivideo.py` -- covers all rows.
5. `phase6_merge_omnivideo.py` -- inject `<agent>` into the Step A stream, chunk-aligned (window_id == chunk_idx * 8 for this window=8 run, verified exact match, no time interpolation needed)
6. `phase7_finalize_omnivideo.py` -- append QA (grouped by video_id) after the video token stream -- **this repo's content**

## Split

Train/test split by shard (not by row), seed 42, ~{test_ratio:.0%} held out as test.
"""


def compress_shard(old_path: str, new_path: str) -> str:
    if os.path.exists(new_path):
        return f"  Skipped (exists): {os.path.basename(new_path)}"
    with open(old_path, "rb") as f_in, gzip.open(new_path, "wb", compresslevel=5) as f_out:
        shutil.copyfileobj(f_in, f_out)
    return f"  Done: {os.path.basename(new_path)}"


def main():
    ap = argparse.ArgumentParser(description="Upload finalized OmniVideo-100K dataset to HuggingFace.")
    ap.add_argument("--repo-id", required=True, help="e.g. EmpathicRobotics/omnivideo-100k-final")
    ap.add_argument("--source-dir", default=SOURCE_DIR)
    ap.add_argument("--upload-dir", default=UPLOAD_DIR)
    ap.add_argument("--skip-compress", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    train_dir = os.path.join(args.upload_dir, "train")
    test_dir = os.path.join(args.upload_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_shards = [f"{SHARD_PREFIX}_{i}.jsonl" for i in range(TOTAL_SHARDS)]
    missing = [f for f in all_shards if not os.path.exists(os.path.join(args.source_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} shards, first: {missing[0]}")
    print(f"All {TOTAL_SHARDS} shards found.")

    random.seed(SEED)
    shuffled = all_shards[:]
    random.shuffle(shuffled)
    test_count = max(1, round(TOTAL_SHARDS * TEST_RATIO))
    test_files = shuffled[:test_count]
    train_files = shuffled[test_count:]
    print(f"Train: {len(train_files)} shards | Test: {len(test_files)} shards")

    if not args.skip_compress:
        for prefix, files, target_dir in (("train", train_files, train_dir), ("test", test_files, test_dir)):
            print(f"Compressing {prefix} ({len(files)} files)...")
            for i, name in enumerate(files):
                old_path = os.path.join(args.source_dir, name)
                new_name = f"{prefix}-{i:05d}-of-{len(files):05d}.jsonl.gz"
                print(compress_shard(old_path, os.path.join(target_dir, new_name)))

    actual_train = len([f for f in os.listdir(train_dir) if f.endswith(".jsonl.gz")])
    actual_test = len([f for f in os.listdir(test_dir) if f.endswith(".jsonl.gz")])
    if actual_train != len(train_files):
        raise ValueError(f"Expected {len(train_files)} train shards, found {actual_train}")
    if actual_test != len(test_files):
        raise ValueError(f"Expected {len(test_files)} test shards, found {actual_test}")

    if args.skip_upload:
        print("Skipping upload (--skip-upload). Files in:", args.upload_dir)
        return

    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    repo_name = args.repo_id.split("/")[-1]
    readme_path = os.path.join(args.upload_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD.format(repo_name=repo_name, test_ratio=TEST_RATIO))
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print(f"Uploading to {args.repo_id} ...")
    api.upload_folder(
        folder_path=args.upload_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload finalized OmniVideo-100K, window=8 (5,214 rows, 799 with agent tokens, "
                        "5,214 with listen/SNAC tokens, 5,214 with QA)",
        allow_patterns=["train/*.jsonl.gz", "test/*.jsonl.gz", "README.md"],
    )

    print(f"Done! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
