#!/usr/bin/env python3
"""
Upload the flattened Harmony4D agent-token dataset to HF.

2026-07-26 update: re-windowed at **window=8** (was window=24), matching the
w8_new mix used to train v6 -- see `data_prep/harmony4d/phase3_normalize.py`'s
--window-size/--stride flags and `flatten_harmony4d.py`'s --input-dir/
--output-path flags, both added 2026-07-25/26 specifically so window=24 and
window=8 outputs can coexist (never overwrite, see
feedback_never_delete_dataset_versions memory). Oversample factor unchanged
at 20x (the compounding-with-Megatron issue found 2026-07-26 was fixed via
a config weight change, not by reducing this file's own oversample -- see
project_harmony4d_oversampling memory).

Source: single file at
  /e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/harmony4d/harmony4d_flat_w8.jsonl
  (8,320 rows = 416 tracks x 20x oversample, unchanged from the window=24 run --
  ~15,254 whitespace-split tokens/row, ~2.85x the window=24 run's ~5,344/row,
  consistent with window=8 producing ~3x more windows/track than window=24.
  See project_harmony4d_oversampling memory + PROGRESS_VI.md 2026-07-23/26 for
  the full pipeline: normalize-only Phase 3 (no monocular filters -- ground-truth
  multi-camera data, 416/416 tracks pass clean, vs FineVideo's 44%), Phase 5
  adaptive-PCHIP tokenize, Qwen2.5-VL caption + category text, 20x oversample.)

Each row: {"id": <track_id + "_os" + oversample_idx>, "text": <flattened record>}
  USER: <category text + VLM caption> ASSISTANT: <agent> <fps_30> <pelvis>...</agent>

Text-to-pose only -- no video/cosmos/seed2 channel (Harmony4D has no ordinary-
camera footage, only the multi-camera Aria capture rig).

Unlike laion_emotional_roleplay's 14 pre-sharded input files, this dataset is
a single flat file -- this script splits it into shards itself (by row, not
by source file).

Usage:
    export HF_TOKEN='hf_...'
    python data_prep/harmony4d/upload_hf.py --repo-id EmpathicRobotics/harmony4d-flattened
    python data_prep/harmony4d/upload_hf.py --repo-id ... --skip-compress   # reuse existing .gz
    python data_prep/harmony4d/upload_hf.py --repo-id ... --skip-upload     # compress only
"""
import argparse
import gzip
import json
import os
import random

from huggingface_hub import HfApi, login

SOURCE_FILE = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/harmony4d/harmony4d_flat_w8.jsonl"
UPLOAD_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/harmony4d/harmony4d_flat_w8_hf_upload"
NUM_SHARDS = 8
TEST_RATIO = 0.06  # ~1/16 tracks held out (split by base track, not by oversampled row -- see note below)
SEED = 42

DATASET_CARD = """---
license: cc-by-4.0
---

# {repo_name}

**2026-07-26 update: re-windowed at window=8** (was window=24), to match
`w8_new` -- the mix used to train v6 after v2 (also window=8) beat every
window=24 successor on every metric measured. Same 416 tracks, same 20x
oversample, same category+VLM-caption instruction text -- only the pose
windowing granularity changed (24-frame windows -> 8-frame windows, so ~3x
more windows/track are now concatenated per row).

Flattened agent-token (3D pose) dataset from [Harmony4D](https://jyuntins.github.io/harmony4d/)
(multi-person, multi-camera close-interaction motion capture -- hugging,
martial arts, sword fighting, ballroom dancing), prepared for the PAB-Spline /
omni-modal VLA project to fill an occlusion / multi-person gap FineVideo-VLA's
monocular YouTube pipeline can't cover (FineVideo drops ~56% of windows to
occlusion/hallucination filters; Harmony4D is ground-truth multi-camera data,
needs none of those filters, 416/416 tracks pass clean).

- **8,320 rows** = 416 source tracks x 20x oversample (id suffix `_os0`..`_os19`)
  -- oversampled because 416 tracks / ~2.8M raw agent tokens is tiny next to
  the rest of the training mix; see `project_harmony4d_oversampling` project
  note for the full reasoning (20x is a starting point, not yet tuned against
  final mix token budget)
- Instruction = category text (e.g. "Two people hugging each other...") +
  Qwen2.5-VL-3B-Instruct caption, combined (category alone is always
  accurate; VLM caption adds detail but sometimes generic -- combining both
  was the deliberate choice over replacing one with the other)
- **Text-to-pose only** -- no video/cosmos/seed2 (Harmony4D has no ordinary
  single-camera footage)
- Same adaptive-PCHIP 17-joint agent-token scheme as FineVideo-VLA
  (`<fps_30> <pelvis> <pelvis_t_N> <pelvis_x_N> ... </pelvis> ...`)
- Pose pipeline: **normalize-only** (`KinematicPreprocessor.split_root_motion()`
  + `.normalize_bone_lengths()` + `create_windows()`) -- deliberately skips
  the hallucination/ID-switch/temporal-smooth/kinematic-anomaly/stiff-leg
  filters FineVideo's monocular pipeline needs, since those are tuned for
  monocular-estimation error and don't apply to Harmony4D's multi-camera
  ground truth

## Format

Each row is `{{"id": ..., "text": ...}}` with `text`:

```
USER: <instruction> ASSISTANT:
<agent> <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_N> ... </pelvis> ...(17 joints)... </agent>
```

## Split

Train/test split by **base track** (not by oversampled row -- all 20 `_osN`
copies of a track stay together on the same side), seed 42, ~{test_ratio:.0%}
of base tracks held out as test, {num_shards} shards.
"""


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def base_track_id(row_id):
    # e.g. "01_hugging_001_hugging_aria01_os0" -> "01_hugging_001_hugging_aria01"
    idx = row_id.rfind("_os")
    return row_id[:idx] if idx != -1 else row_id


def write_shards(rows, out_dir, prefix, num_shards):
    os.makedirs(out_dir, exist_ok=True)
    shard_files = [
        gzip.open(os.path.join(out_dir, f"{prefix}-{i:05d}-of-{num_shards:05d}.jsonl.gz"), "wt")
        for i in range(num_shards)
    ]
    try:
        for i, row in enumerate(rows):
            shard_files[i % num_shards].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for f in shard_files:
            f.close()


def main():
    ap = argparse.ArgumentParser(description="Upload Harmony4D flattened dataset to HuggingFace.")
    ap.add_argument("--repo-id", required=True, help="e.g. EmpathicRobotics/harmony4d-flattened")
    ap.add_argument("--source-file", default=SOURCE_FILE)
    ap.add_argument("--upload-dir", default=UPLOAD_DIR)
    ap.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    ap.add_argument("--skip-compress", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    train_dir = os.path.join(args.upload_dir, "train")
    test_dir = os.path.join(args.upload_dir, "test")

    if not args.skip_compress:
        print(f"Loading {args.source_file} ...")
        rows = load_rows(args.source_file)
        print(f"{len(rows)} rows loaded")

        base_ids = sorted({base_track_id(r["id"]) for r in rows})
        random.seed(SEED)
        shuffled = base_ids[:]
        random.shuffle(shuffled)
        test_count = max(1, round(len(base_ids) * TEST_RATIO))
        test_bases = set(shuffled[:test_count])

        train_rows = [r for r in rows if base_track_id(r["id"]) not in test_bases]
        test_rows = [r for r in rows if base_track_id(r["id"]) in test_bases]
        print(f"Train: {len(train_rows)} rows ({len(base_ids) - test_count} base tracks) | "
              f"Test: {len(test_rows)} rows ({test_count} base tracks)")

        print("Writing train shards...")
        write_shards(train_rows, train_dir, "train", args.num_shards)
        print("Writing test shards...")
        write_shards(test_rows, test_dir, "test", max(1, args.num_shards // 4))

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
        f.write(DATASET_CARD.format(repo_name=repo_name, test_ratio=TEST_RATIO, num_shards=args.num_shards))
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
        commit_message="Upload Harmony4D flattened agent-token dataset, window=8 (8,320 rows, 416 tracks x 20x oversample)",
        allow_patterns=["train/*.jsonl.gz", "test/*.jsonl.gz", "README.md"],
    )

    print(f"Done! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
