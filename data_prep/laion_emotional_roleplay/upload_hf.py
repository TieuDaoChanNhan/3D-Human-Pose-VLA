#!/usr/bin/env python3
"""
Upload the SNAC-tokenized laion/emotional-roleplay-finetuning-dataset to HF.

Source: 14 shards at
  /e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/laion_emotional_roleplay_flattened_speak/
  roleplay_snac_speak_flat_{00000..00013}.jsonl
  (67,459/67,459 rows, 0 errors -- see PROGRESS_VI.md 2026-07-23 entry.)

Each row: {"id": <source row id>, "text": <flattened training record>}
  USER: <text> [Voice: <voice_description>] ASSISTANT:
  <speak> <snac_N> <snac_N> ... </snac_N> </speak>

speak-format = 7 tokens per 12.5Hz base frame (all 3 SNAC hierarchical
codebook levels, Leo/Orpheus offset scheme), replacing the older
listen-format upload (3 tok/frame, L2 zero-filled) from 2026-07-20.
`<speak>` marks this audio as the model's own generated turn (role tag, not
a format/voice-clone distinction -- see PROGRESS_VI.md 2026-07-23 for the
listen vs speak design decision).

Usage:
    export HF_TOKEN='hf_...'
    python data_prep/laion_emotional_roleplay/upload_hf.py --repo-id EmpathicRobotics/<name-you-pick>
    python data_prep/laion_emotional_roleplay/upload_hf.py --repo-id ... --skip-compress   # reuse existing .gz
    python data_prep/laion_emotional_roleplay/upload_hf.py --repo-id ... --skip-upload     # compress only
"""
import argparse
import gzip
import os
import random
import shutil

from huggingface_hub import HfApi, login

SOURCE_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/laion_emotional_roleplay_flattened_speak"
UPLOAD_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/laion_emotional_roleplay_hf_upload_speak"
SHARD_PREFIX = "roleplay_snac_speak_flat"
TOTAL_SHARDS = 14
TEST_RATIO = 0.07  # 1/14 shards held out
SEED = 42

DATASET_CARD = """---
license: cc-by-4.0
---

# {repo_name}

SNAC-tokenized version of [laion/emotional-roleplay-finetuning-dataset]\
(https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset),
prepared for the PAB-Spline / omni-modal VLA project (audio<->text modality
pair, per Huu's instruction: "concatenate the text and interleave with snac
and/or moss tokens").

**2026-07-23 update: re-tokenized in speak-format**, replacing the original
listen-format upload. speak-format encodes all 3 SNAC hierarchical codebook
levels (7 tokens/12.5Hz base frame, ~87.5 tok/s) using the same token-offset
scheme (`OFFSET_L2 = [136458, 140554, 148746, 152842]`, interleaved order
`L0, L1a, L2_0, L2_1, L1b, L2_2, L2_3`) as the production Leo/Orpheus SNAC
pipeline, vs. the old listen-format's 3 tok/frame with the finest codebook
level dropped entirely. Every row is now wrapped in `<speak>...</speak>`
(role tag: this audio is always the model's own generated turn, independent
of whether the source voice has been voice-cloned).

- **67,459 rows** (67,491 source rows minus 32 with out-of-range `adherence_score`)
- **54,578,440 SNAC tokens** total, `hubertsiuzdak/snac_24khz`, speak-format
  encoding (7 tokens per 12.5Hz base frame -> 87.5 tok/s)
- Source audio: synthetic TTS (MOSS-TTS-Local v1.5 fine-tune), mono 24kHz MP3,
  German-majority multilingual, ~184 hours total

## Format

Each row is `{{"id": ..., "text": ...}}` with `text`:

```
USER: <text> [Voice: <voice_description>] ASSISTANT:
<speak> <snac_N> <snac_N> ... </speak>
```

`voice_description` is the judge-verified (audio-listened) description of the
realized voice; `instruction`/`req_*` intent fields are intentionally dropped
(source README's Limitations section notes the model does not reliably follow
them -- see source dataset card for detail).

## Known data-quality note

103/67,459 rows (0.15%) have audio noticeably shorter than their text would
imply (as low as 0.16s for a full sentence) -- verified as truncated/corrupted
audio in the **source** parquet (e.g. an 813-byte MP3 that ffprobe confirms is
genuinely 0.14s), not a tokenization bug. Left in as-is; filter by token count
before training if this matters for your use case.

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
    ap = argparse.ArgumentParser(description="Upload roleplay SNAC dataset to HuggingFace.")
    ap.add_argument("--repo-id", required=True, help="e.g. EmpathicRobotics/roleplay-snac-audio")
    ap.add_argument("--source-dir", default=SOURCE_DIR)
    ap.add_argument("--upload-dir", default=UPLOAD_DIR)
    ap.add_argument("--skip-compress", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    train_dir = os.path.join(args.upload_dir, "train")
    test_dir = os.path.join(args.upload_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_shards = [f"{SHARD_PREFIX}_{i:05d}.jsonl" for i in range(TOTAL_SHARDS)]
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
        commit_message="Upload SNAC-tokenized laion/emotional-roleplay-finetuning-dataset "
                        "(67,459 rows, 23.39M SNAC tokens)",
        allow_patterns=["train/*.jsonl.gz", "test/*.jsonl.gz", "README.md"],
    )

    print(f"Done! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
