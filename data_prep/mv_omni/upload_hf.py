#!/usr/bin/env python3
"""
Upload the converted MixtureVitae-Omni (`valid_snac`) dataset to HF.

Source: 6 pre-gzipped shards at
  /e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/flatten/mv_omni/mv_omni_converted/
  mv_omni_snac_{0..5}.jsonl.gz (~30GB total, already gzip -- NOT re-compressed by
  this script, just copied/renamed into train/test dirs)

Original source: HF `mixture-vitae/MixtureVitae-Omni` (`data/data/valid_snac_*.jsonl.gz`),
downloaded to /p/data1/mmlaion/nguyen38/inventory_cache/hf_snac/, then `<seed_N>` ->
`<seed2_N>` converted (SNAC ids already matched our vocab range, no conversion needed
there). See datasets.md section "2. MixtureVitae-Omni (valid_snac)" for the full
investigation. No agent/cosmos content (MV-Omni has no pose/spatial-video channel).

Row count verified 2026-07-26 by decompressing all 6 shards (not from the datasets.md
inventory estimate of "~1.78M", which was wrong): 1,593,301 real rows
(296,651 + 290,169 + 294,657 + 296,095 + 296,365 + 119,364).

Each row: {"text": ..., "metadata": [{"source": ..., "params": {"id": <YouTube id>, ...}}]}
  text example: "Q: Listen to this and tell me what you heard. <listen><snac_N>...</listen>"
Already in flat `{"text": ...}` schema (like our own Phase 7 output) -- no separate
flatten step needed, unlike the hierarchical FineVideo pipeline.

License: mixture-vitae/MixtureVitae-Omni has no documented license confirmation in
this project's records (unlike synth_llava, which has an explicit "confirmed
permissive by Huu directly" note) -- tagged `license: other` per Van Khue's
2026-07-26 decision, pending an explicit confirmation from Huu if a more specific
SPDX tag is ever needed.

Usage:
    export HF_TOKEN='hf_...'
    python data_prep/mv_omni/upload_hf.py --repo-id EmpathicRobotics/MV-Omni
    python data_prep/mv_omni/upload_hf.py --repo-id ... --skip-copy     # reuse existing split dir
    python data_prep/mv_omni/upload_hf.py --repo-id ... --skip-upload  # copy only, no push
"""
import argparse
import os
import shutil

from huggingface_hub import HfApi, login

SOURCE_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/flatten/mv_omni/mv_omni_converted"
UPLOAD_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/flatten/mv_omni/mv_omni_hf_upload"
NUM_SHARDS = 6
# Row counts per shard, verified 2026-07-26 (decompressed all 6 files) -- used only
# for the dataset card, not for the split logic (split is by whole shard, not by row).
SHARD_ROW_COUNTS = [296651, 290169, 294657, 296095, 296365, 119364]
TEST_SHARD_INDEX = 5  # smallest shard (119,364 rows, ~7.5% of total) held out as test

DATASET_CARD = """---
license: other
---

# {repo_name}

Converted [mixture-vitae/MixtureVitae-Omni](https://huggingface.co/datasets/mixture-vitae/MixtureVitae-Omni)
(`valid_snac` split), prepared for the PAB-Spline / omni-modal VLA project
(audio+text modality pair, no pose/video). Used in the `w8_new` mix that
trained `vla-1.7b-qwen3-v6` as the primary non-action/language-backbone
source (39.58% of that mix's total weight).

- **1,593,301 rows** across 6 shards (296,651 / 290,169 / 294,657 / 296,095 /
  296,365 / 119,364) -- verified by full decompression 2026-07-26, correcting
  an earlier inventory estimate of "~1.78M rows" that was never actually counted.
- `<seed_N>` tokens converted to `<seed2_N>` to match this project's vocab
  (SNAC ids already fell inside the project's registered range,
  `128266-148745`, L0 + L1a + L1b bands -- no conversion needed there).
- No `<agent>` (pose) or `<cosmos>` (spatial video) content -- MixtureVitae-Omni
  has no pose/ordinary-video channel, only audio (SNAC) + seed2 (image) + text.
- **License note:** no documented license confirmation for this specific HF
  repo was found in this project's records (unlike `synth_llava`, which has an
  explicit "confirmed permissive by Huu directly" note) -- both `mixture-vitae`
  and `mixture-vitae-backup` are orgs controlled by the project lead, so this
  is presumed to be his own data, but tagged `license: other` (not a specific
  SPDX id) pending an explicit confirmation, per Van Khue's decision 2026-07-26.

## Format

Each row is `{{"text": ..., "metadata": [...]}}`:

```
Q: Listen to this and tell me what you heard. <listen><snac_N><snac_N>...</listen>
```

`metadata` carries source provenance (e.g. YouTube id) from the original
MixtureVitae-Omni pipeline, unchanged.

## Split

Train/test split **by whole shard** (not by row, since shards were already
gzip-compressed upstream and re-splitting by row would require decompressing
+ recompressing ~30GB for no real benefit): shard {test_shard} ({test_rows:,}
rows, ~{test_pct:.1f}% of total) held out as test, remaining {n_train_shards}
shards as train.
"""


def main():
    ap = argparse.ArgumentParser(description="Upload converted MixtureVitae-Omni (valid_snac) dataset to HuggingFace.")
    ap.add_argument("--repo-id", required=True, help="e.g. EmpathicRobotics/MV-Omni")
    ap.add_argument("--source-dir", default=SOURCE_DIR)
    ap.add_argument("--upload-dir", default=UPLOAD_DIR)
    ap.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    ap.add_argument("--test-shard-index", type=int, default=TEST_SHARD_INDEX)
    ap.add_argument("--skip-copy", action="store_true", help="Reuse existing train/test split dir")
    ap.add_argument("--skip-upload", action="store_true", help="Only copy/rename, don't push")
    args = ap.parse_args()

    train_dir = os.path.join(args.upload_dir, "train")
    test_dir = os.path.join(args.upload_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_shards = [f"mv_omni_snac_{i}.jsonl.gz" for i in range(args.num_shards)]
    missing = [f for f in all_shards if not os.path.exists(os.path.join(args.source_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} shards, first: {missing[0]}")
    print(f"All {args.num_shards} shards found.")

    if not args.skip_copy:
        train_i = 0
        for i, name in enumerate(all_shards):
            src = os.path.join(args.source_dir, name)
            if i == args.test_shard_index:
                dst = os.path.join(test_dir, "test-00000-of-00001.jsonl.gz")
            else:
                dst = os.path.join(train_dir, f"train-{train_i:05d}-of-{args.num_shards - 1:05d}.jsonl.gz")
                train_i += 1
            if os.path.exists(dst):
                print(f"  Skipped (exists): {os.path.basename(dst)}")
                continue
            print(f"  Copying {name} -> {os.path.basename(dst)} ...")
            shutil.copyfile(src, dst)
        print("Copy complete.")

    actual_train = len([f for f in os.listdir(train_dir) if f.endswith(".jsonl.gz")])
    actual_test = len([f for f in os.listdir(test_dir) if f.endswith(".jsonl.gz")])
    expected_train = args.num_shards - 1
    if actual_train != expected_train:
        raise ValueError(f"Expected {expected_train} train shards, found {actual_train}")
    if actual_test != 1:
        raise ValueError(f"Expected 1 test shard, found {actual_test}")

    if args.skip_upload:
        print("Skipping upload (--skip-upload). Files in:", args.upload_dir)
        return

    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    repo_name = args.repo_id.split("/")[-1]
    test_rows = SHARD_ROW_COUNTS[args.test_shard_index]
    total_rows = sum(SHARD_ROW_COUNTS)
    readme_path = os.path.join(args.upload_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD.format(
            repo_name=repo_name,
            test_shard=args.test_shard_index,
            test_rows=test_rows,
            test_pct=100 * test_rows / total_rows,
            n_train_shards=args.num_shards - 1,
        ))
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
        commit_message="Upload converted MixtureVitae-Omni (valid_snac) dataset "
                        "(1,593,301 rows, seed2 + listen/SNAC + text, no agent/cosmos)",
        allow_patterns=["train/*.jsonl.gz", "test/*.jsonl.gz", "README.md"],
    )

    print(f"Done! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
