#!/usr/bin/env python3
"""
Upload flattened Megatron-LM shards to EmpathicRobotics/FineVideo-Phase7-Flattened.

  - 160 shards split 95/5 train/test (seed 42)
  - gzip compressed in parallel
  - uploaded via huggingface_hub

v7 (default, 2026-07-26): seed2 + cosmos(50%) + agent + listen + caption + speech, 371,892 records,
  10,926,767,551 real tokens (from the Megatron .idx used to train v6, the training run).
  Re-tokenized audio wrapper <snac>...</snac> (v6) -> <listen>...</listen> (v7), matching the
  w8_new mix -- record count/content otherwise unchanged from v6. Requires
  EmpathicRobotics/tokenizer-vla-qwen3-v2, NOT tokenizer-vla-adaptive-v2 (which only has bare
  <snac>, not atomic <listen>).
  source: w8_new/flatten/finevideo_v6_listen/finevideo_v6_flat_listen/flat_final_vla_adaptive_rank_*.jsonl

v6 (previous): seed2 + cosmos(50%) + agent + snac + caption + speech, 371,892 records, 5.443B tokens
  Rebuilt on top of the fps-mismatch fix in Phase 3/4 (affected ~35% of videos with
  native fps != 30 -- agent/pose windows are now correctly time-aligned) and adds
  explicit modality wrapper tokens (<seed2>...</seed2>, <cosmos>...</cosmos>,
  <agent>...</agent>, <snac>...</snac>) that were previously stripped during flattening.
  Per-chunk temporal ordering: [caption?][seed2?][cosmos?][agent?][snac?][speech?] per 8-frame chunk
  Caption/speech anchored to their exact chunk, 0% dropout, no augmentation
  Whole-activity ### Speech: header unchanged (intentionally redundant with inline <speech>)
  source: megatron_dataset_v6/flat_final_vla_adaptive_rank_*.jsonl (all on /e -- /p (JUWELS)
  is not mounted on JUPITER compute nodes)

Usage:
    export HF_TOKEN='hf_...'
    python tools/upload/upload_flattened_hf.py \
        --source-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/flatten/finevideo_v6_listen/finevideo_v6_flat_listen \
        --upload-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/finevideo_v6_listen_hf_upload
    python tools/upload/upload_flattened_hf.py --skip-compress   # reuse existing .gz files
    python tools/upload/upload_flattened_hf.py --skip-upload     # compress only, no push
"""

import argparse
import gzip
import multiprocessing as mp
import os
import random
import shutil

from huggingface_hub import HfApi, login


REPO_ID = "EmpathicRobotics/FineVideo-Phase7-Flattened"
TOTAL_SHARDS = 160
TEST_RATIO = 0.05
SEED = 42


def compress_worker(args):
    old_path, new_path, new_name = args
    try:
        if not os.path.exists(new_path):
            with open(old_path, "rb") as f_in:
                with gzip.open(new_path, "wb", compresslevel=5) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return f"  Done: {new_name}"
        else:
            return f"  Skipped (exists): {new_name}"
    except Exception as e:
        return f"  Error at {new_name}: {e}"


def process_and_compress(file_list, prefix, target_dir, source_dir):
    total = len(file_list)
    print(f"Compressing {prefix} ({total} files)...")

    tasks = []
    for i, old_name in enumerate(file_list):
        old_path = os.path.join(source_dir, old_name)
        new_name = f"{prefix}-{i:05d}-of-{total:05d}.jsonl.gz"
        new_path = os.path.join(target_dir, new_name)
        tasks.append((old_path, new_path, new_name))

    num_cores = min(mp.cpu_count(), 16)
    print(f"Using {num_cores} cores for parallel compression")

    with mp.Pool(num_cores) as pool:
        for res in pool.imap_unordered(compress_worker, tasks):
            print(res)

    print(f"Finished {prefix}!")


def main():
    parser = argparse.ArgumentParser(
        description="Upload flattened adaptive shards to HuggingFace."
    )
    parser.add_argument(
        "--source-dir",
        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/FineVideo-VLA/megatron_dataset_v6",
    )
    parser.add_argument(
        "--upload-dir",
        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/FineVideo-VLA/hf_upload_flattened_v6",
    )
    parser.add_argument(
        "--shard-prefix",
        default="flat_final_vla_adaptive_rank",
        help="Filename stem before _{i}.jsonl (v5 uses 'flat_final_vla_adaptive_rank')",
    )
    parser.add_argument(
        "--skip-compress", action="store_true",
        help="Skip compression step (reuse existing compressed files)",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Only compress, don't upload.",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.upload_dir, "train")
    test_dir = os.path.join(args.upload_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_shards = [f"{args.shard_prefix}_{i}.jsonl" for i in range(TOTAL_SHARDS)]

    print("Verifying all shards exist...")
    missing = [f for f in all_shards if not os.path.exists(os.path.join(args.source_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} shards, first: {missing[0]}")
    print(f"All {TOTAL_SHARDS} shards found.")

    random.seed(SEED)
    random.shuffle(all_shards)

    test_count = int(TOTAL_SHARDS * TEST_RATIO)
    test_files = all_shards[:test_count]
    train_files = all_shards[test_count:]
    print(f"Train: {len(train_files)} | Test: {len(test_files)}")

    if not args.skip_compress:
        process_and_compress(train_files, "train", train_dir, args.source_dir)
        process_and_compress(test_files, "test", test_dir, args.source_dir)
        print("Compression complete.")

    expected_train = len(train_files)
    expected_test = len(test_files)
    actual_train = len([f for f in os.listdir(train_dir) if f.endswith(".jsonl.gz")])
    actual_test = len([f for f in os.listdir(test_dir) if f.endswith(".jsonl.gz")])
    if actual_train != expected_train:
        raise ValueError(f"Expected {expected_train} train shards, found {actual_train}")
    if actual_test != expected_test:
        raise ValueError(f"Expected {expected_test} test shards, found {actual_test}")

    if args.skip_upload:
        print("Skipping upload (--skip-upload). Files in:", args.upload_dir)
        return

    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])

    print(f"Uploading to {REPO_ID} ...")
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    readme_path = os.path.join(os.path.dirname(__file__), "vla_flattened_dataset_card.md")
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Update dataset card for v7: <snac> -> <listen> wrapper, tokenizer-vla-qwen3-v2, 10.93B tokens",
        )
        print("Uploaded dataset card.")

    api.upload_folder(
        folder_path=args.upload_dir,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload v7: re-tokenized audio wrapper <snac> -> <listen>, matching the w8_new mix (371,892 records, 10.93B real tokens)",
    )

    print(f"Done! https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
