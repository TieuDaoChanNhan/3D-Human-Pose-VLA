#!/usr/bin/env python3
"""
Finalize OmniVideo-100K: join QA text onto each video's merged Step A +
agent-token record (phase6_merge_omnivideo.py's output), keyed by video_id.

One record per video (all QA pairs for that video_id are concatenated and
appended after the video token stream, not split into separate records --
avoids repeating the (potentially large) video token stream once per
question; decided with Van Khue 20/07/2026). Videos with no QA (should not
happen -- QA track covers all 5,214) keep just their video tokens; the
reverse (QA video_id not present in Step A) is logged as a warning rather
than silently dropped.

Usage:
    python data_prep/omnivideo_100k/phase7_finalize_omnivideo.py
"""
import argparse
import glob
import json
import os
from collections import defaultdict

DEFAULT_MERGED_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k/video_agent_merged_w24"
DEFAULT_QA_FILE = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k/qa_flat/omnivideo_100k_qa_flat.jsonl"
DEFAULT_OUTPUT_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k/final_w24"


def load_qa_by_video(qa_file):
    qa_by_video = defaultdict(list)
    with open(qa_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qa_by_video[rec["video_id"]].append(rec["text"])
    return qa_by_video


def finalize_one_file(in_path, output_dir, qa_by_video, skip_existing, video_ids_seen):
    base = os.path.basename(in_path)
    out_path = os.path.join(output_dir, base)
    if skip_existing and os.path.exists(out_path):
        return {"file": base, "skipped": True}

    n_in = n_out = n_with_qa = n_without_qa = 0
    tmp_path = out_path + ".tmp"
    with open(in_path) as fin, open(tmp_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            video_id = d["video_id"]
            video_ids_seen.add(video_id)
            qa_texts = qa_by_video.get(video_id)
            if qa_texts:
                final_text = d["text"] + "\n" + "\n".join(qa_texts)
                n_with_qa += 1
            else:
                final_text = d["text"]
                n_without_qa += 1
            fout.write(json.dumps({"video_id": video_id, "text": final_text}, ensure_ascii=False) + "\n")
            n_out += 1
    os.replace(tmp_path, out_path)
    return {
        "file": base, "n_in": n_in, "n_out": n_out,
        "n_with_qa": n_with_qa, "n_without_qa": n_without_qa,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    ap.add_argument("--qa-file", default=DEFAULT_QA_FILE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading QA from {args.qa_file} ...")
    qa_by_video = load_qa_by_video(args.qa_file)
    print(f"QA loaded: {len(qa_by_video)} unique video_id, "
          f"{sum(len(v) for v in qa_by_video.values())} total QA rows")

    files = sorted(glob.glob(os.path.join(args.merged_dir, "step_a_rank_*.jsonl")))
    print(f"{len(files)} merged files from {args.merged_dir}")

    total_in = total_out = total_with_qa = total_without_qa = 0
    video_ids_seen = set()
    for fp in files:
        stats = finalize_one_file(fp, args.output_dir, qa_by_video, args.skip_existing, video_ids_seen)
        if stats.get("skipped"):
            print(f"{stats['file']}: da co, bo qua")
            continue
        total_in += stats["n_in"]
        total_out += stats["n_out"]
        total_with_qa += stats["n_with_qa"]
        total_without_qa += stats["n_without_qa"]
        print(f"{stats['file']}: {stats['n_in']} -> {stats['n_out']} "
              f"(with_qa: {stats['n_with_qa']}, without_qa: {stats['n_without_qa']})")

    qa_video_ids_not_in_step_a = set(qa_by_video.keys()) - video_ids_seen
    print(f"\nTONG: {total_in} -> {total_out} | with_qa: {total_with_qa} | without_qa: {total_without_qa}")
    if qa_video_ids_not_in_step_a:
        print(f"WARNING: {len(qa_video_ids_not_in_step_a)} video_id co QA nhung khong co trong Step A merged: "
              f"{sorted(qa_video_ids_not_in_step_a)[:10]}{'...' if len(qa_video_ids_not_in_step_a) > 10 else ''}")


if __name__ == "__main__":
    main()
