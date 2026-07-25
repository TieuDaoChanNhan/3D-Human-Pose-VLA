#!/usr/bin/env python3
"""Second-pass filter on top of select_sports_subset.py's sports_subset_video_ids.txt:
excludes videos whose video_summary explicitly flags animated/cartoon content.

Motivation (found while investigating the 2 lowest-yield videos in the 24-video
Phase 1 pilot, job 976467): select_sports_subset.py's keyword match ("dancing",
"running", etc.) fires on generic verbs regardless of context, so it lets through
some animated content (a review video mentioning "a clip of a dancing Robot", a
fairy-tale cartoon short with characters "running away"). Worse than the obvious
zero-detection case: one pilot video (Ncl93lkMpJM, an animated music video about
cartoon dinosaur characters per its own video_summary) still got 56.3% frame-level
HRNet "person" detection -- Faster-RCNN/HRNet can false-positive on stylized
anthropomorphic characters, producing keypoints with real-human confidence scores
but no real-human body proportions. MotionBERT (trained on Human3.6M, real humans
only) lifting those would produce silently wrong 3D pose, not just missing data --
worse for training than a clean skip.

This filter is intentionally narrow (matches only when the source's own summary
says the content IS animated/cartoon/CGI) and will NOT catch every animated video
-- e.g. character-driven fairy-tale content whose summary just uses character
names ("Redhead Girl", "Peter Pan Boy") without ever saying "animated" won't
match. Frame-level detection-rate stats from the real Phase 1 run remain the
final backstop for whatever this text heuristic misses.

Usage:
    python data_prep/omnivideo_100k/dataset_prep/filter_animation_content.py
"""

import json
import re

CAPTIONS_JSONL = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/omnivideo_100k/omnivideo_100k_segment_captions.jsonl"
INPUT_IDS = "data_prep/omnivideo_100k/sports_subset_video_ids.txt"
OUTPUT_IDS = "data_prep/omnivideo_100k/sports_subset_video_ids_filtered.txt"

ANIMATION_PATTERN = re.compile(
    r"\b(animat(?:ed|ion)|cartoon|anime|CGI|computer-generated|claymation|stop-motion)\b",
    re.IGNORECASE,
)


def main():
    sports_ids = set(line.strip() for line in open(INPUT_IDS) if line.strip())
    print(f"Input sports subset: {len(sports_ids)}")

    excluded = []
    kept = []
    with open(CAPTIONS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            vid = d["video_id"]
            if vid not in sports_ids:
                continue
            summary = d.get("video_summary", "")
            m = ANIMATION_PATTERN.findall(summary)
            if m:
                excluded.append((vid, m))
            else:
                kept.append(vid)

    with open(OUTPUT_IDS, "w") as f:
        f.write("\n".join(sorted(kept)) + "\n")

    print(f"Excluded (matched animation/cartoon keyword): {len(excluded)}")
    print(f"Remaining: {len(kept)}")
    print(f"Written to: {OUTPUT_IDS}")


if __name__ == "__main__":
    main()
