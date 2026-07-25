#!/usr/bin/env python3
"""
Download Jyun-Ting/Harmony4D from HuggingFace Hub.

Multi-view video dataset of close human-human interactions (wrestling/
grappling, dancing, MMA, karate, sword fighting, hugging) -- 208 sequences,
24 subjects, always 2 people per scene. 3D pose (17-joint, world-metric) +
SMPL mesh with contact-aware fitting for occlusion. License: MIT (verified
directly on the HF repo card, 21/07/2026) -- chosen specifically to fill two
gaps found in our own FineVideo pose pipeline: the YOLO occlusion filter
drops 45.9% of windows, and the pipeline only keeps the single
most-confident bounding box per frame (no multi-person support). See
PROGRESS_VI.md's 21/07/2026 pivot entry and
`.claude/.../memory/project_pivot_pose_dataset_sourcing.md` for the full
decision trail. JRDB-Pose3D was a technically closer fit but rejected
outright (CC BY-NC-SA, non-commercial) -- this project only takes
permissive-licensed data.

Verified via HF tree API 21/07/2026:
  train/  -- 15 zip files, ~287GB (01_hugging.zip .. 15_mma4.zip, named
             ##_<category>[_partN].zip; categories: hugging, grappling,
             sword, ballroom, karate, mma)
  test/   -- 7 zip files, ~65GB (a subset of the same categories)
  README.md is only 24 bytes -- no dataset-card documentation on HF itself,
  structure/format must be learned from the actual paper (arXiv 2410.20294)
  and/or by inspecting a downloaded zip directly.
Total: ~352GB.

Must run from a login node with internet access (confirmed reachable from
this JUPITER login node -- see download_synth_llava.py's note; the older
JUWELS-only assumption in download_omnivideo_100k.py/download_robovqa.py is
outdated). Target dir is on /e (JUPITER's own storage), NOT /p -- verified
21/07/2026 via a real `srun` job on the booster partition that Jupiter
compute nodes cannot see /p at all (only the login node can), and all
processing this dataset will need (unzip, SMPL->17-joint conversion, 20fps->
30fps resample) runs as SLURM jobs on Jupiter/booster. See project memory
`feedback_data_storage_location` -- downloading straight to /e avoids a
352GB copy tax later. Resumable: snapshot_download skips files already
complete in local_dir, and this script retries on transient network errors
-- safe to re-run / leave in a tmux session.

352GB is large -- consider downloading a single small file first to inspect
structure before committing to the full pull, e.g.:
    python3 -c "
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id='Jyun-Ting/Harmony4D', repo_type='dataset',
                     filename='test/01_hugging.zip', local_dir='/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d')
    "
(test/01_hugging.zip is the smallest file in the repo at 1.7GB.)

Usage:
    export HF_TOKEN='hf_...'
    python3 tools/extract/download_harmony4d.py                # full download (~352GB)
    python3 tools/extract/download_harmony4d.py --sample-only   # just the 1.7GB sample above
"""
import argparse
import os
import time

os.environ.setdefault("HF_HOME", "/e/data1/datasets/playground/mmlaion/shared/nguyen38/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet backend flaky on this cluster, see project memory

from huggingface_hub import login, snapshot_download, hf_hub_download

REPO_ID = "Jyun-Ting/Harmony4D"
LOCAL_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/harmony4d"
MAX_WORKERS = 16
RETRY_DELAY_SEC = 30
SAMPLE_FILE = "test/01_hugging.zip"  # smallest file in the repo, 1.7GB


def report_progress(local_dir):
    if not os.path.isdir(local_dir):
        print("  (nothing downloaded yet)", flush=True)
        return
    zips_done = []
    for root, _dirs, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".zip"):
                zips_done.append(os.path.join(root, f))
    size_gb = sum(os.path.getsize(f) for f in zips_done) / 1e9
    print(f"  progress: {len(zips_done)} zip files, {size_gb:.1f} GB on disk so far", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-only", action="store_true",
                    help="Download only the smallest file (test/01_hugging.zip, 1.7GB) to inspect structure.")
    args = p.parse_args()

    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
    login(token=os.environ["HF_TOKEN"])

    os.makedirs(LOCAL_DIR, exist_ok=True)

    if args.sample_only:
        print(f"Sample-only mode: downloading {SAMPLE_FILE} (1.7GB) -> {LOCAL_DIR}", flush=True)
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=SAMPLE_FILE,
                         local_dir=LOCAL_DIR, token=os.environ["HF_TOKEN"])
        print("Sample download complete.", flush=True)
        return

    print(f"Target: {LOCAL_DIR}  (~352GB total: 15 train zips + 7 test zips)", flush=True)

    attempt = 0
    while True:
        attempt += 1
        print(f"\n[attempt {attempt}] snapshot_download starting -> {LOCAL_DIR}", flush=True)
        report_progress(LOCAL_DIR)
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                max_workers=MAX_WORKERS,
            )
            print("snapshot_download completed successfully.", flush=True)
            break
        except Exception as e:
            print(f"[attempt {attempt}] failed: {e!r} -- retrying in {RETRY_DELAY_SEC}s", flush=True)
            report_progress(LOCAL_DIR)
            time.sleep(RETRY_DELAY_SEC)

    report_progress(LOCAL_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
