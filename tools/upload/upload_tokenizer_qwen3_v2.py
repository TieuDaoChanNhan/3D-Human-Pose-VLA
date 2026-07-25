#!/usr/bin/env python3
"""
Upload the promoted VLA tokenizer (Qwen3 base + listen/speak wrapper fix +
MV-Omni's extra SNAC bands) to HuggingFace EmpathicRobotics.

Background (2026-07-23): the production tokenizer (`tokenizer-vla-qwen3`,
106,258 added tokens) predates the `<listen>`/`<speak>` wrapper convention
(decided the same day, replacing the older generic `<snac>` wrapper) -- so
`<listen>`, `</listen>`, `<speak>`, `</speak>` are NOT atomic in it (verified
directly via tokenizers.Tokenizer.encode(), not assumed). It also predates
MixtureVitae-Omni's real SNAC ID range: streaming 50,000 real MV-Omni records
turned up 28,672 unique `<snac_N>` IDs, of which 16,384 (57%) fall in 2 extra
4096-wide bands (136458-144649, 148746-156937) that the production tokenizer
never registered.

This tokenizer (`tokenizer_vla_qwen3_v2`, previously a "_test" build) was
verified 2026-07-23 to have neither gap: all 4 wrapper tags are atomic, and
0/28,672 real MV-Omni SNAC IDs from that same sample are non-atomic. All
106,258 token IDs from the old production tokenizer are unchanged (only
additions, no renumbering) -- confirmed in an earlier session
(PROGRESS_VI.md, tokenizer build verification, same day).

Vocab: 151,643 (Qwen3 base) + 122,918 added VLA tokens = 274,561 real vocab
(vs. 257,901 in the old tokenizer-vla-qwen3 -- +16,660 new tokens, all
additive).

Usage:
    export HF_TOKEN='hf_...'
    python tools/upload/upload_tokenizer_qwen3_v2.py
    python tools/upload/upload_tokenizer_qwen3_v2.py --skip-upload   # dry run, just prints file list
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

TOKENIZER_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/tokenizer_vla_qwen3_v2"
REPO_ID = "EmpathicRobotics/tokenizer-vla-qwen3-v2"

README = """\
---
language: en
tags:
  - vla
  - tokenizer
  - robotics
  - multimodal
  - pose-estimation
  - audio
license: apache-2.0
---

# VLA Tokenizer — Qwen3 v2 (listen/speak wrapper fix + MV-Omni SNAC bands)

Successor to [tokenizer-vla-qwen3](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3)
(used to train `vla-1.7b-qwen3-v2`). Purely additive — every token ID from the
previous tokenizer is unchanged; this only registers new tokens that were
missing.

**Vocab size: 274,561** (151,643 Qwen3 base + 122,918 added VLA tokens, up
from 257,901 / 106,258 in the previous tokenizer — +16,660 new tokens).

## What changed and why

1. **`<listen>`/`</listen>`/`<speak>`/`</speak>` are now atomic.** The
   previous tokenizer was built before the 2026-07-23 decision to replace the
   generic `<snac>`/`</snac>` SNAC wrapper with `<listen>` (ambient/scene
   audio the model perceives — used by FineVideo-VLA and OmniVideo-100K) and
   `<speak>` (a generated reply — used by the emotional-roleplay dataset).
   Verified directly (`tokenizers.Tokenizer.encode()`, not assumed): in the
   old tokenizer these 4 tags split into 3-4 BPE subword pieces each; here
   each encodes to a single token ID.

2. **MixtureVitae-Omni's full SNAC ID range is now covered.** Streaming
   50,000 real records from MV-Omni's `mv_omni_snac_*.jsonl.gz` turned up
   28,672 unique `<snac_N>` IDs. The old tokenizer only registered the 3 bands
   this project's own `tokenize_snac.py` produces (L0/L1a/L1b,
   `128266-132361`/`132362-136457`/`144650-148745`) — 16,384 of the 28,672
   IDs found (57%), spanning 2 further 4096-wide bands
   (`136458-144649`, `148746-156937`), were unregistered and would have
   split into multiple BPE pieces. All 28,672 are atomic here.

Without this tokenizer, essentially all `<listen>`/`<speak>`-wrapped SNAC
audio project-wide (FineVideo-VLA, OmniVideo-100K, emotional-roleplay, and
MixtureVitae-Omni) would silently lose the atomic-token property those
projects' data pipelines assume.

## Compatibility

All 257,901 token IDs from `tokenizer-vla-qwen3` are unchanged (verified —
this is a strict superset, additions only, no renumbering). Text tokenized
with the old tokenizer remains valid under this one.

## Base

[Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) tokenizer,
extended the same way as `tokenizer-vla-qwen3` (see that repo's card for the
full original token-family breakdown: seed2/cosmos/avclm/agent/snac/
caption/speech).

## Related

| Resource | Link |
|----------|------|
| Previous tokenizer | [tokenizer-vla-qwen3](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3) |
| VLA model trained with previous tokenizer | [vla-1.7b-qwen3-v2](https://huggingface.co/EmpathicRobotics/vla-1.7b-qwen3-v2) |
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-upload", action="store_true", help="Dry run: list files, don't push to HF.")
    args = ap.parse_args()

    files = sorted(os.listdir(TOKENIZER_DIR))
    print(f"Source: {TOKENIZER_DIR}")
    print(f"Files: {files}")

    if args.skip_upload:
        print("(--skip-upload) Not uploading.")
        return

    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    api = HfApi()
    print(f"\n=== Uploading {REPO_ID} ===")
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    with tempfile.TemporaryDirectory() as tmp:
        for fname in files:
            shutil.copy2(os.path.join(TOKENIZER_DIR, fname), tmp)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(README)

        api.upload_folder(
            folder_path=tmp,
            repo_id=REPO_ID,
            repo_type="model",
            create_pr=False,
            commit_message="Upload VLA tokenizer v2 (listen/speak wrapper fix + MV-Omni SNAC bands)",
        )

    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
