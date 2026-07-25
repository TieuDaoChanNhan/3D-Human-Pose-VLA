#!/usr/bin/env python3
"""
Upload VLA 1.7B Qwen3 v2 model to HuggingFace.

Usage:
    export HF_TOKEN=hf_...
    python tools/upload/upload_vla_v2_model.py
"""

import os
import tempfile
import shutil

from huggingface_hub import HfApi

REPO_ID = "EmpathicRobotics/vla-1.7b-qwen3-v2"
# 2026-07-23: output_vla moved project1 -> data1 (freed inodes for the
# project1 quota crisis) -- verified byte-for-byte match before the project1
# copy was deleted, see PROGRESS_VI.md same-day entry.
MODEL_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/qwen3_1.7b_vla_v2/hf/iter_0007632"

README = """\
---
license: apache-2.0
language:
  - en
tags:
  - robotics
  - vla
  - vision-language-action
  - 3d-pose
  - qwen3
  - megatron
  - multimodal
pipeline_tag: text-generation
library_name: transformers
---

# VLA 1.7B — Qwen3 v2

A 1.7B parameter Vision-Language-Action model, migrated to a **Qwen3** backbone and
trained on a **5-source, ~32B-token multimodal mix** (video, 3D pose, audio, image+caption).
This is the project's first Qwen3-based VLA model, and the first trained after fixing
the "stuck in one modality" failure mode found in the previous model.

## Key facts

| | |
|---|---|
| **Architecture** | Qwen3 (28 layers, hidden 2048, intermediate 6144, 16 attn heads / 8 KV heads (GQA), qk-layernorm, RoPE θ=1e6, tied embeddings) |
| **Parameters** | 1.94B (including embeddings for 257,920 vocab) |
| **Vocab size** | 257,920 (Qwen3 base ~151,669 + 106,232 VLA tokens, padded) |
| **Tokenizer** | [EmpathicRobotics/tokenizer-vla-qwen3](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3) |
| **Training data** | ~32.01B tokens across 5 sources: FineVideo-VLA v6, MixtureVitae-Omni, OmniVideo-100K, synth-llava, emotional-roleplay |
| **Training** | 7,632 iters (1 epoch), 64 nodes × 4 GH200 GPUs, global batch 1024, seq len 4096 |
| **Final loss** | Train: 1.694, Val: 1.7526 (PPL 5.77), Test: 1.7722 (PPL 5.88) |
| **Precision** | bf16 |
| **Context length** | 4,096 tokens |

## What this model does

Given a text prompt (activity description, image seed2 block, or partial modality
sequence), the model generates an interleaved multimodal token sequence spanning
6 categories it was trained on:

```
<seed2_N> ...                          # 1 FPS semantic image/video keyframes (vocab 8192)
<cosmos_N> ... </cosmos>               # 8-frame spatial video tokens (vocab 64000)
<snac_N> ... </snac>                   # SNAC audio codec tokens (12,288)
<speech> ... </speech>                 # inline spoken-dialogue text
<caption> ... </caption>               # inline visual caption text
<agent> <fps_30> <pelvis> ... </agent> # 3D human pose, 17 H36M joints
```

## Progress vs. the previous model

The first model ([vla-1.7b-pab-spline-adaptive](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive))
passed agent-completion but **failed modality transitions**: it stayed in `seed2` mode and
never transitioned to `cosmos`/`avclm`/`agent` from text alone. This model no longer has
that failure mode — it transitions freely across all 6 trained categories, in both
greedy and sampled decoding.

Strongest evidence: given **only** 32 real `<seed2_N>` tokens from a held-out image record
(no other text hint), the model generated a topically-correct caption closely matching the
real ground truth, then closed `</caption><|im_end|>` cleanly — genuine image↔text
cross-modal binding, not template noise.

It can also produce full agent (3D pose) blocks that decode to valid, non-degenerate
coordinates, and — verified for the first time on this model's own generation, not just
training data — `cosmos` video tokens that decode to a real, playable video via
[`Cosmos-Tokenizer-DV8x16x16`](https://github.com/NVIDIA/Cosmos-Tokenizer), and `snac`
audio tokens that decode to a real, non-silent waveform via
[SNAC](https://github.com/hubertsiuzdak/snac) (`hubertsiuzdak/snac_24khz`).

All 3 non-text modalities the model actually produces in volume (`cosmos`, `snac`, `seed2`)
now have a working decoder in the project repo (`tools/decode/`) and have each been
round-tripped on real ground-truth tokens. `seed2` is generative rather than a
deterministic codec round-trip — see Known limitations.

## Known limitations

- **Greedy decoding can degenerate into repeated-token loops** inside long `cosmos`
  runs (e.g. the same token repeating 6-8 times), which can burn the generation budget
  before reaching `<fps_N>`/`agent`. Sampling with `repetition_penalty>1` mitigates this.
- **Sampling trades accuracy for diversity**: in the image-captioning test, sampled
  generation occasionally hallucinated details (e.g. an invented name) not present in
  the source image; greedy decoding did not.
- **`cosmos` tokens dominate generation**: aggregated across all test prompts, `cosmos`
  is 61-77% of all non-text VLA tokens produced (vs. a minority share for
  agent/seed2/snac combined). This is largely structural (one cosmos chunk costs a fixed
  200 tokens vs. ~1-4 tokens for the others), but it does mean cosmos runs can consume
  most of a generation's token budget before reaching `<fps_N>`/`agent`.
- **`avc_lm` tokens are essentially unused** — discarded at the data-flatten stage before
  training (to control token count), so the model rarely if ever produces them.
- **`seed2`→image reconstruction is generative, not a deterministic round-trip.**
  Seed2Tokenizer has no pixel decoder of its own; reconstruction conditions a diffusion
  img2img pipeline (`StableUnCLIPImg2ImgPipeline`) on the token embeddings to *generate*
  a plausible image, unlike `cosmos`/`snac`'s lossy-but-deterministic codec decoders — two
  runs of the same tokens can come out visually different. Verified end-to-end on 32 real
  ground-truth `<seed2_N>` tokens (`tools/decode/decode_seed2.py`) — the diffusion weights
  now come from a community mirror (`sd2-community/stable-diffusion-2-1-unclip`), since
  the original `stabilityai/stable-diffusion-2-1-unclip` was removed from HuggingFace.
- **Evaluation so far is qualitative** (manual inspection of generated tokens/decoded
  media) — no MPJPE, BLEU/CIDEr, or closed-loop task-success metric has been run yet.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "EmpathicRobotics/vla-1.7b-qwen3-v2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("EmpathicRobotics/vla-1.7b-qwen3-v2")

prompt = (
    "### Context: Person raises both arms above head.\\n"
    "<seed2_3758> <seed2_2157> <cosmos_58567> "
    "<fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>"
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    input_ids, max_new_tokens=500,
    do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.3,
)
print(tokenizer.decode(output[0]))
```

### Encoding real media into tokens (so you can actually prompt the model)

The `## Usage` prompt above uses pre-picked token ids as a demo. To send the
model *real* media -- e.g. "here's a photo, continue the scene" or "here's
a real motion clip, keep going" -- encode it first with the 4 encoders below
(**verified working 2026-07-23**, each tested end-to-end: real media ->
tokens -> decoded/compared back against the original). Bundled in this repo
the same way as the decoders (`tools/encode/`), no separate `git clone`
needed.

```bash
# Image -> <seed2_N> tokens (32 ids, auto-downloads the Q-Former checkpoint
# from ontocord/seed2 if not cached locally)
python tools/encode/encode_seed2.py --image photo.jpg

# 8 video frames -> <cosmos_N> tokens (200 ids -- this model's OLD
# window=8/square-crop convention, NOT the newer 2026-07-23 aspect-preserving
# one; auto-downloads encoder.jit from nvidia/Cosmos-Tokenizer-DV8x16x16)
python tools/encode/encode_cosmos.py --frames f0.png f1.png f2.png f3.png f4.png f5.png f6.png f7.png

# Audio/video file -> <snac_N> tokens (listen-format, <snac> wrapper --
# this model never saw the newer <listen>/<speak> convention or speak-format L2)
python tools/encode/encode_snac.py --input clip.wav

# Real 3D pose (8 frames x 17 joints x xyz, metres, root-centred) -> <agent>
# tokens -- for "give the model a real motion capture / pose-pipeline output,
# have it continue" (same behavior already verified: agent completion PASS)
python tools/encode/encode_agent.py --input pose.npy   # shape (8, 17, 3)
```

Splice the printed token block into your prompt (e.g. after `### Context:
...`) the same way the `## Usage` example does, then call `model.generate()`
as shown there.

### Decoding generated tokens back to media

The decoder scripts + their vendored dependencies are bundled directly in
**this repo** (`tools/`) -- one `snapshot_download` gets everything, no
separate `git clone` needed. (Also mirrored at
[github.com/TieuDaoChanNhan/finevideo-vla](https://github.com/TieuDaoChanNhan/finevideo-vla)
if you'd rather browse/clone the code on its own.) **Verified working
2026-07-23** with no cluster/internal access required, each tested end-to-end
on real tokens this model actually generated.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('EmpathicRobotics/vla-1.7b-qwen3-v2', allow_patterns=['tools/*', 'tools/**/*'])
"
pip install scipy numpy torch torchvision imageio-ffmpeg soundfile snac huggingface_hub
cd <snapshot-download-cache-dir-printed-above>
```

**Agent tokens -> 3D pose** (pure Python, no extra downloads):
```bash
python tools/eval/decode_agent_tokens.py --input generated_tokens.txt --output poses.json
```

**Cosmos tokens -> video** (auto-downloads the ~350MB decoder checkpoint from
[nvidia/Cosmos-Tokenizer-DV8x16x16](https://huggingface.co/nvidia/Cosmos-Tokenizer-DV8x16x16)
on first run):
```bash
python tools/decode/decode_cosmos.py --tokens 58345,57843,... --output out.mp4
# this model's cosmos chunks are exactly 200 raw ids each (8 frames, 160x160,
# square-cropped -- the DV8x16x16 checkpoint's own token grid for that input
# size). A later dataset pivot (2026-07-23, aspect-preserving/896 tokens)
# does NOT apply to this model -- it was trained entirely on the 200-token/
# square-crop convention.
```

**SNAC tokens -> audio** (auto-downloads `hubertsiuzdak/snac_24khz` from HF):
```bash
python tools/decode/decode_snac.py --tokens 130911,134940,... --format listen --output out.wav
# this model only ever saw "listen" format (3 tokens/base-frame, <snac>
# wrapper) -- do NOT use --format speak, that's a newer (2026-07-23)
# convention this model was never trained on.
```

**Seed2 tokens -> image** (auto-downloads the ~2.6GB Q-Former checkpoint from
the tokenizer's own public repo,
[ontocord/seed2](https://huggingface.co/ontocord/seed2), plus a ~5GB
diffusion img2img pipeline on first run -- this one is a generative
*reconstruction*, not a deterministic decode, so expect run-to-run and
prompt-to-prompt variation in the exact pixels even for the same tokens):
```bash
python tools/decode/decode_seed2.py --tokens 6750,680,2472,... --output out.png
# exactly 32 raw ids per image (Seed2Tokenizer's fixed Q-former query length)
```

## Training details

### Loss curve

| Iter | Loss |
|---|---|
| 50 | 6.472 |
| 500 | 2.840 |
| 1000 | 2.154 |
| 2000 | 1.953 |
| 4000 | 1.826 |
| 6000 | 1.767 |
| 7600 | 1.694 |
| 7632 (val) | 1.7526 (PPL 5.77) |
| 7632 (test) | 1.7722 (PPL 5.88) |

### Config

- **Batch**: GBS 1024, seq_len 4096 → 32.01B tokens trained (exactly 1 epoch)
- **Infrastructure**: 64 nodes × 4 GH200 GPUs (256 total), ~284 TFLOP/s/GPU, ~21,800 tok/s/GPU
- **Framework**: Megatron-LM via oellm-autoexp

### Data mix

| Source | Tokens |
|---|---|
| MixtureVitae-Omni | 20.39B |
| FineVideo-VLA v6 | 10.93B |
| OmniVideo-100K (video) | 0.54B |
| synth-llava | 0.10B |
| emotional-roleplay (SNAC TTS) | 0.05B |
| **Total** | **~32.01B** |

## Citation

```bibtex
@misc{empathicrobotics2026vlaqwen3,
  title={VLA 1.7B Qwen3 v2: Multi-Source Multimodal Vision-Language-Action Pretraining},
  author={EmpathicRobotics},
  year={2026},
  url={https://huggingface.co/EmpathicRobotics/vla-1.7b-qwen3-v2}
}
```
"""


TOKENIZER_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/tokenizer_vla_qwen3"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 3d-human-pose/ (file is at tools/upload/<this>.py)
DECODER_FILES = [
    "tools/decode/decode_cosmos.py",
    "tools/decode/decode_snac.py",
    "tools/decode/decode_seed2.py",
    "tools/eval/decode_agent_tokens.py",
    "tools/encode/encode_cosmos.py",
    "tools/encode/encode_snac.py",
    "tools/encode/encode_seed2.py",
    "tools/encode/encode_agent.py",
    "pipeline_pose/phase5_adaptive_pchip.py",  # encode_agent.py's build_token_str()
    # encode_snac.py imports encode_listen() from this file (relative
    # sys.path insert into ../../pipeline_pose) -- bundle it too so that
    # import resolves after a snapshot_download, not just in this git repo.
    "pipeline_pose/snac_finevideo.py",
]
VENDOR_DIR = "tools/decode/vendor"


def main():
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir(MODEL_DIR):
            src = os.path.join(MODEL_DIR, f)
            dst = os.path.join(tmp, f)
            print(f"  Copying {f} ({os.path.getsize(src) / 1e6:.1f} MB)")
            shutil.copy2(src, dst)

        # 2026-07-23: bundle the tokenizer directly into the model repo (not
        # just linked from a separate one) so `AutoTokenizer.from_pretrained(
        # "EmpathicRobotics/vla-1.7b-qwen3-v2")` works standalone -- no need
        # to know/find the separate tokenizer repo id.
        print("  Bundling tokenizer...")
        for f in os.listdir(TOKENIZER_DIR):
            shutil.copy2(os.path.join(TOKENIZER_DIR, f), os.path.join(tmp, f))

        # 2026-07-23: bundle the decoders + vendored cosmos_tokenizer so a
        # single `snapshot_download()`/`git clone` of this repo is enough to
        # run inference AND decode the output, without a second `git clone`
        # of the GitHub repo. Preserves the same relative layout
        # (tools/decode/, tools/eval/) so the decoders' own relative imports
        # (decode_cosmos.py's `vendor/` sys.path insert) still resolve.
        print("  Bundling decoders + vendored cosmos_tokenizer...")
        for rel_path in DECODER_FILES:
            src = os.path.join(REPO_ROOT, rel_path)
            dst = os.path.join(tmp, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        shutil.copytree(os.path.join(REPO_ROOT, VENDOR_DIR), os.path.join(tmp, VENDOR_DIR))

        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(README)

        print(f"\nUploading to {REPO_ID}...")
        api.upload_folder(
            folder_path=tmp,
            repo_id=REPO_ID,
            repo_type="model",
            create_pr=False,
            commit_message="Upload VLA 1.7B Qwen3 v2 with model card",
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
