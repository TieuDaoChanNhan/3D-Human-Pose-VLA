#!/usr/bin/env python3
"""
Upload VLA 1.7B Qwen3 v6 (seq_length=8192, w8_new rebuild) model to HuggingFace.

Usage:
    export HF_TOKEN=hf_...
    python tools/upload/upload_vla_v6_model.py
"""

import os
import tempfile
import shutil

from huggingface_hub import HfApi

REPO_ID = "EmpathicRobotics/vla-1.7b-qwen3-v6"
# 2026-07-26: v6 was launched as 3 parallel seq_length variants (4096/8192/16384,
# jobs 1046012/1046010/1046014) on the w8_new rebuild (window=8 architecture,
# same recipe as v2 + Harmony4D/MV-Omni/synth_llava/roleplay added). seq8192 was
# chosen for public release: best test PPL of the 3 (5.98 vs 6.36/6.12), all 5
# modalities present in full-chain generation, good cosmos-continuity (3
# consecutive chunks). See PROGRESS_VI.md 2026-07-26 entries / project_v6_seq_length_ablation.md.
MODEL_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/qwen3_1.7b_vla_v6_seq8192/hf/iter_0004033"

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

# VLA 1.7B — Qwen3 v6 (seq_length=8192)

A 1.7B parameter Vision-Language-Action model, third Qwen3-backbone release from this
project. v6 rebuilds the **window=8 architecture that produced
[vla-1.7b-qwen3-v2](https://huggingface.co/EmpathicRobotics/vla-1.7b-qwen3-v2)** (still the
project's strongest baseline) and adds newly-integrated data sources on top of it —
**Harmony4D** (multi-person close-interaction motion capture), **MV-Omni**
(MixtureVitae-Omni, audio+image+text), and the **`<listen>`/`<speak>` audio-token
convention** (replacing v2's bare `<snac>` wrapper). This checkpoint uses
`seq_length=8192`, the best-performing of three seq_length variants (4096/8192/16384)
trained in parallel as an ablation.

## Key facts

| | |
|---|---|
| **Architecture** | Qwen3 (28 layers, hidden 2048, intermediate 6144, 16 attn heads / 8 KV heads (GQA), qk-layernorm, RoPE θ=1e6, tied embeddings) |
| **Parameters** | 1.97B (including embeddings for 274,688 padded vocab) |
| **Vocab size** | 274,688 padded (274,561 real tokens: Qwen3 base ~151,669 + ~122,892 VLA tokens) |
| **Tokenizer** | [EmpathicRobotics/tokenizer-vla-qwen3-v2](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3-v2) |
| **Training data** | ~33.83B tokens across 6 sources: FineVideo-VLA (window=8 rebuild), OmniVideo-100K, Harmony4D, MixtureVitae-Omni, synth-llava, emotional-roleplay |
| **Training** | 4,033 iters (1 epoch), 64 nodes × 4 GH200 GPUs, global batch 1024, seq len 8192, micro batch size 2 |
| **Final loss** | Train: 1.7067, Val: 1.8056 (PPL 6.08), Test: 1.7886 (PPL 5.98) |
| **Precision** | bf16 |
| **Context length** | 8,192 tokens |

## What this model does

Given a text prompt (activity description, image seed2 block, or partial modality
sequence), the model generates an interleaved multimodal token sequence spanning
6 categories it was trained on:

```
<seed2_N> ...                          # 1 FPS semantic image/video keyframes (vocab 8192)
<cosmos_N> ... </cosmos>               # 8-frame spatial video tokens (vocab 64000)
<listen> <snac_N> ... </listen>        # SNAC audio codec tokens, "heard" role
<speak> <snac_N> ... </speak>          # SNAC audio codec tokens, model-generated "spoken" role
<caption> ... </caption>               # inline visual caption text
<agent> <fps_30> <pelvis> ... </agent> # 3D human pose, 17 H36M joints
```

## Progress vs. previous versions (v2 → v6)

The project trained v3/v4/v5 on a **window=24** architecture (longer temporal chunks,
different cosmos token cost, doc-packing, adjusted cosmos dropout). None of v3/v4/v5
beat v2 on any measured axis (perplexity, modality-persistence over time, or caption
accuracy) — see the project's full eval writeup. **v6 abandons window=24 and rebuilds
on v2's window=8 recipe instead, only adding new data sources on top.** Result: test
PPL recovers to 5.98–6.36 across the 3 seq_length variants, close to v2's 5.77 and far
better than v3 (27.58) / v4 (15.78) / v5 (16.75). This seq8192 checkpoint had the best
test PPL of the three and produced all 5 trained modalities in a single full-chain
generation test.

**Cosmos-continuity**: this checkpoint returned to `<cosmos>` and generated 3
consecutive real chunks in one sampled-decoding run — the best cosmos-persistence
observed across the whole v2→v6 lineage for a multi-chunk (not just single-return) case.

## Known limitations

- **Instruction-following gap, unchanged since v2**: given a genuinely novel activity
  description not seen in training (e.g. "A person is running in a park"), this model's
  self-generated caption is topically wrong (confabulates an unrelated activity), exactly
  like every prior version (v2 through v5). This is the project's highest-priority open
  problem — diagnosed as a missing instruction-tuning data stage (see the project's
  AnyGPT/AnyInstruct-inspired "VLA-Instruct" proposal), not something any pretraining-recipe
  change (window size, dropout, seq_length, mix ratio) has fixed so far.
- **Modality persistence, improved but only single-seed-verified for this specific
  checkpoint**: the sibling seq4096 variant achieved 4 consecutive non-frozen `<agent>`
  blocks in one run (best in project history), but only verified on 1 of 4 standard eval
  seeds — treat any single-seed persistence claim (including this checkpoint's
  cosmos-continuity result above) as encouraging, not proven-robust.
- **`avc_lm` tokens are essentially unused** — discarded at the data-flatten stage before
  training (to control token count), so the model rarely if ever produces them.
- **`seed2`→image reconstruction is generative, not a deterministic round-trip** (see
  `tools/decode/decode_seed2.py` — conditions a diffusion img2img pipeline on the token
  embeddings, unlike `cosmos`/`snac`'s lossy-but-deterministic codec decoders).
- **Evaluation so far is qualitative** (manual inspection of generated tokens/decoded
  media, plus PPL) — no MPJPE, BLEU/CIDEr, or closed-loop task-success metric has been
  run yet.
- **Two other seq_length variants exist internally** (seq4096, test PPL 6.36;
  seq16384, test PPL 6.12) but were not chosen for public release — seq8192 had the best
  overall balance of PPL and multi-modality coverage in the project's eval suite.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "EmpathicRobotics/vla-1.7b-qwen3-v6",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("EmpathicRobotics/vla-1.7b-qwen3-v6")

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
a real motion clip, keep going" -- encode it first with the 4 encoders below.
Bundled in this repo the same way as the decoders (`tools/encode/`), no
separate `git clone` needed.

```bash
# Image -> <seed2_N> tokens (32 ids, auto-downloads the Q-Former checkpoint
# from ontocord/seed2 if not cached locally)
python tools/encode/encode_seed2.py --image photo.jpg

# 8 video frames -> <cosmos_N> tokens (200 ids -- window=8/square-crop
# convention; the window=24/aspect-preserving convention (896 tokens) was NOT
# used for this model)
python tools/encode/encode_cosmos.py --frames f0.png f1.png f2.png f3.png f4.png f5.png f6.png f7.png

# Audio/video file -> <snac_N> tokens, wrapped in <listen> (this model's
# "heard" convention, replacing v2's bare <snac>)
python tools/encode/encode_snac.py --input clip.wav

# Real 3D pose (8 frames x 17 joints x xyz, metres, root-centred) -> <agent>
# tokens -- for "give the model a real motion capture / pose-pipeline output,
# have it continue"
python tools/encode/encode_agent.py --input pose.npy   # shape (8, 17, 3)
```

Splice the printed token block into your prompt (e.g. after `### Context:
...`) the same way the `## Usage` example does, then call `model.generate()`
as shown there.

### Decoding generated tokens back to media

The decoder scripts + their vendored dependencies are bundled directly in
**this repo** (`tools/`) -- one `snapshot_download` gets everything, no
separate `git clone` needed.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('EmpathicRobotics/vla-1.7b-qwen3-v6', allow_patterns=['tools/*', 'tools/**/*'])
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
# square-cropped) -- the window=24/aspect-preserving convention (896 tokens)
# does NOT apply to this model.
```

**SNAC tokens -> audio** (auto-downloads `hubertsiuzdak/snac_24khz` from HF):
```bash
python tools/decode/decode_snac.py --tokens 130911,134940,... --format listen --output out.wav
# use --format listen for <listen>-wrapped tokens (input/"heard" role) or
# --format speak for <speak>-wrapped tokens (model-generated "spoken" role) --
# this model was trained on both, unlike v2 which only ever saw bare <snac>.
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
| 50 | 6.634 |
| 500 | 2.801 |
| 1000 | 2.142 |
| 1500 | 1.980 |
| 2000 | 1.890 |
| 2500 | 1.828 |
| 3000 | 1.905 |
| 3500 | 1.801 |
| 4000 | 1.707 |
| 4033 (val) | 1.8056 (PPL 6.08) |
| 4033 (test) | 1.7886 (PPL 5.98) |

### Config

- **Batch**: GBS 1024, seq_len 8192, micro_batch_size 2 → 33.83B tokens trained (exactly 1 epoch of the w8_new corpus)
- **Infrastructure**: 64 nodes × 4 GH200 GPUs (256 total), ~303 TFLOP/s/GPU, ~20,650 tok/s/GPU
- **Framework**: Megatron-LM via oellm-autoexp

### Data mix

| Source | Tokens | Role |
|---|---|---|
| MixtureVitae-Omni | 20.39B | Audio+image+text, largest source, "language backbone" |
| FineVideo-VLA (window=8 rebuild) | 10.93B | Flagship video+pose branch |
| OmniVideo-100K | 1.98B | Video + multimodal QA |
| Harmony4D | 0.32B | Multi-person close-interaction pose (20x oversampled from 416 real tracks) |
| emotional-roleplay (SNAC TTS, "speak") | 0.05B | Teaches the `<speak>` (model-generated audio) role |
| synth-llava / synth-llava2 | 0.02B (seed2-only count) | Static image + caption pairs |
| **Total** | **~33.83B** | |

## Citation

```bibtex
@misc{empathicrobotics2026vlaqwen3v6,
  title={VLA 1.7B Qwen3 v6: Window=8 Architecture Rebuild with Expanded Multimodal Sources},
  author={EmpathicRobotics},
  year={2026},
  url={https://huggingface.co/EmpathicRobotics/vla-1.7b-qwen3-v6}
}
```
"""


TOKENIZER_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/tokenizer_vla_qwen3_v2"
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

        # Bundle the full tokenizer directly into the model repo (not just
        # linked from a separate one) so `AutoTokenizer.from_pretrained(
        # "EmpathicRobotics/vla-1.7b-qwen3-v6")` works standalone.
        print("  Bundling tokenizer...")
        for f in os.listdir(TOKENIZER_DIR):
            shutil.copy2(os.path.join(TOKENIZER_DIR, f), os.path.join(tmp, f))

        # Bundle the decoders + vendored cosmos_tokenizer so a single
        # `snapshot_download()`/`git clone` of this repo is enough to run
        # inference AND decode the output, without a second `git clone` of
        # the GitHub repo. Preserves the same relative layout (tools/decode/,
        # tools/eval/) so the decoders' own relative imports still resolve.
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
            commit_message="Upload VLA 1.7B Qwen3 v6 (seq8192) with model card",
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
