#!/usr/bin/env python3
"""
Upload VLA 1.7B Qwen3 v7 (format-standardization, w8_new corpus) model to HuggingFace.

Usage:
    export HF_TOKEN=hf_...
    python tools/upload/upload_vla_v7_model.py
"""

import os
import tempfile
import shutil

from huggingface_hub import HfApi

REPO_ID = "EmpathicRobotics/vla-1.7b-qwen3-v7"
# 2026-07-31: v7 reformats the same 6-source w8_new corpus v6 trained on into a
# universal USER:/ASSISTANT:/<think> wire format (replacing v6's per-source ad
# hoc headers -- ### Title:/### Context: for FineVideo, Q:/A: for MV-Omni,
# etc.), per Huu's format-standardization directive (2026-07-30). Along the
# way, 2 real data-pipeline bugs were found and fixed: omnivideo's
# USER/ASSISTANT split point was wrong (~99% of every record's modality
# tokens landed on the USER side), and mv_omni's own Q:/A: -> USER:/ASSISTANT:
# reformat silently dropped/mis-split ~64% of records. Both fixes recovered
# real training signal (mv_omni: ~9.65M real turns kept vs ~1.7M under the
# old buggy script at matched record counts). Job 1137346, 64 nodes, 2544
# iters (1 epoch of 21.34B tokens).
MODEL_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/qwen3_1.7b_vla_v7_format_std/hf/iter_0002544"

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

# VLA 1.7B — Qwen3 v7 (format-standardization)

A 1.7B parameter Vision-Language-Action model, fourth Qwen3-backbone release from this
project. v7 keeps the exact same window=8, 6-source `w8_new` training corpus as
[vla-1.7b-qwen3-v6](https://huggingface.co/EmpathicRobotics/vla-1.7b-qwen3-v6) — it does
**not** add new data — and instead standardizes the prompt/wire format all 6 sources use:
a universal `USER: ... ASSISTANT:` instruction wrapper plus an empty `<think>\\n</think>`
immediately after `ASSISTANT:` on every single example, present even when there's nothing
to reason about. The `<think>` tag's job is structural, not reasoning: it always marks the
instruction/response boundary, so the model has one consistent pattern to condition on
across all 6 sources, and reuses the exact literal token sequences (`USER:`, ChatML-style
turn markers) that a much larger general-purpose instruction corpus also uses — the
intended transfer-learning benefit (see `project_format_standardization_directive` in the
project's internal notes for Huu's full rationale).

While standardizing the format, 2 real data-pipeline bugs were found and fixed:
**omnivideo_100k**'s `USER:`/`ASSISTANT:` split point was wrong (~99% of every record's
modality tokens landed on the `USER:` side instead of `ASSISTANT:`), and **mv_omni**'s own
`Q:`/`A:` → `USER:`/`ASSISTANT:` conversion silently dropped or mis-split ~64% of records
(a missing-preamble case and a `"\\nA:"`-without-space case). Both were caught and fixed
before this training run — mv_omni alone went from ~1.7M real turns kept to ~9.65M turns
kept at matched record counts once fixed.

## Key facts

| | |
|---|---|
| **Architecture** | Qwen3 (28 layers, hidden 2048, intermediate 6144, 16 attn heads / 8 KV heads (GQA), qk-layernorm, RoPE θ=1e6, tied embeddings) |
| **Parameters** | 1.97B (including embeddings for 274,688 padded vocab) |
| **Vocab size** | 274,688 padded (same tokenizer as v6 — no vocab expansion needed; `<think>`/`</think>`/`<\\|im_start\\|>`/`<\\|im_end\\|>`/`<\\|endoftext\\|>` were already atomic special tokens in the base Qwen3 tokenizer) |
| **Tokenizer** | [EmpathicRobotics/tokenizer-vla-qwen3-v2](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3-v2) |
| **Training data** | 21.34B tokens across the same 6 sources as v6, reformatted (see Data mix below) |
| **Training** | 2,544 iters (1 epoch), 64 nodes × 4 GH200 GPUs, global batch 1024, seq len 8192, micro batch size 2 |
| **Final loss** | Train: 2.609 (iter 2500), Val: 2.6153 (PPL 13.67), Test: 2.5738 (PPL 13.12) |
| **Precision** | bf16 |
| **Context length** | 8,192 tokens |

## What this model does

Given a text prompt (activity description, image seed2 block, or partial modality
sequence), the model generates an interleaved multimodal token sequence spanning
6 categories it was trained on:

```
<think> ... </think>                   # structural turn-boundary marker, usually empty
<seed2_N> ...                          # 1 FPS semantic image/video keyframes (vocab 8192)
<cosmos_N> ... </cosmos>               # 8-frame spatial video tokens (vocab 64000)
<listen> <snac_N> ... </listen>        # SNAC audio codec tokens, "heard" role
<speak> <snac_N> ... </speak>          # SNAC audio codec tokens, model-generated "spoken" role
<caption> ... </caption>               # inline visual caption text
<agent> <fps_30> <pelvis> ... </agent> # 3D human pose, 17 H36M joints
```

Prompt shape (FineVideo example, from `investigations/format_standardization/reformat_finevideo.py`):

```
USER: Continue this video activity titled "Morning stretch". Context: A person raises
both arms above their head. ASSISTANT:
<think>
</think>
<caption> ... </caption> <seed2_N> ... <agent> ... </agent> ...
```

## Progress vs. v6 — sampling matters more than decoding strategy might suggest

Direct comparison run (same eval harness, same 5-prompt suite structure, multi-seed where
noted) against v6 and v2:

- **Autonomous full-chain generation from a bare text header** (no modality primed —
  the hardest test in the suite: does the model spontaneously open `<think>`, then
  `<caption>`, `<seed2>`, and eventually a decodable `<agent>` block on its own?) — **v7
  succeeded with a valid decoded pose in 4/4 sampled seeds tested. v6 and v2 both failed
  in 0/4 combined runs** (greedy and sampled, seed 42) — decoding an all-zero pose every
  time. This is the clearest signal so far that the format-standardization changed
  something real, not just cosmetic, about the model's ability to self-initiate a full
  modality chain and close it with `<\\|im_end\\|>`.
- **Greedy decoding (no sampling) is not usable on this checkpoint** — without
  `repetition_penalty`, generation collapses into token-repetition loops (a single token
  repeated up to the full `max_new_tokens` budget, or a `<cosmos>` chunk repeating the same
  handful of ids for hundreds of tokens). This is not unique to v7 (v6 and v2 show a
  related-but-different failure under greedy — technically more varied tokens, but still no
  valid decodable output on the hardest test above) — but it means **this model must always
  be sampled** (`do_sample=True`, `temperature≈0.8`, `top_p≈0.9`, `repetition_penalty≈1.3`)
  for usable output. The bundled demo (Colab notebook in this repo) samples by default.
- **Perplexity is not directly comparable to v6** (13.67 val / 13.12 test vs v6's 6.08 /
  5.98) — the extra `<think>` and ChatML-style boundary tokens change the loss landscape
  and the corpus was re-split, so this is not an apples-to-apples regression signal by
  itself. Content quality (caption accuracy on genuinely novel prompts) is unchanged from
  v2 through v7 — still the project's top open problem, see Known limitations.

## Known limitations

- **Instruction-following / content-accuracy gap, unchanged since v2**: even when the
  model correctly opens `<caption>` and closes it cleanly, the actual described content is
  frequently wrong on genuinely novel prompts (e.g. describing unrelated objects/scenes).
  Format-standardization measurably improved *structural* autonomy (see above) but was
  never expected to fix this — it's a data-composition problem (most non-video sources are
  media→text, i.e. understanding/captioning direction, not text→media generation), flagged
  as a concrete target for the project's planned VLA-Instruct SFT stage.
- **Greedy decoding is unreliable** (see above) — always sample.
- **Probabilistic, not universal, on the "from-scratch" test**: even under sampling, 1 of 4
  seeds tested failed the hardest "generate an agent block from just a seed2 prime, no
  agent context at all" sub-test (decoded an all-zero pose) — treat the from-scratch
  full-chain result as "usually works, not guaranteed every draw," consistent with the
  "modality drift is probabilistic" pattern documented across the whole v2→v7 lineage.
- **`avc_lm` tokens are essentially unused** — discarded at the data-flatten stage before
  training, so the model rarely if ever produces them.
- **`seed2`→image reconstruction is generative, not a deterministic round-trip** (see
  `tools/decode/decode_seed2.py` — conditions a diffusion img2img pipeline on the token
  embeddings; expect run-to-run pixel variation for the same input tokens, and note the
  diffusion decode step itself (default 20 inference steps) is the slowest single decode
  step in the demo by design, not a bug).
- **Evaluation so far is qualitative** (manual inspection of generated tokens/decoded
  media, plus PPL, plus the structural comparison above) — no MPJPE, BLEU/CIDEr, or
  closed-loop task-success metric has been run yet.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "EmpathicRobotics/vla-1.7b-qwen3-v7",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("EmpathicRobotics/vla-1.7b-qwen3-v7")

prompt = (
    'USER: Continue this video activity titled "Morning stretch". Context: A person '
    "raises both arms above their head. ASSISTANT:\\n<think>\\n</think>\\n"
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    input_ids, max_new_tokens=1200,
    do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.3,
)
print(tokenizer.decode(output[0]))
```

**Always sample** (`do_sample=True` + `repetition_penalty>1.0`) — greedy decoding on this
checkpoint reliably collapses into repetition loops on generations longer than a couple
hundred tokens (see Progress vs. v6 above).

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
# "heard" convention)
python tools/encode/encode_snac.py --input clip.wav

# Real 3D pose (8 frames x 17 joints x xyz, metres, root-centred) -> <agent>
# tokens -- for "give the model a real motion capture / pose-pipeline output,
# have it continue"
python tools/encode/encode_agent.py --input pose.npy   # shape (8, 17, 3)
```

Splice the printed token block into your prompt (after `ASSISTANT:\\n<think>\\n</think>\\n`)
the same way the `## Usage` example does, then call `model.generate()` as shown there.

### Decoding generated tokens back to media

The decoder scripts + their vendored dependencies are bundled directly in
**this repo** (`tools/`) -- one `snapshot_download` gets everything, no
separate `git clone` needed.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('EmpathicRobotics/vla-1.7b-qwen3-v7', allow_patterns=['tools/*', 'tools/**/*'])
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
# --format speak for <speak>-wrapped tokens (model-generated "spoken" role).
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
| 50 | 8.214 |
| 500 | 3.998 |
| 1000 | 3.042 |
| 1500 | 2.830 |
| 2000 | 2.718 |
| 2500 | 2.609 |
| 2544 (val) | 2.6153 (PPL 13.67) |
| 2544 (test) | 2.5738 (PPL 13.12) |

### Config

- **Batch**: GBS 1024, seq_len 8192, micro_batch_size 2 → 21.34B tokens trained (1 epoch of the w8_new format-standardized corpus)
- **Infrastructure**: 64 nodes × 4 GH200 GPUs (256 total), ~302 TFLOP/s/GPU, ~20,650 tok/s/GPU
- **Framework**: Megatron-LM via oellm-autoexp
- **Runtime**: 1h27m wall clock (job 1137346)

### Data mix (same 6 sources as v6, reformatted -- 2 sources' real token counts changed after bug fixes)

| Source | Tokens | % of total | Notes |
|---|---|---|---|
| FineVideo-VLA (window=8 rebuild) | 10.93B | 51.2% | Unchanged from v6 — flagship video+pose branch |
| MixtureVitae-Omni (mv_omni) | 7.89B | 37.0% | +0.04B vs v6's pre-fix count — the reformat bug fixes moved WHERE text landed (USER vs ASSISTANT), not how much content existed |
| OmniVideo-100K | 1.98B | 9.3% | Unchanged token count from v6 — the reformat bug fix moved the split point, ~5 tokens differ across the whole corpus |
| Harmony4D | 0.32B | 1.5% | Multi-person close-interaction pose (20x oversampled from 416 real tracks), ChatML-tier format |
| synth-llava / synth-llava2 | 0.11B | 0.5% | Static image + caption pairs |
| emotional-roleplay (SNAC TTS, "speak") | 0.12B | 0.5% | Teaches the `<speak>` (model-generated audio) role |
| **Total** | **21.34B** | 100% | |

## Citation

```bibtex
@misc{{empathicrobotics2026vlaqwen3v7,
  title={{VLA 1.7B Qwen3 v7: Format-Standardized Prompt Wire Format Across 6 Multimodal Sources}},
  author={{EmpathicRobotics}},
  year={{2026}},
  url={{https://huggingface.co/EmpathicRobotics/vla-1.7b-qwen3-v7}}
}}
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
        # "EmpathicRobotics/vla-1.7b-qwen3-v7")` works standalone.
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
            commit_message="Upload VLA 1.7B Qwen3 v7 (format-standardization) with model card",
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
