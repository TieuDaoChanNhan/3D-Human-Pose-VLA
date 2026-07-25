#!/usr/bin/env python3
"""
Any-mode-to-any-mode test against REAL pretrain records (not synthetic prompts),
per Huu's direct request in the 2026-07-22 Discord thread (REPORT.md #33, item 9):
"try doing stuff from the pretrain dataset itself... like an image of a chemical
bond, or something from one of the valid speech thing, to see if it will echo
back some speaking, or try feeding it cosmo or seed tokens of a woman walking and
see what it does, it should do any mode to any mode."

3 tests, each pulled live from its real tokenized source file (not hand-copied,
so there's no transcription risk and the exact record id is traceable):

  1. chemical_bond_real   -- synth_llava2_001131949, a real chemical-bond diagram.
                             Prompt = seed2 block only. Does the model caption it
                             correctly (image -> text)?
  2. woman_walking_real   -- FineVideo-VLA v6 record captioned "The woman is
                             walking outside." Prompt = header + caption + seed2
                             + first 200-token cosmos chunk (1 chunk, open tag
                             only). Does the model continue coherently (real
                             ground truth is close-cosmos + snac + speech), and
                             does decoding the model's own next cosmos chunk show
                             any sign of unrelated-scene bleed (Huu's "finger from
                             a cutting video" suspicion)?
  3. roleplay_speech_real -- a real laion/emotional-roleplay USER turn (voice-
                             acted instruction), fed with NO snac hint at all.
                             Does the model produce a plausible <snac> speech
                             response from scratch (text -> audio)?

Ground truth is written into every input_output.txt for direct comparison, and
every modality present in the full prompt+output text is decoded to real media
via the project's existing decoders (reusing decode_media from eval_vla_v2_media).

Usage:
    python tools/eval/eval_any_mode_real.py
    python tools/eval/eval_any_mode_real.py --only woman_walking --modes greedy
"""
import argparse
import json
import os
import re
import sys
from datetime import date

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from eval_vla_v2_sanity import MODEL_PATH, TOKENIZER_PATH
from eval_vla_v2_media import _patched_tokenizer_path, decode_media

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

SYNTH_LLAVA2_SHARD = "/p/data1/mmlaion/shared/vla/synth_llava_flat/synth_llava2_shard-0000025.jsonl"
FINEVIDEO_V6_SHARD = ("/e/data1/datasets/playground/mmlaion/shared/nguyen38/window8_legacy/"
                      "FineVideo-VLA/megatron_dataset_v6/flat_final_vla_adaptive_rank_1.jsonl")
ROLEPLAY_SHARD = "/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/flattened/roleplay_snac_flat_00000.jsonl"

COSMOS_CHUNK_TOKENS = 200


def load_chemical_bond_real():
    record_id = "synth_llava2_001131949"
    with open(SYNTH_LLAVA2_SHARD) as f:
        for line in f:
            d = json.loads(line)
            if d.get("id") == record_id:
                text = d["text"]
                cut = text.index("</seed2>") + len("</seed2>")
                return text[:cut], text[cut:].strip(), record_id
    raise RuntimeError(f"{record_id} not found in {SYNTH_LLAVA2_SHARD}")


def load_woman_walking_real():
    needle = "The woman is walking outside."
    with open(FINEVIDEO_V6_SHARD) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("text", "")
            if needle in t:
                idx = t.find(needle)
                hdr_start = t.rfind("### Title:", 0, idx)
                seg = t[hdr_start:]
                break
        else:
            raise RuntimeError(f"{needle!r} not found in {FINEVIDEO_V6_SHARD}")

    m_caption = re.search(r"### Title:.*?</caption>", seg, re.DOTALL)
    header_caption = m_caption.group(0)
    rest = seg[m_caption.end():]

    m_seed2 = re.search(r"<seed2>.*?</seed2>", rest, re.DOTALL)
    seed2_block = m_seed2.group(0)
    after_seed2 = rest[m_seed2.end():]

    m_cosmos = re.search(r"<cosmos>(.*?)</cosmos>", after_seed2, re.DOTALL)
    cosmos_ids = re.findall(r"<cosmos_\d+>", m_cosmos.group(1))
    if len(cosmos_ids) < COSMOS_CHUNK_TOKENS:
        raise RuntimeError(f"only {len(cosmos_ids)} cosmos ids in first chunk, need {COSMOS_CHUNK_TOKENS}")
    first_chunk = cosmos_ids[:COSMOS_CHUNK_TOKENS]

    prompt = header_caption + " " + seed2_block + " <cosmos> " + " ".join(first_chunk)

    m_gt = re.search(r"</cosmos>\s*<snac>.*?</snac>\s*<speech>.*?</speech>",
                      after_seed2[m_cosmos.start():], re.DOTALL)
    ground_truth = m_gt.group(0) if m_gt else "(closing tag / next spans not found)"
    return prompt, ground_truth, "FineVideo-VLA v6, header 'Pickings Charge'"


def load_roleplay_speech_real():
    with open(ROLEPLAY_SHARD) as f:
        d = json.loads(f.readline())
    text = d["text"]
    marker = "ASSISTANT:"
    idx = text.index(marker) + len(marker)
    prompt = text[:idx].strip()
    ground_truth = text[idx:].strip()
    return prompt, ground_truth, d["id"]


def build_prompts():
    cb_prompt, cb_gt, cb_id = load_chemical_bond_real()
    ww_prompt, ww_gt, ww_id = load_woman_walking_real()
    rp_prompt, rp_gt, rp_id = load_roleplay_speech_real()
    return [
        {
            "name": "chemical_bond_real", "category": "image_to_text",
            "source": f"{cb_id} ({SYNTH_LLAVA2_SHARD})",
            "prompt": cb_prompt, "ground_truth": cb_gt, "max_new_tokens": 300,
            "description": "Real chemical-bond diagram (seed2 only) -- Huu: 'try an image of a "
                            "chemical bond ... from the pretrain dataset itself'. Expect a caption "
                            "naming carbon/bonds/structure, matching the real ground truth.",
        },
        {
            "name": "woman_walking_real", "category": "any_to_any",
            "source": f"{ww_id}, {ww_id.split(',')[0]}",
            "prompt": ww_prompt, "ground_truth": ww_gt, "max_new_tokens": 400,
            "description": "Real header+caption+seed2+1 cosmos chunk of a woman walking outside -- "
                            "Huu: 'try feeding it cosmo or seed tokens of a woman walking and see "
                            "what it does, it should do any mode to any mode'. Also checks whether "
                            "the model's own next cosmos chunk shows unrelated-scene content "
                            "(Huu's 'finger from a cutting video' cross-contamination suspicion).",
        },
        {
            "name": "roleplay_speech_real", "category": "text_to_audio",
            "source": f"{rp_id} ({ROLEPLAY_SHARD})",
            "prompt": rp_prompt, "ground_truth": rp_gt, "max_new_tokens": 450,
            "description": "Real emotional-roleplay USER turn, NO snac hint given at all -- Huu: "
                            "'something from one of the valid speech thing, to see if it will echo "
                            "back some speaking'. Expect a plausible <snac> response from scratch.",
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--only", default=None, help="Comma-separated substrings; only run matching prompt names")
    ap.add_argument("--modes", default="greedy,sample", help="Comma-separated subset of: greedy,sample")
    args = ap.parse_args()

    prompts = build_prompts()
    if args.only:
        needles = args.only.split(",")
        prompts = [p for p in prompts if any(n in p["name"] for n in needles)]
        if not prompts:
            raise SystemExit(f"--only matched nothing (have: {[p['name'] for p in build_prompts()]})")
    modes = [m for m in (("greedy", False), ("sample", True)) if m[0] in args.modes.split(",")]

    run_dir = args.run_dir or os.path.join(
        REPO_ROOT, "samples", "qwen3_1.7b_vla_v2_eval", f"{date.today().isoformat()}_any_mode_real")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Model:     {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(_patched_tokenizer_path(TOKENIZER_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    summary_rows = []

    for idx, pinfo in enumerate(prompts, start=1):
        for mode, sample in modes:
            test_name = f"{idx:02d}_{pinfo['name']}_{mode}"
            test_dir = os.path.join(run_dir, test_name)
            os.makedirs(test_dir, exist_ok=True)
            print(f"\n{'=' * 60}\n{test_name}  [{pinfo['category']}]\n{'=' * 60}")
            print(f"  source: {pinfo['source']}")
            print(f"  {pinfo['description']}")

            input_ids = tokenizer.encode(pinfo["prompt"], return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            prompt_len = input_ids.shape[1]
            print(f"  prompt_len tokens: {prompt_len}")

            gen_kwargs = dict(max_new_tokens=pinfo["max_new_tokens"], pad_token_id=tokenizer.eos_token_id)
            if sample:
                torch.manual_seed(42)
                gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.3)
            else:
                gen_kwargs.update(do_sample=False)

            with torch.no_grad():
                output_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

            generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
            full_text = pinfo["prompt"] + generated_text

            text_path = os.path.join(test_dir, "input_output.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(f"CATEGORY: {pinfo['category']}\n")
                f.write(f"SOURCE: {pinfo['source']}\n")
                f.write(f"DESCRIPTION: {pinfo['description']}\n")
                f.write(f"MODE: {mode}\n\n")
                f.write("INPUT PROMPT (fed to the model, real pretrain tokens):\n")
                f.write(pinfo["prompt"] + "\n\n")
                f.write("GROUND TRUTH (what the real record actually has next):\n")
                f.write(pinfo["ground_truth"] + "\n\n")
                f.write("MODEL OUTPUT (everything generated after the prompt):\n")
                f.write(generated_text + "\n")

            print(f"  Saved: {text_path}")

            media_status = decode_media(text_path, full_text, test_dir)
            for modality, (ok, detail) in media_status.items():
                print(f"  decode[{modality}]: {'OK' if ok else 'FAIL'} - {detail}")

            summary_rows.append({
                "test": test_name, "category": pinfo["category"], "mode": mode,
                "media": media_status,
            })

    summary_path = os.path.join(run_dir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# qwen3_1.7b_vla_v2 any-mode-to-any-mode eval (REAL pretrain records) -- {date.today().isoformat()}\n\n")
        f.write(f"Model: `{MODEL_PATH}`\n\n")
        f.write("Requested directly by Huu in the 2026-07-22 Discord thread (REPORT.md #33, item 9).\n\n")
        f.write("| Test | Category | Mode | seed2 | cosmos | snac | agent |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in summary_rows:
            def cell(m):
                if m not in row["media"]:
                    return "-"
                ok, detail = row["media"][m]
                return f"{'PASS' if ok else 'FAIL'}: {detail}"
            f.write(f"| [{row['test']}]({row['test']}/input_output.txt) | {row['category']} | {row['mode']} | "
                     f"{cell('seed2')} | {cell('cosmos')} | {cell('snac')} | {cell('agent')} |\n")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
