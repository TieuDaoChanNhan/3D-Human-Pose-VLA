#!/usr/bin/env python3
"""
VLA v7 (Qwen3, format-standardization -- USER:/ASSISTANT:/<think> universal
wrapper, window=8, 6-source mix) model sanity check -- token atomicity +
greedy/sampled generation.

Differences from eval_vla_v6_sanity.py:
  - v7's format-standardization pass wraps every source in USER: ... ASSISTANT:
    followed by a universal <think>\\n</think> (see project_format_standardization_directive
    memory) -- prompts here use that wrapper, v6's prompts predate it (raw
    ### Context:/### Title:/### Keywords: headers).
  - real record pulled from the reformatted flat text (post-fix,
    w8_new/flatten/finevideo_v6_listen/finevideo_v6_flat_listen_userassistant/
    flat_final_vla_adaptive_rank_0.jsonl, line 408, "Timid Feeding Gone Wrong"
    video) -- includes a real <cosmos> block between seed2 and listen/agent,
    unlike v6's chosen record which had none.

Run on login node with one GPU:
    module --force purge
    module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
    source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
    python tools/eval/eval_vla_v7_sanity.py --model-path <hf ckpt>
"""

import argparse
import re
import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from decode_agent_tokens import decode, to_json, JOINT_NAMES, JOINT_INDEX

TOKENIZER_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/tokenizer_vla_qwen3_v2"

REAL_RECORD_PATH = (
    "/e/data1/datasets/playground/mmlaion/shared/nguyen38/w8_new/flatten/"
    "finevideo_v6_listen/finevideo_v6_flat_listen_userassistant/"
    "flat_final_vla_adaptive_rank_0.jsonl"
)
REAL_RECORD_LINE = 408


# ── Test 1: Token atomicity ──────────────────────────────────────────────────

ATOMICITY_TESTS = [
    "<seed2>", "</seed2>", "<seed2_1137>", "<seed2_0>", "<seed2_8191>",
    "<cosmos>", "</cosmos>", "<cosmos_58567>", "<cosmos_0>", "<cosmos_63999>",
    "<fps_30>",
    "<agent>", "</agent>",
    "<pelvis_t_0>", "<pelvis_t_7>",
    "<pelvis_x_0>", "<pelvis_x_128>", "<pelvis_x_255>",
    "<r_wrist_y_200>", "<l_shoulder_z_50>",
    "<r_hip>", "</r_hip>", "<head_top>", "</head_top>",
    "<listen>", "</listen>", "<speak>", "</speak>", "<speech>", "</speech>",
    "<snac_128266>", "<snac_132361>", "<snac_132362>", "<snac_136457>",
    "<snac_136458>", "<snac_144649>", "<snac_144650>", "<snac_148745>",
    "<snac_148746>", "<snac_156937>",
    "<caption>", "</caption>",
    # v7-specific: universal ChatML/think wrapper tokens.
    "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
]


def test_atomicity(tokenizer):
    print("=" * 60)
    print("TEST 1: Token atomicity")
    print("=" * 60)

    passed = 0
    failed = 0
    for token_str in ATOMICITY_TESTS:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        ok = len(ids) == 1
        status = "OK" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            decoded = [tokenizer.decode([i]) for i in ids]
            print(f"  {status}  {token_str:30s} -> {len(ids)} ids: {decoded}")

    if failed == 0:
        print(f"  All {passed} tokens are atomic (single token ID each)")
    else:
        print(f"\n  {passed} passed, {failed} FAILED")

    print()
    return failed == 0


# ── Test 2: Greedy/sampled generation ────────────────────────────────────────

def load_real_record():
    import json
    with open(REAL_RECORD_PATH) as f:
        for i, line in enumerate(f):
            if i == REAL_RECORD_LINE:
                return json.loads(line)["text"]
    raise RuntimeError(f"line {REAL_RECORD_LINE} not found in {REAL_RECORD_PATH}")


def parse_real_record(text):
    hdr_end = text.find("ASSISTANT:") + len("ASSISTANT:")
    header_nochat = text[:hdr_end]

    think_start = text.find("<think>", hdr_end)
    think_end = text.find("</think>", think_start) + len("</think>")
    header_with_think = text[:think_end] + "\n"

    cap_start = text.find("<caption>", think_end)
    cap_end = text.find("</caption>", cap_start) + len("</caption>")
    caption_block = text[cap_start:cap_end] + " "

    seed2_start = text.find("<seed2>", cap_end)
    seed2_end = text.find("</seed2>", seed2_start) + len("</seed2>")
    seed2_block = text[seed2_start:seed2_end] + " "

    agent_start = text.find("<agent>")
    agent_end = text.find("</agent>", agent_start) + len("</agent>")
    agent_block = text[agent_start:agent_end]

    r_knee_close = agent_block.find("</r_knee>") + len("</r_knee>")
    agent_partial = agent_block[:r_knee_close]

    after_agent = text[agent_end:]
    listen_start = after_agent.find("<listen>")
    listen_after_agent = after_agent[listen_start:]
    # first 3 of 9 snac tokens in the real trailing listen block
    snac_ids = re.findall(r"<snac_\d+>", listen_after_agent)
    listen_partial = "<listen> " + " ".join(snac_ids[:3])

    return {
        "header_nochat": header_nochat,
        "header_with_think": header_with_think,
        "caption": caption_block,
        "seed2": seed2_block,
        "agent_partial": agent_partial,
        "agent_full": agent_block,
        "listen_partial": listen_partial,
    }


def build_prompts():
    text = load_real_record()
    r = parse_real_record(text)

    return [
        {
            "name": "full_prompt_header",
            "prompt": r["header_nochat"],
            "max_new_tokens": 2000,
            "description": "REAL record header ending at 'ASSISTANT:' (no <think> "
                            "yet) -- expect model to emit <think></think><caption>"
                            "...<seed2>...<agent>...<listen> chain matching train-"
                            "time wiring",
        },
        {
            "name": "agent_continuation",
            "prompt": r["header_with_think"] + r["caption"] + r["seed2"] + r["agent_partial"],
            "max_new_tokens": 1200,
            "description": "REAL partial <agent> block (pelvis/r_hip/r_knee only, "
                            "w8, adaptive CP counts), expect model completes "
                            "remaining 14 joints with valid t_0-7 range and closes "
                            "</agent>",
            "ground_truth_next_token": "<r_ankle>",
        },
        {
            "name": "seed2_to_agent_from_scratch",
            "prompt": r["header_with_think"] + r["caption"] + r["seed2"],
            "max_new_tokens": 1500,
            "description": "REAL header+caption+full seed2 block with NOTHING "
                            "after -- expect model opens <agent> (or <cosmos>, "
                            "also valid per real wiring) on its own",
        },
        {
            "name": "listen_continuation",
            "prompt": r["header_with_think"] + r["caption"] + r["seed2"] + r["agent_full"] + " " + r["listen_partial"],
            "max_new_tokens": 400,
            "description": "Partial <listen> block (3/9 real snac tokens from "
                            "ground truth) -- expect model continues with valid "
                            "<snac_N> tokens and eventually closes </listen>",
        },
        {
            "name": "image_caption_synth_llava",
            # Unaffected by header format (image-only, no agent/cosmos) -- reused
            # verbatim from eval_vla_v3/v6_sanity.py, real synth_llava2 record.
            "prompt": (
                "<seed2> <seed2_6750> <seed2_2157> <seed2_8000> <seed2_6026> <seed2_680> "
                "<seed2_4247> <seed2_3771> <seed2_5189> <seed2_3771> <seed2_2157> <seed2_2157> "
                "<seed2_2745> <seed2_1131> <seed2_6115> <seed2_1227> <seed2_5619> <seed2_2157> "
                "<seed2_4247> <seed2_2157> <seed2_884> <seed2_3771> <seed2_3771> <seed2_3771> "
                "<seed2_2157> <seed2_6115> <seed2_884> <seed2_716> <seed2_1088> <seed2_4174> "
                "<seed2_3395> <seed2_1227> <seed2_1088> </seed2>"
            ),
            "max_new_tokens": 300,
            "description": "REAL synth_llava2 record (id=synth_llava2_003266024), "
                            "seed2 block only, expect model to emit <caption>...</caption>",
            "ground_truth": (
                "<caption> The image is a portrait of a young boy wearing a green graduation "
                "cap and gown. He is smiling and looking directly at the camera. The cap is "
                "green with a yellow tassel hanging from the top. The gown is a dark green "
                "color with a white collar and a green tie. The background is a solid green "
                "color. The image is framed in a green ornate frame. </caption>"
            ),
        },
    ]


def classify_token(token_str):
    if re.match(r"<seed2_?\d*>", token_str):
        return "seed2"
    if re.match(r"<cosmos_?\d*>", token_str):
        return "cosmos"
    if re.match(r"<snac_?\d*>", token_str) or token_str in (
        "<listen>", "</listen>", "<speak>", "</speak>", "<speech>", "</speech>",
        "<snac>", "</snac>",
    ):
        return "snac"
    if token_str in ("<caption>", "</caption>"):
        return "caption"
    if token_str in ("<think>", "</think>"):
        return "think"
    if token_str in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        return "chatml"
    if re.match(r"<fps_\d+>", token_str):
        return "agent"
    if re.match(r"<\w+_[txyz]_\d+>", token_str):
        return "agent"
    if re.match(r"</?[a-z_]+>", token_str):
        return "agent"
    return "text"


def validate_agent_structure(tokens):
    """Check if generated agent tokens have valid structure (window=8: t 0-7)."""
    errors = []
    warnings = []

    vla_tokens = [t for t in tokens if classify_token(t) == "agent"]
    if not vla_tokens:
        errors.append("No agent tokens generated")
        return errors, warnings

    fps_count = sum(1 for t in vla_tokens if t.startswith("<fps_"))
    if fps_count == 0:
        errors.append("No <fps_N> token found")

    found_joints = set()
    for t in vla_tokens:
        m = re.match(r"^<([a-z_]+)>$", t)
        if m and m.group(1) in JOINT_INDEX:
            found_joints.add(m.group(1))

    if found_joints:
        missing = set(JOINT_NAMES) - found_joints
        if missing:
            warnings.append(f"Missing joints: {sorted(missing)}")
    else:
        warnings.append("No joint open tags found")

    for t in vla_tokens:
        for dim in ("x", "y", "z"):
            m = re.match(rf"<\w+_{dim}_(\d+)>", t)
            if m and int(m.group(1)) > 255:
                errors.append(f"Out-of-range value in {t}")
        m = re.match(r"<\w+_t_(\d+)>", t)
        if m and int(m.group(1)) > 7:
            errors.append(f"Frame index > 7 in {t} (window=8 standard for v7)")

    return errors, warnings


def test_generation(model, tokenizer, device, prompts, max_new_tokens=500, sample=False,
                     temperature=0.8, top_p=0.9, repetition_penalty=1.3, seed=42):
    mode = "Sampled" if sample else "Greedy"
    print("=" * 60)
    print(f"TEST 2: {mode} generation" + (
        f" (T={temperature}, top_p={top_p}, rep_penalty={repetition_penalty})" if sample else ""))
    print("=" * 60)

    if sample:
        torch.manual_seed(seed)

    all_ok = True

    for pinfo in prompts:
        print(f"\n{'#' * 60}")
        print(f"--- {pinfo['name']} ---")
        print(f"  {pinfo['description']}")
        print(f"\n  >>> FULL INPUT PROMPT:")
        print(f"  {pinfo['prompt']}")
        if "ground_truth" in pinfo:
            print(f"\n  >>> GROUND TRUTH (what training data actually has next):")
            print(f"  {pinfo['ground_truth']}")
        if "ground_truth_next_token" in pinfo:
            print(f"\n  >>> GROUND TRUTH next open tag: {pinfo['ground_truth_next_token']}")

        input_ids = tokenizer.encode(pinfo["prompt"], return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        gen_len = pinfo.get("max_new_tokens", max_new_tokens)

        gen_kwargs = dict(
            max_new_tokens=gen_len,
            pad_token_id=tokenizer.eos_token_id,
        )
        if sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        generated_ids = output_ids[0, prompt_len:]
        generated_tokens = [tokenizer.decode([tid]).strip() for tid in generated_ids.tolist()]

        counts = {}
        for t in generated_tokens:
            cat = classify_token(t)
            counts[cat] = counts.get(cat, 0) + 1

        print(f"\n  Generated {len(generated_tokens)} tokens")
        print(f"  Breakdown: {counts}")

        full_output = " ".join(generated_tokens)
        print(f"\n  >>> FULL MODEL OUTPUT:")
        print(f"  {full_output}")

        transitions = []
        prev_cat = None
        for i, t in enumerate(generated_tokens):
            cat = classify_token(t)
            if cat != prev_cat:
                transitions.append((i, cat, t))
                prev_cat = cat
        if len(transitions) > 1:
            print(f"  Transitions: ", end="")
            print(" -> ".join(f"{cat}@{i}" for i, cat, _ in transitions[:15]))

        if any(classify_token(t) == "agent" for t in generated_tokens):
            errors, warnings = validate_agent_structure(generated_tokens)
            for e in errors:
                print(f"  ERROR: {e}")
                all_ok = False
            for w in warnings:
                print(f"  WARN:  {w}")

            agent_str = " ".join(t for t in generated_tokens if classify_token(t) == "agent")
            if agent_str:
                try:
                    trajectories = decode(agent_str)
                    if trajectories:
                        result = to_json(trajectories)
                        print(f"  Decoded {result['n_windows']} pose windows, "
                              f"shape {result['shape']}, "
                              f"range {result['value_range_m']} m")
                        for w in result["windows"][:2]:
                            movers = ", ".join(f"{n} {d:.3f}m" for n, d in w["top_movers"][:3])
                            print(f"    Window {w['window']}: {movers}")
                    else:
                        print("  Could not decode agent tokens into poses")
                except Exception as e:
                    print(f"  Decode error: {e}")

    print()
    return all_ok


def main():
    p = argparse.ArgumentParser(description="VLA v7 model sanity check (format-standardization, window=8)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decoding")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Model:     {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    atomicity_ok = test_atomicity(tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n_params / 1e9:.2f}B params, vocab {model.config.vocab_size}")
    print()

    prompts = build_prompts()

    generation_ok = test_generation(
        model, tokenizer, device, prompts, args.max_new_tokens,
        sample=args.sample, temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
    )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Token atomicity: {'PASS' if atomicity_ok else 'FAIL'}")
    print(f"  Generation:      {'PASS' if generation_ok else 'NEEDS REVIEW'}")
    print()


if __name__ == "__main__":
    main()
