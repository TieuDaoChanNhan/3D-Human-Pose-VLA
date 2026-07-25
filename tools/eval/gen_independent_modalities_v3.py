#!/usr/bin/env python3
"""
Independent single-modality generation test for vla-1.7b-qwen3-v3: for each
modality, prime the prompt with ONLY free-text context + the opening tag
(no other modality's tokens feeding in first, unlike gen_full_chain_v3.py's
free-running chain), and check whether the model can produce that modality
on its own from a text description alone. Decodes each into a real file.

Usage:
    python tools/eval/gen_independent_modalities_v3.py --output-dir /path/to/outdir
"""
import argparse
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decode"))
from decode_agent_tokens import decode as decode_agent, to_json, JOINT_NAMES

MODEL_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/qwen3_1.7b_vla_v3/hf/iter_0000881"
TOKENIZER_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/tokenizer_vla_qwen3_v2"

# Each test: free-text context (never seen in training) + the opening tag only
# -- nothing else primes the model toward that modality.
TESTS = [
    {
        "name": "image_independent",
        "modality": "seed2",
        "prompt": "### Context: A golden retriever puppy sits on a green lawn under a blue sky.\n<seed2>",
        "max_new_tokens": 60,
    },
    {
        "name": "video_independent",
        "modality": "cosmos",
        "prompt": "### Context: A red sports car speeds down a highway at sunset.\n<cosmos>",
        "max_new_tokens": 950,
    },
    {
        "name": "pose_independent",
        "modality": "agent",
        "prompt": "### Context: A gymnast performs a cartwheel on a mat.\n<agent> <fps_30>",
        "max_new_tokens": 400,
    },
    {
        "name": "audio_independent",
        "modality": "listen",
        "prompt": "### Context: A woman warmly says thank you so much for your help today.\n<listen>",
        "max_new_tokens": 60,
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    summary = {}

    for test in TESTS:
        name, modality, prompt = test["name"], test["modality"], test["prompt"]
        print(f"\n{'#'*60}\n--- {name} ---\nPrompt: {prompt}")

        torch.manual_seed(args.seed)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=test["max_new_tokens"], do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False)

        with open(os.path.join(args.output_dir, f"{name}_raw.txt"), "w") as f:
            f.write(prompt + gen_text)

        # Only keep the leading run of tokens matching this modality's prefix
        # (stop at the first token that belongs to something else -- tests
        # whether the model can produce a clean, self-contained block from a
        # bare text prompt, not whether it eventually wanders elsewhere).
        ids = [int(x) for x in re.findall(rf"<{modality}_(\d+)>", gen_text)]
        leading = []
        for tid in re.findall(r"<[^>]+>", gen_text):
            m = re.match(rf"<{modality}_(\d+)>", tid)
            if m:
                leading.append(int(m.group(1)))
            elif tid in (f"</{modality}>",):
                break
            else:
                break
        print(f"  {len(ids)} total {modality} ids found in generation, "
              f"{len(leading)} in the leading contiguous run")

        ok = False
        try:
            if modality == "seed2" and len(leading) >= 8:
                from decode_seed2 import decode_seed2_tokens
                out_path = os.path.join(args.output_dir, f"{name}.png")
                decode_seed2_tokens(leading[:32], out_path)
                print(f"  Decoded -> {out_path}")
                ok = True
            elif modality == "cosmos":
                print(f"  {len(leading)} cosmos tokens (need exactly 200 for the "
                      f"OLD decoder format; v3 uses ~896/chunk -- same known gap "
                      f"as gen_full_chain_v3.py, decode_cosmos.py not updated yet)")
                if len(leading) == 200:
                    from decode_cosmos import decode_cosmos_chunk
                    out_path = os.path.join(args.output_dir, f"{name}.mp4")
                    decode_cosmos_chunk(leading, out_path)
                    print(f"  Decoded -> {out_path}")
                    ok = True
            elif modality == "agent":
                # re-prepend <agent> <fps_30> since the prompt itself opened them
                block = "<agent> <fps_30> " + gen_text
                trajectories = decode_agent(block)
                if trajectories:
                    result = to_json(trajectories)
                    out_json = os.path.join(args.output_dir, f"{name}.json")
                    import json
                    with open(out_json, "w") as f:
                        json.dump(result, f, indent=1)
                    print(f"  Decoded {result['n_windows']} window(s), shape {result['shape']}, "
                          f"range {result['value_range_m']} m -> {out_json}")
                    for w in result["windows"][:1]:
                        movers = ", ".join(f"{n} {d:.3f}m" for n, d in w["top_movers"][:3])
                        print(f"    top movers: {movers}")
                    ok = True
            elif modality == "listen" and len(leading) >= 3:
                from decode_snac import decode_snac_tokens
                n = (len(leading) // 3) * 3
                out_path = os.path.join(args.output_dir, f"{name}.wav")
                decode_snac_tokens(leading[:n], out_path)
                print(f"  Decoded -> {out_path}")
                ok = True
        except Exception as e:
            print(f"  decode FAILED: {e}")

        summary[name] = ok

    print("\n" + "=" * 60)
    print("SUMMARY (decoded to a real file):", summary)


if __name__ == "__main__":
    main()
