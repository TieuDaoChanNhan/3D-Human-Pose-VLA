"""
Gradio demo UI for EmpathicRobotics/vla-1.7b-qwen3-v6.

In:  a title and a content/description (text).
Out: the media the model's tokens decode to —

        pose  -> .mp4   (3D skeleton animation, <agent> tokens)
        video -> .mp4   (<cosmos_N> tokens, all complete chunks concatenated)
        image -> .png   (<seed2_N> tokens)
        audio -> .wav   (<snac_N> tokens, from whichever of <listen>/<speak>
                         the model produced -- <speak> is new in v6)

The text becomes a `### Title:`/`### Context:` prompt (the same header shape
training records use), the model continues it as an interleaved multimodal
token stream, and each modality present in that stream is decoded
independently — one failing (or simply not being generated) never blocks the
others.

Run:
    python vla_app.py
or, in Colab, import it and call build_ui().launch(share=True).
"""

from __future__ import annotations

import traceback

import gradio as gr
import torch

import vla_core as vc

MODEL = None
TOK = None

EXAMPLE_TITLE = "Morning stretch"
EXAMPLE_CONTENT = (
    "A person stands in a bright living room and slowly raises both arms "
    "above their head, then lowers them back down."
)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_model(model_id: str = vc.MODEL_ID):
    """Load the LM once. bf16 where the GPU really supports it (Ampere+),
    fp16 on older cards (T4) where bf16 is emulated and slow."""
    global MODEL, TOK
    if MODEL is not None:
        return MODEL, TOK

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    TOK = AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(device_map="auto", trust_remote_code=True)
    try:
        MODEL = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:                      # transformers < 4.56 spells it torch_dtype
        MODEL = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    MODEL.eval()
    print(f"Loaded {model_id} as {dtype} on {MODEL.device}")
    return MODEL, TOK


def _err(e: Exception) -> str:
    if isinstance(e, gr.Error):                 # expected, user-facing
        return f"⚠️ {getattr(e, 'message', e)}"
    traceback.print_exc()                       # real failure -> full trace in the log
    return f"❌ `{type(e).__name__}: {e}`"


# --------------------------------------------------------------------------
# prompt + generation
# --------------------------------------------------------------------------

def build_prompt(title: str, content: str) -> str:
    """Title + content -> the same `### Title:` / `### Context:` header shape
    training records use (see PROJECT_OVERVIEW.md 2.2.4). Emitting them as
    two separate lines (rather than folding title into one merged
    `### Context:` line) keeps the prompt in-distribution with what the model
    actually saw during training."""
    title, content = (title or "").strip(), (content or "").strip()
    if not title and not content:
        raise gr.Error("Enter a title or some content first.")
    lines = []
    if title:
        lines.append(f"### Title: {title}")
    if content:
        lines.append(f"### Context: {content}")
    return "\n".join(lines) + "\n"


def generate_once(prompt, max_new_tokens, do_sample, temperature, top_p,
                  repetition_penalty, seed):
    model, tok = load_model()
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))

    input_ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    n_in = input_ids.shape[1]
    # Read the real context length from the model's own config instead of
    # hardcoding one model's number (v2 was 4096, v6 is 8192) -- keeps this
    # correct across model swaps without needing another manual edit here.
    ctx = getattr(model.config, "max_position_embeddings", 4096)
    max_new_tokens = max(16, min(int(max_new_tokens), ctx - n_in))

    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=bool(do_sample),
                  repetition_penalty=float(repetition_penalty),
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if do_sample:
        kwargs.update(temperature=float(temperature), top_p=float(top_p))

    with torch.no_grad():
        out = model.generate(input_ids, **kwargs)

    text = tok.decode(out[0][n_in:], skip_special_tokens=False)
    return text.replace("<|endoftext|>", "").replace("<|im_end|>", "")


def _have(counts: dict) -> dict:
    """Which modalities are actually decodable from this generation."""
    return {
        # a bare <fps_30> is not a decodable window -- need real xyz tokens
        "pose": counts["agent_coords"] >= 3,
        "video": counts["cosmos"] >= vc.COSMOS_CHUNK,
        "image": counts["seed2"] >= vc.SEED2_QUERY_LEN,
        "audio": counts["speak_snac"] >= vc.SPEAK_GROUP
                 or counts["listen_snac"] >= vc.SNAC_GROUP,
    }


# --------------------------------------------------------------------------
# the one pipeline the UI runs
# --------------------------------------------------------------------------

SLOTS = ["status", "pose", "pose_note", "video", "video_note",
         "image", "image_note", "audio", "audio_note", "text", "prompt"]


def _frame(st: dict) -> tuple:
    return tuple(st.get(k) for k in SLOTS)


def run(title, content, max_new_tokens, do_sample, temperature, top_p,
        repetition_penalty, seed, attempts, want_image, max_chunks):
    """Generate, then decode every modality present. Yields as each finishes."""
    st = {k: None for k in SLOTS}
    for k in ("pose_note", "video_note", "image_note", "audio_note"):
        st[k] = "_waiting…_"

    try:
        prompt = build_prompt(title, content)
    except Exception as e:
        st["status"] = _err(e)
        yield _frame(st)
        return

    st["prompt"] = prompt
    st["status"] = "⏳ Loading the model…" if MODEL is None else "⏳ Generating…"
    yield _frame(st)

    # ---- generate (optionally re-roll until all four modalities appear) ----
    text, counts, have = "", {}, {}
    try:
        for i in range(max(1, int(attempts))):
            st["status"] = f"⏳ Generating… (attempt {i + 1}/{int(attempts)})"
            yield _frame(st)
            text = generate_once(prompt, max_new_tokens, do_sample, temperature,
                                 top_p, repetition_penalty,
                                 -1 if int(seed) < 0 else int(seed) + i)
            counts = vc.summarize(text)
            have = _have(counts)
            if all(have.values()):
                break
    except Exception as e:
        st["status"] = _err(e)
        yield _frame(st)
        return

    st["text"] = text
    missing = [k for k, v in have.items() if not v]
    st["status"] = (
        f"✅ Generated {counts['seed2']} seed2 / {counts['cosmos']} cosmos / "
        f"{counts['listen_snac']} listen-snac / {counts['speak_snac']} speak-snac / "
        f"{counts['agent_tokens']} pose tokens."
        + (f"\n\nNot produced this run: **{', '.join(missing)}** — re-run, raise "
           f"*max new tokens*, or increase *attempts*." if missing else "")
    )
    for k in ("pose", "video", "image", "audio"):
        if not have[k]:
            st[f"{k}_note"] = "_not produced in this generation_"
    inline = vc.inline_text(text)
    if inline:
        st["status"] += f"\n\n{inline}"
    yield _frame(st)

    # ---- decode, cheapest first so the UI fills in fast ----
    if have["pose"]:
        st["pose_note"] = "⏳ decoding…"
        yield _frame(st)
        try:
            trajectories, summary = vc.decode_agent(text)
            st["pose"] = vc.render_pose_video(trajectories)
            movers = ", ".join(f"{n} {d:.2f}m" for n, d in
                               summary["windows"][0]["top_movers"][:3])
            st["pose_note"] = (f"{summary['n_windows']} window(s), "
                               f"{summary['total_frames']} frames · top movers: {movers}")
        except Exception as e:
            st["pose_note"] = _err(e)
        yield _frame(st)

    if have["audio"]:
        # <speak> (model's own generated voice, full 3-level codebook) is
        # preferred over <listen> (input/"heard" role, level 2 dropped) when
        # both are present -- it's the more informative output and the new
        # capability this version adds over v2. Each format needs its own
        # decoder: same <snac_N> token family, different tokens-per-frame
        # layout, so picking the wrong one silently produces garbled audio.
        speak_ids = vc.wrapped_snac_runs(text, "speak")
        listen_ids = vc.wrapped_snac_runs(text, "listen")
        use_speak = len(speak_ids) >= vc.SPEAK_GROUP
        st["audio_note"] = ("⏳ decoding speak-format audio (model's own voice)…"
                            if use_speak else
                            "⏳ decoding listen-format audio…") + \
                           " (first run fetches SNAC, ~100 MB)"
        yield _frame(st)
        try:
            if use_speak:
                n = len(speak_ids) - len(speak_ids) % vc.SPEAK_GROUP
                st["audio"] = vc.decode_speak(speak_ids[:n])
                st["audio_note"] = (f"{n // vc.SPEAK_GROUP} frames ≈ "
                                    f"{n / vc.SPEAK_GROUP / 12.5:.2f}s @24 kHz "
                                    f"(speak format — all 3 codebook levels, this is "
                                    f"the model's own generated voice)")
            else:
                n = len(listen_ids) - len(listen_ids) % vc.SNAC_GROUP
                st["audio"] = vc.decode_snac(listen_ids[:n])
                st["audio_note"] = (f"{n // vc.SNAC_GROUP} base frames ≈ "
                                    f"{n / vc.SNAC_GROUP / 12.5:.2f}s @24 kHz "
                                    f"(listen format — level 2 was never encoded, so it "
                                    f"sounds coarse)")
        except Exception as e:
            st["audio_note"] = _err(e)
        yield _frame(st)

    if have["video"]:
        st["video_note"] = "⏳ decoding… (first run fetches the Cosmos decoder, ~350 MB)"
        yield _frame(st)
        try:
            # Every complete <cosmos_N> run across the WHOLE generation, not
            # just the first <cosmos>...</cosmos> block -- this is what lets
            # the clip run longer than a single 8-frame/~0.27s chunk. How
            # many chunks actually chain together is up to the model (see
            # the cosmos-persistence notes on the model card); `max_chunks`
            # only caps how many of the model's own chunks get decoded.
            ids = [i for run in vc.token_runs(text, "cosmos") for i in run]
            chunks, leftover = vc.chunked(ids, vc.COSMOS_CHUNK)
            used = chunks[:int(max_chunks)]
            st["video"] = vc.decode_cosmos(used)
            note = f"{len(used)}/{len(chunks)} chunk(s) = {len(used) * 8} frames @6 fps"
            if leftover:
                note += f" · dropped {leftover} trailing token(s) (incomplete chunk)"
            st["video_note"] = note
        except Exception as e:
            st["video_note"] = _err(e)
        yield _frame(st)

    if have["image"]:
        if not want_image:
            st["image_note"] = "_skipped (enable “decode image” in Advanced)_"
        else:
            st["image_note"] = ("⏳ decoding… (first run fetches ~7.6 GB: Q-former "
                                "+ diffusion — this one takes a while)")
            yield _frame(st)
            try:
                run0 = next(r for r in vc.token_runs(text, "seed2")
                            if len(r) >= vc.SEED2_QUERY_LEN)
                st["image"] = vc.decode_seed2(run0[:vc.SEED2_QUERY_LEN])
                st["image_note"] = ("generative reconstruction (diffusion img2img), "
                                    "not a codec round-trip — pixels vary run to run")
            except Exception as e:
                st["image_note"] = _err(e)
    yield _frame(st)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def build_ui():
    # Deliberately no theme= / show_copy_button= / other version-specific
    # kwargs below: this has to build on whatever gradio 4/5/6 pip resolves to.
    with gr.Blocks(title="VLA 1.7B Qwen3 v6 — multimodal demo") as demo:
        gr.Markdown(
            "# VLA 1.7B — Qwen3 v6\n"
            "Type a **title** and **content**. The model continues them as an "
            "interleaved multimodal token stream, which is decoded back into "
            "**pose video, video, image and audio** using the decoders bundled "
            "in the model repo (`tools/decode`, `tools/eval`). Audio may come "
            "back as `<listen>` (heard) or `<speak>` (the model's own "
            "generated voice, new in v6) — whichever the model produced."
        )

        with gr.Row():
            with gr.Column(scale=2):
                title = gr.Textbox(label="Title", value=EXAMPLE_TITLE,
                                   placeholder="Morning stretch")
                content = gr.Textbox(label="Content", value=EXAMPLE_CONTENT,
                                     lines=4,
                                     placeholder="Describe the scene or activity…")
            with gr.Column(scale=1):
                go = gr.Button("Generate", variant="primary")
                with gr.Accordion("Advanced", open=False):
                    max_new = gr.Slider(128, 6000, value=1200, step=64,
                                        label="max new tokens (context is 8192 for v6)")
                    attempts = gr.Slider(1, 3, value=1, step=1,
                                         label="attempts (stops early once all "
                                               "4 modalities appear)")
                    want_image = gr.Checkbox(
                        value=True, label="decode image (~7.6 GB on first use)")
                    max_chunks = gr.Slider(1, 15, value=3, step=1,
                                           label="max video chunks to decode "
                                                 "(more chunks = longer video, capped by "
                                                 "how many the model actually generated)")
                    do_sample = gr.Checkbox(value=True, label="sampling")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                            label="temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                    rep = gr.Slider(1.0, 2.0, value=1.3, step=0.05,
                                    label="repetition penalty")
                    seed = gr.Number(value=0, precision=0, label="seed (-1 = random)")

        status = gr.Markdown()

        with gr.Row():
            with gr.Column():
                pose_out = gr.Video(label="Pose video (.mp4)", height=300)
                pose_note = gr.Markdown()
            with gr.Column():
                video_out = gr.Video(label="Video (.mp4)", height=300)
                video_note = gr.Markdown()
        with gr.Row():
            with gr.Column():
                image_out = gr.Image(label="Image (.png)", type="filepath", height=300)
                image_note = gr.Markdown()
            with gr.Column():
                audio_out = gr.Audio(label="Audio (.wav) — listen or speak", type="filepath")
                audio_note = gr.Markdown()

        with gr.Accordion("Raw tokens / prompt", open=False):
            text_out = gr.Textbox(label="Generated tokens", lines=8, max_lines=20)
            prompt_out = gr.Textbox(label="Prompt sent to the model", lines=3)

        outputs = [status, pose_out, pose_note, video_out, video_note,
                   image_out, image_note, audio_out, audio_note,
                   text_out, prompt_out]
        inputs = [title, content, max_new, do_sample, temperature, top_p,
                  rep, seed, attempts, want_image, max_chunks]

        go.click(run, inputs, outputs)
        content.submit(run, inputs, outputs)
        title.submit(run, inputs, outputs)
    return demo


if __name__ == "__main__":
    if vc.REPO_DIR is None:
        from huggingface_hub import snapshot_download
        vc.setup(snapshot_download(
            vc.MODEL_ID,
            allow_patterns=["tools/*", "tools/**/*", "pipeline_pose/*"]))
    load_model()
    build_ui().launch(share=True)