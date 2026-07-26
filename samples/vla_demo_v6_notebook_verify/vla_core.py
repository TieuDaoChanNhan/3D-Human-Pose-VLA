"""
Encoding / decoding layer for EmpathicRobotics/vla-1.7b-qwen3-v6.

Every token convention here is delegated to the encoder/decoder scripts that
ship inside the model repo itself (``tools/encode``, ``tools/decode``,
``tools/eval``, ``pipeline_pose``), so the demo cannot drift from the model
card:

  image  -> <seed2_N>   32 raw ids / image   (tools/encode/encode_seed2.py)
  8 frames -> <cosmos_N> 200 raw ids / chunk (tools/encode/encode_cosmos.py)
  audio (heard)  -> <listen> <snac_N> ... </listen>
                    3 ids / base frame, 2 of 3 codebook levels
                                            (tools/encode/encode_snac.py)
  audio (spoken) -> <speak> <snac_N> ... </speak>
                    7 ids / base frame, all 3 codebook levels -- the model's
                    own generated voice, new in v6 (v2 never produced this)
                                            (tools/decode/decode_snac.py::decode_speak_tokens)
  pose   -> <agent> ...  adaptive PCHIP      (tools/encode/encode_agent.py)

and back:

  <cosmos_N> -> mp4              (tools/decode/decode_cosmos.py)
  <snac_N> inside <listen> -> wav (tools/decode/decode_snac.py::decode_snac_tokens)
  <snac_N> inside <speak>  -> wav (tools/decode/decode_snac.py::decode_speak_tokens)
  <seed2_N>  -> png               (tools/decode/decode_seed2.py, generative)
  <agent>    -> 3D pose           (tools/eval/decode_agent_tokens.py)

listen and speak share the same <snac_N> token family but different
tokens-per-frame layouts (3 vs 7) and codebook coverage (2 vs 3 levels) --
decoding one as the other doesn't raise, it just produces garbled audio (a
token's offset gets mapped to the wrong band). This module tracks which
*wrapper* a run came from (wrapped_snac_runs()), not just the token family,
so the caller always picks the matching decoder.

The only places this file does not literally call a repo function are noted
inline (cached model handles + multi-chunk cosmos decode), and they reproduce
the repo's own math/constants exactly.
"""

from __future__ import annotations

import functools
import glob
import os
import re
import subprocess
import sys
import tempfile

MODEL_ID = "EmpathicRobotics/vla-1.7b-qwen3-v6"

# Fixed by the checkpoints the model was trained against -- see model card.
SEED2_QUERY_LEN = 32     # Seed2Tokenizer Q-former query length, 1 image
COSMOS_CHUNK = 200       # DV8x16x16 grid (2,10,10) for an 8-frame/160x160 chunk
SNAC_GROUP = 3           # <listen> format: L0, L1a, L1b per base frame (2 of 3 codebook levels)
SPEAK_GROUP = 7          # <speak> format: L0, L1a, L2_0, L2_1, L1b, L2_2, L2_3 per base frame
                         # (all 3 codebook levels -- the model's own generated voice, v6-only)
AGENT_WINDOW = 8         # this model's window (NOT the newer 24-frame pipeline)
N_JOINTS = 17

REPO_DIR: str | None = None
_DEVICE: str | None = None


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

# The model card's pip line ("scipy numpy torch torchvision imageio-ffmpeg
# soundfile snac huggingface_hub") is incomplete: the vendored cosmos_tokenizer
# imports three more transitively -- video_lib -> utils -> mediapy, then
# utils -> networks -> loguru, and networks -> modules -> einops. Missing any
# of them only shows up as a ModuleNotFoundError once you decode a video, which
# is a long way into a demo, so check up front instead.
_RUNTIME_DEPS = {
    "mediapy": "mediapy",              # cosmos_tokenizer.utils
    "loguru": "loguru",                # cosmos_tokenizer.networks
    "einops": "einops",                # cosmos_tokenizer.modules
    "tqdm": "tqdm",                    # cosmos_tokenizer.video_lib
    "imageio_ffmpeg": "imageio-ffmpeg",  # every mp4/wav muxing step
    "snac": "snac",                    # audio decode
    "soundfile": "soundfile",          # audio decode
    "scipy": "scipy",                  # pose decode (PCHIP)
    "matplotlib": "matplotlib",        # pose render
    "diffusers": "diffusers",          # seed2 -> image
    "timm": "timm",                    # seed2 tokenizer (vendored BLIP-2 code)
}


def missing_deps() -> list[str]:
    """pip names of any runtime dependency that isn't importable."""
    import importlib.util

    out = []
    for mod, pkg in _RUNTIME_DEPS.items():
        try:
            ok = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            ok = False
        if not ok:
            out.append(pkg)
    return out


def setup(repo_dir: str) -> None:
    """Point this module at a snapshot_download() of the model repo."""
    global REPO_DIR
    REPO_DIR = os.path.abspath(repo_dir)
    for sub in ("tools/decode", "tools/decode/vendor", "tools/encode",
                "tools/eval", "pipeline_pose"):
        p = os.path.join(REPO_DIR, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    if not os.path.isdir(os.path.join(REPO_DIR, "tools")):
        raise RuntimeError(
            f"{REPO_DIR} has no tools/ directory -- download it with "
            f"snapshot_download('{MODEL_ID}', allow_patterns=['tools/*','tools/**/*',"
            f"'pipeline_pose/*'])"
        )
    miss = missing_deps()
    if miss:
        print("WARNING: the bundled decoders need packages that aren't installed.\n"
              f"         pip install {' '.join(miss)}")
    else:
        print(f"vla_core ready: {REPO_DIR}")


def device() -> str:
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _tmp(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, dir=tempfile.gettempdir())
    os.close(fd)
    return path


class _KeepCwd:
    """tools/decode/decode_seed2.py::_load_seed2_tokenizer() os.chdir()s into
    the seed2 checkpoint dir and never chdirs back. Harmless for a one-shot
    CLI, fatal for a long-lived UI process (every later relative path breaks),
    so restore it around any call that reaches that loader."""

    def __enter__(self):
        self._cwd = os.getcwd()
        return self

    def __exit__(self, *exc):
        os.chdir(self._cwd)
        return False


# --------------------------------------------------------------------------
# token-stream parsing
# --------------------------------------------------------------------------

def token_runs(text: str, family: str) -> list[list[int]]:
    """Maximal contiguous runs of ``<family_N>`` tokens, as lists of raw ids.

    A run is broken by any non-whitespace text between two tokens -- so the
    wrapper tags (``<seed2>``/``</seed2>``, ``<cosmos>``…) and interleaved
    caption/speech text all split runs correctly, and an unwrapped stream
    (the format the model card's own Usage example prompts with) still works.
    """
    pat = re.compile(rf"<{family}_(\d+)>")
    runs: list[list[int]] = []
    cur: list[int] = []
    prev_end = None
    for m in pat.finditer(text):
        if prev_end is not None and text[prev_end:m.start()].strip():
            if cur:
                runs.append(cur)
            cur = []
        cur.append(int(m.group(1)))
        prev_end = m.end()
    if cur:
        runs.append(cur)
    return runs


def wrapped_snac_runs(text: str, wrapper: str) -> list[int]:
    """All <snac_N> ids found strictly inside <wrapper>...</wrapper> blocks
    (wrapper: "listen" or "speak"), concatenated in order across every such
    block in the text.

    token_runs(text, "snac") only knows the ``snac`` token family, not which
    wrapper a given run came from -- but listen (3 tok/frame, 2 codebook
    levels) and speak (7 tok/frame, all 3 levels) are different layouts of
    the *same* token family, so telling them apart requires looking at the
    wrapper tag, not just the family regex. Decoding one format as the other
    doesn't raise, it silently produces garbled audio.
    """
    ids: list[int] = []
    for m in re.finditer(rf"<{wrapper}>(.*?)</{wrapper}>", text, re.DOTALL):
        ids.extend(int(x) for x in re.findall(r"<snac_(\d+)>", m.group(1)))
    return ids


def chunked(ids: list[int], size: int) -> tuple[list[list[int]], int]:
    """Split into fixed-size groups; returns (groups, n_leftover)."""
    n = len(ids) // size
    return [ids[i * size:(i + 1) * size] for i in range(n)], len(ids) - n * size


def summarize(text: str) -> dict:
    """Per-modality token counts for a generated sequence."""
    counts = {
        f: sum(len(r) for r in token_runs(text, f))
        for f in ("seed2", "cosmos", "snac")
    }
    counts["listen_snac"] = len(wrapped_snac_runs(text, "listen"))
    counts["speak_snac"] = len(wrapped_snac_runs(text, "speak"))
    counts["agent_tokens"] = len(re.findall(r"<(?:fps_\d+|[a-z_]+_[txyz]_\d+)>", text))
    # Coordinate tokens specifically: <fps_30> alone is not a decodable window.
    counts["agent_coords"] = len(re.findall(r"<[a-z_]+_[xyz]_\d+>", text))
    counts["caption_blocks"] = len(re.findall(r"<caption>", text))
    counts["speech_blocks"] = len(re.findall(r"<speech>", text))
    counts["listen_blocks"] = len(re.findall(r"<listen>", text))
    counts["speak_blocks"] = len(re.findall(r"<speak>", text))
    counts["agent_blocks"] = len(re.findall(r"<agent>", text))
    return counts


def inline_text(text: str) -> str:
    """Pull the readable <caption>/<speech> spans out of a generation."""
    out = []
    for tag in ("caption", "speech"):
        for m in re.finditer(rf"<{tag}>(.*?)(?:</{tag}>|$)", text, re.DOTALL):
            body = m.group(1).strip()
            if body:
                out.append(f"[{tag}] {body}")
    return "\n\n".join(out)


# --------------------------------------------------------------------------
# ENCODE:  image -> <seed2_N>
# --------------------------------------------------------------------------

def _patch_transformers_compat():
    """Backfill `find_pruneable_heads_and_indices` for transformers v5.

    seed2_tokenizer.py is vendored BLIP-2/LAVIS code written against
    transformers ~4.15 and imports three helpers from `modeling_utils` that
    later moved to `pytorch_utils`. The repo's own shim in
    decode_seed2._load_seed2_tokenizer() copies them back -- but v5 deleted
    one of the three from `pytorch_utils` as well, so that shim itself dies
    with `module 'transformers.pytorch_utils' has no attribute
    'find_pruneable_heads_and_indices'`.

    Verified against the real packages: 4.57.6's pytorch_utils defines all
    three; 5.14.1 keeps only apply_chunking_to_forward and prune_linear_layer.
    Restoring the missing one (4.57.6's implementation, verbatim) is enough --
    with it, `import seed2_tokenizer` succeeds on transformers 5.14.1 /
    timm 1.0.28 / torch 2.13, and every other v4-era import in that file
    (file_utils.ModelOutput, LlamaTokenizer, the modeling_outputs classes,
    timm's legacy module paths) still resolves.

    Must run *before* the repo's own shim loop, which then copies whatever
    modeling_utils is still missing out of pytorch_utils as usual.
    """
    import torch
    import transformers.modeling_utils as _mu
    import transformers.pytorch_utils as _pu

    def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                         already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    for mod in (_pu, _mu):
        if not hasattr(mod, "find_pruneable_heads_and_indices"):
            mod.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # v5 also dropped head-masking from ModuleUtilsMixin. The vendored Q-Former's
    # BertModel.forward() calls self.get_head_mask(...) unconditionally, so this
    # one fails at *inference* time rather than load time. `invert_attention_mask`
    # and `get_extended_attention_mask` -- the other two mixin methods that file
    # uses -- do still exist in v5 and are left alone.
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        return head_mask.to(dtype=self.dtype)

    # The other two mixin methods the vendored file calls -- invert_attention_mask
    # (x3) and get_extended_attention_mask (x2) -- DO still exist in 5.14.1, but
    # using them prints "deprecated and will be removed in v5.12.0". Since that
    # version has already shipped, the removal can land any release. These are
    # 4.57.6's implementations, installed only if the real ones are gone, so
    # today they are inert.
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        device = attention_mask.device
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)
        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [torch.ones((batch_size, seq_length, prefix_seq_len), device=device,
                            dtype=causal_mask.dtype), causal_mask], axis=-1)
        return causal_mask[:, None, :, :] * attention_mask[:, None, None, :]

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            ext = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            ext = encoder_attention_mask[:, None, None, :]
        ext = ext.to(dtype=self.dtype)
        return (1.0 - ext) * torch.finfo(self.dtype).min

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if attention_mask.dim() == 3:
            ext = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                maker = getattr(_mu.ModuleUtilsMixin,
                                "create_extended_attention_mask_for_decoder",
                                create_extended_attention_mask_for_decoder)
                ext = maker(input_shape, attention_mask)
            else:
                ext = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or "
                f"attention_mask (shape {attention_mask.shape})")
        ext = ext.to(dtype=dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    for _name, _fn in (("get_head_mask", get_head_mask),
                       ("_convert_head_mask_to_5d", _convert_head_mask_to_5d),
                       ("invert_attention_mask", invert_attention_mask),
                       ("get_extended_attention_mask", get_extended_attention_mask),
                       ("create_extended_attention_mask_for_decoder",
                        staticmethod(create_extended_attention_mask_for_decoder))):
        if not hasattr(_mu.ModuleUtilsMixin, _name):
            setattr(_mu.ModuleUtilsMixin, _name, _fn)


def _patch_seed2_loader():
    """Give Seed2Tokenizer the `all_tied_weights_keys` attribute v5 expects.

    In transformers v5 that attribute is an *instance* attribute set in exactly
    one place -- `PreTrainedModel.post_init()`. Seed2Tokenizer is v4.15-era
    code: its `__init__` calls `super().__init__(config)` and stops, so
    post_init never runs and `from_pretrained` dies in
    `mark_tied_weights_as_initialized` with `'Seed2Tokenizer' object has no
    attribute 'all_tied_weights_keys'`.

    An empty dict is the correct value, not just a silencer: the model has no
    tied weights, and the repo's own `get_output_embeddings -> None` guard
    already makes tying a no-op. Every v5 site that touches it either reads it
    or mutates it only while iterating its (empty) contents, so the shared
    class-level dict is never written to -- asserted in testing.

    Patches the module attribute rather than the class we get back, because
    `decode_seed2_tokens()` calls `_load_seed2_tokenizer()` through the module
    global -- this way the image *decode* path is covered too.
    """
    import decode_seed2

    if getattr(decode_seed2, "_vla_demo_patched", False):
        return
    _original = decode_seed2._load_seed2_tokenizer

    def _load_and_fix():
        cls, seed2_dir = _original()
        if not hasattr(cls, "all_tied_weights_keys"):
            cls.all_tied_weights_keys = {}
        return cls, seed2_dir

    decode_seed2._load_seed2_tokenizer = _load_and_fix
    decode_seed2._vla_demo_patched = True


def _prepare_seed2():
    """Both compat patches, applied before anything touches seed2."""
    _patch_transformers_compat()
    _patch_seed2_loader()


@functools.lru_cache(maxsize=1)
def _seed2_tokenizer():
    """Same load sequence as tools/encode/encode_seed2.py::encode_image(),
    kept as a cached handle so the UI doesn't re-read the 2.6GB Q-former on
    every click."""
    import torch
    _prepare_seed2()
    import decode_seed2                    # call through the module: see above

    with _KeepCwd():
        Seed2Tokenizer, seed2_dir = decode_seed2._load_seed2_tokenizer()
        tok = Seed2Tokenizer.from_pretrained(seed2_dir).eval()
        if torch.cuda.is_available():
            tok = tok.cuda()
    return tok


def encode_image(image_path: str) -> list[int]:
    """Real image -> 32 `<seed2_N>` ids (0-8191, no vocab offset)."""
    from PIL import Image
    from decode_seed2 import NUM_IMAGE_TOKENS

    tok = _seed2_tokenizer()
    # Seed2Tokenizer.encode_image() does its own CLIP-style 224x224 resize.
    ids = tok.encode_image(image_pil=Image.open(image_path).convert("RGB"))
    ids = ids.view(-1).tolist()
    bad = [t for t in ids if not (0 <= t < NUM_IMAGE_TOKENS)]
    if bad:
        raise ValueError(f"encode_image produced out-of-range ids: {bad[:5]}")
    return ids


def seed2_block(ids: list[int]) -> str:
    return "<seed2> " + " ".join(f"<seed2_{i}>" for i in ids) + " </seed2>"


# --------------------------------------------------------------------------
# ENCODE:  video / 8 frames -> <cosmos_N>
# --------------------------------------------------------------------------

def extract_frames(video_path: str, start_sec: float = 0.0,
                   fps: int = 30, n: int = 8) -> list[str]:
    """Pull n consecutive frames at `fps` starting at `start_sec`.

    The training pipeline chunked video into 8 consecutive 30fps frames, so
    that is the default here; the cosmos encoder then squashes each frame to
    160x160 (aspect-distorting on purpose -- this model's own convention).
    """
    import imageio_ffmpeg

    outdir = tempfile.mkdtemp(prefix="vla_frames_")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", str(start_sec),
           "-i", video_path, "-vf", f"fps={fps}", "-frames:v", str(n),
           os.path.join(outdir, "f_%02d.png")]
    res = subprocess.run(cmd, capture_output=True)
    files = sorted(glob.glob(os.path.join(outdir, "f_*.png")))
    if len(files) < n:
        raise RuntimeError(
            f"Only got {len(files)}/{n} frames from {os.path.basename(video_path)} "
            f"at t={start_sec}s. Pick an earlier start time.\n"
            f"{res.stderr.decode(errors='replace')[-400:]}"
        )
    return files[:n]


def encode_frames(frame_paths: list[str]) -> list[int]:
    """8 frames -> 200 `<cosmos_N>` ids, via the repo's own encoder."""
    from encode_cosmos import encode_frames as _encode_frames
    return _encode_frames(list(frame_paths))


def cosmos_block(ids: list[int]) -> str:
    return "<cosmos> " + " ".join(f"<cosmos_{i}>" for i in ids) + " </cosmos>"


# --------------------------------------------------------------------------
# ENCODE:  audio -> <snac_N>  (listen format)
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _snac_model():
    from snac import SNAC
    return SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device())


def encode_audio(input_path: str, max_seconds: float = 4.0) -> list[str]:
    """Audio/video file -> `<snac_N>` strings (3 per 12.5Hz base frame).

    Uses the repo's ffmpeg extraction + its verbatim ``encode_listen()``.
    Trimmed to ``max_seconds`` because SNAC costs 37.5 tokens/sec and the
    model's context is only 4096.
    """
    from encode_snac import extract_audio, SAMPLE_RATE
    from snac_finevideo import encode_listen

    audio = extract_audio(input_path)
    if max_seconds:
        audio = audio[:int(max_seconds * SAMPLE_RATE)]
    return encode_listen(audio, _snac_model(), device())


def snac_block(tokens: list[str]) -> str:
    return "<snac> " + " ".join(tokens) + " </snac>"


# --------------------------------------------------------------------------
# ENCODE:  3D pose -> <agent> ...
# --------------------------------------------------------------------------

def encode_pose(states) -> tuple[str, dict]:
    """(8, 17, 3) root-centred metres -> `<agent>` token block."""
    import numpy as np
    from phase5_adaptive_pchip import build_token_str, TARGET_FPS

    states = np.asarray(states, dtype=np.float32)
    if states.shape != (AGENT_WINDOW, N_JOINTS, 3):
        raise ValueError(
            f"Expected shape ({AGENT_WINDOW}, {N_JOINTS}, 3), got {states.shape}. "
            f"This model was trained on 8-frame windows."
        )
    pelvis = states[:, 0, :]
    if abs(pelvis).max() >= 1e-4:
        states = states - pelvis[:, None, :]      # root-centre, as the encoder requires
    token_str, cp_counts = build_token_str(states, fps=TARGET_FPS)
    return f"<agent> {token_str} </agent>", cp_counts


def load_pose_file(path: str):
    """.npy / .json ({"states": [...]}) -> (8,17,3) array, repo's own loader."""
    from encode_agent import load_states
    return load_states(path)


def demo_pose():
    """A synthetic 8-frame 'raise both arms' clip, so the UI is demoable with
    no motion-capture data on hand. Root-centred metres, H36M joint order."""
    import numpy as np

    base = np.array([
        [0.00,  0.00, 0.00],  # 0 pelvis
        [-0.13, -0.05, 0.00],  # 1 r_hip
        [-0.14, -0.45, 0.00],  # 2 r_knee
        [-0.15, -0.88, 0.00],  # 3 r_ankle
        [0.13, -0.05, 0.00],  # 4 l_hip
        [0.14, -0.45, 0.00],  # 5 l_knee
        [0.15, -0.88, 0.00],  # 6 l_ankle
        [0.00,  0.23, 0.00],  # 7 spine
        [0.00,  0.45, 0.00],  # 8 thorax
        [0.00,  0.57, 0.03],  # 9 nose
        [0.00,  0.68, 0.00],  # 10 head_top
        [0.18,  0.42, 0.00],  # 11 l_shoulder
        [0.34,  0.18, 0.00],  # 12 l_elbow
        [0.38, -0.08, 0.00],  # 13 l_wrist
        [-0.18,  0.42, 0.00],  # 14 r_shoulder
        [-0.34,  0.18, 0.00],  # 15 r_elbow
        [-0.38, -0.08, 0.00],  # 16 r_wrist
    ], dtype=np.float32)

    states = np.repeat(base[None], AGENT_WINDOW, axis=0)
    for t in range(AGENT_WINDOW):
        a = t / (AGENT_WINDOW - 1)                     # 0 -> 1, arms go up
        for elbow, wrist, side in ((12, 13, 1.0), (15, 16, -1.0)):
            states[t, elbow] = [side * (0.34 - 0.10 * a), 0.18 + 0.34 * a, 0.0]
            states[t, wrist] = [side * (0.38 - 0.16 * a), -0.08 + 0.86 * a, 0.0]
    return states


# --------------------------------------------------------------------------
# DECODE:  <cosmos_N> -> mp4
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _cosmos_decoder():
    from decode_cosmos import _resolve_checkpoint_dec
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    # CausalVideoTokenizer defaults to device="cuda" and torch.jit.load()s the
    # checkpoint straight onto it, so a CPU-only box dies inside torch's
    # map_location with an unrelated-looking error rather than falling back.
    # Pass the resolved device instead (the repo's decode_cosmos.py leaves the
    # default and only calls .to(device) afterwards, i.e. GPU-only).
    return CausalVideoTokenizer(checkpoint_dec=_resolve_checkpoint_dec(),
                                device=device()).to(device())


def decode_cosmos(chunks: list[list[int]], out_path: str | None = None,
                  fps: int = 6) -> str:
    """Decode N 200-token chunks into one mp4 -- this is what makes the
    output longer than a single 8-frame/~0.27s chunk: every complete 200-id
    group found across the whole generation (not just the first `<cosmos>`
    block) gets concatenated into one clip. Length is capped by `max_chunks`
    in the UI (and, upstream of that, by how many complete chunks the model
    actually produced before switching modality or hitting the token budget
    -- see the model card's cosmos-persistence notes).

    Same math as tools/decode/decode_cosmos.py::decode_cosmos_chunk() -- grid
    (2,10,10), output rescaled from [-1,1] to [0,1], ffmpeg at 6fps with
    nearest-neighbour upscale to 320x320 -- but the ~350MB jit decoder is
    loaded once and all chunks are concatenated into a single clip instead of
    one file per chunk.
    """
    import imageio_ffmpeg
    import torch
    import torchvision.transforms as T
    from decode_cosmos import CHUNK_GRID, CHUNK_TOKENS

    if not chunks:
        raise ValueError("No complete cosmos chunks to decode.")
    for c in chunks:
        if len(c) != CHUNK_TOKENS:
            raise ValueError(f"Expected {CHUNK_TOKENS} tokens per chunk, got {len(c)}")

    out_path = os.path.abspath(out_path or _tmp(".mp4"))
    dec = _cosmos_decoder()
    frame_dir = tempfile.mkdtemp(prefix="vla_cosmos_")
    to_pil = T.ToPILImage()

    idx = 0
    for chunk in chunks:
        indices = torch.tensor(chunk, dtype=torch.int64,
                               device=device()).view(1, *CHUNK_GRID)
        with torch.no_grad():
            out = dec.decode(indices)                     # (1,3,T,H,W) ~[-1,1]
        out = ((out.float() + 1.0) / 2.0).clamp(0, 1).squeeze(0)
        for i in range(out.shape[1]):
            to_pil(out[:, i].cpu()).save(os.path.join(frame_dir, f"frame_{idx:04d}.png"))
            idx += 1

    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps),
         "-i", os.path.join(frame_dir, "frame_%04d.png"),
         "-vf", "scale=320:320:flags=neighbor", "-pix_fmt", "yuv420p", out_path],
        check=True, capture_output=True,
    )
    return out_path


# --------------------------------------------------------------------------
# DECODE:  <snac_N> -> wav
# --------------------------------------------------------------------------

def decode_snac(ids: list[int], out_path: str | None = None) -> str:
    """Listen-format triplets -> 24kHz wav, via the repo's decoder.

    Use for ids pulled from a <listen>...</listen> block. For <speak>...
    </speak> ids, use decode_speak() instead -- different tokens-per-frame
    layout, calling this on speak-format ids produces garbled audio."""
    from decode_snac import decode_snac_tokens

    out_path = os.path.abspath(out_path or _tmp(".wav"))
    decode_snac_tokens(list(ids), out_path)
    return out_path


def decode_speak(ids: list[int], out_path: str | None = None) -> str:
    """Speak-format groups-of-7 -> 24kHz wav, via the repo's decoder.

    Unlike decode_snac() (listen-only: 3 tok/frame, level 2 zero-filled),
    this reconstructs the real level-2 codes -- this is the model's own
    generated voice, new in v6 (v2 never produced <speak>). Use for ids
    pulled from a <speak>...</speak> block."""
    from decode_snac import decode_speak_tokens

    out_path = os.path.abspath(out_path or _tmp(".wav"))
    decode_speak_tokens(list(ids), out_path)
    return out_path


# --------------------------------------------------------------------------
# DECODE:  <seed2_N> -> png  (generative reconstruction, not a round-trip)
# --------------------------------------------------------------------------

def decode_seed2(ids: list[int], out_path: str | None = None,
                 guidance_scale: float = 10.0, steps: int = 20) -> str:
    _prepare_seed2()                       # same v5 gaps as the encoder; see above
    from decode_seed2 import decode_seed2_tokens

    out_path = os.path.abspath(out_path or _tmp(".png"))
    with _KeepCwd():                       # the loader chdirs; see _KeepCwd
        decode_seed2_tokens(list(ids), out_path, guidance_scale, steps)
    return out_path


# --------------------------------------------------------------------------
# DECODE:  <agent> -> 3D pose (+ animation)
# --------------------------------------------------------------------------

# H36M skeleton, matching decode_agent_tokens.JOINT_NAMES order.
BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
         (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]


def decode_agent(text: str):
    """`<agent>` tokens -> (trajectories, json_summary) via the repo decoder.

    decode() already concatenates every window/block found in `text` (not
    just the first), so a generation with several consecutive <agent> blocks
    decodes to one longer trajectory, same as the multi-chunk cosmos path
    above."""
    from decode_agent_tokens import decode as _decode, to_json

    trajectories = _decode(text)
    if not trajectories:
        raise ValueError("No parseable <agent> window in this text.")
    return trajectories, to_json(trajectories)


def render_pose_video(trajectories, out_path: str | None = None,
                      fps: int = 6, supersample: int = 5) -> str:
    """Animate the decoded skeleton to an mp4.

    Raw per-window trajectories are only 8 real frames each -- stitching just
    those at a low fps (needed to make ~8 frames watchable at all) plays back
    as a hard-cut slideshow. `supersample` re-evaluates a PCHIP curve through
    the same frames at `supersample`x density before rendering, so motion
    looks continuous instead of stroboscopic; total playback duration is
    unchanged (output fps = fps * supersample). Pass supersample=1 for the
    old raw-frame behavior.
    """
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frames = np.concatenate(trajectories, axis=0)          # (F, 17, 3)
    if supersample > 1:
        from scipy.interpolate import PchipInterpolator
        n = frames.shape[0]
        t_in = np.arange(n, dtype=np.float64)
        t_out = np.linspace(0, n - 1, (n - 1) * supersample + 1)
        dense = np.zeros((len(t_out), frames.shape[1], 3), dtype=np.float32)
        for j in range(frames.shape[1]):
            for d in range(3):
                dense[:, j, d] = PchipInterpolator(t_in, frames[:, j, d])(t_out)
        frames = dense
        fps = fps * supersample

    out_path = os.path.abspath(out_path or _tmp(".mp4"))
    frame_dir = tempfile.mkdtemp(prefix="vla_pose_")

    lo, hi = float(frames.min()) - 0.1, float(frames.max()) + 0.1
    fig = plt.figure(figsize=(4.0, 4.0), dpi=120)          # 480x480, even dims
    ax = fig.add_subplot(111, projection="3d")

    for f in range(frames.shape[0]):
        ax.clear()
        p = frames[f]
        # x right, z depth, y up -- plot (x, z, y) so the figure stands up.
        for a, b in BONES:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 2], p[b, 2]],
                    [p[a, 1], p[b, 1]], lw=2.2, color="#2b6cb0")
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=14, color="#e53e3e")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_title(f"frame {f + 1}/{frames.shape[0]}", fontsize=9)
        fig.savefig(os.path.join(frame_dir, f"frame_{f:04d}.png"))
    plt.close(fig)

    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps),
         "-i", os.path.join(frame_dir, "frame_%04d.png"),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",       # yuv420p needs even dims
         "-pix_fmt", "yuv420p", out_path],
        check=True, capture_output=True,
    )
    return out_path