# Minimal stub so `vla_app.py` (which does `import gradio as gr` at module
# level) can be imported for a headless logic test, without installing the
# real gradio package into the shared env_stable_vla env (installing it for
# real pulled in an incompatible huggingface_hub>=1.0, breaking transformers
# -- reverted). Only the names vla_app.py actually references at import time
# / in the code paths this test exercises (gr.Error) need to exist.
class Error(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class _Stub:
    def __getattr__(self, name):
        raise AttributeError(
            f"gradio stub: '{name}' not implemented -- this test doesn't call "
            f"build_ui()/launch(), only the core generate/decode logic."
        )


Blocks = _Stub
Markdown = _Stub
Row = _Stub
Column = _Stub
Accordion = _Stub
Textbox = _Stub
Button = _Stub
Slider = _Stub
Checkbox = _Stub
Number = _Stub
Video = _Stub
Image = _Stub
Audio = _Stub
