#!/usr/bin/env python3
"""
app.py — Hugging Face Spaces entry point for AI Music Supervisor.

WHY THIS APPROACH
─────────────────
HF Spaces requires the entry file to be named app.py.
ai_music_supervisor.py contains module-level st.set_page_config(),
st.markdown(CSS) and st.session_state initialisation calls that MUST
execute inside Streamlit's script-runner context on every rerun.

Importing the module (from ai_music_supervisor import main) would fire
those calls once at import time and never again on reruns, breaking
session state, CSS injection and the GUI layout on every widget
interaction.

The correct solution is to exec() the script's source code directly
inside this file's own Streamlit execution context.  This is
equivalent to running:
    streamlit run ai_music_supervisor.py
but satisfies HF Spaces' requirement for an app.py entry point.
"""

import pathlib as _pathlib

_src = (_pathlib.Path(__file__).parent / "ai_music_supervisor.py").read_text(encoding="utf-8")
exec(compile(_src, "ai_music_supervisor.py", "exec"), {"__name__": "__main__", "__file__": "ai_music_supervisor.py"})
