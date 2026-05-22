#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# VERSION
# ============================================================
SCRIPT_NAME      = "screenplay_parser_engine.py"
SCRIPT_VERSION   = "3.07"
PIPELINE_VERSION = f"{SCRIPT_NAME[:-3]}_v{SCRIPT_VERSION}"

"""
screenplay_parser_engine.py — v3.05

Business logic extracted verbatim from cl_screenplay_parser.py v3.14.4.
Zero Streamlit imports — importable from tests, CLI, or notebooks.

Change log
──────────
v3.07 (2026-05-22)
  • BUG FIX: read_docx_bytes() ignored <w:br/> (soft return / SHIFT+ENTER)
    elements inside paragraphs. Polish screenplays written in MS Word commonly
    use SHIFT+ENTER to separate dialogue lines within a single <w:p> block.
    The previous parser only walked <w:t> nodes, so every line in the paragraph
    concatenated without any separator — producing a single run-on string.
    Fix: the parser now iterates all child elements of each <w:p>; a <w:br/>
    emits "\n" and a <w:br w:type="page"/> emits "\n\n"; text nodes are
    collected as before. normalize_whitespace() then folds excess newlines.
v3.06 (2026-05-13)
  • Added return type annotation ``psycopg2.extensions.connection``
    to the module-level get_connection() — the only function in this
    module that was missing one.
v3.05 (2026-05-13)
  • ISSUE-02: Replaced old-style typing generics (List[…], Tuple[…])
    with built-in lowercase equivalents (list[…], tuple[…]) throughout.
    from __future__ import annotations already active; no runtime cost.
    Removed List, Tuple from `from typing import …`.
  • ISSUE-03: Added __all__ — declares the public API surface so
    `from screenplay_parser_engine import *` and IDE tooling only expose
    the intended symbols.
v3.04 (2026-05-10)
  • BUG FIX: invalid escape sequence '\.' in docstring at line 572
    (_extract_scene_keyword examples block). Replaced with '\\.' —
    Python 3.12+ raises SyntaxWarning for unrecognised escape sequences
    in plain strings; HF Spaces log confirmed the warning on startup.
v3.0 (2026-05-07)
  • Extracted verbatim from cl_screenplay_parser.py v3.14.4.
    New file name: screenplay_parser_engine.py (GUI: screenplay_parser.py).
  • HIGH: import streamlit removed; zero st.* calls in this file.
  • HIGH: _runtime_openai_key() no longer reads st.session_state.
    GUI injects the active key via set_runtime_openai_key(key).
    Falls back to OPENAI_API_KEY env var when not injected.
  • HIGH: _ANTHROPIC_CLIENT singleton added — mirrors _OPENAI_CLIENT;
    _call_anthropic_cached() uses _get_anthropic_client() instead of
    creating anthropic.Anthropic() on every call.
  • HIGH: run_llm2_embed_save() is now the single Stage-3 implementation.
    The inline duplicate loop in render_processing_section() is removed
    from the GUI; GUI calls this function with an on_progress callback.
  • MEDIUM: get_uploaded_file_bytes() removed (accepted Streamlit
    UploadedFile, coupling engine to st). GUI extracts (name, bytes)
    and calls read_screenplay(name, bytes) directly.
  • MEDIUM: _validate_openai_key() and _invalidate_openai_singleton()
    moved here so GUI can call them without re-importing OpenAI.
  • LOW: make_scenes_query() marked DeprecationWarning (legacy shim).
v3.02 (2026-05-07)
  • split_scenes(): replaced raw regex pattern in the "No scene headers
    found" error with a human-readable keyword extracted from the pattern
    (e.g. "SCENA" or "SCENE"). Added _extract_scene_keyword() helper.
"""

import io
import json
import logging
import os
import re
import subprocess
import threading
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from openai import OpenAI
from psycopg2.extras import Json
from pydantic import BaseModel, Field, field_validator
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from transformers import ClapModel, ClapProcessor

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OPENAI_API_KEY:    str  = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str  = os.getenv("ANTHROPIC_API_KEY", "")
MODEL:             str  = os.getenv("MODEL", "gpt-4.1")

# Set USE_ANTHROPIC_CACHE=1 when routing through the Anthropic Messages API.
# OpenAI benefits from automatic prefix caching with no markup needed.
USE_ANTHROPIC_CACHE: bool = os.getenv("USE_ANTHROPIC_CACHE", "0").lower() in ("1", "true", "yes")

CLAP_MODEL:  str          = os.getenv("CLAP_MODEL", "laion/clap-htsat-unfused")
VECTOR_DIM:  int          = int(os.getenv("VECTOR_DIM", "512"))
OUTPUT_DIR:   Path         = Path(os.getenv("OUTPUT_DIR", "output"))
HF_TOKEN:     Optional[str] = os.getenv("HF_TOKEN")
CLOUD_MUSIC:  bool          = os.getenv("CLOUD_MUSIC", "false").lower() in ("1", "true", "yes")
BUCKET_NAME:  str           = os.getenv("BUCKET_NAME", "")
MUSIC_DIR:    str           = os.getenv("MUSIC_DIR", "")
PGDATABASE:   str           = os.getenv("PGDATABASE", "")

_DEFAULT_SCENE_PATTERN = r"(?mi)^\s*SCENA\s+(\d+)\s*[\.\:\-]?\s*(.*)$"
SCENE_HEADER_RE: re.Pattern = re.compile(
    os.getenv("SCENE_HEADER_PATTERN", _DEFAULT_SCENE_PATTERN)
)

SUPPORTED_SCREENPLAY_SUFFIXES = {".doc", ".docx"}

DB_CONFIG = {
    "host":            os.getenv("PGHOST"),
    "dbname":          os.getenv("PGDATABASE"),
    "user":            os.getenv("PGUSER"),
    "password":        os.getenv("PGPASSWORD"),
    "sslmode":         os.getenv("PGSSLMODE"),
    "channel_binding": os.getenv("PGCHANNELBINDING"),
}

# CACHE-2 — module-level OpenAI client singleton; keeps the TCP connection alive
# and ensures the prompt prefix hash is stable across sequential scene calls.
_OPENAI_CLIENT:      Optional[OpenAI]    = None
_OPENAI_CLIENT_LOCK: threading.Lock      = threading.Lock()

# Anthropic client singleton — mirrors _OPENAI_CLIENT pattern.
# Prevents a new HTTPS connection on every _call_anthropic_cached() call.
_ANTHROPIC_CLIENT:      Optional[Any]  = None
_ANTHROPIC_CLIENT_LOCK: threading.Lock = threading.Lock()

# Thread-safe CLAP singleton
_CLAP_LOCK:           threading.Lock         = threading.Lock()
_CLAP_PROCESSOR:      Optional[ClapProcessor] = None
_CLAP_MODEL_INSTANCE: Optional[ClapModel]     = None
_CLAP_DEVICE:         str                     = "cuda" if torch.cuda.is_available() else "cpu"


# Runtime API-key injection — GUI calls set_runtime_openai_key() instead of
# writing st.session_state so the engine never imports Streamlit.
_RUNTIME_OPENAI_KEY: str            = ""
_RUNTIME_KEY_LOCK:   threading.Lock = threading.Lock()


# ============================================================
# PUBLIC API
# ============================================================
__all__ = [
    # Pydantic data models
    "ThemeBase", "ThemeList",
    "SceneMusicSemantics", "SceneMusicTargets", "SceneTagSelection",
    "ClapPromptEnsemble", "ThemeQuery", "SceneQueryRecord", "SceneEmbeddings",
    # Runtime key management
    "set_runtime_openai_key",
    "_runtime_openai_key",
    "_validate_openai_key",
    "_invalidate_openai_singleton",
    # LLM pipeline
    "LLM_1_theme_list", "LLM_2_theme_query", "make_theme_query_record",
    # CLAP / embeddings
    "get_clap_model", "release_clap_model",
    "encode_text", "encode_texts_batch", "create_embeddings",
    # Screenplay I/O
    "read_screenplay", "split_scenes",
    # Database
    "get_connection",
    "delete_episode_scene_queries",
    "insert_scene_query_row", "save_theme_query_record",
    "fetch_scene_query_records",
    # Orchestration
    "run_llm1_all_scenes", "run_llm2_embed_save", "make_scenes_query",
    # PDF export
    "register_pdf_fonts", "build_pdf_story", "create_pdf_scenes",
    # Misc config
    "SUPPORTED_SCREENPLAY_SUFFIXES", "MODEL",
    "CLOUD_MUSIC", "BUCKET_NAME", "MUSIC_DIR", "PGDATABASE",
    "SCRIPT_NAME", "SCRIPT_VERSION", "PIPELINE_VERSION",
]


def set_runtime_openai_key(key: str) -> None:
    """Inject the active OpenAI key from the GUI (thread-safe).

    Call this whenever the user saves or changes their key.
    Invalidates the OpenAI client singleton when the key changes.
    """
    global _RUNTIME_OPENAI_KEY
    key = (key or "").strip()
    with _RUNTIME_KEY_LOCK:
        if key != _RUNTIME_OPENAI_KEY:
            _RUNTIME_OPENAI_KEY = key
            _invalidate_openai_singleton()


def _runtime_openai_key() -> str:
    """Return active OpenAI key: injected runtime key > OPENAI_API_KEY env."""
    with _RUNTIME_KEY_LOCK:
        if _RUNTIME_OPENAI_KEY:
            return _RUNTIME_OPENAI_KEY
    return OPENAI_API_KEY or ""


def _get_anthropic_client() -> Any:
    """Return a cached anthropic.Anthropic client, creating one if needed."""
    global _ANTHROPIC_CLIENT
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "anthropic package required when USE_ANTHROPIC_CACHE=1. "
            "Install: pip install anthropic"
        ) from exc
    with _ANTHROPIC_CLIENT_LOCK:
        if _ANTHROPIC_CLIENT is None or _ANTHROPIC_CLIENT.api_key != ANTHROPIC_API_KEY:
            _ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return _ANTHROPIC_CLIENT


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    key = _runtime_openai_key()
    if not key:
        raise RuntimeError(
            "OpenAI API key is not set. "
            "Enter your key in the sidebar and click Save."
        )
    with _OPENAI_CLIENT_LOCK:
        # Reset singleton when the key has changed (e.g. user entered a new one)
        if _OPENAI_CLIENT is not None and _OPENAI_CLIENT.api_key != key:
            _OPENAI_CLIENT = None
        if _OPENAI_CLIENT is None:
            _OPENAI_CLIENT = OpenAI(api_key=key)
        return _OPENAI_CLIENT


def _validate_openai_key(key: str) -> Optional[str]:
    """Check whether key is accepted by the OpenAI API.

    Returns None on success, or an error string on failure.
    Moved from GUI section to engine so GUI does not need to import OpenAI.
    """
    try:
        client = OpenAI(api_key=key)
        client.models.list()
        return None
    except Exception as exc:
        msg = str(exc)
        if "Incorrect API key" in msg or "invalid_api_key" in msg:
            return "Incorrect API key — please check and try again."
        if "401" in msg:
            return "Authentication failed (401) — key is invalid or expired."
        if "Connection" in msg or "Timeout" in msg or "timeout" in msg:
            return "Could not reach OpenAI — check your internet connection."
        return f"OpenAI rejected the key: {msg[:120]}"


def _invalidate_openai_singleton() -> None:
    """Reset the module-level OpenAI client so the next call creates a fresh one."""
    global _OPENAI_CLIENT
    with _OPENAI_CLIENT_LOCK:
        _OPENAI_CLIENT = None


# ============================================================
# PYDANTIC MODELS — LLM #1
# ============================================================

class ThemeBase(BaseModel):
    theme_title_pl: str = Field(..., min_length=1)
    theme_title_en: str = Field(..., min_length=1)
    description_en: str = Field(..., min_length=1)
    theme_txt:      str = Field(..., min_length=1)

    @field_validator("theme_title_pl", "theme_title_en", "description_en", "theme_txt")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty.")
        return value


class ThemeList(BaseModel):
    themes:               list[ThemeBase]
    segmentation_reason:  str = Field(..., min_length=1)

    @field_validator("themes")
    @classmethod
    def validate_themes_not_empty(cls, value: list[ThemeBase]) -> list[ThemeBase]:
        if not value:
            raise ValueError("themes cannot be empty.")
        return value

    @field_validator("segmentation_reason")
    @classmethod
    def strip_segmentation_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("segmentation_reason cannot be empty.")
        return value


# ============================================================
# PYDANTIC MODELS — LLM #2
# ============================================================

class SceneMusicSemantics(BaseModel):
    emotional_direction:    list[str]
    narrative_function:     list[str]
    weight_profile:         str = Field(..., min_length=1)
    dialogue_safe_required: bool

    @field_validator("weight_profile")
    @classmethod
    def strip_weight_profile(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("weight_profile cannot be empty.")
        return value


class SceneMusicTargets(BaseModel):
    energy_target:          str = Field(..., min_length=1)
    tempo_target:           str = Field(..., min_length=1)
    rhythm_target:          str = Field(..., min_length=1)
    intensity_shape_target: str = Field(..., min_length=1)
    sound_character_target: list[str]

    @field_validator("energy_target", "tempo_target", "rhythm_target", "intensity_shape_target")
    @classmethod
    def strip_targets(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Target field cannot be empty.")
        return value


class SceneTagSelection(BaseModel):
    should_have_tags: list[str]
    must_not_tags:    list[str]


# LLM2-C4 — max_length=120 enforces the "max 14 words" contract at the Pydantic layer.
class ClapPromptEnsemble(BaseModel):
    semantic_scene_prompt: str = Field(..., min_length=1, max_length=120)
    music_for_scene_prompt: str = Field(..., min_length=1, max_length=120)
    emotion_prompt:         str = Field(..., min_length=1, max_length=120)
    narrative_prompt:       str = Field(..., min_length=1, max_length=120)
    sonic_prompt:           str = Field(..., min_length=1, max_length=120)
    tag_prompt:             str = Field(..., min_length=1, max_length=120)
    concise_core_prompt:    str = Field(..., min_length=1, max_length=120)

    @field_validator(
        "semantic_scene_prompt", "music_for_scene_prompt", "emotion_prompt",
        "narrative_prompt", "sonic_prompt", "tag_prompt", "concise_core_prompt",
    )
    @classmethod
    def strip_prompts(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("CLAP prompt cannot be empty.")
        return value


class ThemeQuery(BaseModel):
    # LLM2-C5 — max_length=120 enforces the "max 14 words" contract
    tags_summary_en:        str = Field(..., min_length=1, max_length=120)
    scene_music_semantics:  SceneMusicSemantics
    scene_music_targets:    SceneMusicTargets
    scene_tag_selection:    SceneTagSelection
    clap_prompt_ensemble:   ClapPromptEnsemble

    @field_validator("tags_summary_en")
    @classmethod
    def strip_tags_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tags_summary_en cannot be empty.")
        return value


# ============================================================
# PYDANTIC MODELS — FINAL RECORD
# ============================================================

class SceneQueryRecord(BaseModel):
    episode_nr:           int
    scene_nr:             int
    theme_nr:             int
    segmentation_reason:  str = Field(..., min_length=1)
    theme_title_pl:       str = Field(..., min_length=1)
    theme_title_en:       str = Field(..., min_length=1)
    description_en:       str = Field(..., min_length=1)
    theme_txt:            str = Field(..., min_length=1)
    tags_summary_en:      str = Field(..., min_length=1)
    scene_music_semantics: SceneMusicSemantics
    scene_music_targets:   SceneMusicTargets
    scene_tag_selection:   SceneTagSelection
    clap_prompt_ensemble:  ClapPromptEnsemble


class SceneEmbeddings(BaseModel):
    embedding_main:          list[float]
    embedding_tags:          list[float]
    embedding_clap_ensemble: list[float]
    embedding_hybrid:        list[float]


# ============================================================
# DB HELPERS
# ============================================================

def get_connection() -> psycopg2.extensions.connection:
    """Open and return a new psycopg2 connection using module-level DB_CONFIG."""
    return psycopg2.connect(**DB_CONFIG)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _delete_episode_scene_queries_cur(cur, episode_nr: int) -> None:
    cur.execute("DELETE FROM scene_query WHERE episode_nr = %s", (episode_nr,))


def delete_episode_scene_queries(episode_nr: int) -> None:
    if not isinstance(episode_nr, int) or episode_nr <= 0:
        raise ValueError("episode_nr must be a positive integer.")
    conn = get_connection()
    cur  = conn.cursor()
    try:
        _delete_episode_scene_queries_cur(cur, episode_nr)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_episode_nr(file_name: str) -> int:
    match = re.search(r"(\d{1,6})", Path(file_name).stem)
    if not match:
        raise ValueError(f"Cannot extract episode number from filename: {file_name}")
    return int(match.group(1))


# ============================================================
# SCREENPLAY READING
# ============================================================

def read_docx_bytes(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        xml_data = zf.read("word/document.xml")
    root = ET.fromstring(xml_data)
    ns    = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    para_strings: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts: list[str] = []
        for elem in para.iter():
            # Strip namespace prefix to get the local tag name
            local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if local == "t" and elem.text:
                parts.append(elem.text)
            elif local == "br":
                # w:br w:type="page"  → paragraph / page break → double newline
                # w:br (no type) or w:type="textWrapping" → soft return (SHIFT+ENTER)
                br_type = elem.get(f"{{{ns['w']}}}type", "")
                parts.append("\n\n" if br_type == "page" else "\n")
        para_strings.append("".join(parts))
    return "\n".join(para_strings)


def _find_soffice() -> Optional[str]:
    """
    Return the full path to the LibreOffice soffice executable, or None.
    Checks PATH first, then common Windows installation directories.
    """
    import shutil
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    # Common Windows installation paths
    win_candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for candidate in win_candidates:
        if Path(candidate).exists():
            return candidate
    return None


def read_doc_bytes(file_name: str, data: bytes) -> str:
    """
    Extract plain text from a legacy .doc file supplied as raw bytes.

    All strategies work from the in-memory bytes directly.
    A NamedTemporaryFile is written once and reused by every strategy
    that needs a filesystem path (antiword, textract, soffice).

    Strategy 1 — antiword   (text-only, fast)
    Strategy 2 — textract   (Python package)
    Strategy 3 — LibreOffice soffice --headless
                 Converts .doc → .docx in a sibling temp dir,
                 then parses the result with read_docx_bytes().
    """
    suffix = Path(file_name).suffix.lower()
    if suffix != ".doc":
        raise ValueError("read_doc_bytes supports only .doc files.")

    import tempfile

    errors: list[str] = []

    # Write bytes to a single NamedTemporaryFile; all strategies share it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        # ── Strategy 1: antiword ──────────────────────────────────────────
        try:
            result = subprocess.run(
                ["antiword", str(tmp_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True,
            )
            if result.stdout.strip():
                return result.stdout
        except FileNotFoundError:
            errors.append("antiword: not found")
        except Exception as exc:
            errors.append(f"antiword: {exc}")

        # ── Strategy 2: textract ──────────────────────────────────────────
        try:
            import textract  # type: ignore
            raw = textract.process(str(tmp_path))
            text = raw.decode("utf-8", errors="replace")
            if text.strip():
                return text
        except ImportError:
            errors.append("textract: not installed")
        except Exception as exc:
            errors.append(f"textract: {exc}")

        # ── Strategy 3: LibreOffice headless ─────────────────────────────
        soffice_exe = _find_soffice()
        if soffice_exe is None:
            errors.append(
                "soffice/libreoffice: not found on PATH or common install paths"
            )
        else:
            with tempfile.TemporaryDirectory() as out_dir:
                try:
                    subprocess.run(
                        [
                            soffice_exe, "--headless",
                            "--convert-to", "docx",
                            "--outdir", out_dir,
                            str(tmp_path),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120,
                        check=True,
                    )
                except Exception as exc:
                    errors.append(f"soffice conversion failed: {exc}")
                else:
                    converted = Path(out_dir) / (tmp_path.stem + ".docx")
                    if converted.exists():
                        return read_docx_bytes(converted.read_bytes())
                    errors.append(
                        f"soffice ran but produced no .docx in {out_dir}"
                    )

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    raise RuntimeError(
        "Failed to read .doc file. All three strategies failed:\n"
        + "\n".join(f"  · {e}" for e in errors)
        + "\nOptions: install antiword or textract, install LibreOffice, "
        "or convert the file to .docx before uploading."
    )


def load_screenplay_from_bytes(file_name: str, data: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_SCREENPLAY_SUFFIXES:
        raise ValueError(
            f"Unsupported screenplay format: {suffix}. "
            f"Supported: {sorted(SUPPORTED_SCREENPLAY_SUFFIXES)}"
        )
    return read_docx_bytes(data) if suffix == ".docx" else read_doc_bytes(file_name, data)


def _extract_scene_keyword(pattern: re.Pattern) -> str:
    """Extract the human-readable scene keyword from the compiled regex.

    Looks for the first ALL-CAPS word (2+ letters) that is a literal in
    the pattern source.  Falls back to the raw pattern string so the
    message is always non-empty.

    Examples
    --------
    '(?mi)^\\s*SCENA\\s+(\\d+)...'  →  "SCENA"
    '(?mi)^\\s*SCENE\\s+(\\d+)...'  →  "SCENE"
    '(?mi)^\\s*INT\\.\\s+...'          →  "INT"
    """
    # Find all sequences of uppercase ASCII letters (2+ chars) that appear
    # as literals in the pattern source — these are the keywords we care about.
    candidates = re.findall(r"[A-Z]{2,}", pattern.pattern)
    if candidates:
        return candidates[0]
    return pattern.pattern


def split_scenes(full_text: str) -> list[str]:
    matches = list(SCENE_HEADER_RE.finditer(full_text))
    if not matches:
        keyword = _extract_scene_keyword(SCENE_HEADER_RE)
        raise ValueError(
            f"No scene headers found. "
            f"Expected keyword: \"{keyword}\"\n"
            f"Make sure the screenplay uses that word to start every scene "
            f"(e.g. \"{keyword} 1.\" or \"{keyword} 1 INT. LIVING ROOM\"). "
            f"You can change the keyword by setting the SCENE_HEADER_PATTERN "
            f"environment variable."
        )
    scenes: list[str] = []
    for idx, match in enumerate(matches):
        start     = match.start()
        end       = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        scene_txt = full_text[start:end].strip()
        if scene_txt:
            scenes.append(scene_txt)
    if not scenes:
        raise ValueError("No non-empty scenes extracted from screenplay.")
    return scenes


def read_screenplay(file_name: str, file_bytes: bytes) -> tuple[int, list[str]]:
    episode_nr = extract_episode_nr(file_name)
    full_text  = load_screenplay_from_bytes(file_name, file_bytes)
    full_text  = normalize_whitespace(full_text)
    if not full_text:
        raise ValueError("Loaded screenplay text is empty.")
    return episode_nr, split_scenes(full_text)




# ============================================================
# LLM PROMPTS — LLM #1
#
# LLM1-C1: All field rules live in system; user prompt is just the raw scene text.
# LLM1-C2: segmentation_reason is visually separated as a ThemeList-level field.
# LLM1-C3: Explicit empty-input handling instruction.
# LLM1-C4: Inline ThemeList JSON schema pushes system prompt past the 1 024-token
#           caching threshold (~700 tokens after embedding).
# ============================================================

LLM1_SYSTEM_PROMPT = """\
Jesteś analitykiem semantycznym scen filmowych i serialowych.
Twoim zadaniem jest podzielić pojedynczą scenę na semantycznie spójne tematy
i przygotować podstawowe definicje tych tematów do dalszego wyszukiwania muzyki.

=== WYMAGANA STRUKTURA JSON (ThemeList) ===

{
  "themes": [
    {
      "theme_title_pl": "string — krótki tytuł po polsku, 2-6 słów, semantyczny, nie techniczny",
      "theme_title_en": "string — krótki tytuł po angielsku, 2-6 słów, semantyczny, nie dosłowny",
      "description_en": "string — maks. 16 słów, po angielsku; opisuje sytuację, emocję i sens tematu; nie może przepisywać nagłówka, nie może kopiować dialogu, nie może zawierać znaczników scenariusza",
      "theme_txt": "string — DOKŁADNY tekst sceny należący do tego tematu, bez parafrazy i bez skracania"
    }
  ],
  "segmentation_reason": "string — pole ThemeList (NIE per-theme); 1-2 zdania po angielsku; wyjaśnia decyzję segmentacyjną dla CAŁEJ sceny"
}

=== ZASADY SEGMENTACJI (pole themes) ===

Dziel scenę na tematy tylko wtedy, gdy jest to uzasadnione semantycznie.
Rozbijaj scenę na maksymalnie 2 tematy, nawet jeśli jest długa i złożona.
Jeden temat jeśli scena jest semantycznie spójna.
Dwa tematy tylko wtedy, jeśli zmienia się co najmniej jedno z poniższych:
  - główny temat sceny
  - dominująca emocja
  - funkcja narracyjna
  - oczekiwana rola muzyki
Nie dziel sceny tylko ze względu na długość lub liczbę wypowiedzi.

=== ZASADY pola segmentation_reason (ThemeList-level) ===

Jeden wpis dla całej sceny, niezależnie od liczby tematów.
Wyjaśnia logikę podziału lub braku podziału.

=== OBSŁUGA BŁĘDÓW ===

Jeśli scene_txt jest pusty, nieczytelny lub zawiera wyłącznie whitespace / numerację stron,
zwróć: "themes": [] oraz segmentation_reason: "Input text is empty or unreadable."

=== ZAKRES ZADANIA ===

Nie analizuj muzyki. Nie twórz tagów. Nie twórz promptów CLAP.
Zwróć WYŁĄCZNIE obiekt JSON zgodny ze strukturą ThemeList powyżej.
"""

# LLM1-C1: user prompt is minimal — just the raw scene text, no repeated rules.
LLM1_USER_PROMPT_TEMPLATE = "Scena:\n{scene_txt}"


# ============================================================
# LLM PROMPTS — LLM #2
#
# LLM2-C1: 'release' namespace collision resolved — narrative category renames
#           the value to 'narrative_release' to avoid ambiguity with
#           intensity_shape::release in tag dictionaries.
# LLM2-C2: weight_profile (scene type) vs narrative_function (music role)
#           explicitly differentiated.
# LLM2-C3: dialogue_safe_required given a precise operational definition.
# LLM2-C6: Full ThemeQuery schema embedded (system already > 1 024 tokens).
# LLM2-C1: User prompt minimal — only theme_txt.
# ============================================================

LLM2_SYSTEM_PROMPT = """\
Jesteś AI Music Supervisor przygotowującym jeden temat sceny do wyszukiwania muzyki.
Otrzymujesz theme_txt i zwracasz obiekt ThemeQuery.

=== WYMAGANA STRUKTURA JSON (ThemeQuery) ===

{
  "tags_summary_en": "string — maks. 14 słów po angielsku; streszczenie kluczowych tagów i targetów muzycznych; nie jest mechaniczną kopią should_have_tags",

  "scene_music_semantics": {
    "emotional_direction":    ["string", "..."],
    "narrative_function":     ["string", "..."],
    "weight_profile":         "string",
    "dialogue_safe_required": true
  },

  "scene_music_targets": {
    "energy_target":          "string",
    "tempo_target":           "string",
    "rhythm_target":          "string",
    "intensity_shape_target": "string",
    "sound_character_target": ["string", "..."]
  },

  "scene_tag_selection": {
    "should_have_tags": ["string", "..."],
    "must_not_tags":    ["string", "..."]
  },

  "clap_prompt_ensemble": {
    "semantic_scene_prompt":  "string — maks. 14 słów po angielsku",
    "music_for_scene_prompt": "string — maks. 14 słów po angielsku",
    "emotion_prompt":         "string — maks. 14 słów po angielsku",
    "narrative_prompt":       "string — maks. 14 słów po angielsku",
    "sonic_prompt":           "string — maks. 14 słów po angielsku",
    "tag_prompt":             "string — maks. 14 słów po angielsku",
    "concise_core_prompt":    "string — maks. 14 słów po angielsku"
  }
}

=== DOZWOLONE WARTOŚCI ===

emotional_direction (wybierz 2-4 elementów):
  tension, suspense, fear, anxiety, danger, sadness, grief, melancholy, loneliness, nostalgia,
  joy, happiness, hope, relief, love, romance, warmth, anger, aggression, determination,
  courage, triumph, epic, heroism, mystery, curiosity, wonder, awe, serenity, calm, peace, spirituality

narrative_function (wybierz 1-3 elementów):
  DEFINICJA: ROLA MUZYKI względem narracji (nie typ sceny — tym jest weight_profile).
  background, foreground, underscore, emotional_support, tension_building, suspense_building,
  anticipation, narrative_release, payoff, transition, bridge, accent, stinger, climax_support
  UWAGA: używaj "narrative_release" (nie "release") — aby uniknąć kolizji z intensity_shape::release.

weight_profile (dokładnie 1 wartość):
  DEFINICJA: TYP SCENY (nie rola muzyki — tym jest narrative_function).
  dialogue, conversation, investigation, action, dramatic_scene, horror_scene,
  thriller_scene, romantic_scene, comedic_scene, transition, montage, climax, resolution, aftermath

dialogue_safe_required:
  true  — muzyka NIE MOŻE zawierać wokalu, narracji ani żadnego głosu, który mógłby maskować dialogi postaci.
  false — muzyka może zawierać dowolne elementy wokalne.

energy_target         (1 wartość): very_low, low, medium, high, very_high
tempo_target          (1 wartość): very_slow, slow, moderate, fast, very_fast
rhythm_target         (1 wartość): steady, syncopated, irregular, driving, floating, free
intensity_shape_target (1 wartość):
  static, gradual_build, crescendo, peak, climax_peak, drop, release, wave, pulsating
sound_character_target (1-4 wartości):
  atmospheric, textured, dense, sparse, wide, intimate, distorted, clean, noisy, glitchy, analog, digital

=== DOZWOLONY SŁOWNIK TAGÓW (should_have_tags i must_not_tags) ===

Tagi w obu listach NIE MOGĄ się pokrywać.
should_have_tags: 4-8 elementów. must_not_tags: 2-6 elementów.
UWAGA: tag "release" pochodzi z kategorii intensity_shape (NIE z kategorii narrative).

emotion:         tension, suspense, fear, anxiety, danger, sadness, grief, melancholy, loneliness, nostalgia,
                 joy, happiness, hope, relief, love, romance, warmth, anger, aggression, determination,
                 courage, triumph, epic, heroism, mystery, curiosity, wonder, awe, serenity, calm, peace, spirituality
scene:           dialogue, conversation, transition, montage, investigation, action, climax, resolution, aftermath,
                 horror_scene, thriller_scene, romantic_scene, dramatic_scene, comedic_scene
energy:          very_low, low, medium, high, very_high
tempo:           very_slow, slow, moderate, fast, very_fast
rhythm:          steady, syncopated, irregular, driving, floating, free
intensity_shape: static, gradual_build, crescendo, peak, climax_peak, drop, release, wave, pulsating
narrative:       background, foreground, underscore, emotional_support, tension_building, suspense_building,
                 anticipation, narrative_release, payoff, transition, bridge, accent, stinger, climax_support
sound_char:      atmospheric, textured, dense, sparse, wide, intimate, distorted, clean, noisy, glitchy, analog, digital
usage:           background_music, film_score, trailer_music
instrument:      hits, impacts, pulses, drums, strings, piano, synth, guitar, choir, pads
atmosphere:      dark, eerie, warm, uplifting, emotional, serious, playful, intimate, expansive
special:         dialogue_safe, restrained, cinematic

=== PRZYKŁADY CLAP PROMPT ENSEMBLE ===

Poniżej 4 pełne przykłady oparte na rzeczywistych scenach serialu KLAN (odcinek 4695).
Traktuj je jako wzorzec kalibracji — nie jako szablony do kopiowania.
Każdy prompt musi być napisany od nowa, specyficznie dla theme_txt.

--- PRZYKŁAD 1 (KLAN 4695, scena 1) ---
Rodzina przy śniadaniu. Witold ogłasza, że potencjalny kupiec domu wycofał
się, bo odkrył plany budowy drogi. Anna próbuje zmienić temat, Iwona tłumi
złość, Witold spokojnie twierdzi, że tylko stwierdza fakty. Pod uprzejmą
rozmową tli się konflikt o pieniądze i dziedzictwo — żadnych ostrych słów,
same spojrzenia i zaciśnięte wargi.

clap_prompt_ensemble:
  semantic_scene_prompt:  "tense family breakfast concealing conflict over inheritance"
  music_for_scene_prompt: "soft restrained domestic underscore beneath polite argument"
  emotion_prompt:         "tension anxiety loneliness melancholy"
  narrative_prompt:       "background low underscore barely audible beneath dialogue"
  sonic_prompt:           "sparse intimate warm analog piano gentle strings"
  tag_prompt:             "tension dialogue_safe underscore restrained warm conversation"
  concise_core_prompt:    "quiet warm tension domestic family conflict underscore"

--- PRZYKŁAD 2 (KLAN 4695, scena 9) ---
Norbert wraca do mieszkania Sandry. Ona zawija naleśniki, wzruszona —
wczorajsza rozmowa ją poruszyła. Zgadza się ponownie wziąć ślub. Delikatny
pocałunek. Dzieci wbiegają i skandują "gorzko". Norbert się śmieje, ale na
blacie leży telefon ekranem w dół — to Roksi dzwoniła. Ciepło i ulga,
ale z cichym cieniem czegoś niedokończonego.

clap_prompt_ensemble:
  semantic_scene_prompt:  "tender kitchen reconciliation second chance at love"
  music_for_scene_prompt: "warm hopeful strings for quiet domestic reunion scene"
  emotion_prompt:         "love warmth hope relief nostalgia"
  narrative_prompt:       "emotional_support underscore swelling gently beneath reunion"
  sonic_prompt:           "warm intimate sparse strings piano analog"
  tag_prompt:             "warmth hope love relief strings piano romantic_scene underscore"
  concise_core_prompt:    "warm bittersweet hopeful reconciliation strings intimate"

--- PRZYKŁAD 3 (KLAN 4695, scena 10A) ---
Kornel i jego zespół siedzą w hipsterskiej kawiarni, omawiają strategię
po kompromitującym wywiadzie u influencerki. Kornel dostrzega tę samą
influencerkę — Ostrą — która robi live'a w tym samym lokalu i drwi z
artystów. Wstaje, celowo potyka się, oblewa ją napojem, obejmuje ramieniem
i nawinuje do jej własnej kamery, że rock nigdy się nie zestarzeeje. Ostra
krzyżuje miny. Komedia z ostrym zębem — żenada, prowokacja i rozbawiony
tłum w tle.

clap_prompt_ensemble:
  semantic_scene_prompt:  "awkward comic confrontation between rock star and influencer"
  music_for_scene_prompt: "quirky playful score for social media prank gone public"
  emotion_prompt:         "anger determination curiosity awe"
  narrative_prompt:       "foreground comedic accent punctuating social collision"
  sonic_prompt:           "clean playful digital sparse quirky"
  tag_prompt:             "playful comedic_scene curiosity determination cinematic digital"
  concise_core_prompt:    "playful quirky comedic confrontation awkward viral moment"

--- PRZYKŁAD 4 (KLAN 4695, scena 7) ---
Sala prób. Roksi i tancerki kończą choreografię flamenco efektowną pozą.
Norbert stał w drzwiach i obserwował — przyszedł tylko oddać portfel
Miłoszowi. Kiedy Roksi kończy i chce do niego podejść, on już wychodzi.
Patrzy za siebie raz — nostalgiczny uśmiech — i znika. Muzyka flamenco
i klaskanie staccato podkreślają to, czego nie da się powiedzieć głośno.

clap_prompt_ensemble:
  semantic_scene_prompt:  "flamenco rehearsal melancholic farewell glance at the door"
  music_for_scene_prompt: "bittersweet flamenco underscore for unspoken longing"
  emotion_prompt:         "melancholy loneliness nostalgia sadness"
  narrative_prompt:       "emotional_support foreground flamenco rhythm carrying grief"
  sonic_prompt:           "intimate acoustic analog guitar percussion sparse warm"
  tag_prompt:             "melancholy loneliness nostalgia guitar dramatic_scene underscore"
  concise_core_prompt:    "melancholic acoustic flamenco longing unspoken farewell"

=== ZASADY OGÓLNE ===

1. Traktuj theme_txt jako jeden spójny temat — nie dziel go dalej.
2. Nie używaj wartości spoza dozwolonych słowników.
3. Myśl w kategoriach dopasowania muzyki do sceny, nie streszczenia fabuły.
4. Każdy prompt CLAP: maks. 14 słów, po angielsku, bez przepisywania tekstu sceny.
5. Zwróć WYŁĄCZNIE obiekt JSON zgodny z ThemeQuery — bez żadnych dodatkowych pól.
"""

# LLM2-C1: user prompt is minimal.
LLM2_USER_PROMPT_TEMPLATE = "theme_txt:\n{theme_txt}"


# ============================================================
# LLM CALLING
# ============================================================

def call_llm(
    system_prompt:  str,
    user_prompt:    str,
    response_model: type[BaseModel],
) -> BaseModel:
    """
    Call OpenAI (or Anthropic) with structured output.

    Caching notes
    ─────────────
    OpenAI (gpt-4.1):
      Automatic prefix caching applies when the prompt prefix ≥ 1 024 tokens.
      No markup needed. The module-level client singleton (CACHE-2) keeps the
      TCP connection alive for stable prefix hashing across sequential scenes.

    Anthropic Messages API (Claude models, USE_ANTHROPIC_CACHE=1):
      Explicit cache_control breakpoints are required.  _call_anthropic_cached()
      places them at the end of each static content block (system prompt and
      schema).  Both LLM1 (~700 tokens) and LLM2 (~1 400+ tokens) exceed the
      1 024-token minimum for all current Claude Sonnet/Opus models.
      Default TTL: 5 minutes (suitable for a sequential episode run).
      Set ANTHROPIC_CACHE_TTL_1H=1 for 1-hour TTL on long batch jobs.

    OpenAI strategy order:
      1. Responses API structured parse  (openai >= 1.75)
      2. Chat Completions beta parse
      3. Chat Completions JSON mode + manual Pydantic validation
    """
    if USE_ANTHROPIC_CACHE:
        return _call_anthropic_cached(system_prompt, user_prompt, response_model)

    client = _get_openai_client()

    # v3.01: _MAX_TOKENS raised from 1500 to 4096.
    # LLM #1 must echo verbatim scene text inside theme_txt — a long Polish
    # soap opera scene can exceed 1000 tokens of text alone, leaving no room
    # for the JSON wrapper under a 1500-token cap.  4096 comfortably fits the
    # largest scenes while keeping latency reasonable.  _MAX_TOKENS_RETRY is
    # used for a single automatic retry when the first response is truncated
    # (finish_reason == "length").
    _MAX_TOKENS       = 4096
    _MAX_TOKENS_RETRY = 8192
    _TIMEOUT          = 60   # seconds per API call (raised from 30 for long scenes)

    def _s1_attempt(max_tok: int) -> Optional[BaseModel]:
        """Strategy 1 — Responses API structured parse (openai >= 1.75)."""
        try:
            if hasattr(client, "responses") and hasattr(client.responses, "parse"):
                resp = client.responses.parse(
                    model=MODEL,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    text_format=response_model,
                    max_output_tokens=max_tok,
                    timeout=_TIMEOUT,
                )
                parsed = getattr(resp, "output_parsed", None)
                if parsed is not None:
                    return parsed
        except Exception:
            pass
        return None

    def _s2_attempt(max_tok: int) -> Optional[BaseModel]:
        """Strategy 2 — Chat Completions beta parse."""
        try:
            resp = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format=response_model,
                max_tokens=max_tok,
                timeout=_TIMEOUT,
            )
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise RuntimeError(
                    f"Strategy 2 truncated at {max_tok} tokens (finish_reason=length). "
                    "The scene text is too long for this token budget."
                )
            parsed = choice.message.parsed
            if parsed is not None:
                return parsed
        except RuntimeError:
            raise
        except Exception:
            pass
        return None

    def _s3_attempt(max_tok: int) -> BaseModel:
        """Strategy 3 — Chat Completions JSON mode + manual validation."""
        schema      = response_model.model_json_schema()
        schema_json = json.dumps(schema, ensure_ascii=False)
        fallback_sys = (
            system_prompt
            + "\n\nReturn ONLY valid JSON matching this JSON Schema exactly:\n"
            + schema_json
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": fallback_sys},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tok,
            timeout=_TIMEOUT,
        )
        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish == "length":
            raise RuntimeError(
                f"Strategy 3 truncated at {max_tok} tokens (finish_reason=length). "
                "The scene text is too long for this token budget."
            )
        output_text = choice.message.content
        if not output_text:
            raise RuntimeError("Strategy 3 returned empty response text.")
        try:
            return response_model.model_validate_json(output_text)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to validate LLM response as {response_model.__name__}: {exc}"
            ) from exc

    # ── Run strategies, with one automatic retry at double tokens on truncation ──
    for max_tok in (_MAX_TOKENS, _MAX_TOKENS_RETRY):
        result = _s1_attempt(max_tok)
        if result is not None:
            return result

        result = _s2_attempt(max_tok)
        if result is not None:
            return result

        try:
            return _s3_attempt(max_tok)
        except RuntimeError as exc:
            if "finish_reason=length" in str(exc) and max_tok == _MAX_TOKENS:
                logger.warning(
                    "LLM response truncated at %d tokens — retrying with %d tokens.",
                    max_tok, _MAX_TOKENS_RETRY,
                )
                continue   # retry loop with _MAX_TOKENS_RETRY
            raise         # non-truncation error or second attempt: propagate

    raise RuntimeError(
        f"All LLM strategies failed for {response_model.__name__} "
        f"even at {_MAX_TOKENS_RETRY} tokens."
    )


def _call_anthropic_cached(
    system_prompt:  str,
    user_prompt:    str,
    response_model: type[BaseModel],
) -> BaseModel:
    """
    CACHE-1 — Anthropic Messages API with explicit cache_control breakpoints.

    The system content is split into two blocks so the cache breakpoint covers
    both the instruction text and the embedded JSON schema:

      block 0: system instruction text  → cache_control: ephemeral
      block 1: JSON schema reminder      → cache_control: ephemeral

    The user message carries only the variable scene/theme text (no caching).

    Cache hit monitoring: cache_creation_input_tokens and cache_read_input_tokens
    are logged at DEBUG level for each call.
    """
    use_1h = os.getenv("ANTHROPIC_CACHE_TTL_1H", "0").lower() in ("1", "true", "yes")
    cache_control: dict = {"type": "ephemeral"}
    if use_1h:
        cache_control["ttl"] = "1h"

    extra_headers: dict = {}
    if use_1h:
        extra_headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"

    schema_json = json.dumps(
        response_model.model_json_schema(), ensure_ascii=False, indent=2
    )

    anth_client = _get_anthropic_client()   # ← singleton, not a fresh instance

    system_blocks = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": cache_control,
        },
        {
            "type": "text",
            "text": (
                "\n\nJSON Schema for your response (follow it exactly):\n"
                + schema_json
            ),
            "cache_control": cache_control,
        },
    ]

    response = anth_client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_blocks,
        messages=[{"role": "user", "content": user_prompt}],
        extra_headers=extra_headers or None,
    )

    usage = getattr(response, "usage", None)
    if usage:
        logger.debug(
            "Anthropic cache — write: %d tokens, read: %d tokens",
            getattr(usage, "cache_creation_input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
        )

    output_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    if not output_text:
        raise RuntimeError("Anthropic LLM returned empty response.")

    # Strip markdown fences the model may add around JSON
    output_text = re.sub(r"^```(?:json)?\s*", "", output_text.strip())
    output_text = re.sub(r"\s*```$", "", output_text)

    try:
        return response_model.model_validate_json(output_text)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to validate Anthropic response as {response_model.__name__}: {exc}"
        ) from exc


# ============================================================
# LLM WRAPPERS
# ============================================================

def LLM_1_theme_list(scene_txt: str) -> ThemeList:
    if not isinstance(scene_txt, str):
        raise TypeError("scene_txt must be a string.")
    scene_txt = scene_txt.strip()
    if not scene_txt:
        raise ValueError("scene_txt cannot be empty.")
    result = call_llm(
        system_prompt=LLM1_SYSTEM_PROMPT,
        user_prompt=LLM1_USER_PROMPT_TEMPLATE.format(scene_txt=scene_txt),
        response_model=ThemeList,
    )
    if not isinstance(result, ThemeList):
        result = ThemeList.model_validate(result)
    # LLM1-C3: empty themes is the model's error signal for unreadable input
    if not result.themes:
        raise ValueError(
            f"LLM_1_theme_list: empty themes. "
            f"Reason: {result.segmentation_reason!r}"
        )
    return result


def LLM_2_theme_query(theme_txt: str) -> ThemeQuery:
    if not isinstance(theme_txt, str):
        raise TypeError("theme_txt must be a string.")
    theme_txt = theme_txt.strip()
    if not theme_txt:
        raise ValueError("theme_txt cannot be empty.")
    result = call_llm(
        system_prompt=LLM2_SYSTEM_PROMPT,
        user_prompt=LLM2_USER_PROMPT_TEMPLATE.format(theme_txt=theme_txt),
        response_model=ThemeQuery,
    )
    if not isinstance(result, ThemeQuery):
        result = ThemeQuery.model_validate(result)
    if not result.tags_summary_en.strip():
        raise ValueError("LLM_2_theme_query returned empty tags_summary_en.")
    should_have = set(result.scene_tag_selection.should_have_tags)
    must_not    = set(result.scene_tag_selection.must_not_tags)
    overlap = should_have.intersection(must_not)
    if overlap:
        raise ValueError(
            f"LLM_2_theme_query: overlapping tags: {sorted(overlap)}"
        )
    return result


def make_theme_query_record(
    episode_nr:          int,
    scene_nr:            int,
    theme_nr:            int,
    segmentation_reason: str,
    theme_base:          ThemeBase,
    theme_query:         ThemeQuery,
) -> SceneQueryRecord:
    if not isinstance(episode_nr, int) or episode_nr <= 0:
        raise ValueError("episode_nr must be a positive integer.")
    if not isinstance(scene_nr, int) or scene_nr <= 0:
        raise ValueError("scene_nr must be a positive integer.")
    if not isinstance(theme_nr, int) or theme_nr <= 0:
        raise ValueError("theme_nr must be a positive integer.")
    segmentation_reason = str(segmentation_reason).strip()
    if not segmentation_reason:
        raise ValueError("segmentation_reason cannot be empty.")
    if not isinstance(theme_base, ThemeBase):
        theme_base = ThemeBase.model_validate(theme_base)
    if not isinstance(theme_query, ThemeQuery):
        theme_query = ThemeQuery.model_validate(theme_query)
    should_have = set(theme_query.scene_tag_selection.should_have_tags)
    must_not    = set(theme_query.scene_tag_selection.must_not_tags)
    overlap = should_have.intersection(must_not)
    if overlap:
        raise ValueError(f"Overlapping tags: {sorted(overlap)}")
    return SceneQueryRecord(
        episode_nr=episode_nr,
        scene_nr=scene_nr,
        theme_nr=theme_nr,
        segmentation_reason=segmentation_reason,
        theme_title_pl=theme_base.theme_title_pl,
        theme_title_en=theme_base.theme_title_en,
        description_en=theme_base.description_en,
        theme_txt=theme_base.theme_txt,
        tags_summary_en=theme_query.tags_summary_en,
        scene_music_semantics=theme_query.scene_music_semantics,
        scene_music_targets=theme_query.scene_music_targets,
        scene_tag_selection=theme_query.scene_tag_selection,
        clap_prompt_ensemble=theme_query.clap_prompt_ensemble,
    )


# ============================================================
# CLAP MODEL (thread-safe singleton)
# ============================================================

def get_clap_model() -> tuple[ClapProcessor, ClapModel]:
    global _CLAP_PROCESSOR, _CLAP_MODEL_INSTANCE
    with _CLAP_LOCK:
        if _CLAP_PROCESSOR is None or _CLAP_MODEL_INSTANCE is None:
            logger.info("Loading CLAP model: %s onto %s", CLAP_MODEL, _CLAP_DEVICE)
            _CLAP_PROCESSOR     = ClapProcessor.from_pretrained(CLAP_MODEL)
            _CLAP_MODEL_INSTANCE = ClapModel.from_pretrained(CLAP_MODEL).to(_CLAP_DEVICE)
            _CLAP_MODEL_INSTANCE.eval()
        return _CLAP_PROCESSOR, _CLAP_MODEL_INSTANCE


def release_clap_model() -> None:
    global _CLAP_PROCESSOR, _CLAP_MODEL_INSTANCE
    with _CLAP_LOCK:
        if _CLAP_MODEL_INSTANCE is not None:
            del _CLAP_MODEL_INSTANCE
            _CLAP_MODEL_INSTANCE = None
        if _CLAP_PROCESSOR is not None:
            del _CLAP_PROCESSOR
            _CLAP_PROCESSOR = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("CLAP model released from memory.")


# ============================================================
# CLAP ENCODING
# ============================================================

def _tensor_from_clap_output(output: Any) -> "torch.Tensor":
    """
    Safely extract a raw tensor from whatever get_text_features() returns.

    Depending on the transformers version / CLAP variant the call may return:
      - a plain torch.Tensor                       (most common, contrastive head)
      - BaseModelOutputWithPooling / similar        (base encoder fallback)
          .text_embeds   – contrastive projection, may be None on base-only builds
          .pooler_output – [CLS] pooled hidden state, always present
          .last_hidden_state – full sequence, needs mean-pooling
      - a tuple whose first element is a tensor
    """
    if torch.is_tensor(output):
        return output

    # HuggingFace dataclass outputs — try each attribute in preference order
    for attr in ("text_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if val is not None and torch.is_tensor(val):
            return val

    if hasattr(output, "last_hidden_state"):
        lhs = output.last_hidden_state
        if lhs is not None and torch.is_tensor(lhs):
            # mean-pool over the sequence dimension
            return lhs.mean(dim=1)

    # Tuple / list fallback (some older transformers versions)
    if isinstance(output, (tuple, list)) and len(output) > 0:
        first = output[0]
        if torch.is_tensor(first):
            return first

    raise TypeError(
        f"Cannot extract a tensor from CLAP output of type {type(output).__name__}. "
        f"Attributes: {[a for a in dir(output) if not a.startswith('_')]}"
    )


def _extract_text_features_to_numpy(output: Any) -> np.ndarray:
    """Convert any CLAP get_text_features() output to a 1-D float32 numpy array."""
    tensor = _tensor_from_clap_output(output)
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[0] if arr.shape[0] == 1 else arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Unexpected embedding shape after squeeze: {arr.shape}")
    return arr


def encode_texts_batch(texts: list[str]) -> list[np.ndarray]:
    """Encode a list of texts in a single CLAP forward pass.

    Handles both the contrastive-projection output (plain tensor) and the
    BaseModelOutputWithPooling fallback returned by some transformers builds.
    Returns L2-normalised float32 numpy vectors of length VECTOR_DIM.
    """
    if not texts:
        return []
    processor, model = get_clap_model()
    # Cast through Any: Pyrefly stubs for ProcessorMixin are incomplete and
    # incorrectly reject return_tensors / padding at analysis time.
    # Runtime behaviour is identical — ClapProcessor accepts these kwargs.
    _proc: Any = processor
    inputs = _proc(text=texts, return_tensors="pt", padding=True).to(_CLAP_DEVICE)
    with torch.no_grad():
        raw = model.get_text_features(**inputs)

    # raw may be a tensor or a HuggingFace output dataclass — normalise first
    tensor = _tensor_from_clap_output(raw)
    arr = tensor.detach().cpu().numpy().astype(np.float32)  # (N, dim)

    results: list[np.ndarray] = []
    for row in arr:
        norm = float(np.linalg.norm(row))
        if norm == 0.0:
            raise ValueError("Zero-norm CLAP embedding encountered.")
        results.append(row / norm)
    return results


def encode_text(text: str) -> list[float]:
    text = str(text).strip()
    if not text:
        raise ValueError("Text for encoding cannot be empty.")
    vecs = encode_texts_batch([text])
    vec  = vecs[0]
    if vec.shape[0] != VECTOR_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: got {vec.shape[0]}, expected {VECTOR_DIM}."
        )
    return vec.tolist()


# ============================================================
# EMBEDDINGS
# ============================================================

_ENSEMBLE_PROMPT_FIELDS = [
    ("semantic_scene_prompt",  0.25),
    ("music_for_scene_prompt", 0.20),
    ("emotion_prompt",         0.15),
    ("narrative_prompt",       0.15),
    ("sonic_prompt",           0.10),
    ("tag_prompt",             0.10),
    ("concise_core_prompt",    0.05),
]


def create_embeddings(scene_query_record: SceneQueryRecord) -> SceneEmbeddings:
    if not isinstance(scene_query_record, SceneQueryRecord):
        scene_query_record = SceneQueryRecord.model_validate(scene_query_record)

    title_en       = scene_query_record.theme_title_en.strip()
    description_en = scene_query_record.description_en.strip()
    tags_summary   = scene_query_record.tags_summary_en.strip()

    if not title_en:       raise ValueError("theme_title_en cannot be empty.")
    if not description_en: raise ValueError("description_en cannot be empty.")
    if not tags_summary:   raise ValueError("tags_summary_en cannot be empty.")

    main_text        = f"{title_en}. {description_en}"
    ensemble_obj     = scene_query_record.clap_prompt_ensemble
    ensemble_texts   = [getattr(ensemble_obj, f).strip() for f, _ in _ENSEMBLE_PROMPT_FIELDS]
    ensemble_weights = np.asarray([w for _, w in _ENSEMBLE_PROMPT_FIELDS], dtype=np.float32)

    all_vecs = encode_texts_batch([main_text, tags_summary] + ensemble_texts)

    for v in all_vecs:
        if v.shape[0] != VECTOR_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: got {v.shape[0]}, expected {VECTOR_DIM}."
            )

    embedding_main          = all_vecs[0]
    embedding_tags          = all_vecs[1]
    ensemble_stack          = np.stack(all_vecs[2:], axis=0)
    ensemble_raw            = (ensemble_stack * ensemble_weights[:, None]).sum(axis=0)
    ensemble_norm           = float(np.linalg.norm(ensemble_raw))
    if ensemble_norm == 0.0:
        raise ValueError("Zero-norm CLAP ensemble embedding.")
    embedding_clap_ensemble = ensemble_raw / ensemble_norm

    hybrid_raw  = (0.45 * embedding_main
                   + 0.20 * embedding_tags
                   + 0.35 * embedding_clap_ensemble)
    hybrid_norm = float(np.linalg.norm(hybrid_raw))
    if hybrid_norm == 0.0:
        raise ValueError("Zero-norm hybrid embedding.")
    embedding_hybrid = hybrid_raw / hybrid_norm

    return SceneEmbeddings(
        embedding_main=embedding_main.tolist(),
        embedding_tags=embedding_tags.tolist(),
        embedding_clap_ensemble=embedding_clap_ensemble.tolist(),
        embedding_hybrid=embedding_hybrid.tolist(),
    )


# ============================================================
# DB WRITE
# ============================================================

def insert_scene_query_row(
    cur,
    record:     SceneQueryRecord,
    embeddings: SceneEmbeddings,
) -> int:
    cur.execute(
        """
        INSERT INTO scene_query (
            episode_nr, scene_nr, theme_nr, segmentation_reason,
            theme_title_pl, theme_title_en, description_en, theme_txt,
            tags_summary_en,
            scene_music_semantics, scene_music_targets,
            scene_tag_selection, clap_prompt_ensemble,
            embedding_main, embedding_tags,
            embedding_clap_ensemble, embedding_hybrid
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s
        )
        ON CONFLICT (episode_nr, scene_nr, theme_nr) DO UPDATE SET
            segmentation_reason     = EXCLUDED.segmentation_reason,
            theme_title_pl          = EXCLUDED.theme_title_pl,
            theme_title_en          = EXCLUDED.theme_title_en,
            description_en          = EXCLUDED.description_en,
            theme_txt               = EXCLUDED.theme_txt,
            tags_summary_en         = EXCLUDED.tags_summary_en,
            scene_music_semantics   = EXCLUDED.scene_music_semantics,
            scene_music_targets     = EXCLUDED.scene_music_targets,
            scene_tag_selection     = EXCLUDED.scene_tag_selection,
            clap_prompt_ensemble    = EXCLUDED.clap_prompt_ensemble,
            embedding_main          = EXCLUDED.embedding_main,
            embedding_tags          = EXCLUDED.embedding_tags,
            embedding_clap_ensemble = EXCLUDED.embedding_clap_ensemble,
            embedding_hybrid        = EXCLUDED.embedding_hybrid,
            updated_at              = NOW()
        RETURNING id
        """,
        (
            record.episode_nr, record.scene_nr, record.theme_nr,
            record.segmentation_reason,
            record.theme_title_pl, record.theme_title_en,
            record.description_en, record.theme_txt,
            record.tags_summary_en,
            Json(record.scene_music_semantics.model_dump()),
            Json(record.scene_music_targets.model_dump()),
            Json(record.scene_tag_selection.model_dump()),
            Json(record.clap_prompt_ensemble.model_dump()),
            embeddings.embedding_main,
            embeddings.embedding_tags,
            embeddings.embedding_clap_ensemble,
            embeddings.embedding_hybrid,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("INSERT INTO scene_query returned no id.")
    return int(row[0])


def save_theme_query_record(scene_query_record: SceneQueryRecord) -> int:
    if not isinstance(scene_query_record, SceneQueryRecord):
        scene_query_record = SceneQueryRecord.model_validate(scene_query_record)
    embeddings = create_embeddings(scene_query_record)
    conn = get_connection()
    cur  = conn.cursor()
    try:
        row_id = insert_scene_query_row(cur, scene_query_record, embeddings)
        conn.commit()
        return row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ============================================================
# BATCH ORCHESTRATION  —  three explicit stages
# ============================================================
# Stage 1: read_and_split_screenplay()  →  (episode_nr, scenes)
# Stage 2: run_llm1_all_scenes()        →  list[ThemeList]
# Stage 3: run_llm2_embed_save()        →  list[SceneQueryRecord]
#
# This separation lets the UI show results and ask for confirmation
# between each stage so no LLM money is spent without user approval.
# ============================================================

def run_llm1_all_scenes(
    scenes:            list[str],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[ThemeList]:
    """
    Stage 2: run LLM #1 (semantic segmentation) for every scene.
    Returns a parallel list of ThemeList objects, one per scene.
    No DB writes, no embeddings, no LLM #2 calls.
    """
    if not scenes:
        raise ValueError("scenes list is empty.")
    total   = len(scenes)
    results: list[ThemeList] = []

    for idx, scene_txt in enumerate(scenes, start=1):
        if progress_callback:
            progress_callback(f"LLM #1 — scene {idx}/{total}: segmentation…", idx - 1, total)

        theme_list = LLM_1_theme_list(scene_txt)
        if not isinstance(theme_list, ThemeList):
            theme_list = ThemeList.model_validate(theme_list)
        if not theme_list.themes:
            raise ValueError(f"LLM #1 returned empty themes for scene {idx}.")
        results.append(theme_list)

    if progress_callback:
        progress_callback(f"LLM #1 complete — {total} scenes segmented.", total, total)
    return results


def run_llm2_embed_save(
    episode_nr:        int,
    scenes:            list[str],
    theme_lists:       list[ThemeList],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[SceneQueryRecord]:
    """
    Stage 3: for every theme from Stage 2, run LLM #2, create embeddings,
    and write to DB. Each row is committed individually so a mid-run failure
    preserves already-saved rows and the DB is never left in an empty state.
    Existing rows for the episode are deleted once, before the first write.
    """
    if not isinstance(episode_nr, int) or episode_nr <= 0:
        raise ValueError("episode_nr must be a positive integer.")
    if len(scenes) != len(theme_lists):
        raise ValueError("scenes and theme_lists must have the same length.")

    total_themes = sum(len(tl.themes) for tl in theme_lists)
    theme_counter = 0

    # Delete old rows once, with its own short-lived connection
    delete_episode_scene_queries(episode_nr)

    saved: list[SceneQueryRecord] = []

    for scene_idx, (scene_txt, theme_list) in enumerate(zip(scenes, theme_lists), start=1):
        segmentation_reason = theme_list.segmentation_reason.strip()

        for theme_idx, theme_base in enumerate(theme_list.themes, start=1):
            theme_counter += 1
            if progress_callback:
                progress_callback(
                    f"Scene {scene_idx}, theme {theme_idx}/{len(theme_list.themes)}: "
                    f"LLM #2 + embeddings + save…",
                    theme_counter - 1,
                    total_themes,
                )

            if not isinstance(theme_base, ThemeBase):
                theme_base = ThemeBase.model_validate(theme_base)

            theme_query = LLM_2_theme_query(theme_base.theme_txt)
            record = make_theme_query_record(
                episode_nr=episode_nr,
                scene_nr=scene_idx,
                theme_nr=theme_idx,
                segmentation_reason=segmentation_reason,
                theme_base=theme_base,
                theme_query=theme_query,
            )

            embeddings = create_embeddings(record)

            # Commit each row individually — partial progress is preserved on failure
            conn = get_connection()
            cur  = conn.cursor()
            try:
                insert_scene_query_row(cur, record, embeddings)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
                conn.close()

            saved.append(record)

    if progress_callback:
        progress_callback(
            f"Completed — {len(saved)} theme records saved for episode {episode_nr}.",
            total_themes,
            total_themes,
        )
    return saved


def make_scenes_query(
    episode_nr:        int,
    scenes:            list[str],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[SceneQueryRecord]:
    """
    Legacy single-call entry point kept for backward compatibility
    (used by save_theme_query_record standalone wrapper and tests).
    Runs all three stages sequentially without a confirmation gate.
    """
    import warnings
    warnings.warn(
        "make_scenes_query() is deprecated. "
        "Prefer calling run_llm1_all_scenes() and run_llm2_embed_save() separately.",
        DeprecationWarning,
        stacklevel=2,
    )
    theme_lists = run_llm1_all_scenes(scenes, progress_callback=progress_callback)
    return run_llm2_embed_save(
        episode_nr, scenes, theme_lists, progress_callback=progress_callback
    )


# ============================================================
# DB READ
# ============================================================

def fetch_scene_query_records(episode_nr: int) -> list[SceneQueryRecord]:
    if not isinstance(episode_nr, int) or episode_nr <= 0:
        raise ValueError("episode_nr must be a positive integer.")
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            """
            SELECT episode_nr, scene_nr, theme_nr, segmentation_reason,
                   theme_title_pl, theme_title_en, description_en, theme_txt,
                   tags_summary_en, scene_music_semantics, scene_music_targets,
                   scene_tag_selection, clap_prompt_ensemble
            FROM scene_query
            WHERE episode_nr = %s
            ORDER BY scene_nr, theme_nr
            """,
            (episode_nr,),
        )
        rows    = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [
            SceneQueryRecord.model_validate(dict(zip(columns, row)))
            for row in rows
        ]
    finally:
        cur.close()
        conn.close()


# ============================================================
# PDF GENERATION
# ============================================================

def register_pdf_fonts() -> tuple[str, str]:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ),
    ]
    win = os.environ.get("WINDIR", r"C:\Windows")
    candidates += [
        (str(Path(win) / "Fonts" / "arial.ttf"),   str(Path(win) / "Fonts" / "arialbd.ttf")),
        (str(Path(win) / "Fonts" / "calibri.ttf"), str(Path(win) / "Fonts" / "calibrib.ttf")),
    ]
    for regular_path, bold_path in candidates:
        if Path(regular_path).exists() and Path(bold_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("SceneQueryFont",     regular_path))
                pdfmetrics.registerFont(TTFont("SceneQueryFontBold", bold_path))
                return "SceneQueryFont", "SceneQueryFontBold"
            except Exception:
                continue
    raise RuntimeError(
        "No Unicode-capable TrueType font found. "
        "Install: apt install fonts-dejavu-core"
    )


def _pdf_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_jsonb_for_pdf(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict):
        lines = []
        for key, subvalue in value.items():
            rendered = (", ".join(str(x) for x in subvalue)
                        if isinstance(subvalue, list) else str(subvalue))
            lines.append(f"{key}: {rendered}")
        return "<br/>".join(_pdf_escape(line) for line in lines)
    if isinstance(value, list):
        return "<br/>".join(_pdf_escape(str(x)) for x in value)
    return _pdf_escape(str(value)).replace("\n", "<br/>")


def build_pdf_story(records: list[SceneQueryRecord], episode_nr: int) -> list[Any]:
    font_name, font_bold = register_pdf_fonts()
    styles = getSampleStyleSheet()
    styles["Title"].fontName    = font_bold
    styles["Title"].fontSize    = 18
    styles["Title"].leading     = 22
    styles["BodyText"].fontName = font_name
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading  = 11

    if "FieldLabel" not in styles.byName:
        styles.add(ParagraphStyle(
            name="FieldLabel", parent=styles["BodyText"],
            fontName=font_bold, textColor=colors.HexColor("#294866"), alignment=TA_LEFT,
        ))
    if "FieldValue" not in styles.byName:
        styles.add(ParagraphStyle(
            name="FieldValue", parent=styles["BodyText"],
            fontName=font_name, alignment=TA_LEFT,
        ))

    _TS = TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND",    (0, 0), (0,  0),  colors.HexColor("#eaf1f7")),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d4e2")),
        ("LINEBEFORE",    (1, 0), (1,  0),  0.5, colors.HexColor("#c7d4e2")),
    ])

    def _row(label: str, value: Any) -> Table:
        t = Table(
            [[Paragraph(_pdf_escape(label), styles["FieldLabel"]),
              Paragraph(format_jsonb_for_pdf(value), styles["FieldValue"])]],
            colWidths=[45 * mm, 125 * mm], hAlign="LEFT",
        )
        t.setStyle(_TS)
        return t

    long_fields = {"theme_txt", "description_en", "segmentation_reason", "tags_summary_en"}

    def add_field(story: list[Any], label: str, value: Any) -> None:
        if label in long_fields:
            h = Table(
                [[Paragraph(_pdf_escape(label), styles["FieldLabel"]),
                  Paragraph("", styles["FieldValue"])]],
                colWidths=[45 * mm, 125 * mm], hAlign="LEFT",
            )
            h.setStyle(_TS)
            story.extend([h, Spacer(1, 2)])
            for chunk in format_jsonb_for_pdf(value).split("<br/>"):
                chunk = chunk.strip()
                if chunk:
                    story.append(Paragraph(chunk, styles["FieldValue"]))
            story.append(Spacer(1, 4))
        else:
            story.extend([_row(label, value), Spacer(1, 3)])

    ordered = [
        "episode_nr", "scene_nr", "theme_nr", "segmentation_reason",
        "theme_title_pl", "theme_title_en", "description_en", "theme_txt",
        "tags_summary_en", "scene_music_semantics", "scene_music_targets",
        "scene_tag_selection", "clap_prompt_ensemble",
    ]

    story: list[Any] = [
        Paragraph(f"Scene Queries — Episode {episode_nr}", styles["Title"]),
        Spacer(1, 6),
        Paragraph(
            f"Control export from scene_query. Total themes: {len(records)}",
            styles["BodyText"],
        ),
        Spacer(1, 10),
    ]

    for idx, record in enumerate(records, start=1):
        story.append(Paragraph(
            _pdf_escape(
                f"SceneQuery #{idx}  |  scene_nr={record.scene_nr}"
                f"  |  theme_nr={record.theme_nr}"
            ),
            styles["FieldLabel"],
        ))
        story.append(Spacer(1, 4))
        rd = record.model_dump()
        for field_name in ordered:
            add_field(story, field_name, rd.get(field_name))
        story.append(Spacer(1, 8))

    return story


def create_pdf_scenes(episode_nr: int) -> tuple[str, bytes]:
    if not isinstance(episode_nr, int) or episode_nr <= 0:
        raise ValueError("episode_nr must be a positive integer.")
    records = fetch_scene_query_records(episode_nr)
    if not records:
        raise ValueError(f"No scene_query rows found for episode_nr={episode_nr}.")
    story  = build_pdf_story(records, episode_nr)
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=14 * mm,
        title=f"sceneQueries_{episode_nr}",
    )
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return f"sceneQueries_{episode_nr}.pdf", pdf_bytes
