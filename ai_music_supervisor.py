#!/usr/bin/env python3
from __future__ import annotations

# ============================================================
# VERSION
# ============================================================
SCRIPT_NAME      = "ai_music_supervisor.py"
SCRIPT_VERSION   = "4.08"
PIPELINE_VERSION = f"{SCRIPT_NAME[:-3]}_v{SCRIPT_VERSION}"

# ── Change log ──────────────────────────────────────────────
# v4.08 (2026-05-13)
#   • Config page — changed display all weights
#     Screenplay Parser page — added instruction text to PDF export section.
# v4.07 (2026-05-13)
#   • _start_import(): _worker now reads stats directly from
#     ImportResult.to_stats_dict() instead of splitting the
#     "Final | inserted: N | …" log string.  duration_s and
#     skipped are now surfaced in session state as well.
# v4.06 (2026-05-13)
#   • ISSUE-02: Removed old-style typing generics.
# v4.05 (2026-05-12)
#   • BUG FIX (P0): Removed duplicate module-level helper block
#     that had been pasted inside the Importer page section
#     (previously lines ~2994–3059).  Five functions were
#     defined twice: _safe, _badge, _metrics, _pipe, _header.
#     The second _score_bar definition (2-arg signature) silently
#     overrode the primary 4-arg version used by
#     _render_score_breakdown(), causing a TypeError at runtime
#     on any score-breakdown render. Removed duplicates; the
#     canonical definitions at lines ~648–741 are now the only
#     copy. _render_log() (Importer-only helper) moved up into
#     the shared SHARED UI HELPERS section where it belongs.
# v4.03 (2026-05-12)
#   • BUG FIX: All pages — background colour remained Streamlit
#     default white/grey instead of the amber warm off-white
#     (#fdf6ee) used by music_library_importer.py.
#     Root cause: the base rule setting background:var(--bg) was
#     scoped to body[data-theme="blue"] in ai_music_supervisor,
#     whereas in music_library_importer it is an unscoped rule
#     on html, body, [data-testid="stApp"]. The scoped rule
#     only applies after _set_theme("blue") fires its JS, and
#     never applies to pages that haven't called _set_theme.
#     Fix: added an unscoped base rule (identical to
#     music_library_importer) immediately after the :root block,
#     setting background:var(--bg), color:var(--txt), and
#     font-family:var(--sans) on html/body/stApp for all pages.
#     The parser theme overrides this with its own green
#     background via the existing body[data-theme="parser"] rule.
# v4.02 (2026-05-12)
#   • BUG FIX: Run Importer page — pipeline never exited
#     "Pipeline running…" state after completion.
#     Root cause: _sync_import() was present in
#     music_library_importer.py v3.36 and called at the top of
#     its main() on every rerun, but was not ported into
#     ai_music_supervisor.py when the importer pages were moved.
#     Without it, the background worker thread updated the shared
#     _imp_result dict (setting done=True, running=False), but
#     nothing ever read that dict back into session state, so
#     import_running stayed True and import_done stayed False
#     permanently — keeping the progress bar spinning.
#     Fix: added _sync_import() function and call it as the
#     first statement in main() on every rerun, matching the
#     pattern in music_library_importer.py v3.36.
# v4.01 (2026-05-12)
#   • CONFLICT FIX: removed stale commented-out import line
#     (from music_library_importer import _page_config).
#   • CONFLICT FIX: get_connection — two engines exported
#     functions with the same name but different signatures.
#     scene_music_matcher_engine.get_connection(db_config) is
#     kept as get_connection (used by all matcher/scoring/pdf
#     pages which call get_connection(DB_CONFIG)).
#     importer_pipeline.get_connection() is aliased to
#     _imp_get_connection and used only by the importer pages
#     (_page_importer, _page_inspector, _start_import).
#   • CONFLICT FIX: CLOUD_MUSIC, BUCKET_NAME — imported from
#     both scene_music_matcher_engine and importer_pipeline
#     (same env vars, last import silently overwrote first).
#     Removed from importer_pipeline import block; single
#     authoritative values come from scene_music_matcher_engine.
#   • CONFLICT FIX: TAGS_FILE — same env var, same Path type,
#     imported from both engines. Removed from importer_pipeline
#     import; single value from scene_music_matcher_engine used
#     throughout (already a Path, compatible with all call sites).
#   • CONFLICT FIX: MUSIC_DIR — defined three times (importer_
#     pipeline as Path, _SP_MUSIC_DIR alias from parser engine,
#     and a post-import string re-definition at module level).
#     Kept the importer_pipeline Path import as MUSIC_DIR.
#     Removed the redundant module-level string re-definition
#     (line 296 of v4.00: MUSIC_DIR = _os.getenv(...) or ...).
# v3.50 (2026-05-12)
#  •  Import music_library_importer
# v3.473 (2026-05-11)
#   • BUG FIX: Sidebar collapse button displayed "keyb" text instead of
#     the arrow icon. Streamlit renders the keyboard shortcut label inside
#     span/kbd/p child elements; the previous body[data-theme] scoped
#     font-size:0 rule did not reach those children.
#     Fix: replaced with the comprehensive unscoped multi-selector block
#     from music_library_importer, targeting the button and every child
#     element type (span, kbd, p, div:not(:has(svg))) with font-size:0,
#     line-height:0, letter-spacing:0. Added section[data-testid=
#     "stSidebar"] kbd { display:none } for belt-and-braces coverage.
#     The unscoped rules apply to both blue and parser themes.
# v3.472 (2026-05-11)
#   • BUG FIX: Scene-Music Scoring / Manual Selection — "Starts with"
#     returned too many results for single-letter queries (same root
#     cause as music_library_importer v3.34).
#     SQL fix: prefix mode now searches filename ONLY with no OR against
#     semantic_title_en (which caused every track whose title contained
#     the letter anywhere to be included).
#     Widget fix: "Starts with" checkbox moved outside the fc1 column
#     context to the main expander flow above the Search button, so its
#     session state is committed before the button executes.
#     Search button now reads name, starts_with and tags directly from
#     st.session_state to avoid column-buffer scoping issues.
# v3.471 (2026-05-11)
#   • BUG FIX: Scene-Music Scoring / Manual Selection —
#     "Starts with" checkbox had no effect on search results.
#     Root cause: st.checkbox and st.text_input were given both
#     a key= and a value= parameter pointing to different session
#     state keys ("scr_starts_with_chk" vs "scr_manual_starts_with",
#     "scr_name_{ver}" vs "scr_manual_name"). In Streamlit, when a
#     keyed widget also receives value=, the value= re-applies the
#     old session state on every rerun, silently resetting the
#     widget before the Search button logic reads it.
#     Fix: removed value= from both widgets — state is now driven
#     exclusively by the widget key. The mirror writes to
#     scr_manual_starts_with and scr_manual_name are kept for
#     downstream compatibility but no longer feed back into the
#     widget via value=.
#   • BUG FIX: Sidebar rendered in Streamlit's default grey instead
#     of the amber --bg-panel (#faeee0).
#     Root cause: the previous body[data-theme="blue"] scoped rule
#     was overridden by Streamlit's config.toml secondaryBackground.
#     Fix: added unscoped [data-testid="stSidebar"] rule (identical
#     to music_library_importer pattern) before the themed block —
#     unscoped rules win the specificity battle against config.toml.
#     Removed the now-redundant body[data-theme="blue"] scoped
#     sidebar background/border/font-family rules.
#   • Palette unified with music_library_importer v3.2.
#     Blue (#2c6fad) theme replaced by amber (#c96a1a) palette.
#     Added full :root design-token block (identical to
#     music_library_importer): --bg, --bg-panel, --bg-raised,
#     --bg-card, --border, --border-s, --accent, --accent-lt,
#     --accent-dim, --accent-bg, --green, --red, --yellow,
#     --txt, --txt-2, --txt-3, --log-bg/txt/ok/err/warn/info,
#     --sb-high/mid/low, --accent-shadow/border/hover/chip,
#     --info-bg, --mono, --sans, --serif.
#     All blue-theme component class rules (.page-header,
#     .page-title, .card, .b-ok/.b-warn/.b-err/.b-info,
#     .m-tile, .p-step, .score-formula, .score-bar-*)
#     rewritten to reference :root variables — no hardcoded hex.
#     _score_bar() inline color map updated to amber equivalents;
#     label/track/value colours updated (#5a3e28, #f0dcc4, #6b4423).
#     Five _render_score_breakdown() sub-header divs updated
#     from #4a6888 → #6b4423 (var(--txt-3) equivalent).
#     Sidebar uses var(--accent) and var(--txt-3) throughout.
#     Green parser theme is unchanged.
#   • Unscoped [data-testid="stDataFrame"] rule moved inside
#     body[data-theme="blue"] selector — no longer bleeds into
#     the green parser page.
#   • BUG FIX: _score_bar() rendered as plain text on HF Spaces.
#     CSS classes (score-bar-row/label/track/fill/val/wgt) and
#     CSS variable var(--accent) are defined in the module-level
#     st.markdown() block; Streamlit 1.45+ does not guarantee
#     global stylesheet availability inside st.expander() context.
#     Fix: replaced all CSS class references and var(--) variables
#     in _score_bar() with fully self-contained inline styles.
#     Function now renders correctly in any Streamlit container.
#   • BUG FIX: StreamlitAPIException "Expanders may not be nested
#     inside other expanders" in _render_score_breakdown().
#     Replaced all 5 st.expander() calls with st.container() +
#     .s-sub markdown section header. (carried from v3.45 work)
# v3.44 (2026-05-10)
#   • BUG FIX: replaced all width="stretch" with use_container_width=True
#     in st.dataframe(), st.button(), st.download_button() calls —
#     Streamlit 1.45+ / Python 3.13 on HF Spaces requires an int for
#     the width parameter; "stretch" raised TypeError: Cannot set
#     Arrow.width to 'stretch'. 19 occurrences fixed across all pages.
#   • st.selectbox() and st.number_input() width="stretch" removed
#     (those widgets do not support use_container_width).
# v3.422 (2026-05-08)
#   • Active sidebar button colour fix: added high-specificity
#     [data-testid="stSidebar"] selector rules that override
#     Streamlit's config.toml primaryColor (which was rendering
#     red and defeating the previous !important rules).
#     Active button now shows --accent-lt (#4a90d9) correctly.
# v3.41 (2026-05-07)
#   • Unified GUI typography across all pages using Screenplay
#     Parser as the reference.
#   • Added _page_header() — 2.4rem serif title + mono tagline
#     + env line, matching sp-title/sp-tagline structure exactly.
#     Used on Dashboard, Music Matcher, PDF Reports, Scoring.
#   • Added _section_header() — 1.5rem serif + mono uppercase
#     subtitle, matching sec-title/sec-subtitle exactly.
#     _header() is now an alias so engine-sourced pages inherit it.
#   • Removed all emojis from page titles and card titles.
#     Sidebar nav icons (⬡ ✦ ▶) are kept — navigation only.
#   • Fixed sec-title/sec-subtitle CSS collision: parser page had
#     its own definitions (DM Serif, green) that were overwritten
#     by the blue-theme definitions (Source Serif, dark). Now
#     parser overrides use body[data-theme="parser"] selectors.
#   • Added .score-formula, .accent, .danger, .accent-lg CSS
#     classes — removed all hardcoded inline font-size/color from
#     the score breakdown card in _render_score_breakdown().
#   • Removed obsolete .app-header / .app-title / .app-sub classes
#     from Python (kept in CSS for backward compat, unused in HTML).
#   • Dashboard card buttons unified to "Open" label.
# v3.404 (2026-05-07)
#   • Screenplay Parser page: fixed centering.
#     .block-container selector did not match — modern Streamlit
#     uses [data-testid="stMainBlockContainer"]. Rule now covers
#     all three known selector forms + explicit margin:auto.
# v3.403 (2026-05-07)
#   • Fixed theme-flash on page switch.
#     Previous approach injected a second <style> block inside
#     _page_parser() — its !important rules persisted in the
#     browser DOM across page switches, causing a visible green
#     flash before the blue theme re-asserted itself.
#     New approach: both themes live in the single module-level
#     <style> block using body[data-theme="blue"] and
#     body[data-theme="parser"] CSS selectors.  Each page
#     function calls _set_theme("blue"|"parser") as its first
#     statement, which emits a synchronous <script> that sets
#     document.body.dataset.theme in the same render frame —
#     the correct theme is active before any element is painted.
# v3.402 (2026-05-07)
#   • Screenplay Parser page: CSS approach replaced.
#     Removed broken .sp-page scoped CSS (Streamlit widgets render
#     outside the div wrapper, so scoping never reached them).
#     Added _SP_PAGE_CSS module-level constant — the full original
#     screenplay_parser.py green theme with !important overrides —
#     injected via st.markdown() at the top of _page_parser().
#     This restores: DM Serif Display / DM Mono / DM Sans fonts,
#     green background (#f0f7f2), correct btn/input/expander colours,
#     block-container centering (max-width 1060px), .sp-title /
#     .sp-tagline classes, and all badge / metric / log styles.
#   • _page_parser() no longer wraps content in <div class="sp-page">.
# v3.401 (2026-05-07)
#   • BUG FIX: _page_matcher, _page_pdf, _page_scoring and
#     render_episode_selection_section were rewritten from scratch
#     in v3.40 instead of using the exact source from
#     scene_music_matcher.py — causing KeyError 'scene_count',
#     wrong match_episode_v6 call signature, wrong export_episode_results_v6
#     call signature, and a simplified _render_score_breakdown missing
#     all sub-score expanders.  All four functions now match the
#     original scene_music_matcher.py v3.31 exactly.
# v3.40 (2026-05-07)
#   • Renamed script: scene_music_matcher.py → ai_music_supervisor.py.
#   • Added Screenplay Parser as a sidebar nav page ("parser").
#   • Screenplay Parser page renders the full screenplay_parser.py
#     GUI inline: CSS injection, session-state init, header,
#     API-key widget, upload, processing pipeline, and PDF export.
#   • Sidebar nav updated to five items:
#       ⬡  Dashboard | ✦  Screenplay Parser | ▶  Music Matcher
#       📄  PDF Reports | 🎶  Scene - Music Scoring
#   • Dashboard cards updated to include Screenplay Parser entry point.
#   • page_config title updated to "AI Music Supervisor".
#   • SCRIPT_VERSION/NAME/PIPELINE_VERSION updated accordingly.
# ────────────────────────────────────────────────────────────

import html as _html
import json
from pathlib import Path
from typing import Any, Optional

import threading
import time
import traceback

import streamlit as st

# ── Import everything from the matcher engine ────────────────────────────────
from scene_music_matcher_engine import (
    # config
    SCRIPT_VERSION as _ENGINE_VERSION,
    DB_CONFIG, OUTPUT_DIR, TAGS_FILE,
    CLOUD_MUSIC, BUCKET_NAME,
    # runtime config
    load_runtime_config, ensure_output_dir, get_default_weights,
    # data models
    NormalizedSceneQuery, NormalizedTrackQuery, RankedMatchV6,
    # DB helpers — get_connection(db_config) signature
    get_connection, _db_connect,
    fetch_scene_queries_for_episode,
    fetch_scene_query_by_id,
    fetch_scoring_episode_numbers,
    fetch_scoring_scene_numbers,
    fetch_scoring_theme_numbers,
    fetch_scene_query_by_episode_scene_theme,
    fetch_matches_for_theme,
    fetch_track_detail,
    fetch_track_query_candidates_by_hybrid,
    fetch_track_query_count,
    fetch_pdf_episode_numbers,
    fetch_matches_as_ranked_v6,
    fetch_episode_options,
    fetch_recent_episode_numbers,
    fetch_track_query_rows,
    # normalizers
    normalize_scene_query_row,
    normalize_track_query_row,
    build_input_validation_report,
    # pipeline
    match_episode_v6,
    save_episode_matches_v6,
    export_episode_results_v6,
    # scoring / debug
    debug_compare_scene_track,
    # audio
    get_audio_for_player,
    # utils
    round_floats_3, format_float_columns_df,
)

# ── Import everything from the screenplay parser engine ──────────────────────
from screenplay_parser_engine import (
    SCRIPT_VERSION as _PARSER_ENGINE_VERSION,
    MODEL,
    CLOUD_MUSIC as _SP_CLOUD_MUSIC,
    BUCKET_NAME as _SP_BUCKET_NAME,
    MUSIC_DIR as _SP_MUSIC_DIR,
    PGDATABASE as _SP_PGDATABASE,
    SUPPORTED_SCREENPLAY_SUFFIXES,
    ThemeList, SceneQueryRecord,
    set_runtime_openai_key,
    _runtime_openai_key,
    _validate_openai_key,
    _invalidate_openai_singleton,
    release_clap_model,
    read_screenplay,
    run_llm1_all_scenes,
    run_llm2_embed_save,
    LLM_1_theme_list,
    create_pdf_scenes,
    get_connection as _sp_get_connection,
)

# ── Import everything from the music library engine ───────────────────────────
from importer_pipeline import (
    # identity
    SCRIPT_VERSION as _PIPELINE_SCRIPT_VERSION,
    PIPELINE_VERSION,
    # runtime constants for importer GUI display and _start_import args
    # MUSIC_DIR is the Path from importer_pipeline (same env var as matcher engine)
    # CLOUD_MUSIC, BUCKET_NAME, TAGS_FILE come from scene_music_matcher_engine above
    MUSIC_DIR, CLAP_MODEL_NAME, DEVICE,
    TARGET_SR, SEGMENT_SECONDS, NUM_SEGMENTS,
    WORKERS, CHUNK_SIZE, BATCH_SIZE, AUDIO_BATCH_SIZE, TEXT_BATCH_SIZE,
    VECTOR_DIM, SKIP_EXISTING,
    # DB helper — no-arg signature get_connection(); aliased to avoid collision
    # with scene_music_matcher_engine.get_connection(db_config)
    get_connection as _imp_get_connection,
    # ETL orchestrator
    run_import,
)

# ── Shared pipeline constants (for Config page display) ───────────────────────
from pipeline_config import (
    CATEGORY_RULES,
    TAG_PROMPT_WEIGHTS,
    ENSEMBLE_PROMPT_WEIGHTS,
    TEXT_HYBRID_WEIGHTS,
    CONTRARY_TAGS,
)


import logging as _logging
import os as _os

_logger = _logging.getLogger(__name__)


# ============================================================
# STREAMLIT PAGE CONFIG  (must be the first st.* call)
# ============================================================

st.set_page_config(
    page_title=f"AI Music Supervisor v{SCRIPT_VERSION}",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Unified CSS — both themes live here, activated by body[data-theme] ───────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Design tokens (amber palette — shared with music_library_importer) ──── */
:root {
    /* Pastel-orange light palette */
    --bg:         #fdf6ee;   /* warm off-white page */
    --bg-panel:   #faeee0;   /* sidebar: slightly deeper peach */
    --bg-raised:  #fff8f2;   /* inputs, raised surfaces */
    --bg-card:    #ffffff;   /* cards: pure white for contrast */

    /* Borders */
    --border:     #e8c9a8;   /* medium warm border */
    --border-s:   #f0dcc4;   /* subtle border */

    /* Accent — deep amber-orange, readable on white */
    --accent:     #c96a1a;   /* primary brand colour */
    --accent-lt:  #e8874a;   /* hover / lighter accent */
    --accent-dim: #d9875a;   /* muted accent for secondary text */
    --accent-bg:  #fef0e4;   /* tinted accent background */

    /* Semantic colours */
    --green:      #2d8a5e;
    --red:        #c0392b;
    --yellow:     #b07d10;

    /* Text hierarchy */
    --txt:        #1a1208;   /* near-black, warm */
    --txt-2:      #5a3e28;   /* secondary text */
    --txt-3:      #6b4423;   /* tertiary / labels — AA contrast on bg */

    /* Log console — dark surface tinted to palette */
    --log-bg:     #1e1208;   /* console background */
    --log-txt:    #f5d9b8;   /* console base text */
    --log-ok:     #4aab7a;   /* ✅ / success lines */
    --log-err:    #d95f4b;   /* ❌ / [ERROR] lines */
    --log-warn:   #c9960a;   /* ⚠  lines */
    --log-info:   #e8a96a;   /* [INFO] lines */

    /* Score-bar thresholds */
    --sb-high:    var(--green);   /* score ≥ 65 % */
    --sb-mid:     var(--accent);  /* score ≥ 35 % */
    --sb-low:     var(--red);     /* score  < 35 % */

    /* Accent-derived alpha values */
    --accent-shadow: rgba(180,120,60,.07);   /* card box-shadow */
    --accent-border: rgba(201,106,26,.30);   /* .b-warn border */
    --accent-hover:  rgba(201,106,26,.18);   /* primary button hover */
    --accent-chip:   rgba(201,106,26,.25);   /* tag chip border */
    --info-bg:       rgba(90,62,40,.07);     /* .b-info badge background */

    /* Typography stacks */
    --mono:  'JetBrains Mono', 'Courier New', monospace;
    --sans:  'Inter', system-ui, sans-serif;
    --serif: 'Source Serif 4', Georgia, serif;
    
    
}
/* ── Base — unscoped so amber background applies on all pages ──────────── */
html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--txt) !important;
    font-family: var(--sans) !important;
    font-size: 17px !important;
}

/* ── Score bar ───────────────────────────────────────────────────────────── */
.sb-wrap  { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.sb-label { font-family: var(--mono); font-size: 0.68rem; color: var(--txt-2); width: 210px; }
.sb-track { flex: 1; height: 4px; background: var(--border-s); border-radius: 3px; }
.sb-fill  { height: 4px; border-radius: 3px; }
.sb-val   { font-family: var(--mono); font-size: 0.68rem; color: var(--txt-3); width: 36px; text-align: right; }

/* ── Log console ─────────────────────────────────────────────────────────── */
.log { background: var(--log-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 15px; font-family: var(--mono); font-size: 0.78rem; color: var(--log-txt); max-height: 360px; overflow-y: auto; white-space: pre-wrap; line-height: 1.65; }

/* --- Lightened Brown Global Button Override --- */
.stButton > button[kind="primary"] {
    background-color: #785135 !important; /* Medium-Dark Brown */
    color: #ffffff !important;           /* Keep white text for AA contrast */
    border: 1px solid #785135 !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease-in-out !important;
}

/* Hover state: Lighten further and add a slight lift */
.stButton > button[kind="primary"]:hover {
    background-color: #8c6346 !important; /* Lighter teak brown */
    color: #ffffff !important;
    border-color: #8c6346 !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    transform: translateY(-1px); /* Subtle physical lift effect */
}

/* Ensure disabled buttons don't stay brown */
.stButton > button[kind="primary"]:disabled {
    background-color: #d1c4b9 !important;
    border-color: #d1c4b9 !important;
    color: #8a7e72 !important;
}


/* ════════════════════════════════════════════════════════════
   AMBER THEME  (default — dashboard / matcher / pdf / scoring)
   All rules reference :root variables — a single :root edit re-themes.
   ════════════════════════════════════════════════════════════ */

/* ── Sidebar — unscoped so it wins over Streamlit's config.toml grey ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }

body[data-theme="blue"] html,
body[data-theme="blue"],
body[data-theme="blue"] [data-testid="stApp"] { background:var(--bg) !important; color:var(--txt) !important; font-family:var(--sans) !important; font-size:15px !important; }
body[data-theme="blue"] .stButton > button { background:var(--bg-card) !important; border:1px solid var(--border) !important; color:var(--txt) !important; font-family:var(--sans) !important; font-size:0.88rem !important; border-radius:7px !important; transition:all 0.13s !important; }
body[data-theme="blue"] .stButton > button:hover { border-color:var(--accent) !important; color:var(--accent) !important; background:var(--accent-bg) !important; }
body[data-theme="blue"] .stButton > button[kind="primary"] { background:var(--accent-bg) !important; border-color:var(--accent) !important; color:var(--accent) !important; font-weight:500 !important; }
body[data-theme="blue"] .stButton > button[kind="primary"]:hover { background:var(--accent-hover) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"]:focus,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:active { background:var(--accent-bg) !important; border:1px solid var(--accent) !important; color:var(--accent) !important; font-weight:500 !important; box-shadow:none !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background:var(--accent-hover) !important; border-color:var(--accent) !important; color:var(--accent) !important; }
body[data-theme="blue"] div[data-testid="stTextInput"] input,
body[data-theme="blue"] div[data-testid="stNumberInput"] input,
body[data-theme="blue"] div[data-testid="stTextArea"] textarea { background:var(--bg-raised) !important; border:1px solid var(--border) !important; color:var(--txt) !important; font-family:var(--sans) !important; font-size:0.88rem !important; border-radius:7px !important; }
body[data-theme="blue"] .stProgress > div > div { background:var(--accent) !important; }
body[data-theme="blue"] div[data-testid="stExpander"] { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
body[data-theme="blue"] div[data-testid="stExpander"] summary { color:var(--txt-2) !important; font-family:var(--sans) !important; }
body[data-theme="blue"] button[data-baseweb="tab"] { font-family:var(--sans) !important; font-size:0.88rem !important; color:var(--txt-2) !important; }
body[data-theme="blue"] button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent) !important; border-bottom-color:var(--accent) !important; }
body[data-theme="blue"] label,
body[data-theme="blue"] .stCheckbox label { color:var(--txt-2) !important; font-family:var(--sans) !important; font-size:0.88rem !important; }
body[data-theme="blue"] div[data-testid="stSelectbox"] label,
body[data-theme="blue"] div[data-testid="stFileUploader"] label { color:var(--txt-2) !important; }
body[data-theme="blue"] hr { border-color:var(--border) !important; }
body[data-theme="blue"] [data-testid="stDataFrame"] { border-radius:8px; border:1px solid var(--border); }
body[data-theme="blue"] [data-testid="stSidebarCollapseButton"] { font-size:0 !important; }
body[data-theme="blue"] [data-testid="stSidebarCollapsedControl"] { font-size:0 !important; }

/* Sidebar collapse-button "keyb…" label — hide text, keep SVG arrow.
   Unscoped so it applies to both themes. Targets every child element
   that could render text while preserving the SVG arrow icon. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] > *:not(svg),
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] kbd,
[data-testid="stSidebarCollapseButton"] p,
[data-testid="stSidebarCollapseButton"] div:not(:has(svg)) {
    font-size: 0 !important; line-height: 0 !important; letter-spacing: 0 !important;
}
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] > *:not(svg),
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stSidebarCollapsedControl"] kbd,
[data-testid="stSidebarCollapsedControl"] p,
[data-testid="stSidebarCollapsedControl"] div:not(:has(svg)) {
    font-size: 0 !important; line-height: 0 !important; letter-spacing: 0 !important;
}
section[data-testid="stSidebar"] kbd { display: none !important; }

/* ── Amber-theme component classes (non-parser pages) ── */
.page-header  { padding:2rem 0 1.1rem; border-bottom:2px solid var(--border); margin-bottom:1.6rem; }
.page-title   { font-family:var(--mono); font-size:2.4rem; font-weight:400; color:var(--accent); letter-spacing:-0.02em; line-height:1; margin:0; }
.page-tagline { font-family:var(--mono); font-size:1.05rem; color:var(--txt-3); letter-spacing:0.12em; text-transform:uppercase; margin:0; padding:6px 0 0; }
.page-env     { font-family:var(--mono); font-size:1.02rem; color:var(--txt-3); margin-top:4px; }
.sec-title    { font-family:var(--mono); font-size:1.5rem; font-weight:400; color:var(--accent); margin-bottom:0.2rem; }
.sec-subtitle { font-family:var(--mono); font-size:1.02rem; color:var(--txt-3); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:1.4rem; }
.s-title { font-family:var(--serif); font-size:1.5rem; font-weight:400; color:var(--txt); margin-bottom:0.2rem; }
.s-sub   { font-family:var(--mono); font-size:1.02rem; color:var(--txt-3); letter-spacing:0.09em; text-transform:uppercase; margin-bottom:1.1rem; }
.card       { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:1.1rem 1.3rem; margin-bottom:0.9rem; box-shadow:0 1px 3px var(--accent-shadow); }
.card-title { font-family:var(--serif); font-size:1.00rem; font-weight:400; color:var(--accent); margin-bottom:0.35rem; }
.card-body  { font-family:var(--sans); font-size:0.85rem; color:var(--txt-2); line-height:1.65; }
.score-formula { font-family:var(--mono); font-size:0.88rem; line-height:2.0; }
.score-formula .accent    { color:var(--accent); }
.score-formula .danger    { color:var(--red); }
.score-formula .accent-lg { font-size:1.05rem; color:var(--accent); }
.b-ok   { background:rgba(45,138,94,.10);  color:var(--green); border:1px solid rgba(45,138,94,.28); }
.b-warn { background:var(--accent-bg);     color:var(--accent); border:1px solid var(--accent-border); }
.b-err  { background:rgba(192,57,43,.09);  color:var(--red);   border:1px solid rgba(192,57,43,.28); }
.b-info { background:var(--info-bg);       color:var(--txt-2); border:1px solid var(--border); }
.m-row  { display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap; }
.m-tile { flex:1; min-width:100px; background:var(--accent-bg); border:1px solid var(--border-s); border-radius:9px; padding:12px 14px; text-align:center; }
.m-val  { font-family:var(--serif); font-size:1.70rem; color:var(--accent); line-height:1; margin-bottom:3px; }
.m-lbl  { font-family:var(--mono); font-size:0.90rem; color:var(--txt-3); text-transform:uppercase; letter-spacing:0.10em; }
.pipe   { display:flex; margin-bottom:1.3rem; gap:3px; }
.p-step { flex:1; padding:8px 4px; text-align:center; font-family:var(--mono); font-size:0.93rem; letter-spacing:0.05em; text-transform:uppercase; color:var(--txt-3); border-top:2px solid var(--border); background:var(--bg-raised); border-radius:0 0 4px 4px; }
.p-done   { color:var(--green); border-top-color:var(--green); background:rgba(45,138,94,.06); }
.p-active { color:var(--accent); border-top-color:var(--accent); background:var(--accent-bg); }
.score-bar-row   { display:flex; align-items:center; gap:10px; margin-bottom:5px; }
.score-bar-label { font-family:var(--mono); font-size:1.02rem; color:var(--txt-2); width:200px; }
.score-bar-track { flex:1; height:6px; background:var(--border-s); border-radius:3px; }
.score-bar-fill  { height:6px; border-radius:3px; }
.score-bar-val   { font-family:var(--mono); font-size:1.02rem; color:var(--txt-3); width:42px; text-align:right; }
.score-bar-wgt   { font-family:var(--mono); font-size:0.95rem; color:var(--txt-3); width:58px; text-align:right; }
body[data-theme="blue"] [data-testid="stDataFrame"] { border-radius:8px; border:1px solid var(--border); }

/* ════════════════════════════════════════════════════════════
   GREEN THEME  (parser page only)
   ════════════════════════════════════════════════════════════ */
body[data-theme="parser"] { --sp-bg:#f0f7f2; --sp-panel:#e8f3eb; --sp-raised:#ddeee2; --sp-card:#d4e9db; --sp-border:#b2d4bc; --sp-border-soft:#c8e0cf; --sp-gold:#2e7d52; --sp-teal:#1a7a6e; --sp-red:#c0392b; --sp-txt:#1a2e22; --sp-txt2:#3d5c47; --sp-muted:#6b8f74; }

body[data-theme="parser"] html,
body[data-theme="parser"],
body[data-theme="parser"] [data-testid="stApp"] { background:var(--sp-bg) !important; color:var(--sp-txt) !important; font-family:'DM Sans',sans-serif !important; font-size:15px !important; }
body[data-theme="parser"] .block-container,
body[data-theme="parser"] [data-testid="stMainBlockContainer"],
body[data-theme="parser"] [data-testid="block-container"] { max-width:1060px !important; padding-top:0 !important; padding-bottom:2rem !important; margin-left:auto !important; margin-right:auto !important; }
body[data-theme="parser"] [data-testid="stSidebar"] { background:var(--sp-panel) !important; border-right:1px solid var(--sp-border) !important; }
body[data-theme="parser"] [data-testid="stSidebar"] * { font-family:'DM Sans',sans-serif !important; }
body[data-theme="parser"] .stButton > button { background:var(--sp-raised) !important; border:1px solid var(--sp-border) !important; color:var(--sp-txt2) !important; font-family:'DM Sans',sans-serif !important; border-radius:7px !important; font-size:0.88rem !important; transition:all 0.15s !important; }
body[data-theme="parser"] .stButton > button:hover:not(:disabled) { border-color:var(--sp-gold) !important; color:var(--sp-gold) !important; background:rgba(46,125,82,0.08) !important; }
body[data-theme="parser"] .stButton > button[kind="primary"] { background:rgba(46,125,82,0.15) !important; border-color:var(--sp-gold) !important; color:var(--sp-gold) !important; font-weight:500 !important; }
body[data-theme="parser"] .stButton > button[kind="primary"]:hover:not(:disabled) { background:rgba(46,125,82,0.28) !important; }
body[data-theme="parser"] .stButton > button:disabled { background:var(--sp-card) !important; border-color:var(--sp-border-soft) !important; color:var(--sp-muted) !important; opacity:0.55 !important; cursor:not-allowed !important; }
body[data-theme="parser"] div[data-testid="stTextInput"] input,
body[data-theme="parser"] div[data-testid="stNumberInput"] input,
body[data-theme="parser"] div[data-testid="stSelectbox"] > div,
body[data-theme="parser"] div[data-testid="stTextArea"] textarea { background:var(--sp-raised) !important; border:1px solid var(--sp-border) !important; color:var(--sp-txt) !important; font-family:'DM Sans',sans-serif !important; border-radius:7px !important; }
body[data-theme="parser"] div[data-testid="stFileUploader"] { background:var(--sp-raised) !important; border:1px dashed var(--sp-border) !important; border-radius:10px !important; }
body[data-theme="parser"] .stProgress > div > div { background:var(--sp-gold) !important; }
body[data-theme="parser"] div[data-testid="stExpander"] { background:var(--sp-card) !important; border:1px solid var(--sp-border) !important; border-radius:8px !important; }
body[data-theme="parser"] div[data-testid="stExpander"] summary { font-family:'DM Sans',sans-serif !important; color:var(--sp-txt2) !important; }
body[data-theme="parser"] label,
body[data-theme="parser"] .stCheckbox label { color:var(--sp-txt2) !important; font-family:'DM Sans',sans-serif !important; font-size:0.88rem !important; }
body[data-theme="parser"] div[data-testid="stSelectbox"] label,
body[data-theme="parser"] div[data-testid="stFileUploader"] label { color:var(--sp-txt2) !important; }
body[data-theme="parser"] .stAlert { border-radius:8px !important; font-family:'DM Sans',sans-serif !important; }
body[data-theme="parser"] hr { border-color:var(--sp-border) !important; }

/* ── Parser-specific classes (green theme) ── */
.sp-title { font-family:'DM Serif Display',Georgia,serif; font-size:2.4rem; font-weight:400; color:#2e7d52; letter-spacing:-0.02em; line-height:1; margin:0; }
.sp-tagline { font-family:'DM Mono',monospace; font-size:1.05rem; color:#6b8f74; letter-spacing:0.12em; text-transform:uppercase; margin:0; padding-bottom:3px; }
body[data-theme="parser"] .sec-title { font-family:'DM Serif Display',Georgia,serif !important; font-size:1.5rem !important; color:#1a2e22 !important; margin-bottom:0.2rem !important; font-weight:400 !important; }
body[data-theme="parser"] .sec-subtitle { font-family:'DM Mono',monospace !important; font-size:1.02rem !important; color:#6b8f74 !important; letter-spacing:0.1em !important; text-transform:uppercase !important; margin-bottom:1.4rem !important; }
.pipe-steps { display:flex; gap:0; margin-bottom:1.6rem; }
.pipe-step { flex:1; padding:10px 6px; text-align:center; font-family:'DM Mono',monospace; font-size:0.95rem; letter-spacing:0.05em; text-transform:uppercase; color:#6b8f74; border-top:2px solid #b2d4bc; }
.pipe-step.done   { color:#1a7a6e; border-top-color:#1a7a6e; }
.pipe-step.active { color:#2e7d52; border-top-color:#2e7d52; }
.metric-row { display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap; }
.metric-tile { flex:1; min-width:100px; background:#ddeee2; border:1px solid #c8e0cf; border-radius:8px; padding:12px 14px; text-align:center; }
.metric-val { font-family:'DM Serif Display',Georgia,serif; font-size:1.7rem; color:#2e7d52; line-height:1; margin-bottom:3px; }
.metric-lbl { font-family:'DM Mono',monospace; font-size:0.93rem; color:#6b8f74; text-transform:uppercase; letter-spacing:0.1em; }
.badge { display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:1.02rem; font-weight:500; letter-spacing:0.07em; text-transform:uppercase; }
.badge-ok   { background:rgba(26,122,110,0.10); color:#1a7a6e; border:1px solid rgba(26,122,110,0.30); }
.badge-warn { background:rgba(46,125,82,0.12);  color:#2e7d52; border:1px solid rgba(46,125,82,0.30); }
.badge-err  { background:rgba(192,57,43,0.10);  color:#c0392b; border:1px solid rgba(192,57,43,0.30); }
.badge-info { background:rgba(107,143,116,0.15); color:#3d5c47; border:1px solid #b2d4bc; }
.log-console { background:#f4fbf6; border:1px solid #b2d4bc; border-radius:8px; padding:12px 14px; font-family:'DM Mono',monospace; font-size:1.14rem; color:#1e5c35; max-height:280px; overflow-y:auto; white-space:pre-wrap; line-height:1.6; }
.rec-table { width:100%; border-collapse:collapse; font-size:1.00rem; }
.rec-table td { padding:6px 10px; vertical-align:top; border-bottom:1px solid #c8e0cf; color:#1a2e22; }
.rec-table td:first-child { font-family:'DM Mono',monospace; font-size:1.05rem; color:#6b8f74; white-space:nowrap; width:160px; }
</style>
""", unsafe_allow_html=True)

# ── Theme switcher helpers ────────────────────────────────────────────────────
# Each page calls _set_theme("parser") or _set_theme("blue") as its FIRST
# statement.  The JS runs synchronously in the same render frame as the
# page content, so the correct body attribute is set before any repaint.
# CSS in the global <style> block above uses body[data-theme="..."] selectors
# to apply the right palette with no flash on switching.

def _set_theme(theme: str) -> None:
    """Emit a synchronous JS snippet that sets body[data-theme] immediately."""
    st.markdown(
        f'<script>document.body.setAttribute("data-theme","{theme}");</script>',
        unsafe_allow_html=True,
    )

# ── Session-state defaults ────────────────────────────────────────────────────
_SS_DEFAULTS: dict[str, Any] = {
    # ── navigation ──
    "page":                        "dashboard",
    "db_status":                   {},
    # ── matcher ──
    "last_result_bundle":          None,
    "last_export_payload":         None,
    "scr_selected_track_id":       None,
    "scr_selected_scene_query_id": None,
    "scr_selected_rank":           None,
    "scr_selected_score":          None,
    "scr_selected_explanation":    None,
    "scr_last_selection":          None,
    "scr_autoplay":                False,
    "scr_mode":                    "Ranked matches",
    "scr_manual_name":             "",
    "scr_name_input_ver":          0,
    "scr_manual_starts_with":      False,
    "scr_manual_results":          None,
    # ── screenplay parser ──
    "parser_stage":                None,
    "parser_error":                None,
    "parser_records":              [],
    "parser_episode_nr":           None,
    "key_entry_mode":              False,
    "openai_api_key":              "",
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# SHARED UI HELPERS  (matcher-style)
# ============================================================

def _safe(val: Any) -> str:
    return _html.escape(str(val)) if val is not None else "–"


def _badge(text: str, kind: str = "info") -> str:
    k = {"ok": "b-ok", "warn": "b-warn", "err": "b-err"}.get(kind, "b-info")
    return f'<span class="badge {k}">{_safe(text)}</span>'


def _metrics(tiles: list[tuple[str, str]]) -> None:
    parts = "".join(
        f'<div class="m-tile"><div class="m-val">{_safe(v)}</div>'
        f'<div class="m-lbl">{_safe(l)}</div></div>'
        for l, v in tiles
    )
    st.markdown(f'<div class="m-row">{parts}</div>', unsafe_allow_html=True)


def _pipe(steps: list[str], done: int, active: int = -1) -> None:
    html = '<div class="pipe">'
    for i, s in enumerate(steps):
        cls   = "p-done" if i < done else ("p-active" if i == active else "")
        html += f'<div class="p-step {cls}">{_safe(s)}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _page_header(title: str, sub: str = "") -> None:
    """Large page title — mirrors sp-title / sp-tagline structure of the Parser page."""
    pgdb      = _os.getenv("PGDATABASE", "—")
    tracks_val = BUCKET_NAME or "—" if CLOUD_MUSIC else MUSIC_DIR or "—"
    env_line = (
        f"Database:&nbsp;<b>{_safe(pgdb)}</b>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"Tracks:&nbsp;<b>{_safe(tracks_val)}</b>"
    )
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-title">{_safe(title)}<br><br></div>'
        f'<div class="page-tagline">{_safe(sub)}</div>'
#        f'<div class="page-env">{env_line}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _section_header(title: str, sub: str = "") -> None:
    """Section heading — mirrors sec-title / sec-subtitle of the Parser page."""
    st.markdown(f'<div class="sec-title">{_safe(title)}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-subtitle">{_safe(sub)}</div>', unsafe_allow_html=True)


def _header(title: str, sub: str = "") -> None:
    """Alias for _section_header — kept for compatibility with engine-sourced page functions."""
    _section_header(title, sub)


def _score_bar(label: str, value: float, weight: float, color: str = "#c96a1a") -> None:
    """Render a single score bar row with fully inline styles.

    All styles are inlined so the component renders correctly inside
    st.expander() on HF Spaces, where the global stylesheet injected
    by the module-level st.markdown(CSS) block may not be in scope.
    The default color uses the hex equivalent of var(--accent) from
    the amber palette.  Pass "#c0392b" for penalty (red) bars.
    """
    pct          = max(0.0, min(1.0, value)) * 100
    contribution = weight * value
    # Resolve common CSS variable aliases to amber-palette hex equivalents
    _color_map = {
        "var(--accent)":    "#c96a1a",
        "var(--accent-lt)": "#e8874a",
        "var(--red)":       "#c0392b",
        "var(--green)":     "#2d8a5e",
    }
    bar_color = _color_map.get(color, color)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">'
        f'<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
        f'font-size:0.80rem;color:#5a3e28;width:250px;flex-shrink:0;">{label}</div>'
        f'<div style="flex:1;height:6px;background:#f0dcc4;border-radius:3px;">'
        f'<div style="width:{pct:.1f}%;height:6px;border-radius:3px;'
        f'background:{bar_color};"></div>'
        f'</div>'
        f'<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
        f'font-size:0.80rem;color:#6b4423;width:48px;text-align:right;">'
        f'{value:.2f}</div>',
#        f'<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
#        f'font-size:0.90rem;color:#6b4423;width:72px;text-align:right;">'
#        f'×{weight:.2f}={contribution:.3f}</div>'
#        f'</div>',
        unsafe_allow_html=True,
    )


def _render_log(lines: list[str]) -> None:
    """Render import-log lines as a styled HTML console block."""
    raw  = "\n".join(lines[-150:]) if lines else "— awaiting run —"
    text = _html.escape(raw)
    for old, new in [
        ("✅", '<span style="color:var(--log-ok)">✅</span>'),
        ("❌", '<span style="color:var(--log-err)">❌</span>'),
        ("⚠",  '<span style="color:var(--log-warn)">⚠</span>'),
        ("[ERROR]", '<span style="color:var(--log-err)">[ERROR]</span>'),
        ("[INFO]",  '<span style="color:var(--log-info)">[INFO]</span>'),
    ]:
        text = text.replace(old, new)
    st.markdown(f'<div class="log">{text}</div>', unsafe_allow_html=True)


# ============================================================
# DB STATUS CHECK
# ============================================================

def _check_db() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False, "error": None,
        "track_query": 0, "scene_query": 0, "matches": 0,
    }
    try:
        conn = get_connection(DB_CONFIG)
        cur  = conn.cursor()
        for tbl, key in [
            ("track_query",            "track_query"),
            ("scene_query",            "scene_query"),
            ("scene_music_matches_v6", "matches"),
        ]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                result[key] = int(cur.fetchone()[0])
            except Exception:
                conn.rollback()
                result[key] = 0
        cur.close()
        conn.close()
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ============================================================
# SIDEBAR
# ============================================================

def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="padding:1.2rem 0 0.7rem">'
            '<div style="font-family:var(--serif);font-size:1.35rem;font-weight:400;'
            'color:var(--accent);line-height:1.15;letter-spacing:-0.01em">🎬 AI Music Supervisor</div>'
            f'<div style="font-family:var(--mono);font-size:0.98rem;color:var(--txt-3);'
            f'letter-spacing:.10em;text-transform:uppercase;margin-top:5px">'
            f'v. {SCRIPT_VERSION}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        for pid, icon, label in [
            ("dashboard", "⬡",  "Dashboard"),
            ("parser",    "✦",  "Screenplay Parser"),
            ("matcher",   "🎵", "Scene - Music Matcher"),
            ("scoring",   "🎶", "Scene - Music Scoring"),
            ("pdf",       "📝", "Scene - Music Reports"),
        ]:
            active = st.session_state.page == pid
            if st.button(
                f"{icon}  {label}", key=f"nav_{pid}",
                type="primary" if active else "secondary",
            ):
                st.session_state.page = pid
                st.rerun()

#        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown(
            '<div style="padding:1.2rem 0 0.7rem">'
            '<div style="font-family:var(--serif);font-size:1.25rem;font-weight:400;'
            'color:var(--accent);line-height:1.15;letter-spacing:-0.01em">🎷 Music Library Importer</div>'
            f'<div style="font-family:var(--mono);font-size:0.98rem;color:var(--txt-3);'
            f'letter-spacing:.10em;text-transform:uppercase;margin-top:5px">'
#            f'v. {SCRIPT_VERSION}</div>'
            '</div>',
            unsafe_allow_html=True,)
        
        for pid, icon, label in [
            ("importer",  "▶", "Run Importer"),
            ("inspector", "🔍", "Track Inspector"),
            ("config",    "⚙", "Configuration"),
        ]:
            active = st.session_state.page == pid
            if st.button(
                f"{icon}  {label}", key=f"nav_{pid}",
                type="primary" if active else "secondary",
            ):
                st.session_state.page = pid
                st.rerun()
            
        st.markdown("<hr>", unsafe_allow_html=True)

        pgdb       = _os.getenv("PGDATABASE", "—")
        tracks_val = BUCKET_NAME or "—" if CLOUD_MUSIC else MUSIC_DIR or "—"

        st.markdown(
            f'<div style="font-family:var(--mono);font-size:1.02rem;'
            f'color:var(--txt-2);line-height:2.2;margin-top:4px">'
            f'<b>Database:</b> {_safe(pgdb)}<br>'
            f'<b>Tracks:</b> {_safe(tracks_val)}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: DASHBOARD
# ============================================================

def _page_dashboard() -> None:
    _set_theme("blue")
    _page_header(
        "🎬 AI Music Supervisor",
        f"v. {SCRIPT_VERSION} · {SCRIPT_NAME} · scene-to-music ranking engine",
    )
    
    # db = st.session_state.db_status
    # ok = db.get("ok", False)
    # if db:
    #     _metrics([
    #         ("Tracks indexed", str(db.get("track_query", "–"))),
    #         ("Scenes parsed",  str(db.get("scene_query", "–"))),
    #         ("Match rows",     str(db.get("matches",     "–"))),
    #         ("Last check",     "✓ DB reached" if ok else "✗ unreachable"),
    #     ])
    # if st.button("↻  Refresh DB stats", disabled=ok):
    #     with st.spinner("Querying…"):
    #         st.session_state.db_status = _check_db()
    #     st.rerun()
    # st.markdown("<hr>", unsafe_allow_html=True)
   
    c1, c2, c3, c4 = st.columns(4)
    for col, title, body, page in [
        (c1, "✦ Screenplay Parser",
         "Upload a .docx screenplay. LLM splits into scenes, extracts themes, "
         "generates CLAP embeddings and saves to <code>scene_query</code>.",
         "parser"),
        (c2, "🎵 Scene - Music Matcher",
         "Retrieves candidate tracks per scene, scores by embeddings + semantics "
         "+ targets + tags, applies penalties. Writes <code>scene_music_matches_v6</code>.",
         "matcher"),
        (c3, "🎶 Scene - Music Scoring",
         "Inspect and compare component scores for any scene ↔ track pair: "
         "embedding, semantic, targets, tag-selection, and penalties.",
         "scoring"),
        (c4, "📝 Scene - Music Reports",
         "Download ranked match results as full or compact PDF and XLSX files, "
         "grouped by scene and theme. Select an episode to generate all four formats.",
         "pdf"),
 
    ]:
        with col:
            st.markdown(
                f'<div class="card"><div class="card-title">{_safe(title)}</div>'
                f'<div class="card-body">{body}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Open", key=f"d_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    _section_header("Pipeline stages", "four-step workflow")
    _pipe(
        [ "Parse screenplay", "Match tracks to scenes",  "Inspect scores", "Export match reports" ],
        done=0,
    )
    st.markdown("Run each step in order. Screenplay Parser generates semantic music embeddings.  "
                "Scene - Music Matcher seeks candidate tracks and scores them against scenes and themes using a combination of "
                "embedding + semantic + targets + tag scoring with contextual penalties.  "
                "Scene - Music Scoring lets you inspect the component scores for any scene-track pair,"
                " and Scene - Music Reports exports the ranked matches in PDF and XLSX formats, grouped by scene and theme.", unsafe_allow_html=True)


# ============================================================
# PAGE: SCREENPLAY PARSER
# ============================================================

# ── Screenplay Parser UI helpers (scoped to .sp-page) ────────────────────────

def _sp_safe(val: Any) -> str:
    return _html.escape(str(val)) if val is not None else "–"


def _sp_badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def _sp_section_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _sp_pipe_steps(steps: list[str], done_up_to: int, active: int) -> None:
    html = '<div class="pipe-steps">'
    for i, s in enumerate(steps):
        cls = "done" if i < done_up_to else ("active" if i == active else "")
        html += f'<div class="pipe-step {cls}">{s}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _sp_metric_tiles(tiles: list[tuple[str, str]]) -> None:
    parts = "".join(
        f'<div class="metric-tile"><div class="metric-val">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>'
        for lbl, val in tiles
    )
    st.markdown(f'<div class="metric-row">{parts}</div>', unsafe_allow_html=True)


def _sp_log_console(lines: list[str]) -> None:
    text = "\n".join(lines[-80:]) if lines else "— awaiting run —"
    st.markdown(f'<div class="log-console">{text}</div>', unsafe_allow_html=True)


def _sp_record_table_html(record: SceneQueryRecord) -> str:
    rd   = record.model_dump()
    rows = ""
    fields = [
        ("scene_nr / theme_nr",   f"{rd['scene_nr']} / {rd['theme_nr']}"),
        ("theme_title_pl",        rd["theme_title_pl"]),
        ("theme_title_en",        rd["theme_title_en"]),
        ("description_en",        rd["description_en"]),
        ("tags_summary_en",       rd["tags_summary_en"]),
        ("weight_profile",        rd["scene_music_semantics"]["weight_profile"]),
        ("emotional_direction",   ", ".join(rd["scene_music_semantics"]["emotional_direction"])),
        ("energy / tempo",        f"{rd['scene_music_targets']['energy_target']} / {rd['scene_music_targets']['tempo_target']}"),
        ("should_have_tags",      ", ".join(rd["scene_tag_selection"]["should_have_tags"])),
        ("must_not_tags",         ", ".join(rd["scene_tag_selection"]["must_not_tags"])),
        ("semantic_scene_prompt", rd["clap_prompt_ensemble"]["semantic_scene_prompt"]),
        ("concise_core_prompt",   rd["clap_prompt_ensemble"]["concise_core_prompt"]),
    ]
    for lbl, val in fields:
        rows += f'<tr><td>{lbl}</td><td>{val}</td></tr>'
    return f'<table class="rec-table"><tbody>{rows}</tbody></table>'


def _sp_reset_parser_state() -> None:
    for k in ["parser_stage", "parser_episode_nr", "parser_scenes",
              "parser_theme_lists", "parser_records", "parser_error", "parser_log"]:
        st.session_state.pop(k, None)


def _sp_append_log(msg: str) -> None:
    if "parser_log" not in st.session_state:
        st.session_state["parser_log"] = []
    st.session_state["parser_log"].append(msg)


def _sp_render_log(expand: bool = False) -> None:
    lines = st.session_state.get("parser_log", [])
    if not lines:
        return
    with st.expander(f"Process log ({len(lines)} lines)", expanded=expand):
        _sp_log_console(lines)


def _sp_render_error() -> None:
    err = st.session_state.get("parser_error")
    if not err:
        return
    st.markdown(
        f'<div style="margin-top:0.6rem;">{_sp_badge("Pipeline error", "err")}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Error details"):
        st.code(err)




def _sp_render_api_key_inline() -> bool:
    """Render OpenAI API key widget. Returns True when a valid key is active."""
    current_key = _runtime_openai_key()
    has_key     = bool(current_key)
    entry_mode  = st.session_state.get("key_entry_mode", False)

    _sp_section_header("OpenAI API key", "required to run LLM stages")

    if has_key and not entry_mode:
        masked = current_key[:8] + "…" + current_key[-4:]
        col_badge, col_btn = st.columns([3, 1])
        with col_badge:
            st.markdown(_sp_badge(f"Active: {masked}", "ok"), unsafe_allow_html=True)
        with col_btn:
            if st.button("⟳  Change key", key="sp_btn_change_key",
                         use_container_width=True):
                st.session_state["key_entry_mode"] = True
                _invalidate_openai_singleton()
                st.rerun()
        return True

    if not has_key:
        st.markdown(
            _sp_badge("No API key — enter below to continue", "err"),
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

    btn_label = "Enter key" if not has_key else "Change key"
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        new_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="Paste your OpenAI API key…",
            label_visibility="collapsed",
            key="sp_api_key_input",
        )
    with col_btn:
        save_clicked = st.button(
            btn_label, key="sp_btn_save_key",
            use_container_width=True, type="primary",
        )

    if save_clicked:
        val = new_key.strip()
        if not val:
            st.markdown(_sp_badge("Please paste a valid API key", "err"), unsafe_allow_html=True)
        else:
            with st.spinner("Validating key…"):
                err = _validate_openai_key(val)
            if err:
                st.markdown(_sp_badge(err, "err"), unsafe_allow_html=True)
            else:
                st.session_state["openai_api_key"] = val
                st.session_state["key_entry_mode"]  = False
                set_runtime_openai_key(val)
                st.rerun()

    if entry_mode and has_key:
        if st.button("✕  Cancel", key="sp_btn_cancel_key"):
            st.session_state["key_entry_mode"] = False
            st.rerun()

    return False


def _sp_render_upload_section() -> Optional[Any]:
    _sp_section_header("Upload", "step 1 · .doc / .docx screenplay file")
    with st.expander("⬆  Drop screenplay file", expanded=True):
        uploaded_file = st.file_uploader(
            "Drag & drop a screenplay (.doc / .docx)",
            type=["doc", "docx"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.markdown(
                _sp_badge(f"Loaded: {uploaded_file.name}", "ok"),
                unsafe_allow_html=True,
            )
    return uploaded_file


def _sp_render_processing_section(uploaded_file: Optional[Any]) -> None:
    if uploaded_file is None:
        if st.session_state.get("parser_stage") or st.session_state.get("parser_error"):
            _sp_reset_parser_state()

    _sp_section_header("Parse", "step 2 · three-stage pipeline")

    stage   = st.session_state.get("parser_stage")
    has_err = bool(st.session_state.get("parser_error"))

    stage_to_done   = {None: 0, "split": 1, "llm1": 3, "running_llm2": 3, "done": 5}
    stage_to_active = {None: 0, "split": 1, "llm1": 3, "running_llm2": 4, "done": -1}
    _sp_pipe_steps(
        ["Upload", "Split scenes", "LLM #1 · all scenes", "LLM #2 · all themes", "Embed + save"],
        done_up_to = stage_to_done.get(stage, 0),
        active     = stage_to_active.get(stage, 0),
    )

    st.markdown("---")

    # ── Stage 1 — Split scenes ──────────────────────────────────────────────
    _sp_section_header("Stage 1", "read file & detect scenes — no API calls")
    s1_ready = uploaded_file is not None and stage is None and not has_err
    if st.button("▶  Split scenes", type="primary",
                 disabled=not s1_ready, key="sp_btn_split"):
        _sp_reset_parser_state()
        st.session_state["parser_log"] = []
        with st.spinner("Reading and splitting screenplay…"):
            try:
                file_name  = uploaded_file.name
                file_bytes = uploaded_file.getvalue()
                if not file_bytes:
                    raise ValueError("Uploaded file is empty.")
                _sp_append_log(f"Reading: {file_name}")
                episode_nr, scenes = read_screenplay(file_name, file_bytes)
                st.session_state["parser_episode_nr"] = episode_nr
                st.session_state["parser_scenes"]     = scenes
                st.session_state["parser_stage"]      = "split"
                _sp_append_log(f"✓ Episode {episode_nr} — {len(scenes)} scenes detected")
            except Exception as exc:
                st.session_state["parser_error"] = str(exc)
                _sp_append_log(f"[ERROR] {exc}")
                _logger.exception("Stage 1 failed")
        st.rerun()

    scenes = st.session_state.get("parser_scenes")
    ep     = st.session_state.get("parser_episode_nr")
    if stage in ("split", "llm1", "done") and scenes and ep:
        _sp_metric_tiles([
            ("Episode",         str(ep)),
            ("Scenes detected", str(len(scenes))),
        ])
        with st.expander(f"Scene list ({len(scenes)} scenes)", expanded=False):
            for i, s in enumerate(scenes, 1):
                first_line = s.splitlines()[0][:120] if s.strip() else "(empty)"
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:1.08rem;'
                    f'color:var(--txt-3);padding:2px 0;">{i}. {_sp_safe(first_line)}</div>',
                    unsafe_allow_html=True,
                )

    _sp_render_log(expand=False)
    _sp_render_error()
    if has_err:
        return

    st.markdown("---")

    # ── Stage 2 — LLM #1 ───────────────────────────────────────────────────
    _sp_section_header("Stage 2", "LLM #1 — semantic segmentation for every scene")
    s2_ready = stage == "split" and scenes is not None and not has_err
    if st.button("▶  Run LLM #1 (all scenes)", type="primary",
                 disabled=not s2_ready, key="sp_btn_llm1"):
        total    = len(scenes)
        progress = st.progress(0, text="LLM #1 starting…")
        try:
            theme_lists: list[ThemeList] = []
            for idx, scene_txt in enumerate(scenes, 1):
                progress.progress(
                    (idx - 1) / total,
                    text=f"LLM #1 — scene {idx}/{total}…",
                )
                _sp_append_log(f"[{idx:>2}/{total}] LLM #1 — scene {idx}…")
                tl = LLM_1_theme_list(scene_txt)
                theme_lists.append(tl)
                _sp_append_log(
                    f"  → {len(tl.themes)} theme(s): "
                    + ", ".join(f'"{tb.theme_title_en}"' for tb in tl.themes)
                )
            progress.progress(1.0, text="LLM #1 complete")
            st.session_state["parser_theme_lists"] = [tl.model_dump() for tl in theme_lists]
            st.session_state["parser_stage"]       = "llm1"
            total_themes = sum(len(tl.themes) for tl in theme_lists)
            _sp_append_log(f"✓ LLM #1 done — {total} scenes, {total_themes} themes total")
        except Exception as exc:
            st.session_state["parser_error"] = str(exc)
            _sp_append_log(f"[ERROR] {exc}")
            _logger.exception("Stage 2 failed")
        st.rerun()

    _theme_list_dicts = st.session_state.get("parser_theme_lists")
    theme_lists_stored: Optional[list[ThemeList]] = (
        [ThemeList.model_validate(d) for d in _theme_list_dicts]
        if _theme_list_dicts else None
    )
    if stage in ("llm1", "done") and theme_lists_stored:
        total_themes = sum(len(tl.themes) for tl in theme_lists_stored)
        _sp_metric_tiles([
            ("Scenes segmented", str(len(theme_lists_stored))),
            ("Total themes",     str(total_themes)),
        ])
        with st.expander("LLM #1 results — theme summary", expanded=True):
            for i, tl in enumerate(theme_lists_stored, 1):
                titles = ", ".join(f'"{tb.theme_title_en}"' for tb in tl.themes)
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:1.08rem;'
                    f'color:var(--txt-3);padding:3px 0;">'
                    f'Scene {i}: {len(tl.themes)} theme(s) — {_sp_safe(titles)}</div>',
                    unsafe_allow_html=True,
                )

    _sp_render_log(expand=False)
    _sp_render_error()
    if has_err:
        return

    st.markdown("---")

    # ── Stage 3 — LLM #2 + embed + save ────────────────────────────────────
    s3_subtitle = (
        "⏳ running — do not close the page…"
        if stage == "running_llm2"
        else "LLM #2 + CLAP embeddings + DB save — one theme at a time"
    )
    _sp_section_header("Stage 3", s3_subtitle)
    s3_ready = (
        stage == "llm1"
        and theme_lists_stored is not None
        and ep is not None
        and not has_err
    )
    if st.button("▶  Run LLM #2 & Save", type="primary",
                 disabled=not s3_ready, key="sp_btn_llm2"):
        st.session_state["parser_stage"] = "running_llm2"
        progress = st.progress(0, text="LLM #2 starting…")

        def _on_progress(text: str, done: int, total: int) -> None:
            frac = (done / max(1, total)) * 0.95
            progress.progress(min(frac, 0.95), text=text)
            _sp_append_log(text)

        try:
            saved = run_llm2_embed_save(
                episode_nr        = ep,
                scenes            = scenes,
                theme_lists       = theme_lists_stored,
                progress_callback = _on_progress,
            )
            st.session_state["parser_records"] = saved
            _sp_append_log(f"✓ Saved {len(saved)} records for episode {ep}")
        except Exception as exc:
            st.session_state["parser_error"] = str(exc)
            _sp_append_log(f"[ERROR] {exc}")
            _logger.exception("Stage 3 failed")
        finally:
            release_clap_model()
            if not st.session_state.get("parser_error"):
                progress.progress(1.0, text="✓ All themes saved")
                st.session_state["parser_stage"] = "done"
        st.rerun()

    records = st.session_state.get("parser_records", [])
    if stage == "done" and records:
        _sp_metric_tiles([
            ("Episode",             str(ep)),
            ("Theme records saved", str(len(records))),
            ("Status",              "✓ done"),
        ])

    _sp_render_log(expand=stage == "done")
    _sp_render_error()

    if records:
        st.markdown("<hr>", unsafe_allow_html=True)
        _sp_section_header("Preview", "first SceneQueryRecord from this run")
        col_prev, col_json = st.columns([3, 2])
        with col_prev:
            st.markdown(_sp_record_table_html(records[0]), unsafe_allow_html=True)
        with col_json:
            with st.expander("Raw JSON"):
                st.json(records[0].model_dump())


def _sp_render_pdf_section() -> None:
    _sp_section_header("Export PDF", "step 3 · generate control PDF from scene_query")
    st.markdown("Enter an episode number to generate a PDF report")
    ep_default  = st.session_state.get("parser_episode_nr", "")
    col_input, col_btn = st.columns([2, 1])
    with col_input:
        episode_value = st.text_input(
            "Episode number",
            value=str(ep_default) if ep_default else "",
            placeholder="e.g. 4695",
            label_visibility="collapsed",
            key="sp_pdf_episode_input",
        )
    with col_btn:
        gen_clicked = st.button("⬇  Generate PDF", use_container_width=True,
                                key="sp_btn_gen_pdf")
    if not gen_clicked:
        return
    try:
        if not str(episode_value).strip():
            raise ValueError("Please enter an episode number.")
        episode_nr = int(str(episode_value).strip())
        with st.spinner(f"Building PDF for episode {episode_nr}…"):
            file_name, pdf_bytes = create_pdf_scenes(episode_nr)
        st.markdown(
            _sp_badge(f"PDF ready — {len(pdf_bytes) // 1024} KB", "ok"),
            unsafe_allow_html=True,
        )
        st.download_button(
            label=f"⬇  Download {file_name}",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True,
            key="sp_dl_pdf",
        )
    except Exception as exc:
        st.markdown(
            f'<div style="margin-top:0.5rem;">{_sp_badge("PDF generation failed", "err")}</div>',
            unsafe_allow_html=True,
        )
        st.error(str(exc))


def _page_parser() -> None:
    """Screenplay Parser page — activates green theme via body[data-theme]."""
    _set_theme("parser")

    # Sync any key stored in session_state into the engine singleton
    stored_key = st.session_state.get("openai_api_key", "")
    if stored_key:
        set_runtime_openai_key(stored_key)

    _page_header("✦ Screenplay Parser", "LLM segmentation · CLAP embeddings · scene_query")
    api_key_ok = _sp_render_api_key_inline()
    st.markdown("<hr>", unsafe_allow_html=True)

    if not api_key_ok:
        return

    uploaded_file = _sp_render_upload_section()
    st.markdown("<hr>", unsafe_allow_html=True)
    _sp_render_processing_section(uploaded_file)
    st.markdown("<hr>", unsafe_allow_html=True)
    _sp_render_pdf_section()


# ============================================================
# PAGE: MATCHER / PDF / SCORING  — exact originals from scene_music_matcher.py
# ============================================================

def render_episode_selection_section(config: dict[str, Any]) -> dict[str, Any]:
    """Render episode selector and runtime parameter controls."""
    ep_options: list[dict[str, Any]] = []
    try:
        conn = get_connection(DB_CONFIG)
        cur  = conn.cursor()
        ep_options = fetch_episode_options(cur)
        cur.close()
        conn.close()
    except Exception:
        ep_options = []

    if not ep_options:
        st.warning("No episodes found in scene_query. Run the Screenplay Parser first.")
        return {
            "episode_nr":    0,
            "preview_only":  False,
            "preview_top_n": int(config.get("preview_top_n", 5)),
            "rank_top_n":    int(config.get("rank_top_n",    10)),
        }

    st.markdown(
        '<div class="s-sub">Select an episode from scene_query. Episodes already '
        'matched are marked <em>- matched</em>.</div>',
        unsafe_allow_html=True,
    )
    outer_left, center, outer_right = st.columns([1.2, 2.2, 1.2])
    with center:
        selected_opt = st.selectbox(
            "Select episode", options=ep_options,
            format_func=lambda x: x["label"], index=0,
        )
        rank_top_n = st.number_input(
            "Rank top-N", min_value=1, step=1,
            value=int(config.get("rank_top_n", 10)),
        )
    episode_nr = int(selected_opt["episode_nr"]) if selected_opt else 0
    return {
        "episode_nr":    episode_nr,
        "preview_only":  False,
        "preview_top_n": int(config.get("preview_top_n", 5)),
        "rank_top_n":    int(rank_top_n),
    }


def render_input_validation_section(cur, ui_state: dict[str, Any]) -> dict[str, Any]:
    """Render compact input-data validation panel for selected episode."""
    episode_nr = int(ui_state.get("episode_nr", 0) or 0)
    if episode_nr <= 0:
        return {}
    scene_rows  = fetch_scene_queries_for_episode(cur, episode_nr)
    scenes:      list[NormalizedSceneQuery] = []
    scene_errors: list[str]                = []
    for row in scene_rows:
        try:
            scenes.append(normalize_scene_query_row(row))
        except Exception as exc:
            scene_errors.append(str(exc))
    track_sample: list[NormalizedTrackQuery] = []
    track_count = 0
    try:
        track_count = fetch_track_query_count(cur)
        if scenes:
            sample_rows = fetch_track_query_candidates_by_hybrid(cur, scenes[0].embedding_hybrid, 20)
            for r in sample_rows:
                try:
                    track_sample.append(normalize_track_query_row(r))
                except Exception:
                    pass
    except Exception as exc:
        scene_errors.append(f"track_query count failed: {exc}")
    report = build_input_validation_report(scenes, track_sample)
    report.update({
        "episode_nr":             episode_nr,
        "scene_query_rows":       len(scene_rows),
        "scene_query_normalized": len(scenes),
        "track_query_rows":       track_count,
        "track_sample_validated": len(track_sample),
        "recent_episode_numbers": fetch_recent_episode_numbers(cur, 10),
    })
    if scene_errors:
        report["errors"] = scene_errors[:20]
    return report


def render_run_matching_section(
    cur, ui_state: dict[str, Any], config: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Run matching and auto-save results to scene_music_matches_v6."""
    left, center, right = st.columns([1.2, 2.2, 1.2])
    with center:
        if not st.button("Run matching", type="primary", use_container_width=True):
            return st.session_state.get("last_result_bundle")
    episode_nr = int(ui_state.get("episode_nr", 0) or 0)
    if episode_nr <= 0:
        st.error("episode_nr must be > 0")
        return None
    merged_config = dict(config)
    merged_config["preview_top_n"] = int(ui_state.get("preview_top_n", config.get("preview_top_n", 5)))
    merged_config["rank_top_n"]    = int(ui_state.get("rank_top_n",    config.get("rank_top_n",    10)))

    progress_bar  = st.progress(0.0, text="Starting matcher…")
    total_scenes  = [0]

    def _on_progress(done: int, total: int, text: str) -> None:
        if total_scenes[0] == 0 and total > 0:
            total_scenes[0] = total
        frac = (done / max(1, total)) * 0.8
        progress_bar.progress(min(frac, 0.80), text=text)

    result_bundle = match_episode_v6(
        cur, episode_nr, merged_config,
        preview_only=False, on_progress=_on_progress,
    )
    progress_bar.progress(0.9, text="Saving results…")

    matches  = result_bundle.get("matches", [])
    inserted = save_episode_matches_v6(cur, episode_nr, matches, overwrite=True)
    cur.connection.commit()
    progress_bar.progress(1.0, text="Completed.")

    st.session_state["last_result_bundle"]  = result_bundle
    st.session_state["last_export_payload"] = export_episode_results_v6(
        result_bundle, OUTPUT_DIR, episode_nr
    )
    st.session_state["matcher_last_saved_episode"] = episode_nr

    left_m, center_m, right_m = st.columns([1.2, 2.2, 1.2])
    with center_m:
        st.markdown(
            f'<div class="card" style="text-align:center">'
            f'<span style="color:var(--green);font-weight:500">'
            f'✓ Saved/updated {inserted} rows</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.rerun()
    return result_bundle


def render_export_section(
    result_bundle: dict[str, Any],
    export_paths:  Optional[dict[str, Any]],
) -> None:
    """Render XLSX + PDF export download widgets."""
    payload = export_paths or st.session_state.get("last_export_payload")
    if not payload:
        return
    _header("Export", "XLSX and PDF downloads")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("XLSX full",  data=payload["full_bytes"],     file_name=payload["full_name"],     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c2:
        st.download_button("XLSX short", data=payload["short_bytes"],    file_name=payload["short_name"],    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c3:
        st.download_button("PDF full",   data=payload["pdf_full_bytes"], file_name=payload["pdf_full_name"], mime="application/pdf", use_container_width=True)
    with c4:
        st.download_button("PDF short",  data=payload["pdf_short_bytes"],file_name=payload["pdf_short_name"],mime="application/pdf", use_container_width=True)


def _page_matcher() -> None:
    _set_theme("blue")
    _page_header("🎵 Scene - Music Matcher", "episode selection · candidate retrieval · scoring")
   
    config = load_runtime_config()
    ensure_output_dir(OUTPUT_DIR)
    conn = None
    cur  = None
    try:
        conn = get_connection(DB_CONFIG)
        cur  = conn.cursor()
        ui_state      = render_episode_selection_section(config)
        result_bundle = render_run_matching_section(cur, ui_state, config)
    finally:
        if cur  is not None: cur.close()
        if conn is not None: conn.close()
    _pipe(["Select episode", "Retrieve candidates", "Score pairs", "Rank & penalize", "Save"], done=0)

def _render_score_breakdown(payload: dict[str, Any], title: str) -> None:
    """Render a structured visual score breakdown from a debug_compare payload."""
    if not payload:
        st.info("No score breakdown available.")
        return

    import pandas as pd

    comp = payload.get("component_scores", {})
    if not comp:
        comp = {
            **payload.get("embedding",     {}),
            **payload.get("semantic",      {}),
            **payload.get("targets",       {}),
            **payload.get("tag_selection", {}),
        }
    pen_grp = payload.get("penalties", {})

    def _g(key: str, default: float = 0.0) -> float:
        v = comp.get(key, pen_grp.get(key, default))
        try:
            return round(float(v), 4)
        except Exception:
            return default

    emb_s = _g("embedding_score")
    sem_s = _g("semantic_score")
    tgt_s = _g("targets_score")
    tag_s = _g("tag_selection_score")
    pen_t = _g("penalty_total")
    final = max(0.0, min(1.0, 0.40*emb_s + 0.25*sem_s + 0.20*tgt_s + 0.15*tag_s - pen_t))

    st.markdown(
        f'<div class="card score-formula">'
        f'<b>Final score</b> = '
        f'0.40 × <b class="accent">{emb_s:.3f}</b> (emb) + '
        f'0.25 × <b class="accent">{sem_s:.3f}</b> (sem) + '
        f'0.20 × <b class="accent">{tgt_s:.3f}</b> (tgt) + '
        f'0.15 × <b class="accent">{tag_s:.3f}</b> (tag) − '
        f'<b class="danger">{pen_t:.3f}</b> (pen) '
        f'= <b class="accent-lg">{final:.3f}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _score_bar("Embedding score",     emb_s, 0.40)
    _score_bar("Semantic score",      sem_s, 0.25)
    _score_bar("Targets score",       tgt_s, 0.20)
    _score_bar("Tag selection score", tag_s, 0.15)
    _score_bar("Penalty total",       pen_t, 1.00, color="var(--red)")

    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)

    with sc1:
        with st.container():
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
                'font-size:0.88rem;color:#6b4423;letter-spacing:0.09em;'
                'text-transform:uppercase;margin-bottom:0.6rem;">'
                'Embedding similarities</div>',
                unsafe_allow_html=True,
            )
            emb_rows = [
                {"component": lbl, "similarity": f"{_g(key):.3f}"}
                for key, lbl in [
                    ("hybrid_similarity",    "Hybrid (×0.50)"),
                    ("ensemble_similarity",  "CLAP ensemble (×0.20)"),
                    ("main_similarity",      "Main text (×0.15)"),
                    ("tags_similarity",      "Tags text (×0.10)"),
                    ("audio_similarity_aux", "Audio aux (×0.05)"),
                ]
            ]
            st.dataframe(pd.DataFrame(emb_rows), hide_index=True, use_container_width=True)

    with sc2:
        with st.container():
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
                'font-size:0.88rem;color:#6b4423;letter-spacing:0.09em;'
                'text-transform:uppercase;margin-bottom:0.6rem;">'
                'Penalty breakdown</div>',
                unsafe_allow_html=True,
            )
            pen_rows = []
            for key, lbl in [
                ("dialogue_conflict",              "Dialogue conflict"),
                ("duration_conflict",              "Duration conflict"),
                ("forbidden_tag_conflict",         "Forbidden tag conflict"),
                ("style_redundancy_penalty",       "Style redundancy"),
                ("same_track_consecutive_penalty", "Consecutive track"),
                ("missing_data_penalty",           "Missing data"),
            ]:
                v = _g(key)
                pen_rows.append({"penalty": lbl, "value": f"{v:.3f}", "active": "✗" if v > 0 else "—"})
            st.dataframe(pd.DataFrame(pen_rows), hide_index=True, use_container_width=True)

    sc3, sc4 = st.columns(2)
    with sc3:
        with st.container():
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
                'font-size:0.88rem;color:#6b4423;letter-spacing:0.09em;'
                'text-transform:uppercase;margin-bottom:0.6rem;">'
                'Semantic sub-scores</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame([
                {"component": "Emotion direction (×0.35)",  "score": f"{_g('emotion_match_score'):.3f}"},
                {"component": "Narrative function (×0.25)", "score": f"{_g('narrative_match_score'):.3f}"},
                {"component": "Weight profile (×0.20)",     "score": f"{_g('weight_profile_match_score'):.3f}"},
                {"component": "Dialogue safety (×0.20)",    "score": f"{_g('dialogue_match_score'):.3f}"},
            ]), hide_index=True, use_container_width=True)
    with sc4:
        with st.container():
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
                'font-size:0.88rem;color:#6b4423;letter-spacing:0.09em;'
                'text-transform:uppercase;margin-bottom:0.6rem;">'
                'Target sub-scores</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame([
                {"component": "Energy (×0.20)",           "score": f"{_g('energy_match_score'):.3f}"},
                {"component": "Tempo (×0.20)",            "score": f"{_g('tempo_match_score'):.3f}"},
                {"component": "Intensity shape (×0.20)",  "score": f"{_g('intensity_shape_match_score'):.3f}"},
                {"component": "Sound character (×0.25)",  "score": f"{_g('sound_character_match_score'):.3f}"},
                {"component": "Rhythm (×0.15)",           "score": f"{_g('rhythm_match_score'):.3f}"},
            ]), hide_index=True, use_container_width=True)

    with st.container():
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',\'Courier New\',monospace;'
            'font-size:0.88rem;color:#6b4423;letter-spacing:0.09em;'
            'text-transform:uppercase;margin-bottom:0.6rem;">'
            'Tag selection detail</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame([
            {"metric": "Should-have hit rate (×0.75)",       "value": f"{_g('should_have_hit_rate'):.3f}"},
            {"metric": "Must-not conflict rate (×0.25 pen)", "value": f"{_g('must_not_conflict_rate'):.3f}"},
            {"metric": "Tag selection score",                 "value": f"{_g('tag_selection_score'):.3f}"},
        ]), hide_index=True, use_container_width=True)


def _load_tag_vocab() -> dict[str, list[str]]:
    """Load tag vocabulary from TAGS_FILE."""
    try:
        raw = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
        return {cat: sorted(data.get("tags", {}).keys()) for cat, data in raw.items()}
    except Exception:
        return {}


def _run_manual_track_search(
    cur,
    name_query:  str,
    sel_tags:    list[str],
    starts_with: bool = False,
) -> None:
    """Search track_query and store results in scr_manual_results."""
    import pandas as pd
    try:
        conditions: list[str] = []
        params:     list[Any] = []
        if name_query:
            if starts_with:
                # Prefix mode: filename only — no OR with semantic_title_en,
                # which would pull in any track whose title contains the letter.
                conditions.append("filename ILIKE %s")
                params.append(f"{name_query}%")
            else:
                # Contains mode: search both filename and semantic title.
                conditions.append("(filename ILIKE %s OR semantic_title_en ILIKE %s)")
                params.extend([f"%{name_query}%", f"%{name_query}%"])
        for tag in sel_tags:
            conditions.append("track_tag_selection->'should_have_tags' @> %s::jsonb")
            params.append(json.dumps([tag]))
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cur.execute(
            f"""
            SELECT id, filepath, filename, duration_sec, bpm, musical_key,
                   semantic_title_en, tags_summary_en, updated_at
            FROM track_query
            {where}
            ORDER BY updated_at DESC
            """,
            params,
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        st.session_state.scr_manual_results = pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"Search error: {exc}")
        st.session_state.scr_manual_results = pd.DataFrame()


def _page_pdf() -> None:
    _set_theme("blue")
    _page_header("📝 Scene - Music Reports", "select episode · generate · download XLSX and PDF")
    st.markdown(
        '<div class="card"><div class="card-title">Match result files</div>'
        '<div class="card-body">'
        'Select an episode from the list below. Episodes are loaded from '
        '<code>scene_music_matches_v6</code>. Once selected, generate the four '
        'download files — XLSX full, XLSX short, PDF full, PDF short.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    episodes: list[int] = []
    db_error = ""
    try:
        with _db_connect(DB_CONFIG) as (conn, cur):
            episodes = fetch_pdf_episode_numbers(cur)
    except Exception as exc:
        db_error = str(exc)

    if db_error:
        st.markdown(_badge("DB error", "err"), unsafe_allow_html=True)
        st.error(db_error)
        return
    if not episodes:
        st.info("No episodes found in scene_music_matches_v6. Run Scene Music Matcher first.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    left, center, right = st.columns([1.2, 2.2, 1.2])
    with center:
        selected_episode = st.selectbox(
            "Episodes", options=episodes,
            format_func=lambda x: f"Episode {x}",
            key="pdf_episode_select",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button(
            "⚙  Generate export files", type="primary",
            key="pdf_generate_btn", use_container_width=True,
        )

    cache_key = f"pdf_payload_{selected_episode}"
    cached    = st.session_state.get(cache_key)

    if generate_clicked or cached is None:
        if generate_clicked:
            with st.spinner(f"Building export files for Episode {selected_episode}…"):
                try:
                    with _db_connect(DB_CONFIG) as (conn, cur):
                        matches = fetch_matches_as_ranked_v6(cur, selected_episode)
                except Exception as exc:
                    st.error(f"DB error while loading matches: {exc}")
                    return
                if not matches:
                    st.warning(f"No match rows found for Episode {selected_episode}.")
                    return
                result_bundle = {"matches": matches, "episode_nr": selected_episode}
                payload = export_episode_results_v6(result_bundle, OUTPUT_DIR, selected_episode)
                st.session_state[cache_key] = payload
                cached = payload

    if cached:
        ep = selected_episode
        st.markdown(
            _badge(f"Episode {ep}  ·  {len(cached.get('full_rows', []))} rows", "ok"),
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, card_title, card_body, dl_label, key_sfx, data_key, fname_key, mime in [
            (c1, "XLSX — full",  "All component scores, penalties and metadata.",
             "⬇  XLSX full",  "xlsx_full",  "full_bytes",      "full_name",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            (c2, "XLSX — short", "Top-3 tracks per scene: filename, title, score.",
             "⬇  XLSX short", "xlsx_short", "short_bytes",     "short_name",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            (c3, "PDF — full",   "All ranked tracks per scene with complete scores.",
             "⬇  PDF full",   "pdf_full",   "pdf_full_bytes",  "pdf_full_name",
             "application/pdf"),
            (c4, "PDF — short",  "Top-3 tracks per scene in a compact table.",
             "⬇  PDF short",  "pdf_short",  "pdf_short_bytes", "pdf_short_name",
             "application/pdf"),
        ]:
            with col:
                st.markdown(
                    f'<div class="card"><div class="card-title">{card_title}</div>'
                    f'<div class="card-body">{card_body}</div></div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    dl_label, data=cached[data_key], file_name=cached[fname_key],
                    mime=mime, key=f"dl_{key_sfx}_{ep}", use_container_width=True,
                )
    else:
        left2, center2, right2 = st.columns([1.2, 2.2, 1.2])
        with center2:
            st.markdown(
                '<div class="card" style="text-align:center;font-family:var(--mono);'
                'font-size:0.75rem;color:var(--txt-3)">'
                'Select an episode and click ⚙ Generate export files to activate the download buttons.'
                '</div>',
                unsafe_allow_html=True,
            )


def _page_scoring() -> None:
    _set_theme("blue")
    _page_header("🎶 Scene - Music Scoring",
                 "episode · scene · theme selection · scene detail · ranked matches")

    try:
        conn = get_connection(DB_CONFIG)
        cur  = conn.cursor()
    except Exception as exc:
        st.error(f"DB connection failed: {exc}")
        return

    try:
        episodes: list[int] = []
        try:
            episodes = fetch_scoring_episode_numbers(cur)
        except Exception as exc:
            st.error(f"Could not load episodes: {exc}")
            return

        if not episodes:
            st.info("No episodes found. Run Scene Music Matcher and write results to DB first.")
            return

        st.markdown("<br>", unsafe_allow_html=True)

        col_ep, col_sc, col_th = st.columns(3)
        with col_ep:
            sel_episode = st.selectbox(
                "Episodes", options=episodes,
                format_func=lambda x: f"Ep. {x}", key="scr_episode",
            )

        scenes: list[int] = []
        try:
            scenes = fetch_scoring_scene_numbers(cur, sel_episode)
        except Exception as exc:
            st.error(f"Could not load scenes: {exc}")
            return
        if not scenes:
            st.warning(f"No scenes for Episode {sel_episode}.")
            return

        with col_sc:
            sel_scene = st.selectbox(
                "Scenes", options=scenes,
                format_func=lambda x: f"Sc. {x}", key="scr_scene",
            )

        themes: list[int] = []
        try:
            themes = fetch_scoring_theme_numbers(cur, sel_episode, sel_scene)
        except Exception as exc:
            st.error(f"Could not load themes: {exc}")
            return
        if not themes:
            st.warning(f"No themes for Ep. {sel_episode} / Sc. {sel_scene}.")
            return

        with col_th:
            sel_theme = st.selectbox(
                "Themes", options=themes,
                format_func=lambda x: f"Th. {x}", key="scr_theme",
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        current_selection = (sel_episode, sel_scene, sel_theme)
        if st.session_state.get("scr_last_selection") != current_selection:
            st.session_state.scr_last_selection          = current_selection
            st.session_state.scr_selected_track_id       = None
            st.session_state.scr_selected_scene_query_id = None
            st.session_state.scr_selected_rank           = None
            st.session_state.scr_selected_score          = None
            st.session_state.scr_selected_explanation    = None

        scene_row = fetch_scene_query_by_episode_scene_theme(
            cur, sel_episode, sel_scene, sel_theme
        )

        if scene_row is None:
            st.warning(
                f"No scene_query row found for "
                f"Episode {sel_episode} / Scene {sel_scene} / Theme {sel_theme}."
            )
        else:
            _header(
                f"Episode {sel_episode}  ·  Scene {sel_scene}  ·  Theme {sel_theme}",
                str(scene_row.get("theme_title_en") or ""),
            )
            for label, key in [
                ("Theme text",          "theme_txt"),
                ("Description",         "description_en"),
                ("Segmentation reason", "segmentation_reason"),
            ]:
                value = scene_row.get(key)
                if not value:
                    continue
                safe_html = (
                    _safe(value)
                    .replace("\r\n", "<br>")
                    .replace("\r", "<br>")
                    .replace("\n", "<br>")
                )
                label_div = (
                    f'<div style="margin-bottom:0.3rem;font-family:var(--mono);'
                    f'font-size:1.00rem;color:var(--txt-3);letter-spacing:0.08em;'
                    f'text-transform:uppercase">{_safe(label)}</div>'
                )
                if key == "theme_txt":
                    content_div = (
                        f'<div class="card" style="max-height:260px;overflow-y:auto">'
                        f'{safe_html}</div>'
                    )
                else:
                    content_div = (
                        f'<div style="font-family:var(--sans);font-size:0.88rem;'
                        f'color:var(--txt-2);line-height:1.65;margin-bottom:1.1rem">'
                        f'{safe_html}</div>'
                    )
                st.markdown(label_div + content_div, unsafe_allow_html=True)

            EMBED_KEYS = {"embedding_main", "embedding_tags", "embedding_clap_ensemble", "embedding_hybrid"}
            PROSE_KEYS = {"theme_title_en", "theme_txt", "description_en", "segmentation_reason"}
            JSONB_KEYS = {"scene_music_semantics", "scene_music_targets", "scene_tag_selection", "clap_prompt_ensemble"}

            scalar_rows = [
                {"field": k, "value": str(v) if v is not None else ""}
                for k, v in scene_row.items()
                if k not in EMBED_KEYS and k not in PROSE_KEYS and k not in JSONB_KEYS
            ]
            if scalar_rows:
                import pandas as pd
                with st.expander("Scene metadata", expanded=True):
                    st.dataframe(pd.DataFrame(scalar_rows), hide_index=True, use_container_width=True)

            jcol1, jcol2 = st.columns(2)
            with jcol1:
                sem_val = scene_row.get("scene_music_semantics")
                if sem_val:
                    with st.expander("Music semantics", expanded=False):
                        st.json(sem_val if isinstance(sem_val, dict) else {})
            with jcol2:
                tgt_val = scene_row.get("scene_music_targets")
                if tgt_val:
                    with st.expander("Music targets", expanded=False):
                        st.json(tgt_val if isinstance(tgt_val, dict) else {})
            jcol3, jcol4 = st.columns(2)
            with jcol3:
                tag_val = scene_row.get("scene_tag_selection")
                if tag_val:
                    with st.expander("Tag selection", expanded=False):
                        st.json(tag_val if isinstance(tag_val, dict) else {})
            with jcol4:
                clap_val = scene_row.get("clap_prompt_ensemble")
                if clap_val:
                    with st.expander("CLAP prompt ensemble", expanded=False):
                        st.json(clap_val if isinstance(clap_val, dict) else {})

        st.markdown("<hr>", unsafe_allow_html=True)

        scr_mode = st.radio(
            "Track selection mode",
            options=["Ranked matches", "Manual selection"],
            index=0 if st.session_state.get("scr_mode", "Ranked matches") == "Ranked matches" else 1,
            horizontal=True, key="scr_mode_radio",
        )
        st.session_state.scr_mode = scr_mode

        if st.session_state.get("scr_mode_last") != (current_selection, scr_mode):
            st.session_state["scr_mode_last"]              = (current_selection, scr_mode)
            st.session_state.scr_selected_track_id         = None
            st.session_state.scr_selected_scene_query_id   = None
            st.session_state.scr_selected_rank             = None
            st.session_state.scr_selected_score            = None
            st.session_state.scr_selected_explanation      = None

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Branch A: Ranked matches ──────────────────────────────────────────
        if scr_mode == "Ranked matches":
            _header("Ranked matches", "select a row to load the track player and detail")
            match_rows: list[dict[str, Any]] = []
            try:
                match_rows = fetch_matches_for_theme(cur, sel_episode, sel_scene, sel_theme)
            except Exception as exc:
                st.error(f"Could not load match rows: {exc}")

            if not match_rows:
                st.info("No match rows found for this episode / scene / theme combination.")
            else:
                import pandas as pd
                df = pd.DataFrame(match_rows)
                float_cols = list(df.select_dtypes(include=["float64", "float32"]).columns)
                col_cfg    = {c: st.column_config.NumberColumn(format="%.2f") for c in float_cols}
                priority   = [
                    "rank_position", "track_filename", "track_semantic_title_en",
                    "duration_sec", "bpm", "musical_key",
                    "final_score", "embedding_score", "semantic_score",
                    "targets_score", "tag_selection_score", "dialogue_score",
                    "main_similarity", "tags_similarity", "ensemble_similarity",
                    "hybrid_similarity", "audio_similarity_aux",
                    "penalty_total", "dialogue_conflict", "duration_conflict",
                    "forbidden_tag_conflict", "style_redundancy_penalty",
                    "same_track_consecutive_penalty", "missing_data_penalty",
                    "style_signature", "dialogue_safe_applied", "track_query_id",
                ]
                ordered = [c for c in priority if c in df.columns]
                rest    = [c for c in df.columns if c not in ordered]
                df      = df[ordered + rest]
                event   = st.dataframe(
                    df, hide_index=True, use_container_width=True,
                    on_select="rerun", selection_mode="single-row",
                    column_config=col_cfg,
                    key=f"scr_matches_{sel_episode}_{sel_scene}_{sel_theme}",
                )
                st.caption(
                    f"{len(df)} ranked track{'s' if len(df) != 1 else ''} for "
                    f"Episode {sel_episode} · Scene {sel_scene} · Theme {sel_theme}"
                )
                selected_rows = event.selection.rows if event and event.selection else []
                if selected_rows:
                    selected_idx = selected_rows[0]
                    if 0 <= selected_idx < len(df):
                        row_data = df.iloc[selected_idx]
                        st.session_state.scr_selected_track_id       = int(row_data["track_query_id"])
                        st.session_state.scr_selected_scene_query_id = int(row_data["scene_query_id"])
                        st.session_state.scr_selected_rank           = int(row_data.get("rank_position", 0))
                        st.session_state.scr_selected_score          = float(row_data.get("final_score", 0.0))
                        raw_expl = row_data.get("match_explanation")
                        if isinstance(raw_expl, str):
                            try:
                                raw_expl = json.loads(raw_expl)
                            except Exception:
                                raw_expl = {}
                        st.session_state.scr_selected_explanation = (
                            raw_expl if isinstance(raw_expl, dict) else {}
                        )

        # ── Branch B: Manual selection ────────────────────────────────────────
        else:
            _header("Manual track selection",
                    "search by name or tag — select a track to play and analyse")
            tag_vocab = _load_tag_vocab()
            with st.expander("Search filters", expanded=True):
                fc1, fc2 = st.columns([2, 3])
                with fc1:
                    fn1, fn2 = st.columns([5, 1])
                    with fn1:
                        _ver = st.session_state.get("scr_name_input_ver", 0)
                        name_query = st.text_input(
                            "Track name",
                            placeholder="e.g. tension, piano, cue_042 …",
                            help="Searches filename and semantic_title_en (case-insensitive).",
                            key=f"scr_name_{_ver}",
                        )
                        st.session_state.scr_manual_name = name_query
                    with fn2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("✕ Clear", key="scr_manual_clear",
                                     help="Clear the track-name field"):
                            st.session_state.scr_manual_name     = ""
                            st.session_state.scr_name_input_ver  = _ver + 1
                            st.rerun()
                with fc2:
                    sel_cats = st.multiselect(
                        "Tag categories", options=sorted(tag_vocab.keys()),
                        key="scr_manual_cats", help="Select categories to filter by tag.",
                    )
                    available_tags: list[str] = sorted(
                        set(t for c in sel_cats for t in tag_vocab.get(c, []))
                    )
                    sel_tags = st.multiselect(
                        "Tags (within selected categories)", options=available_tags,
                        key="scr_manual_tags", disabled=not available_tags,
                        help="Track must contain ALL selected tags in its should_have_tags list.",
                    )
                # Checkbox outside all column contexts — committed to session state
                # before the Search button executes (no column-buffer delay).
                st.checkbox(
                    "Starts with",
                    key="scr_starts_with_chk",
                )
                st.session_state.scr_manual_starts_with = bool(
                    st.session_state.get("scr_starts_with_chk", False)
                )
                if st.button("🔍  Search", type="primary", key="scr_manual_search"):
                    _ver2         = st.session_state.get("scr_name_input_ver", 0)
                    _name_query   = str(st.session_state.get(f"scr_name_{_ver2}", "")).strip()
                    _starts_with  = bool(st.session_state.get("scr_starts_with_chk", False))
                    _sel_tags     = list(st.session_state.get("scr_manual_tags", []))
                    if not _name_query and not _sel_tags:
                        st.warning("Enter a name or select at least one tag to search.")
                    else:
                        st.session_state.scr_selected_track_id       = None
                        st.session_state.scr_selected_scene_query_id = None
                        st.session_state.scr_selected_rank           = None
                        st.session_state.scr_selected_score          = None
                        st.session_state.scr_selected_explanation    = None
                        _run_manual_track_search(cur, _name_query, _sel_tags,
                                                 starts_with=_starts_with)

            import pandas as pd
            results = st.session_state.get("scr_manual_results")
            if results is not None:
                if len(results) == 0:
                    st.info("No tracks matched the search criteria.")
                else:
                    n = len(results)
                    st.markdown(
                        _badge(f"{n} track{'s' if n != 1 else ''} found", "ok"),
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    display_cols = [
                        c for c in
                        ["id", "filename", "bpm", "musical_key",
                         "semantic_title_en", "tags_summary_en", "updated_at"]
                        if c in results.columns
                    ]
                    man_event = st.dataframe(
                        results[display_cols], use_container_width=True, hide_index=True,
                        on_select="rerun", selection_mode="single-row",
                        column_config={
                            "id":                st.column_config.NumberColumn("ID", width="small"),
                            "filename":          st.column_config.TextColumn("Filename"),
                            "bpm":               st.column_config.NumberColumn("BPM", format="%d"),
                            "musical_key":       st.column_config.TextColumn("Key", width="small"),
                            "semantic_title_en": st.column_config.TextColumn("Semantic title"),
                            "tags_summary_en":   st.column_config.TextColumn("Tags summary"),
                            "updated_at":        st.column_config.DatetimeColumn("Updated", format="DD MMM YYYY"),
                        },
                        key="scr_manual_results_df",
                    )
                    man_selected = (
                        man_event.selection.rows
                        if man_event and hasattr(man_event, "selection") and man_event.selection
                        else []
                    )
                    if man_selected:
                        man_idx = man_selected[0]
                        man_tid = int(results.iloc[man_idx]["id"])
                        if man_tid != st.session_state.get("scr_selected_track_id"):
                            st.session_state.scr_selected_track_id       = man_tid
                            st.session_state.scr_selected_rank           = None
                            st.session_state.scr_selected_score          = None
                            st.session_state.scr_selected_explanation    = None
                            sq_row = fetch_scene_query_by_episode_scene_theme(
                                cur, sel_episode, sel_scene, sel_theme
                            )
                            st.session_state.scr_selected_scene_query_id = (
                                int(sq_row["id"]) if sq_row else None
                            )
                    st.markdown("<br>", unsafe_allow_html=True)
                    left_ap, center_ap, right_ap = st.columns([1.2, 2.2, 1.2])
                    with center_ap:
                        st.session_state.scr_autoplay = st.checkbox(
                            "Auto play — start playing immediately when a track is selected",
                            value=st.session_state.get("scr_autoplay", False),
                            key="scr_autoplay_chk",
                        )

        # Auto play toggle — Ranked matches mode
        if scr_mode == "Ranked matches":
            st.markdown("<br>", unsafe_allow_html=True)
            left_ap, center_ap, right_ap = st.columns([1.2, 2.2, 1.2])
            with center_ap:
                st.session_state.scr_autoplay = st.checkbox(
                    "Auto play — start playing immediately when a track is selected",
                    value=st.session_state.get("scr_autoplay", False),
                    key="scr_autoplay_chk",
                )

        # Player + track detail — shared across both modes
        track_id = st.session_state.get("scr_selected_track_id")
        if track_id is not None:
            track = None
            try:
                track = fetch_track_detail(cur, track_id)
            except Exception as exc:
                st.error(f"Could not load track detail: {exc}")

            if track:
                st.markdown("<hr>", unsafe_allow_html=True)
                _header(
                    str(track.get("filename") or f"Track {track_id}"),
                    f"track_query id: {track_id}",
                )

                if scr_mode == "Ranked matches":
                    sel_rank  = st.session_state.get("scr_selected_rank")
                    sel_score = st.session_state.get("scr_selected_score")
                    if sel_rank is not None and sel_score is not None:
                        st.markdown(
                            f'<div class="card-body">'
                            f'<b>Rank position:</b> <b style="color:var(--accent)">{sel_rank}</b>'
                            f'&nbsp;&nbsp;&nbsp;'
                            f'<b>Final score:</b> <b style="color:var(--accent)">{sel_score:.2f}</b>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                filepath = str(track.get("filepath") or "")
                filename = str(track.get("filename") or "")
                audio_bytes, local_path, audio_error = get_audio_for_player(filepath, filename)

                left_pl, center_pl, right_pl = st.columns([1.2, 2.2, 1.2])
                with center_pl:
                    if audio_error:
                        st.warning(audio_error)
                    elif audio_bytes is not None:
                        st.audio(audio_bytes, autoplay=st.session_state.scr_autoplay)
                    elif local_path:
                        st.audio(local_path, autoplay=st.session_state.scr_autoplay)

                ca, cb = st.columns(2)
                with ca:
                    source_label = "R2 bucket" if CLOUD_MUSIC else "Local path"
                    source_value = BUCKET_NAME if CLOUD_MUSIC else _safe(track.get("filepath"))
                    st.markdown(
                        f'<div class="card">'
                        f'<div class="card-title">{_safe(track.get("filename"))}</div>'
                        f'<div class="card-body">'
                        f'<b>ID:</b> {_safe(track.get("id"))}<br>'
                        f'<b>{source_label}:</b> {_safe(source_value)}<br>'
                        f'<b>Duration:</b> {_safe(track.get("duration_sec"))} s<br>'
                        f'<b>BPM:</b> {_safe(track.get("bpm"))}<br>'
                        f'<b>Key:</b> {_safe(track.get("musical_key"))}<br>'
                        f'<b>Updated:</b> {_safe(track.get("updated_at"))}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                with cb:
                    st.markdown(
                        f'<div class="card">'
                        f'<div class="card-title">Semantic contract</div>'
                        f'<div class="card-body">'
                        f'<b>Title:</b> {_safe(track.get("semantic_title_en"))}<br>'
                        f'<b>Description:</b> {_safe(track.get("description_en"))}<br>'
                        f'<b>Tags summary:</b> {_safe(track.get("tags_summary_en"))}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                te1, te2 = st.columns(2)
                with te1:
                    with st.expander("Tag selection", expanded=False):
                        tag_sel = track.get("track_tag_selection") or {}
                        if isinstance(tag_sel, str):
                            tag_sel = json.loads(tag_sel)
                        should  = tag_sel.get("should_have_tags", []) or []
                        mustnot = tag_sel.get("must_not_tags",    []) or []
                        chips   = " ".join(
                            f'<span class="badge b-info">{_safe(x)}</span>'
                            for x in should
                        )
                        st.markdown(f"**Should have:** {chips}", unsafe_allow_html=True)
                        st.markdown(
                            f"**Must not:** `{'`, `'.join(_safe(x) for x in mustnot)}`"
                            if mustnot else "**Must not:** —"
                        )
                with te2:
                    with st.expander("Audio analysis", expanded=False):
                        aa = track.get("audio_analysis") or {}
                        if isinstance(aa, str):
                            aa = json.loads(aa)
                        st.json(aa)

                te3, te4 = st.columns(2)
                with te3:
                    with st.expander("Music semantics & targets", expanded=False):
                        sem = track.get("track_music_semantics") or {}
                        if isinstance(sem, str):
                            sem = json.loads(sem)
                        tgt = track.get("track_music_targets") or {}
                        if isinstance(tgt, str):
                            tgt = json.loads(tgt)
                        st.json(sem)
                        st.json(tgt)
                with te4:
                    with st.expander("Segmentation metadata", expanded=False):
                        seg = track.get("segmentation") or {}
                        if isinstance(seg, str):
                            seg = json.loads(seg)
                        st.json(seg)

                te5, te6 = st.columns(2)
                with te5:
                    with st.expander("Prompt ensemble", expanded=False):
                        ens = track.get("track_clap_prompt_ensemble") or {}
                        if isinstance(ens, str):
                            ens = json.loads(ens)
                        for k, v in (ens.items() if isinstance(ens, dict) else {}.items()):
                            st.markdown(
                                f'<div class="s-sub">{_safe(k)}</div>'
                                f'<div class="card-body" style="margin-bottom:8px">{_safe(v)}</div>',
                                unsafe_allow_html=True,
                            )

                st.markdown("<hr>", unsafe_allow_html=True)

                if scr_mode == "Ranked matches":
                    with st.expander(
                        "📋  Score breakdown — stored  (why the algorithm ranked this track)",
                        expanded=True,
                    ):
                        stored_expl = st.session_state.get("scr_selected_explanation")
                        if stored_expl:
                            st.markdown(
                                '<div class="s-sub">'
                                'Exact scores from the ranking run — includes contextual penalties '
                                '(style redundancy, consecutive track) that cannot be reproduced in isolation.'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                            _render_score_breakdown(stored_expl, "Stored breakdown")
                        else:
                            st.info("No stored match_explanation found for this row.")

                scene_query_id_sel = st.session_state.get("scr_selected_scene_query_id")
                with st.expander(
                    "⚙  Score breakdown — live recompute  (current weights)",
                    expanded=False,
                ):
                    st.markdown(
                        '<div class="s-sub">'
                        'Re-runs the full scoring engine on the current config weights. '
                        'Contextual penalties (style redundancy, consecutive track) are '
                        'computed without episode context and will show 0.'
                        + (" — use the stored breakdown above for those."
                           if scr_mode == "Ranked matches" else "")
                        + '</div>',
                        unsafe_allow_html=True,
                    )
                    if scene_query_id_sel and track_id:
                        recompute_key = f"scr_live_{scene_query_id_sel}_{track_id}"
                        if st.button(
                            "⚙  Recompute with current weights",
                            key=f"btn_{recompute_key}", type="primary",
                        ):
                            config_live  = load_runtime_config()
                            live_payload = debug_compare_scene_track(
                                cur, scene_query_id_sel, track_id, config_live
                            )
                            st.session_state[recompute_key] = live_payload
                        live_payload = st.session_state.get(recompute_key)
                        if live_payload:
                            if "error" in live_payload:
                                st.error(live_payload["error"])
                            else:
                                _render_score_breakdown(live_payload, "Live breakdown")
                    else:
                        st.info(
                            "Select a track to enable live recompute."
                            if scr_mode == "Manual selection"
                            else "Select a row in the Ranked matches table to enable live recompute."
                        )

    finally:
        if cur  is not None: cur.close()
        if conn is not None: conn.close()


## ── Import pipeline page ───────────────────────────────────────────────
def _page_importer() -> None:
    _page_header("▶ Run Importer", "music directory · start pipeline · live log")

    active = st.session_state.import_stage if st.session_state.import_running else -1
    done   = 5 if st.session_state.import_done else 0
    _pipe(
        ["Scan files", "Preprocess (CPU)", "CLAP encode (GPU)", "Tag + contract", "DB upsert"],
        done=done, active=active,
    )

    with st.expander("⚙  Parameters", expanded=True):
        _music_dir_value = BUCKET_NAME if CLOUD_MUSIC else str(MUSIC_DIR)
        _music_dir_label = (
            "Music source — R2 bucket (BUCKET_NAME)"
            if CLOUD_MUSIC else
            "Music directory (MUSIC_DIR)"
        )
        _music_dir_help = (
            "Cloud mode: tracks are streamed from this Cloudflare R2 bucket. "
            "Edit BUCKET_NAME in .env to change."
            if CLOUD_MUSIC else
            "Local mode: audio files are scanned from this folder. "
            "Edit MUSIC_DIR in .env to change."
        )
        st.text_input(
            _music_dir_label,
            value=_music_dir_value,
            disabled=True,
            help=_music_dir_help,
        )

        st.checkbox(
            "Skip already-indexed tracks (SKIP_EXISTING)",
            value=st.session_state.p_skip,
            key="p_skip",
            help="When checked, files whose filepath already exists in track_query "
                 "are skipped. Uncheck to force a full re-import of all files.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.68rem;'
            'color:var(--txt-3);letter-spacing:0.06em;text-transform:uppercase;'
            'margin-bottom:6px">Locked parameters — edit via .env file</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("Tag vocabulary (TAGS_FILE)",       value=str(TAGS_FILE),    disabled=True)
            st.text_input("CLAP model (CLAP_MODEL_NAME)",     value=CLAP_MODEL_NAME,   disabled=True)
            st.number_input("CPU workers (WORKERS)",          value=WORKERS,           disabled=True)
        with c2:
            st.number_input("Chunk size (CHUNK_SIZE)",        value=CHUNK_SIZE,        disabled=True)
            st.number_input("GPU batch size (BATCH_SIZE)",    value=BATCH_SIZE,        disabled=True)
            st.number_input("Audio CLAP batch (AUDIO_BATCH_SIZE)", value=AUDIO_BATCH_SIZE, disabled=True)
        with c3:
            st.number_input("Text CLAP batch (TEXT_BATCH_SIZE)",   value=TEXT_BATCH_SIZE,  disabled=True)
            st.number_input("Segments per track (NUM_SEGMENTS)",    value=NUM_SEGMENTS,     disabled=True)
            st.number_input("Segment duration, s (SEGMENT_SECONDS)", value=SEGMENT_SECONDS, disabled=True)

        st.caption(
            "Skip already-indexed tracks is editable. "
            "Music directory is set from BUCKET_NAME (cloud) or MUSIC_DIR (local). "
            "All other parameters are locked to the values from the environment at startup — "
            "edit your .env file and restart the app to change them."
        )

    lbl = "▶  Start Import" if not st.session_state.import_running else "⏳  Running…"
    _bl, _bc, _br = st.columns([3, 1, 3])
    with _bc:
        if st.button(lbl, type="primary",
                     disabled=st.session_state.import_running,
                     use_container_width=True):
            _music_src = BUCKET_NAME if CLOUD_MUSIC else str(MUSIC_DIR)
            _start_import(
                music_dir        = _music_src,
                tags_file        = str(TAGS_FILE),
                clap_model_name  = CLAP_MODEL_NAME,
                workers          = WORKERS,
                chunk_size       = CHUNK_SIZE,
                batch_size       = BATCH_SIZE,
                audio_batch_size = AUDIO_BATCH_SIZE,
                text_batch_size  = TEXT_BATCH_SIZE,
                num_segments     = NUM_SEGMENTS,
                segment_seconds  = SEGMENT_SECONDS,
                skip_existing    = bool(st.session_state.p_skip),
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    _header("Console", "import log")
    _render_log(st.session_state.import_log)

    if st.session_state.import_done and st.session_state.import_stats:
        s = st.session_state.import_stats
        st.success("Import completed.")
        _metrics([
            ("Inserted", str(s.get("inserted", "–"))),
            ("Updated",  str(s.get("updated",  "–"))),
            ("Failed",   str(s.get("failed",   "–"))),
            ("In DB",    str(s.get("total",    "–"))),
        ])

    if st.session_state.import_running:
        st.progress(0.5, text="Pipeline running…")
        time.sleep(2)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# GUI DB QUERIES  (inspector / search — no embedding columns)
# ══════════════════════════════════════════════════════════════════════════════

def _load_inspector_track(track_id: int) -> None:
    """Load full record for one track (no embeddings) into session state."""
    try:
        conn = _imp_get_connection()
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT id, filepath, filename, duration_sec, bpm, musical_key,
                   semantic_title_en, description_en, tags_summary_en,
                   track_music_semantics, track_music_targets,
                   track_tag_selection, track_clap_prompt_ensemble,
                   audio_analysis, segmentation, updated_at
            FROM track_query WHERE id = %s
            """,
            (track_id,),
        )
        row  = cur.fetchone()
        cols = [d[0] for d in cur.description]
        cur.close()
        conn.close()
        if row:
            t                     = dict(zip(cols, row))
            analysis              = t.get("audio_analysis") or {}
            seg                   = t.get("segmentation")   or {}
            t["rms_mean"]         = round(float(analysis.get("rms", 0.0)), 6)
            t["zcr_mean"]         = round(float(analysis.get("zero_crossing_rate", 0.0)), 6)
            t["pipeline_version"] = seg.get("pipeline_version", "–")
            st.session_state.insp_track = t
        else:
            st.warning(f"No track with id={track_id}")
            st.session_state.insp_track = None
    except Exception as exc:
        st.error(f"DB error: {exc}")


def _run_inspector_search(
    name_query:  str,
    sel_tags:    list[str],
    starts_with: bool = False,
) -> None:
    """Execute search — no row limit. Empty args return all tracks."""
    import pandas as pd
    try:
        conn = _imp_get_connection()
        cur  = conn.cursor()
        conditions: list[str] = []
        params: list[Any]     = []
        if name_query:
            if starts_with:
                conditions.append("filename ILIKE %s")
                params.append(f"{name_query}%")
            else:
                conditions.append(
                    "(filename ILIKE %s OR semantic_title_en ILIKE %s)"
                )
                params.extend([f"%{name_query}%", f"%{name_query}%"])
        for tag in sel_tags:
            conditions.append("track_tag_selection->'should_have_tags' @> %s::jsonb")
            params.append(json.dumps([tag]))
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cur.execute(
            f"""
            SELECT id, filepath, filename, duration_sec, bpm, musical_key,
                   semantic_title_en, tags_summary_en, updated_at
            FROM track_query {where} ORDER BY filename ASC
            """,
            params,
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        cur.close()
        conn.close()
        st.session_state.insp_over_limit = False
        df = pd.DataFrame(rows, columns=cols)
        st.session_state.insp_results   = df
        st.session_state.insp_last_name = name_query
        st.session_state.insp_last_tags = list(sel_tags)
        if not df.empty:
            _first_id = int(df.iloc[0]["id"])
            st.session_state.insp_selected_id = _first_id
            _load_inspector_track(_first_id)
        else:
            st.session_state.insp_track       = None
            st.session_state.insp_selected_id = None
    except Exception as exc:
        st.error(f"Search error: {exc}")
        st.session_state.insp_results    = pd.DataFrame()
        st.session_state.insp_over_limit = False


def _render_inspector_track(t: dict[str, Any]) -> None:
    """Render the full track detail card."""
    ca, cb = st.columns(2)
    with ca:
        st.markdown(
            f'<div class="card"><div class="card-title">{_safe(t.get("filename"))}</div>'
            f'<div class="card-body">'
            f'<b>ID:</b> {_safe(t.get("id"))}<br>'
            f'<b>Path:</b> {_safe(t.get("filepath"))}<br>'
            f'<b>Duration:</b> {_safe(t.get("duration_sec"))} s<br>'
            f'<b>BPM:</b> {_safe(t.get("bpm"))}<br>'
            f'<b>Key:</b> {_safe(t.get("musical_key"))}<br>'
            f'<b>RMS:</b> {_safe(t.get("rms_mean"))}<br>'
            f'<b>ZCR:</b> {_safe(t.get("zcr_mean"))}<br>'
            f'<b>Pipeline:</b> {_safe(t.get("pipeline_version"))}<br>'
            f'<b>Updated:</b> {_safe(t.get("updated_at"))}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with cb:
        st.markdown(
            f'<div class="card"><div class="card-title">Semantic contract</div>'
            f'<div class="card-body">'
            f'<b>Title:</b> {_safe(t.get("semantic_title_en"))}<br>'
            f'<b>Description:</b> {_safe(t.get("description_en"))}<br>'
            f'<b>Tags summary:</b> {_safe(t.get("tags_summary_en"))}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("Tag selection", expanded=False):
        tag_sel = t.get("track_tag_selection") or {}
        if isinstance(tag_sel, str):
            tag_sel = json.loads(tag_sel)
        should  = tag_sel.get("should_have_tags", [])
        mustnot = tag_sel.get("must_not_tags",    [])
        conf    = tag_sel.get("confidence_profile", {})
        chips = " ".join(
            f'<span style="display:inline-flex;padding:2px 9px;border-radius:12px;'
            f'background:var(--accent-bg);border:1px solid var(--accent-chip);'
            f'font-family:var(--mono);font-size:0.67rem;color:var(--accent);'
            f'margin:2px">{_safe(x)}</span>'
            for x in should
        )
        st.markdown(f"**Should have:** {chips}", unsafe_allow_html=True)
        st.markdown(
            f"**Must not:** `{'`, `'.join(_safe(x) for x in mustnot)}`"
        )
        st.json(conf)

    with st.expander("Music semantics & targets", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            sem = t.get("track_music_semantics") or {}
            if isinstance(sem, str):
                sem = json.loads(sem)
            st.json(sem)
        with c2:
            tgt = t.get("track_music_targets") or {}
            if isinstance(tgt, str):
                tgt = json.loads(tgt)
            st.json(tgt)

    with st.expander("Prompt ensemble", expanded=False):
        ens = t.get("track_clap_prompt_ensemble") or {}
        if isinstance(ens, str):
            ens = json.loads(ens)
        for k, v in ens.items():
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.70rem;'
                f'color:var(--accent-dim);margin-bottom:1px">{_safe(k)}</div>'
                f'<div style="font-size:0.83rem;color:var(--txt-2);'
                f'margin-bottom:8px">{_safe(v)}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("Audio analysis (raw)", expanded=False):
        st.json(t.get("audio_analysis") or {})

    with st.expander("Segmentation metadata (raw)", expanded=False):
        st.json(t.get("segmentation") or {})

def _page_inspector() -> None:
    """Track Inspector — search track_query by name or tags, view full record."""
    _page_header("🔍 Track Inspector", "search by name · filter by tag · view full record")

    _tag_vocab: dict[str, list[str]] = {}
    try:
        raw = json.loads(Path(TAGS_FILE).read_text(encoding="utf-8"))
        for cat, cat_data in raw.items():
            _tag_vocab[cat] = sorted(cat_data.get("tags", {}).keys())
    except Exception:
        for cat in CATEGORY_RULES:
            _tag_vocab[cat] = []

    with st.expander("Search filters", expanded=True):
        sc1, sc2 = st.columns([2, 3])
        with sc1:
            nq1, nq2 = st.columns([5, 1])
            with nq1:
                name_query = st.text_input(
                    "Track name",
                    placeholder="e.g. tension, piano, cue_042 ...",
                    help="Searches filename and semantic_title_en (case-insensitive).",
                    key=f"insp_name_input_{st.session_state.insp_name_key}",
                )
                st.session_state.insp_name = name_query
            with nq2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✕ Clear", key="insp_clear_name",
                             help="Clear the track-name field"):
                    st.session_state.insp_name = ""
                    st.session_state.insp_name_key += 1
                    st.rerun()

        with sc2:
            sel_cats = st.multiselect(
                "Tag categories",
                options=sorted(_tag_vocab.keys()),
                key="insp_cats",
                help="Select one or more categories to filter by tag.",
            )
            available_tags: list[str] = sorted(
                set(t for c in sel_cats for t in _tag_vocab.get(c, []))
            )
            sel_tags = st.multiselect(
                "Tags (within selected categories)",
                options=available_tags,
                key="insp_tags",
                disabled=not available_tags,
                help="Track must contain ALL selected tags in its should_have_tags list.",
            )

        st.checkbox(
            "Match from start only (prefix search on filename)",
            key="insp_starts_with",
            help="When checked, only tracks whose filename BEGINS with the "
                 "entered text are returned.",
        )

        if st.button("Search", type="primary"):
            _ver         = st.session_state.get("insp_name_key", 0)
            _name_query  = str(st.session_state.get(f"insp_name_input_{_ver}", "")).strip()
            _starts_with = bool(st.session_state.get("insp_starts_with", False))
            _sel_tags    = list(st.session_state.get("insp_tags", []))
            st.session_state.insp_last_starts_with = _starts_with
            _run_inspector_search(_name_query, _sel_tags, starts_with=_starts_with)

    results    = st.session_state.insp_results
    over_limit = st.session_state.get("insp_over_limit", False)
    if results is not None:
        _last_name = st.session_state.get("insp_last_name", "")
        _last_tags = st.session_state.get("insp_last_tags", [])
        if not _last_name and not _last_tags:
            _parts = "<b>Result for:</b> all tracks"
        else:
            _sw       = st.session_state.get("insp_last_starts_with", False)
            _mode     = " (starts with)" if _sw else " (contains)"
            _name_part = f'<b>Result for:</b> {_safe(_last_name)}{_mode}' if _last_name else ""
            _tags_part = f'<b>Tags:</b> {_safe(", ".join(_last_tags))}' if _last_tags else ""
            _parts     = " &nbsp;·&nbsp; ".join(p for p in [_name_part, _tags_part] if p)
        if len(results) == 0:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.78rem;color:var(--txt-2);margin-bottom:8px">'
                f'{_parts}</div>',
                unsafe_allow_html=True,
            )
            st.info("No tracks matched the search criteria.")
        else:
            import pandas as pd
            n = len(results)
            _count_lbl = f"{n:,} track{'s' if n != 1 else ''} found"
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.78rem;color:var(--txt-2);margin-bottom:4px">'
                f'{_parts}</div>'
                f'<div style="margin-bottom:10px">' + _badge(_count_lbl, "ok") + f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            display_cols = [c for c in
                ["id", "filename", "bpm", "musical_key",
                 "semantic_title_en", "tags_summary_en", "updated_at"]
                if c in results.columns]
            results_sorted = results[display_cols].sort_values(
                "filename", ascending=True).reset_index(drop=True)

            event = st.dataframe(
                results_sorted, key="insp_df_widget", use_container_width=True,
                hide_index=True, on_select="rerun", selection_mode="single-row",
                column_config={
                    "id":                st.column_config.NumberColumn("ID", width="small"),
                    "filename":          st.column_config.TextColumn("Filename"),
                    "bpm":               st.column_config.NumberColumn("BPM", format="%d"),
                    "musical_key":       st.column_config.TextColumn("Key", width="small"),
                    "semantic_title_en": st.column_config.TextColumn("Semantic title"),
                    "tags_summary_en":   st.column_config.TextColumn("Tags summary"),
                    "updated_at":        st.column_config.DatetimeColumn("Updated", format="DD MMM YYYY"),
                },
            )
            try:
                _sel_rows = event.selection.rows if event else []
            except AttributeError:
                _sel_rows = []
            if _sel_rows:
                _clicked_id = int(results_sorted.iloc[_sel_rows[0]]["id"])
                if _clicked_id != (st.session_state.insp_track or {}).get("id"):
                    st.session_state.insp_selected_id = _clicked_id
                    _load_inspector_track(_clicked_id)

            st.markdown("<hr>", unsafe_allow_html=True)
            _header("Track detail", "full record — all fields except embeddings")
            valid_ids  = results["id"].tolist()
            _cur_id    = st.session_state.get("insp_selected_id")
            _cur_id    = int(_cur_id) if (_cur_id and _cur_id in valid_ids) else int(valid_ids[0])
            _cur_track = st.session_state.insp_track or {}
            _filepath  = _cur_track.get("filepath", "")
            _filename  = _cur_track.get("filename", "")

            _l, _c, _r = st.columns([1.2, 2.2, 1.2])
            with _c:
                st.session_state.insp_autoplay = st.checkbox(
                    "Auto play — start playing immediately when a track is selected",
                    key="insp_autoplay_chk",
                )

            _left_pl, _center_pl, _right_pl = st.columns([1.2, 2.2, 1.2])
            with _center_pl:
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:0.72rem;color:var(--txt-3);'
                    f'margin-bottom:3px;text-transform:uppercase;letter-spacing:.08em">'
 #                   f'Track ID &nbsp;·&nbsp; '
                    f'<span style="font-size:1.10rem;color:var(--accent);font-weight:600">'
                    f'{_cur_id}</span></div>',
                    unsafe_allow_html=True,
                )
                if get_audio_for_player is not None and (_filepath or _filename):
                    _audio_bytes, _local_path, _audio_error = get_audio_for_player(
                        _filepath, _filename
                    )
                    if _audio_error:
                        st.warning(_audio_error)
                    elif _audio_bytes is not None:
                        st.audio(_audio_bytes, autoplay=st.session_state.insp_autoplay)
                    elif _local_path:
                        st.audio(_local_path, autoplay=st.session_state.insp_autoplay)
                elif _filepath or _cur_id:
                    st.markdown(
                        '<div style="font-family:var(--mono);font-size:0.72rem;'
                        'color:var(--txt-3);margin-top:6px;font-style:italic">'
                        '⚠ Audio engine unavailable (scene_music_matcher_engine not found)</div>',
                        unsafe_allow_html=True,
                    )

    t = st.session_state.insp_track
    if t:
        _render_inspector_track(t)

def _page_config() -> None:
    _page_header("Configuration", "constants · embedding weights · category rules · contrary tags")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Run constants", "Embedding weights", "Category rules", "Contrary tags"]
    )

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        for name, val in [
            ("PIPELINE_VERSION",  PIPELINE_VERSION),
            ("MUSIC_DIR",         str(MUSIC_DIR)),
            ("TAGS_FILE",         str(TAGS_FILE)),
            ("CLAP_MODEL_NAME",   CLAP_MODEL_NAME),
            ("DEVICE",            DEVICE),
            ("TARGET_SR",         TARGET_SR),
            ("SEGMENT_SECONDS",   SEGMENT_SECONDS),
            ("NUM_SEGMENTS",      NUM_SEGMENTS),
            ("WORKERS",           WORKERS),
            ("CHUNK_SIZE",        CHUNK_SIZE),
            ("BATCH_SIZE",        BATCH_SIZE),
            ("AUDIO_BATCH_SIZE",  AUDIO_BATCH_SIZE),
            ("TEXT_BATCH_SIZE",   TEXT_BATCH_SIZE),
            ("VECTOR_DIM",        VECTOR_DIM),
            ("SKIP_EXISTING",     SKIP_EXISTING),
        ]:
            st.markdown(
                f'<div style="display:flex;gap:14px;padding:5px 0;'
                f'border-bottom:1px solid var(--border-s);'
                f'font-family:var(--mono);font-size:0.77rem;">'
                f'<span style="color:var(--accent);width:220px">{_safe(name)}</span>'
                f'<span style="color:var(--txt-2)">{_safe(val)}</span></div>',
                unsafe_allow_html=True,
            )
        st.caption("Change values via .env file or the Run Import parameter panel.")

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            _header("Hybrid embedding weights", "text_hybrid_weights")
            st.caption(
                "Controls `embedding_hybrid`. Defined in pipeline_config.py — "
                "shared with scene_music_matcher_engine and screenplay_parser_engine."
            )
            for lbl, val in [
                ("embedding_main",    TEXT_HYBRID_WEIGHTS[0]),
                ("embedding_tags",    TEXT_HYBRID_WEIGHTS[1]),
                ("embedding_ensemble", TEXT_HYBRID_WEIGHTS[2]),
            ]:
                _score_bar(lbl, val, 1.0)
            st.markdown("<br>", unsafe_allow_html=True)
            _header("Tag prompt weights", "tag_prompt_weights")
            for lbl, val in [
                ("Prompt A — cinematic … with tag", TAG_PROMPT_WEIGHTS[0]),
                ("Prompt B — music tagged as …",    TAG_PROMPT_WEIGHTS[1]),
                ("Prompt C — film underscore …",    TAG_PROMPT_WEIGHTS[2]),
            ]:
                _score_bar(lbl, val, 1.0)
        with c2:
            _header("Ensemble prompt weights", "ensemble_prompt_weights")
            st.caption("7 prompts averaged → `embedding_clap_ensemble`.")
            for lbl, val in zip(
                ["semantic_scene_prompt", "music_for_scene_prompt", "emotion_prompt",
                 "narrative_prompt", "sonic_prompt", "tag_prompt", "concise_core_prompt"],
                ENSEMBLE_PROMPT_WEIGHTS,
            ):
                _score_bar(lbl, val, 1.0)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "`top_k` = max tags selected · `threshold` = min cosine score · "
            "`margin` = min gap to next · `required` = always pick ≥1"
        )
        import pandas as pd
        df = pd.DataFrame(CATEGORY_RULES).T.reset_index().rename(columns={"index": "category"})
        st.dataframe(df, use_container_width=True, hide_index=True,
            column_config={
                "category":  st.column_config.TextColumn("Category", width="medium"),
                "top_k":     st.column_config.NumberColumn("top_k",     format="%d"),
                "threshold": st.column_config.NumberColumn("threshold", format="%.3f"),
                "margin":    st.column_config.NumberColumn("margin",    format="%.3f"),
                "required":  st.column_config.NumberColumn("required",  format="%d"),
            })

    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Tags that must NOT appear together. Drives `must_not_tags` in the contract.")
        import pandas as pd
        rows_ct = [{"tag": k, "contrary tags": ", ".join(v)} for k, v in CONTRARY_TAGS.items()]
        st.dataframe(pd.DataFrame(rows_ct), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

_SS_DEFAULTS: dict[str, Any] = {
    "page":           "dashboard",
    "import_log":     [],
    "import_running": False,
    "import_done":    False,
    "import_stats":   {},
    "import_stage":   -1,
    "db_status":      {},
    "p_music_dir":    str(MUSIC_DIR),
    "p_skip":         SKIP_EXISTING,
    # Track inspector search state
    "insp_results":          None,
    "insp_track":            None,
    "insp_starts_with":      False,
    "insp_name":             "",
    "insp_over_limit":       False,
    "insp_name_key":         0,
    "insp_last_name":        "",
    "insp_last_tags":        [],
    "insp_last_starts_with": False,
    "insp_selected_id":      None,
    "insp_autoplay":         False,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_print_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT THREAD BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

def _start_import(**kwargs) -> None:
    log: list[str]         = ["[INFO] Starting import pipeline…"]
    result: dict[str, Any] = {"running": True, "done": False, "stats": {}}

    st.session_state.import_log     = log
    st.session_state.import_running = True
    st.session_state.import_done    = False
    st.session_state.import_stats   = {}
    st.session_state.import_stage   = 0
    st.session_state["_imp_result"] = result

    def _stage_cb(msg: str, stage: int) -> None:
        log.append(msg)
        if stage >= 0:
            result["stage"] = stage

    def _worker(log: list[str], result: dict[str, Any]) -> None:
        try:
            # run_import now returns an ImportResult — no log-string parsing needed.
            imp_result = run_import(**kwargs, log_fn=_stage_cb)

            total: Any = "–"
            try:
                conn = _imp_get_connection()
                cur  = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM track_query")
                total = int(cur.fetchone()[0])
                cur.close()
                conn.close()
            except Exception:
                pass

            stats = imp_result.to_stats_dict()
            stats["total"] = total
            result["stats"] = stats
            result["done"]  = True
        except Exception as exc:
            log.append(f"[ERROR] {exc}")
            log.append(traceback.format_exc())
        finally:
            result["running"] = False

    threading.Thread(target=_worker, args=(log, result), daemon=True).start()
    time.sleep(0.4)
    st.rerun()


def _sync_import() -> None:
    """Sync the background import thread result back into session state.

    Called at the top of main() on every Streamlit rerun so that when the
    worker thread finishes, import_running is cleared, import_done is set,
    and import_stats are populated — causing the page to exit the
    'Pipeline running…' spinner and show 'Import completed.'
    """
    r = st.session_state.get("_imp_result", {})
    if r:
        if not r.get("running", True):
            st.session_state.import_running = False
        if "stage" in r:
            st.session_state.import_stage = r["stage"]
        if r.get("done") and not st.session_state.import_done:
            st.session_state.import_done  = True
            st.session_state.import_stats = r.get("stats", {})


# ============================================================
# ENTRY POINT
# ============================================================


def main() -> None:
    """Streamlit application entrypoint."""
    # ── Sync background import thread → session state on every rerun ─────
    _sync_import()

    # ── Session-state initialization (must happen before any key access) ──
    _SS_DEFAULTS: dict = {
        "db_status":             {},
        "page":                  "dashboard",
        "last_result_bundle":    None,
        "last_export_payload":   None,
    }
    for _k, _v in _SS_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── DB check (only when not yet populated) ────────────────────────────
    if not st.session_state.db_status:
        st.session_state.db_status = _check_db()

    _sidebar()

    page = st.session_state.page
    if   page == "dashboard": _page_dashboard()
    elif page == "parser":    _page_parser()
    elif page == "matcher":   _page_matcher()
    elif page == "pdf":       _page_pdf()
    elif page == "scoring":   _page_scoring()
    elif page == "importer":  _page_importer()
    elif page == "inspector": _page_inspector()
    elif page == "config":    _page_config()
    else:                     _page_dashboard()


if __name__ == "__main__":
    main()
