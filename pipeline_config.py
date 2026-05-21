#!/usr/bin/env python3
"""
pipeline_config.py  —  AI Music Supervisor · Shared Pipeline Configuration
===========================================================================

Single source of truth for all constants that are shared across:
  - music_library_importer.py   (track ingestion pipeline)
  - importer_pipeline.py        (ETL business logic)
  - scene_music_matcher_engine.py  (v6 scoring pipeline)
  - screenplay_parser_engine.py    (LLM segmentation + embedding)

Previously these constants were duplicated in each script.  Any weight
or rule change now requires editing only this file.

Changes vs initial creation (v1.0)
------------------------------------
1.  Extracted from music_library_importer.py v3.35:
      CATEGORY_RULES, TAG_PROMPT_WEIGHTS, ENSEMBLE_PROMPT_WEIGHTS,
      TEXT_HYBRID_WEIGHTS, PROFILE_HINTS, CONTRARY_TAGS.
2.  These replace the inline definitions in music_library_importer.py,
    importer_pipeline.py, scene_music_matcher_engine.py, and
    screenplay_parser_engine.py — import from here instead.
"""

from __future__ import annotations

SCRIPT_NAME      = "pipeline_config.py"
SCRIPT_VERSION   = "1.1"
PIPELINE_VERSION = f"{SCRIPT_NAME[:-3]}_v{SCRIPT_VERSION}"

# ── Change log ──────────────────────────────────────────────
# v1.1 (2026-05-13)
#   • Added PIPELINE_VERSION — was the only module without it
#     (ISSUE-06).
#   • Added validate_weights() guard — raises ValueError when a
#     weight list does not sum to 1.0, preventing silent mis-
#     configuration (ISSUE-06).
#   • Removed old-style typing imports (Dict, List); replaced
#     with built-in generics (dict, list) — from __future__
#     import annotations already active (ISSUE-02).
# ────────────────────────────────────────────────────────────

# ── Tag embedding weights ──────────────────────────────────────────────────────
# Three prompts per tag, weighted when building the per-tag embedding vector.
TAG_PROMPT_WEIGHTS: list[float] = [0.52, 0.28, 0.20]

# Seven CLAP prompts per track/scene, weighted into embedding_clap_ensemble.
ENSEMBLE_PROMPT_WEIGHTS: list[float] = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]

# Weights for embedding_hybrid.
# Index 0 = embedding_main, 1 = embedding_tags, 2 = embedding_clap_ensemble.
# CRITICAL: these values MUST be identical in every script that writes or reads
# embedding_hybrid.  Cross-table cosine similarity is only valid when both the
# scene_query and track_query hybrid vectors were produced with the same weights.
TEXT_HYBRID_WEIGHTS: list[float] = [0.45, 0.20, 0.35]


def validate_weights(weights: list[float], name: str = "weights") -> None:
    """Assert that a weight list sums to 1.0 within floating-point tolerance.

    Raises ValueError with a descriptive message when the check fails.
    Call at module initialisation time to catch config drift early.

    Example
    -------
    >>> validate_weights(TAG_PROMPT_WEIGHTS, "TAG_PROMPT_WEIGHTS")
    """
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"{name} must sum to 1.0, got {total:.8f} "
            f"(values: {weights})"
        )


# Validate all weight lists at import time so misconfiguration is caught
# immediately rather than silently producing wrong embeddings.
validate_weights(TAG_PROMPT_WEIGHTS,      "TAG_PROMPT_WEIGHTS")
validate_weights(ENSEMBLE_PROMPT_WEIGHTS, "ENSEMBLE_PROMPT_WEIGHTS")
validate_weights(TEXT_HYBRID_WEIGHTS,     "TEXT_HYBRID_WEIGHTS")


# ── Tag selection rules ────────────────────────────────────────────────────────
# Per-category selection parameters used by select_best_tags() in the importer
# and validated by the scorer in scene_music_matcher_engine.
CATEGORY_RULES: dict[str, dict[str, float]] = {
    "emotion":         {"top_k": 3, "threshold": 0.255, "margin": 0.020, "required": 0},
    "scene":           {"top_k": 2, "threshold": 0.245, "margin": 0.020, "required": 0},
    "energy":          {"top_k": 1, "threshold": 0.215, "margin": 0.015, "required": 1},
    "tempo":           {"top_k": 1, "threshold": 0.215, "margin": 0.015, "required": 1},
    "rhythm":          {"top_k": 1, "threshold": 0.215, "margin": 0.015, "required": 1},
    "intensity_shape": {"top_k": 1, "threshold": 0.220, "margin": 0.015, "required": 1},
    "narrative":       {"top_k": 3, "threshold": 0.235, "margin": 0.018, "required": 1},
    "sound_character": {"top_k": 3, "threshold": 0.230, "margin": 0.018, "required": 1},
    "usage":           {"top_k": 2, "threshold": 0.225, "margin": 0.018, "required": 0},
    "instrumentation": {"top_k": 3, "threshold": 0.225, "margin": 0.018, "required": 0},
    "atmosphere":      {"top_k": 2, "threshold": 0.225, "margin": 0.018, "required": 0},
    "special":         {"top_k": 2, "threshold": 0.225, "margin": 0.018, "required": 0},
}

# ── Scene-type → expected-tag mapping ─────────────────────────────────────────
# Used by derive_weight_profile_candidates() to score how well a track's
# selected tags match each known scene/dramatic profile.
PROFILE_HINTS: dict[str, set] = {
    "dialogue":       {"dialogue", "conversation", "background_music", "underscore",
                       "dialogue_safe", "restrained", "sparse", "intimate"},
    "investigation":  {"investigation", "mystery", "curiosity", "tension",
                       "suspense", "textured", "pulsating"},
    "action":         {"action", "danger", "aggression", "driving", "fast",
                       "very_fast", "high", "very_high", "hits", "impacts", "pulses"},
    "dramatic_scene": {"dramatic_scene", "sadness", "grief", "melancholy",
                       "hope", "emotional_support", "strings", "piano"},
    "horror_scene":   {"horror_scene", "fear", "suspense", "dark",
                       "eerie", "tension_building", "textured"},
    "thriller_scene": {"thriller_scene", "tension", "suspense", "anxiety",
                       "investigation", "driving", "pulsating"},
    "romantic_scene": {"romantic_scene", "love", "romance", "warmth", "intimate", "sparse"},
    "comedic_scene":  {"comedic_scene", "playful", "joy", "happiness"},
    "transition":     {"transition", "bridge", "release", "background",
                       "underscore", "moderate"},
    "montage":        {"montage", "determination", "hope", "uplifting",
                       "driving", "gradual_build"},
    "climax":         {"climax", "climax_peak", "epic", "heroism",
                       "very_high", "hits", "impacts", "trailer_music"},
    "resolution":     {"resolution", "relief", "peace", "release", "warm", "calm"},
    "aftermath":      {"aftermath", "grief", "sadness", "calm", "peace",
                       "sparse", "very_slow", "slow"},
}

# ── Mutually-exclusive tag pairs ───────────────────────────────────────────────
# Drives must_not_tags in every track/scene contract.
# If tag A is selected for a track, each tag in CONTRARY_TAGS[A] is a
# candidate for must_not_tags (unless A itself is already selected).
CONTRARY_TAGS: dict[str, list[str]] = {
    "very_low":         ["very_high", "high"],
    "low":              ["very_high"],
    "high":             ["very_low", "dialogue_safe"],
    "very_high":        ["very_low", "low", "dialogue_safe", "restrained"],
    "very_slow":        ["very_fast", "fast"],
    "slow":             ["very_fast"],
    "fast":             ["very_slow", "dialogue_safe"],
    "very_fast":        ["very_slow", "slow"],
    "dialogue_safe":    ["hits", "impacts", "very_high", "climax_peak", "trailer_music"],
    "restrained":       ["hits", "impacts", "very_high", "climax_peak", "trailer_music"],
    "background_music": ["hits", "impacts"],
    "trailer_music":    ["dialogue_safe", "restrained", "very_low", "very_slow"],
    "dark":             ["warm", "uplifting", "playful"],
    "uplifting":        ["dark", "eerie", "fear"],
    "playful":          ["dark", "suspense", "fear", "grief"],
    "joy":              ["fear", "grief", "melancholy"],
    "fear":             ["joy", "relief", "peace"],
    "peace":            ["aggression", "danger", "very_high"],
    "intimate":         ["wide"],
    "wide":             ["intimate"],
    "sparse":           ["dense"],
    "dense":            ["sparse"],
    "clean":            ["distorted", "noisy"],
    "distorted":        ["clean"],
    "analog":           ["digital"],
    "digital":          ["analog"],
}
