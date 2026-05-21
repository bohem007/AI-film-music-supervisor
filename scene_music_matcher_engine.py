#!/usr/bin/env python3
from __future__ import annotations

# ============================================================
# VERSION
# ============================================================
SCRIPT_NAME      = "scene_music_matcher_engine.py"
SCRIPT_VERSION   = "3.3"
PIPELINE_VERSION = f"{SCRIPT_NAME[:-3]}_v{SCRIPT_VERSION}"

# ── Change log ──────────────────────────────────────────────
# v3.3 (2026-05-13)
#   • Added return type annotation to _db_connect() —
#     contextlib.AbstractContextManager[tuple[...]] — the only
#     public function in the module that was missing one.
# v3.2 (2026-05-13)
#   • Added ScoringWeights dataclass; get_default_weights()
#     delegates to ScoringWeights.default().to_dict().
# v3.1 (2026-05-13)
#   • Added __all__.
# v3.0 (2026-04-29)
#   • Extracted from cl_scene_music_matcher.py v3.22.
#     Contains all business logic: config, data models, DB access,
#     normalizers/validators, scoring engine, penalties, ranking
#     pipeline, persistence, export, and audio helper.
#     Zero Streamlit imports — safe to import from tests or CLI.
#   • HIGH: module-level st.* calls removed; engine is now importable
#     without triggering Streamlit.
#   • HIGH: ensure_scene_music_matches_v6_exists() result cached via
#     _table_existence_cache set; DB round-trips reduced from N to 1
#     per connection lifetime.
#   • HIGH: get_audio_for_player S3 client cached in a module-level
#     dict keyed by endpoint+key; boto3.client() no longer re-created
#     on every audio request.
#   • HIGH: progress callback added to match_episode_v6() so GUI can
#     display progress without direct st.progress coupling.
#   • MEDIUM: RankedMatchV6 gains ScoreBundle sub-dataclass that groups
#     the 14 float score/penalty fields; constructor remains backward-
#     compatible via keyword arguments.
#   • MEDIUM: schema-drift try/except in
#     fetch_scene_query_by_episode_scene_theme replaced by probing once
#     per connection via _has_column() helper.
#   • MEDIUM: get_connection wrapped in _db_connect context manager
#     for automatic close-on-exit; existing callers still work.
#   • MEDIUM: round_floats_3 and format_float_columns_df moved here
#     (were misplaced before imports in v3.22).
#   • LOW: get_default_weights() relocated next to scoring engine.
#   • LOW: normalize_scene_query_row and normalize_track_query_row
#     reformatted onto multiple lines (were unreadable single-liners).
#   • LOW: fetch_episode_options / fetch_recent_episode_numbers moved
#     here from the GUI zone (they are pure DB helpers).
# ────────────────────────────────────────────────────────────

import contextlib
import io
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import psycopg2
from psycopg2.extras import Json, execute_values
from dotenv import load_dotenv
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# ============================================================
# UTILITY HELPERS
# ============================================================

def round_floats_3(value: Any) -> Any:
    """Recursively round floats to 3 decimals for UI and export."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {k: round_floats_3(v) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats_3(v) for v in value]
    return value


def format_float_columns_df(df: Any) -> Any:
    """Round float columns in a pandas DataFrame to 3 decimals."""
    try:
        float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns
        if len(float_cols) > 0:
            df = df.copy()
            df[float_cols] = df[float_cols].round(3)
    except Exception:
        pass
    return df


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
TAGS_FILE  = Path(os.getenv("TAGS_FILE",  "tags_v2.json"))

# ── Cloud / R2 storage config ─────────────────────────────────────────────────
CLOUD_MUSIC          = os.getenv("CLOUD_MUSIC", "false").strip().lower() == "true"
CF_ACCOUNT_ID        = os.getenv("CF_ACCOUNT_ID",        "")
CF_TOKEN_VALUE       = os.getenv("CF_TOKEN_VALUE",        "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID",     "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL",      "")
BUCKET_NAME          = os.getenv("BUCKET_NAME",          "")

DB_CONFIG: dict[str, Any] = {
    "host":            os.getenv("PGHOST"),
    "dbname":          os.getenv("PGDATABASE"),
    "user":            os.getenv("PGUSER"),
    "password":        os.getenv("PGPASSWORD"),
    "sslmode":         os.getenv("PGSSLMODE"),
    "channel_binding": os.getenv("PGCHANNELBINDING"),
}


# ============================================================
# PUBLIC API
# ============================================================
__all__ = [
    # Data models
    "ScoreBundle", "NormalizedSceneQuery", "NormalizedTrackQuery", "RankedMatchV6",
    "ScoringWeights",
    # Config / runtime
    "get_default_weights", "load_runtime_config", "ensure_output_dir",
    "DB_CONFIG", "OUTPUT_DIR", "TAGS_FILE",
    "CLOUD_MUSIC", "BUCKET_NAME",
    "SCRIPT_NAME", "SCRIPT_VERSION", "PIPELINE_VERSION",
    # DB connection
    "get_connection", "_db_connect",
    # DB fetch helpers
    "fetch_scene_queries_for_episode", "fetch_scene_query_by_id",
    "fetch_scoring_episode_numbers", "fetch_scoring_scene_numbers",
    "fetch_scoring_theme_numbers", "fetch_scene_query_by_episode_scene_theme",
    "fetch_matches_for_theme", "fetch_track_detail",
    "fetch_track_query_candidates_by_hybrid", "fetch_track_query_candidates_by_ensemble",
    "fetch_track_query_rows", "fetch_existing_episode_matches",
    "fetch_pdf_episode_numbers", "fetch_matches_as_ranked_v6",
    "fetch_track_query_count", "fetch_episode_options", "fetch_recent_episode_numbers",
    "delete_episode_matches_v6",
    # Normalizers / validators
    "parse_vector", "normalize_jsonb_object",
    "normalize_scene_query_row", "normalize_track_query_row",
    "validate_scene_query_record", "validate_track_query_record",
    "build_input_validation_report",
    # Candidate retrieval
    "union_candidate_ids", "filter_candidates_minimum_validity",
    "retrieve_track_candidates_for_scene",
    # Scoring
    "cosine_similarity_safe", "compute_embedding_similarities",
    "compute_embedding_score", "compute_emotional_direction_score",
    "compute_narrative_function_score", "compute_weight_profile_score",
    "compute_dialogue_score", "compute_semantic_score",
    "compute_scalar_target_score", "compute_sound_character_score",
    "compute_targets_score", "compute_tag_selection_score",
    "compute_dialogue_safe_penalty", "compute_duration_penalty",
    "compute_forbidden_tag_penalty",
    # Utilities
    "round_floats_3", "format_float_columns_df",
    "ensure_scene_music_matches_v6_exists",
]


# ============================================================
# RUNTIME MODELS / TYPED STRUCTURES
# ============================================================

@dataclass
class ScoreBundle:
    """Groups the 14 numeric score and penalty fields of one ranked match."""
    main_similarity:               float = 0.0
    tags_similarity:               float = 0.0
    ensemble_similarity:           float = 0.0
    hybrid_similarity:             float = 0.0
    audio_similarity_aux:          float = 0.0
    embedding_score:               float = 0.0
    semantic_score:                float = 0.0
    targets_score:                 float = 0.0
    tag_selection_score:           float = 0.0
    dialogue_score:                float = 0.0
    dialogue_conflict:             float = 0.0
    duration_conflict:             float = 0.0
    forbidden_tag_conflict:        float = 0.0
    style_redundancy_penalty:      float = 0.0
    same_track_consecutive_penalty: float = 0.0
    missing_data_penalty:          float = 0.0
    penalty_total:                 float = 0.0


@dataclass
class NormalizedSceneQuery:
    """Normalized runtime representation of one scene_query row."""
    id:                    int
    episode_nr:            int
    scene_nr:              int
    theme_nr:              int
    theme_title_pl:        str
    theme_title_en:        str
    description_en:        str
    theme_txt:             str
    tags_summary_en:       str
    scene_music_semantics: dict[str, Any]
    scene_music_targets:   dict[str, Any]
    scene_tag_selection:   dict[str, Any]
    clap_prompt_ensemble:  dict[str, Any]
    embedding_main:        list[float]
    embedding_tags:        list[float]
    embedding_clap_ensemble: list[float]
    embedding_hybrid:      list[float]


@dataclass
class NormalizedTrackQuery:
    """Normalized runtime representation of one track_query row."""
    id:                        int
    filename:                  str
    filepath:                  str
    duration_sec:              float
    bpm:                       Optional[int]
    musical_key:               Optional[str]
    semantic_title_en:         str
    description_en:            str
    tags_summary_en:           str
    track_music_semantics:     dict[str, Any]
    track_music_targets:       dict[str, Any]
    track_tag_selection:       dict[str, Any]
    track_clap_prompt_ensemble: dict[str, Any]
    embedding_audio:           list[float]
    embedding_main:            list[float]
    embedding_tags:            list[float]
    embedding_clap_ensemble:   list[float]
    embedding_hybrid:          list[float]
    audio_analysis:            dict[str, Any]
    segmentation:              dict[str, Any]


@dataclass
class RankedMatchV6:
    """One ranked match result for scene_query ↔ track_query."""
    episode_nr:              int
    scene_query_id:          int
    scene_nr:                int
    theme_nr:                int
    track_query_id:          int
    rank_position:           int
    scene_theme_title_en:    str
    track_semantic_title_en: str
    track_filename:          str
    track_filepath:          str
    duration_sec:            Optional[float]
    bpm:                     Optional[int]
    musical_key:             Optional[str]
    final_score:             float
    # ── Score / penalty fields (grouped in ScoreBundle but kept flat here
    #    for full backward-compatibility with save/fetch SQL column mapping) ──
    main_similarity:               float = 0.0
    tags_similarity:               float = 0.0
    ensemble_similarity:           float = 0.0
    hybrid_similarity:             float = 0.0
    audio_similarity_aux:          float = 0.0
    embedding_score:               float = 0.0
    semantic_score:                float = 0.0
    targets_score:                 float = 0.0
    tag_selection_score:           float = 0.0
    dialogue_score:                float = 0.0
    dialogue_conflict:             float = 0.0
    duration_conflict:             float = 0.0
    forbidden_tag_conflict:        float = 0.0
    style_redundancy_penalty:      float = 0.0
    same_track_consecutive_penalty: float = 0.0
    missing_data_penalty:          float = 0.0
    penalty_total:                 float = 0.0
    style_signature:               str   = ""
    dialogue_safe_applied:         bool  = False
    match_metadata:   dict[str, Any] = field(default_factory=dict)
    match_explanation: dict[str, Any] = field(default_factory=dict)


# ============================================================
# CONFIG HELPERS
# ============================================================

@dataclass
class ScoringWeights:
    """Validated weight set for the v6 scene-music scoring pipeline.

    All four top-level weights (``embedding``, ``semantic``, ``targets``,
    ``tag_selection``) must sum to exactly 1.0; ``__post_init__`` enforces
    this so misconfigured weights raise immediately rather than silently
    degrading match quality.

    Usage
    -----
    Build from defaults::

        w = ScoringWeights.default()

    Override one component and re-validate::

        w = ScoringWeights(embedding=0.45, semantic=0.25, targets=0.15, tag_selection=0.15)

    Pass to ``rank_one_candidate_v6`` via ``config["weights"]``::

        config["weights"] = w.to_dict()
    """
    # ── Top-level final-score weights ──────────────────────────
    embedding:    float = 0.40
    semantic:     float = 0.25
    targets:      float = 0.20
    tag_selection: float = 0.15

    # ── Embedding sub-weights ───────────────────────────────────
    emb_main:      float = 0.15
    emb_tags:      float = 0.10
    emb_ensemble:  float = 0.20
    emb_hybrid:    float = 0.50
    emb_audio_aux: float = 0.05

    # ── Semantic sub-weights ────────────────────────────────────
    sem_emotion:        float = 0.35
    sem_narrative:      float = 0.25
    sem_weight_profile: float = 0.20
    sem_dialogue:       float = 0.20

    # ── Targets sub-weights ─────────────────────────────────────
    tgt_energy:          float = 0.20
    tgt_tempo:           float = 0.20
    tgt_rhythm:          float = 0.15
    tgt_intensity_shape: float = 0.20
    tgt_sound_character: float = 0.25

    # ── Tag-selection sub-weights ───────────────────────────────
    tag_should_have: float = 0.75
    tag_must_not:    float = 0.25

    def __post_init__(self) -> None:
        self._validate_group(
            [self.embedding, self.semantic, self.targets, self.tag_selection],
            "final weights (embedding + semantic + targets + tag_selection)",
        )
        self._validate_group(
            [self.emb_main, self.emb_tags, self.emb_ensemble,
             self.emb_hybrid, self.emb_audio_aux],
            "embedding sub-weights",
        )
        self._validate_group(
            [self.sem_emotion, self.sem_narrative,
             self.sem_weight_profile, self.sem_dialogue],
            "semantic sub-weights",
        )
        self._validate_group(
            [self.tgt_energy, self.tgt_tempo, self.tgt_rhythm,
             self.tgt_intensity_shape, self.tgt_sound_character],
            "targets sub-weights",
        )
        self._validate_group(
            [self.tag_should_have, self.tag_must_not],
            "tag_selection sub-weights",
        )

    @staticmethod
    def _validate_group(weights: list[float], name: str) -> None:
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ScoringWeights: {name} must sum to 1.0, got {total:.8f}"
            )

    @classmethod
    def default(cls) -> "ScoringWeights":
        """Return a ScoringWeights instance with the standard production weights."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoringWeights":
        """Construct from a nested dict as returned by the legacy get_default_weights().

        Accepts both flat ``{"embedding": 0.40, …}`` dicts and the nested
        ``{"final": {"embedding": 0.40}, "embedding": {…}, …}`` form.
        """
        flat: dict[str, float] = {}
        if "final" in d:
            flat.update({k: float(v) for k, v in d["final"].items()})
        else:
            for k in ("embedding", "semantic", "targets", "tag_selection"):
                if k in d:
                    flat[k] = float(d[k])
        if "embedding" in d and isinstance(d["embedding"], dict):
            sub = d["embedding"]
            flat.update({
                "emb_main":      float(sub.get("main",      0.15)),
                "emb_tags":      float(sub.get("tags",      0.10)),
                "emb_ensemble":  float(sub.get("ensemble",  0.20)),
                "emb_hybrid":    float(sub.get("hybrid",    0.50)),
                "emb_audio_aux": float(sub.get("audio_aux", 0.05)),
            })
        if "semantic" in d and isinstance(d["semantic"], dict):
            sub = d["semantic"]
            flat.update({
                "sem_emotion":        float(sub.get("emotion",        0.35)),
                "sem_narrative":      float(sub.get("narrative",      0.25)),
                "sem_weight_profile": float(sub.get("weight_profile", 0.20)),
                "sem_dialogue":       float(sub.get("dialogue",       0.20)),
            })
        if "targets" in d and isinstance(d["targets"], dict):
            sub = d["targets"]
            flat.update({
                "tgt_energy":          float(sub.get("energy",          0.20)),
                "tgt_tempo":           float(sub.get("tempo",           0.20)),
                "tgt_rhythm":          float(sub.get("rhythm",          0.15)),
                "tgt_intensity_shape": float(sub.get("intensity_shape", 0.20)),
                "tgt_sound_character": float(sub.get("sound_character", 0.25)),
            })
        if "tag_selection" in d and isinstance(d["tag_selection"], dict):
            sub = d["tag_selection"]
            flat.update({
                "tag_should_have": float(sub.get("should_have", 0.75)),
                "tag_must_not":    float(sub.get("must_not",    0.25)),
            })
        return cls(**flat)

    def to_dict(self) -> dict[str, Any]:
        """Return the nested dict form expected by rank_one_candidate_v6().

        Preserves backward compatibility with ``config.get("weights", {})``.
        """
        return {
            "final": {
                "embedding":     self.embedding,
                "semantic":      self.semantic,
                "targets":       self.targets,
                "tag_selection": self.tag_selection,
            },
            "embedding": {
                "main":      self.emb_main,
                "tags":      self.emb_tags,
                "ensemble":  self.emb_ensemble,
                "hybrid":    self.emb_hybrid,
                "audio_aux": self.emb_audio_aux,
            },
            "semantic": {
                "emotion":        self.sem_emotion,
                "narrative":      self.sem_narrative,
                "weight_profile": self.sem_weight_profile,
                "dialogue":       self.sem_dialogue,
            },
            "targets": {
                "energy":          self.tgt_energy,
                "tempo":           self.tgt_tempo,
                "rhythm":          self.tgt_rhythm,
                "intensity_shape": self.tgt_intensity_shape,
                "sound_character": self.tgt_sound_character,
            },
            "tag_selection": {
                "should_have": self.tag_should_have,
                "must_not":    self.tag_must_not,
            },
        }


def get_default_weights() -> dict[str, Any]:
    """Return default scoring weights as a nested dict.

    Delegates to ``ScoringWeights.default().to_dict()`` so the returned
    structure is always validated before use.  Existing callers that do
    ``config.get("weights", {}).get("final", {})`` continue to work
    unchanged.
    """
    return ScoringWeights.default().to_dict()


def load_runtime_config() -> dict[str, Any]:
    """Load runtime configuration and defaults for matcher execution."""
    return {
        "db_config":              DB_CONFIG,
        "output_dir":             OUTPUT_DIR,
        "rank_top_n":             int(os.getenv("RANK_TOP_N",             "10")),
        "preview_top_n":          int(os.getenv("PREVIEW_TOP_N",          "5")),
        "candidate_pool_hybrid":  int(os.getenv("CANDIDATE_POOL_HYBRID",  "150")),
        "candidate_pool_ensemble":int(os.getenv("CANDIDATE_POOL_ENSEMBLE","60")),
        "candidate_union_limit":  int(os.getenv("CANDIDATE_UNION_LIMIT",  "180")),
        "enable_audio_aux":       os.getenv("ENABLE_AUDIO_AUX",       "1") == "1",
        "strict_dialogue_safe":   os.getenv("STRICT_DIALOGUE_SAFE",   "1") == "1",
        "style_duplicate_limit":  int(os.getenv("STYLE_DUPLICATE_LIMIT",  "2")),
        "low_confidence_threshold": float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.45")),
        "weights":                get_default_weights(),
    }


def ensure_output_dir(output_dir: Path) -> Path:
    """Create output directory if needed and return it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# DB ACCESS
# ============================================================

# Per-connection cache: stores the set of connections where
# scene_music_matches_v6 existence has already been verified.
# Keyed by id(connection) to avoid repeated SELECT to_regclass().
_table_existence_cache: set[int] = set()


def get_connection(db_config: dict[str, Any]) -> psycopg2.extensions.connection:
    """Create a PostgreSQL connection."""
    return psycopg2.connect(**db_config)


@contextlib.contextmanager
def _db_connect(
    db_config: dict[str, Any],
) -> contextlib.AbstractContextManager[tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]]:
    """Context manager: open a connection and guarantee close on exit.

    Usage::

        with _db_connect(DB_CONFIG) as (conn, cur):
            rows = cur.execute(...).fetchall()
    """
    conn = get_connection(db_config)
    cur  = conn.cursor()
    try:
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
        _table_existence_cache.discard(id(conn))


def ensure_scene_music_matches_v6_exists(cur) -> None:
    """Validate that scene_music_matches_v6 exists — cached per connection.

    The SELECT to_regclass() round-trip is executed at most once per
    connection object; subsequent calls within the same connection are
    no-ops.  This eliminates the 5 × RTT overhead that occurred in v3.22
    when every fetch function called this independently.
    """
    conn_id = id(cur.connection)
    if conn_id in _table_existence_cache:
        return
    cur.execute("SELECT to_regclass('public.scene_music_matches_v6')")
    row = cur.fetchone()
    if not row or row[0] is None:
        raise RuntimeError(
            "scene_music_matches_v6 table is missing. Run DB setup first."
        )
    _table_existence_cache.add(conn_id)


def _has_column(cur, table: str, column: str) -> bool:
    """Return True when *column* exists in *table* (information_schema probe).

    Used to detect optional columns added by schema migrations so that
    try/except schema-drift workarounds are no longer needed.
    """
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = %s
          AND column_name  = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def fetch_scene_queries_for_episode(
    cur, episode_nr: int
) -> list[dict[str, Any]]:
    """Fetch ordered scene_query rows for one episode."""
    cur.execute(
        """
        SELECT id, episode_nr, scene_nr, theme_nr,
               theme_title_pl, theme_title_en, description_en, theme_txt,
               tags_summary_en, scene_music_semantics, scene_music_targets,
               scene_tag_selection, clap_prompt_ensemble,
               embedding_main, embedding_tags,
               embedding_clap_ensemble, embedding_hybrid
        FROM scene_query
        WHERE episode_nr = %s
        ORDER BY scene_nr, theme_nr
        """,
        (episode_nr,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_scene_query_by_id(
    cur, scene_query_id: int
) -> Optional[dict[str, Any]]:
    """Fetch one scene_query row by id."""
    cur.execute(
        """
        SELECT id, episode_nr, scene_nr, theme_nr,
               theme_title_pl, theme_title_en, description_en, theme_txt,
               tags_summary_en, scene_music_semantics, scene_music_targets,
               scene_tag_selection, clap_prompt_ensemble,
               embedding_main, embedding_tags,
               embedding_clap_ensemble, embedding_hybrid
        FROM scene_query WHERE id = %s
        """,
        (scene_query_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def fetch_scoring_episode_numbers(cur) -> list[int]:
    """Return distinct episode numbers from scene_music_matches_v6, newest first."""
    ensure_scene_music_matches_v6_exists(cur)
    cur.execute(
        "SELECT DISTINCT episode_nr FROM scene_music_matches_v6 ORDER BY episode_nr DESC"
    )
    return [int(r[0]) for r in cur.fetchall()]


def fetch_scoring_scene_numbers(cur, episode_nr: int) -> list[int]:
    """Return distinct scene numbers for one episode from scene_music_matches_v6."""
    ensure_scene_music_matches_v6_exists(cur)
    cur.execute(
        "SELECT DISTINCT scene_nr FROM scene_music_matches_v6 WHERE episode_nr=%s ORDER BY scene_nr",
        (episode_nr,),
    )
    return [int(r[0]) for r in cur.fetchall()]


def fetch_scoring_theme_numbers(cur, episode_nr: int, scene_nr: int) -> list[int]:
    """Return distinct theme numbers for one episode+scene, ascending."""
    ensure_scene_music_matches_v6_exists(cur)
    cur.execute(
        """
        SELECT DISTINCT theme_nr FROM scene_music_matches_v6
        WHERE episode_nr=%s AND scene_nr=%s ORDER BY theme_nr
        """,
        (episode_nr, scene_nr),
    )
    return [int(r[0]) for r in cur.fetchall()]


def fetch_scene_query_by_episode_scene_theme(
    cur,
    episode_nr: int,
    scene_nr:   int,
    theme_nr:   int,
) -> Optional[dict[str, Any]]:
    """Fetch the full scene_query row identified by episode · scene · theme.

    Includes segmentation_reason when the column exists in the schema
    (detected once via _has_column instead of the previous try/except
    schema-drift workaround).
    """
    has_seg = _has_column(cur, "scene_query", "segmentation_reason")
    extra   = ", segmentation_reason" if has_seg else ""
    cur.execute(
        f"""
        SELECT id, episode_nr, scene_nr, theme_nr,
               theme_title_pl, theme_title_en, description_en, theme_txt,
               tags_summary_en{extra},
               scene_music_semantics, scene_music_targets,
               scene_tag_selection, clap_prompt_ensemble
        FROM scene_query
        WHERE episode_nr=%s AND scene_nr=%s AND theme_nr=%s
        LIMIT 1
        """,
        (episode_nr, scene_nr, theme_nr),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def fetch_matches_for_theme(
    cur,
    episode_nr: int,
    scene_nr:   int,
    theme_nr:   int,
) -> list[dict[str, Any]]:
    """Fetch all ranked match rows for one episode · scene · theme."""
    ensure_scene_music_matches_v6_exists(cur)
    cur.execute(
        """
        SELECT
            rank_position, scene_query_id, track_query_id,
            track_filename, track_semantic_title_en,
            duration_sec, bpm, musical_key,
            final_score, embedding_score, semantic_score,
            targets_score, tag_selection_score, dialogue_score,
            main_similarity, tags_similarity, ensemble_similarity,
            hybrid_similarity, audio_similarity_aux,
            dialogue_conflict, duration_conflict, forbidden_tag_conflict,
            style_redundancy_penalty, same_track_consecutive_penalty,
            missing_data_penalty, penalty_total,
            style_signature, dialogue_safe_applied, match_explanation
        FROM scene_music_matches_v6
        WHERE episode_nr=%s AND scene_nr=%s AND theme_nr=%s
        ORDER BY rank_position
        """,
        (episode_nr, scene_nr, theme_nr),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def fetch_track_detail(
    cur, track_query_id: int
) -> Optional[dict[str, Any]]:
    """Fetch one track_query row by id, excluding all embedding vectors."""
    cur.execute(
        """
        SELECT id, filename, filepath, duration_sec, bpm, musical_key,
               semantic_title_en, description_en, tags_summary_en,
               track_music_semantics, track_music_targets,
               track_tag_selection, track_clap_prompt_ensemble,
               audio_analysis, segmentation, updated_at
        FROM track_query
        WHERE id = %s
        """,
        (track_query_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def fetch_track_query_candidates_by_hybrid(
    cur,
    scene_embedding_hybrid: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch candidate track_query rows using embedding_hybrid similarity."""
    cur.execute(
        """
        SELECT id, filename, filepath, duration_sec, bpm, musical_key,
               semantic_title_en, description_en, tags_summary_en,
               track_music_semantics, track_music_targets,
               track_tag_selection, track_clap_prompt_ensemble,
               embedding_audio, embedding_main, embedding_tags,
               embedding_clap_ensemble, embedding_hybrid,
               audio_analysis, segmentation,
               (embedding_hybrid <=> %s::vector) AS distance
        FROM track_query
        ORDER BY embedding_hybrid <=> %s::vector
        LIMIT %s
        """,
        (scene_embedding_hybrid, scene_embedding_hybrid, limit),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_track_query_candidates_by_ensemble(
    cur,
    scene_embedding_ensemble: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch candidate track_query rows using embedding_clap_ensemble similarity."""
    cur.execute(
        """
        SELECT id, filename, filepath, duration_sec, bpm, musical_key,
               semantic_title_en, description_en, tags_summary_en,
               track_music_semantics, track_music_targets,
               track_tag_selection, track_clap_prompt_ensemble,
               embedding_audio, embedding_main, embedding_tags,
               embedding_clap_ensemble, embedding_hybrid,
               audio_analysis, segmentation,
               (embedding_clap_ensemble <=> %s::vector) AS distance
        FROM track_query
        ORDER BY embedding_clap_ensemble <=> %s::vector
        LIMIT %s
        """,
        (scene_embedding_ensemble, scene_embedding_ensemble, limit),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_track_query_rows(
    cur, track_query_ids: list[int]
) -> list[dict[str, Any]]:
    """Fetch full track_query rows for selected ids."""
    if not track_query_ids:
        return []
    cur.execute(
        """
        SELECT id, filename, filepath, duration_sec, bpm, musical_key,
               semantic_title_en, description_en, tags_summary_en,
               track_music_semantics, track_music_targets,
               track_tag_selection, track_clap_prompt_ensemble,
               embedding_audio, embedding_main, embedding_tags,
               embedding_clap_ensemble, embedding_hybrid,
               audio_analysis, segmentation
        FROM track_query
        WHERE id = ANY(%s)
        """,
        (track_query_ids,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_existing_episode_matches(
    cur, episode_nr: int
) -> list[dict[str, Any]]:
    """Fetch existing scene_music_matches_v6 rows for one episode."""
    ensure_scene_music_matches_v6_exists(cur)
    cur.execute(
        "SELECT * FROM scene_music_matches_v6 WHERE episode_nr=%s "
        "ORDER BY scene_nr, theme_nr, rank_position",
        (episode_nr,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_pdf_episode_numbers(cur) -> list[int]:
    """Return distinct episode numbers from scene_music_matches_v6, newest first."""
    ensure_scene_music_matches_v6_exists(cur)
    cur.execute(
        "SELECT DISTINCT episode_nr FROM scene_music_matches_v6 ORDER BY episode_nr DESC"
    )
    return [int(r[0]) for r in cur.fetchall()]


def fetch_matches_as_ranked_v6(cur, episode_nr: int) -> list[RankedMatchV6]:
    """Reconstruct RankedMatchV6 objects from saved scene_music_matches_v6 rows."""
    rows = fetch_existing_episode_matches(cur, episode_nr)
    matches: list[RankedMatchV6] = []
    for r in rows:
        matches.append(RankedMatchV6(
            episode_nr              = int(r.get("episode_nr", 0)),
            scene_query_id          = int(r.get("scene_query_id", 0)),
            scene_nr                = int(r.get("scene_nr", 0)),
            theme_nr                = int(r.get("theme_nr", 0)),
            track_query_id          = int(r.get("track_query_id", 0)),
            rank_position           = int(r.get("rank_position", 0)),
            scene_theme_title_en    = str(r.get("scene_theme_title_en") or ""),
            track_semantic_title_en = str(r.get("track_semantic_title_en") or ""),
            track_filename          = str(r.get("track_filename") or ""),
            track_filepath          = str(r.get("track_filepath") or ""),
            duration_sec    = float(r["duration_sec"]) if r.get("duration_sec") is not None else None,
            bpm             = int(r["bpm"])            if r.get("bpm")           is not None else None,
            musical_key     = str(r["musical_key"])    if r.get("musical_key")   is not None else None,
            final_score                    = float(r.get("final_score",          0.0)),
            main_similarity                = float(r.get("main_similarity",       0.0)),
            tags_similarity                = float(r.get("tags_similarity",       0.0)),
            ensemble_similarity            = float(r.get("ensemble_similarity",   0.0)),
            hybrid_similarity              = float(r.get("hybrid_similarity",     0.0)),
            audio_similarity_aux           = float(r.get("audio_similarity_aux",  0.0)),
            embedding_score                = float(r.get("embedding_score",       0.0)),
            semantic_score                 = float(r.get("semantic_score",        0.0)),
            targets_score                  = float(r.get("targets_score",         0.0)),
            tag_selection_score            = float(r.get("tag_selection_score",   0.0)),
            dialogue_score                 = float(r.get("dialogue_score",        0.0)),
            dialogue_conflict              = float(r.get("dialogue_conflict",     0.0)),
            duration_conflict              = float(r.get("duration_conflict",     0.0)),
            forbidden_tag_conflict         = float(r.get("forbidden_tag_conflict",0.0)),
            style_redundancy_penalty       = float(r.get("style_redundancy_penalty",       0.0)),
            same_track_consecutive_penalty = float(r.get("same_track_consecutive_penalty", 0.0)),
            missing_data_penalty           = float(r.get("missing_data_penalty",  0.0)),
            penalty_total                  = float(r.get("penalty_total",         0.0)),
            style_signature                = str(r.get("style_signature") or ""),
            dialogue_safe_applied          = bool(r.get("dialogue_safe_applied", False)),
            match_metadata                 = dict(r.get("match_metadata") or {}),
            match_explanation              = dict(r.get("match_explanation") or {}),
        ))
    return matches


def delete_episode_matches_v6(cur, episode_nr: int) -> None:
    """Delete existing scene_music_matches_v6 rows for one episode before overwrite."""
    ensure_scene_music_matches_v6_exists(cur)
    cur.execute("DELETE FROM scene_music_matches_v6 WHERE episode_nr=%s", (episode_nr,))


def fetch_track_query_count(cur) -> int:
    """Return total number of rows currently available in track_query."""
    cur.execute("SELECT COUNT(*) FROM track_query")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def fetch_episode_options(cur) -> list[dict[str, Any]]:
    """Return all episodes from scene_query DESC, enriched with match status.

    Each entry: {'episode_nr': int, 'matched': bool, 'label': str}.
    """
    cur.execute(
        "SELECT DISTINCT episode_nr FROM scene_query ORDER BY episode_nr DESC"
    )
    episodes = [int(r[0]) for r in cur.fetchall()]
    if not episodes:
        return []
    matched_set: set[int] = set()
    try:
        cur.execute("SELECT DISTINCT episode_nr FROM scene_music_matches_v6")
        matched_set = {int(r[0]) for r in cur.fetchall()}
    except Exception:
        pass  # table may not exist yet
    return [
        {
            "episode_nr": ep,
            "matched":    ep in matched_set,
            "label":      f"Episode {ep} - matched" if ep in matched_set else f"Episode {ep}",
        }
        for ep in episodes
    ]


def fetch_recent_episode_numbers(cur, limit: int = 10) -> list[int]:
    """Backward-compatible wrapper — returns episode numbers only, newest first."""
    opts = fetch_episode_options(cur)
    return [o["episode_nr"] for o in opts[:limit]]


# ============================================================
# NORMALIZERS / VALIDATORS
# ============================================================

def parse_vector(value: Any) -> list[float]:
    """Parse a DB vector representation into a Python list of floats.

    Handles: list/tuple, str (pgvector text "[...]"), memoryview,
    objects with .tolist() (pgvector native / numpy).
    """
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return [float(x) for x in value.tolist()]
    if isinstance(value, memoryview):
        value = bytes(value).decode("utf-8")
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    if isinstance(value, str):
        txt = value.strip().strip("[]()")
        if not txt:
            return []
        return [float(x.strip()) for x in txt.split(",") if x.strip()]
    try:
        return [float(x) for x in value]
    except Exception:
        raise ValueError(f"Cannot parse vector from value type={type(value)}")


def normalize_jsonb_object(value: Any, field_name: str) -> dict[str, Any]:
    """Normalize a JSONB field into a Python dict and validate its shape."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        obj = json.loads(value)
        if not isinstance(obj, dict):
            raise ValueError(f"{field_name} must be a JSON object")
        return obj
    raise ValueError(f"{field_name} must be a dict-like JSON object")


def normalize_scene_query_row(row: dict[str, Any]) -> NormalizedSceneQuery:
    """Convert one raw scene_query row into NormalizedSceneQuery."""
    return NormalizedSceneQuery(
        id           = int(row["id"]),
        episode_nr   = int(row["episode_nr"]),
        scene_nr     = int(row["scene_nr"]),
        theme_nr     = int(row["theme_nr"]),
        theme_title_pl  = str(row.get("theme_title_pl")  or ""),
        theme_title_en  = str(row.get("theme_title_en")  or ""),
        description_en  = str(row.get("description_en")  or ""),
        theme_txt       = str(row.get("theme_txt")        or ""),
        tags_summary_en = str(row.get("tags_summary_en")  or ""),
        scene_music_semantics = normalize_jsonb_object(
            row.get("scene_music_semantics"), "scene_music_semantics"
        ),
        scene_music_targets   = normalize_jsonb_object(
            row.get("scene_music_targets"),   "scene_music_targets"
        ),
        scene_tag_selection   = normalize_jsonb_object(
            row.get("scene_tag_selection"),   "scene_tag_selection"
        ),
        clap_prompt_ensemble  = normalize_jsonb_object(
            row.get("clap_prompt_ensemble"),  "clap_prompt_ensemble"
        ),
        embedding_main          = parse_vector(row.get("embedding_main")),
        embedding_tags          = parse_vector(row.get("embedding_tags")),
        embedding_clap_ensemble = parse_vector(row.get("embedding_clap_ensemble")),
        embedding_hybrid        = parse_vector(row.get("embedding_hybrid")),
    )


def normalize_track_query_row(row: dict[str, Any]) -> NormalizedTrackQuery:
    """Convert one raw track_query row into NormalizedTrackQuery."""
    return NormalizedTrackQuery(
        id           = int(row["id"]),
        filename     = str(row.get("filename")  or ""),
        filepath     = str(row.get("filepath")  or ""),
        duration_sec = float(row.get("duration_sec") or 0.0),
        bpm          = int(row["bpm"])        if row.get("bpm")         is not None else None,
        musical_key  = str(row["musical_key"]) if row.get("musical_key") is not None else None,
        semantic_title_en = str(row.get("semantic_title_en") or ""),
        description_en    = str(row.get("description_en")    or ""),
        tags_summary_en   = str(row.get("tags_summary_en")   or ""),
        track_music_semantics     = normalize_jsonb_object(
            row.get("track_music_semantics"),     "track_music_semantics"
        ),
        track_music_targets       = normalize_jsonb_object(
            row.get("track_music_targets"),       "track_music_targets"
        ),
        track_tag_selection       = normalize_jsonb_object(
            row.get("track_tag_selection"),       "track_tag_selection"
        ),
        track_clap_prompt_ensemble = normalize_jsonb_object(
            row.get("track_clap_prompt_ensemble"), "track_clap_prompt_ensemble"
        ),
        embedding_audio         = parse_vector(row.get("embedding_audio")),
        embedding_main          = parse_vector(row.get("embedding_main")),
        embedding_tags          = parse_vector(row.get("embedding_tags")),
        embedding_clap_ensemble = parse_vector(row.get("embedding_clap_ensemble")),
        embedding_hybrid        = parse_vector(row.get("embedding_hybrid")),
        audio_analysis = normalize_jsonb_object(row.get("audio_analysis"), "audio_analysis"),
        segmentation   = normalize_jsonb_object(row.get("segmentation"),   "segmentation"),
    )


def validate_scene_query_record(scene: NormalizedSceneQuery) -> list[str]:
    """Return a list of validation issues for one normalized scene row."""
    issues: list[str] = []
    if not scene.theme_title_en:
        issues.append("missing theme_title_en")
    if not scene.description_en:
        issues.append("missing description_en")
    if not scene.embedding_hybrid:
        issues.append("missing embedding_hybrid")
    should  = set(scene.scene_tag_selection.get("should_have_tags", []) or [])
    must    = set(scene.scene_tag_selection.get("must_not_tags",    []) or [])
    overlap = should.intersection(must)
    if overlap:
        issues.append(f"overlapping scene tags: {sorted(overlap)}")
    return issues


def validate_track_query_record(track: NormalizedTrackQuery) -> list[str]:
    """Return a list of validation issues for one normalized track row."""
    issues: list[str] = []
    if not track.semantic_title_en:
        issues.append("missing semantic_title_en")
    if not track.embedding_hybrid:
        issues.append("missing embedding_hybrid")
    should  = set(track.track_tag_selection.get("should_have_tags", []) or [])
    must    = set(track.track_tag_selection.get("must_not_tags",    []) or [])
    overlap = should.intersection(must)
    if overlap:
        issues.append(f"overlapping track tags: {sorted(overlap)}")
    return issues


def build_input_validation_report(
    scenes: list[NormalizedSceneQuery],
    tracks: list[NormalizedTrackQuery],
) -> dict[str, Any]:
    """Build a high-level validation report for GUI preview."""
    scene_issues = sum((validate_scene_query_record(s) for s in scenes), [])
    track_issues = sum((validate_track_query_record(t) for t in tracks), [])
    return {
        "scene_count":        len(scenes),
        "track_count":        len(tracks),
        "scene_issue_count":  len(scene_issues),
        "track_issue_count":  len(track_issues),
        "scene_issue_samples": scene_issues[:10],
        "track_issue_samples": track_issues[:10],
    }


# ============================================================
# RETRIEVAL
# ============================================================

def union_candidate_ids(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    max_total: int,
) -> list[int]:
    """Merge candidate ids from multiple retrieval passes, preserving order."""
    out:  list[int] = []
    seen: set[int]  = set()
    for row in rows_a + rows_b:
        rid = int(row["id"])
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
        if len(out) >= max_total:
            break
    return out


def filter_candidates_minimum_validity(
    candidates: list[NormalizedTrackQuery],
) -> list[NormalizedTrackQuery]:
    """Remove candidates that lack embedding_hybrid (minimum contract)."""
    return [
        c for c in candidates
        if "missing embedding_hybrid" not in validate_track_query_record(c)
    ]


def retrieve_track_candidates_for_scene(
    cur,
    scene: NormalizedSceneQuery,
    candidate_pool_hybrid:   int,
    candidate_pool_ensemble: int,
    candidate_union_limit:   int,
) -> list[NormalizedTrackQuery]:
    """Retrieve and normalize track candidates for one scene."""
    rows_h = fetch_track_query_candidates_by_hybrid(
        cur, scene.embedding_hybrid, candidate_pool_hybrid
    )
    rows_e = fetch_track_query_candidates_by_ensemble(
        cur, scene.embedding_clap_ensemble, candidate_pool_ensemble
    )
    candidate_ids = union_candidate_ids(rows_h, rows_e, candidate_union_limit)
    rows = fetch_track_query_rows(cur, candidate_ids)
    return filter_candidates_minimum_validity(
        [normalize_track_query_row(r) for r in rows]
    )


# ============================================================
# SCORING: EMBEDDINGS
# ============================================================

def cosine_similarity_safe(a: list[float], b: list[float]) -> float:
    """Safely compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def compute_embedding_similarities(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    enable_audio_aux: bool,
) -> dict[str, float]:
    """Compute all embedding-level similarities for scene ↔ track."""
    return {
        "main_similarity":     cosine_similarity_safe(scene.embedding_main,          track.embedding_main),
        "tags_similarity":     cosine_similarity_safe(scene.embedding_tags,          track.embedding_tags),
        "ensemble_similarity": cosine_similarity_safe(scene.embedding_clap_ensemble, track.embedding_clap_ensemble),
        "hybrid_similarity":   cosine_similarity_safe(scene.embedding_hybrid,        track.embedding_hybrid),
        "audio_similarity_aux": (
            cosine_similarity_safe(scene.embedding_hybrid, track.embedding_audio)
            if enable_audio_aux else 0.0
        ),
    }


def compute_embedding_score(
    similarities: dict[str, float],
    weights: dict[str, Any],
) -> float:
    """Aggregate embedding similarities into one embedding_score."""
    w     = (weights or {}).get("embedding", {})
    main_w    = float(w.get("main",      0.15))
    tags_w    = float(w.get("tags",      0.10))
    ensemble_w = float(w.get("ensemble", 0.20))
    hybrid_w  = float(w.get("hybrid",   0.50))
    audio_w   = float(w.get("audio_aux",0.05))
    total = main_w + tags_w + ensemble_w + hybrid_w + audio_w
    if total <= 0:
        return 0.0
    score = (
        main_w     * float(similarities.get("main_similarity",     0.0)) +
        tags_w     * float(similarities.get("tags_similarity",     0.0)) +
        ensemble_w * float(similarities.get("ensemble_similarity", 0.0)) +
        hybrid_w   * float(similarities.get("hybrid_similarity",   0.0)) +
        audio_w    * float(similarities.get("audio_similarity_aux",0.0))
    ) / total
    return max(0.0, min(1.0, score))


# ============================================================
# SCORING: SEMANTICS
# ============================================================

def compute_emotional_direction_score(
    scene_semantics: dict[str, Any],
    track_semantics: dict[str, Any],
) -> float:
    """Compare emotional_direction blocks."""
    scene_vals = set(scene_semantics.get("emotional_direction", []) or [])
    track_vals = set(track_semantics.get("emotional_direction", []) or [])
    if not scene_vals or not track_vals:
        return 0.0
    return len(scene_vals.intersection(track_vals)) / max(1, len(scene_vals))


def compute_narrative_function_score(
    scene_semantics: dict[str, Any],
    track_semantics: dict[str, Any],
) -> float:
    """Compare narrative_function blocks."""
    scene_vals = set(scene_semantics.get("narrative_function", []) or [])
    track_vals = set(track_semantics.get("narrative_function", []) or [])
    if not scene_vals or not track_vals:
        return 0.0
    return len(scene_vals.intersection(track_vals)) / max(1, len(scene_vals))


def compute_weight_profile_score(
    scene_semantics: dict[str, Any],
    track_semantics: dict[str, Any],
) -> float:
    """Compare scene weight_profile against track weight_profile_candidates."""
    scene_value = str(scene_semantics.get("weight_profile", "") or "").strip()
    candidates  = set(track_semantics.get("weight_profile_candidates", []) or [])
    if not scene_value:
        return 0.0
    return 1.0 if scene_value in candidates else 0.0


def compute_dialogue_score(
    scene_semantics: dict[str, Any],
    track_semantics: dict[str, Any],
) -> float:
    """Compare dialogue-safe requirements between scene and track."""
    required = bool(scene_semantics.get("dialogue_safe_required", False))
    provided = bool(track_semantics.get("dialogue_safe", False))
    if required:
        return 1.0 if provided else 0.0
    return 1.0


def compute_semantic_score(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    weights: dict[str, Any],
) -> dict[str, Any]:
    """Compute full semantic component scoring bundle."""
    scene_sem = scene.scene_music_semantics or {}
    track_sem = track.track_music_semantics or {}
    emotion_match_score        = compute_emotional_direction_score(scene_sem, track_sem)
    narrative_match_score      = compute_narrative_function_score(scene_sem, track_sem)
    weight_profile_match_score = compute_weight_profile_score(scene_sem, track_sem)
    dialogue_match_score       = compute_dialogue_score(scene_sem, track_sem)
    w = (weights or {}).get("semantic", {})
    score = (
        float(w.get("emotion",        0.35)) * emotion_match_score        +
        float(w.get("narrative",      0.25)) * narrative_match_score      +
        float(w.get("weight_profile", 0.20)) * weight_profile_match_score +
        float(w.get("dialogue",       0.20)) * dialogue_match_score
    )
    return {
        "emotion_match_score":        emotion_match_score,
        "narrative_match_score":      narrative_match_score,
        "weight_profile_match_score": weight_profile_match_score,
        "dialogue_match_score":       dialogue_match_score,
        "semantic_score":             max(0.0, min(1.0, score)),
    }


# ============================================================
# SCORING: TARGETS
# ============================================================

def compute_scalar_target_score(
    scene_value: str,
    track_value: str,
    ordered_scale: Optional[list[str]] = None,
) -> float:
    """Compare one scalar target value, optionally with ordered-distance fallback.

    Penalty per step is 0.25 so adjacent values score 0.75, two steps 0.50,
    four+ steps reach 0.0.
    """
    scene_value = (scene_value or "").strip()
    track_value = (track_value or "").strip()
    if not scene_value or not track_value:
        return 0.0
    if scene_value == track_value:
        return 1.0
    if (ordered_scale
            and scene_value in ordered_scale
            and track_value in ordered_scale):
        dist = abs(ordered_scale.index(scene_value) - ordered_scale.index(track_value))
        return max(0.0, 1.0 - 0.25 * dist)
    return 0.0


def compute_sound_character_score(
    scene_targets: dict[str, Any],
    track_targets: dict[str, Any],
) -> float:
    """Compare sound_character_target lists (Jaccard similarity)."""
    scene_vals = set(scene_targets.get("sound_character_target", []) or [])
    track_vals = set(track_targets.get("sound_character_target", []) or [])
    if not scene_vals or not track_vals:
        return 0.0
    inter = len(scene_vals.intersection(track_vals))
    union = len(scene_vals.union(track_vals))
    return inter / max(1, union)


def compute_targets_score(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    weights: dict[str, Any],
) -> dict[str, Any]:
    """Compute full targets component scoring bundle."""
    scene_tgt = scene.scene_music_targets or {}
    track_tgt = track.track_music_targets or {}
    energy_match_score = compute_scalar_target_score(
        scene_tgt.get("energy_target", ""), track_tgt.get("energy_target", ""),
        ["very_low", "low", "medium", "high", "very_high"],
    )
    tempo_match_score = compute_scalar_target_score(
        scene_tgt.get("tempo_target", ""), track_tgt.get("tempo_target", ""),
        ["very_slow", "slow", "moderate", "fast", "very_fast"],
    )
    rhythm_match_score = compute_scalar_target_score(
        scene_tgt.get("rhythm_target", ""), track_tgt.get("rhythm_target", ""), None,
    )
    intensity_shape_match_score = compute_scalar_target_score(
        scene_tgt.get("intensity_shape_target", ""),
        track_tgt.get("intensity_shape_target", ""),
        ["static", "gradual_build", "crescendo", "peak", "climax_peak",
         "drop", "release", "wave", "pulsating"],
    )
    sound_character_match_score = compute_sound_character_score(scene_tgt, track_tgt)
    w = (weights or {}).get("targets", {})
    score = (
        float(w.get("energy",          0.20)) * energy_match_score          +
        float(w.get("tempo",           0.20)) * tempo_match_score           +
        float(w.get("rhythm",          0.15)) * rhythm_match_score          +
        float(w.get("intensity_shape", 0.20)) * intensity_shape_match_score +
        float(w.get("sound_character", 0.25)) * sound_character_match_score
    )
    return {
        "energy_match_score":          energy_match_score,
        "tempo_match_score":           tempo_match_score,
        "rhythm_match_score":          rhythm_match_score,
        "intensity_shape_match_score": intensity_shape_match_score,
        "sound_character_match_score": sound_character_match_score,
        "targets_score":               max(0.0, min(1.0, score)),
    }


# ============================================================
# SCORING: TAG SELECTION
# ============================================================

def extract_scene_should_have(scene: NormalizedSceneQuery) -> list[str]:
    return list((scene.scene_tag_selection or {}).get("should_have_tags", []) or [])


def extract_scene_must_not(scene: NormalizedSceneQuery) -> list[str]:
    return list((scene.scene_tag_selection or {}).get("must_not_tags", []) or [])


def extract_track_should_have(track: NormalizedTrackQuery) -> list[str]:
    return list((track.track_tag_selection or {}).get("should_have_tags", []) or [])


def compute_tag_selection_score(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    weights: dict[str, Any],
) -> dict[str, Any]:
    """Compute tag-selection matching and conflicts."""
    should      = set(extract_scene_should_have(scene))
    must_not    = set(extract_scene_must_not(scene))
    track_should = set(extract_track_should_have(track))
    should_have_hit_rate  = (
        len(should.intersection(track_should)) / max(1, len(should))
        if should else 0.0
    )
    must_not_conflict_rate = (
        len(must_not.intersection(track_should)) / max(1, len(must_not))
        if must_not else 0.0
    )
    w = (weights or {}).get("tag_selection", {})
    score = (
        float(w.get("should_have", 0.75)) * should_have_hit_rate
        - float(w.get("must_not",  0.25)) * must_not_conflict_rate
    )
    return {
        "should_have_hit_rate":  should_have_hit_rate,
        "must_not_conflict_rate": must_not_conflict_rate,
        "tag_selection_score":   max(0.0, min(1.0, score)),
    }


# ============================================================
# PENALTIES / CONSTRAINTS
# ============================================================

def compute_dialogue_safe_penalty(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    strict_dialogue_safe: bool,
) -> float:
    """Penalty for dialogue-safe mismatch."""
    required = bool((scene.scene_music_semantics or {}).get("dialogue_safe_required", False))
    provided = bool((track.track_music_semantics or {}).get("dialogue_safe", False))
    if required and not provided:
        return 0.20 if strict_dialogue_safe else 0.10
    return 0.0


def compute_duration_penalty(
    scene: NormalizedSceneQuery,  # noqa: ARG001 — kept for API symmetry
    track: NormalizedTrackQuery,
) -> float:
    """Penalty for problematic track duration."""
    dur = float(track.duration_sec or 0.0)
    if dur <= 0:   return 0.10
    if dur < 30:   return 0.15
    if dur < 60:   return 0.05
    return 0.0


def compute_forbidden_tag_penalty(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
) -> float:
    """Penalty for forbidden tag overlap."""
    must_not     = set(extract_scene_must_not(scene))
    track_should = set(extract_track_should_have(track))
    conflicts    = len(must_not.intersection(track_should))
    return min(0.25, 0.08 * conflicts)


def build_style_signature(track: NormalizedTrackQuery) -> str:
    """Build style signature used for redundancy control."""
    sem = track.track_music_semantics or {}
    tgt = track.track_music_targets   or {}
    parts = [
        str((sem.get("weight_profile_candidates", []) or [""])[0]),
        str(tgt.get("energy_target",          "")),
        str(tgt.get("tempo_target",           "")),
        str(tgt.get("intensity_shape_target", "")),
        "-".join(sorted((tgt.get("sound_character_target", []) or [])[:3])),
    ]
    return "|".join(parts)


def compute_style_redundancy_penalty(
    style_signature: str,
    already_used_signatures: list[str],
    style_duplicate_limit: int,
) -> float:
    """Penalty for repeating very similar style signatures."""
    if not style_signature:
        return 0.0
    count = sum(1 for s in already_used_signatures if s == style_signature)
    return 0.10 * max(0, count - max(0, style_duplicate_limit - 1))


def compute_same_track_consecutive_penalty(
    track: NormalizedTrackQuery,
    previous_scene_track_query_ids: list[int],
) -> float:
    """Penalty for reusing the same track in consecutive scenes."""
    return 0.12 if track.id in set(previous_scene_track_query_ids or []) else 0.0


def compute_missing_data_penalty(
    scene_errors: list[str],
    track_errors: list[str],
) -> float:
    """Penalty for missing critical scene/track data.

    Floor of 0.05 applied whenever any validation error is present so that even
    a single missing field produces a visible penalty.
    """
    count = len(scene_errors) + len(track_errors)
    if count <= 0:
        return 0.0
    return min(0.25, max(0.05, 0.02 * count))


def compute_penalties(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    context: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute all penalties and aggregate penalty_total."""
    scene_errors = validate_scene_query_record(scene)
    track_errors = validate_track_query_record(track)
    style_signature = build_style_signature(track)
    already_used_signatures        = list(context.get("already_used_signatures", []))
    previous_scene_track_query_ids = list(context.get("previous_scene_track_query_ids", []))

    dialogue_conflict              = compute_dialogue_safe_penalty(
        scene, track, bool(config.get("strict_dialogue_safe", False))
    )
    duration_conflict              = compute_duration_penalty(scene, track)
    forbidden_tag_conflict         = compute_forbidden_tag_penalty(scene, track)
    style_redundancy_penalty       = compute_style_redundancy_penalty(
        style_signature, already_used_signatures,
        int(config.get("style_duplicate_limit", 2))
    )
    same_track_consecutive_penalty = compute_same_track_consecutive_penalty(
        track, previous_scene_track_query_ids
    )
    missing_data_penalty           = compute_missing_data_penalty(scene_errors, track_errors)
    penalty_total = sum([
        dialogue_conflict, duration_conflict, forbidden_tag_conflict,
        style_redundancy_penalty, same_track_consecutive_penalty, missing_data_penalty,
    ])
    return {
        "dialogue_conflict":              dialogue_conflict,
        "duration_conflict":              duration_conflict,
        "forbidden_tag_conflict":         forbidden_tag_conflict,
        "style_redundancy_penalty":       style_redundancy_penalty,
        "same_track_consecutive_penalty": same_track_consecutive_penalty,
        "missing_data_penalty":           missing_data_penalty,
        "penalty_total":                  penalty_total,
        "style_signature":                style_signature,
        "dialogue_safe_applied": bool(
            (scene.scene_music_semantics or {}).get("dialogue_safe_required", False)
        ),
    }


# ============================================================
# RANKING ORCHESTRATION
# ============================================================

def build_match_explanation(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    component_scores: dict[str, Any],
    penalties: dict[str, Any],
) -> dict[str, Any]:
    """Create structured explanation JSON for one ranked match."""
    return {
        "scene": {
            "scene_query_id": scene.id,
            "theme_title_en": scene.theme_title_en,
        },
        "track": {
            "track_query_id":   track.id,
            "semantic_title_en": track.semantic_title_en,
            "filename":         track.filename,
        },
        "component_scores": component_scores,
        "penalties":        penalties,
    }


def rank_one_candidate_v6(
    scene: NormalizedSceneQuery,
    track: NormalizedTrackQuery,
    context: dict[str, Any],
    config: dict[str, Any],
) -> RankedMatchV6:
    """Score one scene ↔ track pair and return RankedMatchV6."""
    emb             = compute_embedding_similarities(scene, track, bool(config.get("enable_audio_aux", True)))
    embedding_score = compute_embedding_score(emb, config.get("weights", {}))
    sem             = compute_semantic_score(scene, track, config.get("weights", {}))
    tgt             = compute_targets_score(scene, track, config.get("weights", {}))
    tag             = compute_tag_selection_score(scene, track, config.get("weights", {}))
    penalties       = compute_penalties(scene, track, context, config)

    fw    = (config.get("weights") or {}).get("final", {})
    w_emb = float(fw.get("embedding",     0.40))
    w_sem = float(fw.get("semantic",      0.25))
    w_tgt = float(fw.get("targets",       0.20))
    w_tag = float(fw.get("tag_selection", 0.15))
    final_score = max(0.0, min(1.0,
        w_emb * embedding_score       +
        w_sem * sem["semantic_score"] +
        w_tgt * tgt["targets_score"]  +
        w_tag * tag["tag_selection_score"] -
        penalties["penalty_total"]
    ))
    component_scores = {**emb, "embedding_score": embedding_score, **sem, **tgt, **tag}
    explanation = build_match_explanation(scene, track, component_scores, penalties)
    return RankedMatchV6(
        episode_nr              = scene.episode_nr,
        scene_query_id          = scene.id,
        scene_nr                = scene.scene_nr,
        theme_nr                = scene.theme_nr,
        track_query_id          = track.id,
        rank_position           = 0,
        scene_theme_title_en    = scene.theme_title_en,
        track_semantic_title_en = track.semantic_title_en,
        track_filename          = track.filename,
        track_filepath          = track.filepath,
        duration_sec            = track.duration_sec,
        bpm                     = track.bpm,
        musical_key             = track.musical_key,
        final_score             = final_score,
        main_similarity                = emb["main_similarity"],
        tags_similarity                = emb["tags_similarity"],
        ensemble_similarity            = emb["ensemble_similarity"],
        hybrid_similarity              = emb["hybrid_similarity"],
        audio_similarity_aux           = emb["audio_similarity_aux"],
        embedding_score                = embedding_score,
        semantic_score                 = sem["semantic_score"],
        targets_score                  = tgt["targets_score"],
        tag_selection_score            = tag["tag_selection_score"],
        dialogue_score                 = sem["dialogue_match_score"],
        dialogue_conflict              = penalties["dialogue_conflict"],
        duration_conflict              = penalties["duration_conflict"],
        forbidden_tag_conflict         = penalties["forbidden_tag_conflict"],
        style_redundancy_penalty       = penalties["style_redundancy_penalty"],
        same_track_consecutive_penalty = penalties["same_track_consecutive_penalty"],
        missing_data_penalty           = penalties["missing_data_penalty"],
        penalty_total                  = penalties["penalty_total"],
        style_signature                = penalties["style_signature"],
        dialogue_safe_applied          = penalties["dialogue_safe_applied"],
        match_metadata  = {"scene_query_id": scene.id, "track_query_id": track.id},
        match_explanation = explanation,
    )


def rank_candidates_v6(
    scene: NormalizedSceneQuery,
    candidates: list[NormalizedTrackQuery],
    context: dict[str, Any],
    config: dict[str, Any],
) -> list[RankedMatchV6]:
    """Rank all candidate tracks for one scene."""
    ranked = [rank_one_candidate_v6(scene, track, context, config) for track in candidates]
    ranked.sort(
        key=lambda x: (
            x.final_score, x.embedding_score,
            x.semantic_score, x.targets_score, x.hybrid_similarity,
        ),
        reverse=True,
    )
    for idx, item in enumerate(ranked, start=1):
        item.rank_position = idx
    return ranked


def match_episode_v6(
    cur,
    episode_nr:   int,
    config:       dict[str, Any],
    preview_only: bool = True,
    on_progress:  Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, Any]:
    """Main batch orchestrator for all scene_query rows of one episode.

    *on_progress* is an optional callback ``(done: int, total: int, text: str) -> None``
    so the GUI can display progress without coupling this function to Streamlit.
    """
    scene_rows = fetch_scene_queries_for_episode(cur, episode_nr)
    scenes     = [normalize_scene_query_row(r) for r in scene_rows]
    all_matches:   list[RankedMatchV6]      = []
    preview_rows:  list[dict[str, Any]]     = []
    telemetry = {
        "scene_count":              len(scenes),
        "scenes_processed":         0,
        "scene_without_candidates": 0,
    }
    previous_top_track_query_ids: list[int] = []
    already_used_signatures:      list[str] = []
    preview_top_n = int(config.get("preview_top_n", 5))
    rank_top_n    = int(config.get("rank_top_n",    10))

    for idx, scene in enumerate(scenes):
        if on_progress:
            on_progress(idx, len(scenes), f"Scene {scene.scene_nr} / theme {scene.theme_nr}…")
        candidates = retrieve_track_candidates_for_scene(
            cur, scene,
            int(config.get("candidate_pool_hybrid",  200)),
            int(config.get("candidate_pool_ensemble", 80)),
            int(config.get("candidate_union_limit",  240)),
        )
        if not candidates:
            telemetry["scene_without_candidates"] += 1
            telemetry["scenes_processed"]         += 1
            continue
        context = {
            "previous_scene_track_query_ids": previous_top_track_query_ids,
            "already_used_signatures":        already_used_signatures,
        }
        ranked    = rank_candidates_v6(scene, candidates, context, config)
        top_ranked = ranked[:rank_top_n]
        all_matches.extend(top_ranked)
        for rm in ranked[:preview_top_n]:
            preview_rows.append({
                "scene_query_id":          rm.scene_query_id,
                "scene_nr":                rm.scene_nr,
                "theme_nr":                rm.theme_nr,
                "theme_title_en":          rm.scene_theme_title_en,
                "track_query_id":          rm.track_query_id,
                "track_filename":          rm.track_filename,
                "track_semantic_title_en": rm.track_semantic_title_en,
                "rank_position":           rm.rank_position,
                "final_score":             rm.final_score,
                "embedding_score":         rm.embedding_score,
                "semantic_score":          rm.semantic_score,
                "targets_score":           rm.targets_score,
                "tag_selection_score":     rm.tag_selection_score,
                "penalty_total":           rm.penalty_total,
                "style_signature":         rm.style_signature,
            })
        if ranked:
            previous_top_track_query_ids = [ranked[0].track_query_id]
            already_used_signatures.append(ranked[0].style_signature)
        telemetry["scenes_processed"] += 1

    if on_progress:
        on_progress(len(scenes), len(scenes), "Completed.")

    low_confidence = extract_low_confidence_matches(
        all_matches, float(config.get("low_confidence_threshold", 0.45))
    )
    summary = build_episode_summary(all_matches, telemetry)
    return {
        "episode_nr":    episode_nr,
        "preview_only":  preview_only,
        "matches":       all_matches,
        "preview_rows":  preview_rows,
        "low_confidence": low_confidence,
        "telemetry":     telemetry,
        "summary":       summary,
    }


def match_single_scene_preview(
    cur, scene_query_id: int, config: dict[str, Any]
) -> dict[str, Any]:
    """Run preview ranking for one selected scene_query row."""
    row = fetch_scene_query_by_id(cur, scene_query_id)
    if row is None:
        return {"error": "scene_query not found"}
    scene      = normalize_scene_query_row(row)
    candidates = retrieve_track_candidates_for_scene(
        cur, scene,
        int(config.get("candidate_pool_hybrid",   150)),
        int(config.get("candidate_pool_ensemble",  60)),
        int(config.get("candidate_union_limit",   180)),
    )
    ranked = rank_candidates_v6(
        scene, candidates,
        {"previous_scene_track_query_ids": [], "already_used_signatures": []},
        config,
    )
    return {
        "scene_query_id": scene_query_id,
        "matches":        ranked[: int(config.get("preview_top_n", 5))],
    }


# ============================================================
# PERSISTENCE
# ============================================================

def save_episode_matches_v6(
    cur,
    episode_nr:     int,
    ranked_matches: list[RankedMatchV6],
    overwrite:      bool = True,
) -> int:
    """Persist a full episode ranking batch into scene_music_matches_v6.

    Uses psycopg2.extras.execute_values with server-side parameter binding so
    that % characters inside string values (e.g. Windows paths) never cause
    'not all arguments converted during string formatting' TypeError.
    """
    ensure_scene_music_matches_v6_exists(cur)
    if overwrite:
        delete_episode_matches_v6(cur, episode_nr)
    if not ranked_matches:
        return 0

    def _row(m: RankedMatchV6) -> tuple:
        return (
            m.episode_nr, m.scene_query_id, m.scene_nr, m.theme_nr,
            m.track_query_id, m.rank_position,
            m.scene_theme_title_en, m.track_semantic_title_en,
            m.track_filename, m.track_filepath,
            m.duration_sec, m.bpm, m.musical_key, m.final_score,
            m.main_similarity, m.tags_similarity, m.ensemble_similarity,
            m.hybrid_similarity, m.audio_similarity_aux,
            m.embedding_score, m.semantic_score, m.targets_score,
            m.tag_selection_score, m.dialogue_score,
            m.dialogue_conflict, m.duration_conflict, m.forbidden_tag_conflict,
            m.style_redundancy_penalty, m.same_track_consecutive_penalty,
            m.missing_data_penalty, m.penalty_total,
            m.style_signature, m.dialogue_safe_applied,
            Json(m.match_metadata), Json(m.match_explanation),
        )

    sql = """
        INSERT INTO scene_music_matches_v6 (
            episode_nr, scene_query_id, scene_nr, theme_nr, track_query_id, rank_position,
            scene_theme_title_en, track_semantic_title_en, track_filename, track_filepath,
            duration_sec, bpm, musical_key, final_score,
            main_similarity, tags_similarity, ensemble_similarity, hybrid_similarity, audio_similarity_aux,
            embedding_score, semantic_score, targets_score, tag_selection_score, dialogue_score,
            dialogue_conflict, duration_conflict, forbidden_tag_conflict,
            style_redundancy_penalty, same_track_consecutive_penalty, missing_data_penalty, penalty_total,
            style_signature, dialogue_safe_applied, match_metadata, match_explanation
        ) VALUES %s
        ON CONFLICT (scene_query_id, track_query_id) DO UPDATE SET
            rank_position              = EXCLUDED.rank_position,
            final_score                = EXCLUDED.final_score,
            main_similarity            = EXCLUDED.main_similarity,
            tags_similarity            = EXCLUDED.tags_similarity,
            ensemble_similarity        = EXCLUDED.ensemble_similarity,
            hybrid_similarity          = EXCLUDED.hybrid_similarity,
            audio_similarity_aux       = EXCLUDED.audio_similarity_aux,
            embedding_score            = EXCLUDED.embedding_score,
            semantic_score             = EXCLUDED.semantic_score,
            targets_score              = EXCLUDED.targets_score,
            tag_selection_score        = EXCLUDED.tag_selection_score,
            dialogue_score             = EXCLUDED.dialogue_score,
            dialogue_conflict          = EXCLUDED.dialogue_conflict,
            duration_conflict          = EXCLUDED.duration_conflict,
            forbidden_tag_conflict     = EXCLUDED.forbidden_tag_conflict,
            style_redundancy_penalty   = EXCLUDED.style_redundancy_penalty,
            same_track_consecutive_penalty = EXCLUDED.same_track_consecutive_penalty,
            missing_data_penalty       = EXCLUDED.missing_data_penalty,
            penalty_total              = EXCLUDED.penalty_total,
            style_signature            = EXCLUDED.style_signature,
            dialogue_safe_applied      = EXCLUDED.dialogue_safe_applied,
            match_metadata             = EXCLUDED.match_metadata,
            match_explanation          = EXCLUDED.match_explanation,
            updated_at                 = NOW()
    """
    execute_values(cur, sql, [_row(m) for m in ranked_matches], page_size=100)
    return len(ranked_matches)


def save_scene_matches_v6(
    cur,
    scene_query_id: int,
    ranked_matches: list[RankedMatchV6],
) -> int:
    """Persist ranking rows for one scene_query only."""
    ensure_scene_music_matches_v6_exists(cur)
    count = 0
    for match in ranked_matches:
        if match.scene_query_id == scene_query_id:
            count += save_episode_matches_v6(
                cur, match.episode_nr, [match], overwrite=False
            )
    return count


# ============================================================
# EXPORT
# ============================================================

def build_export_rows_full(matches: list[RankedMatchV6]) -> list[dict[str, Any]]:
    """Build full export rows for XLSX/CSV."""
    rows = []
    for m in matches:
        rows.append({
            "episode_nr":              m.episode_nr,
            "scene_query_id":          m.scene_query_id,
            "scene_nr":                m.scene_nr,
            "theme_nr":                m.theme_nr,
            "rank_position":           m.rank_position,
            "scene_theme_title_en":    m.scene_theme_title_en,
            "track_query_id":          m.track_query_id,
            "track_semantic_title_en": m.track_semantic_title_en,
            "track_filename":          m.track_filename,
            "track_filepath":          m.track_filepath,
            "final_score":             m.final_score,
            "embedding_score":         m.embedding_score,
            "semantic_score":          m.semantic_score,
            "targets_score":           m.targets_score,
            "tag_selection_score":     m.tag_selection_score,
            "dialogue_score":          m.dialogue_score,
            "main_similarity":         m.main_similarity,
            "tags_similarity":         m.tags_similarity,
            "ensemble_similarity":     m.ensemble_similarity,
            "hybrid_similarity":       m.hybrid_similarity,
            "audio_similarity_aux":    m.audio_similarity_aux,
            "dialogue_conflict":       m.dialogue_conflict,
            "duration_conflict":       m.duration_conflict,
            "forbidden_tag_conflict":  m.forbidden_tag_conflict,
            "style_redundancy_penalty":       m.style_redundancy_penalty,
            "same_track_consecutive_penalty": m.same_track_consecutive_penalty,
            "missing_data_penalty":    m.missing_data_penalty,
            "penalty_total":           m.penalty_total,
            "style_signature":         m.style_signature,
            "dialogue_safe_applied":   m.dialogue_safe_applied,
        })
    return [round_floats_3(row) for row in rows]


def build_export_rows_short(
    matches: list[RankedMatchV6], short_top_n: int
) -> list[dict[str, Any]]:
    """Build compact export rows (top-N per scene) for short XLSX."""
    grouped: dict[int, list[RankedMatchV6]] = {}
    for m in matches:
        grouped.setdefault(m.scene_query_id, []).append(m)
    rows = []
    for items in grouped.values():
        items.sort(key=lambda x: x.rank_position)
        for m in items[:short_top_n]:
            rows.append({
                "episode_nr":           m.episode_nr,
                "scene_nr":             m.scene_nr,
                "theme_nr":             m.theme_nr,
                "rank_position":        m.rank_position,
                "scene_theme_title_en": m.scene_theme_title_en,
                "track_filename":       m.track_filename,
                "track_semantic_title_en": m.track_semantic_title_en,
                "final_score":          m.final_score,
            })
    return [round_floats_3(row) for row in rows]


def _xlsx_bytes_from_rows(rows: list[dict[str, Any]]) -> bytes:
    wb = Workbook()
    ws = wb.active
    if not rows:
        ws.append(["no_data"])
    else:
        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([
                json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for v in (row.get(h) for h in headers)
            ])
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def write_excel_full(
    rows: list[dict[str, Any]], output_dir: Path, file_name: str
) -> Path:
    """Write full XLSX export and return file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    path.write_bytes(_xlsx_bytes_from_rows(rows))
    return path


def write_excel_short(
    rows: list[dict[str, Any]], output_dir: Path, file_name: str
) -> Path:
    """Write short XLSX export and return file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    path.write_bytes(_xlsx_bytes_from_rows(rows))
    return path


def export_episode_results_v6(
    result_bundle: dict[str, Any],
    output_dir:    Path,
    episode_nr:    int,
) -> dict[str, Any]:
    """Create XLSX and PDF exports in memory and return download payloads."""
    matches     = result_bundle.get("matches", [])
    full_rows   = build_export_rows_full(matches)
    short_rows  = build_export_rows_short(matches, short_top_n=3)
    full_bytes  = _xlsx_bytes_from_rows(full_rows)
    short_bytes = _xlsx_bytes_from_rows(short_rows)
    pdf_full_bytes  = _pdf_bytes_from_matches(matches, short=False)
    pdf_short_bytes = _pdf_bytes_from_matches(matches, short=True)
    return {
        "full_rows":        full_rows,
        "short_rows":       short_rows,
        "full_bytes":       full_bytes,
        "short_bytes":      short_bytes,
        "pdf_full_bytes":   pdf_full_bytes,
        "pdf_short_bytes":  pdf_short_bytes,
        "full_name":        f"scene_music_matches_v6_episode_{episode_nr}_full.xlsx",
        "short_name":       f"scene_music_matches_v6_episode_{episode_nr}_short.xlsx",
        "pdf_full_name":    f"scene_music_matches_v6_episode_{episode_nr}_full.pdf",
        "pdf_short_name":   f"scene_music_matches_v6_episode_{episode_nr}_short.pdf",
    }


def _group_matches_for_pdf(
    matches: list[RankedMatchV6],
) -> list[tuple[tuple[int, int, int, str], list[RankedMatchV6]]]:
    grouped: dict[tuple[int, int, int, str], list[RankedMatchV6]] = {}
    for m in matches:
        key = (m.episode_nr, m.scene_nr, m.theme_nr, m.scene_theme_title_en or "")
        grouped.setdefault(key, []).append(m)
    ordered = []
    for key in sorted(grouped, key=lambda x: (x[0], x[1], x[2], x[3])):
        ordered.append((key, sorted(grouped[key], key=lambda x: x.rank_position)))
    return ordered


def _pdf_bytes_from_matches(
    matches: list[RankedMatchV6], short: bool = False
) -> bytes:
    """Build PDF bytes in grouped scene/theme layout."""
    bio = io.BytesIO()
    doc = SimpleDocTemplate(
        bio,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles  = getSampleStyleSheet()
    story   = []
    title   = ("Scene Music Matcher v6.1 — PDF short"
               if short else "Scene Music Matcher v6.1 — PDF full")
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 6))

    grouped  = _group_matches_for_pdf(matches)
    per_scene = 3 if short else None

    for (episode_nr, scene_nr, theme_nr, scene_theme_title_en), items in grouped:
        shown  = items[:per_scene] if per_scene else items
        header = f"episode_nr: {episode_nr}   scene_nr: {scene_nr}   theme_nr: {theme_nr}"
        story.append(Paragraph(header, styles["Heading3"]))
        story.append(Paragraph(scene_theme_title_en or "—", styles["BodyText"]))
        story.append(Spacer(1, 3))
        table_data = [["rank_position", "final_score", "track_filename"]]
        for m in shown:
            table_data.append([
                str(m.rank_position),
                f"{m.final_score:.3f}",
                m.track_filename or "",
            ])
        table = Table(table_data, colWidths=[30 * mm, 30 * mm, 120 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.black),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f7f9fc")]),
            ("VALIGN",   (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEADING",  (0, 0), (-1, -1), 11),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    if not grouped:
        story.append(Paragraph("No data.", styles["BodyText"]))

    doc.build(story)
    return bio.getvalue()


def write_pdf_full(
    matches: list[RankedMatchV6], output_dir: Path, file_name: str
) -> Path:
    """Deprecated — no-op stub retained for interface compatibility."""
    import warnings
    warnings.warn(
        "write_pdf_full() is deprecated. Use export_episode_results_v6() instead.",
        DeprecationWarning, stacklevel=2,
    )
    return output_dir / file_name


def write_pdf_short(
    matches: list[RankedMatchV6], output_dir: Path, file_name: str
) -> Path:
    """Deprecated — no-op stub retained for interface compatibility."""
    import warnings
    warnings.warn(
        "write_pdf_short() is deprecated. Use export_episode_results_v6() instead.",
        DeprecationWarning, stacklevel=2,
    )
    return output_dir / file_name


# ============================================================
# AUDIO SOURCE HELPER
# ============================================================

# Module-level S3 client cache keyed by (endpoint_url, access_key_id).
# Avoids re-creating boto3.client() on every audio request in cloud mode.
_s3_client_cache: dict[tuple[str, str], Any] = {}


def get_audio_for_player(
    filepath: str,
    filename: str,
) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Return (audio_bytes, local_path, error_message) for st.audio().

    Cloud mode  — downloads object from Cloudflare R2 (S3-compatible, boto3).
                  S3 client is cached per (endpoint, key) pair.
    Local mode  — returns filesystem path string if the file exists.
    """
    if not filepath and not filename:
        return None, None, "No filepath or filename recorded for this track."

    if CLOUD_MUSIC:
        if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, BUCKET_NAME]):
            return None, None, (
                "Cloud storage is enabled (CLOUD_MUSIC=true) but one or more "
                "R2 credentials are missing in .env "
                "(R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, BUCKET_NAME)."
            )
        if not filepath:
            return None, None, "No filepath recorded for this track in track_query."
        try:
            import boto3
            from botocore.config import Config as BotoConfig

            cache_key = (R2_ENDPOINT_URL, R2_ACCESS_KEY_ID)
            if cache_key not in _s3_client_cache:
                _s3_client_cache[cache_key] = boto3.client(
                    "s3",
                    endpoint_url         = R2_ENDPOINT_URL,
                    aws_access_key_id    = R2_ACCESS_KEY_ID,
                    aws_secret_access_key = R2_SECRET_ACCESS_KEY,
                    config               = BotoConfig(signature_version="s3v4"),
                )
            s3          = _s3_client_cache[cache_key]
            response    = s3.get_object(Bucket=BUCKET_NAME, Key=filepath)
            audio_bytes = response["Body"].read()
            return audio_bytes, None, None
        except Exception as exc:
            return None, None, f"R2 download failed for '{filepath}': {exc}"
    else:
        if not filepath:
            return None, None, "No filepath recorded for this track in track_query."
        local = Path(filepath)
        if local.exists():
            return None, str(local), None
        return None, None, f"Audio file not found on disk: {filepath}"


# ============================================================
# DEBUG / QUALITY VALIDATION
# ============================================================

def debug_compare_scene_track(
    cur,
    scene_query_id: int,
    track_query_id: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute detailed debug comparison for one scene_query ↔ track_query pair."""
    scene_row  = fetch_scene_query_by_id(cur, scene_query_id)
    track_rows = fetch_track_query_rows(cur, [track_query_id])
    if scene_row is None or not track_rows:
        return {"error": "scene_query or track_query not found"}
    scene     = normalize_scene_query_row(scene_row)
    track     = normalize_track_query_row(track_rows[0])
    emb_sims  = compute_embedding_similarities(scene, track, bool(config.get("enable_audio_aux", True)))
    emb_score = compute_embedding_score(emb_sims, config.get("weights", {}))
    sem_bundle = compute_semantic_score(scene, track, config.get("weights", {}))
    tgt_bundle = compute_targets_score(scene, track, config.get("weights", {}))
    tag_bundle = compute_tag_selection_score(scene, track, config.get("weights", {}))
    penalties  = compute_penalties(
        scene, track,
        {"previous_scene_track_query_ids": [], "already_used_signatures": []},
        config,
    )
    return {
        "how_it_works": {
            "embedding":     "Compares main, tags, ensemble, hybrid and optional audio-aux vectors.",
            "semantic":      "Compares emotional_direction, narrative_function, weight_profile and dialogue safety.",
            "targets":       "Compares energy, tempo, rhythm, intensity_shape and sound_character targets.",
            "tag_selection": "Rewards should-have overlaps and penalizes must-not conflicts.",
            "penalties":     "Adds dialogue, duration, forbidden-tag, style redundancy, consecutive-track and missing-data penalties.",
        },
        "scene":       scene_row,
        "track":       track_rows[0],
        "embedding":   {**emb_sims, "embedding_score": emb_score},
        "semantic":    sem_bundle,
        "targets":     tgt_bundle,
        "tag_selection": tag_bundle,
        "penalties":   penalties,
    }


def extract_low_confidence_matches(
    matches: list[RankedMatchV6], threshold: float
) -> list[RankedMatchV6]:
    """Return matches below the configured confidence threshold."""
    return [m for m in matches if m.final_score < threshold]


def build_episode_summary(
    matches: list[RankedMatchV6],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    """Build final episode summary for GUI and exports."""
    if not matches:
        return {"match_count": 0, **telemetry}
    return {
        "match_count":        len(matches),
        "avg_final_score":    float(sum(m.final_score    for m in matches) / len(matches)),
        "avg_embedding_score": float(sum(m.embedding_score for m in matches) / len(matches)),
        **telemetry,
    }
