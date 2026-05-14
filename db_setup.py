#!/usr/bin/env python3
from __future__ import annotations

# ============================================================
# VERSION
# ============================================================
SCRIPT_NAME      = "db_setup.py"
SCRIPT_VERSION   = "1.3"
PIPELINE_VERSION = f"{SCRIPT_NAME[:-3]}_v{SCRIPT_VERSION}"

# ── Change log ──────────────────────────────────────────────
# v1.3 (2026-05-13)
#   • Added get_or_create_connection() — returns an existing open
#     connection unchanged, or opens a new one via
#     DatabaseConfig.default().  Eliminates the pattern of
#     unconditionally calling get_connection() when a caller may
#     already hold a live connection (avoids redundant TCP
#     handshakes on Neon serverless endpoints).
#   • Added __all__ — declares the public API surface of db_setup
#     so import * and tooling expose only the intended symbols.
# v1.2 (2026-05-13)
#   • Added DatabaseConfig dataclass with get_connection() and
#     get_connection_with_retry().
# v1.1 (2026-05-13)
#   • Added SCRIPT_NAME, SCRIPT_VERSION, PIPELINE_VERSION.
# ────────────────────────────────────────────────────────────

"""
db_setup.py

Single database setup script for the current pipeline:
- screenplay_parser.py
- parallel_importer.py (track_query-only architecture)
- scene_music_matcher.py (track_query_id-only architecture)

What this script creates:
1. pgvector extension
2. scene_query table + indexes + updated_at trigger
3. track_query table + indexes + updated_at trigger
4. scene_music_matches_v6 table + indexes + updated_at trigger
5. debug / QA views used by the latest matcher workflow

Important design note:
- There is NO tracks table in this schema.
- track_query is the authoritative track registry.
- scene_music_matches_v6 references track_query_id only.

The script is idempotent:
- CREATE TABLE IF NOT EXISTS
- CREATE INDEX IF NOT EXISTS
- CREATE OR REPLACE VIEW
- triggers are recreated safely
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional
import psycopg2
from dotenv import load_dotenv

load_dotenv()

VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))
MATCH_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.45"))

DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "sslmode": os.getenv("PGSSLMODE"),
    "channel_binding": os.getenv("PGCHANNELBINDING"),
}


# ============================================================
# DATABASE CONFIG
# ============================================================

@dataclass
class DatabaseConfig:
    """Single authoritative source for all Neon/PostgreSQL connection parameters.

    Replaces the four scattered ``get_connection()`` implementations that had
    incompatible signatures across db_setup, importer_pipeline,
    scene_music_matcher_engine, and screenplay_parser_engine.

    Usage
    -----
    Use the class-level factory for the standard env-var driven config::

        cfg  = DatabaseConfig.default()
        conn = cfg.get_connection()

    Or construct directly for tests / overrides::

        cfg = DatabaseConfig(host="localhost", dbname="test_db", ...)
    """
    host:            str         = field(default_factory=lambda: os.getenv("PGHOST", ""))
    dbname:          str         = field(default_factory=lambda: os.getenv("PGDATABASE", ""))
    user:            str         = field(default_factory=lambda: os.getenv("PGUSER", ""))
    password:        str         = field(default_factory=lambda: os.getenv("PGPASSWORD", ""))
    sslmode:         str         = field(default_factory=lambda: os.getenv("PGSSLMODE", "require"))
    channel_binding: str         = field(default_factory=lambda: os.getenv("PGCHANNELBINDING", ""))
    connect_timeout: int         = 10

    @classmethod
    def default(cls) -> "DatabaseConfig":
        """Return a DatabaseConfig populated entirely from environment variables."""
        return cls()

    def as_kwargs(self) -> dict[str, str | int]:
        """Return a kwargs dict suitable for ``psycopg2.connect(**kwargs)``.

        Keys with empty-string values are omitted so psycopg2 falls back to
        its own defaults (important for optional fields like channel_binding).
        """
        raw = {
            "host":            self.host,
            "dbname":          self.dbname,
            "user":            self.user,
            "password":        self.password,
            "sslmode":         self.sslmode,
            "channel_binding": self.channel_binding,
            "connect_timeout": self.connect_timeout,
        }
        return {k: v for k, v in raw.items() if v not in ("", None)}

    def get_connection(self) -> psycopg2.extensions.connection:
        """Open and return a new psycopg2 connection."""
        return psycopg2.connect(**self.as_kwargs())

    def get_connection_with_retry(
        self,
        max_attempts: int   = 3,
        base_delay:   float = 0.5,
    ) -> psycopg2.extensions.connection:
        """Open a connection with exponential backoff for transient Neon timeouts.

        Neon serverless Postgres can take up to ~1 s to wake a cold endpoint.
        This retries up to *max_attempts* times, sleeping
        ``base_delay * 2^(attempt-1)`` seconds between attempts.

        Parameters
        ----------
        max_attempts:
            Maximum total connection attempts (default 3).
        base_delay:
            Initial sleep in seconds before the second attempt (default 0.5).
            Subsequent sleeps: 1.0 s, 2.0 s, …

        Raises
        ------
        psycopg2.OperationalError
            Re-raised after all attempts are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return psycopg2.connect(**self.as_kwargs())
            except psycopg2.OperationalError as exc:
                last_exc = exc
                if attempt == max_attempts:
                    raise
                sleep_secs = base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_secs)
        raise last_exc  # type: ignore[misc]  — unreachable, satisfies type checker


def get_connection() -> psycopg2.extensions.connection:
    """Module-level convenience wrapper — delegates to ``DatabaseConfig.default()``.

    Backward-compatible with the pre-v1.2 signature used by setup_database().
    """
    return DatabaseConfig.default().get_connection()


def get_or_create_connection(
    existing: Optional[psycopg2.extensions.connection] = None,
) -> psycopg2.extensions.connection:
    """Return *existing* if it is still open, otherwise open a new connection.

    Avoids redundant TCP handshakes on Neon serverless endpoints when a caller
    may already hold a live connection.  A connection is considered open when
    ``psycopg2.extensions.connection.closed == 0``.

    Parameters
    ----------
    existing:
        A previously opened psycopg2 connection, or ``None``.

    Returns
    -------
    psycopg2.extensions.connection
        Either *existing* (unchanged) or a freshly opened connection via
        ``DatabaseConfig.default().get_connection()``.

    Examples
    --------
    >>> conn = get_or_create_connection()          # opens new
    >>> conn2 = get_or_create_connection(conn)     # reuses same
    >>> conn3 = get_or_create_connection(None)     # opens new
    """
    if existing is not None and existing.closed == 0:
        return existing
    return DatabaseConfig.default().get_connection()


# ============================================================
# PUBLIC API
# ============================================================
__all__ = [
    # Connection helpers
    "DatabaseConfig",
    "get_connection",
    "get_or_create_connection",
    # DDL — extensions
    "create_extensions",
    # DDL — trigger helpers
    "create_generic_updated_at_function",
    "recreate_updated_at_trigger",
    # DDL — scene_query
    "create_scene_query_table",
    "create_scene_query_indexes",
    # DDL — track_query
    "create_track_query_table",
    "create_track_query_indexes",
    "create_track_query_views",
    # DDL — scene_music_matches_v6
    "create_scene_music_matches_v6_table",
    "create_scene_music_matches_v6_indexes",
    "create_scene_music_matches_v6_views",
    # Orchestration
    "setup_database",
    # Identity
    "SCRIPT_NAME", "SCRIPT_VERSION", "PIPELINE_VERSION",
]


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

def create_extensions(cur) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


# ---------------------------------------------------------------------------
# Generic updated_at trigger helpers
# ---------------------------------------------------------------------------

def create_generic_updated_at_function(cur) -> None:
    cur.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def recreate_updated_at_trigger(cur, table_name: str, trigger_name: str) -> None:
    cur.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON {table_name};")
    cur.execute(
        f"""
        CREATE TRIGGER {trigger_name}
        BEFORE UPDATE ON {table_name}
        FOR EACH ROW
        EXECUTE FUNCTION set_updated_at();
        """
    )


# ---------------------------------------------------------------------------
# scene_query (screenplay_parser_final.py)
# ---------------------------------------------------------------------------

def create_scene_query_table(cur) -> None:
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scene_query (
            id BIGSERIAL PRIMARY KEY,

            episode_nr INT NOT NULL,
            scene_nr INT NOT NULL,
            theme_nr INT NOT NULL,
            segmentation_reason TEXT NOT NULL,

            theme_title_pl TEXT NOT NULL,
            theme_title_en TEXT NOT NULL,
            description_en TEXT NOT NULL,
            theme_txt TEXT NOT NULL,

            tags_summary_en TEXT NOT NULL,

            scene_music_semantics JSONB NOT NULL,
            scene_music_targets JSONB NOT NULL,
            scene_tag_selection JSONB NOT NULL,
            clap_prompt_ensemble JSONB NOT NULL,

            embedding_main VECTOR({VECTOR_DIM}),
            embedding_tags VECTOR({VECTOR_DIM}),
            embedding_clap_ensemble VECTOR({VECTOR_DIM}),
            embedding_hybrid VECTOR({VECTOR_DIM}),

            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),

            CONSTRAINT uq_scene_query_episode_scene_theme
                UNIQUE (episode_nr, scene_nr, theme_nr),

            CONSTRAINT chk_scene_music_semantics_object
                CHECK (jsonb_typeof(scene_music_semantics) = 'object'),
            CONSTRAINT chk_scene_music_targets_object
                CHECK (jsonb_typeof(scene_music_targets) = 'object'),
            CONSTRAINT chk_scene_tag_selection_object
                CHECK (jsonb_typeof(scene_tag_selection) = 'object'),
            CONSTRAINT chk_clap_prompt_ensemble_object
                CHECK (jsonb_typeof(clap_prompt_ensemble) = 'object'),

            CONSTRAINT chk_scene_music_semantics_required_keys
                CHECK (
                    scene_music_semantics ? 'emotional_direction' AND
                    scene_music_semantics ? 'narrative_function' AND
                    scene_music_semantics ? 'weight_profile' AND
                    scene_music_semantics ? 'dialogue_safe_required'
                ),
            CONSTRAINT chk_scene_music_targets_required_keys
                CHECK (
                    scene_music_targets ? 'energy_target' AND
                    scene_music_targets ? 'tempo_target' AND
                    scene_music_targets ? 'rhythm_target' AND
                    scene_music_targets ? 'intensity_shape_target' AND
                    scene_music_targets ? 'sound_character_target'
                ),
            CONSTRAINT chk_scene_tag_selection_required_keys
                CHECK (
                    scene_tag_selection ? 'should_have_tags' AND
                    scene_tag_selection ? 'must_not_tags'
                ),
            CONSTRAINT chk_clap_prompt_ensemble_required_keys
                CHECK (
                    clap_prompt_ensemble ? 'semantic_scene_prompt' AND
                    clap_prompt_ensemble ? 'music_for_scene_prompt' AND
                    clap_prompt_ensemble ? 'emotion_prompt' AND
                    clap_prompt_ensemble ? 'narrative_prompt' AND
                    clap_prompt_ensemble ? 'sonic_prompt' AND
                    clap_prompt_ensemble ? 'tag_prompt' AND
                    clap_prompt_ensemble ? 'concise_core_prompt'
                ),

            CONSTRAINT chk_scene_music_semantics_array_types
                CHECK (
                    jsonb_typeof(scene_music_semantics->'emotional_direction') = 'array' AND
                    jsonb_typeof(scene_music_semantics->'narrative_function') = 'array'
                ),
            CONSTRAINT chk_scene_music_targets_array_types
                CHECK (
                    jsonb_typeof(scene_music_targets->'sound_character_target') = 'array'
                ),
            CONSTRAINT chk_scene_tag_selection_array_types
                CHECK (
                    jsonb_typeof(scene_tag_selection->'should_have_tags') = 'array' AND
                    jsonb_typeof(scene_tag_selection->'must_not_tags') = 'array'
                )
        );
        """
    )


def create_scene_query_indexes(cur) -> None:
    statements = [
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_episode_scene_theme
        ON scene_query (episode_nr, scene_nr, theme_nr);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_theme_title_en
        ON scene_query (theme_title_en);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_description_en
        ON scene_query (description_en);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_scene_music_semantics_gin
        ON scene_query USING GIN (scene_music_semantics);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_scene_music_targets_gin
        ON scene_query USING GIN (scene_music_targets);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_scene_tag_selection_gin
        ON scene_query USING GIN (scene_tag_selection);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_clap_prompt_ensemble_gin
        ON scene_query USING GIN (clap_prompt_ensemble);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_embedding_main_ivfflat
        ON scene_query USING ivfflat (embedding_main vector_cosine_ops)
        WITH (lists = 50);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_embedding_tags_ivfflat
        ON scene_query USING ivfflat (embedding_tags vector_cosine_ops)
        WITH (lists = 50);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_embedding_clap_ensemble_ivfflat
        ON scene_query USING ivfflat (embedding_clap_ensemble vector_cosine_ops)
        WITH (lists = 50);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_query_embedding_hybrid_ivfflat
        ON scene_query USING ivfflat (embedding_hybrid vector_cosine_ops)
        WITH (lists = 50);
        """,
    ]
    for sql in statements:
        cur.execute(sql)


# ---------------------------------------------------------------------------
# track_query (new_parallel_importer_v9_0.py, track_query-only)
# ---------------------------------------------------------------------------

def create_track_query_table(cur) -> None:
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS track_query (
            id BIGSERIAL PRIMARY KEY,

            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            duration_sec FLOAT,
            bpm INT,
            musical_key TEXT,

            semantic_title_en TEXT NOT NULL,
            description_en TEXT NOT NULL,
            tags_summary_en TEXT NOT NULL,

            track_music_semantics JSONB NOT NULL,
            track_music_targets JSONB NOT NULL,
            track_tag_selection JSONB NOT NULL,
            track_clap_prompt_ensemble JSONB NOT NULL,

            embedding_audio VECTOR({VECTOR_DIM}),
            embedding_main VECTOR({VECTOR_DIM}),
            embedding_tags VECTOR({VECTOR_DIM}),
            embedding_clap_ensemble VECTOR({VECTOR_DIM}),
            embedding_hybrid VECTOR({VECTOR_DIM}),

            audio_analysis JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            segmentation JSONB NOT NULL DEFAULT '{{}}'::jsonb,

            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),

            CONSTRAINT uq_track_query_filepath UNIQUE (filepath),

            CONSTRAINT chk_track_music_semantics_object
                CHECK (jsonb_typeof(track_music_semantics) = 'object'),
            CONSTRAINT chk_track_music_targets_object
                CHECK (jsonb_typeof(track_music_targets) = 'object'),
            CONSTRAINT chk_track_tag_selection_object
                CHECK (jsonb_typeof(track_tag_selection) = 'object'),
            CONSTRAINT chk_track_clap_prompt_ensemble_object
                CHECK (jsonb_typeof(track_clap_prompt_ensemble) = 'object'),
            CONSTRAINT chk_audio_analysis_object
                CHECK (jsonb_typeof(audio_analysis) = 'object'),
            CONSTRAINT chk_segmentation_object
                CHECK (jsonb_typeof(segmentation) = 'object'),

            CONSTRAINT chk_track_music_semantics_required_keys
                CHECK (
                    track_music_semantics ? 'emotional_direction' AND
                    track_music_semantics ? 'narrative_function' AND
                    track_music_semantics ? 'weight_profile_candidates' AND
                    track_music_semantics ? 'dialogue_safe_score' AND
                    track_music_semantics ? 'dialogue_safe'
                ),

            CONSTRAINT chk_track_music_targets_required_keys
                CHECK (
                    track_music_targets ? 'energy_target' AND
                    track_music_targets ? 'tempo_target' AND
                    track_music_targets ? 'rhythm_target' AND
                    track_music_targets ? 'intensity_shape_target' AND
                    track_music_targets ? 'sound_character_target'
                ),

            CONSTRAINT chk_track_tag_selection_required_keys
                CHECK (
                    track_tag_selection ? 'should_have_tags' AND
                    track_tag_selection ? 'must_not_tags'
                ),

            CONSTRAINT chk_track_clap_prompt_ensemble_required_keys
                CHECK (
                    track_clap_prompt_ensemble ? 'semantic_scene_prompt' AND
                    track_clap_prompt_ensemble ? 'music_for_scene_prompt' AND
                    track_clap_prompt_ensemble ? 'emotion_prompt' AND
                    track_clap_prompt_ensemble ? 'narrative_prompt' AND
                    track_clap_prompt_ensemble ? 'sonic_prompt' AND
                    track_clap_prompt_ensemble ? 'tag_prompt' AND
                    track_clap_prompt_ensemble ? 'concise_core_prompt'
                ),

            CONSTRAINT chk_track_music_semantics_array_types
                CHECK (
                    jsonb_typeof(track_music_semantics->'emotional_direction') = 'array' AND
                    jsonb_typeof(track_music_semantics->'narrative_function') = 'array' AND
                    jsonb_typeof(track_music_semantics->'weight_profile_candidates') = 'array'
                ),

            CONSTRAINT chk_track_music_targets_array_types
                CHECK (
                    jsonb_typeof(track_music_targets->'sound_character_target') = 'array'
                ),

            CONSTRAINT chk_track_tag_selection_array_types
                CHECK (
                    jsonb_typeof(track_tag_selection->'should_have_tags') = 'array' AND
                    jsonb_typeof(track_tag_selection->'must_not_tags') = 'array'
                )
        );
        """
    )


def create_track_query_indexes(cur) -> None:
    statements = [
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_filename
        ON track_query (filename);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_filepath
        ON track_query (filepath);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_track_music_semantics_gin
        ON track_query USING GIN (track_music_semantics);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_track_music_targets_gin
        ON track_query USING GIN (track_music_targets);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_track_tag_selection_gin
        ON track_query USING GIN (track_tag_selection);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_track_clap_prompt_ensemble_gin
        ON track_query USING GIN (track_clap_prompt_ensemble);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_audio_analysis_gin
        ON track_query USING GIN (audio_analysis);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_segmentation_gin
        ON track_query USING GIN (segmentation);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_embedding_audio_ivfflat
        ON track_query USING ivfflat (embedding_audio vector_cosine_ops)
        WITH (lists = 100);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_embedding_main_ivfflat
        ON track_query USING ivfflat (embedding_main vector_cosine_ops)
        WITH (lists = 100);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_embedding_tags_ivfflat
        ON track_query USING ivfflat (embedding_tags vector_cosine_ops)
        WITH (lists = 100);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_embedding_clap_ensemble_ivfflat
        ON track_query USING ivfflat (embedding_clap_ensemble vector_cosine_ops)
        WITH (lists = 100);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_track_query_embedding_hybrid_ivfflat
        ON track_query USING ivfflat (embedding_hybrid vector_cosine_ops)
        WITH (lists = 100);
        """,
    ]
    for sql in statements:
        cur.execute(sql)


def create_track_query_views(cur) -> None:
    cur.execute(
        """
        CREATE OR REPLACE VIEW track_query_debug AS
        SELECT
            tq.id,
            tq.filename,
            tq.filepath,
            tq.duration_sec,
            tq.bpm,
            tq.musical_key,
            tq.semantic_title_en,
            tq.description_en,
            tq.tags_summary_en,
            tq.track_music_semantics,
            tq.track_music_targets,
            tq.track_tag_selection,
            tq.track_clap_prompt_ensemble,
            tq.audio_analysis,
            tq.segmentation,
            tq.created_at,
            tq.updated_at
        FROM track_query tq;
        """
    )

    cur.execute(
        """
        CREATE OR REPLACE VIEW track_scene_contract_debug AS
        SELECT
            tq.id AS track_query_id,
            tq.filename,
            tq.filepath,
            tq.semantic_title_en,
            tq.description_en,
            tq.tags_summary_en,
            tq.track_music_semantics,
            tq.track_music_targets,
            tq.track_tag_selection,
            tq.track_clap_prompt_ensemble
        FROM track_query tq;
        """
    )


# ---------------------------------------------------------------------------
# scene_music_matches_v6 (scene_music_matcher_v6_1.py, track_query_id-only)
# ---------------------------------------------------------------------------

def create_scene_music_matches_v6_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_music_matches_v6 (
            id BIGSERIAL PRIMARY KEY,

            episode_nr INT NOT NULL,

            scene_query_id BIGINT NOT NULL
                REFERENCES scene_query(id) ON DELETE CASCADE,

            scene_nr INT NOT NULL,
            theme_nr INT NOT NULL,

            track_query_id BIGINT NOT NULL
                REFERENCES track_query(id) ON DELETE CASCADE,

            rank_position INT NOT NULL,

            scene_theme_title_en TEXT,
            track_semantic_title_en TEXT,
            track_filename TEXT,
            track_filepath TEXT,

            duration_sec FLOAT,
            bpm INT,
            musical_key TEXT,

            final_score FLOAT NOT NULL,

            main_similarity FLOAT,
            tags_similarity FLOAT,
            ensemble_similarity FLOAT,
            hybrid_similarity FLOAT,
            audio_similarity_aux FLOAT,

            embedding_score FLOAT,
            semantic_score FLOAT,
            targets_score FLOAT,
            tag_selection_score FLOAT,
            dialogue_score FLOAT,

            dialogue_conflict FLOAT,
            duration_conflict FLOAT,
            forbidden_tag_conflict FLOAT,
            style_redundancy_penalty FLOAT,
            same_track_consecutive_penalty FLOAT,
            missing_data_penalty FLOAT,
            penalty_total FLOAT,

            style_signature TEXT,
            dialogue_safe_applied BOOLEAN DEFAULT FALSE,

            match_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            match_explanation JSONB NOT NULL DEFAULT '{}'::jsonb,

            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),

            CONSTRAINT uq_scene_music_matches_v6_scene_rank
                UNIQUE (scene_query_id, rank_position),

            CONSTRAINT uq_scene_music_matches_v6_scene_track
                UNIQUE (scene_query_id, track_query_id),

            CONSTRAINT chk_match_metadata_object
                CHECK (jsonb_typeof(match_metadata) = 'object'),

            CONSTRAINT chk_match_explanation_object
                CHECK (jsonb_typeof(match_explanation) = 'object')
        );
        """
    )


def create_scene_music_matches_v6_indexes(cur) -> None:
    statements = [
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_episode
        ON scene_music_matches_v6 (episode_nr, scene_nr, theme_nr, rank_position);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_scene
        ON scene_music_matches_v6 (scene_query_id, rank_position);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_track
        ON scene_music_matches_v6 (track_query_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_final_score
        ON scene_music_matches_v6 (final_score DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_style_signature
        ON scene_music_matches_v6 (style_signature);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_dialogue_safe
        ON scene_music_matches_v6 (dialogue_safe_applied);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_match_metadata_gin
        ON scene_music_matches_v6 USING GIN (match_metadata);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scene_music_matches_v6_match_explanation_gin
        ON scene_music_matches_v6 USING GIN (match_explanation);
        """,
    ]
    for sql in statements:
        cur.execute(sql)


def create_scene_music_matches_v6_views(cur) -> None:
    threshold_sql = f"{MATCH_LOW_CONFIDENCE_THRESHOLD:.6f}"

    cur.execute(
        """
        CREATE OR REPLACE VIEW scene_music_matches_v6_debug AS
        SELECT
            m.id,
            m.episode_nr,
            m.scene_query_id,
            m.scene_nr,
            m.theme_nr,
            sq.theme_title_en AS scene_theme_title_en,
            sq.description_en AS scene_description_en,
            m.track_query_id,
            tq.semantic_title_en AS track_semantic_title_en,
            tq.description_en AS track_description_en,
            tq.tags_summary_en AS track_tags_summary_en,
            m.track_filename,
            m.track_filepath,
            m.rank_position,
            m.final_score,
            m.main_similarity,
            m.tags_similarity,
            m.ensemble_similarity,
            m.hybrid_similarity,
            m.audio_similarity_aux,
            m.embedding_score,
            m.semantic_score,
            m.targets_score,
            m.tag_selection_score,
            m.dialogue_score,
            m.dialogue_conflict,
            m.duration_conflict,
            m.forbidden_tag_conflict,
            m.style_redundancy_penalty,
            m.same_track_consecutive_penalty,
            m.missing_data_penalty,
            m.penalty_total,
            m.style_signature,
            m.dialogue_safe_applied,
            m.duration_sec,
            m.bpm,
            m.musical_key,
            m.match_metadata,
            m.match_explanation,
            m.created_at,
            m.updated_at
        FROM scene_music_matches_v6 m
        JOIN scene_query sq ON sq.id = m.scene_query_id
        JOIN track_query tq ON tq.id = m.track_query_id
        ORDER BY m.episode_nr, m.scene_nr, m.theme_nr, m.rank_position;
        """
    )

    cur.execute(
        """
        CREATE OR REPLACE VIEW scene_music_matches_v6_top1 AS
        SELECT DISTINCT ON (m.scene_query_id)
            m.id,
            m.episode_nr,
            m.scene_query_id,
            m.scene_nr,
            m.theme_nr,
            sq.theme_title_en AS scene_theme_title_en,
            sq.description_en AS scene_description_en,
            m.track_query_id,
            tq.semantic_title_en AS track_semantic_title_en,
            m.track_filename,
            m.track_filepath,
            m.rank_position,
            m.final_score,
            m.embedding_score,
            m.semantic_score,
            m.targets_score,
            m.tag_selection_score,
            m.dialogue_score,
            m.penalty_total,
            m.style_signature,
            m.dialogue_safe_applied,
            m.match_explanation
        FROM scene_music_matches_v6 m
        JOIN scene_query sq ON sq.id = m.scene_query_id
        JOIN track_query tq ON tq.id = m.track_query_id
        ORDER BY m.scene_query_id, m.rank_position ASC, m.final_score DESC;
        """
    )

    cur.execute(
        """
        CREATE OR REPLACE VIEW scene_track_compare_debug AS
        SELECT
            m.id AS match_id,
            m.episode_nr,
            m.scene_query_id,
            m.scene_nr,
            m.theme_nr,
            sq.theme_title_en,
            sq.description_en AS scene_description_en,
            sq.tags_summary_en AS scene_tags_summary_en,
            sq.scene_music_semantics,
            sq.scene_music_targets,
            sq.scene_tag_selection,
            sq.clap_prompt_ensemble AS scene_clap_prompt_ensemble,
            m.track_query_id,
            tq.semantic_title_en,
            tq.description_en AS track_description_en,
            tq.tags_summary_en AS track_tags_summary_en,
            tq.track_music_semantics,
            tq.track_music_targets,
            tq.track_tag_selection,
            tq.track_clap_prompt_ensemble,
            tq.audio_analysis,
            tq.segmentation,
            m.final_score,
            m.embedding_score,
            m.semantic_score,
            m.targets_score,
            m.tag_selection_score,
            m.dialogue_score,
            m.penalty_total,
            m.match_explanation
        FROM scene_music_matches_v6 m
        JOIN scene_query sq ON sq.id = m.scene_query_id
        JOIN track_query tq ON tq.id = m.track_query_id;
        """
    )

    cur.execute(
        """
        CREATE OR REPLACE VIEW scene_music_matches_v6_episode_summary AS
        SELECT
            episode_nr,
            COUNT(*) AS total_match_rows,
            COUNT(DISTINCT scene_query_id) AS total_scene_themes,
            COUNT(DISTINCT track_query_id) AS distinct_tracks_used,
            AVG(final_score) AS avg_final_score,
            MAX(final_score) AS max_final_score,
            MIN(final_score) AS min_final_score,
            AVG(embedding_score) AS avg_embedding_score,
            AVG(semantic_score) AS avg_semantic_score,
            AVG(targets_score) AS avg_targets_score,
            AVG(tag_selection_score) AS avg_tag_selection_score,
            AVG(dialogue_score) AS avg_dialogue_score,
            AVG(penalty_total) AS avg_penalty_total
        FROM scene_music_matches_v6
        GROUP BY episode_nr
        ORDER BY episode_nr;
        """
    )

    cur.execute(
        """
        CREATE OR REPLACE VIEW scene_music_matches_v6_style_redundancy AS
        SELECT
            episode_nr,
            scene_nr,
            theme_nr,
            style_signature,
            COUNT(*) AS style_count,
            ARRAY_AGG(track_query_id ORDER BY rank_position) AS track_query_ids,
            ARRAY_AGG(track_filename ORDER BY rank_position) AS track_filenames
        FROM scene_music_matches_v6
        GROUP BY episode_nr, scene_nr, theme_nr, style_signature
        HAVING COUNT(*) > 1
        ORDER BY episode_nr, scene_nr, theme_nr, style_count DESC;
        """
    )

    cur.execute(
        f"""
        CREATE OR REPLACE VIEW scene_music_matches_v6_low_confidence AS
        SELECT
            m.id,
            m.episode_nr,
            m.scene_nr,
            m.theme_nr,
            m.scene_query_id,
            m.track_query_id,
            m.track_filename,
            m.rank_position,
            m.final_score,
            m.embedding_score,
            m.semantic_score,
            m.targets_score,
            m.tag_selection_score,
            m.dialogue_score,
            m.penalty_total,
            m.match_explanation
        FROM scene_music_matches_v6 m
        WHERE
            m.final_score < {threshold_sql}
            OR m.semantic_score < 0.40
            OR m.targets_score < 0.40
            OR m.penalty_total > 0.25
        ORDER BY m.episode_nr, m.scene_nr, m.theme_nr, m.rank_position;
        """
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def setup_database() -> None:
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        print(f"🔧 Ensuring pgvector extension (VECTOR_DIM={VECTOR_DIM})...")
        create_extensions(cur)

        print("📦 Creating scene_query table...")
        create_scene_query_table(cur)
        print("⚡ Creating scene_query indexes...")
        create_scene_query_indexes(cur)

        print("📦 Creating track_query table...")
        create_track_query_table(cur)
        print("⚡ Creating track_query indexes...")
        create_track_query_indexes(cur)
        print("👁 Creating track_query views...")
        create_track_query_views(cur)

        print("📦 Creating scene_music_matches_v6 table...")
        create_scene_music_matches_v6_table(cur)
        print("⚡ Creating scene_music_matches_v6 indexes...")
        create_scene_music_matches_v6_indexes(cur)
        print("👁 Creating scene_music_matches_v6 views...")
        create_scene_music_matches_v6_views(cur)

        print("🛠 Creating updated_at function and triggers...")
        create_generic_updated_at_function(cur)
        recreate_updated_at_trigger(cur, "scene_query", "trg_scene_query_updated_at")
        recreate_updated_at_trigger(cur, "track_query", "trg_track_query_updated_at")
        recreate_updated_at_trigger(cur, "scene_music_matches_v6", "trg_scene_music_matches_v6_updated_at")

        print("✅ Database setup completed successfully")
        print("   - scene_query ready")
        print("   - track_query ready (track_query-only architecture)")
        print("   - scene_music_matches_v6 ready (track_query_id-only architecture)")

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    setup_database()
