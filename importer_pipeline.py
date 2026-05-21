#!/usr/bin/env python3
"""
importer_pipeline.py  —  AI Music Supervisor · Track Import Pipeline  (v1.01)
==============================================================================

Purpose
-------
Pure business-logic layer for the Music Library Importer.  Contains no
Streamlit code.  Can be called directly from the command line, from a
batch job, or imported by music_library_importer.py (the Streamlit GUI).

Extracted from music_library_importer.py v3.35 as part of the Option B
3-layer split.  All shared embedding weights and tag rules are now imported
from pipeline_config.py instead of being defined here.

Public API
----------
  run_import(...)          — full ETL orchestrator; accepts a log_fn callback
  preprocess_file(path)    — librosa audio analysis + segment extraction
  scan_music_files(dir)    — walk local filesystem for audio files
  scan_cloud_files()       — list audio objects in Cloudflare R2
  build_track_contract(...)— semantic contract builder
  create_track_embeddings(...)  — 5-vector embedding package

Changes vs v1.1
-----------------
v1.11 (2026-05-13)
  • Added return type annotation ``Any`` to _get_s3_client() — the
    boto3 S3 client has no public type stub so Any is the correct
    annotation (avoids a mypy error while remaining truthful).
  • Added get_or_create_connection() to __all__ export list.
v1.1 (2026-05-13)
  • Added EmbeddingPackage dataclass — typed wrapper for the five
    embedding vectors produced per track.  create_track_embeddings()
    now returns EmbeddingPackage; upsert_track_query() calls
    .to_db_dict() to obtain the plain lists it passes to psycopg2.
  • Added TagIndex dataclass — typed wrapper for the CLAP tag
    embedding index.  build_tag_embedding_index() now returns a
    TagIndex; select_best_tags() accepts TagIndex directly.
  • Added ImportResult dataclass — typed return value for
    run_import().  GUI (_start_import) reads structured fields
    instead of parsing "Final |" log strings.
  • __all__ updated to include the three new classes.
v1.02 (2026-05-13)
  • ISSUE-02: Removed old-style typing generics (List[…], Dict[…],
    Tuple[…]); replaced with built-in lowercase equivalents throughout.
    Mapping and Sequence kept from typing (no built-in alias before 3.12).
  • ISSUE-03: Added __all__ to declare the public API surface.
v1.01 (2026-05-12)
  • BUG FIX (P0): cosine() replaced with cosine_similarity_safe().
    The old implementation lacked a zero-vector guard on the size/shape
    checks before normalisation; a silent all-zero audio embedding
    (produced when librosa returns no signal) would cause a
    ZeroDivisionError in select_best_tags().  The new implementation
    matches the safe guard logic used in scene_music_matcher_engine.py
    and additionally uses numpy for performance.  The single call site
    in select_best_tags() is updated to cosine_similarity_safe().
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import librosa
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from psycopg2.extras import Json
from transformers import ClapModel, ClapProcessor

# ── Shared pipeline constants (single source of truth) ────────────────────────
from pipeline_config import (
    CATEGORY_RULES,
    TAG_PROMPT_WEIGHTS,
    ENSEMBLE_PROMPT_WEIGHTS,
    TEXT_HYBRID_WEIGHTS,
    PROFILE_HINTS,
    CONTRARY_TAGS,
)

load_dotenv()

# ── Script identity ────────────────────────────────────────────────────────────
SCRIPT_NAME      = "importer_pipeline.py"
SCRIPT_VERSION   = "1.11"
PIPELINE_VERSION = f"{SCRIPT_NAME[:-3]}_v{SCRIPT_VERSION}"

# ── Runtime constants (all env-var defaults; overridable at run time) ──────────
MUSIC_DIR        = Path(os.getenv("MUSIC_DIR", "clanMusic"))
TAGS_FILE        = Path(os.getenv("TAGS_FILE", "tags_v2.json"))
TARGET_SR        = int(os.getenv("TARGET_SR", "48000"))
SEGMENT_SECONDS  = int(os.getenv("SEGMENT_SECONDS", "20"))
NUM_SEGMENTS     = int(os.getenv("NUM_SEGMENTS", "5"))
WORKERS          = int(os.getenv("WORKERS", "6"))
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "64"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "8"))
TEXT_BATCH_SIZE  = int(os.getenv("TEXT_BATCH_SIZE", "32"))
AUDIO_BATCH_SIZE = int(os.getenv("AUDIO_BATCH_SIZE", "8"))
VECTOR_DIM       = int(os.getenv("VECTOR_DIM", "512"))
SKIP_EXISTING    = os.getenv("SKIP_EXISTING", "1") == "1"
CLAP_MODEL_NAME  = os.getenv("CLAP_MODEL_NAME", "laion/clap-htsat-unfused")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
HF_TOKEN         = os.getenv("HF_TOKEN")

CLOUD_MUSIC          = os.getenv("CLOUD_MUSIC", "false").lower() == "true"
CF_ACCOUNT_ID        = os.getenv("CF_ACCOUNT_ID", "")
CF_TOKEN_VALUE       = os.getenv("CF_TOKEN_VALUE", "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL", "")
BUCKET_NAME          = os.getenv("BUCKET_NAME", "")

DB_CONFIG: dict[str, Optional[str]] = {
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
    # Data classes
    "EmbeddingPackage", "TagIndex", "ImportResult",
    # Audio / file scanning
    "scan_music_files", "scan_cloud_files", "preprocess_file",
    # Tag helpers
    "load_tags", "flatten_tags", "build_tag_prompts",
    "select_best_tags", "tags_by_category",
    "build_tag_embedding_index",
    # Math helpers
    "cosine_similarity_safe", "l2_normalize", "weighted_average",
    # Contract / embedding builders
    "build_track_contract", "create_track_embeddings",
    "build_track_music_semantics", "build_track_music_targets",
    "build_track_tag_selection", "build_track_texts",
    "build_track_clap_prompt_ensemble",
    # DB helpers
    "get_connection", "get_or_create_connection",
    "safe_close_cursor", "safe_close_connection",
    "ensure_track_query_exists", "upsert_track_query",
    "get_existing_filepaths", "get_existing_track_stats",
    "get_track_query_vector_dim", "validate_vector_dim",
    # Orchestration
    "run_import",
    # Encoding
    "encode_texts", "encode_audio_segments",
    # Utility
    "ensure_serializable",
    # Identity
    "SCRIPT_NAME", "SCRIPT_VERSION", "PIPELINE_VERSION",
    # Runtime constants (needed by the GUI)
    "MUSIC_DIR", "CLAP_MODEL_NAME", "DEVICE",
    "TARGET_SR", "SEGMENT_SECONDS", "NUM_SEGMENTS",
    "WORKERS", "CHUNK_SIZE", "BATCH_SIZE",
    "AUDIO_BATCH_SIZE", "TEXT_BATCH_SIZE",
    "VECTOR_DIM", "SKIP_EXISTING",
]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class EmbeddingPackage:
    """Typed container for the five embedding vectors produced per track.

    Replaces the raw ``dict[str, Any]`` previously returned by
    ``create_track_embeddings()``.  The ``to_db_dict()`` method
    serialises all arrays to Python lists for psycopg2/pgvector.

    Attributes
    ----------
    embedding_audio:
        Mean-pooled CLAP audio embedding from raw waveform segments.
    embedding_main:
        CLAP text embedding of ``"<title>. <description>"``.
    embedding_tags:
        CLAP text embedding of the tags summary string.
    embedding_clap_ensemble:
        Weighted average of the seven CLAP prompt embeddings.
    embedding_hybrid:
        Weighted combination of main + tags + ensemble (TEXT_HYBRID_WEIGHTS).
    hybrid_weights:
        Dict recording the weights used so the DB row is self-documenting.
    """
    embedding_audio:         np.ndarray
    embedding_main:          np.ndarray
    embedding_tags:          np.ndarray
    embedding_clap_ensemble: np.ndarray
    embedding_hybrid:        np.ndarray
    hybrid_weights:          dict[str, Any]

    @property
    def dim(self) -> int:
        """Dimensionality of the hybrid vector (representative of all vectors)."""
        return int(self.embedding_hybrid.shape[0])

    def to_db_dict(self) -> dict[str, Any]:
        """Serialise all arrays to Python lists for psycopg2 / pgvector.

        Returns the same key set that ``upsert_track_query`` expects so
        it can be passed directly as the ``embeddings`` argument.
        """
        return {
            "embedding_audio":         self.embedding_audio.tolist(),
            "embedding_main":          self.embedding_main.tolist(),
            "embedding_tags":          self.embedding_tags.tolist(),
            "embedding_clap_ensemble": self.embedding_clap_ensemble.tolist(),
            "embedding_hybrid":        self.embedding_hybrid.tolist(),
            "hybrid_weights":          self.hybrid_weights,
        }


@dataclass
class TagIndex:
    """Typed container for the CLAP tag embedding index.

    Replaces the opaque ``dict[tuple[str, str], dict[str, Any]]``
    previously returned by ``build_tag_embedding_index()``.
    Provides a ``nearest()`` method for top-k similarity lookup.
    """
    _index: dict[tuple[str, str], dict[str, Any]]

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def values(self):
        """Iterate over tag payload dicts — mirrors ``dict.values()``."""
        return self._index.values()

    def items(self):
        """Iterate over ``((category, name), payload)`` pairs."""
        return self._index.items()

    def nearest(self, query_vec: np.ndarray, top_k: int = 20) -> list[dict[str, Any]]:
        """Return the *top_k* most similar tags by cosine similarity.

        Parameters
        ----------
        query_vec:
            Audio or text embedding to compare against all tag vectors.
        top_k:
            Number of nearest tags to return.

        Returns
        -------
        list[dict]
            Tag payload dicts sorted by descending similarity, each augmented
            with a ``"nearest_score"`` key holding the cosine similarity.
        """
        payloads   = list(self._index.values())
        embeddings = np.stack([p["embedding"] for p in payloads], axis=0).astype(np.float32)
        norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed     = embeddings / np.maximum(norms, 1e-9)
        q          = np.asarray(query_vec, dtype=np.float32)
        q_norm     = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        sims = normed @ q
        idxs = np.argsort(sims)[::-1][:top_k]
        return [{**payloads[int(i)], "nearest_score": float(sims[i])} for i in idxs]


@dataclass
class ImportResult:
    """Structured return value for ``run_import()``.

    Replaces the ``None`` return and the "Final | inserted: N | …" log
    string that ``_start_import()`` previously parsed with string splitting.

    Attributes
    ----------
    inserted:
        Number of new rows inserted into track_query.
    updated:
        Number of existing rows updated (ON CONFLICT DO UPDATE).
    skipped:
        Number of files skipped because they were already indexed
        and ``skip_existing=True``.
    failed:
        Number of files that raised an exception during preprocessing
        or CLAP encoding.
    errors:
        Short error messages for each failed file (capped at 50 entries).
    duration_s:
        Wall-clock time of the full import in seconds.
    """
    inserted:   int        = 0
    updated:    int        = 0
    skipped:    int        = 0
    failed:     int        = 0
    errors:     list[str]  = field(default_factory=list)
    duration_s: float      = 0.0

    @property
    def total_attempted(self) -> int:
        """Files that entered the pipeline (excluding pre-scan skips)."""
        return self.inserted + self.updated + self.failed

    @property
    def success_rate(self) -> float:
        """Fraction of attempted files that were successfully indexed."""
        return (self.inserted + self.updated) / max(1, self.total_attempted)

    def to_stats_dict(self) -> dict[str, Any]:
        """Return the flat stats dict that ``_start_import`` stores in session state."""
        return {
            "inserted":   self.inserted,
            "updated":    self.updated,
            "skipped":    self.skipped,
            "failed":     self.failed,
            "duration_s": round(self.duration_s, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG)


def safe_close_cursor(cursor) -> None:
    try:
        if cursor is not None and not getattr(cursor, "closed", True):
            cursor.close()
    except Exception:
        pass


def safe_close_connection(conn) -> None:
    try:
        if conn is not None and not getattr(conn, "closed", True):
            conn.close()
    except Exception:
        pass


def ensure_track_query_exists(cursor) -> None:
    cursor.execute("SELECT to_regclass('public.track_query')")
    row = cursor.fetchone()
    if not row or row[0] is None:
        raise RuntimeError("track_query table is missing. Run db_setup.py first.")


def get_existing_filepaths(cursor, scanned_paths: Sequence[str]) -> set:
    if not scanned_paths:
        return set()
    cursor.execute(
        "SELECT filepath FROM track_query WHERE filepath = ANY(%s)",
        (list(scanned_paths),),
    )
    return {str(row[0]) for row in cursor.fetchall()}


def get_existing_track_stats(cursor, scanned_paths: Sequence[str]) -> dict[str, int]:
    cursor.execute("SELECT COUNT(*) FROM track_query")
    total_saved = int(cursor.fetchone()[0])
    if not scanned_paths:
        return {
            "total_saved_in_track_query":    total_saved,
            "already_saved_in_current_scan": 0,
            "new_in_current_scan":           0,
        }
    cursor.execute(
        "SELECT COUNT(*) FROM track_query WHERE filepath = ANY(%s)",
        (list(scanned_paths),),
    )
    already_saved = int(cursor.fetchone()[0])
    return {
        "total_saved_in_track_query":    total_saved,
        "already_saved_in_current_scan": already_saved,
        "new_in_current_scan":           int(len(scanned_paths) - already_saved),
    }


def get_track_query_vector_dim(cursor, column_name: str) -> Optional[int]:
    cursor.execute(
        """
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'track_query'::regclass AND attname = %s
        """,
        (column_name,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    atttypmod = row[0]
    if atttypmod is None or atttypmod < 0:
        return None
    return int(atttypmod)


def validate_vector_dim(cursor, expected_dim: int) -> None:
    for col in [
        "embedding_audio", "embedding_main", "embedding_tags",
        "embedding_clap_ensemble", "embedding_hybrid",
    ]:
        db_dim = get_track_query_vector_dim(cursor, col)
        if db_dim is not None and db_dim != expected_dim:
            raise ValueError(
                f"Vector dim mismatch for track_query.{col}: "
                f"DB expects {db_dim}, importer produced {expected_dim}"
            )


def upsert_track_query(
    cursor,
    file_row: Mapping[str, Any],
    contract: Mapping[str, Any],
    embeddings: "EmbeddingPackage | Mapping[str, Any]",
    seg_sec: int,
    clap_model_name: str,
) -> str:
    # Accept both the new EmbeddingPackage and the legacy plain dict.
    emb: Mapping[str, Any] = (
        embeddings.to_db_dict()
        if isinstance(embeddings, EmbeddingPackage)
        else embeddings
    )
    cursor.execute(
        """
        INSERT INTO track_query (
            filename, filepath, duration_sec, bpm, musical_key,
            semantic_title_en, description_en, tags_summary_en,
            track_music_semantics, track_music_targets,
            track_tag_selection, track_clap_prompt_ensemble,
            embedding_audio, embedding_main, embedding_tags,
            embedding_clap_ensemble, embedding_hybrid,
            audio_analysis, segmentation
        ) VALUES (
            %s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s
        )
        ON CONFLICT (filepath) DO UPDATE SET
            filename=EXCLUDED.filename,
            filepath=EXCLUDED.filepath,
            duration_sec=EXCLUDED.duration_sec,
            bpm=EXCLUDED.bpm,
            musical_key=EXCLUDED.musical_key,
            semantic_title_en=EXCLUDED.semantic_title_en,
            description_en=EXCLUDED.description_en,
            tags_summary_en=EXCLUDED.tags_summary_en,
            track_music_semantics=EXCLUDED.track_music_semantics,
            track_music_targets=EXCLUDED.track_music_targets,
            track_tag_selection=EXCLUDED.track_tag_selection,
            track_clap_prompt_ensemble=EXCLUDED.track_clap_prompt_ensemble,
            embedding_audio=EXCLUDED.embedding_audio,
            embedding_main=EXCLUDED.embedding_main,
            embedding_tags=EXCLUDED.embedding_tags,
            embedding_clap_ensemble=EXCLUDED.embedding_clap_ensemble,
            embedding_hybrid=EXCLUDED.embedding_hybrid,
            audio_analysis=EXCLUDED.audio_analysis,
            segmentation=EXCLUDED.segmentation,
            updated_at=NOW()
        RETURNING (xmax = 0) AS inserted
        """,
        (
            file_row["filename"],
            file_row["path"],
            file_row["duration"],
            file_row["bpm"],
            file_row["key"],
            contract["semantic_title_en"],
            contract["description_en"],
            contract["tags_summary_en"],
            Json(ensure_serializable(contract["track_music_semantics"])),
            Json(ensure_serializable(contract["track_music_targets"])),
            Json(ensure_serializable(contract["track_tag_selection"])),
            Json(ensure_serializable(contract["track_clap_prompt_ensemble"])),
            emb["embedding_audio"] if isinstance(emb["embedding_audio"], list)
                else emb["embedding_audio"].tolist(),
            emb["embedding_main"] if isinstance(emb["embedding_main"], list)
                else emb["embedding_main"].tolist(),
            emb["embedding_tags"] if isinstance(emb["embedding_tags"], list)
                else emb["embedding_tags"].tolist(),
            emb["embedding_clap_ensemble"] if isinstance(emb["embedding_clap_ensemble"], list)
                else emb["embedding_clap_ensemble"].tolist(),
            emb["embedding_hybrid"] if isinstance(emb["embedding_hybrid"], list)
                else emb["embedding_hybrid"].tolist(),
            Json(ensure_serializable(file_row["analysis"])),
            Json({
                "segment_seconds":  seg_sec,
                "num_segments":     len(file_row["segments"]),
                "segment_ranges":   file_row["segment_ranges"],
                "pipeline_version": PIPELINE_VERSION,
                "clap_model_name":  clap_model_name,
                "hybrid_weights":   ensure_serializable(emb["hybrid_weights"]),
            }),
        ),
    )
    row = cursor.fetchone()
    return "inserted" if (row and row[0]) else "updated"


# ══════════════════════════════════════════════════════════════════════════════
# MATH & SERIALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity_safe(a: Sequence[float], b: Sequence[float]) -> float:
    """Safely compute cosine similarity between two vectors.

    Returns 0.0 for empty, mismatched, or zero-norm inputs rather than
    raising ZeroDivisionError.  Uses numpy for performance.
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.size == 0 or b_arr.size == 0 or a_arr.shape != b_arr.shape:
        return 0.0
    a_norm = float(np.linalg.norm(a_arr))
    b_norm = float(np.linalg.norm(b_arr))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))


def l2_normalize(vec: Sequence[float]) -> np.ndarray:
    arr  = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else arr / norm


def weighted_average(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> np.ndarray:
    vectors_arr = np.asarray(vectors, dtype=np.float32)
    weights_arr = np.asarray(weights, dtype=np.float32)
    if vectors_arr.size == 0:
        raise ValueError("No vectors provided")
    if vectors_arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={vectors_arr.shape}")
    if vectors_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError("vectors and weights length mismatch")
    total = float(weights_arr.sum())
    if total == 0.0:
        raise ValueError("sum(weights) must be > 0")
    out = (vectors_arr * weights_arr[:, None]).sum(axis=0) / total
    return l2_normalize(out)


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


# ══════════════════════════════════════════════════════════════════════════════
# TAG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_tags(tags_file: Path = TAGS_FILE) -> dict[str, dict[str, Any]]:
    with tags_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_tags(raw_tags: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for category, category_data in raw_tags.items():
        category_pl = category_data["category_pl"]
        for tag_en, tag_pl in category_data["tags"].items():
            rows.append({
                "category":    category,
                "category_pl": category_pl,
                "name":        tag_en,
                "name_pl":     tag_pl,
            })
    return rows


def build_tag_prompts(tag_row: Mapping[str, str]) -> list[str]:
    category = tag_row["category"].replace("_", " ")
    tag_en   = tag_row["name"].replace("_", " ")
    return [
        f"cinematic instrumental music with {tag_en}",
        f"music tagged as {category}: {tag_en}",
        f"film underscore with {tag_en}",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def is_probably_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def scan_music_files(music_dir: Path, log_fn=None) -> list[str]:
    files:          list[str] = []
    skipped_hidden: int       = 0
    for root, _dirs, names in os.walk(music_dir):
        for name in names:
            p = Path(root) / name
            if is_probably_appledouble(p):
                skipped_hidden += 1
                continue
            if name.lower().endswith(AUDIO_EXTENSIONS):
                files.append(str(p))
    files.sort()
    if skipped_hidden and log_fn is not None:
        log_fn(f"⚠ Skipped AppleDouble sidecar files: {skipped_hidden}", -1)
    return files


def scan_cloud_files(log_fn=None) -> list[str]:
    """List all audio objects in the R2 bucket (S3-compatible API).

    Returns a list of object keys (strings) that look like audio files.
    These keys are used as the canonical "path" throughout the pipeline
    when CLOUD_MUSIC=true; they are stored in the filepath column of
    track_query so re-import skips already-indexed keys.
    """
    try:
        import boto3
    except ImportError:
        if log_fn:
            log_fn("⚠ boto3 not installed — run: pip install boto3", -1)
        return []

    try:
        session = boto3.session.Session()
        s3 = session.client(
            service_name          = "s3",
            endpoint_url          = R2_ENDPOINT_URL,
            aws_access_key_id     = R2_ACCESS_KEY_ID,
            aws_secret_access_key = R2_SECRET_ACCESS_KEY,
        )
        keys: list[str] = []
        skipped_hidden = 0
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME):
            for obj in page.get("Contents", []):
                key  = obj["Key"]
                name = key.split("/")[-1]
                if is_probably_appledouble(Path(name)):
                    skipped_hidden += 1
                    continue
                if name.lower().endswith(AUDIO_EXTENSIONS):
                    keys.append(key)
        keys.sort()
        if skipped_hidden and log_fn:
            log_fn(f"⚠ Skipped AppleDouble sidecar files: {skipped_hidden}", -1)
        return keys
    except Exception as exc:
        if log_fn:
            log_fn(f"⚠ R2 scan error: {exc}", -1)
        return []


# Module-level S3 client — created once, reused for every download.
_s3_client = None


def _get_s3_client() -> Any:
    """Return the shared boto3 S3 client, creating it on first call.

    Returns ``Any`` because boto3 does not ship public type stubs;
    the underlying type is ``botocore.client.S3``.
    """
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    import boto3
    from botocore.config import Config
    _s3_client = boto3.session.Session().client(
        service_name          = "s3",
        endpoint_url          = R2_ENDPOINT_URL,
        aws_access_key_id     = R2_ACCESS_KEY_ID,
        aws_secret_access_key = R2_SECRET_ACCESS_KEY,
        config                = Config(
            connect_timeout = 10,
            read_timeout    = 60,
            retries         = {"max_attempts": 3, "mode": "standard"},
        ),
    )
    return _s3_client


def _r2_download(key: str) -> bytes:
    """Download a single object from R2 using the shared S3 client."""
    return _get_s3_client().get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()


def sample_segments(
    audio: np.ndarray,
    sr: int,
    segment_seconds: int = SEGMENT_SECONDS,
    num_segments: int    = NUM_SEGMENTS,
) -> list[tuple[np.ndarray, tuple[float, float]]]:
    segment_len = segment_seconds * sr
    if len(audio) < segment_len:
        padded = np.pad(audio, (0, segment_len - len(audio)))
        return [(padded, (0.0, float(len(audio) / sr)))]
    if num_segments <= 1:
        return [(audio[:segment_len], (0.0, float(segment_seconds)))]
    max_start = len(audio) - segment_len
    starts    = np.linspace(0, max_start, num_segments, dtype=int)
    return [
        (audio[s: s + segment_len], (s / sr, (s + segment_len) / sr))
        for s in starts.tolist()
    ]


def bpm_to_tempo_target(bpm: Optional[int]) -> str:
    if bpm is None: return "moderate"
    if bpm < 55:    return "very_slow"
    if bpm < 80:    return "slow"
    if bpm < 118:   return "moderate"
    if bpm < 145:   return "fast"
    return "very_fast"


def rms_to_energy_target(rms: float) -> str:
    if rms < 0.030: return "very_low"
    if rms < 0.060: return "low"
    if rms < 0.110: return "medium"
    if rms < 0.180: return "high"
    return "very_high"


def zcr_to_rhythm_target(zcr: float, bpm: Optional[int]) -> str:
    if bpm is not None and bpm >= 120: return "driving"
    if zcr < 0.035: return "floating"
    if zcr < 0.070: return "steady"
    if zcr < 0.110: return "syncopated"
    return "irregular"


def infer_intensity_shape(rms: float, zcr: float) -> str:
    if rms > 0.14: return "climax_peak"
    if zcr > 0.09: return "pulsating"
    if rms > 0.08: return "gradual_build"
    return "static"


def centroid_to_sound_hints(centroid: float) -> list[str]:
    if centroid < 1600: return ["intimate", "analog"]
    if centroid < 3200: return ["atmospheric", "textured"]
    return ["clean", "wide"]


def preprocess_file(
    path: str,
    segment_seconds: int = SEGMENT_SECONDS,
    num_segments:    int = NUM_SEGMENTS,
) -> dict[str, Any]:
    """BPM, key, RMS, ZCR, spectral features via librosa.

    path is a local filesystem path when CLOUD_MUSIC=false, or an R2
    object key when CLOUD_MUSIC=true.  In cloud mode the file is
    downloaded and written to a temp file — librosa.load requires a
    real seekable file path for some formats (BytesIO can freeze).
    """
    try:
        if CLOUD_MUSIC:
            import tempfile
            data   = _r2_download(path)
            suffix = ("." + path.rsplit(".", 1)[-1]) if "." in path else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                audio, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        else:
            audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception as exc:
        return {
            "path":       path,
            "filename":   path.split("/")[-1] if CLOUD_MUSIC else os.path.basename(path),
            "status":     "error",
            "error_type": type(exc).__name__,
            "error":      str(exc),
        }

    duration   = float(librosa.get_duration(y=audio, sr=sr))
    segments   = sample_segments(audio, sr,
                                 segment_seconds=segment_seconds,
                                 num_segments=num_segments)
    seg_audio  = [seg for seg, _ in segments]
    seg_ranges = [
        {"start_sec": round(s, 3), "end_sec": round(e, 3)}
        for _, (s, e) in segments
    ]

    bpm: Optional[int] = None
    key: Optional[str] = None
    try:
        bpm_val, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = int(round(float(np.asarray(bpm_val).item())))
    except Exception:
        pass

    if len(audio) >= 2048:
        try:
            chroma      = librosa.feature.chroma_cqt(y=audio, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            keys        = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            key         = keys[int(chroma_mean.argmax())]
        except Exception:
            pass

    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    zcr = float(librosa.feature.zero_crossing_rate(audio).mean()) if len(audio) else 0.0
    spectral_centroid = 0.0
    spectral_rolloff  = 0.0
    if len(audio) >= 1024:
        try:
            spectral_centroid = float(
                librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
            )
            spectral_rolloff = float(
                librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()
            )
        except Exception:
            pass

    analysis = {
        "rms":                   rms,
        "zero_crossing_rate":    zcr,
        "spectral_centroid":     spectral_centroid,
        "spectral_rolloff":      spectral_rolloff,
        "energy_target_hint":    rms_to_energy_target(rms),
        "tempo_target_hint":     bpm_to_tempo_target(bpm),
        "rhythm_target_hint":    zcr_to_rhythm_target(zcr, bpm),
        "intensity_shape_hint":  infer_intensity_shape(rms, zcr),
        "sound_character_hints": centroid_to_sound_hints(spectral_centroid),
    }

    return {
        "path":           path,
        "filename":       path.split("/")[-1] if CLOUD_MUSIC else os.path.basename(path),
        "status":         "ok",
        "duration":       duration,
        "segments":       seg_audio,
        "segment_ranges": seg_ranges,
        "bpm":            bpm,
        "key":            key,
        "analysis":       analysis,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLAP ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def _to_numpy_features(output: Any) -> np.ndarray:
    if torch.is_tensor(output):
        return output.detach().cpu().numpy()
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds.detach().cpu().numpy()
    if hasattr(output, "audio_embeds") and output.audio_embeds is not None:
        return output.audio_embeds.detach().cpu().numpy()
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output.detach().cpu().numpy()
    raise TypeError(f"Unsupported CLAP output type: {type(output)}")


def encode_texts(
    texts: Sequence[str],
    processor: ClapProcessor,
    model: ClapModel,
    batch_size: int = TEXT_BATCH_SIZE,
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch  = list(texts[i: i + batch_size])
        inputs = processor(text=batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            raw = model.get_text_features(**inputs)
            emb = _to_numpy_features(raw)
        vectors.extend([l2_normalize(row) for row in emb])
    return vectors


def encode_audio_segments(
    segments: Sequence[np.ndarray],
    processor: ClapProcessor,
    model: ClapModel,
    batch_size: int = AUDIO_BATCH_SIZE,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    for i in range(0, len(segments), batch_size):
        batch  = list(segments[i: i + batch_size])
        inputs = processor(
            audio=batch, sampling_rate=TARGET_SR,
            return_tensors="pt", padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            raw = model.get_audio_features(**inputs)
            emb = _to_numpy_features(raw)
        encoded.extend([l2_normalize(row) for row in emb])
    return np.asarray(encoded, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TAG EMBEDDING INDEX & SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def build_tag_embedding_index(
    tags: Sequence[Mapping[str, str]],
    processor: ClapProcessor,
    model: ClapModel,
) -> TagIndex:
    """Build a CLAP embedding index for all supplied tag rows.

    Returns a :class:`TagIndex` so callers get typed access and the
    ``nearest()`` method for quick similarity lookup.
    """
    all_prompts:   list[str]       = []
    prompt_groups: list[list[str]] = []
    for tag_row in tags:
        prompts = build_tag_prompts(tag_row)
        prompt_groups.append(prompts)
        all_prompts.extend(prompts)
    prompt_vectors = encode_texts(
        all_prompts, processor=processor, model=model, batch_size=TEXT_BATCH_SIZE
    )
    index: dict[tuple[str, str], dict[str, Any]] = {}
    ptr = 0
    for tag_row, prompts in zip(tags, prompt_groups):
        group_vecs = prompt_vectors[ptr: ptr + len(prompts)]
        ptr       += len(prompts)
        ensemble   = weighted_average(group_vecs, TAG_PROMPT_WEIGHTS)
        index[(tag_row["category"], tag_row["name"])] = {
            **dict(tag_row), "prompts": prompts, "embedding": ensemble,
        }
    return TagIndex(index)


def select_best_tags(
    audio_embedding: np.ndarray,
    tag_index: "TagIndex | Mapping[tuple[str, str], Mapping[str, Any]]",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for payload in tag_index.values():
        row = {
            "category":    payload["category"],
            "category_pl": payload["category_pl"],
            "name":        payload["name"],
            "name_pl":     payload["name_pl"],
            "score":       float(cosine_similarity_safe(audio_embedding, payload["embedding"])),
        }
        grouped.setdefault(str(payload["category"]), []).append(row)

    selected: list[dict[str, Any]] = []
    for category, rows in grouped.items():
        rows.sort(key=lambda x: x["score"], reverse=True)
        rule      = CATEGORY_RULES.get(
            category, {"top_k": 1, "threshold": 0.22, "margin": 0.015, "required": 0}
        )
        top_k     = int(rule["top_k"])
        threshold = float(rule["threshold"])
        margin    = float(rule["margin"])
        required  = int(rule["required"])
        kept = 0
        for idx, row in enumerate(rows[: top_k + 1]):
            if idx >= top_k:
                break
            next_score   = rows[idx + 1]["score"] if idx + 1 < len(rows) else -1.0
            local_margin = float(row["score"] - next_score)
            if row["score"] >= threshold and (
                local_margin >= margin
                or kept < required
                or row["score"] >= threshold + 0.035
            ):
                selected.append(row)
                kept += 1
        if kept == 0 and required > 0 and rows and rows[0]["score"] >= threshold - 0.020:
            selected.append(rows[0])

    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def tags_by_category(
    selected_tags: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for tag in selected_tags:
        grouped.setdefault(str(tag["category"]), []).append(dict(tag))
    for rows in grouped.values():
        rows.sort(key=lambda x: float(x["score"]), reverse=True)
    return grouped


def top_names(rows: Sequence[Mapping[str, Any]], limit: int) -> list[str]:
    return [str(row["name"]) for row in rows[:limit]]


def score_lookup(selected_tags: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {str(tag["name"]): float(tag["score"]) for tag in selected_tags}


def derive_dialogue_safe_score(scores: Mapping[str, float]) -> float:
    safe = (
        0.45 * scores.get("dialogue_safe", 0.0)
        + 0.28 * scores.get("restrained", 0.0)
        + 0.24 * scores.get("background_music", 0.0)
        + 0.22 * scores.get("underscore", 0.0)
        + 0.18 * scores.get("sparse", 0.0)
        + 0.16 * scores.get("intimate", 0.0)
        + 0.14 * scores.get("low", 0.0)
        + 0.10 * scores.get("very_low", 0.0)
    )
    risk = (
        0.34 * scores.get("hits", 0.0)
        + 0.34 * scores.get("impacts", 0.0)
        + 0.25 * scores.get("trailer_music", 0.0)
        + 0.21 * scores.get("climax_peak", 0.0)
        + 0.18 * scores.get("very_high", 0.0)
        + 0.14 * scores.get("peak", 0.0)
        + 0.10 * scores.get("dense", 0.0)
    )
    return max(0.0, min(1.0, 0.50 + safe - risk))


def derive_weight_profile_candidates(
    scores: Mapping[str, float],
    grouped: Mapping[str, list[dict[str, Any]]],
    dialogue_safe_score: float,
) -> list[str]:
    selected_names = set(scores.keys())
    candidates: list[tuple[str, float]] = []
    for profile, hints in PROFILE_HINTS.items():
        overlap = selected_names.intersection(hints)
        if not overlap:
            continue
        score = sum(scores.get(name, 0.0) for name in overlap)
        if profile == "dialogue":
            score += 0.45 * dialogue_safe_score
        if score >= 0.24:
            candidates.append((profile, score))
    if not candidates:
        if dialogue_safe_score >= 0.62:
            return ["dialogue", "transition"]
        if grouped.get("scene"):
            return top_names(grouped["scene"], 2)
        return ["dramatic_scene"]
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0][1]
    return [name for name, val in candidates if val >= max(0.24, best - 0.12)][:3]


def derive_must_not_tags(
    selected_tags: Sequence[Mapping[str, Any]],
    semantics: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> list[str]:
    selected_names = [str(t["name"]) for t in selected_tags]
    selected_set   = set(selected_names)
    scores         = score_lookup(selected_tags)
    ranked: dict[str, float] = {}
    for name in selected_names:
        for opposite in CONTRARY_TAGS.get(name, []):
            if opposite not in selected_set:
                ranked[opposite] = max(
                    ranked.get(opposite, 0.0), max(0.20, scores.get(name, 0.0))
                )
    dialogue_safe_score = float(semantics.get("dialogue_safe_score", 0.0))
    if dialogue_safe_score >= 0.62:
        for name in ["hits", "impacts", "trailer_music", "very_high", "climax_peak"]:
            if name not in selected_set:
                ranked[name] = max(
                    ranked.get(name, 0.0), 0.50 + 0.30 * dialogue_safe_score
                )
    for profile in semantics.get("weight_profile_candidates", [])[:2]:
        if profile == "dialogue":
            for name in ["hits", "impacts", "trailer_music", "very_high"]:
                if name not in selected_set:
                    ranked[name] = max(ranked.get(name, 0.0), 0.42)
        if profile == "action":
            for name in ["dialogue_safe", "restrained", "very_low"]:
                if name not in selected_set:
                    ranked[name] = max(ranked.get(name, 0.0), 0.42)
    for target_key in ["energy_target", "tempo_target"]:
        tag_name = str(targets.get(target_key, ""))
        for opposite in CONTRARY_TAGS.get(tag_name, []):
            if opposite not in selected_set:
                ranked[opposite] = max(ranked.get(opposite, 0.0), 0.30)
    return [
        name for name, _ in
        sorted(ranked.items(), key=lambda x: x[1], reverse=True)
        if name not in selected_set
    ][:6]


def build_track_music_semantics(
    selected_tags: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped             = tags_by_category(selected_tags)
    scores              = score_lookup(selected_tags)
    dialogue_safe_score = derive_dialogue_safe_score(scores)
    emotional_direction = top_names(grouped.get("emotion", []), 4)
    narrative_function  = top_names(grouped.get("narrative", []), 3) or ["underscore"]
    return {
        "emotional_direction":       emotional_direction,
        "narrative_function":        narrative_function,
        "weight_profile_candidates": derive_weight_profile_candidates(
            scores, grouped, dialogue_safe_score
        ),
        "dialogue_safe_score": round(dialogue_safe_score, 6),
        "dialogue_safe":       dialogue_safe_score >= 0.60,
    }


def build_track_music_targets(
    selected_tags: Sequence[Mapping[str, Any]],
    file_row: Mapping[str, Any],
) -> dict[str, Any]:
    grouped  = tags_by_category(selected_tags)
    analysis = file_row.get("analysis", {}) or {}

    def first_or(category: str, fallback: str) -> str:
        rows = grouped.get(category, [])
        return str(rows[0]["name"]) if rows else fallback

    sound_character = top_names(grouped.get("sound_character", []), 4)
    if not sound_character:
        sound_character = list(analysis.get("sound_character_hints", []))[:2] or ["atmospheric"]
    if "atmospheric" not in sound_character and len(sound_character) < 4:
        sound_character.append("atmospheric")

    return {
        "energy_target":          first_or("energy",          str(analysis.get("energy_target_hint",   "medium"))),
        "tempo_target":           first_or("tempo",           str(analysis.get("tempo_target_hint",    "moderate"))),
        "rhythm_target":          first_or("rhythm",          str(analysis.get("rhythm_target_hint",   "steady"))),
        "intensity_shape_target": first_or("intensity_shape", str(analysis.get("intensity_shape_hint", "static"))),
        "sound_character_target": list(dict.fromkeys(sound_character))[:4],
    }


def build_track_tag_selection(
    selected_tags: Sequence[Mapping[str, Any]],
    semantics: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    should_have = [str(t["name"]) for t in selected_tags[:8]]
    must_not    = derive_must_not_tags(selected_tags, semantics, targets)
    should_set  = set(should_have)
    must_not    = [x for x in must_not if x not in should_set][:6]
    scores_list = [float(t["score"]) for t in selected_tags] if selected_tags else [0.0]
    return {
        "should_have_tags": should_have,
        "must_not_tags":    must_not,
        "confidence_profile": {
            "mean_selected_tag_score": float(np.mean(scores_list)),
            "max_selected_tag_score":  float(np.max(scores_list)),
            "min_selected_tag_score":  float(np.min(scores_list)),
        },
    }


def build_track_texts(
    selected_tags: Sequence[Mapping[str, Any]],
    file_row: Mapping[str, Any],
    semantics: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, str]:
    emotions  = semantics.get("emotional_direction", []) or ["cinematic"]
    narrative = semantics.get("narrative_function",  []) or ["underscore"]
    intensity = str(targets.get("intensity_shape_target", "static")).replace("_", " ")
    energy    = str(targets.get("energy_target",          "medium")).replace("_", " ")
    sound     = ", ".join(
        str(x).replace("_", " ") for x in targets.get("sound_character_target", [])[:2]
    ) or "atmospheric"
    semantic_title_en = (
        f"{emotions[0].replace('_',' ').title()} "
        f"{narrative[0].replace('_',' ').title()} Cue"
    )
    description_en = (
        f"Instrumental cue with {', '.join(emotions[:2]).replace('_',' ')} and "
        f"{', '.join(narrative[:2]).replace('_',' ')}; {energy} energy, "
        f"{intensity} shape, {sound} character."
    )
    tag_summary_items = [str(t["name"]).replace("_", " ") for t in selected_tags[:6]]
    tags_summary_en   = (
        ", ".join(tag_summary_items) if tag_summary_items
        else "cinematic instrumental underscore"
    )
    return {
        "semantic_title_en": semantic_title_en[:120],
        "description_en":    description_en[:240],
        "tags_summary_en":   tags_summary_en[:240],
    }


def build_track_clap_prompt_ensemble(
    texts: Mapping[str, str],
    semantics: Mapping[str, Any],
    targets: Mapping[str, Any],
    tag_selection: Mapping[str, Any],
) -> dict[str, str]:
    emotions  = ", ".join(semantics.get("emotional_direction", [])[:3]).replace("_", " ") or "cinematic"
    narrative = ", ".join(semantics.get("narrative_function",  [])[:3]).replace("_", " ") or "underscore"
    sound     = ", ".join(targets.get("sound_character_target", [])[:3]).replace("_", " ") or "atmospheric"
    tags      = ", ".join(tag_selection.get("should_have_tags", [])[:6]).replace("_", " ") or "cinematic"
    return {
        "semantic_scene_prompt":  texts["description_en"][:120],
        "music_for_scene_prompt": f"instrumental music for {texts['semantic_title_en'].lower()}"[:120],
        "emotion_prompt":         emotions[:120],
        "narrative_prompt":       narrative[:120],
        "sonic_prompt":           sound[:120],
        "tag_prompt":             tags[:120],
        "concise_core_prompt":    texts["tags_summary_en"][:120],
    }


def build_track_contract(
    selected_tags: Sequence[Mapping[str, Any]],
    file_row: Mapping[str, Any],
) -> dict[str, Any]:
    semantics       = build_track_music_semantics(selected_tags)
    targets         = build_track_music_targets(selected_tags, file_row)
    tag_selection   = build_track_tag_selection(selected_tags, semantics, targets)
    texts           = build_track_texts(selected_tags, file_row, semantics, targets)
    prompt_ensemble = build_track_clap_prompt_ensemble(texts, semantics, targets, tag_selection)
    return {
        **texts,
        "track_music_semantics":      semantics,
        "track_music_targets":        targets,
        "track_tag_selection":        tag_selection,
        "track_clap_prompt_ensemble": prompt_ensemble,
    }


def create_track_embeddings(
    contract: Mapping[str, Any],
    audio_embedding: np.ndarray,
    processor: ClapProcessor,
    model: ClapModel,
    text_batch_size: int = TEXT_BATCH_SIZE,
) -> EmbeddingPackage:
    """Build all five embedding vectors for a track and return them as an
    :class:`EmbeddingPackage`.

    Pass the result directly to :func:`upsert_track_query`; it calls
    ``embeddings.to_db_dict()`` internally.
    """
    main_text      = f"{contract['semantic_title_en']}. {contract['description_en']}".strip()
    embedding_main = np.asarray(
        encode_texts([main_text], processor, model, batch_size=1)[0], dtype=np.float32
    )
    embedding_tags = np.asarray(
        encode_texts([contract["tags_summary_en"]], processor, model, batch_size=1)[0],
        dtype=np.float32,
    )
    ensemble_texts = [
        contract["track_clap_prompt_ensemble"]["semantic_scene_prompt"],
        contract["track_clap_prompt_ensemble"]["music_for_scene_prompt"],
        contract["track_clap_prompt_ensemble"]["emotion_prompt"],
        contract["track_clap_prompt_ensemble"]["narrative_prompt"],
        contract["track_clap_prompt_ensemble"]["sonic_prompt"],
        contract["track_clap_prompt_ensemble"]["tag_prompt"],
        contract["track_clap_prompt_ensemble"]["concise_core_prompt"],
    ]
    ensemble_vectors        = encode_texts(ensemble_texts, processor, model, batch_size=7)
    embedding_clap_ensemble = weighted_average(ensemble_vectors, ENSEMBLE_PROMPT_WEIGHTS)
    embedding_hybrid        = weighted_average(
        [embedding_main, embedding_tags, embedding_clap_ensemble],
        TEXT_HYBRID_WEIGHTS,
    )
    return EmbeddingPackage(
        embedding_audio         = audio_embedding,
        embedding_main          = embedding_main,
        embedding_tags          = embedding_tags,
        embedding_clap_ensemble = embedding_clap_ensemble,
        embedding_hybrid        = embedding_hybrid,
        hybrid_weights          = {
            "main":          TEXT_HYBRID_WEIGHTS[0],
            "tags":          TEXT_HYBRID_WEIGHTS[1],
            "clap_ensemble": TEXT_HYBRID_WEIGHTS[2],
            "audio":         0.0,
            "mode":          "scene_query_compatible_textual_hybrid",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN IMPORT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def iter_chunks(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[i: i + size]) for i in range(0, len(items), size)]


def run_import(
    music_dir:        str      = str(MUSIC_DIR),
    tags_file:        str      = str(TAGS_FILE),
    clap_model_name:  str      = CLAP_MODEL_NAME,
    workers:          int      = WORKERS,
    chunk_size:       int      = CHUNK_SIZE,
    batch_size:       int      = BATCH_SIZE,
    audio_batch_size: int      = AUDIO_BATCH_SIZE,
    text_batch_size:  int      = TEXT_BATCH_SIZE,
    num_segments:     int      = NUM_SEGMENTS,
    segment_seconds:  int      = SEGMENT_SECONDS,
    skip_existing:    bool     = SKIP_EXISTING,
    log_fn: Optional[Callable[[str, int], None]] = None,
) -> ImportResult:
    """Full import pipeline.

    All parameters default to the module-level constants but can be
    overridden by the GUI or a direct call.

    Parameters
    ----------
    log_fn:
        Optional callback ``log_fn(message, stage_index)`` called at
        each stage boundary.  stage_index values:
        0=scan, 1=preprocess, 2=CLAP, 3=contract, 4=upsert, -1=info/warn.

    Returns
    -------
    ImportResult
        Structured summary with inserted/updated/skipped/failed counts
        and total wall-clock duration.
    """
    _start_time = time.monotonic()
    _result     = ImportResult()
    def _log(msg: str, stage: int = -1) -> None:
        if log_fn is not None:
            log_fn(msg, stage)

    _log("[INFO] Loading CLAP model…", 0)
    from transformers import ClapModel as _ClapModel, ClapProcessor as _ClapProcessor
    processor = _ClapProcessor.from_pretrained(clap_model_name, token=HF_TOKEN)
    model     = _ClapModel.from_pretrained(clap_model_name, token=HF_TOKEN).to(DEVICE)
    model.eval()
    _log(f"[INFO] CLAP model loaded on {DEVICE}", 0)

    _log("[INFO] Loading tag vocabulary…", 0)
    raw_tags  = load_tags(Path(tags_file))
    flat_tags = flatten_tags(raw_tags)
    _log(f"[INFO] Building tag embedding index ({len(flat_tags)} tags)…", 0)
    tag_index = build_tag_embedding_index(flat_tags, processor=processor, model=model)
    _log(f"[INFO] Tag index ready — {len(tag_index)} entries", 0)

    _log("[INFO] Scanning files…", 0)
    if CLOUD_MUSIC:
        files = scan_cloud_files(log_fn=_log)
        _log(f"[INFO] R2 scan complete — {len(files)} audio objects", 0)
    else:
        files = scan_music_files(Path(music_dir), log_fn=_log)
        _log(f"[INFO] Local scan complete — {len(files)} audio files", 0)

    conn = get_connection()
    conn.autocommit = True
    cur  = conn.cursor()
    try:
        ensure_track_query_exists(cur)
        stats          = get_existing_track_stats(cur, files)
        _log(f"   Total already in DB:   {stats['total_saved_in_track_query']}")
        _log(f"   Already saved (scan):  {stats['already_saved_in_current_scan']} / {len(files)}")
        _log(f"   New files to process:  {stats['new_in_current_scan']}")
        existing_paths = get_existing_filepaths(cur, files) if skip_existing else set()
        _log(f"   Skipping existing:     {len(existing_paths)}")
        if existing_paths:
            _result.skipped = len(existing_paths)
            files = [p for p in files if p not in existing_paths]
        _log(f"   Files left to import:  {len(files)}")
    finally:
        safe_close_cursor(cur)
        safe_close_connection(conn)

    if not files:
        _log("✅ All files already indexed in track_query. Nothing to do.")
        _result.duration_s = time.monotonic() - _start_time
        return _result

    inserted_count    = 0
    updated_count     = 0
    failed_count      = 0
    first_dim_checked = False
    chunks            = iter_chunks(files, chunk_size)
    total_chunks      = len(chunks)

    _preprocess = partial(
        preprocess_file,
        segment_seconds=segment_seconds,
        num_segments=num_segments,
    )

    for chunk_idx, chunk_files in enumerate(chunks, start=1):
        lo = (chunk_idx - 1) * chunk_size + 1
        hi = lo + len(chunk_files) - 1
        _log(f"🧩 Chunk {chunk_idx}/{total_chunks} | files {lo}–{hi} / {len(files)}")
        _log(f"   Preprocessing {len(chunk_files)} files…", stage=1)

        processed_all = []
        for _fi, _f in enumerate(chunk_files, start=1):
            if CLOUD_MUSIC:
                _log(f"   ☁  [{_fi}/{len(chunk_files)}] {_f.split('/')[-1]}")
            processed_all.append(_preprocess(_f))

        processed       = [r for r in processed_all if r.get("status") == "ok"]
        failed_in_chunk = [r for r in processed_all if r.get("status") != "ok"]
        failed_count   += len(failed_in_chunk)
        if failed_in_chunk:
            _log(f"   ⚠  Failed in chunk: {len(failed_in_chunk)}")
            for r in failed_in_chunk[:5]:
                _log(f"      - {r['path']} :: {r.get('error_type')} :: {r.get('error')}")
                if len(_result.errors) < 50:
                    _result.errors.append(
                        f"{r.get('path','')} :: {r.get('error_type','')} :: {r.get('error','')}"
                    )
        if not processed:
            _log("   ⚠  No readable files in this chunk")
            continue

        _log(f"   CLAP encoding {len(processed)} files…", stage=2)

        for batch_start in range(0, len(processed), batch_size):
            batch = processed[batch_start: batch_start + batch_size]

            valid_batch:    list[dict[str, Any]] = []
            all_segments:   list[np.ndarray]     = []
            segment_counts: list[int]            = []
            for row in batch:
                if not row["segments"]:
                    _log(f"   ⚠  Skipping {row['filename']} — no segments produced")
                    failed_count += 1
                    if len(_result.errors) < 50:
                        _result.errors.append(f"{row['filename']} :: no segments produced")
                    continue
                valid_batch.append(row)
                all_segments.extend(row["segments"])
                segment_counts.append(len(row["segments"]))

            if not valid_batch:
                continue

            batch_embeddings = encode_audio_segments(
                all_segments, processor=processor, model=model,
                batch_size=audio_batch_size,
            )

            conn = get_connection()
            conn.autocommit = False
            cur  = conn.cursor()
            try:
                ensure_track_query_exists(cur)
                ptr            = 0
                batch_inserted = 0
                batch_updated  = 0
                for row, seg_count in zip(valid_batch, segment_counts):
                    seg_vectors     = batch_embeddings[ptr: ptr + seg_count]
                    ptr            += seg_count
                    audio_embedding = weighted_average(seg_vectors, [1.0] * len(seg_vectors))
                    if not first_dim_checked:
                        validate_vector_dim(cur, int(len(audio_embedding)))
                        first_dim_checked = True
                    _log(f"   Building contract: {row['filename']}", stage=3)
                    selected_tags = select_best_tags(audio_embedding, tag_index=tag_index)
                    contract      = build_track_contract(selected_tags, row)
                    embeddings    = create_track_embeddings(
                        contract, audio_embedding,
                        processor=processor, model=model,
                        text_batch_size=text_batch_size,
                    )
                    _log(f"   Upserting: {row['filename']}", stage=4)
                    status = upsert_track_query(
                        cur, row, contract, embeddings,
                        seg_sec=segment_seconds,
                        clap_model_name=clap_model_name,
                    )
                    if status == "inserted":
                        inserted_count += 1
                        batch_inserted += 1
                    else:
                        updated_count += 1
                        batch_updated += 1
                conn.commit()
                done_so_far = min(batch_start + len(valid_batch), len(processed))
                _log(
                    f"   ✅ Saved {done_so_far}/{len(processed)} in chunk | "
                    f"inserted: {inserted_count} | updated: {updated_count}"
                )
            except Exception as exc:
                conn.rollback()
                _log(f"❌ Batch failed: {exc}")
                _log(traceback.format_exc())
                raise
            finally:
                safe_close_cursor(cur)
                safe_close_connection(conn)

        processed_so_far = min(chunk_idx * chunk_size, len(files))
        _log(
            f"✅ Progress {processed_so_far}/{len(files)} source files | "
            f"inserted: {inserted_count} | updated: {updated_count} | failed: {failed_count}"
        )

    _result.inserted   = inserted_count
    _result.updated    = updated_count
    _result.failed     = failed_count
    _result.duration_s = time.monotonic() - _start_time
    _log("🎉 Import finished successfully")
    _log(
        f"   Final | inserted: {inserted_count} | updated: {updated_count} | "
        f"failed: {failed_count} | duration: {_result.duration_s:.1f}s"
    )
    return _result


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    def _cli_log(msg: str, stage: int = -1) -> None:
        print(msg, flush=True)

    print(f"{SCRIPT_NAME} v{SCRIPT_VERSION} — CLI mode", flush=True)
    print(f"CLAP model: {CLAP_MODEL_NAME} | Device: {DEVICE}", flush=True)
    print(f"Music source: {'R2:' + BUCKET_NAME if CLOUD_MUSIC else str(MUSIC_DIR)}", flush=True)
    try:
        run_import(log_fn=_cli_log)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        raise
