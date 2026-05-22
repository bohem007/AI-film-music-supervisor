#!/usr/bin/env python3
"""
music_library_importer.py  —  AI Music Supervisor · Track Importer  (v3.0)
=========================================================================

Purpose
-------
Streamlit GUI for importing a local (or Cloudflare R2 bucket) 
music library and indexing audio tracks into the track_query
PostgreSQL table.  Each track is preprocessed with librosa, CLAP-encoded
with laion/clap-htsat-unfused, tagged semantically, and stored with five
embedding vectors for downstream scene-music matching.

Pages
-----
  Dashboard       — DB stats, quick links to other pages.
  Run Import      — Start the import pipeline; live console log (auto-refresh).
  Track Inspector — Search track_query by name or tag; full contract + player.
  Configuration   — View all pipeline constants and category rules.

Key options (via .env or environment variables)
-----------------------------------------------
  CLOUD_MUSIC       Use Cloudflare R2 instead of local folder (true/false)
  CF_ACCOUNT_ID     Cloudflare account ID (R2 dashboard sidebar)
  CF_TOKEN_VALUE    Cloudflare API token
  R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT_URL — S3 credentials
  BUCKET_NAME       R2 bucket name
  MUSIC_DIR         Local folder to scan when CLOUD_MUSIC=false (default: clanMusic)
  TAGS_FILE         Tag vocabulary JSON               (default: tags_v2.json)
  CLAP_MODEL_NAME   HuggingFace model ID              (default: laion/clap-htsat-unfused)
  SKIP_EXISTING     Skip already-indexed tracks       (default: 1)
  WORKERS           Preprocessing parallelism         (default: 6)
  CHUNK_SIZE        Files per import chunk            (default: 64)
  TARGET_SR         Audio sample rate                 (default: 48000)
  NUM_SEGMENTS      Segments sampled per track        (default: 5)
  SEGMENT_SECONDS   Segment length in seconds         (default: 20)
  PGHOST / PGDATABASE / PGUSER / PGPASSWORD / PGSSLMODE  — Neon DB connection

Changes vs v3.01
------------------
1.  Version bumped to 3.02.
2.  Cloud mode: S3 client created once and reused (_get_s3_client).
    Creating a new boto3 client per file caused a full TLS handshake
    on every download — with 64 files/chunk this looked like a freeze.
    A module-level _s3_client is now initialised on first call and
    reused for all subsequent downloads.  connect_timeout=10 s,
    read_timeout=60 s, max_attempts=3 added via botocore Config.
3.  Cloud mode: per-file progress logged during chunk preprocessing
    so the Console shows "☁ [N/64] filename" for each download.
"""

from __future__ import annotations

# Remove any stale _NumbaAutoStub entries left by previous sessions.
import sys as _sys_cleanup
_stub_keys = [k for k, v in list(_sys_cleanup.modules.items())
              if type(v).__name__ == "_NumbaAutoStub"]
for _k in _stub_keys:
    _sys_cleanup.modules.pop(_k, None)
del _sys_cleanup, _stub_keys


import html as _html
import json
import os
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
import psycopg2
import streamlit as st
import torch
from dotenv import load_dotenv
from psycopg2.extras import Json
from transformers import ClapModel, ClapProcessor

load_dotenv()


# ── CONSTANTS  (all env-var defaults; overridable at run time) ──

SCRIPT_NAME        = "music_library_importer.py"
SCRIPT_VERSION     = "3.02"
PIPELINE_VERSION   = f"{SCRIPT_NAME[:-3]}_v{SCRIPT_VERSION}"
MUSIC_DIR          = Path(os.getenv("MUSIC_DIR", "clanMusic"))
TAGS_FILE          = Path(os.getenv("TAGS_FILE", "tags_v2.json"))
TARGET_SR          = int(os.getenv("TARGET_SR", "48000"))
SEGMENT_SECONDS    = int(os.getenv("SEGMENT_SECONDS", "20"))
NUM_SEGMENTS       = int(os.getenv("NUM_SEGMENTS", "5"))
WORKERS            = int(os.getenv("WORKERS", "6"))
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "64"))
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "8"))
TEXT_BATCH_SIZE    = int(os.getenv("TEXT_BATCH_SIZE", "32"))
AUDIO_BATCH_SIZE   = int(os.getenv("AUDIO_BATCH_SIZE", "8"))
VECTOR_DIM         = int(os.getenv("VECTOR_DIM", "512"))
SKIP_EXISTING      = os.getenv("SKIP_EXISTING", "1") == "1"
CLAP_MODEL_NAME    = os.getenv("CLAP_MODEL_NAME", "laion/clap-htsat-unfused")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_EXTENSIONS   = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
HF_TOKEN           = os.getenv("HF_TOKEN")

CLOUD_MUSIC        = os.getenv("CLOUD_MUSIC", "false").lower() == "true"
CF_ACCOUNT_ID      = os.getenv("CF_ACCOUNT_ID", "")
CF_TOKEN_VALUE     = os.getenv("CF_TOKEN_VALUE", "")
R2_ACCESS_KEY_ID   = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL    = os.getenv("R2_ENDPOINT_URL", "")
BUCKET_NAME        = os.getenv("BUCKET_NAME", "")

DB_CONFIG: Dict[str, Optional[str]] = {
    "host":            os.getenv("PGHOST"),
    "dbname":          os.getenv("PGDATABASE"),
    "user":            os.getenv("PGUSER"),
    "password":        os.getenv("PGPASSWORD"),
    "sslmode":         os.getenv("PGSSLMODE"),
    "channel_binding": os.getenv("PGCHANNELBINDING"),
}

CATEGORY_RULES: Dict[str, Dict[str, float]] = {
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

TAG_PROMPT_WEIGHTS      = [0.52, 0.28, 0.20]
ENSEMBLE_PROMPT_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
TEXT_HYBRID_WEIGHTS     = [0.45, 0.20, 0.35]   # must match screenplay_parser

PROFILE_HINTS: Dict[str, set] = {
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

CONTRARY_TAGS: Dict[str, List[str]] = {
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



# ── DATABASE HELPERS ──

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


def get_existing_track_stats(cursor, scanned_paths: Sequence[str]) -> Dict[str, int]:
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
    embeddings: Mapping[str, Any],
    seg_sec: int,
    clap_model_name: str,
) -> str:
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
            embeddings["embedding_audio"].tolist(),
            embeddings["embedding_main"].tolist(),
            embeddings["embedding_tags"].tolist(),
            embeddings["embedding_clap_ensemble"].tolist(),
            embeddings["embedding_hybrid"].tolist(),
            Json(ensure_serializable(file_row["analysis"])),
            Json({
                "segment_seconds":  seg_sec,
                "num_segments":     len(file_row["segments"]),
                "segment_ranges":   file_row["segment_ranges"],
                "pipeline_version": PIPELINE_VERSION,
                "clap_model_name":  clap_model_name,
                "hybrid_weights":   ensure_serializable(embeddings["hybrid_weights"]),
            }),
        ),
    )
    row = cursor.fetchone()
    return "inserted" if (row and row[0]) else "updated"



# ── MATH & SERIALIZATION HELPERS ──

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.size == 0 or b_arr.size == 0 or a_arr.shape != b_arr.shape:
        return 0.0
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
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



# ── TAG HELPERS ──

def load_tags(tags_file: Path = TAGS_FILE) -> Dict[str, Dict[str, Any]]:
    with tags_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_tags(raw_tags: Mapping[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
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


def build_tag_prompts(tag_row: Mapping[str, str]) -> List[str]:
    category = tag_row["category"].replace("_", " ")
    tag_en   = tag_row["name"].replace("_", " ")
    return [
        f"cinematic instrumental music with {tag_en}",
        f"music tagged as {category}: {tag_en}",
        f"film underscore with {tag_en}",
    ]



# ── AUDIO PREPROCESSING ──

def is_probably_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def scan_music_files(music_dir: Path, log_fn=None) -> List[str]:
    files:          List[str] = []
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



def scan_cloud_files(log_fn=None) -> List[str]:
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
        keys: List[str] = []
        skipped_hidden = 0
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME):
            for obj in page.get("Contents", []):
                key = obj["Key"]
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
# Creating a new client per file causes massive overhead (TLS handshake,
# credential resolution) and makes sequential downloads appear frozen.
_s3_client = None

def _get_s3_client():
    """Return the shared S3 client, creating it on first call."""
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
            connect_timeout   = 10,
            read_timeout      = 60,
            retries           = {"max_attempts": 3, "mode": "standard"},
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
) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
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


def centroid_to_sound_hints(centroid: float) -> List[str]:
    if centroid < 1600: return ["intimate", "analog"]
    if centroid < 3200: return ["atmospheric", "textured"]
    return ["clean", "wide"]


def preprocess_file(
    path: str,
    segment_seconds: int = SEGMENT_SECONDS,
    num_segments:    int = NUM_SEGMENTS,
) -> Dict[str, Any]:
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



# ── CLAP ENCODING ──

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
) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []
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
    encoded: List[np.ndarray] = []
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



# ── TAG EMBEDDING INDEX & SELECTION ──

def build_tag_embedding_index(
    tags: Sequence[Mapping[str, str]],
    processor: ClapProcessor,
    model: ClapModel,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    all_prompts:   List[str]       = []
    prompt_groups: List[List[str]] = []
    for tag_row in tags:
        prompts = build_tag_prompts(tag_row)
        prompt_groups.append(prompts)
        all_prompts.extend(prompts)
    prompt_vectors = encode_texts(
        all_prompts, processor=processor, model=model, batch_size=TEXT_BATCH_SIZE
    )
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    ptr = 0
    for tag_row, prompts in zip(tags, prompt_groups):
        group_vecs = prompt_vectors[ptr: ptr + len(prompts)]
        ptr       += len(prompts)
        ensemble   = weighted_average(group_vecs, TAG_PROMPT_WEIGHTS)
        index[(tag_row["category"], tag_row["name"])] = {
            **dict(tag_row), "prompts": prompts, "embedding": ensemble,
        }
    return index


def select_best_tags(
    audio_embedding: np.ndarray,
    tag_index: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for payload in tag_index.values():
        row = {
            "category":    payload["category"],
            "category_pl": payload["category_pl"],
            "name":        payload["name"],
            "name_pl":     payload["name_pl"],
            "score":       float(cosine(audio_embedding, payload["embedding"])),
        }
        grouped.setdefault(str(payload["category"]), []).append(row)

    selected: List[Dict[str, Any]] = []
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



# ── CONTRACT BUILDERS ──

def tags_by_category(
    selected_tags: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for tag in selected_tags:
        grouped.setdefault(str(tag["category"]), []).append(dict(tag))
    for rows in grouped.values():
        rows.sort(key=lambda x: float(x["score"]), reverse=True)
    return grouped


def top_names(rows: Sequence[Mapping[str, Any]], limit: int) -> List[str]:
    return [str(row["name"]) for row in rows[:limit]]


def score_lookup(selected_tags: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
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
    grouped: Mapping[str, List[Dict[str, Any]]],
    dialogue_safe_score: float,
) -> List[str]:
    selected_names = set(scores.keys())
    candidates: List[Tuple[str, float]] = []
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
) -> List[str]:
    selected_names = [str(t["name"]) for t in selected_tags]
    selected_set   = set(selected_names)
    scores         = score_lookup(selected_tags)
    ranked: Dict[str, float] = {}
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
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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
) -> Dict[str, str]:
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
) -> Dict[str, str]:
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
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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
    return {
        "embedding_audio":         audio_embedding,
        "embedding_main":          embedding_main,
        "embedding_tags":          embedding_tags,
        "embedding_clap_ensemble": embedding_clap_ensemble,
        "embedding_hybrid":        embedding_hybrid,
        "hybrid_weights": {
            "main":          TEXT_HYBRID_WEIGHTS[0],
            "tags":          TEXT_HYBRID_WEIGHTS[1],
            "clap_ensemble": TEXT_HYBRID_WEIGHTS[2],
            "audio":         0.0,
            "mode":          "scene_query_compatible_textual_hybrid",
        },
    }



# ── MAIN IMPORT PIPELINE ──

def iter_chunks(items: Sequence[str], size: int) -> List[List[str]]:
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
) -> None:
    """
    Full import pipeline.  All parameters default to the module-level
    constants but can be overridden by the GUI or a direct call.

    log_fn(message, stage_index) is called at each stage boundary.
    stage_index: 0=scan/init, 1=preprocess, 2=encode, 3=contract, 4=upsert
    """
    def _log(msg: str, stage: int = -1) -> None:
        if log_fn is not None:
            log_fn(msg, stage)

    music_path = Path(music_dir)
    tags_path  = Path(tags_file)

    _log(f"🎵 Loading CLAP model: {clap_model_name}", stage=0)
    processor = ClapProcessor.from_pretrained(clap_model_name, token=HF_TOKEN)
    model     = ClapModel.from_pretrained(clap_model_name, token=HF_TOKEN).to(DEVICE)
    model.eval()

    _log(f"🏷  Loading tags: {tags_path}", stage=0)
    raw_tags  = load_tags(tags_path)
    flat_tags = flatten_tags(raw_tags)
    _log(f"🧠 Building tag embedding index for {len(flat_tags)} tags…", stage=0)
    tag_index = build_tag_embedding_index(flat_tags, processor=processor, model=model)

    if CLOUD_MUSIC:
        _log(f"☁  Scanning R2 bucket: {BUCKET_NAME}", stage=0)
        files = scan_cloud_files(log_fn=log_fn)
    else:
        _log(f"📂 Scanning music folder: {music_path}", stage=0)
        files = scan_music_files(music_path, log_fn=log_fn)
    _log(f"   Tracks found: {len(files)}", stage=0)
    if not files:
        _log("⚠  No audio files found. Nothing to import.")
        return

    conn = get_connection()
    cur  = conn.cursor()
    try:
        ensure_track_query_exists(cur)
        stats = get_existing_track_stats(cur, files)
        _log(f"   Registry table: track_query")
        _log(f"   Total already in DB:   {stats['total_saved_in_track_query']}")
        _log(f"   Already saved (scan):  {stats['already_saved_in_current_scan']} / {len(files)}")
        _log(f"   New files to process:  {stats['new_in_current_scan']}")
        existing_paths = get_existing_filepaths(cur, files) if skip_existing else set()
        _log(f"   Skipping existing:     {len(existing_paths)}")
        if existing_paths:
            files = [p for p in files if p not in existing_paths]
        _log(f"   Files left to import:  {len(files)}")
    finally:
        safe_close_cursor(cur)
        safe_close_connection(conn)

    if not files:
        _log("✅ All files already indexed in track_query. Nothing to do.")
        return

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
        if not processed:
            _log("   ⚠  No readable files in this chunk")
            continue

        _log(f"   CLAP encoding {len(processed)} files…", stage=2)

        for batch_start in range(0, len(processed), batch_size):
            batch = processed[batch_start: batch_start + batch_size]

            valid_batch:    List[Dict[str, Any]] = []
            all_segments:   List[np.ndarray]     = []
            segment_counts: List[int]            = []
            for row in batch:
                if not row["segments"]:
                    _log(f"   ⚠  Skipping {row['filename']} — no segments produced")
                    failed_count += 1
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

    _log("🎉 Import finished successfully")
    _log(
        f"   Final | inserted: {inserted_count} | updated: {updated_count} | failed: {failed_count}"
    )



# ── STREAMLIT GUI ──

st.set_page_config(
    page_title=f"AI Music Supervisor — Track Importer v{SCRIPT_VERSION}",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Design tokens ────────────────────────────────────────────────────────── */
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

    /* Typography stacks */
    --mono:  'JetBrains Mono', 'Courier New', monospace;
    --sans:  'Inter', system-ui, sans-serif;
    --serif: 'Source Serif 4', Georgia, serif;
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--txt) !important;
    font-family: var(--sans) !important;
    font-size: 15px !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }

/* ── App header (full-width title bar) ───────────────────────────────────── */
.app-header {
    padding: 1.6rem 0 1.1rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 1.6rem;
}
.app-title {
    font-family: var(--serif);
    font-size: 2.1rem;
    font-weight: 400;
    color: var(--accent);
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.01em;
}
.app-sub {
    font-family: var(--mono);
    font-size: 0.70rem;
    color: var(--txt-3);
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-top: 5px;
}

/* ── Section headings ────────────────────────────────────────────────────── */
.s-title {
    font-family: var(--serif);
    font-size: 1.35rem;
    font-weight: 400;
    color: var(--txt);
    margin-bottom: 2px;
    letter-spacing: -0.01em;
}
.s-sub {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--txt-3);
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 3px rgba(180,120,60,.07);
}
.card-title {
    font-family: var(--serif);
    font-size: 1.00rem;
    font-weight: 400;
    color: var(--accent);
    margin-bottom: 0.35rem;
}
.card-body {
    font-family: var(--sans);
    font-size: 0.85rem;
    color: var(--txt-2);
    line-height: 1.65;
}

/* ── Badges ──────────────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 0.67rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.b-ok   { background: rgba(45,138,94,.10);  color: var(--green);  border: 1px solid rgba(45,138,94,.28); }
.b-warn { background: var(--accent-bg);     color: var(--accent); border: 1px solid rgba(201,106,26,.30); }
.b-err  { background: rgba(192,57,43,.09);  color: var(--red);    border: 1px solid rgba(192,57,43,.28); }
.b-info { background: rgba(90,62,40,.07);   color: var(--txt-2);  border: 1px solid var(--border); }

/* ── Metric tiles ────────────────────────────────────────────────────────── */
.m-row  { display: flex; gap: 10px; margin-bottom: 1rem; flex-wrap: wrap; }
.m-tile {
    flex: 1;
    min-width: 100px;
    background: var(--accent-bg);
    border: 1px solid var(--border-s);
    border-radius: 9px;
    padding: 12px 14px;
    text-align: center;
}
.m-val  { font-family: var(--serif); font-size: 1.70rem; color: var(--accent); line-height: 1; margin-bottom: 3px; }
.m-lbl  { font-family: var(--mono);  font-size: 0.60rem; color: var(--txt-3); text-transform: uppercase; letter-spacing: 0.10em; }

/* ── Pipeline step indicator ─────────────────────────────────────────────── */
.pipe   { display: flex; margin-bottom: 1.3rem; gap: 3px; }
.p-step {
    flex: 1;
    padding: 8px 4px;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--txt-3);
    border-top: 2px solid var(--border);
    background: var(--bg-raised);
    border-radius: 0 0 4px 4px;
}
.p-done   { color: var(--green); border-top-color: var(--green);  background: rgba(45,138,94,.06); }
.p-active { color: var(--accent); border-top-color: var(--accent); background: var(--accent-bg); }

/* ── Log console ─────────────────────────────────────────────────────────── */
.log {
    background: #1e1208;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 15px;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #f5d9b8;
    max-height: 360px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.65;
}

/* ── Score bar ───────────────────────────────────────────────────────────── */
.sb-wrap  { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.sb-label { font-family: var(--mono); font-size: 0.68rem; color: var(--txt-2); width: 210px; }
.sb-track { flex: 1; height: 4px; background: var(--border-s); border-radius: 3px; }
.sb-fill  { height: 4px; border-radius: 3px; }
.sb-val   { font-family: var(--mono); font-size: 0.68rem; color: var(--txt-3); width: 36px; text-align: right; }

/* ── Streamlit widget overrides ──────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--txt) !important;
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
    border-radius: 7px !important;
    transition: all 0.13s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-bg) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent-bg) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 500 !important;
}
.stButton > button[kind="primary"]:hover {
    background: rgba(201,106,26,.18) !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--txt) !important;
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
    border-radius: 7px !important;
}
.stProgress > div > div { background: var(--accent) !important; }
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] summary { color: var(--txt-2) !important; font-family: var(--sans) !important; }
button[data-baseweb="tab"] {
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
    color: var(--txt-2) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
label, .stCheckbox label {
    color: var(--txt-2) !important;
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stFileUploader"] label { color: var(--txt-2) !important; }
hr { border-color: var(--border) !important; }
[data-testid="stDataFrame"] { border-radius: 8px; border: 1px solid var(--border); }

/* Sidebar collapse-button "keybo…" label — hide text, keep SVG arrow. */
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

/* Audio player — match warm amber palette */
audio {
    width: 100% !important; height: 36px !important;
    border-radius: 8px !important; border: 1px solid var(--border) !important;
    background: var(--bg-raised) !important; accent-color: var(--accent) !important;
    outline: none !important;
}
audio::-webkit-media-controls-panel { background: var(--bg-raised) !important; border-radius: 8px !important; }
audio::-webkit-media-controls-play-button,
audio::-webkit-media-controls-mute-button { color: var(--accent) !important; }
audio::-webkit-media-controls-timeline { accent-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session-state defaults ────────────────────────────────────────────────────
_SS_DEFAULTS: Dict[str, Any] = {
    "page":           "dashboard",
    "import_log":     [],
    "import_running": False,
    "import_done":    False,
    "import_stats":   {},
    "import_stage":   -1,
    "db_status":      {},
    "lib_files":      None,
    "lib_db_rows":    None,
    "lib_track":      None,
    # Editable parameters — stored in session state so they persist across reruns.
    "p_music_dir":   str(MUSIC_DIR),
    "p_skip":        SKIP_EXISTING,
    # Track inspector search state
    "insp_results":       None,   # pd.DataFrame | None — last search results
    "insp_track":         None,   # dict | None — currently loaded track detail
    "insp_starts_with":   False,  # bool — prefix-only name match when True
    "insp_name":          "",     # str  — current name search text
    "insp_over_limit":    False,  # bool — True when >200 results exist
    "insp_name_key":      0,      # int  — incremented on Clear to force widget remount
    "insp_last_name":     "",     # str  — name query used in last search
    "insp_last_tags":     [],     # list — tags used in last search
    "insp_selected_id":   None,   # int|None — ID from last row click / auto-load
    "lib_selected_id":    None,   # int|None — ID from last row click (Library)
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_print_lock = threading.Lock()


# ── UI helpers ────────────────────────────────────────────────────────────────

def _safe(val: Any) -> str:
    """HTML-escape any value before inserting into unsafe_allow_html markup."""
    return _html.escape(str(val)) if val is not None else "–"


def _badge(text: str, kind: str = "info") -> str:
    k = {"ok": "b-ok", "warn": "b-warn", "err": "b-err"}.get(kind, "b-info")
    return f'<span class="badge {k}">{_safe(text)}</span>'


def _metrics(tiles: List[Tuple[str, str]]) -> None:
    parts = "".join(
        f'<div class="m-tile"><div class="m-val">{_safe(v)}</div>'
        f'<div class="m-lbl">{_safe(l)}</div></div>'
        for l, v in tiles
    )
    st.markdown(f'<div class="m-row">{parts}</div>', unsafe_allow_html=True)


def _pipe(steps: List[str], done: int, active: int = -1) -> None:
    html = '<div class="pipe">'
    for i, s in enumerate(steps):
        cls  = "p-done" if i < done else ("p-active" if i == active else "")
        html += f'<div class="p-step {cls}">{_safe(s)}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _header(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="s-title">{_safe(title)}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="s-sub">{_safe(sub)}</div>', unsafe_allow_html=True)


def _log(lines: List[str]) -> None:
    raw  = "\n".join(lines[-150:]) if lines else "— awaiting run —"
    text = _html.escape(raw)
    for old, new in [
        ("✅", '<span style="color:#4aab7a">✅</span>'),
        ("❌", '<span style="color:#d95f4b">❌</span>'),
        ("⚠",  '<span style="color:#c9960a">⚠</span>'),
        ("[ERROR]", '<span style="color:#d95f4b">[ERROR]</span>'),
        ("[INFO]",  '<span style="color:#e8a96a">[INFO]</span>'),
    ]:
        text = text.replace(old, new)
    st.markdown(f'<div class="log">{text}</div>', unsafe_allow_html=True)


def _score_bar(label: str, value: float) -> None:
    pct   = max(0.0, min(1.0, value)) * 100
    color = "#2d8a5e" if pct >= 65 else "#c96a1a" if pct >= 35 else "#c0392b"
    st.markdown(
        f'<div class="sb-wrap">'
        f'<div class="sb-label">{_safe(label)}</div>'
        f'<div class="sb-track">'
        f'<div class="sb-fill" style="width:{pct:.1f}%;background:{color}"></div>'
        f'</div>'
        f'<div class="sb-val">{value:.2f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── DB check ──────────────────────────────────────────────────────────────────

def _check_db() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False, "error": None,
        "track_query": 0, "scene_query": 0, "matches": 0,
    }
    try:
        conn = get_connection()
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
                # subsequent queries on this connection succeed.
                conn.rollback()
                result[key] = 0
        cur.close()
        conn.close()
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ── Import thread ──────────────────────────────────────────────────────────────

def _start_import(**kwargs) -> None:
    log: List[str]         = ["[INFO] Starting import pipeline…"]
    result: Dict[str, Any] = {"running": True, "done": False, "stats": {}}

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

    def _worker(log: List[str], result: Dict[str, Any]) -> None:
        try:
            run_import(**kwargs, log_fn=_stage_cb)

            ins = upd = fail = 0
            for line in reversed(log):
                if "Final |" in line and "inserted:" in line:
                    for part in line.split("|"):
                        p = part.strip()
                        if p.startswith("inserted:"): ins  = int(p.split(":")[1].strip())
                        elif p.startswith("updated:"): upd  = int(p.split(":")[1].strip())
                        elif p.startswith("failed:"):  fail = int(p.split(":")[1].strip())
                    break

            total: Any = "–"
            try:
                conn = get_connection()
                cur  = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM track_query")
                total = int(cur.fetchone()[0])
                cur.close()
                conn.close()
            except Exception:
                pass

            result["stats"] = {"inserted": ins, "updated": upd, "failed": fail, "total": total}
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
    r = st.session_state.get("_imp_result", {})
    if r:
        if not r.get("running", True):
            st.session_state.import_running = False
        if "stage" in r:
            st.session_state.import_stage = r["stage"]
        if r.get("done") and not st.session_state.import_done:
            st.session_state.import_done  = True
            st.session_state.import_stats = r.get("stats", {})


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def _page_dashboard() -> None:
    st.markdown(
        '<div class="app-header">'
        '<div class="app-title">🎵 Track Importer</div>'
        f'<div class="app-sub">AI Music Supervisor · {SCRIPT_NAME} · v{SCRIPT_VERSION} · CLAP audio embedding pipeline</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    db = st.session_state.db_status
    ok = db.get("ok", False)

    # Show metrics immediately — they are populated by the auto-check in main().
    # The Refresh button is active only when the last check failed.
    if db:
        _metrics([
            ("Tracks indexed", str(db.get("track_query", "–"))),
            ("Scenes parsed",  str(db.get("scene_query", "–"))),
            ("Match rows",     str(db.get("matches",     "–"))),
            ("Last check",     "✓ DB reached" if ok else "✗ unreachable"),
        ])

    if st.button("↻  Refresh DB stats", disabled=ok):
        with st.spinner("Querying…"):
            st.session_state.db_status = _check_db()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, title, body, page in [
        (c1, "▶  Run Import",
         "Scan the library, preprocess in parallel, CLAP-encode, select tags, "
         "build contracts, upsert into <code>track_query</code>.",
         "importer"),
        (c2, "🔍  Track Inspector",
         "Search <code>track_query</code> by track name or tag category. "
         "Select a result to view its full contract and audio analysis.",
         "inspector"),
        (c3, "⚙  Configuration",
         "Review all pipeline constants: batch sizes, embedding weights, "
         "category rules, CONTRARY_TAGS.",
         "config"),
    ]:
        with col:
            st.markdown(
                f'<div class="card"><div class="card-title">{_safe(title)}</div>'
                f'<div class="card-body">{body}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"→ {title.split()[-1]}", key=f"d_{page}", width="stretch"):
                st.session_state.page = page
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    _header("Pipeline stages", "five-stage ETL flow")
    _pipe(
        ["Scan files", "Preprocess (CPU)", "CLAP encode (GPU)", "Tag + contract", "DB upsert"],
        done=5 if st.session_state.import_done else 0,
    )
    st.caption(
        "Each track: librosa preprocessing (BPM · key · RMS · ZCR) → "
        "CLAP audio embedding → tag cosine scoring → contract build → "
        "5 embeddings (audio, main, tags, ensemble, hybrid) → ON CONFLICT DO UPDATE upsert."
    )

    if st.session_state.import_done and st.session_state.import_stats:
        st.markdown("<hr>", unsafe_allow_html=True)
        _header("Last run", "session statistics")
        s = st.session_state.import_stats
        _metrics([
            ("Inserted",    str(s.get("inserted", "–"))),
            ("Updated",     str(s.get("updated",  "–"))),
            ("Failed",      str(s.get("failed",   "–"))),
            ("Total in DB", str(s.get("total",    "–"))),
        ])


def _page_importer() -> None:
    _header("Run Import", "set music directory · start pipeline · live log")

    active = st.session_state.import_stage if st.session_state.import_running else -1
    done   = 5 if st.session_state.import_done else 0
    _pipe(
        ["Scan files", "Preprocess (CPU)", "CLAP encode (GPU)", "Tag + contract", "DB upsert"],
        done=done, active=active,
    )

    # ── Parameters panel ─────────────────────────────────────────────────────
    with st.expander("⚙  Parameters", expanded=True):

        st.text_input(
            "Music directory",
            value=st.session_state.p_music_dir,
            key="p_music_dir",
            help="Path to the folder containing audio files to import. "
                 "Edit the MUSIC_DIR variable in .env to change the default.",
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
            st.text_input(
                "Tag vocabulary (TAGS_FILE)",
                value=str(TAGS_FILE),
                disabled=True,
            )
            st.text_input(
                "CLAP model (CLAP_MODEL_NAME)",
                value=CLAP_MODEL_NAME,
                disabled=True,
            )
            st.number_input(
                "CPU workers (WORKERS)",
                value=WORKERS,
                disabled=True,
            )
        with c2:
            st.number_input(
                "Chunk size (CHUNK_SIZE)",
                value=CHUNK_SIZE,
                disabled=True,
            )
            st.number_input(
                "GPU batch size (BATCH_SIZE)",
                value=BATCH_SIZE,
                disabled=True,
            )
            st.number_input(
                "Audio CLAP batch (AUDIO_BATCH_SIZE)",
                value=AUDIO_BATCH_SIZE,
                disabled=True,
            )
        with c3:
            st.number_input(
                "Text CLAP batch (TEXT_BATCH_SIZE)",
                value=TEXT_BATCH_SIZE,
                disabled=True,
            )
            st.number_input(
                "Segments per track (NUM_SEGMENTS)",
                value=NUM_SEGMENTS,
                disabled=True,
            )
            st.number_input(
                "Segment duration, s (SEGMENT_SECONDS)",
                value=SEGMENT_SECONDS,
                disabled=True,
            )

        st.caption(
            "Music directory and Skip already-indexed tracks are editable. "
            "All other parameters are locked to the values from the environment at startup — "
            "edit your .env file and restart the app to change them."
        )

    # ── Start button ─────────────────────────────────────────────────────────
    lbl = "▶  Start Import" if not st.session_state.import_running else "⏳  Running…"
    if st.button(lbl, type="primary",
                 disabled=st.session_state.import_running,
                 width="stretch"):
        _start_import(
            music_dir        = st.session_state.p_music_dir,
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
    _log(st.session_state.import_log)

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


def _page_config() -> None:
    _header("Configuration", "constants · embedding weights · category rules · contrary tags")

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
                "Controls `embedding_hybrid`. Must match `screenplay_parser_final.py` "
                "for cross-table cosine similarity to be valid."
            )
            for lbl, val in [
                ("embedding_main  (0.45)",    TEXT_HYBRID_WEIGHTS[0]),
                ("embedding_tags  (0.20)",    TEXT_HYBRID_WEIGHTS[1]),
                ("embedding_ensemble (0.35)", TEXT_HYBRID_WEIGHTS[2]),
            ]:
                _score_bar(lbl, val)
            st.markdown("<br>", unsafe_allow_html=True)
            _header("Tag prompt weights", "tag_prompt_weights")
            for lbl, val in [
                ("Prompt A — cinematic … with tag (0.52)", TAG_PROMPT_WEIGHTS[0]),
                ("Prompt B — music tagged as … (0.28)",    TAG_PROMPT_WEIGHTS[1]),
                ("Prompt C — film underscore … (0.20)",    TAG_PROMPT_WEIGHTS[2]),
            ]:
                _score_bar(lbl, val)
        with c2:
            _header("Ensemble prompt weights", "ensemble_prompt_weights")
            st.caption("7 prompts averaged → `embedding_clap_ensemble`.")
            for lbl, val in zip(
                ["semantic_scene_prompt", "music_for_scene_prompt", "emotion_prompt",
                 "narrative_prompt", "sonic_prompt", "tag_prompt", "concise_core_prompt"],
                ENSEMBLE_PROMPT_WEIGHTS,
            ):
                _score_bar(lbl, val)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "`top_k` = max tags selected · `threshold` = min cosine score · "
            "`margin` = min gap to next · `required` = always pick ≥1"
        )
        import pandas as pd
        df = pd.DataFrame(CATEGORY_RULES).T.reset_index().rename(columns={"index": "category"})
        st.dataframe(df, width="stretch", hide_index=True,
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
        st.dataframe(pd.DataFrame(rows_ct), width="stretch", hide_index=True)


def _lib_load_track(track_id: int) -> None:
    """Load full track record into lib_track session state (Library Browser)."""
    try:
        conn = get_connection()
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
        cur.close(); conn.close()
        if row:
            t = dict(zip(cols, row))
            analysis = t.get("audio_analysis") or {}
            seg      = t.get("segmentation")   or {}
            t["rms_mean"]         = round(float(analysis.get("rms", 0.0)), 6)
            t["zcr_mean"]         = round(float(analysis.get("zero_crossing_rate", 0.0)), 6)
            t["pipeline_version"] = seg.get("pipeline_version", "–")
            st.session_state.lib_track = t
        else:
            st.warning(f"No track with id={track_id}")
            st.session_state.lib_track = None
    except Exception as exc:
        st.error(f"DB error: {exc}")


def _page_library() -> None:
    """Library Browser — full track_query table with row-click detail + player."""
    _header("Library Browser", "track_query index · select track · full detail")
    import pandas as pd

    _db_total  = st.session_state.db_status.get("track_query", 0)
    _lbl_total = f"{_db_total:,} track{'s' if _db_total != 1 else ''} in index" if _db_total else "tracks in index"
    if st.button(f"⟳  Load / Refresh from DB  ({_lbl_total})", type="primary"):
        try:
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute("""
                SELECT id, filepath, filename, duration_sec, bpm, musical_key,
                       semantic_title_en, tags_summary_en, updated_at
                FROM track_query ORDER BY filename ASC
            """)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            cur.close(); conn.close()
            df = pd.DataFrame(rows, columns=cols)
            st.session_state.lib_db_rows   = df
            st.session_state.lib_selected_id = None
            st.session_state.lib_track       = None
            if not df.empty:
                _first_id = int(df.iloc[0]["id"])
                st.session_state.lib_selected_id = _first_id
                _lib_load_track(_first_id)
        except Exception as exc:
            st.error(f"DB error: {exc}")

    if st.session_state.lib_db_rows is not None:
        df = st.session_state.lib_db_rows
        n  = len(df)
        st.markdown(
            '<div style="margin-bottom:8px">' +
            _badge(f"{n:,} track{'s' if n != 1 else ''} loaded", "ok") +
            '</div>',
            unsafe_allow_html=True,
        )
        display_cols = [c for c in
            ["id", "filename", "bpm", "musical_key",
             "semantic_title_en", "tags_summary_en", "updated_at"]
            if c in df.columns]
        df_sorted = df[display_cols].sort_values("filename", ascending=True).reset_index(drop=True)

        event = st.dataframe(
            df_sorted, key="lib_df_widget", width="stretch",
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
            _clicked_id = int(df_sorted.iloc[_sel_rows[0]]["id"])
            if _clicked_id != (st.session_state.lib_track or {}).get("id"):
                st.session_state.lib_selected_id = _clicked_id
                _lib_load_track(_clicked_id)

        st.markdown("<hr>", unsafe_allow_html=True)
        _header("Track Detail", "full record — all fields except embeddings")
        valid_ids = df["id"].tolist()
        _cur_id   = st.session_state.get("lib_selected_id")
        _cur_id   = int(_cur_id) if (_cur_id and _cur_id in valid_ids) else int(valid_ids[0])
        t         = st.session_state.lib_track or {}
        _filepath = t.get("filepath", "")

        _id_col, _player_col = st.columns([1, 3])
        with _id_col:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.72rem;color:var(--txt-3);'
                f'margin-bottom:3px;text-transform:uppercase;letter-spacing:.08em">Track ID</div>'
                f'<div style="font-family:var(--mono);font-size:1.50rem;'
                f'color:var(--accent);font-weight:600;line-height:1">{_cur_id}</div>',
                unsafe_allow_html=True,
            )
        with _player_col:
            if _filepath and Path(_filepath).exists():
                st.markdown(
                    '<div style="margin-top:4px;padding:6px 10px;background:var(--bg-raised);'
                    'border:1px solid var(--border);border-radius:9px;">',
                    unsafe_allow_html=True,
                )
                st.audio(str(_filepath), autoplay=True)
                st.markdown("</div>", unsafe_allow_html=True)
            elif _filepath or _cur_id:
                st.markdown(
                    '<div style="font-family:var(--mono);font-size:0.72rem;'
                    'color:var(--txt-3);margin-top:6px;font-style:italic">'
                    '⚠ Track does not exist</div>',
                    unsafe_allow_html=True,
                )
        if t:
            _render_inspector_track(t)


def _page_inspector() -> None:
    """Track Inspector — search track_query by name or tags, view full record."""
    _header("Track Inspector", "search by name · filter by tag · view full record")

    # Load tag vocabulary from tags_v2.json
    _tag_vocab: Dict[str, List[str]] = {}
    try:
        import json as _json
        raw = _json.loads(Path(TAGS_FILE).read_text(encoding="utf-8"))
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
                    value=st.session_state.insp_name,
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

            starts_with = st.checkbox(
                "Match from start only (prefix)",
                key="insp_starts_with",
                help="When checked, only tracks whose name BEGINS with the "
                     "entered text are returned (faster, more precise).",
            )

        with sc2:
            sel_cats = st.multiselect(
                "Tag categories",
                options=sorted(_tag_vocab.keys()),
                key="insp_cats",
                help="Select one or more categories to filter by tag.",
            )
            available_tags: List[str] = sorted(
                set(t for c in sel_cats for t in _tag_vocab.get(c, []))
            )
            sel_tags = st.multiselect(
                "Tags (within selected categories)",
                options=available_tags,
                key="insp_tags",
                disabled=not available_tags,
                help="Track must contain ALL selected tags in its should_have_tags list.",
            )

        if st.button("Search", type="primary"):
            # Empty fields → search all tracks (no WHERE clause).
            _run_inspector_search(
                name_query.strip(), sel_tags,
                starts_with=starts_with,
            )

    results    = st.session_state.insp_results
    over_limit = st.session_state.get("insp_over_limit", False)
    if results is not None:
        _last_name = st.session_state.get("insp_last_name", "")
        _last_tags = st.session_state.get("insp_last_tags", [])
        if not _last_name and not _last_tags:
            _parts = "<b>Result for:</b> all tracks"
        else:
            _name_part = f'<b>Result for:</b> {_safe(_last_name)}' if _last_name else ""
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
            n = len(results)
            _count_lbl = f"{n:,} track{'s' if n != 1 else ''} found"
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.78rem;color:var(--txt-2);margin-bottom:4px">'
                f'{_parts}</div>'
                f'<div style="margin-bottom:10px">' + _badge(_count_lbl, "ok") + f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            import pandas as pd
            display_cols = [c for c in
                ["id", "filename", "bpm", "musical_key",
                 "semantic_title_en", "tags_summary_en", "updated_at"]
                if c in results.columns]
            results_sorted = results[display_cols].sort_values(
                "filename", ascending=True).reset_index(drop=True)

            event = st.dataframe(
                results_sorted, key="insp_df_widget", width="stretch",
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
            _id_col, _player_col = st.columns([1, 3])
            with _id_col:
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:0.72rem;color:var(--txt-3);'
                    f'margin-bottom:3px;text-transform:uppercase;letter-spacing:.08em">Track ID</div>'
                    f'<div style="font-family:var(--mono);font-size:1.50rem;'
                    f'color:var(--accent);font-weight:600;line-height:1">{_cur_id}</div>',
                    unsafe_allow_html=True,
                )
            with _player_col:
                if _filepath and Path(_filepath).exists():
                    st.markdown(
                        '<div style="margin-top:4px;padding:6px 10px;background:var(--bg-raised);'
                        'border:1px solid var(--border);border-radius:9px;">',
                        unsafe_allow_html=True,
                    )
                    st.audio(str(_filepath), autoplay=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif _filepath or _cur_id:
                    st.markdown(
                        '<div style="font-family:var(--mono);font-size:0.72rem;'
                        'color:var(--txt-3);margin-top:6px;font-style:italic">'
                        '⚠ Track does not exist</div>',
                        unsafe_allow_html=True,
                    )

    t = st.session_state.insp_track
    if t:
        _render_inspector_track(t)


def _run_inspector_search(
    name_query:  str,
    sel_tags:    List[str],
    starts_with: bool = False,
) -> None:
    """Execute search — no row limit. Empty args return all tracks."""
    import pandas as pd
    import json as _json
    try:
        conn = get_connection()
        cur  = conn.cursor()
        conditions: List[str] = []
        params: List[Any]     = []
        if name_query:
            like = f"{name_query}%" if starts_with else f"%{name_query}%"
            conditions.append("(filename ILIKE %s OR semantic_title_en ILIKE %s)")
            params.extend([like, like])
        for tag in sel_tags:
            conditions.append("track_tag_selection->'should_have_tags' @> %s::jsonb")
            params.append(_json.dumps([tag]))
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


def _load_inspector_track(track_id: int) -> None:
    """Load full record for one track (no embeddings) into session state."""
    try:
        conn = get_connection()
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
            t            = dict(zip(cols, row))
            analysis     = t.get("audio_analysis") or {}
            seg          = t.get("segmentation")   or {}
            t["rms_mean"]         = round(float(analysis.get("rms", 0.0)), 6)
            t["zcr_mean"]         = round(float(analysis.get("zero_crossing_rate", 0.0)), 6)
            t["pipeline_version"] = seg.get("pipeline_version", "–")
            st.session_state.insp_track = t
        else:
            st.warning(f"No track with id={track_id}")
            st.session_state.insp_track = None
    except Exception as exc:
        st.error(f"DB error: {exc}")


def _render_inspector_track(t: Dict[str, Any]) -> None:
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
        import json as _j
        tag_sel = t.get("track_tag_selection") or {}
        if isinstance(tag_sel, str):
            tag_sel = _j.loads(tag_sel)
        should  = tag_sel.get("should_have_tags", [])
        mustnot = tag_sel.get("must_not_tags",    [])
        conf    = tag_sel.get("confidence_profile", {})
        chips = " ".join(
            f'<span style="display:inline-flex;padding:2px 9px;border-radius:12px;'
            f'background:var(--accent-bg);border:1px solid rgba(201,106,26,.25);'
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
                import json as _j2; sem = _j2.loads(sem)
            st.json(sem)
        with c2:
            tgt = t.get("track_music_targets") or {}
            if isinstance(tgt, str):
                import json as _j3; tgt = _j3.loads(tgt)
            st.json(tgt)

    with st.expander("Prompt ensemble", expanded=False):
        ens = t.get("track_clap_prompt_ensemble") or {}
        if isinstance(ens, str):
            import json as _j4; ens = _j4.loads(ens)
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="padding:1.2rem 0 0.7rem">'
            '<div style="font-family:var(--serif);font-size:1.35rem;font-weight:400;'
            'color:var(--accent);line-height:1.15;letter-spacing:-0.01em">🎵 Track Importer</div>'
            f'<div style="font-family:var(--mono);font-size:0.65rem;color:#6b4423;'
            f'letter-spacing:.10em;text-transform:uppercase;margin-top:5px">'
            f'AI Music Supervisor · v{SCRIPT_VERSION}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        for pid, icon, label in [
            ("dashboard", "⬡", "Dashboard"),
            ("importer",  "▶", "Run Import"),
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
        db = st.session_state.db_status
        if db.get("ok"):
            st.markdown(
                _badge("DB reached", "ok") +
                f'<div style="margin-top:9px;font-family:var(--mono);font-size:0.67rem;'
                f'color:var(--txt-3);line-height:2.1">'
                f'tracks: {_safe(db.get("track_query","–"))}<br>'
                f'scenes: {_safe(db.get("scene_query","–"))}<br>'
                f'matches: {_safe(db.get("matches","–"))}</div>',
                unsafe_allow_html=True,
            )
        elif db.get("error"):
            st.markdown(_badge("DB error", "err"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        dev_badge = "ok" if DEVICE == "cuda" else "warn"
        st.markdown(_badge(f"device: {DEVICE}", dev_badge), unsafe_allow_html=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _sync_import()
    if not st.session_state.db_status:
        st.session_state.db_status = _check_db()
    _sidebar()
    page = st.session_state.page
    if   page == "dashboard": _page_dashboard()
    elif page == "importer":  _page_importer()
    elif page == "inspector": _page_inspector()
    elif page == "config":    _page_config()
    else:                     _page_dashboard()


if __name__ == "__main__":
    main()
