#!/usr/bin/env python3
from __future__ import annotations

"""
music_library_importer.py

A utility for importing a local music library into the track_query table with CLAP-based semantic encoding and tag inference.

Key design decisions:
- no legacy writes to embeddings / track_metadata / track_scores,
- no hardcoded scene templates,
- primary and only persisted output is track_query,
- no writes to tracks; track_query is the authoritative track registry.

The track contract mirrors scene_query as closely as possible:
- semantic_title_en
- description_en
- tags_summary_en
- track_music_semantics
- track_music_targets
- track_tag_selection
- track_clap_prompt_ensemble
- embedding_main
- embedding_tags
- embedding_clap_ensemble
- embedding_hybrid
- embedding_audio (track-only auxiliary embedding)
"""

import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from psycopg2.extras import Json
from transformers import ClapModel, ClapProcessor

load_dotenv()

PIPELINE_VERSION = "new_parallel_importer_v9_0"
MUSIC_DIR = Path(os.getenv("MUSIC_DIR", "clanMusic"))
TAGS_FILE = Path(os.getenv("TAGS_FILE", "tags_v2.json"))
TARGET_SR = int(os.getenv("TARGET_SR", "48000"))
SEGMENT_SECONDS = int(os.getenv("SEGMENT_SECONDS", "20"))
NUM_SEGMENTS = int(os.getenv("NUM_SEGMENTS", "5"))
WORKERS = int(os.getenv("WORKERS", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "64"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
TEXT_BATCH_SIZE = int(os.getenv("TEXT_BATCH_SIZE", "32"))
AUDIO_BATCH_SIZE = int(os.getenv("AUDIO_BATCH_SIZE", "8"))
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))
SKIP_EXISTING = os.getenv("SKIP_EXISTING", "1") == "1"
CLAP_MODEL_NAME = os.getenv("CLAP_MODEL_NAME", "laion/clap-htsat-unfused")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

HF_TOKEN = os.getenv("HF_TOKEN")

# MIN_TRACK_ID = 1144

DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "sslmode": os.getenv("PGSSLMODE"),
    "channel_binding": os.getenv("PGCHANNELBINDING"),
}

# Directly aligned to screenplay_parser_final.py allowed vocab.
CATEGORY_RULES: Dict[str, Dict[str, float]] = {
    "emotion": {"top_k": 3, "threshold": 0.255, "margin": 0.020, "required": 0},
    "scene": {"top_k": 2, "threshold": 0.245, "margin": 0.020, "required": 0},
    "energy": {"top_k": 1, "threshold": 0.215, "margin": 0.015, "required": 1},
    "tempo": {"top_k": 1, "threshold": 0.215, "margin": 0.015, "required": 1},
    "rhythm": {"top_k": 1, "threshold": 0.215, "margin": 0.015, "required": 1},
    "intensity_shape": {"top_k": 1, "threshold": 0.220, "margin": 0.015, "required": 1},
    "narrative": {"top_k": 3, "threshold": 0.235, "margin": 0.018, "required": 1},
    "sound_character": {"top_k": 3, "threshold": 0.230, "margin": 0.018, "required": 1},
    "usage": {"top_k": 2, "threshold": 0.225, "margin": 0.018, "required": 0},
    "instrumentation": {"top_k": 3, "threshold": 0.225, "margin": 0.018, "required": 0},
    "atmosphere": {"top_k": 2, "threshold": 0.225, "margin": 0.018, "required": 0},
    "special": {"top_k": 2, "threshold": 0.225, "margin": 0.018, "required": 0},
}

TAG_PROMPT_WEIGHTS = [0.52, 0.28, 0.20]
ENSEMBLE_PROMPT_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
TEXT_HYBRID_WEIGHTS = [0.45, 0.20, 0.35]  # same as screenplay_parser_final.py

PROFILE_HINTS = {
    "dialogue": {"dialogue", "conversation", "background_music", "underscore", "dialogue_safe", "restrained", "sparse", "intimate"},
    "investigation": {"investigation", "mystery", "curiosity", "tension", "suspense", "textured", "pulsating"},
    "action": {"action", "danger", "aggression", "driving", "fast", "very_fast", "high", "very_high", "hits", "impacts", "pulses"},
    "dramatic_scene": {"dramatic_scene", "sadness", "grief", "melancholy", "hope", "emotional_support", "strings", "piano"},
    "horror_scene": {"horror_scene", "fear", "suspense", "dark", "eerie", "tension_building", "textured"},
    "thriller_scene": {"thriller_scene", "tension", "suspense", "anxiety", "investigation", "driving", "pulsating"},
    "romantic_scene": {"romantic_scene", "love", "romance", "warmth", "intimate", "sparse"},
    "comedic_scene": {"comedic_scene", "playful", "joy", "happiness"},
    "transition": {"transition", "bridge", "release", "background", "underscore", "moderate"},
    "montage": {"montage", "determination", "hope", "uplifting", "driving", "gradual_build"},
    "climax": {"climax", "climax_peak", "epic", "heroism", "very_high", "hits", "impacts", "trailer_music"},
    "resolution": {"resolution", "relief", "peace", "release", "warm", "calm"},
    "aftermath": {"aftermath", "grief", "sadness", "calm", "peace", "sparse", "very_slow", "slow"},
}

CONTRARY_TAGS: Dict[str, List[str]] = {
    "very_low": ["very_high", "high"],
    "low": ["very_high"],
    "high": ["very_low", "dialogue_safe"],
    "very_high": ["very_low", "low", "dialogue_safe", "restrained"],
    "very_slow": ["very_fast", "fast"],
    "slow": ["very_fast"],
    "fast": ["very_slow", "dialogue_safe"],
    "very_fast": ["very_slow", "slow"],
    "dialogue_safe": ["hits", "impacts", "very_high", "climax_peak", "trailer_music"],
    "restrained": ["hits", "impacts", "very_high", "climax_peak", "trailer_music"],
    "background_music": ["hits", "impacts"],
    "trailer_music": ["dialogue_safe", "restrained", "very_low", "very_slow"],
    "dark": ["warm", "uplifting", "playful"],
    "uplifting": ["dark", "eerie", "fear"],
    "playful": ["dark", "suspense", "fear", "grief"],
    "joy": ["fear", "grief", "melancholy"],
    "fear": ["joy", "relief", "peace"],
    "peace": ["aggression", "danger", "very_high"],
    "intimate": ["wide"],
    "wide": ["intimate"],
    "sparse": ["dense"],
    "dense": ["sparse"],
    "clean": ["distorted", "noisy"],
    "distorted": ["clean"],
    "analog": ["digital"],
    "digital": ["analog"],
}

# ----------------------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------------------

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def safe_close_cursor(cursor):
    try:
        if cursor is not None and not getattr(cursor, "closed", True):
            cursor.close()
    except Exception:
        pass


def safe_close_connection(conn):
    try:
        if conn is not None and not getattr(conn, "closed", True):
            conn.close()
    except Exception:
        pass


def ensure_track_query_exists(cursor):
    cursor.execute("SELECT to_regclass('public.track_query')")
    row = cursor.fetchone()
    if not row or row[0] is None:
        raise RuntimeError("track_query table is missing. Run track_query_setup.py first.")


def get_existing_filepaths(cursor, scanned_paths: Sequence[str]) -> set[str]:
    if not scanned_paths:
        return set()
    cursor.execute("SELECT filepath FROM track_query WHERE filepath = ANY(%s)", (list(scanned_paths),))
    return {str(row[0]) for row in cursor.fetchall()}


def get_existing_track_stats(cursor, scanned_paths: Sequence[str]) -> Mapping[str, int]:
    cursor.execute("SELECT COUNT(*) FROM track_query")
    total_saved = int(cursor.fetchone()[0])
    if not scanned_paths:
        return {
            "total_saved_in_track_query": total_saved,
            "already_saved_in_current_scan": 0,
            "new_in_current_scan": 0,
        }
    cursor.execute("SELECT COUNT(*) FROM track_query WHERE filepath = ANY(%s)", (list(scanned_paths),))
    already_saved = int(cursor.fetchone()[0])
    return {
        "total_saved_in_track_query": total_saved,
        "already_saved_in_current_scan": already_saved,
        "new_in_current_scan": int(len(scanned_paths) - already_saved),
    }


def get_track_query_vector_dim(cursor, column_name: str) -> Optional[int]:
    cursor.execute(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'track_query'::regclass
          AND attname = %s
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


def validate_vector_dim(cursor, expected_dim: int):
    for col in ["embedding_audio", "embedding_main", "embedding_tags", "embedding_clap_ensemble", "embedding_hybrid"]:
        db_dim = get_track_query_vector_dim(cursor, col)
        if db_dim is not None and db_dim != expected_dim:
            raise ValueError(f"Vector dim mismatch for track_query.{col}: DB expects {db_dim}, importer produced {expected_dim}")


def upsert_track_query(cursor, file_row: Mapping[str, Any], contract: Mapping[str, Any], embeddings: Mapping[str, np.ndarray]) -> str:
    cursor.execute("SELECT id FROM track_query WHERE filepath = %s", (file_row["path"],))
    existing = cursor.fetchone()
    status = "updated" if existing else "inserted"
    cursor.execute(
        """
        INSERT INTO track_query (
            filename, filepath, duration_sec, bpm, musical_key,
            semantic_title_en, description_en, tags_summary_en,
            track_music_semantics, track_music_targets, track_tag_selection, track_clap_prompt_ensemble,
            embedding_audio, embedding_main, embedding_tags, embedding_clap_ensemble, embedding_hybrid,
            audio_analysis, segmentation
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s
        )
        ON CONFLICT (filepath) DO UPDATE SET
            filename = EXCLUDED.filename,
            filepath = EXCLUDED.filepath,
            duration_sec = EXCLUDED.duration_sec,
            bpm = EXCLUDED.bpm,
            musical_key = EXCLUDED.musical_key,
            semantic_title_en = EXCLUDED.semantic_title_en,
            description_en = EXCLUDED.description_en,
            tags_summary_en = EXCLUDED.tags_summary_en,
            track_music_semantics = EXCLUDED.track_music_semantics,
            track_music_targets = EXCLUDED.track_music_targets,
            track_tag_selection = EXCLUDED.track_tag_selection,
            track_clap_prompt_ensemble = EXCLUDED.track_clap_prompt_ensemble,
            embedding_audio = EXCLUDED.embedding_audio,
            embedding_main = EXCLUDED.embedding_main,
            embedding_tags = EXCLUDED.embedding_tags,
            embedding_clap_ensemble = EXCLUDED.embedding_clap_ensemble,
            embedding_hybrid = EXCLUDED.embedding_hybrid,
            audio_analysis = EXCLUDED.audio_analysis,
            segmentation = EXCLUDED.segmentation,
            updated_at = NOW()
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
            Json(contract["track_music_semantics"]),
            Json(contract["track_music_targets"]),
            Json(contract["track_tag_selection"]),
            Json(contract["track_clap_prompt_ensemble"]),
            embeddings["embedding_audio"].tolist(),
            embeddings["embedding_main"].tolist(),
            embeddings["embedding_tags"].tolist(),
            embeddings["embedding_clap_ensemble"].tolist(),
            embeddings["embedding_hybrid"].tolist(),
            Json(file_row["analysis"]),
            Json({
                "segment_seconds": SEGMENT_SECONDS,
                "num_segments": len(file_row["segments"]),
                "segment_ranges": file_row["segment_ranges"],
                "pipeline_version": PIPELINE_VERSION,
                "clap_model_name": CLAP_MODEL_NAME,
                "hybrid_weights": embeddings["hybrid_weights"],
            }),
        ),
    )
    return status


# ----------------------------------------------------------------------------
# Math / serialization helpers
# ----------------------------------------------------------------------------

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    if a_arr.shape != b_arr.shape:
        return 0.0

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))

def l2_normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else arr / norm


def weighted_average(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> np.ndarray:
    if vectors is None:
        raise ValueError("No vectors provided")
    vectors_arr = np.asarray(vectors, dtype=np.float32)
    weights_arr = np.asarray(weights, dtype=np.float32)
    if vectors_arr.size == 0:
        raise ValueError("No vectors provided")
    if vectors_arr.ndim != 2:
        raise ValueError(f"Expected 2D vectors array, got shape={vectors_arr.shape}")
    if weights_arr.ndim != 1:
        raise ValueError(f"Expected 1D weights array, got shape={weights_arr.shape}")
    if vectors_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError(f"vectors and weights must have the same length: {vectors_arr.shape[0]} != {weights_arr.shape[0]}")
    total = float(weights_arr.sum())
    if total == 0.0:
        raise ValueError("sum(weights) must be > 0")
    out = (vectors_arr * weights_arr[:, None]).sum(axis=0) / total
    return l2_normalize(out)


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [ensure_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# ----------------------------------------------------------------------------
# Tags
# ----------------------------------------------------------------------------

def load_tags(tags_file: Path = TAGS_FILE) -> Dict[str, Dict[str, Any]]:
    with tags_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_tags(raw_tags: Mapping[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for category, category_data in raw_tags.items():
        category_pl = category_data["category_pl"]
        for tag_en, tag_pl in category_data["tags"].items():
            rows.append({
                "category": category,
                "category_pl": category_pl,
                "name": tag_en,
                "name_pl": tag_pl,
            })
    return rows


def build_tag_prompts(tag_row: Mapping[str, str]) -> List[str]:
    category = tag_row["category"].replace("_", " ")
    tag_en = tag_row["name"].replace("_", " ")
    return [
        f"cinematic instrumental music with {tag_en}",
        f"music tagged as {category}: {tag_en}",
        f"film underscore with {tag_en}",
    ]


# ----------------------------------------------------------------------------
# Audio preprocessing
# ----------------------------------------------------------------------------

def is_probably_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def scan_music_files(music_dir: Path) -> List[str]:
    files: List[str] = []
    skipped_hidden = 0
    for root, _dirs, names in os.walk(music_dir):
        for name in names:
            p = Path(root) / name
            if is_probably_appledouble(p):
                skipped_hidden += 1
                continue
            if name.lower().endswith(AUDIO_EXTENSIONS):
                files.append(str(p))
    files.sort()
    if skipped_hidden:
        print(f"⚠ Skipped AppleDouble sidecar files: {skipped_hidden}")
    return files


def sample_segments(audio: np.ndarray, sr: int) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
    segment_len = SEGMENT_SECONDS * sr
    if len(audio) < segment_len:
        padded = np.pad(audio, (0, segment_len - len(audio)))
        return [(padded, (0.0, float(len(audio) / sr)))]
    if NUM_SEGMENTS <= 1:
        return [(audio[:segment_len], (0.0, float(SEGMENT_SECONDS)))]
    max_start = len(audio) - segment_len
    starts = np.linspace(0, max_start, NUM_SEGMENTS, dtype=int)
    return [(audio[start:start + segment_len], (start / sr, (start + segment_len) / sr)) for start in starts.tolist()]


def bpm_to_tempo_target(bpm: Optional[int]) -> str:
    if bpm is None:
        return "moderate"
    if bpm < 55:
        return "very_slow"
    if bpm < 80:
        return "slow"
    if bpm < 118:
        return "moderate"
    if bpm < 145:
        return "fast"
    return "very_fast"


def rms_to_energy_target(rms: float) -> str:
    if rms < 0.030:
        return "very_low"
    if rms < 0.060:
        return "low"
    if rms < 0.110:
        return "medium"
    if rms < 0.180:
        return "high"
    return "very_high"


def zcr_to_rhythm_target(zcr: float, bpm: Optional[int]) -> str:
    if bpm is not None and bpm >= 120:
        return "driving"
    if zcr < 0.035:
        return "floating"
    if zcr < 0.070:
        return "steady"
    if zcr < 0.110:
        return "syncopated"
    return "irregular"


def infer_intensity_shape(rms: float, zcr: float) -> str:
    if rms > 0.14:
        return "climax_peak"
    if zcr > 0.09:
        return "pulsating"
    if rms > 0.08:
        return "gradual_build"
    return "static"


def centroid_to_sound_hints(centroid: float) -> List[str]:
    if centroid < 1600:
        return ["intimate", "analog"]
    if centroid < 3200:
        return ["atmospheric", "textured"]
    return ["clean", "wide"]


def preprocess_file(path: str) -> Dict[str, Any]:
    try:
        audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception as exc:
        return {
            "path": path,
            "filename": os.path.basename(path),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    duration = float(librosa.get_duration(y=audio, sr=sr))
    segments = sample_segments(audio, sr)
    seg_audio = [seg for seg, _ in segments]
    seg_ranges = [{"start_sec": round(s, 3), "end_sec": round(e, 3)} for _, (s, e) in segments]

    bpm = None
    key = None
    try:
        bpm_val, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = int(round(float(bpm_val)))
    except Exception:
        pass

    if len(audio) >= 2048:
        try:
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key = keys[int(chroma_mean.argmax())]
        except Exception:
            pass

    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    zcr = float(librosa.feature.zero_crossing_rate(audio).mean()) if len(audio) else 0.0
    spectral_centroid = 0.0
    spectral_rolloff = 0.0
    if len(audio) >= 1024:
        try:
            spectral_centroid = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())
            spectral_rolloff = float(librosa.feature.spectral_rolloff(y=audio, sr=sr).mean())
        except Exception:
            pass

    analysis = {
        "rms": rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "energy_target_hint": rms_to_energy_target(rms),
        "tempo_target_hint": bpm_to_tempo_target(bpm),
        "rhythm_target_hint": zcr_to_rhythm_target(zcr, bpm),
        "intensity_shape_hint": infer_intensity_shape(rms, zcr),
        "sound_character_hints": centroid_to_sound_hints(spectral_centroid),
    }

    return {
        "path": path,
        "filename": os.path.basename(path),
        "status": "ok",
        "duration": duration,
        "segments": seg_audio,
        "segment_ranges": seg_ranges,
        "bpm": bpm,
        "key": key,
        "analysis": analysis,
    }


# ----------------------------------------------------------------------------
# CLAP encoding
# ----------------------------------------------------------------------------

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


def encode_texts(texts: Sequence[str], processor: ClapProcessor, model: ClapModel, batch_size: int = TEXT_BATCH_SIZE) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        inputs = processor(text=batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            raw = model.get_text_features(**inputs)
            emb = _to_numpy_features(raw)
        vectors.extend([l2_normalize(row) for row in emb])
    return vectors


def encode_audio_segments(segments: Sequence[np.ndarray], processor: ClapProcessor, model: ClapModel, batch_size: int = AUDIO_BATCH_SIZE) -> np.ndarray:
    encoded: List[np.ndarray] = []
    for i in range(0, len(segments), batch_size):
        batch = list(segments[i:i + batch_size])
        inputs = processor(audio=batch, sampling_rate=TARGET_SR, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            raw = model.get_audio_features(**inputs)
            emb = _to_numpy_features(raw)
        encoded.extend([l2_normalize(row) for row in emb])
    return np.asarray(encoded, dtype=np.float32)


# ----------------------------------------------------------------------------
# Tag embedding index and selection
# ----------------------------------------------------------------------------

def build_tag_embedding_index(tags: Sequence[Mapping[str, str]], processor: ClapProcessor, model: ClapModel) -> Dict[Tuple[str, str], Dict[str, Any]]:
    all_prompts: List[str] = []
    prompt_groups: List[List[str]] = []
    for tag_row in tags:
        prompts = build_tag_prompts(tag_row)
        prompt_groups.append(prompts)
        all_prompts.extend(prompts)
    prompt_vectors = encode_texts(all_prompts, processor=processor, model=model, batch_size=TEXT_BATCH_SIZE)
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    ptr = 0
    for tag_row, prompts in zip(tags, prompt_groups):
        group_vecs = prompt_vectors[ptr:ptr + len(prompts)]
        ptr += len(prompts)
        ensemble = weighted_average(group_vecs, TAG_PROMPT_WEIGHTS)
        index[(tag_row["category"], tag_row["name"])] = {**dict(tag_row), "prompts": prompts, "embedding": ensemble}
    return index


def select_best_tags(audio_embedding: np.ndarray, tag_index: Mapping[Tuple[str, str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for payload in tag_index.values():
        row = {
            "category": payload["category"],
            "category_pl": payload["category_pl"],
            "name": payload["name"],
            "name_pl": payload["name_pl"],
            "score": float(cosine(audio_embedding, payload["embedding"])),
        }
        grouped.setdefault(str(payload["category"]), []).append(row)

    selected: List[Dict[str, Any]] = []
    for category, rows in grouped.items():
        rows.sort(key=lambda x: x["score"], reverse=True)
        rule = CATEGORY_RULES.get(category, {"top_k": 1, "threshold": 0.22, "margin": 0.015, "required": 0})
        top_k = int(rule["top_k"])
        threshold = float(rule["threshold"])
        margin = float(rule["margin"])
        required = int(rule["required"])
        kept = 0
        for idx, row in enumerate(rows[: top_k + 1]):
            if idx >= top_k:
                break
            next_score = rows[idx + 1]["score"] if idx + 1 < len(rows) else -1.0
            local_margin = float(row["score"] - next_score)
            if row["score"] >= threshold and (local_margin >= margin or kept < required or row["score"] >= threshold + 0.035):
                selected.append(row)
                kept += 1
        if kept == 0 and required > 0 and rows and rows[0]["score"] >= threshold - 0.020:
            selected.append(rows[0])

    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected


# ----------------------------------------------------------------------------
# Contract builders
# ----------------------------------------------------------------------------

def tags_by_category(selected_tags: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
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
        0.45 * scores.get("dialogue_safe", 0.0) +
        0.28 * scores.get("restrained", 0.0) +
        0.24 * scores.get("background_music", 0.0) +
        0.22 * scores.get("underscore", 0.0) +
        0.18 * scores.get("sparse", 0.0) +
        0.16 * scores.get("intimate", 0.0) +
        0.14 * scores.get("low", 0.0) +
        0.10 * scores.get("very_low", 0.0)
    )
    risk = (
        0.34 * scores.get("hits", 0.0) +
        0.34 * scores.get("impacts", 0.0) +
        0.25 * scores.get("trailer_music", 0.0) +
        0.21 * scores.get("climax_peak", 0.0) +
        0.18 * scores.get("very_high", 0.0) +
        0.14 * scores.get("peak", 0.0) +
        0.10 * scores.get("dense", 0.0)
    )
    return max(0.0, min(1.0, 0.50 + safe - risk))


def derive_weight_profile_candidates(scores: Mapping[str, float], grouped: Mapping[str, List[Dict[str, Any]]], dialogue_safe_score: float) -> List[str]:
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
    return [name for name, value in candidates if value >= max(0.24, best - 0.12)][:3]


def derive_must_not_tags(selected_tags: Sequence[Mapping[str, Any]], semantics: Mapping[str, Any], targets: Mapping[str, Any]) -> List[str]:
    selected_names = [str(t["name"]) for t in selected_tags]
    selected_set = set(selected_names)
    scores = score_lookup(selected_tags)
    ranked: Dict[str, float] = {}
    for name in selected_names:
        for opposite in CONTRARY_TAGS.get(name, []):
            if opposite not in selected_set:
                ranked[opposite] = max(ranked.get(opposite, 0.0), max(0.20, scores.get(name, 0.0)))
    dialogue_safe_score = float(semantics.get("dialogue_safe_score", 0.0))
    if dialogue_safe_score >= 0.62:
        for name in ["hits", "impacts", "trailer_music", "very_high", "climax_peak"]:
            if name not in selected_set:
                ranked[name] = max(ranked.get(name, 0.0), 0.50 + 0.30 * dialogue_safe_score)
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
    return [name for name, _ in sorted(ranked.items(), key=lambda x: x[1], reverse=True) if name not in selected_set][:6]


def build_track_music_semantics(selected_tags: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped = tags_by_category(selected_tags)
    scores = score_lookup(selected_tags)
    dialogue_safe_score = derive_dialogue_safe_score(scores)
    emotional_direction = top_names(grouped.get("emotion", []), 4)
    narrative_function = top_names(grouped.get("narrative", []), 3) or ["underscore"]
    return {
        "emotional_direction": emotional_direction,
        "narrative_function": narrative_function,
        "weight_profile_candidates": derive_weight_profile_candidates(scores, grouped, dialogue_safe_score),
        "dialogue_safe_score": round(dialogue_safe_score, 6),
        "dialogue_safe": dialogue_safe_score >= 0.60,
    }


def build_track_music_targets(selected_tags: Sequence[Mapping[str, Any]], file_row: Mapping[str, Any]) -> Dict[str, Any]:
    grouped = tags_by_category(selected_tags)
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
        "energy_target": first_or("energy", str(analysis.get("energy_target_hint", "medium"))),
        "tempo_target": first_or("tempo", str(analysis.get("tempo_target_hint", "moderate"))),
        "rhythm_target": first_or("rhythm", str(analysis.get("rhythm_target_hint", "steady"))),
        "intensity_shape_target": first_or("intensity_shape", str(analysis.get("intensity_shape_hint", "static"))),
        "sound_character_target": list(dict.fromkeys(sound_character))[:4],
    }


def build_track_tag_selection(selected_tags: Sequence[Mapping[str, Any]], semantics: Mapping[str, Any], targets: Mapping[str, Any]) -> Dict[str, Any]:
    should_have = [str(t["name"]) for t in selected_tags[:8]]
    must_not = derive_must_not_tags(selected_tags, semantics, targets)
    should_set = set(should_have)
    must_not = [x for x in must_not if x not in should_set][:6]
    scores = [float(t["score"]) for t in selected_tags] if selected_tags else [0.0]
    return {
        "should_have_tags": should_have,
        "must_not_tags": must_not,
        "confidence_profile": {
            "mean_selected_tag_score": float(np.mean(scores)),
            "max_selected_tag_score": float(np.max(scores)),
            "min_selected_tag_score": float(np.min(scores)),
        },
    }


def build_track_texts(selected_tags: Sequence[Mapping[str, Any]], file_row: Mapping[str, Any], semantics: Mapping[str, Any], targets: Mapping[str, Any]) -> Dict[str, str]:
    emotions = semantics.get("emotional_direction", []) or ["cinematic"]
    narrative = semantics.get("narrative_function", []) or ["underscore"]
    intensity_shape = str(targets.get("intensity_shape_target", "static")).replace("_", " ")
    energy = str(targets.get("energy_target", "medium")).replace("_", " ")
    sound_character = ", ".join([str(x).replace("_", " ") for x in targets.get("sound_character_target", [])[:2]]) or "atmospheric"
    semantic_title_en = f"{emotions[0].replace('_', ' ').title()} {narrative[0].replace('_', ' ').title()} Cue"
    description_en = (
        f"Instrumental cue with {', '.join(emotions[:2]).replace('_', ' ')} and "
        f"{', '.join(narrative[:2]).replace('_', ' ')}; {energy} energy, "
        f"{intensity_shape} shape, {sound_character} character."
    )
    tag_summary_items = [str(t["name"]).replace("_", " ") for t in selected_tags[:6]]
    tags_summary_en = ", ".join(tag_summary_items) if tag_summary_items else "cinematic instrumental underscore"
    return {
        "semantic_title_en": semantic_title_en[:120],
        "description_en": description_en[:240],
        "tags_summary_en": tags_summary_en[:240],
    }


def build_track_clap_prompt_ensemble(texts: Mapping[str, str], semantics: Mapping[str, Any], targets: Mapping[str, Any], tag_selection: Mapping[str, Any]) -> Dict[str, str]:
    emotions = ", ".join(semantics.get("emotional_direction", [])[:3]).replace("_", " ") or "cinematic emotion"
    narrative = ", ".join(semantics.get("narrative_function", [])[:3]).replace("_", " ") or "underscore"
    sound = ", ".join(targets.get("sound_character_target", [])[:3]).replace("_", " ") or "atmospheric"
    tags = ", ".join(tag_selection.get("should_have_tags", [])[:6]).replace("_", " ") or "cinematic"
    return {
        "semantic_scene_prompt": texts["description_en"][:120],
        "music_for_scene_prompt": f"instrumental music for {texts['semantic_title_en'].lower()}"[:120],
        "emotion_prompt": emotions[:120],
        "narrative_prompt": narrative[:120],
        "sonic_prompt": sound[:120],
        "tag_prompt": tags[:120],
        "concise_core_prompt": texts["tags_summary_en"][:120],
    }


def build_track_contract(selected_tags: Sequence[Mapping[str, Any]], file_row: Mapping[str, Any]) -> Dict[str, Any]:
    semantics = build_track_music_semantics(selected_tags)
    targets = build_track_music_targets(selected_tags, file_row)
    tag_selection = build_track_tag_selection(selected_tags, semantics, targets)
    texts = build_track_texts(selected_tags, file_row, semantics, targets)
    prompt_ensemble = build_track_clap_prompt_ensemble(texts, semantics, targets, tag_selection)
    return {
        **texts,
        "track_music_semantics": semantics,
        "track_music_targets": targets,
        "track_tag_selection": tag_selection,
        "track_clap_prompt_ensemble": prompt_ensemble,
    }


def create_track_embeddings(contract: Mapping[str, Any], audio_embedding: np.ndarray, processor: ClapProcessor, model: ClapModel) -> Dict[str, Any]:
    main_text = f"{contract['semantic_title_en']}. {contract['description_en']}".strip()
    embedding_main = np.asarray(encode_texts([main_text], processor, model, batch_size=1)[0], dtype=np.float32)
    embedding_tags = np.asarray(encode_texts([contract["tags_summary_en"]], processor, model, batch_size=1)[0], dtype=np.float32)
    ensemble_texts = [
        contract["track_clap_prompt_ensemble"]["semantic_scene_prompt"],
        contract["track_clap_prompt_ensemble"]["music_for_scene_prompt"],
        contract["track_clap_prompt_ensemble"]["emotion_prompt"],
        contract["track_clap_prompt_ensemble"]["narrative_prompt"],
        contract["track_clap_prompt_ensemble"]["sonic_prompt"],
        contract["track_clap_prompt_ensemble"]["tag_prompt"],
        contract["track_clap_prompt_ensemble"]["concise_core_prompt"],
    ]
    ensemble_vectors = encode_texts(ensemble_texts, processor, model, batch_size=7)
    embedding_clap_ensemble = weighted_average(ensemble_vectors, ENSEMBLE_PROMPT_WEIGHTS)
    # Match scene_query hybrid semantics exactly; audio is auxiliary only.
    embedding_hybrid = weighted_average(
        [embedding_main, embedding_tags, embedding_clap_ensemble],
        TEXT_HYBRID_WEIGHTS,
    )
    return {
        "embedding_audio": audio_embedding,
        "embedding_main": embedding_main,
        "embedding_tags": embedding_tags,
        "embedding_clap_ensemble": embedding_clap_ensemble,
        "embedding_hybrid": embedding_hybrid,
        "hybrid_weights": {
            "main": TEXT_HYBRID_WEIGHTS[0],
            "tags": TEXT_HYBRID_WEIGHTS[1],
            "clap_ensemble": TEXT_HYBRID_WEIGHTS[2],
            "audio": 0.0,
            "mode": "scene_query_compatible_textual_hybrid",
        },
    }


# ----------------------------------------------------------------------------
# Main import flow
# ----------------------------------------------------------------------------

def iter_chunks(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def run_import():
    print(f"🎵 Loading CLAP model: {CLAP_MODEL_NAME}")
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
    model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(DEVICE)
    model.eval()

    print(f"🏷 Loading tags: {TAGS_FILE}")
    raw_tags = load_tags()
    flat_tags = flatten_tags(raw_tags)
    print(f"🧠 Building tag embedding index for {len(flat_tags)} tags...")
    tag_index = build_tag_embedding_index(flat_tags, processor=processor, model=model)

    print(f"📂 Scanning music folder: {MUSIC_DIR}")
    files = scan_music_files(MUSIC_DIR)
    print(f"Tracks found: {len(files)}")
    if not files:
        print("⚠ No audio files found. Nothing to import.")
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        ensure_track_query_exists(cur)
        stats = get_existing_track_stats(cur, files)
        print("Registry table: track_query")
        print(f"Already saved total: {stats['total_saved_in_track_query']}")
        print(f"Already saved from current scan: {stats['already_saved_in_current_scan']} / {len(files)}")
        print(f"New files to process: {stats['new_in_current_scan']}")
        existing_paths = get_existing_filepaths(cur, files) if SKIP_EXISTING else set()
        print(f"Skipping existing files: {len(existing_paths)}")
        if existing_paths:
            files = [p for p in files if p not in existing_paths]
        print(f"Files left for processing: {len(files)}")
        cur.close(); conn.close()
    except Exception:
        safe_close_cursor(cur)
        safe_close_connection(conn)
        raise

    if not files:
        print("✅ Everything from current scan is already saved in track_query.")
        return

    inserted_count = 0
    updated_count = 0
    failed_count = 0
    first_dim_checked = False

    chunks = iter_chunks(files, CHUNK_SIZE)
    total_chunks = len(chunks)

    for chunk_idx, chunk_files in enumerate(chunks, start=1):
        print(f"🧩 Chunk {chunk_idx}/{total_chunks} | files {((chunk_idx-1)*CHUNK_SIZE)+1}-{((chunk_idx-1)*CHUNK_SIZE)+len(chunk_files)} / {len(files)}")
        print(f"   Preprocessing chunk of {len(chunk_files)} files...")
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            processed_all = list(ex.map(preprocess_file, chunk_files))

        processed = [row for row in processed_all if row.get("status") == "ok"]
        failed = [row for row in processed_all if row.get("status") != "ok"]
        failed_count += len(failed)
        if failed:
            print(f"   ⚠ Failed in chunk: {len(failed)}")
            for row in failed[:5]:
                print(f"      - {row['path']} :: {row.get('error_type')} :: {row.get('error')}")

        if not processed:
            print("   ⚠ No readable files in this chunk")
            continue

        conn = get_connection()
        conn.autocommit = False
        cur = conn.cursor()
        try:
            ensure_track_query_exists(cur)
            for batch_start in range(0, len(processed), BATCH_SIZE):
                batch = processed[batch_start: batch_start + BATCH_SIZE]
                all_segments: List[np.ndarray] = []
                segment_counts: List[int] = []
                for row in batch:
                    if not row["segments"]:
                        raise ValueError(f"No segments produced for file: {row['filename']}")
                    all_segments.extend(row["segments"])
                    segment_counts.append(len(row["segments"]))
                batch_embeddings = encode_audio_segments(all_segments, processor=processor, model=model, batch_size=AUDIO_BATCH_SIZE)
                ptr = 0
                batch_inserted = 0
                batch_updated = 0
                for row, seg_count in zip(batch, segment_counts):
                    seg_vectors = batch_embeddings[ptr:ptr + seg_count]
                    ptr += seg_count
                    audio_embedding = weighted_average(seg_vectors, [1.0] * len(seg_vectors))
                    if not first_dim_checked:
                        validate_vector_dim(cur, int(len(audio_embedding)))
                        first_dim_checked = True
                    selected_tags = select_best_tags(audio_embedding, tag_index=tag_index)
                    contract = build_track_contract(selected_tags, row)
                    embeddings = create_track_embeddings(contract, audio_embedding, processor=processor, model=model)
                    tq_status = upsert_track_query(cur, row, contract, embeddings)
                    if tq_status == "inserted":
                        inserted_count += 1
                        batch_inserted += 1
                    else:
                        updated_count += 1
                        batch_updated += 1
                conn.commit()
                print(f"   ✅ Saved chunk batch up to {min(batch_start + len(batch), len(processed))} / {len(processed)} readable in this chunk | inserted: {inserted_count} | updated: {updated_count}")
        except Exception as exc:
            conn.rollback()
            print("❌ Import failed:")
            print(exc)
            print(traceback.format_exc())
            raise
        finally:
            safe_close_cursor(cur)
            safe_close_connection(conn)

        processed_source_files = min(chunk_idx * CHUNK_SIZE, len(files))
        print(f"✅ Progress {processed_source_files} / {len(files)} source files | inserted: {inserted_count} | updated: {updated_count} | failed: {failed_count}")

    print("🎉 Import finished successfully")
    print(f"Final counts | inserted: {inserted_count} | updated: {updated_count} | failed: {failed_count}")


if __name__ == "__main__":
    run_import()
