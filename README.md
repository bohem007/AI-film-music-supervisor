---
title: AI Music Supervisor
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---
# 🎬 AI Music Supervisor

**AI-powered scene-to-music matching engine and library indexing pipeline for film and TV production.**

AI Music Supervisor analyses a screenplay, splits it into scenes and themes, generates semantic CLAP embeddings, then ranks your music library against each scene using a multi-component scoring pipeline (embedding similarity + semantic analysis + music targets + tag selection + contextual penalties). Results are exported as ranked PDF and XLSX reports.

---
## Layer overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  ENTRYPOINT & GUI LAYER (Streamlit & Hugging Face Spaces)           │
│   app.py                   →  HF Spaces bootloader context wrapper  │
│   ai_music_supervisor.py   →  Unified multi-page frontend dashboard │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     │  imports engines & loads configs
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENGINE LAYER (Pure business logic, no Streamlit imports)           │
│   screenplay_parser_engine.py   →  parses scripts, runs LLM / CLAP  │
│   scene_music_matcher_engine.py →  runs vector queries, match score │
│   importer_pipeline.py          →  runs ETL audio CLAP indexing     │
│   pipeline_config.py            →  centralized shared weights/rules │
│   tags_v2.json                  →  loaded by engines at runtime     │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     │  interacts with (setup & retry via db_setup.py)
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DATABASE & STORAGE LAYER (Neon PostgreSQL Cloud / S3 / Local)      │
│   scene_query            ←  populated by parser engine              │
│   track_query            ←  populated by importer pipeline          │
│   scene_music_matches_v6 ←  populated by matcher engine scoring runs│
└─────────────────────────────────────────────────────────────────────┘
```

## Design principles

**GUI/engine separation.** Every engine module is importable without Streamlit. This enables CLI usage, headless batch jobs, and unit testing without a browser.

**Centralized Configuration.** Scoring weights, category selection thresholds, contrary tag rules, and configuration constants are managed centrally within [pipeline_config.py]. Weights are verified at import time via `validate_weights()` to catch config drift early.

**`embedding_hybrid` consistency.** The hybrid embedding is a weighted blend of three vectors: `0.45 × embedding_main + 0.20 × embedding_tags + 0.35 × embedding_clap_ensemble` (configured as `TEXT_HYBRID_WEIGHTS`). This blend is used for pgvector ANN candidate retrieval and as a scoring input. The weights **must be identical** in both the writer (parser/importer) and the reader (matcher).

**`track_query` as authoritative registry.** There is no separate `tracks` table. `track_query` is the single registry of all indexed audio files. `scene_music_matches_v6` references `track_query_id` only.

**Idempotency & Neon Resiliency.** The screenplay parser and importer database updates use SQL upserts (`ON CONFLICT`) to ensure safe re-running. Network connection handshakes are optimized, and cold-start wake-up delays in Neon serverless Postgres are resolved via exponential backoff retries in [db_setup.py].

---

## Database tables

Database schemas, triggers, and indices are defined idempotently in [db_setup.py]

### `track_query` (Music Library description)

Stores one row per audio file. `filepath` is the unique key (relative path within the music directory or R2 key).

| Column group | Columns |
|---|---|
| File | `filename`, `filepath`, `duration_sec` |
| Audio | `bpm`, `musical_key`, `audio_analysis` (JSON), `segmentation` (JSON) |
| LLM output | `semantic_title_en`, `description_en`, `tags_summary_en`, `track_music_semantics`, `track_music_targets`, `track_tag_selection`, `track_clap_prompt_ensemble` |
| Embeddings | `embedding_audio`, `embedding_main`, `embedding_tags`, `embedding_clap_ensemble`, `embedding_hybrid` |

### `scene_query`

Stores one row per (episode, scene, theme) triple. Each scene may have multiple themes (e.g. theme 1 = main dramatic arc, theme 2 = background ambience).

| Column group | Columns |
|---|---|
| Identity | `episode_nr`, `scene_nr`, `theme_nr` |
| Textual | `theme_title_pl`, `theme_title_en`, `description_en`, `theme_txt` |
| LLM output | `scene_music_semantics`, `scene_music_targets`, `scene_tag_selection`, `clap_prompt_ensemble` |
| Embeddings | `embedding_main`, `embedding_tags`, `embedding_clap_ensemble`, `embedding_hybrid` |

Unique constraint: `(episode_nr, scene_nr, theme_nr)`.


### `scene_music_matches_v6`

Stores the ranking output. One row per (episode, scene, theme, rank_position) tuple.

Key columns: `final_score`, `embedding_score`, `semantic_score`, `targets_score`, `tag_selection_score`, `penalty_total`, `match_explanation` (full JSON breakdown), `match_metadata`.

---

## Scoring formula

```
final_score = max(0, min(1,
    0.40 × embedding_score
  + 0.25 × semantic_score
  + 0.20 × targets_score
  + 0.15 × tag_selection_score
  − penalty_total
))
```

### Embedding score

```
embedding_score =
    0.15 × cosine(scene.embedding_main,  track.embedding_main)
  + 0.10 × cosine(scene.embedding_tags,  track.embedding_tags)
  + 0.20 × cosine(scene.embedding_clap_ensemble, track.embedding_clap_ensemble)
  + 0.50 × cosine(scene.embedding_hybrid, track.embedding_hybrid)
  + 0.05 × cosine(scene.embedding_hybrid, track.embedding_audio)  [audio_aux]
```

### Penalties

| Penalty | Trigger |
|---|---|
| `dialogue_conflict` | Track not dialogue-safe when scene requires it |
| `duration_conflict` | Track shorter than minimum scene duration |
| `forbidden_tag_conflict` | Track has a must-not tag |
| `style_redundancy_penalty` | Same style signature used too many times in episode |
| `same_track_consecutive_penalty` | Same track ranked first in consecutive scenes |
| `missing_data_penalty` | Track missing required metadata fields |

---

## Embedding vectors (both tables)

All embeddings are L2-normalised float32 vectors of dimension `VECTOR_DIM` (default 512). Cosine similarity is computed via pgvector `<=>` operator for candidate retrieval, and via `numpy.dot` for scoring (equivalent after L2 normalisation).

| Vector | Source | Description |
|---|---|---|
| `embedding_main` | CLAP text | `"{title}. {description}"` encoded by CLAP |
| `embedding_tags` | CLAP text | `tags_summary_en` encoded by CLAP |
| `embedding_clap_ensemble` | CLAP text | Weighted blend of 7 structured prompts |
| `embedding_hybrid` | Linear blend | `0.45 × main + 0.20 × tags + 0.35 × ensemble` |
| `embedding_audio` | CLAP audio | Averaged CLAP audio embeddings across N segments *(track_query only)* |

---

## CLAP model

Model: `laion/clap-htsat-unfused` (HuggingFace). Produces 512-dimensional embeddings from both text and audio. All text embeddings are produced by `ClapModel.get_text_features()`. Audio embeddings are produced by `ClapModel.get_audio_features()` on librosa-resampled waveform segments. All vectors are L2-normalised before storage.

---

## LLM pipeline (screenplay parser)

**Stage 1 (GPT-4o / Claude):** For each scene chunk, extract a structured JSON containing: `theme_title_pl`, `theme_title_en`, `description_en`, `theme_txt`, `segmentation_reason`, `scene_music_semantics` (emotional direction, narrative function, weight profile, dialogue_safe_required), `scene_music_targets` (energy, tempo, rhythm, intensity shape, sound character), `scene_tag_selection` (should_have, must_not), and `clap_prompt_ensemble` (7 structured music description prompts).

**Stage 2 (CLAP):** For each `SceneQueryRecord` returned by Stage 1, encode 4 embedding vectors and upsert to `scene_query`.

---

## Architecture overview

### Active Modules
- [app.py] — Hugging Face Spaces entry wrapper that compiles and runs the main script inline to preserve the Streamlit script-runner context.
- [ai_music_supervisor.py] — Consolidated multi-page frontend dashboard incorporating AI Music Supervisor tools and Music Library Importer utilities.
- [screenplay_parser_engine.py] — Screenplay reader, LLM segmenter, and CLAP text embedding generator.
- [scene_music_matcher_engine.py] — Scoring engine for retrieval, component grading, ranking, and Excel/PDF exporting.
- [importer_pipeline.py] — ETL indexing pipeline that scans files, processes audio attributes, generates LLM tag contracts, computes audio CLAP embeddings, and pushes records to the library database.
- [pipeline_config.py] — Single source of truth for weighting systems, category rules, contrary tags, and vector dimensions.
- [db_setup.py] — Idempotent database schema constructor and resilient PostgreSQL connection provider.
- [tags_v2.json] — Static tag taxonomy loaded at runtime.
- [packages.txt] — Hugging Face Spaces apt dependencies (`ffmpeg`, `libsndfile1`, `antiword`).
- [cloud_setup.py] - upload local music library to Cloudflare R2

---

## Module descriptions

### `app.py`
The default application entry point file required by Hugging Face Spaces. It reads and compiles `ai_music_supervisor.py` inline via `exec()`. This ensures that all top-level Streamlit initialization logic, CSS injection, and state handling run reliably on every user widget rerun.

---

### `ai_music_supervisor.py` *(v4.08)*
Unified Streamlit application wrapper containing the responsive GUI and layout rules. Page layout switches are managed through the sidebar navigation, mapping 8 consolidated pages:

| Page Group | Page | Description |
|---|---|---|
| **AI Music Supervisor** | **Dashboard** | Displays system overview, database status, loaded scene and track counts, and quick-jump navigators. |
| | **Screenplay Parser** | Document upload interface (.docx or .doc), LLM segmenting (Stage 1), CLAP text embedding generation (Stage 2), and database persistence, as well as a scene-splitting PDF exporter. |
| | **Scene - Music Matcher** | Interactive matching interface. Selects parsed episodes, shows input database validation checks, presents scoring weights selectors (embedding, semantic, targets, tag compliance), runs candidates ranking queries, and writes matches. |
| | **Scene - Music Scoring** | Detailed inspector for any scene-to-track match. Supports playing audio (streamed from local files or Cloudflare R2 bucket), lists complete metadata, and performs live recomputation checks side-by-side with stored database metrics. |
| | **Scene - Music Reports** | Downloader for generating and saving ranked results reports (PDF and Excel format, in full and compact styles) per episode. |
| **Music Library Importer** | **Run Importer** | Interactive portal to trigger and monitor the indexing pipeline (`importer_pipeline.py`) in a background worker thread with real-time log outputs and progress tracking. |
| | **Track Inspector** | Detailed track catalog search tool allowing prefix or contains queries on indexed files, tag category filtering, and direct audio playback alongside full metadata expansion. |
| | **Configuration** | Static and runtime parameter visualizer demonstrating active weights, categories rules, and negative tag restrictions. |

---

### `screenplay_parser_engine.py`
is used to automatically analyze movie scripts and generate semantic music queries for each scene. The result is saved in the `scene_query` table, ready to be matched with tracks from the `track_query` table.

Input data: A screenplay in .docx or .doc format (the .doc format requires LibreOffice or antiword to be installed). The screenplay’s name must include its sequential number. This unique number identifies the screenplay in the database. Using the same number will overwrite the previous screenplay with that number.

Each scene must be preceded by a keyword defining its start. The script uses the keyword SCENE to extract individual scenes. This is defined by a variable in the script:

_DEFAULT_SCENE_PATTERN = r"(?mi)^\s*SCENA\s+(\d+)\s*[\.\:\ -]?\s*(.*)$"

It is possible to define your own variable in the .env file that will split the script into scenes using the English word **SCENE**:

SCENE_HEADER_PATTERN=r“(?mi)^\sSCENE\s+(\d+)\s*[\.\:\ -]?\s*(.*)$”`


Basic functions:
1. **Document Loading:** Reads `.docx` screenplays (via python-docx) and legacy `.doc` files (using command-line antiword and LibreOffice headless converters).
2. **Regex Splitting:** Splits script text into scenes by matching headers.
3. **LLM Segmentation (Stage 1):** Segmenting scenes into thematic parts and building semantic definitions using GPT-4o / Claude.
4. **CLAP Encoding (Stage 2):** Generates 512-dimensional CLAP text vectors (`embedding_main`, `embedding_tags`, `embedding_clap_ensemble`, and normalized `embedding_hybrid`) and upserts them to `scene_query`.
5. **PDF Exporting:** Compiles parsed screenplay summaries to a clean PDF report via ReportLab.

---

### `scene_music_matcher_engine.py`
Pure business-logic script with zero Streamlit dependencies. Handles:
1. **Candidate Retrieval:** Queries database candidates via pgvector hybrid cosine similarity (ANN queries).
2. **Scoring:** Performs component evaluations (similarities, semantics alignment, target parameters matching, tags compliance) and deducts penalty scores.
3. **Persistence:** Orders ranked candidates and writes results to the `scene_music_matches_v6` table.
4. **Reports Generation:** Generates multi-page styled ReportLab PDFs (full/compact layout) and OpenPyXL worksheets (full/compact spreadsheet layouts).

---

### `importer_pipeline.py`
Pure business-logic ETL script with zero Streamlit dependencies. Orchestrates:
1. **Audio Preprocessing:** Scans local or Cloudflare R2 directories, resamples audio to standard 48kHz, segments tracks, and extracts physical properties (RMS, zero-crossing rate).
2. **CLAP Audio Encoding:** Generates 512-dimensional embedding vectors from audio waveforms.
3. **Semantic Contract Generation:** Queries LLM to establish description text, semantic titles, category tags, prompt ensembles, and dialogue safety ratings.
4. **ETL Push:** Embeds text vectors and audio characteristics, assembling the hybrid vector and upserting the results idempotently to the `track_query` registry.

---

### `pipeline_config.py`
Shared parameters module holding:
1. **Weight arrays:** `TAG_PROMPT_WEIGHTS`, `ENSEMBLE_PROMPT_WEIGHTS`, and `TEXT_HYBRID_WEIGHTS`.
2. **Category Selection rules:** thresholds, margin limits, and counts per tag category.
3. **Mutual exclusions:** `CONTRARY_TAGS` constraints.
4. **Static validation:** `validate_weights()` check executes at import.

---

### `cloud_setup.py`
 **CLOUD MODE** Upload local music library to Cloudflare R2

---

### `db_setup.py`
Resilient database client and schema setup:
1. **DatabaseConfig:** Authoritative database settings block offering exponential retry handles to mitigate Neon cold starts.
2. **Migrations:** Creates `pgvector`, `scene_query`, `track_query`, and `scene_music_matches_v6` tables idempotently.
3. **Optimizations:** Mounts indexes, updated_at triggers, and QA views.

---

## Environment variables (`.env`)

### PostgreSQL — Neon Cloud

| Variable | Description | Example |
|---|---|---|
| `PGHOST` | Neon database host | `ep-xxx.eu-central-1.aws.neon.tech` |
| `PGDATABASE` | Database name | `music_supervisor` |
| `PGUSER` | Database user | `music_supervisor_owner` |
| `PGPASSWORD` | Database password | *(your password)* |
| `PGSSLMODE` | SSL mode | `require` |
| `PGCHANNELBINDING` | Channel binding | `disable` |

### OpenAI

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | GPT-4o API key for LLM stages |

### Music library

| Variable | Default | Description |
|---|---|---|
| `MUSIC_DIR` | `clanMusic` | Local music directory path |
| `TAGS_FILE` | `tags_v2.json` | Path to tag taxonomy file |
| `SKIP_EXISTING` | `True` | Skip already-indexed files during indexing |
| `WORKERS` | `4` | Number of concurrent CPU workers |

### CLAP model

| Variable | Default | Description |
|---|---|---|
| `CLAP_MODEL_NAME` | `laion/clap-htsat-unfused` | HuggingFace model ID |
| `VECTOR_DIM` | `512` | Embedding vector dimension |

### Cloud storage (optional — Cloudflare R2)

| Variable | Description |
|---|---|
| `CLOUD_MUSIC` | `true` to enable R2 mode, `false` for local |
| `CF_ACCOUNT_ID` | Cloudflare account ID |
| `CF_TOKEN_VALUE` | Cloudflare API token |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_ENDPOINT_URL` | R2 endpoint URL |
| `BUCKET_NAME` | R2 bucket name |

### Screenplay parser

| Variable | Description |
|---|---|
| `SCENE_HEADER_PATTERN` | r“(?mi)^\sSCENE\s+(\d+)\s*[\.\:\ -]?\s*(.*)$”` |

The English  word **SCENE** splits the script into scenes.

---

## Running order

### Setup & Startup

1. **Configure Environment:** Create a `.env` file pointing to your OpenAI keys and PostgreSQL instance (equipped with the `pgvector` extension).
2. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   ```  
4. **Database Initialization:** Run the database schema initializer setup:
   ```bash
   python db_setup.py
   ```
5. if **CLOUD MODE** Upload music library to Cloudflare R2
      ```bash
   python cloud_setup.py
   ```
6. if **LOCAL MODE** Ensure MUSIC_DIR points to your music folder in .env
7. **Verify packages:** Ensure that system dependencies (`ffmpeg`, `libsndfile1`, and `antiword`) are installed.
8. **Run Streamlit Frontend:**
   ```bash
   streamlit run ai_music_supervisor.py
   ```
   If deploying to Hugging Face Spaces, deploy all files; HF will automatically execute `app.py` bootloader.

### Import music library

After first run ai_music_supervisor.py navigate to **Run Importer** page and scan your music directory (`MUSIC_DIR` or Cloudflare R2 bucket) to index and populate the `track_query` table.


### Matching Workflow

Follow the sequential workflow inside the GUI pages:
1. **Screenplay Parser:** Upload screenplay document (.docx/.doc) -> Parse & Embed -> Results are stored in `scene_query`.
2. **Scene - Music Matcher:** Select episode, configure weights, and hit "Run Matcher" -> Persists matches into `scene_music_matches_v6`.
3. **Scene - Music Scoring:** Play matches, inspect score components, and live-recalculate options.
4. **Scene - Music Reports:** Download PDF/XLSX results summaries.



