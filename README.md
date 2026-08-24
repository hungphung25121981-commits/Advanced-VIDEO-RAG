# video-visual-rag

**A Video-Native Multimodal Hybrid RAG System — 100% free/open-source, single all-in-one model, driven end-to-end by a CLI (`main.py`).**

No Whisper. No audio transcript. This system indexes and reasons over videos using **only what's visible on screen**: OCR'd text, visual scene descriptions, and dynamic video metadata. It is purpose-built for screen recordings, tutorials, news broadcasts, dashboards, and slide-driven content where the audio track is unreliable, unavailable, or simply not the point.

Every phase is exposed as an independent CLI subcommand, so you can re-run just the piece you're iterating on — e.g. tweak `generator.py` and re-run `query`/`shell` repeatedly without re-extracting keyframes or rebuilding the index each time.

---

## Table of Contents

- [Architecture](#architecture)
- [Why a Single Model?](#why-a-single-model)
- [Repository Structure](#repository-structure)
- [Input Data Formats](#input-data-formats)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [Hybrid Search: FAISS + BM25 + RRF](#hybrid-search-faiss--bm25--rrf)
- [Multi-Hop Reasoning](#multi-hop-reasoning)
- [Configuration Reference](#configuration-reference)
- [Memory Management Notes](#memory-management-notes)
- [Limitations](#limitations)

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              RAW VIDEO (.mp4)                │
                         └───────────────────────┬───────────────────────┘
                                                  │
   PHASE 1 — EXTRACTION                          ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ OpenCV SSIM filter → dedup near-identical frames                   │
   │ PySceneDetect → hard-cut scene boundaries                          │
   │ mapper.py → group into 15-30s segments → keyframe_map.csv          │
   └───────────────────────────────────────┬───────────────────────────┘
                                            ▼
   PHASE 2 — VISUAL UNDERSTANDING (single model: Qwen2.5-VL-8B, 4-bit NF4)
   ┌───────────────────────────────────────────────────────────────────┐
   │ PaddleOCR → on-screen text / tables / banners per segment           │
   │ Qwen2.5-VL-8B → dense visual_caption paragraph per segment          │
   │   (prompted with dynamic video_info.json context)                   │
   │ metadata_builder.py → metadata.parquet                              │
   │   full_text_for_embedding = "TITLE: ... | VISUAL: ... | OCR: ..."   │
   └───────────────────────────────────────┬───────────────────────────┘
                                            ▼
   PHASE 3 — HYBRID INDEXING
   ┌───────────────────────────────────────────────────────────────────┐
   │ bge-m3 → dense embeddings (float32)                                 │
   │ faiss_indexer.py → L2-normalize → IndexFlatIP (cosine sim)          │
   │ bm25_indexer.py → rank_bm25 sparse lexical index                    │
   └───────────────────────────────────────┬───────────────────────────┘
                                            ▼
   PHASE 4 — ORCHESTRATION (same single model: Qwen2.5-VL-8B, 4-bit NF4)
   ┌───────────────────────────────────────────────────────────────────┐
   │ multihop_evaluator.py → decompose question → sub-questions          │
   │ hybrid_search.py → FAISS + BM25 per sub-question → RRF fusion       │
   │ multihop_evaluator.py → LLM-judge relevance per segment              │
   │ generator.py → grounded answer + [Source N] → jump_link citations   │
   └───────────────────────────────────────────────────────────────────┘
```

## Why a Single Model?

On a single consumer/free-tier GPU (e.g. a 16GB T4), loading two separate large models (a captioner + a separate LLM) means paying the load/unload VRAM tax twice per run. **Qwen2.5-VL-8B-Instruct**, quantized to **4-bit NF4** via `bitsandbytes`, fits comfortably in ~6-7GB VRAM and is capable of both:

1. **Vision-language captioning** (Phase 2) — it natively accepts multiple images + text.
2. **Pure-text reasoning** (Phase 4) — sub-query decomposition, relevance judging, and answer synthesis are just text-only calls to the *same* loaded model/tokenizer.

`src/phase2_captioning/vlm_captioner.py` exposes a process-wide singleton (`get_qwen_engine()`) so Phase 4 modules reuse the exact same in-memory weights — the model is loaded **once per Kaggle session**.

---

## Repository Structure

```
video-visual-rag/
├── README.md
├── requirements.txt
├── main.py                        # CLI entrypoint: extract / caption / index / query / shell / pipeline
├── config/
│   └── settings.yaml              # all hyperparameters live here
├── main.py                        # CLI entrypoint (extract / caption / index / query / trake / shell / pipeline)
└── src/
    ├── __init__.py
    ├── utils_common.py            # config loader, logger, GPU cleanup, dynamic-metadata helpers
    ├── phase1_extraction/
    │   ├── ssim_filter.py         # OpenCV SSIM near-duplicate frame filter
    │   ├── scene_detect.py        # PySceneDetect wrapper
    │   └── mapper.py              # keyframe → segment grouping, keyframe_map.csv writer
    ├── phase2_captioning/
    │   ├── ocr_engine.py          # PaddleOCR wrapper + segment-level OCR dedup
    │   ├── vlm_captioner.py       # Qwen2.5-VL-8B 4-bit NF4 engine (shared by Phase 2 & 4)
    │   ├── visual_embedder.py     # SigLIP visual-text similarity (fast frame_selector.py path)
    │   └── metadata_builder.py    # video_info.json + OCR + caption → metadata.parquet
    ├── phase3_indexing/
    │   ├── embedder.py            # bge-m3 dense embeddings
    │   ├── faiss_indexer.py       # IndexFlatIP + L2 normalize + segment_id mapping
    │   └── bm25_indexer.py        # rank_bm25 index + pickle serialization
    └── search_engine/
        ├── hybrid_search.py       # FAISS + BM25 + Reciprocal Rank Fusion
        ├── multihop_evaluator.py  # sub-query decomposition + relevance judging (Qwen2.5-VL, text-only)
        ├── reranker.py            # VLM re-scoring of fused candidates (--rerank)
        ├── frame_selector.py      # segment -> exact frame_id (middle / siglip / vlm modes)
        ├── retrieval_cli.py       # search -> rerank -> frame-resolve plumbing for `query --s/--q`
        ├── csv_export.py          # ranked-results CSV read/write (--out-csv, --qa)
        ├── visual_qa.py           # direct VQA from a concrete frame image (--q/--qa)
        ├── trake.py               # multi-moment temporal localization within one video
        └── generator.py           # grounded answer synthesis + timestamp jump_link citations
```

---

## Input Data Formats

### 1. Raw video files
`data/raw_videos/<video_id>.mp4` (or `.mkv`) — any OpenCV-readable container/codec.

### 2. `video_info.json` (one per video, fully dynamic schema)
Place at `data/video_info/<video_id>.json`. **Every key is optional.** The pipeline never assumes a fixed schema — every access goes through `.get(key, default)` via `src/utils_common.py::safe_load_video_info` and `format_dynamic_metadata_block`. Example:

```json
{
  "title": "How to Configure Kubernetes Ingress",
  "description": "A walkthrough of NGINX ingress controllers on GKE.",
  "keywords": ["kubernetes", "ingress", "nginx", "gke"],
  "watch_url": "https://youtube.com/watch?v=abc123",
  "publish_date": "2025-03-14",
  "channel": "CloudOps Weekly"
}
```

If `watch_url` is missing, `jump_link` falls back to `"{video_id}#t={start_time}s"` (see `metadata_builder.py::_build_jump_link`). Missing `title` defaults to `"Unknown Title"`.

### 3. `data/keyframes/*.jpg`
Filtered frames written by Phase 1 (JPEG quality 90–95%, configurable via `phase1.jpeg_quality`).

### 4. `keyframe_map.csv`
| column | type | description |
|---|---|---|
| `frame_id` | str | e.g. `video001_f000042` |
| `video_id` | str | matches `video_info.json` filename stem |
| `timestamp_sec` | float | frame timestamp within the video |
| `segment_id` | int | per-video-local segment id (made globally contiguous in Phase 2) |

### 5. `metadata.parquet`
| column | type | description |
|---|---|---|
| `segment_id` | int | **globally contiguous 0..N-1** — required for FAISS row alignment |
| `video_id` | str | |
| `video_title` | str | defaults to `"Unknown Title"` |
| `start_time` / `end_time` | float | segment boundaries in seconds |
| `jump_link` | str | clickable timestamp URL, dynamically built |
| `keyframes_list` | list[str] | frame_ids belonging to this segment |
| `ocr_screen_text` | str | de-duplicated PaddleOCR output |
| `visual_caption` | str | Qwen2.5-VL-8B dense caption (≤120 words) |
| `full_text_for_embedding` | str | `"TITLE: {title} \| VISUAL: {caption} \| OCR: {ocr}"` |

### 6. Phase 3 index artifacts
- `data/index/video_faiss.index` — FAISS `IndexFlatIP`, IDs = `segment_id`
- `data/index/bm25_corpus.pkl` — pickled `BM25Corpus` (tokenized corpus + `rank_bm25` index)

---

## Installation

```bash
git clone <this-repo-url> video-visual-rag
cd video-visual-rag

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt --no-deps -q
pip install -e .                 # registers `src` as an importable package (also fixes VSCode import warnings)
```

Requires a CUDA GPU with ~8GB+ free VRAM for Qwen2.5-VL-8B in 4-bit NF4 (a 16GB card gives comfortable headroom for the embedder/index phases too). If `faiss-gpu` doesn't detect your GPU (`faiss.get_num_gpus() == 0`), reinstall via conda:

```bash
pip uninstall -y faiss-gpu faiss-cpu
conda install -y -c pytorch -c nvidia faiss-gpu=1.5.0
```

Place your inputs before running anything:

```
data/raw_videos/<video_id>.mp4
data/video_info/<video_id>.json
```

(`data/` is git-ignored — see [Input Data Formats](#input-data-formats) below for the exact schemas expected.)

### Fixing "Import could not be resolved" in VS Code

This warning is a **static-analysis (Pylance) issue only** — it does not affect running the code, and it is unrelated to `git clone`. It appears when:

1. **Dependencies aren't installed yet** in the Python interpreter VS Code is using (`torch`, `transformers`, `faiss`, `cv2`, `paddleocr`, etc. won't resolve until you `pip install -r requirements.txt` inside an active venv).
2. **VS Code hasn't selected that venv as the interpreter.** Run `Python: Select Interpreter` from the Command Palette (`Ctrl+Shift+P`) and pick `.venv`.
3. **Internal imports** (`from src.utils_common import ...`) need the repo root on `PYTHONPATH`. This repo ships `.vscode/settings.json` and `pyproject.toml` to handle this automatically — just make sure you open the **repo root folder** in VS Code (not a subfolder), and reload the window after installing dependencies.

If warnings persist after all three steps: `Ctrl+Shift+P` → `Python: Restart Language Server`.

---

## CLI Usage

Every phase is a subcommand of `main.py`. Defaults for every flag come from `config/settings.yaml`; pass a flag explicitly to override it for a single run without touching the YAML.

```bash
# Run everything end-to-end (Phase 1 -> 2 -> 3)
python main.py pipeline --videos-dir data/raw_videos --video-info-dir data/video_info

# ...or run each phase independently, re-running only what you changed:

# Phase 1: keyframe extraction (SSIM dedup + PySceneDetect + segment grouping)
python main.py extract --videos-dir data/raw_videos --keyframes-dir data/keyframes

# Phase 2: OCR + Qwen2.5-VL captioning -> metadata.parquet
python main.py caption --keyframe-map data/keyframe_map.csv --video-info-dir data/video_info

# Phase 3: build the hybrid FAISS + BM25 index
python main.py index --metadata data/metadata.parquet

# Phase 4: ask a single one-shot question
python main.py query "What error message appeared in the terminal and what command fixed it?"

# Phase 4: interactive REPL — loads the model + index once, ask many questions
python main.py shell
```

### Phase 4, CLI search / QA / TRAKE modes

Four extra modes live under `query` (plus a new `trake` subcommand) for driving the pipeline like a KIS / Q&A / TRAKE-style retrieval competition submission:

```bash
# --s : search only. Ranks matching frames, prints a preview, writes the FULL
#       ranked list to CSV (columns: rank, frame_id, video_id, segment_id,
#       timestamp_sec, score, video_title, jump_link, answer).
python main.py query --s "nguoi mac ao xanh dang chay" --out-csv out.csv --top-n 100 --rerank

# --q : search + answer. Runs the same pipeline as --s internally, then takes
#       the rank-1 result and asks the VLM the question directly against that
#       frame's IMAGE (not just its caption text) -- needed for things like
#       counting ("co bao nhieu nguoi dang an?" -> "2").
python main.py query --q "co bao nhieu nguoi dang an?" --out-csv out.csv --rerank

# --qa : answer-from-existing-CSV ONLY. Reads the rank-1 row of an existing
#        --out-csv (written earlier by --s/--q) and answers from THAT frame --
#        it NEVER re-runs search. If --out-csv doesn't exist yet, it stops
#        with an error instead of silently falling back to searching.
python main.py query --qa "co bao nhieu nguoi dang an?" --out-csv out.csv

# trake : locate N ORDERED sub-moments of one event inside a SINGLE video --
#         e.g. a high jump's 4 stages. Two-stage retrieve-and-align:
#         (A) coarse-localize WHICH video (skip with --video-id if known),
#         (B) per-stage search restricted to that video + a VLM frame-pick,
#         with a monotonic-time constraint so stages come out in order.
#         Output CSV columns: stage_index, stage_query, video_id, segment_id,
#         frame_id, timestamp_sec, score, ok.
python main.py trake \
  --stages "giam nhay" "bay qua xa" "tiep dat" "dung day" \
  --video-id L10_V010 \
  --out-csv trake.csv
```

Flags for `query --s/--q`: `--top-n` (row count, default from `phase4.cli_top_n_default`), `--rerank` (one extra VLM re-scoring pass over the fused candidates), `--video-id` (restrict to one video), `--select-frame` (spend one extra VLM call per row to pick the exact best-matching `frame_id` inside each segment, instead of the default fast SigLIP-embedding pick -- `--q`/`--qa` always do this for their single rank-1 row regardless of this flag).

### Frame selection: SigLIP (fast) vs. VLM (precise)

Every result row needs one concrete `frame_id`, not just a segment. `src/search_engine/frame_selector.py` resolves this three ways, in increasing cost/precision order:

| mode | cost | used by |
|---|---|---|
| `middle` | free, no model call | last-resort fallback only |
| `siglip` **(default)** | one cheap image-text embedding pass, no generation | bulk `--s`/`--q` rows without `--select-frame` |
| `vlm` | one Qwen2.5-VL generation call per row | `--select-frame`, and always for `--q`/`--qa`/`trake`'s chosen frame |

SigLIP (`google/siglip-base-patch16-224` by default, configurable via `phase2.siglip_model_id`) replaces the CLIP-ViT-B/32 slot from the original design -- see `src/phase2_captioning/visual_embedder.py` for the full port notes (fixed-length text padding, explicit L2-normalization, tensor-dimension validation). If SigLIP fails to load (e.g. no cached weights, no internet), frame selection automatically falls back to the `middle` heuristic rather than failing the run; set `phase2.frame_select_fast_mode: "middle"` to skip SigLIP entirely.

`python main.py <command> --help` lists every flag for that command. Run `--help` on the top-level parser too:

```bash
python main.py --help
```

Every printed answer includes a **Sources** section listing `[Source N] title (start-end s): jump_link`, so you can click straight to the moment in the video that grounds each claim.

Because `query`/`shell` are separate from `extract`/`caption`/`index`, iterating on **answer quality** (prompt tweaks in `src/search_engine/generator.py` or `multihop_evaluator.py`) never requires re-running the expensive OCR/captioning/embedding phases — just re-run `python main.py query "..."` against the already-built `metadata.parquet` / `video_faiss.index` / `bm25_corpus.pkl`.

---

## Hybrid Search: FAISS + BM25 + RRF

- **Dense**: query embedded with `bge-m3`, searched against `IndexFlatIP` (vectors L2-normalized at both index-build and query time via `faiss.normalize_L2`, making inner product equivalent to cosine similarity).
- **Sparse**: `rank_bm25.BM25Okapi` over a simple lowercase-alphanumeric tokenizer — catches exact matches (file names, error codes, tickers) that dense embeddings can blur.
- **Fusion**: [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) combines both ranked lists without needing to reconcile incompatible score scales:

  ```
  RRF(segment) = Σ  1 / (rrf_k + rank_r(segment))   for each ranker r
  ```

  `rrf_k` (default `60`) is configurable in `phase4.rrf_k`. Fused scores are normalized to `[0, 1]` and filtered by `phase4.min_relevance_score` before being passed on.

---

## Multi-Hop Reasoning

For complex questions (e.g. *"Compare the CPU chart shown in the first demo to the memory chart shown later, and tell me which metric spiked first"*), `multihop_evaluator.py`:

1. Prompts Qwen2.5-VL-8B (text-only) to decompose the question into up to `phase4.max_subqueries` atomic sub-questions, parsed as strict JSON.
2. Runs `hybrid_search` independently per sub-question.
3. Asks the model to **judge** (JSON `[1,0,1,...]`) which retrieved segments are actually relevant to each sub-question — filtering out RRF-fused-but-irrelevant hits before generation.
4. `generator.py` de-duplicates the relevant segment union across all sub-questions, sorts by fused score, and synthesizes one grounded answer citing `[Source N]` tags mapped to real `jump_link`s.

Both the decomposition and judgment prompts are defensive: if the model's output isn't parseable JSON, the code falls back gracefully (single-question mode / keep-all-segments) rather than crashing the pipeline.

---

## Configuration Reference

All knobs live in `config/settings.yaml`, grouped by phase: `paths`, `phase1` (SSIM/scene thresholds, segment window), `phase2` (Qwen2.5-VL + OCR settings), `phase3` (embedding dim, FAISS type, BM25 k1/b), `phase4` (top_k, RRF k, multi-hop cap), and `runtime` (device_map, memory-cleanup cadence, seed). Load it anywhere via:

```python
from src.utils_common import load_config
cfg = load_config()
```

---

## Memory Management Notes

- `src/utils_common.py::free_gpu_memory()` runs `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`.
- `maybe_free_memory(step_index, cfg)` is called inside the Phase 2 per-segment loop on the cadence set by `runtime.empty_cache_every_n_segments` / `runtime.gc_collect_every_n_segments`.
- `QwenVLEngine.unload()` and `unload_embedder()` explicitly drop model references before switching between the captioning-heavy Phase 2 and the embedding-heavy Phase 3, keeping peak VRAM well under the T4's 16GB.
- FAISS indices are always **saved as CPU indices** (`faiss.index_gpu_to_cpu`) for portability across Kaggle sessions/GPUs, and re-hydrated to GPU on load if available.

---

## Limitations

- **No audio understanding** — content that is only spoken (not shown on screen) is invisible to this system by design.
- OCR quality depends on video resolution/compression; very small on-screen text at low bitrate may be missed.
- `IndexFlatIP` is exact (no ANN approximation) — fine for the scale of a Kaggle-hosted corpus, but will need `IndexIVFFlat`/`IndexHNSWFlat` for corpora beyond a few hundred thousand segments.
- Qwen2.5-VL-8B in 4-bit NF4 trades some captioning/reasoning fidelity for VRAM headroom; if you have more GPU budget, bump `phase2.vlm_quantization` handling in `vlm_captioner.py` to 8-bit or full precision.
