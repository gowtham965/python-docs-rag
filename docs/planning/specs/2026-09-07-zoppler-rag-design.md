# Zoppler Systems RAG — Design Spec

## Purpose

Build a second RAG portfolio piece, `zoppler-rag`, over the public marketing
site at zopplersystems.com (an Indian defence-tech company: radar/EW
systems, AI-driven detection, SWaP-C hardware, simulation/training
platforms, operator displays). Motivation: a demo built specifically around
Zoppler's own content, showable to Zoppler in an interview or application —
"here's a Q&A assistant over your own site, built with a hand-rolled hybrid
retrieval pipeline."

This is a new, standalone repository (`~/Projects/zoppler-rag`), not a mode
of this repo. It reuses this repo's retrieval/generation/eval core as a
dependency rather than duplicating it, which requires one small,
backward-compatible generalization here first (Component 1).

## Context: the corpus

zopplersystems.com is a single-page React SPA (Vite build, client-rendered
— the raw HTML ships an empty `<div id="root">`, so ingestion must render
the page, not fetch it as static HTML). No sitemap, no sub-routes. Content
is organized into ~8 sections plus an FAQ, roughly 2-3k words total:

- Hero / company intro
- AI-Driven Solutions & Custom Tools
- SWaP-C Advanced Engineering (hardware)
- Radar Software Core Products
- Simulation & Training Platforms
- Displays & Operator Systems
- FAQ (a handful of Q/A pairs)

This is a much smaller and thinner corpus than the CPython docs. The design
below (especially eval) is scaled to that — this is a *focused* demo, not an
attempt to make a thin site look like a large corpus.

## Constraints

- Local-only for now — no deployment target in scope. Running the FastAPI
  backend + Vite dev server locally is sufficient.
- No LangChain/LlamaIndex — same hand-built philosophy as the parent
  project, since being able to defend every design decision is the point.
- One-time ingestion — the site is a static marketing page; no scheduled
  re-scrape, no incremental updates.
- Reuse, don't duplicate: retrieval, fusion, reranking, generation, and eval
  logic come from `pyrag` as a dependency. `zoppler-rag` only implements
  what's genuinely new (ingestion for a JS-rendered single page, and its own
  config/prompt/eval data).

## Architecture

```
                    python-docs-rag (this repo)
                    ┌─────────────────────────────┐
                    │ pyrag (pip-installable pkg)  │
                    │  - retrieval (BM25, vector,  │
                    │    RRF fusion, reranker)     │
                    │  - generation (LLM clients,  │
                    │    pipeline, configurable    │
                    │    prompt)                   │
                    │  - eval (metrics, judge,     │
                    │    run_eval)                 │
                    └──────────────┬───────────────┘
                                   │ pip install
                                   │ git+https://github.com/gowtham965/python-docs-rag.git
                                   ▼
                    zoppler-rag (new repo)
                    ┌─────────────────────────────┐
                    │ ingestion/                  │
                    │  - scrape.py (Playwright)   │
                    │  - chunker.py (sections,    │
                    │    FAQ per-Q/A)             │
                    │  - build_index.py           │
                    │    (calls pyrag directly)   │
                    │ config.py (Zoppler prompt,  │
                    │   out-of-scope text)        │
                    │ server.py (thin, mirrors    │
                    │   pyrag.server wiring)      │
                    │ frontend/ (re-skinned copy  │
                    │   of the existing chat UI)  │
                    │ eval_data/zoppler_qa.json   │
                    └─────────────────────────────┘
```

## Components

### 1. Generalize `pyrag`'s hardcoded prompt strings (this repo)

Two strings are currently hardcoded to the Python-docs persona:

- `SYSTEM_INSTRUCTIONS` in `src/pyrag/generation/prompt.py`
- `OUT_OF_SCOPE_MESSAGE` in `src/pyrag/generation/pipeline.py`

Change: add two fields to `Config` (`src/pyrag/config.py`), defaulting to
today's exact text so this repo's behavior is unchanged:

```python
system_instructions: str = (
    "You are a helpful assistant answering questions about the Python "
    "standard library using only the provided documentation excerpts. "
    "Cite the section title for every claim you make, using the format "
    "[Section: <title>]. If the excerpts do not contain the answer, say "
    "\"I don't know based on the provided documentation.\" Do not use "
    "outside knowledge."
)
out_of_scope_message: str = "I couldn't find relevant information in the Python docs for this."
```

`build_prompt(question, chunks, system_instructions)` takes the instructions
as a parameter instead of importing the module-level constant.
`RagPipeline.__init__` takes `out_of_scope_message` (default preserves
current behavior) and uses it instead of the module-level constant.
`wiring.build_pipeline` passes both through from `config`. No other file
changes. Existing tests for `build_prompt`/`RagPipeline` are updated to pass
the default text explicitly (behavior-preserving), plus one new test per
function confirming a custom value is honored.

### 2. New repo scaffold: `zoppler-rag`

`~/Projects/zoppler-rag`, initialized with git, `pyproject.toml` depending
on `pyrag @ git+https://github.com/gowtham965/python-docs-rag.git` plus its
own direct deps (`playwright`). Directory layout mirrors this repo's
conventions (`src/zoppler_rag/`, `data/{raw,processed,chroma}`, `tests/`,
`docs/planning/`) so the two portfolio repos read as one consistent body of
work.

### 3. Ingestion — render + extract (`ingestion/scrape.py`)

Not a crawler (there is nothing to crawl) — a one-shot render:

- Playwright launches headless Chromium, navigates to
  `https://zopplersystems.com`, waits for network idle.
- Walks the rendered DOM for heading elements (`h1`-`h3`) and the text
  content between one heading and the next.
- Writes `data/raw/sections.json`: a list of
  `{"heading": str, "body": str}` in document order. This file is checked
  into the repo — re-running the scraper is a manual, occasional action, not
  part of the build.

### 4. Chunking (`ingestion/chunker.py`)

- Every section becomes one `Chunk` (word-windowed via the same
  400-word/50-overlap logic as `pyrag.ingestion.chunker._split_long_body`
  if a section runs long — none are expected to, given the corpus size, but
  the guard costs nothing).
- The FAQ section is detected by heading match and chunked differently: one
  `Chunk` per Q/A pair, each `Chunk.text` being `"{question}\n{answer}"`.
  Rationale: FAQ entries are the closest thing this corpus has to real user
  queries, and splitting them out gives retrieval its strongest signal
  against a thin corpus.
- `Chunk.source_file` is repurposed as a per-section slug (there's no
  filesystem here) — e.g. `"zopplersystems.com#ai-driven-solutions"`, and
  each FAQ chunk gets its own slug (`"zopplersystems.com#faq-2"`) rather
  than sharing one `"faq"` slug, so retrieval eval (hit@5/MRR, which key off
  `source_file`) can still distinguish between individual FAQ answers.
- `Chunk.section_title` is the heading text (used for citations, same as
  today).

### 5. Index build (`ingestion/build_index.py`)

Thin script calling `pyrag.retrieval.embedder.Embedder`,
`pyrag.retrieval.vector_store.VectorStore`, and the BM25 store construction
directly — same shape as this repo's `build_index.py`, just pointed at
`zoppler-rag`'s own `data/chroma` and `data/processed/chunks.json`.

### 6. Config, server, frontend

- `config.py`: a `Config` subclass/factory that sets `system_instructions`
  and `out_of_scope_message` to Zoppler-specific copy (persona: "answering
  questions about Zoppler Systems' products and capabilities using only the
  provided website excerpts"; out-of-scope: "I couldn't find relevant
  information about Zoppler Systems for this question."). All other config
  (embedding model, reranker, relevance threshold, LLM provider) reuses
  `pyrag`'s defaults.
- `server.py`: calls `pyrag.wiring.build_pipeline` (or a local equivalent
  passing the Zoppler config) — no new endpoint shape, same `POST /chat`
  SSE contract as `pyrag.server`.
- `frontend/`: a copy of this repo's `frontend/` with title, placeholder
  question, and example prompts re-skinned for Zoppler. No new UI
  functionality.

### 7. Eval (`eval_data/zoppler_qa.json`)

~15-20 hand-written questions grounded in the real sections/FAQ, in the
same `{"question", "is_in_scope", "expected_source_file"}` shape
`run_eval.py` already expects, run through `pyrag.eval.run_eval` unchanged.
Produces the same shape of baseline report (hit@5, MRR, faithfulness,
relevance) as this repo's `docs/eval-results/baseline.json`. The README
states the corpus-size caveat plainly (a 2-3k word single page is a much
smaller retrieval problem than the CPython docs) rather than implying
equivalence.

## Error Handling

No new error-handling logic — this reuses `pyrag`'s existing behavior
end-to-end (confidence gate, LLM client retry/backoff, SSE error events).
The only new failure mode is ingestion-time: if Playwright can't render the
page (network failure, site restructure breaking the heading-walk), `scrape.py`
fails loudly and does not overwrite the last-known-good `sections.json`.

## Testing

- `tests/ingestion/test_chunker.py`: unit tests for section chunking and
  FAQ per-Q/A chunking against a small fixture DOM/section list (no live
  Playwright run in tests).
- This repo (`python-docs-rag`): update existing `test_prompt.py` /
  `test_pipeline.py` to pass explicit `system_instructions` /
  `out_of_scope_message`, add one test each for custom values.
- End-to-end verification is manual (via the `run` skill): golden-path
  question answered with citations, an out-of-scope question correctly
  refused, FAQ-derived questions retrieved correctly.

## Out of scope

- Deployment (Render/HF Spaces/etc.) — local-only for now.
- Any content beyond the single public page (no LinkedIn, deck, or PDF
  ingestion).
- Re-scraping automation / staleness detection.
- Changes to `pyrag`'s retrieval, fusion, reranking, or eval logic beyond
  the Component 1 prompt-string generalization.
