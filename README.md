---
title: Python Docs RAG API
emoji: 🐍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Python Docs RAG

A retrieval-augmented Q&A system over the Python standard library docs, with a
hand-built hybrid retrieval pipeline (BM25 + vector search, fused via
Reciprocal Rank Fusion, cross-encoder reranked) and a custom evaluation
harness (retrieval metrics + LLM-as-judge).

## Architecture

Two pipelines, both hand-built (no LangChain/LlamaIndex) so every stage is
easy to open up and explain.

**Ingestion — offline, run once (or whenever the docs change):**

```
CPython Doc/*.rst
  → heading-aware chunker (splits on RST headings first, then windows
    any section over 400 words with 50-word overlap)
  → BAAI/bge-small-en-v1.5 embeddings (local, CPU)
  → Chroma (persisted vector index)  +  chunks.json (canonical chunk store)
```

**Query — one call per user question:**

```
question
  ├─ vector search (Chroma, cosine similarity, top 20)  ─┐
  └─ BM25 keyword search (rank_bm25, top 20)           ─┴→ Reciprocal Rank
                                                            Fusion (k=60)
                                                              ↓
                                                   cross-encoder rerank
                                                   (ms-marco-MiniLM-L-6-v2,
                                                    top 20 → top 5)
                                                              ↓
                                                  confidence gate: is top
                                                  reranked score ≥ 0.3?
                                              ┌───────────────┴───────────────┐
                                       no → "I don't know"        yes → cited answer
                                       (no LLM call)               (Groq / Gemini / OpenAI,
                                                                     swappable via LLM_PROVIDER)
```

Key design choices:
- **Hybrid retrieval over vector-only** — BM25 catches exact-term/API-name
  matches (`os.path.join`) that embedding similarity can blur.
- **Reciprocal Rank Fusion over a weighted score blend** — BM25 and cosine
  scores live on incomparable scales; RRF uses rank position only, so it
  sidesteps that entirely.
- **Cross-encoder rerank after fusion** — fusion is cheap but only a weak
  first-pass signal; a cross-encoder scores query+chunk jointly for far
  better precision, applied only to the narrowed top 20 since it's too
  slow to run over the whole corpus.
- **Confidence gate before generation** — RAG's biggest failure mode is
  confidently answering from irrelevant context, so a low top-rerank-score
  short-circuits straight to "I don't know" without ever calling the LLM.

## Baseline eval results

See `docs/eval-results/baseline.json`.

**Full 50/50 baseline**, run against the `openai` provider (`gpt-4o-mini`
for both generation and judging) specifically to avoid the Groq daily-quota
ceiling that capped an earlier run at 16/50 (see "Known issues" below for
that incident — `run_eval`'s incremental-save/skip-on-failure resilience,
built in response to it, is still in place and still real; this run just
didn't need it).

Summary over all 50 completed rows: hit@5=0.72, MRR=0.55,
faithfulness=4.74/5, relevance=4.36/5.

Unlike the earlier partial run — where every row scored a flat 5/5 on both
judge dimensions (a discriminating-power problem, not a flawless pipeline)
— this full run shows real spread: several rows scored 1/5 on faithfulness
and/or relevance, so the judge is meaningfully separating good answers from
bad ones on this sample.

## Known issues / deviations from the original plan

- **Groq model deprecated.** The originally planned default model,
  `llama-3.3-70b-versatile`, no longer exists on Groq's live catalog (404
  `model_not_found`). The default in `src/pyrag/config.py` was updated to
  `openai/gpt-oss-20b` after querying the live `/models` list and confirming
  it responds.
- **Chroma insert batch limit.** Building the real index produces 8,159
  chunks, but Chroma's client rejects a single `collection.add()` call above
  its configured max batch size (5,461 on this install).
  `VectorStore.add` (`src/pyrag/retrieval/vector_store.py`) was changed to
  insert in batches using `client.get_max_batch_size()`.
- **Groq daily token quota.** This account's `on_demand` tier caps every
  model at 200,000 tokens/day. A full 50-question eval run (~2 Groq calls per
  question, each call carrying several retrieved doc chunks as context) uses
  more than that in a single run. The quota was exhausted on
  `openai/gpt-oss-120b` after ~30 questions, and again on
  `openai/gpt-oss-20b` after 16 questions in the run captured here. Rather
  than get a higher-tier Groq plan or spread the run across multiple quota
  resets, the completed 50/50 baseline above was run against OpenAI instead
  (`LLM_PROVIDER=openai`, added specifically to route around this) — Groq
  remains the free default for normal use.
- **`run_eval` resilience fix.** Originally a single Groq failure crashed
  the whole eval loop with nothing saved. `run_eval` now wraps each
  question in try/except (skipping and logging on failure) and writes the
  report file after every question, so partial progress always survives a
  crash. See `tests/eval/test_run_eval.py`.

## Running locally

Backend:

```bash
pip install -e ".[dev]"
cp .env.example .env  # add your GROQ_API_KEY
python -c "from pyrag.ingestion.fetch_docs import fetch_python_docs; fetch_python_docs('data/raw/cpython')"
python -c "from pyrag.ingestion.build_index import build_index; build_index('data/raw/cpython/Doc', 'data/processed/chunks.json', 'data/chroma')"
uvicorn pyrag.server:app --reload --port 8000
```

Frontend (in a second terminal):

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (defaults to `http://localhost:5173`).

## Running tests

```bash
pytest
```

## Deployment

Backend (FastAPI) deploys as a Docker Space on
[Hugging Face Spaces](https://huggingface.co/spaces/Gowtham8Ai/python-docs-rag)
(free CPU tier, 16GB RAM); frontend (React/Vite) deploys as a static build on
[Vercel](https://vercel.com).

(An earlier iteration deployed the backend on Render — its free/Starter tiers
only offer 512MB RAM, which wasn't enough to load `torch` +
`sentence-transformers` + `chromadb` + the cross-encoder together even after
switching to a CPU-only torch build. HF Spaces' free tier gives 16GB, with
plenty of headroom, at no cost.)

**Index data:** the prebuilt search index (`chunks.json` + the Chroma vector
store) is hosted on a public Hugging Face dataset,
[`Gowtham8Ai/python-docs-rag-index`](https://huggingface.co/datasets/Gowtham8Ai/python-docs-rag-index),
rather than committed to this repo — `chroma.sqlite3` alone is 110MB, over
GitHub's 100MB per-file limit. `download_index.py` fetches it during the
Docker build (see `Dockerfile`).

**Backend setup (Hugging Face Spaces):**
1. The Space's `sdk: docker` + `app_port: 8000` front-matter (top of this
   file) tells HF Spaces to build and run `Dockerfile` directly.
2. Push this repo to the Space's git remote: `git push space main`.
3. Set `GROQ_API_KEY`, `GEMINI_API_KEY`, and `FRONTEND_ORIGIN` as Space
   secrets (Settings → Variables and secrets, or `hf spaces secrets add`).
4. Once live, note the Space's URL
   (`https://gowtham8ai-python-docs-rag.hf.space`).

**Frontend setup (Vercel):**
1. New Project, import this GitHub repo, set the root directory to
   `frontend/`. Vercel auto-detects the Vite build.
2. Set the `VITE_API_URL` env var to the Space URL from above.
3. Deploy, then update the backend's `FRONTEND_ORIGIN` secret to the
   resulting Vercel URL and restart the Space so CORS allows it.
