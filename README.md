# Python Docs RAG

A retrieval-augmented Q&A system over the Python standard library docs, with a
hand-built hybrid retrieval pipeline (BM25 + vector search, fused via
Reciprocal Rank Fusion, cross-encoder reranked) and a custom evaluation
harness (retrieval metrics + LLM-as-judge).

## Architecture

See `docs/superpowers/specs/2026-08-25-python-docs-rag-design.md` for the
full design.

## Baseline eval results

See `docs/eval-results/baseline.json`.

**Note on completeness:** the eval question set has 50 in-scope questions,
but this baseline run only completed **16 of 50** before the Groq account's
daily token quota (200,000 TPD, shared per-model on the `on_demand` tier)
was exhausted mid-run — first on `openai/gpt-oss-120b`, then again on the
smaller `openai/gpt-oss-20b` fallback. `run_eval` now saves its report
incrementally after every question and skips (rather than crashes on)
individual API failures, so the 16 completed rows are real, complete
results — just a partial sample rather than the full 50. See "Known issues"
below.

Summary over the 16 completed rows: hit@5=0.69, MRR=0.45,
faithfulness=5.00/5, relevance=5.00/5.

Every completed row scored a perfect 5/5 on both faithfulness and relevance
— worth treating with some skepticism as a judge-signal issue (little
discriminating power on this sample) rather than as evidence the pipeline is
flawless; see "Known issues."

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
  `openai/gpt-oss-20b` after 16 questions in the run captured here. A
  complete 50/50 baseline needs either a higher-tier Groq plan or spreading
  the eval run across multiple days/quota resets.
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
