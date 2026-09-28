---
project: python-docs-rag
track: ai-ml
level: intermediate
started: 2026-08-25
shipped: 2026-09-07
repo: https://github.com/gowtham965/python-docs-rag
live: https://huggingface.co/spaces/Gowtham8Ai/python-docs-rag  # backend API; TODO: add Vercel frontend URL or a demo video
---

# 1. What this project is

**For a non-technical friend:** a chatbot that answers Python programming
questions using only the official Python documentation. It shows you which
pages the answer came from, and it says "I don't know" instead of making
something up.

**For an engineer:** a hand-built RAG pipeline over CPython's stdlib docs
(8,159 chunks). It runs BM25 and vector search, fuses them with Reciprocal
Rank Fusion, reranks with a cross-encoder and applies a confidence gate
before generation. A custom eval harness (hit@5, MRR, LLM-as-judge
faithfulness/relevance) measures every change.

# 2. Problem it solves

General-purpose LLMs answer Python questions fluently but sometimes invent
APIs or mix up versions, and they never tell you where an answer came from.
This system only answers from retrieved doc sections, cites them, and
refuses when retrieval confidence is low. You can check a wrong answer
against its sources.

The honest second reason: it's a learning and portfolio project for a pivot
into AI engineering. The retrieval and eval internals were deliberately built
by hand, without LangChain or LlamaIndex, so every design decision can be
explained and defended.

# 3. Architecture

```
INGESTION (offline)
CPython Doc/*.rst ─→ heading-aware chunker ─→ bge-small-en-v1.5 ─→ Chroma  +  chunks.json
                     (≤400 words, 50 overlap)   (local, CPU)        (vectors)   (canonical store)

QUERY (per request)                                        FastAPI /chat (streaming)
question ─┬─ vector search (Chroma, top 20) ─┐                     ▲
          └─ BM25 (rank_bm25, top 20) ───────┴─→ RRF (k=60) ─→ cross-encoder rerank (top 5)
                                                                   │
                                          top score ≥ 0.3? ── no ──→ retry with Python-anchored
                                                │                   query, needs score ≥ 2.0
                                               yes                        │ still no
                                                ▼                         ▼
                              cited answer (Groq / Gemini / OpenAI)   "I don't know" (no LLM call)

EVAL (offline)
56 questions (50 in-scope + 6 out-of-scope) ─→ hit@5, MRR ─→ LLM judge ─→ docs/eval-results/*.json
```

Components:
- **Chunker** (`ingestion/chunker.py`) → splits on RST headings first, then windows long sections → heading-first keeps semantically whole sections, which matters more for docs than fixed-size splitting.
- **Embedder** (`retrieval/embedder.py`) → `BAAI/bge-small-en-v1.5` on CPU → free, local and good enough at this corpus size. A hosted embedding API would add cost and a network dependency.
- **Vector store** (`retrieval/vector_store.py`) → Chroma, persisted → embedded, no server to run. pgvector or Qdrant would be overkill for 8k chunks.
- **BM25 store** (`retrieval/bm25_store.py`) → `rank_bm25`, in memory → catches exact API names (`os.path.join`) that embeddings blur.
- **Fusion** (`retrieval/fusion.py`) → hand-written RRF → uses rank only, so BM25 and cosine scores (which aren't on comparable scales) never need normalizing.
- **Reranker** (`retrieval/reranker.py`) → `cross-encoder/ms-marco-MiniLM-L-6-v2` → scores query and chunk jointly for much better precision. It only runs on the fused top 20 because it's too slow to run on the whole corpus.
- **Retriever + confidence gate** (`retrieval/retriever.py`) → orchestrates the steps above and decides whether to answer at all → the main defense against answering confidently from irrelevant context.
- **LLM clients** (`generation/*_client.py`) → Groq (default, free), Gemini, OpenAI behind one interface, with retry, backoff and timeouts → provider swap via `LLM_PROVIDER`, which is how the quota problem in section 7 was worked around.
- **Eval harness** (`eval/`) → retrieval metrics plus LLM-as-judge with per-question isolation and incremental saves → makes changes measurable instead of judged by eye.
- **API + UI** (`server.py`, `frontend/`) → FastAPI streaming endpoint, React/Vite chat UI → replaced the original Streamlit UI to get a real frontend/backend split.
- **Deployment** → Docker on Hugging Face Spaces (backend), Vercel (frontend), index on an HF dataset → $0/month, with enough RAM for torch and the models.

# 4. Key decisions and trade-offs

| Decision | Options I considered | What I chose | Why | What I gave up |
|---|---|---|---|---|
| Framework | LangChain, LlamaIndex, hand-built | Hand-built | Understand and defend every stage; no hidden defaults | Speed of development; ecosystem integrations |
| Retrieval | Vector only, BM25 only, hybrid | Hybrid (BM25 + vector) | Docs questions often hinge on exact identifiers that embeddings blur | Two indexes to build and keep in sync |
| Fusion | Weighted score blend, RRF | RRF (k=60) | Scores on incomparable scales; RRF needs no normalization or weight tuning | Can't express "trust BM25 more" without changing the method |
| Precision | Bigger embedder, cross-encoder rerank | Cross-encoder on top 20 | Joint query+chunk scoring is far more precise than bi-encoder similarity | Extra latency per query; limited to a shortlist |
| Hallucination guard | Prompt-only "say I don't know", score gate | Score gate before the LLM call | Refuses deterministically and skips the LLM cost entirely | Threshold (0.3) is a hand-picked value, not calibrated |
| Ambiguous queries ("What is a library?") | Always prefix a Python anchor, retry only on failure | Retry only on gate failure, stricter 2.0 threshold | Always-prefix dropped MRR 0.550→0.496 and let a Rust question through | One extra rerank on low-confidence queries |
| LLM provider | Groq only, paid API only, pluggable | Pluggable (Groq default) | $0 default, but the eval could run on OpenAI when the Groq quota ran out | Three clients to maintain and test |
| Hosting | Render free, Render Standard, HF Spaces | HF Spaces (Docker) | 16GB RAM free; Render's 512MB OOM-killed the process | Cold starts; less control than a paid host |
| Index storage | Commit to git, LFS, HF dataset | HF dataset, fetched at Docker build | `chroma.sqlite3` is 110MB, over GitHub's limit | Build depends on an external download |

# 5. Skills demonstrated

- [ ] Hybrid retrieval (sparse + dense) evidence: `src/pyrag/retrieval/retriever.py`, `bm25_store.py`, `vector_store.py`
- [ ] Rank fusion implemented from scratch evidence: `src/pyrag/retrieval/fusion.py`, formula-discriminating test in `tests/retrieval/test_fusion.py` (commit `50a2d4b`)
- [ ] Cross-encoder reranking evidence: `src/pyrag/retrieval/reranker.py`
- [ ] Hallucination control via confidence gating evidence: `retriever.py` gate + out-of-scope test in `tests/generation/test_pipeline.py` (commit `fa9e1fd` checks the LLM is never called)
- [ ] Evaluation design (retrieval metrics + LLM-as-judge) evidence: `src/pyrag/eval/`, `eval_data/questions.json`, `docs/eval-results/baseline.json`
- [ ] Parsing unreliable LLM output evidence: `_parse_json_response` in `src/pyrag/eval/judge.py` (commit `ccc2b8f`)
- [ ] Fault tolerance around third-party APIs evidence: retry/backoff/timeout in `generation/gemini_client.py` (commit `387c6f3`), per-question isolation in `eval/run_eval.py` (commit `7d4748d`)
- [ ] Data-driven tuning with regression checks evidence: commit `1cc0ada` (anchored retry, verified against the full eval set)
- [ ] Streaming API design evidence: `POST /chat` in `src/pyrag/server.py`, `answer_stream` in `generation/pipeline.py`
- [ ] Concurrency correctness evidence: double-checked locking in `get_pipeline()` (commit `2078e52`)
- [ ] Containerization + memory-constrained deployment evidence: `Dockerfile` (CPU-only torch), commits `205a710`, `eef38b5`
- [ ] Test-driven development evidence: 75 passing tests under `tests/`, test commits preceding feature commits in `git log`

# 6. Numbers I measured

| Metric | Before | After | How I measured it |
|---|---|---|---|
| Eval questions completed | 16/50 (Groq daily quota hit) | 50/50 | `run_eval` against `eval_data/questions.json`, switched to `LLM_PROVIDER=openai` |
| hit@5 (in-scope) | — | 0.72 (36/50) | `docs/eval-results/baseline.json` |
| MRR (in-scope) | — | 0.55 | same |
| Faithfulness (judge, 1–5) | 5.00 on the 16-row partial run (no spread) | 4.74 (46×5, 1×4, 3×1) | gpt-4o-mini as judge, 50 rows |
| Relevance (judge, 1–5) | 5.00 on the 16-row partial run (no spread) | 4.36 (42×5, 8×1) | same |
| MRR with an always-on Python anchor | 0.550 | 0.496 (rejected) | Full eval rerun: 11 questions ranked worse, 5 better |
| MRR with retry-on-failure anchor | 0.550 | 0.550 (0 questions changed) | Full eval rerun, commit `1cc0ada` |
| Ambiguous in-domain queries answered | 0/2 | 2/2 | Target queries scored 3.75 and 4.18 anchored vs 2.0 threshold |
| Out-of-scope questions refused | 6/6 | 6/6 | Rust false positive scored 1.12 anchored, below the 2.0 threshold |
| Backend RAM on /chat | OOM at 512MB (Render) | Runs on HF Spaces (16GB) | Local `--memory=512m` container + Render's own "exceeded memory limit" report |
| Chunks indexed | Insert failed above 5,461 | 8,159 | Batched `collection.add()` via `get_max_batch_size()` |
| Automated tests | — | 75 passing | `pytest` |
| TODO: p50/p95 latency per request | — | — | Not measured yet |
| TODO: cost per answered question | — | — | Not measured yet (token counts not logged) |

# 7. Things that broke and how I fixed them

1. Symptom: The eval run crashed partway through and every result was lost.
Cause: Groq's free tier caps each model at 200k tokens/day; a 50-question run (about 2 LLM calls per question, each with several chunks of context) goes over that. One exception killed the whole loop, and results were only written at the end.
Fix: try/except around each question (log and skip), save the report after every question, and add an OpenAI provider to run the full baseline.
Lesson: Any long batch job against a rate-limited API saves progress incrementally and survives failures of individual items.

2. Symptom: The deployed backend died silently on the first real `/chat` request, with no traceback.
Cause: Out-of-memory kill. pip's default Linux torch wheel pulls about 15 unused CUDA packages, and even with CPU-only torch, torch + sentence-transformers + chromadb + the cross-encoder need more than 512MB.
Fix: Installed CPU-only torch in the Dockerfile, then moved the backend from Render to HF Spaces (16GB free).
Lesson: A process that dies with no logs is probably an OOM kill. I test containers under the target memory limit before deploying.

3. Symptom: `collection.add()` failed when building the real index.
Cause: 8,159 chunks exceeded Chroma's max batch size (5,461 on that install). The tests used about 20 chunks, so they never hit it.
Fix: Insert in batches sized by `client.get_max_batch_size()`.
Lesson: Test with production-sized data before calling ingestion done.

4. Symptom: The LLM judge crashed the eval with `JSONDecodeError`.
Cause: The model wrapped its JSON in markdown fences or prose despite the prompt.
Fix: `_parse_json_response` strips fences, tries a direct parse, then falls back to extracting the first `{...}` block with a regex.
Lesson: LLM output isn't structured output. I parse defensively (or use native structured-output modes) and test against messy real responses.

5. Symptom: "What is a library?" returned "I don't know", even though the right chunk was the top fusion candidate.
Cause: The cross-encoder had no signal that it was a Python question, so its score fell under the gate.
Fix: My first attempt, prefixing every query with a Python anchor, dropped MRR from 0.550 to 0.496 and let a Rust question through. The final fix only retries with the anchor after the gate fails and requires a stricter 2.0 score. The eval set showed no regressions.
Lesson: A fix that looks right on the failing example can quietly break others. I rerun the full eval before accepting any retrieval change.

6. Symptom: The configured Groq model returned 404 `model_not_found`.
Cause: `llama-3.3-70b-versatile` had been removed from Groq's catalog since the design was written.
Fix: Queried the live `/models` endpoint and picked a model I'd confirmed works (`openai/gpt-oss-20b`).
Lesson: Third-party model IDs are unstable. Make them configurable and check the live catalog rather than the docs.

# 8. What I would do differently at 100x scale

> DRAFT — rewrite in your own words.

- **Move to a real search backend.** At about 800k chunks, an in-process BM25 index rebuilt at every startup and embedded Chroma stop working. I'd use one engine with hybrid search, such as OpenSearch, Qdrant or Postgres with pgvector plus full-text search, with a persisted keyword index and incremental re-indexing when docs change.
- **Split model inference out of the API process.** Serve the embedder and cross-encoder from a GPU inference service (or batch them), so the web tier scales horizontally without each replica loading torch. Add a cache for repeated questions and for embeddings.
- **Make evaluation continuous and calibrated.** Grow the eval set to hundreds of questions, including adversarial and near-miss ones, and run it in CI on every retrieval change. Calibrate the confidence threshold against labeled positive and negative examples, and log production queries with user feedback to find failures the eval set misses.

# 9. Interview answers I have rehearsed

> DRAFT answers built from this repo's history. Rewrite each one in your own
> words and time yourself; they only count once you can say them without reading.
> Note: `docs/interview-prep.md` §5–§6 still quote the old 16/50 numbers and
> the Streamlit deployment; this file has the current numbers.

Q: Walk me through what happens when a user asks a question.
A: The question is embedded with bge-small, and I run vector search in Chroma and BM25 in parallel, top 20 each. Reciprocal Rank Fusion merges the two lists by rank, and a cross-encoder rescores the top 20 against the question and keeps 5. If the best reranked score is below 0.3, I retry once with a Python-anchored version of the query that has to clear a stricter 2.0. If it still fails, I return "I don't know" without calling the LLM. Otherwise the top 5 chunks go into a prompt that requires citations, and the answer streams back over FastAPI.

Q: Why didn't you just use LangChain?
A: The point of the project was to understand retrieval, not assemble it. Building fusion, reranking and the gate myself meant I had to decide things a framework would have hidden, like how to combine BM25 and cosine scores that aren't on comparable scales. That's why I used RRF. In a production team I'd happily use a framework, but I'd know what its defaults do.

Q: How do you know your eval numbers are trustworthy?
A: Partly I don't, and I can show why. My first partial run scored a flat 5/5 on every row, which told me the judge wasn't discriminating, not that the pipeline was perfect. The full 50-question run on gpt-4o-mini shows real spread: 8 rows scored 1 on relevance and 3 scored 1 on faithfulness. Retrieval metrics (hit@5 0.72, MRR 0.55) are deterministic and don't depend on the judge. The weak points are that 50 questions is small and the judge and generator are the same model family.

Q: Tell me about a hard bug.
A: Use story 5 from section 7 (the ambiguous query). It shows a fix that looked right but caused a regression, and that I caught it with the eval.

Q: What happens if retrieval returns confidently wrong chunks?
A: Honestly, nothing catches that today. The gate only catches low-confidence retrieval. The citation prompt helps a reader check the answer, and the judge's faithfulness score detects it offline, but not per request. Next steps would be an answer–context entailment check at request time, or refusing when citations don't support the claim.

# 10. Honest limitations

- The confidence threshold (0.3) and the anchored threshold (2.0) are hand-picked from a few examples, not calibrated on a labeled set.
- The eval set is small (50 in-scope + 6 out-of-scope), written by me, and judged by the same model family that generated the answers.
- hit@5 is 0.72: roughly 1 in 4 in-scope questions doesn't get the expected source file in the top 5.
- Nothing catches confidently wrong retrieval at request time; faithfulness is only measured offline.
- The BM25 index is rebuilt in memory at every startup; fine at 8k chunks, not at 100x.
- No multi-turn memory, single corpus (stdlib docs only), English only. These were deliberate scope cuts.
- Latency and cost per request aren't measured.
- It runs on free-tier hosting with cold starts; there's no auth, rate limiting or monitoring.
- `google-generativeai` is deprecated upstream (noted in `gemini_client.py`).

# 11. How to run it

```bash
git clone https://github.com/gowtham965/python-docs-rag && cd python-docs-rag
pip install -e ".[dev]"
cp .env.example .env   # add GROQ_API_KEY
python -c "from pyrag.ingestion.fetch_docs import fetch_python_docs; fetch_python_docs('data/raw/cpython')"
python -c "from pyrag.ingestion.build_index import build_index; build_index('data/raw/cpython/Doc', 'data/processed/chunks.json', 'data/chroma')"
uvicorn pyrag.server:app --reload --port 8000
# second terminal:
cd frontend && npm install && npm run dev   # open http://localhost:5173
```

Or run the backend in Docker; the build downloads the prebuilt index from HF:

```bash
docker build -t pyrag . && docker run -p 8000:8000 --env-file .env pyrag
```

Required environment variables:
- `GROQ_API_KEY` (always required, even with another provider)
- `LLM_PROVIDER` (optional: `groq` default, `gemini`, `openai`)
- `GEMINI_API_KEY` / `OPENAI_API_KEY` (only if you select that provider)
- `FRONTEND_ORIGIN` (backend CORS, deployment only)
- `VITE_API_URL` (frontend build, deployment only)

# 12. Credits

- CPython documentation source (`Doc/`) from [python/cpython](https://github.com/python/cpython), the corpus.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) and [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) via sentence-transformers.
- [Chroma](https://github.com/chroma-core/chroma) and [rank_bm25](https://github.com/dorianbrown/rank_bm25).
- Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009), the RRF formula and k=60.
- Built with AI pair-programming assistance (Claude Code). TODO: add any tutorials or articles you learned from.
