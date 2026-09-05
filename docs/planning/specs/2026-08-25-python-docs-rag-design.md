# Python Docs RAG — Design Spec

## Purpose

A retrieval-augmented Q&A system over the Python standard library documentation, built as a portfolio project to demonstrate AI engineering skills for a career pivot from a graduate engineer trainee role into AI engineering. The project's differentiator is a real evaluation harness (retrieval metrics + LLM-as-judge) that lets before/after comparisons be shown when the pipeline is tuned — the core interview talking point, alongside a second, related project (an evaluation/benchmarking framework, planned separately).

## Constraints

- **Budget**: $0 ongoing cost. No paid API usage.
- **Skill level**: strong Python, new to AI/ML libraries — the design favors implementing core retrieval/eval logic by hand over black-box frameworks (e.g. LangChain), so the builder learns the underlying concepts and can defend design decisions in interviews.
- **Timeline**: full depth, ~4 weeks, done properly rather than rushed.
- **Deliverable**: a deployed, link-shareable web app (not just a local script/notebook).

## Architecture

Three independent pipelines:

### 1. Ingestion pipeline (offline, run once/periodically)

```
Python docs (CPython repo, Doc/ .rst source files)
   → cleaning & heading-aware chunking
   → embed each chunk (bge-small-en, local, CPU)
   → store in Chroma (vectors) + build BM25 index (rank_bm25) from same chunks
```

### 2. Query pipeline (runtime, per user question)

```
User question
   → embed question (same local model)
   → parallel retrieval: vector search (Chroma) + BM25 keyword search
   → fuse results (Reciprocal Rank Fusion, hand-implemented) → top ~20 candidates
   → rerank with cross-encoder (ms-marco-MiniLM-L-6-v2, local) → top ~5
   → relevance threshold check (see Error Handling)
   → build prompt (question + top chunks, citation-enforcing template)
   → call Groq LLM (llama-3.3-70b-versatile, free tier)
   → return answer + cited source chunks to UI
```

### 3. Eval pipeline (offline, run against a fixed test set)

```
~40-50 hand-written (question, expected-source-doc) pairs, including some
out-of-scope negative examples
   → retrieval-only run → hit rate@k, MRR against expected source docs
   → full-pipeline run → LLM-as-judge (Groq) scores faithfulness & relevance
   → results written to a report, re-run after each pipeline change to
     produce before/after comparisons
```

## Components

1. **Data ingestion** — source is CPython's `Doc/` `.rst` files (cleaner structure than scraped HTML). Chunking splits by section/heading first, then sub-splits long sections to ~300-500 tokens with small overlap.
2. **Indexing** — `bge-small-en` embeddings into Chroma; `rank_bm25` index over the same chunk IDs so both retrievers are directly comparable.
3. **Retrieval + fusion** — vector top-k and BM25 top-k merged via Reciprocal Rank Fusion, implemented by hand.
4. **Reranking** — local cross-encoder re-scores fused candidates against the question, cuts to top ~5.
5. **Generation** — prompt template enforces citations and explicit "say you don't know" behavior when context is insufficient; calls Groq's free-tier API (OpenAI-compatible, so swapping providers later is low-cost).
6. **Eval harness** — standalone module (reusable in the companion evaluation-framework project): retrieval metrics (hit rate@k, MRR) and LLM-as-judge generation metrics (faithfulness, relevance), producing a comparable report after each change.
7. **UI** — Streamlit chat interface with a question box, streamed answer, and an expandable "sources used" panel showing retrieved chunks, to visibly demonstrate grounding.
8. **Deployment** — Streamlit app on Hugging Face Spaces free tier. Only local/CPU components (embedding, reranking, Chroma, BM25) run on the host; generation calls out to Groq, avoiding the need for GPU hosting.

## Error Handling

- **Out-of-scope questions**: if reranked top results fall below a relevance threshold, skip generation and return "I couldn't find relevant information in the Python docs for this" instead of guessing. Threshold and behavior are covered by negative examples in the eval set.
- **Groq API failures/rate limits**: retry with exponential backoff (~3 attempts), surface a clean error in the UI rather than a stack trace.
- **Malformed/empty user input**: validated in the UI layer before entering the pipeline.
- **Ingestion-time malformed docs**: skipped and logged, not fatal to the whole indexing run.

## Testing

- **Unit tests** (pytest) for deterministic hand-written logic: chunking boundaries, RRF fusion math, prompt template construction. Built test-first.
- **Eval harness** serves as the testing layer for non-deterministic AI behavior (retrieval accuracy, answer quality) — not pass/fail, but tracked metrics compared before/after pipeline changes.

## Out of scope

- Multi-turn conversation memory
- Multi-language docs (Python docs only, English only)
- Fine-tuning any model (that's a separate potential project)
- Authentication/user accounts on the deployed app
