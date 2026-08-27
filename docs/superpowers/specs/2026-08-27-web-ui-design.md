# FastAPI Backend + React Frontend — Design Spec

## Purpose

Replace the current Streamlit UI (`src/pyrag/app.py`) with a proper
frontend/backend split: a FastAPI backend exposing the existing `RagPipeline`
over HTTP, and a React + Vite chat frontend consuming it. Motivation is
architectural, not cosmetic — a real API + frontend split is itself worth
showing in this project's portfolio, decoupled from the RAG pipeline it
serves. This supersedes the "UI" and "Deployment" components of
[[2026-08-25-python-docs-rag-design]] (§ Components 7 and 8) — everything
else in that spec (ingestion, retrieval, fusion, reranking, generation, eval)
is unchanged.

## Constraints

- Small deployment cost is acceptable (unlike the original $0 constraint) —
  backend and frontend can move off Hugging Face Spaces' free Streamlit
  hosting to whatever hosts each piece best.
- No change to retrieval/generation behavior or the eval harness — this is a
  transport-layer swap only.
- Keep the existing test-first discipline: new backend behavior (streaming,
  SSE framing, error handling) gets unit tests before/alongside
  implementation, matching the existing `tests/` structure.

## Architecture

```
React (Vite) frontend  --POST /chat, SSE-------->  FastAPI backend
        |                                                 |
   renders streamed                                  RagPipeline.answer_stream()
   tokens + sources                                       |
                                                  (unchanged: Retriever, fusion,
                                                   reranker, LLMClient)
```

Two independently deployable pieces talking over HTTP, replacing the single
in-process Streamlit app. No new persistence, no auth, no session storage
beyond what a single request needs.

## Components

### 1. Streaming support in the LLM clients

`GroqClient` and `GeminiClient` (`src/pyrag/generation/llm_client.py`,
`gemini_client.py`) each get a new method:

```python
def generate_stream(self, prompt: str) -> Iterator[str]:
```

Implemented with each SDK's native streaming (Groq: `stream=True` on
`chat.completions.create`; Gemini: `stream=True` on `generate_content`),
yielding text deltas as they arrive. Retry/backoff behavior on the *initial*
request mirrors the existing `generate()` method; a failure after streaming
has already started is not retried mid-stream (see Error Handling). The
existing `generate()` method is unchanged and keeps serving the eval harness
(`run_eval.py`, `judge.py`), which needs a complete answer, not a token
stream.

### 2. Streaming support in `RagPipeline`

`src/pyrag/generation/pipeline.py` gets a new method:

```python
def answer_stream(self, question: str) -> Iterator[dict]:
```

Behavior:
- Runs retrieval exactly as `answer()` does today.
- If not confident (out-of-scope): yields one
  `{"type": "token", "text": OUT_OF_SCOPE_MESSAGE}` then one
  `{"type": "done", "sources": [], "is_out_of_scope": True}`. No LLM call —
  same cost-saving behavior as today's `answer()`.
- If confident: builds the prompt exactly as today, calls
  `llm_client.generate_stream(prompt)`, yields one
  `{"type": "token", "text": ...}` per delta, then one final
  `{"type": "done", "sources": retrieval.chunks, "is_out_of_scope": False}`.

`answer()` is unchanged and keeps serving `run_eval.py`.

### 3. FastAPI backend (`src/pyrag/server.py`, new)

- `POST /chat` — request body `{"question": str}`. Response is
  `text/event-stream`: each `answer_stream()` dict is sent as one SSE event
  (`data: <json>\n\n`). The frontend distinguishes `token` vs `done` events
  by the `type` field.
- Pipeline is built once at process startup via FastAPI's lifespan hook
  (`build_pipeline()` from `wiring.py`, unchanged) and stored on
  `app.state` — replaces `st.cache_resource`'s role of loading models once
  per process.
- CORS middleware allows the frontend's origin (configurable via an env var,
  e.g. `FRONTEND_ORIGIN`, defaulting to `http://localhost:5173` for local
  dev).
- No other endpoints — no auth, no history/session endpoints. The frontend
  holds chat history client-side only, same scope as today's
  `st.session_state.history`.

### 4. React frontend (`frontend/`, new — Vite + React, no router)

Single chat page:
- Message list (user + assistant turns), text input, submit on enter.
- On submit: POST to `/chat`, read the SSE response body incrementally
  (`fetch` + `ReadableStream`, not `EventSource` — `EventSource` doesn't
  support POST bodies), append each `token` event's text to the in-progress
  assistant message as it arrives.
- On `done`: attach `sources` to that message; render an expandable "Sources
  used" section per assistant message (section title, source file, score —
  same fields as today's Streamlit expander).
- On stream error: show the same friendly message the current app shows on
  pipeline failure ("...may be temporarily rate-limited or unavailable...").
- No global state library, no routing — one screen, component-local state
  (`useState`) is sufficient.

### 5. Removing Streamlit

Once the new stack is verified working end-to-end:
- Delete `src/pyrag/app.py` and the `streamlit` dependency from
  `pyproject.toml`.
- Remove the Hugging Face Spaces front-matter block from `README.md` (`sdk:
  streamlit`, `app_file: ...`) and update the "Running locally" section to
  describe running the FastAPI backend (`uvicorn`) and the Vite dev server
  instead.
- Deployment target (Render/Railway for the backend, Vercel/Netlify for the
  frontend, or similar) is decided at deploy time — no code depends on the
  choice.

## Error Handling

- **Mid-stream LLM failure**: since tokens have already been sent to the
  client, the backend cannot retry transparently. On an exception during
  streaming, `answer_stream` yields a final
  `{"type": "error", "message": "..."}` event instead of `done`; the
  frontend renders the existing friendly error text and stops appending to
  that message. This differs from `answer()`'s all-or-nothing failure mode
  (a `RuntimeError` raised before any output) precisely because streaming
  has already committed partial output to the client.
- **Pre-stream failures** (e.g. retrieval error before any tokens sent):
  same as above — an `error` event, no partial message shown.
- **Malformed/empty request body**: FastAPI's request validation (422)
  handles this; the frontend disables submit on empty/whitespace-only input,
  same as today's Streamlit `chat_input` behavior.

## Testing

- `tests/generation/test_llm_client.py` / `test_gemini_client.py`: unit
  tests for `generate_stream` on both clients (stub the SDK's streaming
  response, assert yielded deltas), alongside the existing `generate` tests.
- `tests/generation/test_pipeline.py`: unit tests for `answer_stream`
  covering the out-of-scope path, the confident/streaming path, and the
  mid-stream error path — same stubbing pattern already used for `answer()`.
- `tests/test_server.py` (new): FastAPI `TestClient` tests against a stubbed
  pipeline — verifies SSE framing (`token`/`done`/`error` events), CORS
  headers, and the out-of-scope path end-to-end through the HTTP layer.
- Frontend: no automated tests (out of scope for a one-page chat UI, YAGNI).
  Verified manually via the `run` skill — golden path (in-scope question
  streams and shows sources) and edge cases (out-of-scope question, a
  simulated backend error).

## Out of scope

- Authentication or user accounts.
- Multi-turn conversation memory / follow-up question handling (already
  out of scope per the original spec).
- Server-side chat history persistence — history lives in the browser tab
  only, same as today.
- Any change to retrieval, fusion, reranking, generation logic, or the eval
  harness.
