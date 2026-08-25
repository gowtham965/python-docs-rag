# Python Docs RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployed, retrieval-augmented Q&A web app over the Python standard library docs, with a hand-built hybrid retrieval pipeline and a custom evaluation harness that produces before/after metrics as the pipeline is tuned.

**Architecture:** Three independent pipelines — offline ingestion (fetch docs → chunk → embed → index), runtime query (embed → hybrid retrieve → fuse → rerank → generate), and offline eval (retrieval metrics + LLM-as-judge against a hand-labeled question set). Core retrieval/fusion/eval logic is hand-implemented rather than delegated to a framework, per the spec's learning goal.

**Tech Stack:** Python 3.10+, `sentence-transformers` (embeddings + cross-encoder reranker), `chromadb` (vector store), `rank-bm25` (keyword search), `groq` (free-tier LLM API), `streamlit` (UI), `pytest` (tests).

**Spec:** `docs/superpowers/specs/2026-08-25-python-docs-rag-design.md`

## Global Constraints

- **Budget:** $0 ongoing cost — no paid API usage anywhere in the pipeline.
- **Generation model:** Groq free tier (`llama-3.3-70b-versatile`), OpenAI-compatible client.
- **Embedding model:** `BAAI/bge-small-en-v1.5`, local, CPU.
- **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, local, CPU.
- **Deployment target:** Hugging Face Spaces free tier (Streamlit SDK).
- **Design principle:** retrieval fusion, reranking orchestration, and eval metrics are hand-implemented, not delegated to a RAG framework.
- **Package layout:** `src/pyrag/` (src-layout, installed editable via `pyproject.toml`).

---

### Task 1: Project setup & config

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/pyrag/__init__.py`
- Create: `src/pyrag/config.py`
- Create: `src/pyrag/models.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass (`groq_api_key`, `groq_model`, `embedding_model`, `reranker_model`, `chroma_path`, `relevance_threshold`) and `load_config() -> Config`, used by every later task's wiring code.
- Produces: `Chunk` dataclass (`id: str`, `text: str`, `source_file: str`, `section_title: str`) and `RetrievedChunk` dataclass (`chunk: Chunk`, `score: float`), used throughout ingestion/retrieval/generation/eval.

- [ ] **Step 1: Create project scaffolding**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pyrag"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "sentence-transformers>=3.0",
    "chromadb>=0.5",
    "rank-bm25>=0.2.2",
    "groq>=0.11",
    "streamlit>=1.38",
    "python-dotenv>=1.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

`requirements.txt`:
```
sentence-transformers>=3.0
chromadb>=0.5
rank-bm25>=0.2.2
groq>=0.11
streamlit>=1.38
python-dotenv>=1.0
numpy>=1.26
```

`.env.example`:
```
GROQ_API_KEY=your-groq-api-key-here
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
data/raw/
data/chroma/
data/processed/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 2: Create shared models**

`src/pyrag/__init__.py`:
```python
```

`src/pyrag/models.py`:
```python
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    section_title: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
```

- [ ] **Step 3: Write the failing test for config**

`tests/test_config.py`:
```python
import pytest
from pyrag.config import load_config


def test_load_config_reads_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    config = load_config()
    assert config.groq_api_key == "test-key-123"
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"


def test_load_config_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        load_config()
```

- [ ] **Step 4: Set up the environment and run the test to verify it fails**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_config.py -v
```

Expected: FAIL with "No module named 'pyrag.config'"

- [ ] **Step 5: Implement config**

`src/pyrag/config.py`:
```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_path: str = "data/chroma"
    relevance_threshold: float = 0.3


def load_config() -> Config:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Config(groq_api_key=api_key)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore src/pyrag/__init__.py src/pyrag/config.py src/pyrag/models.py tests/test_config.py
git commit -m "feat: project scaffolding, config loading, and shared models"
```

---

### Task 2: Heading-aware RST chunker

**Files:**
- Create: `src/pyrag/ingestion/__init__.py`
- Create: `src/pyrag/ingestion/chunker.py`
- Test: `tests/ingestion/test_chunker.py`

**Interfaces:**
- Consumes: `Chunk` from `pyrag.models` (Task 1).
- Produces: `chunk_rst_file(text: str, source_file: str) -> List[Chunk]`, used by Task 8's `collect_chunks`.

- [ ] **Step 1: Write the failing tests**

`tests/ingestion/__init__.py`: (empty file)

`tests/ingestion/test_chunker.py`:
```python
from pyrag.ingestion.chunker import chunk_rst_file, split_into_sections


def test_split_into_sections_detects_headings():
    text = (
        "Intro text here.\n\n"
        "Section One\n"
        "===========\n\n"
        "Body of section one.\n\n"
        "Section Two\n"
        "-----------\n\n"
        "Body of section two.\n"
    )
    sections = split_into_sections(text)
    headings = [h for h, _ in sections]
    assert "Section One" in headings
    assert "Section Two" in headings


def test_chunk_rst_file_produces_chunks_with_metadata():
    text = "My Section\n==========\n\nSome content about Python.\n"
    chunks = chunk_rst_file(text, source_file="library/example.rst")
    assert len(chunks) == 1
    assert chunks[0].section_title == "My Section"
    assert chunks[0].source_file == "library/example.rst"
    assert "Python" in chunks[0].text


def test_chunk_rst_file_splits_long_sections():
    long_body = " ".join(["word"] * 900)
    text = f"Big Section\n===========\n\n{long_body}\n"
    chunks = chunk_rst_file(text, source_file="library/big.rst")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section_title == "Big Section"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingestion/test_chunker.py -v
```

Expected: FAIL with "No module named 'pyrag.ingestion'"

- [ ] **Step 3: Implement the chunker**

`src/pyrag/ingestion/__init__.py`: (empty file)

`src/pyrag/ingestion/chunker.py`:
```python
import uuid
from typing import List, Tuple

from pyrag.models import Chunk

HEADING_CHARS = set("=-~^\"'`#*+.:_")
MAX_WORDS = 400
OVERLAP_WORDS = 50


def _is_heading_underline(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    return len(set(stripped)) == 1 and stripped[0] in HEADING_CHARS


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_heading = "Introduction"
    current_body: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line.strip()
            and _is_heading_underline(next_line)
            and len(next_line.strip()) >= len(line.strip()) * 0.8
        ):
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip()
            current_body = []
            i += 2
            continue
        current_body.append(line)
        i += 1

    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return [(heading, body) for heading, body in sections if body]


def _split_long_body(body: str, max_words: int, overlap_words: int) -> List[str]:
    words = body.split()
    if len(words) <= max_words:
        return [body]

    windows = []
    start = 0
    while start < len(words):
        end = start + max_words
        windows.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return windows


def chunk_rst_file(text: str, source_file: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for heading, body in split_into_sections(text):
        for window_text in _split_long_body(body, MAX_WORDS, OVERLAP_WORDS):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=window_text,
                    source_file=source_file,
                    section_title=heading,
                )
            )
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_chunker.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/ingestion/__init__.py src/pyrag/ingestion/chunker.py tests/ingestion/__init__.py tests/ingestion/test_chunker.py
git commit -m "feat: heading-aware RST chunker"
```

---

### Task 3: Docs fetcher

**Files:**
- Create: `src/pyrag/ingestion/fetch_docs.py`
- Test: `tests/ingestion/test_fetch_docs.py`

**Interfaces:**
- Produces: `fetch_python_docs(dest_dir: str) -> None`, run manually in Task 18 to populate `data/raw/`.

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_fetch_docs.py`:
```python
from unittest.mock import patch

from pyrag.ingestion.fetch_docs import fetch_python_docs, CPYTHON_REPO


def test_fetch_python_docs_runs_sparse_clone(tmp_path):
    dest = tmp_path / "cpython"
    with patch("pyrag.ingestion.fetch_docs.subprocess.run") as mock_run:
        fetch_python_docs(str(dest))

    clone_call = mock_run.call_args_list[0]
    assert "clone" in clone_call.args[0]
    assert CPYTHON_REPO in clone_call.args[0]

    sparse_call = mock_run.call_args_list[1]
    assert sparse_call.args[0] == ["git", "sparse-checkout", "set", "Doc"]
    assert sparse_call.kwargs["cwd"] == str(dest)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/ingestion/test_fetch_docs.py -v
```

Expected: FAIL with "cannot import name 'fetch_python_docs'"

- [ ] **Step 3: Implement the fetcher**

`src/pyrag/ingestion/fetch_docs.py`:
```python
import subprocess
from pathlib import Path

CPYTHON_REPO = "https://github.com/python/cpython.git"


def fetch_python_docs(dest_dir: str) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--filter=blob:none",
            "--sparse", CPYTHON_REPO, str(dest),
        ],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", "Doc"], cwd=str(dest), check=True)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/ingestion/test_fetch_docs.py -v
```

Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/ingestion/fetch_docs.py tests/ingestion/test_fetch_docs.py
git commit -m "feat: sparse-clone fetcher for CPython Doc/ tree"
```

---

### Task 4: Embedder

**Files:**
- Create: `src/pyrag/retrieval/__init__.py`
- Create: `src/pyrag/retrieval/embedder.py`
- Test: `tests/retrieval/test_embedder.py`

**Interfaces:**
- Produces: `Embedder(model_name: str = "BAAI/bge-small-en-v1.5")` with `.embed(texts: List[str]) -> np.ndarray`, used by Task 8 and Task 10.

- [ ] **Step 1: Write the failing test**

`tests/retrieval/__init__.py`: (empty file)

`tests/retrieval/test_embedder.py`:
```python
from pyrag.retrieval.embedder import Embedder


def test_embedder_returns_normalized_vectors():
    embedder = Embedder()
    vectors = embedder.embed(["hello world", "python programming"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-3 for n in norms)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/retrieval/test_embedder.py -v
```

Expected: FAIL with "No module named 'pyrag.retrieval'"

- [ ] **Step 3: Implement the embedder**

`src/pyrag/retrieval/__init__.py`: (empty file)

`src/pyrag/retrieval/embedder.py`:
```python
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/retrieval/test_embedder.py -v
```

Expected: PASS (1 test). Note: first run downloads the model (~130MB) — this can take a minute or two.

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/retrieval/__init__.py src/pyrag/retrieval/embedder.py tests/retrieval/__init__.py tests/retrieval/test_embedder.py
git commit -m "feat: local embedder wrapper (bge-small-en-v1.5)"
```

---

### Task 5: Vector store

**Files:**
- Create: `src/pyrag/retrieval/vector_store.py`
- Test: `tests/retrieval/test_vector_store.py`

**Interfaces:**
- Consumes: `Chunk` from `pyrag.models`.
- Produces: `VectorStore(path: str, collection_name: str = "python_docs")` with `.add(chunks, embeddings) -> None` and `.query(embedding, top_k: int = 20) -> List[str]` (chunk ids), used by Task 8 and Task 10.

- [ ] **Step 1: Write the failing test**

`tests/retrieval/test_vector_store.py`:
```python
import numpy as np

from pyrag.retrieval.vector_store import VectorStore
from pyrag.models import Chunk


def test_vector_store_add_and_query_roundtrip(tmp_path):
    store = VectorStore(path=str(tmp_path / "chroma"))
    chunks = [
        Chunk(id="a", text="apples are fruit", source_file="f1", section_title="s1"),
        Chunk(id="b", text="cars have wheels", source_file="f2", section_title="s2"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    store.add(chunks, embeddings)

    result_ids = store.query(np.array([0.9, 0.1]), top_k=1)
    assert result_ids == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/retrieval/test_vector_store.py -v
```

Expected: FAIL with "cannot import name 'VectorStore'"

- [ ] **Step 3: Implement the vector store**

`src/pyrag/retrieval/vector_store.py`:
```python
from typing import List

import chromadb

from pyrag.models import Chunk


class VectorStore:
    def __init__(self, path: str, collection_name: str = "python_docs"):
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(self, chunks: List[Chunk], embeddings) -> None:
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[
                {"source_file": c.source_file, "section_title": c.section_title}
                for c in chunks
            ],
        )

    def query(self, embedding, top_k: int = 20) -> List[str]:
        result = self._collection.query(
            query_embeddings=[embedding.tolist()], n_results=top_k
        )
        return result["ids"][0]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/retrieval/test_vector_store.py -v
```

Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/retrieval/vector_store.py tests/retrieval/test_vector_store.py
git commit -m "feat: Chroma-backed vector store"
```

---

### Task 6: BM25 keyword store

**Files:**
- Create: `src/pyrag/retrieval/bm25_store.py`
- Test: `tests/retrieval/test_bm25_store.py`

**Interfaces:**
- Consumes: `Chunk` from `pyrag.models`.
- Produces: `BM25Store(chunks: List[Chunk])` with `.query(text: str, top_k: int = 20) -> List[str]` (chunk ids), used by Task 10.

- [ ] **Step 1: Write the failing test**

`tests/retrieval/test_bm25_store.py`:
```python
from pyrag.retrieval.bm25_store import BM25Store
from pyrag.models import Chunk


def test_bm25_store_ranks_exact_keyword_match_first():
    chunks = [
        Chunk(id="a", text="dictionaries map keys to values", source_file="f", section_title="s"),
        Chunk(id="b", text="lists are ordered sequences", source_file="f", section_title="s"),
    ]
    store = BM25Store(chunks)
    result_ids = store.query("dictionaries keys", top_k=2)
    assert result_ids[0] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/retrieval/test_bm25_store.py -v
```

Expected: FAIL with "cannot import name 'BM25Store'"

- [ ] **Step 3: Implement the BM25 store**

`src/pyrag/retrieval/bm25_store.py`:
```python
from typing import List

from rank_bm25 import BM25Okapi

from pyrag.models import Chunk


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def query(self, text: str, top_k: int = 20) -> List[str]:
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [c.id for c, _ in ranked[:top_k]]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/retrieval/test_bm25_store.py -v
```

Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/retrieval/bm25_store.py tests/retrieval/test_bm25_store.py
git commit -m "feat: BM25 keyword store"
```

---

### Task 7: Reciprocal Rank Fusion

**Files:**
- Create: `src/pyrag/retrieval/fusion.py`
- Test: `tests/retrieval/test_fusion.py`

**Interfaces:**
- Produces: `reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[str]`, used by Task 10.

- [ ] **Step 1: Write the failing tests**

`tests/retrieval/test_fusion.py`:
```python
from pyrag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_favors_items_ranked_high_in_multiple_lists():
    list_a = ["x", "y", "z"]
    list_b = ["y", "x", "z"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0] in ("x", "y")
    assert fused[-1] == "z"


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert fused == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/retrieval/test_fusion.py -v
```

Expected: FAIL with "cannot import name 'reciprocal_rank_fusion'"

- [ ] **Step 3: Implement RRF**

`src/pyrag/retrieval/fusion.py`:
```python
from typing import Dict, List


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return [
        item_id
        for item_id, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/retrieval/test_fusion.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/retrieval/fusion.py tests/retrieval/test_fusion.py
git commit -m "feat: Reciprocal Rank Fusion"
```

---

### Task 8: Index build orchestration

**Files:**
- Create: `src/pyrag/ingestion/build_index.py`
- Test: `tests/ingestion/test_build_index.py`

**Interfaces:**
- Consumes: `chunk_rst_file` (Task 2), `Embedder` (Task 4), `VectorStore` (Task 5), `Chunk` (Task 1).
- Produces: `collect_chunks(docs_dir: str) -> List[Chunk]`, `save_chunks(chunks, out_path: str) -> None`, `load_chunks(path: str) -> List[Chunk]`, `build_index(docs_dir: str, chunks_out: str, chroma_path: str) -> int`. `load_chunks` is used by Task 17's wiring; `build_index` is run manually in Task 18.

- [ ] **Step 1: Write the failing tests**

`tests/ingestion/test_build_index.py`:
```python
from pyrag.ingestion.build_index import collect_chunks, save_chunks, load_chunks, build_index


def test_collect_chunks_reads_all_rst_files(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.rst").write_text("Section A\n=========\n\nContent A.\n")
    (docs_dir / "b.rst").write_text("Section B\n=========\n\nContent B.\n")

    chunks = collect_chunks(str(docs_dir))
    assert len(chunks) == 2
    sources = {c.source_file for c in chunks}
    assert sources == {"a.rst", "b.rst"}


def test_collect_chunks_skips_unreadable_file_without_crashing(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "good.rst").write_text("Section A\n=========\n\nContent A.\n")
    (docs_dir / "bad.rst").write_text("Section B\n=========\n\nContent B.\n")

    import pyrag.ingestion.build_index as build_index_module
    original_read_text = build_index_module.Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if self.name == "bad.rst":
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(build_index_module.Path, "read_text", flaky_read_text)

    chunks = collect_chunks(str(docs_dir))
    sources = {c.source_file for c in chunks}
    assert sources == {"good.rst"}


def test_save_and_load_chunks_roundtrip(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.rst").write_text("Section A\n=========\n\nContent A.\n")
    chunks = collect_chunks(str(docs_dir))

    out_path = tmp_path / "chunks.json"
    save_chunks(chunks, str(out_path))
    loaded = load_chunks(str(out_path))

    assert len(loaded) == len(chunks)
    assert loaded[0].text == chunks[0].text


def test_build_index_creates_chroma_collection(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.rst").write_text(
        "Section A\n=========\n\nDictionaries map keys to values.\n"
    )

    chunks_out = tmp_path / "chunks.json"
    chroma_path = tmp_path / "chroma"

    count = build_index(str(docs_dir), str(chunks_out), str(chroma_path))

    assert count == 1
    assert chunks_out.exists()
    assert chroma_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingestion/test_build_index.py -v
```

Expected: FAIL with "cannot import name 'collect_chunks'"

- [ ] **Step 3: Implement build_index**

`src/pyrag/ingestion/build_index.py`:
```python
import json
from pathlib import Path
from typing import List

from pyrag.models import Chunk
from pyrag.ingestion.chunker import chunk_rst_file
from pyrag.retrieval.embedder import Embedder
from pyrag.retrieval.vector_store import VectorStore


def collect_chunks(docs_dir: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for rst_path in sorted(Path(docs_dir).rglob("*.rst")):
        relative = str(rst_path.relative_to(docs_dir))
        try:
            text = rst_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as error:
            print(f"Skipping unreadable doc file {relative}: {error}")
            continue
        chunks.extend(chunk_rst_file(text, source_file=relative))
    return chunks


def save_chunks(chunks: List[Chunk], out_path: str) -> None:
    data = [
        {"id": c.id, "text": c.text, "source_file": c.source_file, "section_title": c.section_title}
        for c in chunks
    ]
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_chunks(path: str) -> List[Chunk]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


def build_index(docs_dir: str, chunks_out: str, chroma_path: str) -> int:
    chunks = collect_chunks(docs_dir)
    save_chunks(chunks, chunks_out)

    embedder = Embedder()
    embeddings = embedder.embed([c.text for c in chunks])

    store = VectorStore(path=chroma_path)
    store.add(chunks, embeddings)

    return len(chunks)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_build_index.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/ingestion/build_index.py tests/ingestion/test_build_index.py
git commit -m "feat: index build orchestration (chunk, embed, store)"
```

---

### Task 9: Cross-encoder reranker

**Files:**
- Create: `src/pyrag/retrieval/reranker.py`
- Test: `tests/retrieval/test_reranker.py`

**Interfaces:**
- Consumes: `Chunk`, `RetrievedChunk` from `pyrag.models`.
- Produces: `Reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2")` with `.rerank(query: str, chunks: List[Chunk], top_k: int = 5) -> List[RetrievedChunk]`, used by Task 10.

- [ ] **Step 1: Write the failing test**

`tests/retrieval/test_reranker.py`:
```python
from pyrag.retrieval.reranker import Reranker
from pyrag.models import Chunk


def test_reranker_ranks_relevant_chunk_higher():
    chunks = [
        Chunk(id="a", text="Dictionaries in Python map keys to values.", source_file="f", section_title="s"),
        Chunk(id="b", text="The history of the Roman Empire.", source_file="f", section_title="s"),
    ]
    reranker = Reranker()
    results = reranker.rerank("How do Python dictionaries work?", chunks, top_k=2)
    assert results[0].chunk.id == "a"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/retrieval/test_reranker.py -v
```

Expected: FAIL with "cannot import name 'Reranker'"

- [ ] **Step 3: Implement the reranker**

`src/pyrag/retrieval/reranker.py`:
```python
from typing import List

from sentence_transformers import CrossEncoder

from pyrag.models import Chunk, RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[RetrievedChunk]:
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [RetrievedChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k]]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/retrieval/test_reranker.py -v
```

Expected: PASS (1 test). Note: first run downloads the cross-encoder model.

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat: cross-encoder reranker"
```

---

### Task 10: Retriever orchestration

**Files:**
- Create: `src/pyrag/retrieval/retriever.py`
- Test: `tests/retrieval/test_retriever.py`

**Interfaces:**
- Consumes: `Chunk`, `RetrievedChunk` (Task 1), `reciprocal_rank_fusion` (Task 7).
- Produces: `RetrievalResult` dataclass (`chunks: List[RetrievedChunk]`, `is_confident: bool`) and `Retriever(embedder, vector_store, bm25_store, reranker, chunk_lookup, relevance_threshold=0.3, fusion_top_k=20, final_top_k=5)` with `.retrieve(query: str) -> RetrievalResult`. Used by Task 13 (pipeline) and Task 16 (eval).

- [ ] **Step 1: Write the failing tests**

`tests/retrieval/test_retriever.py`:
```python
from pyrag.retrieval.retriever import Retriever
from pyrag.models import Chunk, RetrievedChunk


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


class FakeVectorStore:
    def query(self, embedding, top_k):
        return ["a", "b"]


class FakeBM25Store:
    def query(self, text, top_k):
        return ["b", "c"]


class FakeReranker:
    def __init__(self, score_by_id):
        self._score_by_id = score_by_id

    def rerank(self, query, chunks, top_k):
        scored = [RetrievedChunk(chunk=c, score=self._score_by_id[c.id]) for c in chunks]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:top_k]


def _make_chunk_lookup():
    ids = ["a", "b", "c"]
    return {i: Chunk(id=i, text=f"text {i}", source_file="f", section_title="s") for i in ids}


def test_retriever_marks_confident_when_top_score_above_threshold():
    chunk_lookup = _make_chunk_lookup()
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        reranker=FakeReranker({"a": 0.9, "b": 0.5, "c": 0.1}),
        chunk_lookup=chunk_lookup,
        relevance_threshold=0.3,
    )
    result = retriever.retrieve("some question")
    assert result.is_confident is True
    assert result.chunks[0].chunk.id == "a"


def test_retriever_marks_not_confident_when_top_score_below_threshold():
    chunk_lookup = _make_chunk_lookup()
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        reranker=FakeReranker({"a": 0.1, "b": 0.05, "c": 0.01}),
        chunk_lookup=chunk_lookup,
        relevance_threshold=0.3,
    )
    result = retriever.retrieve("some question")
    assert result.is_confident is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/retrieval/test_retriever.py -v
```

Expected: FAIL with "cannot import name 'Retriever'"

- [ ] **Step 3: Implement the retriever**

`src/pyrag/retrieval/retriever.py`:
```python
from dataclasses import dataclass
from typing import List

from pyrag.models import Chunk, RetrievedChunk
from pyrag.retrieval.fusion import reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    is_confident: bool


class Retriever:
    def __init__(
        self,
        embedder,
        vector_store,
        bm25_store,
        reranker,
        chunk_lookup: dict,
        relevance_threshold: float = 0.3,
        fusion_top_k: int = 20,
        final_top_k: int = 5,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._reranker = reranker
        self._chunk_lookup = chunk_lookup
        self._relevance_threshold = relevance_threshold
        self._fusion_top_k = fusion_top_k
        self._final_top_k = final_top_k

    def retrieve(self, query: str) -> RetrievalResult:
        embedding = self._embedder.embed([query])[0]
        vector_ids = self._vector_store.query(embedding, top_k=self._fusion_top_k)
        bm25_ids = self._bm25_store.query(query, top_k=self._fusion_top_k)

        fused_ids = reciprocal_rank_fusion([vector_ids, bm25_ids])[: self._fusion_top_k]
        candidates = [self._chunk_lookup[cid] for cid in fused_ids]

        reranked = self._reranker.rerank(query, candidates, top_k=self._final_top_k)
        is_confident = bool(reranked) and reranked[0].score >= self._relevance_threshold

        return RetrievalResult(chunks=reranked, is_confident=is_confident)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/retrieval/test_retriever.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/retrieval/retriever.py tests/retrieval/test_retriever.py
git commit -m "feat: retriever orchestration (hybrid search, fusion, rerank, confidence gate)"
```

---

### Task 11: Prompt builder

**Files:**
- Create: `src/pyrag/generation/__init__.py`
- Create: `src/pyrag/generation/prompt.py`
- Test: `tests/generation/test_prompt.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `pyrag.models`.
- Produces: `build_prompt(question: str, chunks: List[RetrievedChunk]) -> str`, used by Task 13.

- [ ] **Step 1: Write the failing tests**

`tests/generation/__init__.py`: (empty file)

`tests/generation/test_prompt.py`:
```python
from pyrag.generation.prompt import build_prompt
from pyrag.models import Chunk, RetrievedChunk


def test_build_prompt_includes_question_and_citations():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(id="a", text="dict() creates a dictionary.", source_file="f", section_title="Dictionaries"),
            score=0.9,
        )
    ]
    prompt = build_prompt("How do I create a dictionary?", chunks)
    assert "How do I create a dictionary?" in prompt
    assert "[Section: Dictionaries]" in prompt
    assert "dict() creates a dictionary." in prompt


def test_build_prompt_instructs_model_to_admit_uncertainty():
    prompt = build_prompt("irrelevant question", [])
    assert "I don't know based on the provided documentation." in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/generation/test_prompt.py -v
```

Expected: FAIL with "No module named 'pyrag.generation'"

- [ ] **Step 3: Implement the prompt builder**

`src/pyrag/generation/__init__.py`: (empty file)

`src/pyrag/generation/prompt.py`:
```python
from typing import List

from pyrag.models import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about the Python "
    "standard library using only the provided documentation excerpts. "
    "Cite the section title for every claim you make, using the format "
    "[Section: <title>]. If the excerpts do not contain the answer, say "
    "\"I don't know based on the provided documentation.\" Do not use "
    "outside knowledge."
)


def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[Section: {rc.chunk.section_title}]\n{rc.chunk.text}" for rc in chunks
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Documentation excerpts:\n{context_blocks}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/generation/test_prompt.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/generation/__init__.py src/pyrag/generation/prompt.py tests/generation/__init__.py tests/generation/test_prompt.py
git commit -m "feat: citation-enforcing prompt builder"
```

---

### Task 12: Groq LLM client with retry

**Files:**
- Create: `src/pyrag/generation/llm_client.py`
- Test: `tests/generation/test_llm_client.py`

**Interfaces:**
- Produces: `GroqClient(api_key: str, model: str, max_retries: int = 3, backoff_seconds: float = 1.0)` with `.generate(prompt: str) -> str`, used by Task 13, Task 15, and wiring (Task 17).

- [ ] **Step 1: Write the failing tests**

`tests/generation/test_llm_client.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from pyrag.generation.llm_client import GroqClient


def test_generate_returns_content_on_success():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "the answer"
        MockGroq.return_value.chat.completions.create.return_value = mock_response

        client = GroqClient(api_key="key", model="test-model")
        result = client.generate("a prompt")

        assert result == "the answer"


def test_generate_retries_then_raises_after_max_attempts():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq, patch(
        "pyrag.generation.llm_client.time.sleep"
    ):
        MockGroq.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        client = GroqClient(api_key="key", model="test-model", max_retries=2, backoff_seconds=0.01)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            client.generate("a prompt")

        assert MockGroq.return_value.chat.completions.create.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/generation/test_llm_client.py -v
```

Expected: FAIL with "cannot import name 'GroqClient'"

- [ ] **Step 3: Implement the Groq client**

`src/pyrag/generation/llm_client.py`:
```python
import time

from groq import Groq


class GroqClient:
    def __init__(self, api_key: str, model: str, max_retries: int = 3, backoff_seconds: float = 1.0):
        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as error:
                last_error = error
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_seconds * (2 ** attempt))
        raise RuntimeError(f"Groq API failed after {self._max_retries} attempts") from last_error
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/generation/test_llm_client.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/generation/llm_client.py tests/generation/test_llm_client.py
git commit -m "feat: Groq LLM client with retry/backoff"
```

---

### Task 13: RAG query pipeline

**Files:**
- Create: `src/pyrag/generation/pipeline.py`
- Test: `tests/generation/test_pipeline.py`

**Interfaces:**
- Consumes: `RetrievalResult` (Task 10), `build_prompt` (Task 11).
- Produces: `AnswerResult` dataclass (`answer: str`, `sources: List[RetrievedChunk]`, `is_out_of_scope: bool`), `OUT_OF_SCOPE_MESSAGE` constant, and `RagPipeline(retriever, llm_client)` with `.answer(question: str) -> AnswerResult`. Used by Task 16 (eval) and Task 17 (wiring/UI).

- [ ] **Step 1: Write the failing tests**

`tests/generation/test_pipeline.py`:
```python
from pyrag.generation.pipeline import RagPipeline, OUT_OF_SCOPE_MESSAGE
from pyrag.retrieval.retriever import RetrievalResult
from pyrag.models import Chunk, RetrievedChunk


class FakeRetriever:
    def __init__(self, result):
        self._result = result

    def retrieve(self, query):
        return self._result


class FakeLLMClient:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt):
        return self._response


def test_pipeline_returns_out_of_scope_message_when_not_confident():
    retriever = FakeRetriever(RetrievalResult(chunks=[], is_confident=False))
    pipeline = RagPipeline(retriever=retriever, llm_client=FakeLLMClient("unused"))

    result = pipeline.answer("unrelated question")

    assert result.is_out_of_scope is True
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert result.sources == []


def test_pipeline_returns_generated_answer_when_confident():
    chunk = RetrievedChunk(chunk=Chunk(id="a", text="t", source_file="f", section_title="s"), score=0.9)
    retriever = FakeRetriever(RetrievalResult(chunks=[chunk], is_confident=True))
    pipeline = RagPipeline(retriever=retriever, llm_client=FakeLLMClient("the real answer"))

    result = pipeline.answer("a real question")

    assert result.is_out_of_scope is False
    assert result.answer == "the real answer"
    assert result.sources == [chunk]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/generation/test_pipeline.py -v
```

Expected: FAIL with "cannot import name 'RagPipeline'"

- [ ] **Step 3: Implement the pipeline**

`src/pyrag/generation/pipeline.py`:
```python
from dataclasses import dataclass
from typing import List

from pyrag.models import RetrievedChunk
from pyrag.generation.prompt import build_prompt

OUT_OF_SCOPE_MESSAGE = "I couldn't find relevant information in the Python docs for this."


@dataclass
class AnswerResult:
    answer: str
    sources: List[RetrievedChunk]
    is_out_of_scope: bool


class RagPipeline:
    def __init__(self, retriever, llm_client):
        self._retriever = retriever
        self._llm_client = llm_client

    def answer(self, question: str) -> AnswerResult:
        retrieval = self._retriever.retrieve(question)

        if not retrieval.is_confident:
            return AnswerResult(answer=OUT_OF_SCOPE_MESSAGE, sources=[], is_out_of_scope=True)

        prompt = build_prompt(question, retrieval.chunks)
        answer_text = self._llm_client.generate(prompt)
        return AnswerResult(answer=answer_text, sources=retrieval.chunks, is_out_of_scope=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/generation/test_pipeline.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/generation/pipeline.py tests/generation/test_pipeline.py
git commit -m "feat: RAG query pipeline with out-of-scope guardrail"
```

---

### Task 14: Eval question set & retrieval metrics

**Files:**
- Create: `src/pyrag/eval/__init__.py`
- Create: `src/pyrag/eval/retrieval_metrics.py`
- Create: `eval_data/questions.json`
- Test: `tests/eval/test_retrieval_metrics.py`

**Interfaces:**
- Produces: `hit_rate_at_k(retrieved_source_files: List[str], expected_source_file: str, k: int) -> bool` and `mean_reciprocal_rank(retrieved_source_files: List[str], expected_source_file: str) -> float`, used by Task 16.
- Produces: `eval_data/questions.json`, a seed set of hand-written (question, expected_source_file, is_in_scope) records, used by Task 16 and expanded in Task 18.

- [ ] **Step 1: Write the failing tests**

`tests/eval/__init__.py`: (empty file)

`tests/eval/test_retrieval_metrics.py`:
```python
from pyrag.eval.retrieval_metrics import hit_rate_at_k, mean_reciprocal_rank


def test_hit_rate_at_k_true_when_expected_within_k():
    retrieved = ["library/os.rst", "library/stdtypes.rst", "library/re.rst"]
    assert hit_rate_at_k(retrieved, "library/stdtypes.rst", k=2) is True


def test_hit_rate_at_k_false_when_expected_outside_k():
    retrieved = ["library/os.rst", "library/stdtypes.rst", "library/re.rst"]
    assert hit_rate_at_k(retrieved, "library/re.rst", k=1) is False


def test_mean_reciprocal_rank_scores_by_position():
    retrieved = ["library/os.rst", "library/stdtypes.rst"]
    assert mean_reciprocal_rank(retrieved, "library/stdtypes.rst") == 0.5


def test_mean_reciprocal_rank_zero_when_not_found():
    retrieved = ["library/os.rst"]
    assert mean_reciprocal_rank(retrieved, "library/re.rst") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/eval/test_retrieval_metrics.py -v
```

Expected: FAIL with "No module named 'pyrag.eval'"

- [ ] **Step 3: Implement retrieval metrics**

`src/pyrag/eval/__init__.py`: (empty file)

`src/pyrag/eval/retrieval_metrics.py`:
```python
from typing import List


def hit_rate_at_k(retrieved_source_files: List[str], expected_source_file: str, k: int) -> bool:
    return expected_source_file in retrieved_source_files[:k]


def mean_reciprocal_rank(retrieved_source_files: List[str], expected_source_file: str) -> float:
    for rank, source_file in enumerate(retrieved_source_files, start=1):
        if source_file == expected_source_file:
            return 1.0 / rank
    return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/eval/test_retrieval_metrics.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Create the seed eval question set**

`eval_data/questions.json`:
```json
[
  {"question": "How do I create an empty dictionary in Python?", "expected_source_file": "library/stdtypes.rst", "is_in_scope": true},
  {"question": "How do I open a file for reading in Python?", "expected_source_file": "library/functions.rst", "is_in_scope": true},
  {"question": "What does the itertools.chain function do?", "expected_source_file": "library/itertools.rst", "is_in_scope": true},
  {"question": "How do I parse a JSON string into a Python object?", "expected_source_file": "library/json.rst", "is_in_scope": true},
  {"question": "How do I get the current date and time in Python?", "expected_source_file": "library/datetime.rst", "is_in_scope": true},
  {"question": "How do I compile and match a regular expression in Python?", "expected_source_file": "library/re.rst", "is_in_scope": true},
  {"question": "What is the difference between a list and a tuple in Python?", "expected_source_file": "library/stdtypes.rst", "is_in_scope": true},
  {"question": "How do I run tests using unittest?", "expected_source_file": "library/unittest.rst", "is_in_scope": true},
  {"question": "How do I create an async coroutine and run it with asyncio?", "expected_source_file": "library/asyncio-task.rst", "is_in_scope": true},
  {"question": "How do I join file system paths in Python?", "expected_source_file": "library/os.path.rst", "is_in_scope": true},
  {"question": "What's the best way to center a div in CSS?", "expected_source_file": null, "is_in_scope": false},
  {"question": "How do I train a neural network in PyTorch?", "expected_source_file": null, "is_in_scope": false}
]
```

Note: `expected_source_file` values are best-guess CPython `Doc/` paths — verify and correct them against the actual fetched tree in Task 18, and expand this set to 40-50 questions before running the baseline eval.

- [ ] **Step 6: Commit**

```bash
git add src/pyrag/eval/__init__.py src/pyrag/eval/retrieval_metrics.py eval_data/questions.json tests/eval/__init__.py tests/eval/test_retrieval_metrics.py
git commit -m "feat: retrieval eval metrics and seed question set"
```

---

### Task 15: LLM-as-judge

**Files:**
- Create: `src/pyrag/eval/judge.py`
- Test: `tests/eval/test_judge.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `pyrag.models`.
- Produces: `JudgeScore` class (`faithfulness: int`, `relevance: int`), `build_judge_prompt(question, answer, chunks) -> str`, `judge_answer(question, answer, chunks, llm_client) -> JudgeScore`, used by Task 16.

- [ ] **Step 1: Write the failing tests**

`tests/eval/test_judge.py`:
```python
from pyrag.eval.judge import judge_answer, build_judge_prompt
from pyrag.models import Chunk, RetrievedChunk


class FakeLLMClient:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt):
        return self._response


def test_build_judge_prompt_includes_question_answer_and_context():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(id="a", text="dict maps keys to values", source_file="f", section_title="s"),
            score=0.9,
        )
    ]
    prompt = build_judge_prompt("What is a dict?", "A dict maps keys to values.", chunks)
    assert "What is a dict?" in prompt
    assert "A dict maps keys to values." in prompt
    assert "dict maps keys to values" in prompt


def test_judge_answer_parses_json_score():
    llm_client = FakeLLMClient('{"faithfulness": 5, "relevance": 4}')
    score = judge_answer("q", "a", [], llm_client)
    assert score.faithfulness == 5
    assert score.relevance == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/eval/test_judge.py -v
```

Expected: FAIL with "cannot import name 'judge_answer'"

- [ ] **Step 3: Implement the judge**

`src/pyrag/eval/judge.py`:
```python
import json
from typing import List

from pyrag.models import RetrievedChunk

JUDGE_PROMPT_TEMPLATE = """You are grading an AI-generated answer for a documentation Q&A system.

Question: {question}

Retrieved context:
{context}

Generated answer: {answer}

Score the answer from 1 (worst) to 5 (best) on two dimensions:
- faithfulness: is every claim in the answer supported by the retrieved context (no hallucination)?
- relevance: does the answer actually address the question?

Respond with ONLY a JSON object in this exact format, no other text:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>}}
"""


class JudgeScore:
    def __init__(self, faithfulness: int, relevance: int):
        self.faithfulness = faithfulness
        self.relevance = relevance


def build_judge_prompt(question: str, answer: str, chunks: List[RetrievedChunk]) -> str:
    context = "\n\n".join(rc.chunk.text for rc in chunks)
    return JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)


def judge_answer(question: str, answer: str, chunks: List[RetrievedChunk], llm_client) -> JudgeScore:
    prompt = build_judge_prompt(question, answer, chunks)
    raw_response = llm_client.generate(prompt)
    data = json.loads(raw_response)
    return JudgeScore(faithfulness=int(data["faithfulness"]), relevance=int(data["relevance"]))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/eval/test_judge.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/eval/judge.py tests/eval/test_judge.py
git commit -m "feat: LLM-as-judge faithfulness/relevance scoring"
```

---

### Task 16: Eval runner & report

**Files:**
- Create: `src/pyrag/eval/run_eval.py`
- Test: `tests/eval/test_run_eval.py`

**Interfaces:**
- Consumes: `hit_rate_at_k`, `mean_reciprocal_rank` (Task 14), `judge_answer` (Task 15), `RetrievalResult` (Task 10), `AnswerResult` (Task 13).
- Produces: `EvalRow` dataclass, `load_questions(path) -> List[dict]`, `run_eval(questions_path, retriever, pipeline, judge_llm_client, report_out_path) -> List[EvalRow]`. Run manually in Task 18 against the real pipeline.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_run_eval.py`:
```python
import json

from pyrag.eval.run_eval import run_eval
from pyrag.models import Chunk, RetrievedChunk
from pyrag.retrieval.retriever import RetrievalResult
from pyrag.generation.pipeline import AnswerResult


class FakeRetriever:
    def retrieve(self, query):
        chunk = Chunk(id="a", text="t", source_file="library/stdtypes.rst", section_title="s")
        return RetrievalResult(chunks=[RetrievedChunk(chunk=chunk, score=0.9)], is_confident=True)


class FakePipeline:
    def answer(self, question):
        return AnswerResult(answer="an answer", sources=[], is_out_of_scope=False)


class FakeJudgeLLMClient:
    def generate(self, prompt):
        return '{"faithfulness": 5, "relevance": 5}'


def test_run_eval_writes_report_and_returns_rows(tmp_path):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {"question": "q1", "expected_source_file": "library/stdtypes.rst", "is_in_scope": True},
                {"question": "q2", "expected_source_file": None, "is_in_scope": False},
            ]
        )
    )
    report_path = tmp_path / "report.json"

    rows = run_eval(
        str(questions_path), FakeRetriever(), FakePipeline(), FakeJudgeLLMClient(), str(report_path)
    )

    assert len(rows) == 1
    assert rows[0].hit_at_5 is True
    assert rows[0].faithfulness == 5
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert len(saved) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/eval/test_run_eval.py -v
```

Expected: FAIL with "cannot import name 'run_eval'"

- [ ] **Step 3: Implement the eval runner**

`src/pyrag/eval/run_eval.py`:
```python
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from pyrag.eval.retrieval_metrics import hit_rate_at_k, mean_reciprocal_rank
from pyrag.eval.judge import judge_answer


@dataclass
class EvalRow:
    question: str
    hit_at_5: bool
    mrr: float
    faithfulness: int
    relevance: int


def load_questions(path: str) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_eval(
    questions_path: str,
    retriever,
    pipeline,
    judge_llm_client,
    report_out_path: str,
) -> List[EvalRow]:
    questions = load_questions(questions_path)
    rows: List[EvalRow] = []

    for item in questions:
        if not item["is_in_scope"]:
            continue

        retrieval = retriever.retrieve(item["question"])
        source_files = [rc.chunk.source_file for rc in retrieval.chunks]

        hit = hit_rate_at_k(source_files, item["expected_source_file"], k=5)
        mrr = mean_reciprocal_rank(source_files, item["expected_source_file"])

        result = pipeline.answer(item["question"])
        score = judge_answer(item["question"], result.answer, result.sources, judge_llm_client)

        rows.append(
            EvalRow(
                question=item["question"],
                hit_at_5=hit,
                mrr=mrr,
                faithfulness=score.faithfulness,
                relevance=score.relevance,
            )
        )

    Path(report_out_path).write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/eval/test_run_eval.py -v
```

Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/pyrag/eval/run_eval.py tests/eval/test_run_eval.py
git commit -m "feat: eval runner producing retrieval + judge metrics report"
```

---

### Task 17: Wiring & Streamlit UI

**Files:**
- Create: `src/pyrag/wiring.py`
- Create: `src/pyrag/app.py`
- Test: `tests/test_wiring.py`

**Interfaces:**
- Consumes: `load_config` (Task 1), `load_chunks` (Task 8), `Embedder` (Task 4), `VectorStore` (Task 5), `BM25Store` (Task 6), `Reranker` (Task 9), `Retriever` (Task 10), `GroqClient` (Task 12), `RagPipeline` (Task 13).
- Produces: `build_pipeline(chunks_path: str = "data/processed/chunks.json") -> Tuple[RagPipeline, Retriever]`, used by `app.py` and by Task 18's manual eval run.

- [ ] **Step 1: Write the failing test**

`tests/test_wiring.py`:
```python
import pytest

from pyrag.wiring import build_pipeline


def test_build_pipeline_raises_clear_error_when_chunks_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(FileNotFoundError):
        build_pipeline(chunks_path=str(tmp_path / "missing.json"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_wiring.py -v
```

Expected: FAIL with "No module named 'pyrag.wiring'"

- [ ] **Step 3: Implement wiring**

`src/pyrag/wiring.py`:
```python
from pyrag.config import load_config
from pyrag.ingestion.build_index import load_chunks
from pyrag.retrieval.embedder import Embedder
from pyrag.retrieval.vector_store import VectorStore
from pyrag.retrieval.bm25_store import BM25Store
from pyrag.retrieval.reranker import Reranker
from pyrag.retrieval.retriever import Retriever
from pyrag.generation.llm_client import GroqClient
from pyrag.generation.pipeline import RagPipeline


def build_pipeline(chunks_path: str = "data/processed/chunks.json"):
    config = load_config()
    chunks = load_chunks(chunks_path)
    chunk_lookup = {c.id: c for c in chunks}

    embedder = Embedder(config.embedding_model)
    vector_store = VectorStore(path=config.chroma_path)
    bm25_store = BM25Store(chunks)
    reranker = Reranker(config.reranker_model)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        relevance_threshold=config.relevance_threshold,
    )
    llm_client = GroqClient(api_key=config.groq_api_key, model=config.groq_model)
    return RagPipeline(retriever=retriever, llm_client=llm_client), retriever
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_wiring.py -v
```

Expected: PASS (1 test)

- [ ] **Step 5: Build the Streamlit UI**

`src/pyrag/app.py`:
```python
import streamlit as st

from pyrag.wiring import build_pipeline

st.set_page_config(page_title="Python Docs RAG", page_icon="🐍")
st.title("🐍 Python Docs Q&A")


@st.cache_resource
def get_pipeline():
    pipeline, _ = build_pipeline()
    return pipeline


pipeline = get_pipeline()

question = st.chat_input("Ask a question about the Python standard library...")

if question and question.strip():
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = pipeline.answer(question)
        st.write(result.answer)

        if result.sources:
            with st.expander("Sources used"):
                for rc in result.sources:
                    st.markdown(
                        f"**{rc.chunk.section_title}** ({rc.chunk.source_file}) — score {rc.score:.2f}"
                    )
                    st.caption(rc.chunk.text[:300] + "...")
```

- [ ] **Step 6: Manually verify the UI** (no automated test — this is a UI smoke check)

This step requires the real index to exist (built in Task 18). Once it does:

```bash
streamlit run src/pyrag/app.py
```

Expected: app opens in browser; asking "How do I create a dictionary in Python?" returns an answer with a "Sources used" panel; asking "What's the capital of France?" returns the out-of-scope message with no sources.

- [ ] **Step 7: Commit**

```bash
git add src/pyrag/wiring.py src/pyrag/app.py tests/test_wiring.py
git commit -m "feat: pipeline wiring and Streamlit chat UI"
```

---

### Task 18: Build the real index, expand eval set, capture baseline

This task has no new source code — it runs the pipeline built so far against real data and produces the artifacts the spec's "before/after" story depends on. Treat each step as a manual checklist item.

**Files:**
- Modify: `eval_data/questions.json` (expand to 40-50 questions)
- Create: `docs/eval-results/baseline.json`
- Create: `README.md`

- [ ] **Step 1: Fetch the real docs**

```python
from pyrag.ingestion.fetch_docs import fetch_python_docs
fetch_python_docs("data/raw/cpython")
```

Run via `python -c "..."` or a scratch script. Expected: `data/raw/cpython/Doc/` exists with `.rst` files.

- [ ] **Step 2: Build the real index**

```python
from pyrag.ingestion.build_index import build_index
count = build_index("data/raw/cpython/Doc", "data/processed/chunks.json", "data/chroma")
print(f"Indexed {count} chunks")
```

Expected: a few thousand chunks, `data/processed/chunks.json` and `data/chroma/` populated.

- [ ] **Step 3: Verify and expand the eval question set**

For each entry in `eval_data/questions.json`, confirm the `expected_source_file` actually exists under `data/raw/cpython/Doc/` and contains the answer — correct any that don't match the real tree. Then add questions until the set has 40-50 in-scope entries (covering a broad spread of stdlib modules: collections, os, sys, pathlib, subprocess, logging, typing, dataclasses, csv, argparse, etc.) plus 5-8 out-of-scope negative examples, following the same JSON schema as the seed set.

- [ ] **Step 4: Run the baseline eval**

```python
from pyrag.wiring import build_pipeline
from pyrag.generation.llm_client import GroqClient
from pyrag.config import load_config
from pyrag.eval.run_eval import run_eval

pipeline, retriever = build_pipeline()
config = load_config()
judge_client = GroqClient(api_key=config.groq_api_key, model=config.groq_model)

rows = run_eval("eval_data/questions.json", retriever, pipeline, judge_client, "docs/eval-results/baseline.json")
avg_hit = sum(r.hit_at_5 for r in rows) / len(rows)
avg_mrr = sum(r.mrr for r in rows) / len(rows)
avg_faithfulness = sum(r.faithfulness for r in rows) / len(rows)
print(f"hit@5={avg_hit:.2f} mrr={avg_mrr:.2f} faithfulness={avg_faithfulness:.2f}")
```

Expected: `docs/eval-results/baseline.json` written; note the printed averages — these are your baseline numbers for the resume/interview story.

- [ ] **Step 5: Write the README**

`README.md`:
```markdown
# Python Docs RAG

A retrieval-augmented Q&A system over the Python standard library docs, with a
hand-built hybrid retrieval pipeline (BM25 + vector search, fused via
Reciprocal Rank Fusion, cross-encoder reranked) and a custom evaluation
harness (retrieval metrics + LLM-as-judge).

## Architecture

See `docs/superpowers/specs/2026-08-25-python-docs-rag-design.md` for the
full design.

## Baseline eval results

See `docs/eval-results/baseline.json`. Summary: hit@5=<fill in>,
MRR=<fill in>, faithfulness=<fill in>/5.

## Running locally

\`\`\`bash
pip install -e ".[dev]"
cp .env.example .env  # add your GROQ_API_KEY
python -c "from pyrag.ingestion.fetch_docs import fetch_python_docs; fetch_python_docs('data/raw/cpython')"
python -c "from pyrag.ingestion.build_index import build_index; build_index('data/raw/cpython/Doc', 'data/processed/chunks.json', 'data/chroma')"
streamlit run src/pyrag/app.py
\`\`\`

## Running tests

\`\`\`bash
pytest
\`\`\`
```

Fill in the `<fill in>` placeholders with the actual numbers from Step 4 before committing.

- [ ] **Step 6: Commit**

```bash
git add eval_data/questions.json docs/eval-results/baseline.json README.md
git commit -m "chore: build real index, expand eval set, capture baseline metrics"
```

Note: `data/raw/`, `data/chroma/`, and `data/processed/` are gitignored (Task 1) — only the eval report and question set are committed, not the index itself.

---

### Task 19: Deploy to Hugging Face Spaces

**Files:**
- Modify: `README.md` (add Spaces YAML front matter)

- [ ] **Step 1: Add Spaces metadata to the README**

Add this YAML front matter to the very top of `README.md` (above the existing `# Python Docs RAG` heading):

```yaml
---
title: Python Docs RAG
emoji: 🐍
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: src/pyrag/app.py
pinned: false
---
```

- [ ] **Step 2: Create the Space and push**

```bash
# Create a new Space at huggingface.co/new-space (SDK: Streamlit), then:
git remote add space https://huggingface.co/spaces/<your-username>/python-docs-rag
git push space main
```

- [ ] **Step 3: Set the Groq API key as a secret**

In the Space's Settings → Repository secrets, add `GROQ_API_KEY` with your key. Do not commit `.env` or the key itself.

- [ ] **Step 4: Verify the live deployment**

Open the Space's live URL. Ask "How do I create a dictionary in Python?" — expect an answer with a sources panel. Ask an out-of-scope question — expect the guardrail message. This is your shareable link for interviews/resume.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "chore: add Hugging Face Spaces deployment metadata"
```
