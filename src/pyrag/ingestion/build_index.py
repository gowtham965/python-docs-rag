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
