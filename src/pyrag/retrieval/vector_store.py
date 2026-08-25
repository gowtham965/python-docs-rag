from typing import List

import chromadb

from pyrag.models import Chunk


class VectorStore:
    def __init__(self, path: str, collection_name: str = "python_docs"):
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(self, chunks: List[Chunk], embeddings) -> None:
        try:
            max_batch_size = self._client.get_max_batch_size()
        except Exception:
            max_batch_size = 5000

        for start in range(0, len(chunks), max_batch_size):
            batch_chunks = chunks[start : start + max_batch_size]
            batch_embeddings = embeddings[start : start + max_batch_size]
            self._collection.add(
                ids=[c.id for c in batch_chunks],
                embeddings=[e.tolist() for e in batch_embeddings],
                documents=[c.text for c in batch_chunks],
                metadatas=[
                    {"source_file": c.source_file, "section_title": c.section_title}
                    for c in batch_chunks
                ],
            )

    def query(self, embedding, top_k: int = 20) -> List[str]:
        result = self._collection.query(
            query_embeddings=[embedding.tolist()], n_results=top_k
        )
        return result["ids"][0]
