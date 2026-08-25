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
