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
