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
