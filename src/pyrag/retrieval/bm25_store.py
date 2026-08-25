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
