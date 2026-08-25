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
