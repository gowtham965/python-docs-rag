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


ANCHOR_PREFIX = "In the context of the Python standard library:"


class TwoModeReranker:
    """Returns different scores depending on whether the query was anchored,
    and records every query it was called with so tests can assert the
    anchored retry only happens when the raw query wasn't confident."""

    def __init__(self, raw_scores, anchored_scores=None):
        self._raw_scores = raw_scores
        self._anchored_scores = anchored_scores or {}
        self.queries_seen = []

    def rerank(self, query, chunks, top_k):
        self.queries_seen.append(query)
        scores = self._anchored_scores if query.startswith(ANCHOR_PREFIX) else self._raw_scores
        scored = [RetrievedChunk(chunk=c, score=scores[c.id]) for c in chunks]
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


def test_retriever_skips_anchored_retry_when_raw_query_is_already_confident():
    chunk_lookup = _make_chunk_lookup()
    reranker = TwoModeReranker(raw_scores={"a": 0.9, "b": 0.5, "c": 0.1})
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        relevance_threshold=0.3,
        anchored_relevance_threshold=2.0,
    )
    result = retriever.retrieve("some question")
    assert result.is_confident is True
    assert reranker.queries_seen == ["some question"]


def test_retriever_rescues_ambiguous_query_via_anchored_retry():
    chunk_lookup = _make_chunk_lookup()
    reranker = TwoModeReranker(
        raw_scores={"a": 0.1, "b": 0.05, "c": 0.01},
        anchored_scores={"a": 3.0, "b": 1.0, "c": 0.5},
    )
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        relevance_threshold=0.3,
        anchored_relevance_threshold=2.0,
    )
    result = retriever.retrieve("What is a library?")
    assert result.is_confident is True
    assert result.chunks[0].chunk.id == "a"
    assert reranker.queries_seen == [
        "What is a library?",
        "In the context of the Python standard library: What is a library?",
    ]


def test_retriever_does_not_rescue_when_anchored_score_stays_below_stricter_threshold():
    chunk_lookup = _make_chunk_lookup()
    reranker = TwoModeReranker(
        raw_scores={"a": -0.7, "b": -1.0, "c": -1.5},
        anchored_scores={"a": 1.1, "b": 0.5, "c": 0.1},
    )
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        relevance_threshold=0.3,
        anchored_relevance_threshold=2.0,
    )
    result = retriever.retrieve("How do I declare a variable in Rust?")
    assert result.is_confident is False
