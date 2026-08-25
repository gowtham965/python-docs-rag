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
