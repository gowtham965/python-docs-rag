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
